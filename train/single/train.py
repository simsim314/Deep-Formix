#!/usr/bin/env python3
"""
train.py - Training utilities and functions

Behavior
────────
• Warmup-by-batch (LinearLR) as before.
• If config['lr_schedule'] exists with type=='checkpoint_decay', we DO NOT
  reset LR from config['lr'] after loading; on each checkpoint we multiply the
  optimizer’s current LR (all param groups) by the given 'factor'.
• No counters; optimizer LR is the source of truth across resumes.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any
import multiprocessing as mp  # only for type parity with caller

from model import StackedNNUE
from dataset import StreamingBagDataset, collate


def get_latest_checkpoint(checkpoint_dir: Path, prefix: str) -> Path | None:
    """Find the most recent checkpoint file in a directory."""
    if not checkpoint_dir.exists():
        return None
    ckpts = list(checkpoint_dir.glob(f"{prefix}*.pt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime)
    return ckpts[-1]


def scalar_to_two_bin_probs(tgt: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Convert a scalar target p(win) in [0, 1] to a soft distribution over bins."""
    pos = tgt.squeeze(-1) * 50.0
    idx0 = torch.floor(pos).long()
    idx1 = torch.clamp(idx0 + 1, max=50)
    w = (pos - idx0.float()).clamp(0, 1)

    N = tgt.size(0)
    probs = torch.zeros(N, 51, device=tgt.device)
    probs.scatter_(1, idx0.unsqueeze(1), (1.0 - w).unsqueeze(1))
    probs.scatter_add_(1, idx1.unsqueeze(1), w.unsqueeze(1))
    return probs


class Trainer:
    """Manages the model, optimizer, and training steps."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = StackedNNUE().to(self.device)

        # Optimizer (initial LR used only for fresh runs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'])
        self.step_count = 0

        # Checkpoint dir
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Read schedule from config (optional)
        schedule = config.get('lr_schedule', None)
        if schedule and schedule.get('type') == 'checkpoint_decay':
            self.ckpt_decay_factor = float(schedule['factor'])
        else:
            self.ckpt_decay_factor = None

        # Load checkpoint (model + optimizer + step)
        optimizer_loaded = self._load_checkpoint()

        # IMPORTANT:
        # If a schedule is provided, DO NOT reset LR from config['lr'].
        # If no schedule, only set LR from config['lr'] when we didn't
        # load optimizer state (fresh start).
        #if not optimizer_loaded and self.ckpt_decay_factor is None:
        for g in self.optimizer.param_groups:
            g['lr'] = config['lr']
            g['initial_lr'] = config['lr']

        # Warmup scheduler (per-batch)
        if self.step_count < self.config["warmup_steps"]:
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=self.config["warmup_steps"]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

        # Rolling stats
        self.loss_hist = deque(maxlen=50)
        self.mae_hist = deque(maxlen=50)

    def _load_checkpoint(self) -> bool:
        """Load latest checkpoint if available. Returns True if optimizer loaded."""
        latest = get_latest_checkpoint(self.checkpoint_dir, self.config['checkpoint_prefix'])
        if not latest:
            return False

        print(f"[TRAINER] Loading checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=self.device)

        # Model
        model_state = ckpt.get("model", None)
        if model_state:
            self.model.load_state_dict(model_state, strict=False)

        # Optimizer
        opt_loaded = False
        opt_state = ckpt.get("optimizer", None)
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
                opt_loaded = True
                print("✓ [TRAINER] Optimizer state loaded.")
            except Exception as e:
                print(f"⚠️  [TRAINER] Could not load optimizer state: {e}")

        # Step
        if "step" in ckpt:
            self.step_count = int(ckpt["step"])
            print(f"✓ [TRAINER] Resuming from step {self.step_count}.")
        elif "epoch" in ckpt:
            self.step_count = int(ckpt["epoch"]) * 100000
            print(f"⚠️  [TRAINER] Estimating start step as {self.step_count}.")

        return opt_loaded

    def save_checkpoint(self):
        """Save model and optimizer state, then apply checkpoint LR decay (if configured), and prune."""
        ckpt_path = self.checkpoint_dir / f"{self.config['checkpoint_prefix']}step_{self.step_count:08d}.pt"
        torch.save({
            "step": self.step_count,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, ckpt_path)
        print(f"✓ [TRAINER] Checkpoint saved → {ckpt_path}")

        # Apply checkpoint-based LR decay by multiplying current LR (no counters)
        if self.ckpt_decay_factor is not None:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.ckpt_decay_factor
            print(f"↘ LR decayed on checkpoint: now {self.optimizer.param_groups[0]['lr']:.3e} "
                  f"(× {self.ckpt_decay_factor:.6f} per save)")

        # Prune old checkpoints
        all_ckpts = sorted(
            self.checkpoint_dir.glob(f"{self.config['checkpoint_prefix']}*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        if len(all_ckpts) > self.config['max_checkpoints']:
            for old_ckpt in all_ckpts[:-self.config['max_checkpoints']]:
                try:
                    old_ckpt.unlink()
                except Exception as e:
                    print(f"⚠️  Could not delete old checkpoint {old_ckpt.name}: {e}")

    def train_step(self, batch):
        """Perform a single training step (forward and backward pass)."""
        # OLD path: descriptor index is passed directly; no worker-side masks
        piece, side, ep, cas, fif, tgt, d_idx = (d.to(self.device) for d in batch)

        logits, p_win = self.model(piece, side, ep, cas, fif, d_idx)

        tgt_probs = scalar_to_two_bin_probs(tgt, self.model.bins)

        loss = F.kl_div(logits.log_softmax(dim=-1), tgt_probs, reduction="batchmean")
        mae_pwin = F.l1_loss(p_win, tgt)
        mse_pwin = F.mse_loss(p_win, tgt)

        self.optimizer.zero_grad(set_to_none=True)
        mse_pwin.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.step_count < self.config['warmup_steps']:
            self.scheduler.step()

        self.loss_hist.append(loss.item())
        self.mae_hist.append(mae_pwin.item())
        self.step_count += 1

        return loss.item(), mae_pwin.item()

    def get_stats(self):
        """Get current training statistics for logging."""
        return {
            "avg_loss": sum(self.loss_hist) / len(self.loss_hist) if self.loss_hist else 0.0,
            "avg_mae":  sum(self.mae_hist)  / len(self.mae_hist)  if self.mae_hist  else 0.0,
            "step": self.step_count,
            "lr": self.optimizer.param_groups[0]['lr'],
        }


def continuous_train(config: Dict[str, Any], job_queue: mp.Queue, processed_queue: mp.Queue):
    """The main function for the training process."""
    trainer = Trainer(config)

    dataset = StreamingBagDataset(config['work_dir'], job_queue, processed_queue)

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    trainer.model.train()
    pbar = tqdm(desc="Training", unit="batch")

    for batch in dataloader:
        loss, mae = trainer.train_step(batch)

        stats = trainer.get_stats()
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{stats['avg_loss']:.5f}",
            mae=f"{stats['avg_mae']:.5f}",
            step=f"{stats['step']:,}",
            lr=f"{stats['lr']:.2e}"
        )

        if trainer.step_count % config['checkpoint_interval'] == 0:
            trainer.save_checkpoint()
