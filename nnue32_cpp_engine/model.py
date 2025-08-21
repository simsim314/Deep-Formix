#!/usr/bin/env python3
"""
model.py – Neural-network model with *bucket-specific* embeddings.

Changes (2025-08-02)
────────────────────
• All eight embedding tables (pieces, castling, en-passant, 50-move) are now
  **unique per descriptor index**.  We introduce a small `BucketEmbedding`
  sub-module and keep 32 of them in a `nn.ModuleList`.
• Forward-pass logic picks the correct embedding + MLP head for every
  distinct `descriptor_index` present in the batch.
• If a bucket is unused in the current batch we simply skip its computation,
  so the cost scales with the number of buckets actually present.

No other logic was changed.
"""

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────────────────
# Utility blocks
class ResidualBlock32LN_GELU(nn.Module):
    """32 → 32 → 32 residual block with GELU → LayerNorm."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 32)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(32)

    def forward(self, x):
        y = self.fc1(x)
        y = self.gelu(y)
        y = self.ln1(y)

        y = self.fc2(y)
        y = self.gelu(y)
        y = self.ln2(y)

        return y + x


class MLPHead(nn.Module):
    """
    A simple 2240 → 256 → 32 network with 12 residual blocks,
    producing 51-bin logits.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2240, 256)
        self.gelu = nn.GELU()
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 32)
        self.ln2 = nn.LayerNorm(32)

        self.blocks = nn.Sequential(*[ResidualBlock32LN_GELU() for _ in range(12)])
        self.fc_out = nn.Linear(32, 51)

    def forward(self, x_emb: torch.Tensor):
        x = self.fc1(x_emb)
        x = self.gelu(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.gelu(x)
        x = self.ln2(x)

        x = self.blocks(x)
        logits = self.fc_out(self.gelu(x))
        return logits


# ────────────────────────────────────────────────────────────────────────────────
class BucketEmbedding(nn.Module):
    """
    A *private* set of embedding tables for one descriptor bucket.
    Shapes match the original shared tables.
    """
    def __init__(self):
        super().__init__()
        self.W_white_piece = nn.Parameter(torch.empty(64, 12, 32))
        self.W_black_piece = nn.Parameter(torch.empty(64, 12, 32))
        self.W_white_castle = nn.Parameter(torch.empty(4, 32))
        self.W_black_castle = nn.Parameter(torch.empty(4, 32))
        self.W_white_ep     = nn.Parameter(torch.empty(8, 32))
        self.W_black_ep     = nn.Parameter(torch.empty(8, 32))
        self.W_white_fifty  = nn.Parameter(torch.empty(2, 32))
        self.W_black_fifty  = nn.Parameter(torch.empty(2, 32))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)


# ────────────────────────────────────────────────────────────────────────────────
class StackedNNUE(nn.Module):
    """
    Main model:
        • 32 *independent* `BucketEmbedding` modules
        • 32 matching `MLPHead`s
    Forward pass groups the batch by descriptor bucket so work is not wasted.
    """
    def __init__(self):
        super().__init__()

        # 1. Per-bucket embeddings & heads
        self.embeddings = nn.ModuleList([BucketEmbedding() for _ in range(32)])
        self.networks   = nn.ModuleList([MLPHead()       for _ in range(32)])

        # 2. Fixed bin mid-points (0.00 … 1.00)
        self.register_buffer("bins", torch.linspace(0.0, 1.0, 51))

    # -------------------------------------------------------------------------
    @staticmethod
    def _sel(white: torch.Tensor,
             black: torch.Tensor,
             side_flag: torch.Tensor) -> torch.Tensor:
        """
        Side-to-move selector:
            side_flag == 0 → white,  side_flag == 1 → black
        """
        batch = side_flag.size(0)
        w = white.unsqueeze(0).expand(batch, *white.shape)
        b = black.unsqueeze(0).expand(batch, *black.shape)
        mask = side_flag.view(batch, *([1] * white.dim())).bool()
        return torch.where(mask, b, w)

    # -------------------------------------------------------------------------
    def _compute_x_emb(self,
                       emb: BucketEmbedding,
                       piece_idx: torch.Tensor,
                       side_flag: torch.Tensor,
                       ep_file:   torch.Tensor,
                       castle_ms: torch.Tensor,
                       fifty_a:   torch.Tensor) -> torch.Tensor:
        """
        Produce the 2240-dim concatenated embedding for *one* bucket subset.
        Shapes are identical to the original single-table implementation.
        """
        N, device = piece_idx.size(0), piece_idx.device

        # ––– piece embeddings –––
        W_piece = torch.where(
            side_flag.view(N, 1, 1, 1).bool(),
            emb.W_black_piece.unsqueeze(0).expand(N, -1, -1, -1),
            emb.W_white_piece.unsqueeze(0).expand(N, -1, -1, -1),
        )
        idx_clamp = piece_idx.clamp(min=0)
        gathered = torch.gather(
            W_piece, 2,
            idx_clamp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 32)
        )                               # N × 64 × 1 × 32
        piece_emb = gathered.squeeze(2)  # N × 64 × 32
        # mask out empty squares
        piece_emb *= (piece_idx >= 0).unsqueeze(-1).float()
        pieces_vec = piece_emb.reshape(N, 2048)

        # ––– castling rights –––
        W_castle = self._sel(emb.W_white_castle, emb.W_black_castle, side_flag)
        castle_vec = (castle_ms.unsqueeze(-1) * W_castle).reshape(N, 128)

        # ––– en-passant file –––
        W_ep = self._sel(emb.W_white_ep, emb.W_black_ep, side_flag)
        ep_vec = torch.zeros(N, 32, device=device)
        valid = ep_file >= 0
        if valid.any():
            b_idx  = valid.nonzero(as_tuple=True)[0]
            f_idx  = ep_file[valid]
            ep_vec[valid] = W_ep[b_idx, f_idx]

        # ––– 50-move rule embedding (linear interpolation) –––
        W_fifty = self._sel(emb.W_white_fifty, emb.W_black_fifty, side_flag)
        fifty_vec = ((1.0 - fifty_a).unsqueeze(-1) * W_fifty[:, 0] +
                     fifty_a.unsqueeze(-1)       * W_fifty[:, 1])

        # ––– final concat –––
        return torch.cat([pieces_vec, castle_vec, ep_vec, fifty_vec], dim=1)  # N × 2240

    # -------------------------------------------------------------------------
    def forward(
        self,
        piece_idx:        torch.Tensor,   # [N, 64] int64  (-1 = empty)
        side_flag:        torch.Tensor,   # [N]     int64  (0 = white, 1 = black)
        ep_file:          torch.Tensor,   # [N]     int64  (-1 = none, 0-7 file)
        castle_ms:        torch.Tensor,   # [N, 4]  float32
        fifty_a:          torch.Tensor,   # [N]     float32  (0-1)
        descriptor_index: torch.Tensor    # [N]     int64   (0-31 bucket id)
    ):
        N, device = piece_idx.size(0), piece_idx.device
    
        # Allocate outputs once
        all_logits = torch.empty(N, 51, device=device)
        all_pwins  = torch.empty(N, 1,  device=device)
    
        # Iterate over all possible buckets 0..31
        for idx in range(32):
            mask = descriptor_index == idx
            if not mask.any():
                continue  # skip if no samples for this bucket
    
            # 1. Build embeddings for this bucket’s slice
            x_emb = self._compute_x_emb(
                self.embeddings[idx],
                piece_idx[mask],
                side_flag[mask],
                ep_file[mask],
                castle_ms[mask],
                fifty_a[mask]
            )
    
            # 2. Run the corresponding head
            logits = self.networks[idx](x_emb)
            all_logits[mask] = logits
    
            p_win = (logits.softmax(-1) * self.bins).sum(-1, keepdim=True)
            all_pwins[mask] = p_win
    
        return all_logits, all_pwins

