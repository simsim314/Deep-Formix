#!/usr/bin/env python3
"""
main.py - Main entry point with all configuration. Orchestrates the pipeline.
"""
import time
import json
import multiprocessing as mp
import bagz
from pathlib import Path
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Empty

from download_utils import download_file, get_file_size_gb
from train import continuous_train

# Set the multiprocessing start method to 'spawn'. This is crucial for CUDA
# and provides a clean start for child processes, avoiding many deadlocks.
mp.set_start_method("spawn", force=True)

class ChessPipeline:
    """Orchestrates the download, processing, training, and cleanup threads/processes."""
    
    def __init__(self, config):
        self.config = config
        self.work_dir = Path(config['work_dir'])
        self.work_dir.mkdir(exist_ok=True)
        
        self.status_file = self.work_dir / "pipeline_status.json"
        self.status_lock = Lock()
        self.status = self._load_status()
        
        # Use standard, robust multiprocessing queues
        self.job_queue = mp.Queue()
        self.processed_queue = mp.Queue()
        
        self.active_bags = {} # Tracks files on disk to manage storage space
        self.download_thread = None
        self.cleanup_thread = None
        self.train_process = None
        self.stop_event = Event()

    def _load_status(self):
        """Loads pipeline status from JSON, with defaults for robustness."""
        defaults = {"next_download_index": 0, "bags_processed": 0}
        with self.status_lock:
            if self.status_file.exists():
                try:
                    with open(self.status_file, 'r') as f:
                        status = json.load(f)
                    defaults.update(status)
                    return defaults
                except json.JSONDecodeError:
                    print("Warning: Could not decode status file. Starting fresh.")
            return defaults

    def _save_status(self):
        """Saves the current status to JSON in a thread-safe manner."""
        with self.status_lock:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)

    def download_worker(self):
        """
        A thread that downloads new bag files. It pauses if the disk is full.
        """
        # On startup, queue any bag files that are already on disk.
        for bag_path in self.work_dir.glob("*.bag"):
            try:
                reader = bagz.BagFileReader(str(bag_path))
                # A job is a tuple of (path, number_of_positions)
                self.job_queue.put((str(bag_path), len(reader)))
                self.active_bags[str(bag_path)] = get_file_size_gb(bag_path)
            except Exception as e:
                print(f"Could not queue existing bag {bag_path.name}: {e}")

        while not self.stop_event.is_set():
            try:
                # Only stop the downloader if total disk storage is full
                disk_full = sum(self.active_bags.values()) >= self.config['max_storage_gb']
                
                if disk_full:
                    self.stop_event.wait(timeout=self.config['check_interval'])
                    continue

                idx = self.status['next_download_index']
                bag_path = download_file(idx, self.work_dir, self.config['download_url_template'])
                
                if bag_path:
                    reader = bagz.BagFileReader(str(bag_path))
                    print(f"Downloaded and queued: {bag_path.name}")
                    self.job_queue.put((str(bag_path), len(reader)))
                    self.active_bags[str(bag_path)] = get_file_size_gb(bag_path)
                    
                    with self.status_lock:
                        self.status['next_download_index'] = idx + 1
                    self._save_status()
                else:
                    # Wait a bit longer if a download fails (e.g., 404 error)
                    self.stop_event.wait(timeout=5.0)
            except Exception as e:
                print(f"Download worker error: {e}")
                self.stop_event.wait(timeout=self.config['check_interval'])

    def cleanup_worker(self):
        """A thread that waits for processed file paths and deletes them from disk."""
        while not self.stop_event.is_set():
            try:
                finished_bag_path_str = self.processed_queue.get(timeout=1.0)
                path_obj = Path(finished_bag_path_str)
                
                if path_obj.exists():
                    path_obj.unlink()
                if finished_bag_path_str in self.active_bags:
                    del self.active_bags[finished_bag_path_str]
                
                with self.status_lock:
                    self.status['bags_processed'] += 1
                self._save_status()
            except Empty:
                continue
            except Exception as e:
                print(f"Cleanup worker error: {e}")

    def start(self):
        """Starts all pipeline components."""
        print(f"[{datetime.now()}] Starting chess pipeline...")
        
        self.download_thread = Thread(target=self.download_worker, daemon=True)
        self.cleanup_thread = Thread(target=self.cleanup_worker, daemon=True)
        self.download_thread.start()
        self.cleanup_thread.start()
        
        self.train_process = mp.Process(
            target=continuous_train,
            args=(self.config, self.job_queue, self.processed_queue)
        )
        self.train_process.start()
        
        print(f"[{datetime.now()}] Pipeline started successfully. Press Ctrl+C to stop.")
        try:
            self.train_process.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
            
    def stop(self):
        """Gracefully shuts down all pipeline components."""
        print("\nShutting down pipeline...")
        self.stop_event.set()
        
        # Send a "poison pill" for each worker to ensure they exit the queue.get()
        for _ in range(self.config['num_workers']):
            try:
                self.job_queue.put(None, timeout=1.0)
            except Exception:
                pass
        
        if self.train_process and self.train_process.is_alive():
            print("Waiting for training process to finish...")
            self.train_process.join(timeout=10.0)
            if self.train_process.is_alive():
                print("Training process did not exit gracefully, terminating.")
                self.train_process.terminate()
            print("✓ Training process stopped.")
            
        if self.download_thread.is_alive():
            self.download_thread.join()
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join()
            
        print("✓ Helper threads stopped.")
        print("Pipeline stopped.")


def main():
    """Defines the configuration and starts the pipeline."""
    config = {
        'work_dir': 'chess_pipeline',
        'checkpoint_dir': 'results',
        'checkpoint_prefix': 'nnuew_resnet_',
        'download_url_template': "https://storage.googleapis.com/searchless_chess/data/train/action_value-{:05d}-of-02148_data.bag",
        'max_storage_gb': 10.0,
        'check_interval': 10,
        'num_workers': 5,
        'batch_size': 1024,

        # <<< ADD THESE TWO LINES for the new custom loss >>>
        'score_loss_weight': 1.0, # Weight for the direct score prediction loss
        'diff_loss_weight': 0.1,  # Weight for the weighted difference loss

        # Base LR is still required for fresh runs, but will NOT be re-applied
        # after loading if lr_schedule exists in config.
        'lr': 1e-5,

        'warmup_steps': 100,
        'checkpoint_interval': 250,
        'max_checkpoints': 20,

        # define a schedule to halve LR every 25 checkpoints.
        # If this key exists, train.py will NOT reset LR from config['lr']
        # after loading a checkpoint; it will just multiply the current LR
        # by 'factor' on each save.
        #'lr_schedule': {
        #    'type': 'checkpoint_decay',
        #    'factor': float(0.5 ** (1/25))  # ≈ 0.972682...
        #},
    }
    pipeline = ChessPipeline(config)
    pipeline.start()
    


if __name__ == "__main__":
    main()
