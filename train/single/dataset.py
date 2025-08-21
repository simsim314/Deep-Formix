#!/usr/bin/env python3
"""
dataset.py – Final, robust, file-based orchestration.

Changes (2025-08-02)
────────────────────
• NEW:   Loads bucket_mapping.json (32-bucket greedy mapping).
• NEW:   calculate_descriptor_index() first looks up the mapping:
           key = (has_queen, piece_count, pawn_count)  ➜  bucket_idx
         If the key or the JSON file is missing it falls back to the
         original 5-bit heuristic, so training never crashes.
• No other logic touched.

This file is stand-alone – drop-in replacement.
"""

from __future__ import annotations

import ast
import json
import time
import fcntl
import multiprocessing as mp
from pathlib import Path
from queue import Empty
from typing import Dict, Tuple, Optional
import random 

import bagz
import chess
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

# ────────────────────────────────────────────────────────────────────────────────
# Bucket mapping – load once at import time
BUCKET_MAPPING_FILE = "bucket_mapping.json"
try:
    _raw: Dict[str, int] = json.load(open(BUCKET_MAPPING_FILE, "r"))
    BUCKET_MAP: Dict[Tuple[bool, int, int], int] = {
        ast.literal_eval(k): v for k, v in _raw.items()
    }
    print(f"[dataset.py] Loaded {len(BUCKET_MAP)} bucket mappings "
          f"from '{BUCKET_MAPPING_FILE}'.")
except (FileNotFoundError, json.JSONDecodeError) as e:
    BUCKET_MAP = {}
    print(f"[dataset.py] WARNING – could not load '{BUCKET_MAPPING_FILE}': {e}. "
          "Falling back to heuristic descriptor indices.")

# ────────────────────────────────────────────────────────────────────────────────
# Vectorisation helpers
piece_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def vectorize_position(board: chess.Board):
    """Convert a chess.Board to NN-ready arrays."""
    piece_array = np.full(64, -1, dtype=np.int8)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            piece_array[sq] = piece_to_index[p.symbol()]

    ep_vector = np.zeros(8, dtype=np.int8)
    if board.ep_square is not None:
        ep_vector[chess.square_file(board.ep_square)] = 1

    castling_vector = np.array([
        [board.has_kingside_castling_rights(chess.WHITE)],
        [board.has_queenside_castling_rights(chess.WHITE)],
        [board.has_kingside_castling_rights(chess.BLACK)],
        [board.has_queenside_castling_rights(chess.BLACK)],
    ], dtype=np.int8)

    side_vector = np.array([board.turn], dtype=np.int8)

    a = min(board.halfmove_clock / 100.0, 1.0)
    fifty_vector = np.array([a, 1.0 - a], dtype=np.float32)

    return piece_array, ep_vector, castling_vector, side_vector, fifty_vector


# ────────────────────────────────────────────────────────────────────────────────
def _heuristic_descriptor_index(board: chess.Board) -> int:
    """
    Legacy 5-bit rule:
      Bits 4-2 : piece-count category (0-7)
      Bit  1   : queen presence
      Bit  0   : pawn ≥ pieces flag
    """
    has_queen = bool(board.pieces(chess.QUEEN, chess.WHITE)
                     or board.pieces(chess.QUEEN, chess.BLACK))
    queen_flag = int(has_queen)

    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) \
          + len(board.pieces(chess.ROOK, chess.BLACK))
    knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) \
            + len(board.pieces(chess.KNIGHT, chess.BLACK))
    bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) \
            + len(board.pieces(chess.BISHOP, chess.BLACK))
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) \
           + len(board.pieces(chess.QUEEN, chess.BLACK))
    piece_count = rooks + knights + bishops + queens
    piece_category = min(piece_count // 4, 7)

    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) \
               + len(board.pieces(chess.PAWN, chess.BLACK))
    pawn_dom_flag = int(pawn_count >= piece_count)

    return (piece_category << 2) | (queen_flag << 1) | pawn_dom_flag


def calculate_descriptor_index(board: chess.Board) -> int:
    """
    Preferred: look up the 32-bucket mapping. Fallback: heuristic.
    """
    has_queen = bool(board.pieces(chess.QUEEN, chess.WHITE)
                     or board.pieces(chess.QUEEN, chess.BLACK))

    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) \
          + len(board.pieces(chess.ROOK, chess.BLACK))
    knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) \
            + len(board.pieces(chess.KNIGHT, chess.BLACK))
    bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) \
            + len(board.pieces(chess.BISHOP, chess.BLACK))
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) \
           + len(board.pieces(chess.QUEEN, chess.BLACK))
    piece_count = rooks + knights + bishops + queens

    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) \
               + len(board.pieces(chess.PAWN, chess.BLACK))

    key = (has_queen, piece_count, pawn_count)
    if key in BUCKET_MAP:
        return BUCKET_MAP[key]

    # Not found – graceful fallback
    return _heuristic_descriptor_index(board)


# ────────────────────────────────────────────────────────────────────────────────
class StreamingBagDataset(IterableDataset):
    """
    Iterable dataset reading .bag files in parallel via a file-locked JSON state.
    Workers emit:
        (piece_arrays, 1-win_prob, descriptor_bucket_index)
    """

    def __init__(self, work_dir: str,
                 job_queue: mp.Queue,
                 processed_queue: mp.Queue):
        super().__init__()
        self.work_dir = Path(work_dir)
        self.job_queue = job_queue
        self.processed_queue = processed_queue

        # File-based coordination
        self.state_file = self.work_dir / "worker_state.json"
        self.lock_file = self.work_dir / "state.lock"

        # Worker 0 creates the state files
        info = get_worker_info()
        if info is None or info.id == 0:
            self.lock_file.touch(exist_ok=True)
            with open(self.state_file, "w") as fp:
                json.dump({"path": None, "positions": 0, "done_workers": []}, fp)

    # ────────────────────────────────────────────────────────────────────────────
    def __iter__(self):
        from constants import CODERS  # late import (Torch dataloader quirk)

        info = get_worker_info()
        wid = info.id
        n_workers = info.num_workers

        while True:
            # Acquire lock, read state
            with open(self.lock_file, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    state = json.load(open(self.state_file, "r"))
                except json.JSONDecodeError:
                    fcntl.flock(lf, fcntl.LOCK_UN)
                    time.sleep(0.1)
                    continue

                # Worker-0 fetches a new job if none in progress
                if state["path"] is None and wid == 0:
                    try:
                        job = self.job_queue.get(timeout=1.0)
                        if job is None:                    # sentinel: end-of-jobs
                            state = {"path": "STOP"}
                        else:
                            pth, npos = job
                            state = {"path": pth, "positions": npos,
                                     "done_workers": []}
                        json.dump(state, open(self.state_file, "w"))
                    except Empty:
                        pass
                fcntl.flock(lf, fcntl.LOCK_UN)

            # If no job yet, idle briefly
            if state.get("path") is None:
                time.sleep(0.5)
                continue

            # End-of-epoch sentinel
            if state["path"] == "STOP":
                break

            # Split current .bag among workers
            bag_path = state["path"]
            total = state["positions"]
            chunk = (total + n_workers - 1) // n_workers
            start, end = wid * chunk, min((wid + 1) * chunk, total)

            if start < end:
                try:
                    reader = bagz.BagFileReader(bag_path)
                    for i in range(start, end):
                        try:
                            fen, move, win_prob = CODERS["action_value"].decode(reader[i])
                            
                            if win_prob > 0.015 and win_prob < 0.985 and random.random() > 0.01:
                                continue 
                                
                            board = chess.Board(fen)
                            board.push(chess.Move.from_uci(move))
                            
                            bucket_idx = calculate_descriptor_index(board)
                            vec = vectorize_position(board)

                            yield (vec, 1.0 - win_prob, bucket_idx)
                        except Exception:
                            continue
                except Exception as exc:
                    print(f"[WORKER {wid}] ERROR processing {bag_path}: {exc}")

            # Mark this worker done
            with open(self.lock_file, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                state = json.load(open(self.state_file))
                if state.get("path") == bag_path and wid not in state["done_workers"]:
                    state["done_workers"].append(wid)
                    if len(state["done_workers"]) == n_workers:
                        self.processed_queue.put(bag_path)
                        state = {"path": None, "positions": 0, "done_workers": []}
                    json.dump(state, open(self.state_file, "w"))
                fcntl.flock(lf, fcntl.LOCK_UN)

            # Wait until job changes
            while True:
                try:
                    state = json.load(open(self.state_file))
                except json.JSONDecodeError:
                    state = {"path": bag_path}
                if state.get("path") != bag_path:
                    break
                time.sleep(0.1)


# ────────────────────────────────────────────────────────────────────────────────
def collate(batch):
    """Convert a list of samples into batched tensors."""
    pieces, sides, eps, castles, fifties, targets, descr = [], [], [], [], [], [], []
    for vec, win, bucket in batch:
        piece_arr, ep_vec, castle_vec, side_vec, fifty_vec = vec
        pieces.append(torch.from_numpy(piece_arr.astype(np.int64)))
        sides.append(int(side_vec[0]))
        eps.append(np.argmax(ep_vec) if ep_vec.sum() else -1)
        castles.append(torch.from_numpy(castle_vec.astype(np.float32)).reshape(4))
        fifties.append(float(fifty_vec[0]))
        targets.append(torch.tensor([win], dtype=torch.float32))
        descr.append(bucket)

    return (
        torch.stack(pieces),
        torch.tensor(sides, dtype=torch.long),
        torch.tensor(eps, dtype=torch.long),
        torch.stack(castles),
        torch.tensor(fifties, dtype=torch.float32),
        torch.stack(targets),
        torch.tensor(descr, dtype=torch.long),
    )
