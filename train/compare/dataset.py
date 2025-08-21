#!/usr/bin/env python3
"""
dataset.py – Final, robust, file-based orchestration with mating positions.

Changes (2025-08-02)
────────────────────
• NEW:   Loads bucket_mapping.json (32-bucket greedy mapping).
• NEW:   calculate_descriptor_index() first looks up the mapping:
key = (has_queen, piece_count, pawn_count)  ➜  bucket_idx
If the key or the JSON file is missing it falls back to the
original 5-bit heuristic, so training never crashes.
• NEW:   For extreme scores (>0.99 or <0.01), generates extensive training
data by analyzing the top 15 moves with Stockfish and following the
best line until mate.

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
from typing import Dict, Tuple, Optional, Generator
import random
import subprocess
import tempfile
import itertools

import bagz
import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

#────────────────────────────────────────────────────────────────────────────────
# Bucket mapping – load once at import time
#────────────────────────────────────────────────────────────────────────────────

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

#────────────────────────────────────────────────────────────────────────────────
# Stockfish configuration
#────────────────────────────────────────────────────────────────────────────────

STOCKFISH_PATH   = "/mnt/pacer/Projects/chess_trainer/engines/engines_classic/stockfish"
STOCKFISH_DEPTH = 8
MAX_MATE_PLY = 16 # Max moves to follow a mating sequence

#────────────────────────────────────────────────────────────────────────────────
# Vectorisation helpers
#────────────────────────────────────────────────────────────────────────────────

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

#────────────────────────────────────────────────────────────────────────────────
# Bucket descriptor calculation
#────────────────────────────────────────────────────────────────────────────────

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

#────────────────────────────────────────────────────────────────────────────────
# Dataset class
#────────────────────────────────────────────────────────────────────────────────

class StreamingBagDataset(IterableDataset):
    """
    Iterable dataset reading .bag files in parallel via a file-locked JSON state.
    Workers emit pairs of positions for Siamese training:
    ((vec1, score1, bucket1), (vec2, score2, bucket2))
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

    @staticmethod
    def _get_scored_positions(board: chess.Board, engine: chess.engine.SimpleEngine, num_moves: int, wid: int) -> list:
        """
        Analyzes a position to get the top `num_moves`, and returns a list of
        resulting positions with their scores.
        """
        positions = []
        try:
            analysis = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH), multipv=num_moves)
            if not analysis:
                return []

            for info in analysis:
                if 'pv' not in info or not info['pv']:
                    continue

                move = info['pv'][0]
                score_pov = info['score']
                relative_score = score_pov.relative

                score_for_mover = 0.5
                if relative_score.is_mate():
                    mates_in_plies = relative_score.mate()
                    # --- MODIFICATION: Changed multiplier from 0.0025 to 0.005 ---
                    if mates_in_plies > 0:  # We are delivering mate
                        score_for_mover = 1.0 - 0.005 * mates_in_plies
                    else:  # We are getting mated
                        score_for_mover = 0.005 * abs(mates_in_plies)
                else:
                    # It's a centipawn score.
                    cp = relative_score.score(mate_score=10000)
                    if cp is not None:
                        # Standard sigmoid conversion from centipawns to win prob [0, 1]
                        win_prob = 1.0 / (1.0 + 10**(-cp / 400.0))
                        
                        # --- MODIFICATION: Scale to [0.04, 0.96] ---
                        # new_score = 0.04 + (0.96 - 0.04) * win_prob
                        score_for_mover = 0.04 + 0.92 * win_prob

                next_board = board.copy()
                next_board.push(move)
                yield_score = 1.0 - score_for_mover

                bucket_idx = calculate_descriptor_index(next_board)
                vec = vectorize_position(next_board)
                positions.append((vec, yield_score, bucket_idx))

        except (chess.engine.EngineTerminatedError, BrokenPipeError) as e:
            print(f"[WORKER {wid}] Stockfish engine error during analysis: {e}")
        except Exception as e:
            print(f"[WORKER {wid}] Error during position analysis: {e}")

        return positions

    def __iter__(self):
        from constants import CODERS  # late import (Torch dataloader quirk)

        info = get_worker_info()
        wid = info.id if info else 0
        n_workers = info.num_workers if info else 1

        engine = None
        try:
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception as e:
            print(f"[WORKER {wid}] WARNING: Could not initialize Stockfish: {e}")

        while True:
            with open(self.lock_file, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    state = json.load(open(self.state_file, "r"))
                except json.JSONDecodeError:
                    fcntl.flock(lf, fcntl.LOCK_UN)
                    time.sleep(0.1)
                    continue

                if state["path"] is None and wid == 0:
                    try:
                        job = self.job_queue.get(timeout=1.0)
                        if job is None:
                            state = {"path": "STOP"}
                        else:
                            pth, npos = job
                            state = {"path": pth, "positions": npos,
                                     "done_workers": []}
                        json.dump(state, open(self.state_file, "w"))
                    except Empty:
                        pass
                fcntl.flock(lf, fcntl.LOCK_UN)

            if state.get("path") is None:
                time.sleep(0.5)
                continue

            if state["path"] == "STOP":
                break

            bag_path = state["path"]
            total = state["positions"]
            chunk = (total + n_workers - 1) // n_workers
            start, end = wid * chunk, min((wid + 1) * chunk, total)

            if start < end:
                try:
                    reader = bagz.BagFileReader(bag_path)
                    for i in range(start, end):
                        try:
                            fen, _, win_prob = CODERS["action_value"].decode(reader[i])
                            board = chess.Board(fen)

                            if not engine:
                                continue

                            # Condition 1: Top/Bottom 1% (Extreme scores) -> 45 pairs
                            if win_prob < 0.01 or win_prob > 0.99:
                                scored_positions = self._get_scored_positions(board, engine, 10, wid)
                                if len(scored_positions) < 2: continue
                                for pos1, pos2 in itertools.combinations(scored_positions, 2):
                                    yield (pos1, pos2)

                            # Condition 2: Top/Bottom 4% -> 10 random pairs
                            elif win_prob < 0.05 or win_prob > 0.95:
                                scored_positions = self._get_scored_positions(board, engine, 10, wid)
                                if len(scored_positions) < 2: continue
                                for _ in range(10):
                                    pos1, pos2 = random.sample(scored_positions, 2)
                                    yield (pos1, pos2)

                            # Condition 3: 10% of the rest -> 1 pair
                            else:
                                if random.random() < 0.1:
                                    scored_positions = self._get_scored_positions(board, engine, 2, wid)
                                    if len(scored_positions) == 2:
                                        yield (scored_positions[0], scored_positions[1])

                        except Exception as e:
                            print(f"[WORKER {wid}] Error processing position {i} in {bag_path}: {e}")
                            continue
                except Exception as exc:
                    print(f"[WORKER {wid}] ERROR processing file {bag_path}: {exc}")

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

            while True:
                try:
                    state = json.load(open(self.state_file))
                except json.JSONDecodeError:
                    state = {"path": bag_path}
                if state.get("path") != bag_path:
                    break
                time.sleep(0.1)

        if engine is not None:
            engine.quit()

#────────────────────────────────────────────────────────────────────────────────
# Collate function
#────────────────────────────────────────────────────────────────────────────────

def collate(batch):
    """
    Convert a list of paired samples into two sets of batched tensors
    for Siamese network training.
    """
    # Lists for the first element of each pair
    pieces1, sides1, eps1, castles1, fifties1, targets1, descr1 = [], [], [], [], [], [], []
    # Lists for the second element of each pair
    pieces2, sides2, eps2, castles2, fifties2, targets2, descr2 = [], [], [], [], [], [], []

    for sample1, sample2 in batch:
        # Unpack and process the first sample in the pair
        vec1, win1, bucket1 = sample1
        piece_arr1, ep_vec1, castle_vec1, side_vec1, fifty_vec1 = vec1
        pieces1.append(torch.from_numpy(piece_arr1.astype(np.int64)))
        sides1.append(int(side_vec1[0]))
        eps1.append(np.argmax(ep_vec1) if ep_vec1.sum() else -1)
        castles1.append(torch.from_numpy(castle_vec1.astype(np.float32)).reshape(4))
        fifties1.append(float(fifty_vec1[0]))
        targets1.append(torch.tensor([win1], dtype=torch.float32))
        descr1.append(bucket1)

        # Unpack and process the second sample in the pair
        vec2, win2, bucket2 = sample2
        piece_arr2, ep_vec2, castle_vec2, side_vec2, fifty_vec2 = vec2
        pieces2.append(torch.from_numpy(piece_arr2.astype(np.int64)))
        sides2.append(int(side_vec2[0]))
        eps2.append(np.argmax(ep_vec2) if ep_vec2.sum() else -1)
        castles2.append(torch.from_numpy(castle_vec2.astype(np.float32)).reshape(4))
        fifties2.append(float(fifty_vec2[0]))
        targets2.append(torch.tensor([win2], dtype=torch.float32))
        descr2.append(bucket2)

    # Create batched tensors for the first set
    batch1 = (
        torch.stack(pieces1),
        torch.tensor(sides1, dtype=torch.long),
        torch.tensor(eps1, dtype=torch.long),
        torch.stack(castles1),
        torch.tensor(fifties1, dtype=torch.float32),
        torch.stack(targets1),
        torch.tensor(descr1, dtype=torch.long),
    )

    # Create batched tensors for the second set
    batch2 = (
        torch.stack(pieces2),
        torch.tensor(sides2, dtype=torch.long),
        torch.tensor(eps2, dtype=torch.long),
        torch.stack(castles2),
        torch.tensor(fifties2, dtype=torch.float32),
        torch.stack(targets2),
        torch.tensor(descr2, dtype=torch.long),
    )

    return batch1, batch2
