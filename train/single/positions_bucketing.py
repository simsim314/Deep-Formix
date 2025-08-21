#!/usr/bin/env python3
"""
positions_bucketing.py

Analyse a sample of positions in .bag files, then greedily bin the resulting
(position-feature → percentage) table into exactly 32 buckets.

Each configuration key is a 3-tuple:
    (has_queen: bool, piece_count: int, pawn_count: int)

Output:
  • A JSON file  bucket_mapping.json   with   { key_tuple : bucket_index }.
  • A console report listing the contents of every bucket.

Requires:
  • python-chess
  • bagz
  • The constants.py file from your project (for CODERS).

Author: ChatGPT (2025-08-02)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import bagz

# ────────────────────────────────────────────────────────────────────────────────
# Project-specific decoder
try:
    from constants import CODERS
except ImportError:
    print("Error: Could not import CODERS from constants.py "
          "(make sure this script is inside your project).")
    exit(1)

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
WORK_DIR = "chess_pipeline"          # Folder with .bag files
POSITIONS_TO_ANALYZE = 1_000_000        # Sample size
NUM_BUCKETS = 32                     # Final number of buckets
MAPPING_JSON = "bucket_mapping.json" # Where to save the mapping

# ────────────────────────────────────────────────────────────────────────────────
def scan_positions() -> tuple[list[tuple[tuple[bool, int, int], int]], int]:
    """
    Scan .bag files under WORK_DIR, stop after POSITIONS_TO_ANALYZE positions,
    and return:
        • list of (key_tuple, count) pairs
        • total_positions_processed
    """
    work_dir = Path(WORK_DIR)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory '{WORK_DIR}' not found.")

    bag_files = sorted(work_dir.glob("*.bag"))
    if not bag_files:
        raise FileNotFoundError(f"No .bag files in '{WORK_DIR}'.")

    configuration_counts: Counter[tuple[bool, int, int]] = Counter()
    total_positions_processed = 0

    print(f"Found {len(bag_files)} .bag files – analysing "
          f"up to {POSITIONS_TO_ANALYZE} positions ...")

    try:
        for bag_path in bag_files:
            if total_positions_processed >= POSITIONS_TO_ANALYZE:
                break

            print(f"  • {bag_path.name}")
            reader = bagz.BagFileReader(str(bag_path))

            for idx in range(len(reader)):
                if total_positions_processed >= POSITIONS_TO_ANALYZE:
                    break

                fen, _, _ = CODERS["action_value"].decode(reader[idx])
                board = chess.Board(fen)

                has_queen = bool(
                    board.pieces(chess.QUEEN, chess.WHITE)
                    or board.pieces(chess.QUEEN, chess.BLACK)
                )

                # Major/minor pieces (excluding pawns & kings)
                rooks = len(board.pieces(chess.ROOK,   chess.WHITE)) \
                      + len(board.pieces(chess.ROOK,   chess.BLACK))
                knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) \
                        + len(board.pieces(chess.KNIGHT, chess.BLACK))
                bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) \
                        + len(board.pieces(chess.BISHOP, chess.BLACK))
                queens  = len(board.pieces(chess.QUEEN,  chess.WHITE)) \
                        + len(board.pieces(chess.QUEEN,  chess.BLACK))
                piece_count = rooks + knights + bishops + queens

                pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) \
                           + len(board.pieces(chess.PAWN, chess.BLACK))

                key = (has_queen, piece_count, pawn_count)
                configuration_counts[key] += 1
                total_positions_processed += 1

    except KeyboardInterrupt:
        print("Interrupted by user – proceeding with collected data.")

    if total_positions_processed == 0:
        raise RuntimeError("No positions read – quitting.")

    # Convert to list sorted by (has_queen, piece_count, pawn_count) – all desc.
    sorted_configs = sorted(
        configuration_counts.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
        reverse=True,
    )
    return sorted_configs, total_positions_processed


# ────────────────────────────────────────────────────────────────────────────────
def greedy_bucket(sorted_configs, total_positions, num_buckets=32):
    """
    Bucket (config, count) pairs into num_buckets using the greedy algorithm.
    Returns:
        • mapping_dict  : { config_tuple : bucket_idx }
        • buckets       : list[list[(config_tuple, count, pct)]]
    """
    # Convert counts → percentage (float)
    configs_pct = [
        (cfg, cnt, (cnt / total_positions) * 100.0) for cfg, cnt in sorted_configs
    ]

    mapping = {}
    buckets: list[list[tuple]] = []

    k = num_buckets                   # buckets remaining
    u = 100.0                         # % still unbucketed
    cursor = 0                        # index in configs_pct

    for bucket_idx in range(num_buckets):
        if cursor >= len(configs_pct):
            buckets.append([])        # remaining buckets empty
            continue

        m = u / k                     # target size this round
        current_bucket = []
        bucket_sum = 0.0

        # Always add at least one configuration
        cfg, cnt, pct = configs_pct[cursor]
        current_bucket.append((cfg, cnt, pct))
        bucket_sum += pct
        cursor += 1

        # Then keep adding until next would exceed m (and include that overshoot)
        while cursor < len(configs_pct):
            next_pct = configs_pct[cursor][2]
            if bucket_sum + next_pct > m:
                # Greedy overshoot rule: include the offending item,
                # then stop.  (If you'd rather stop *before* overshoot,
                # comment out the block below.)
                current_bucket.append(configs_pct[cursor])
                bucket_sum += next_pct
                cursor += 1
                break
            # Still room – add it
            current_bucket.append(configs_pct[cursor])
            bucket_sum += next_pct
            cursor += 1

        # Save assignments
        for cfg_entry, *_ in current_bucket:
            mapping[cfg_entry] = bucket_idx

        buckets.append(current_bucket)

        # Update remaining-bucket bookkeeping
        k -= 1
        u -= bucket_sum
        if k == 0:   # just in case
            break

    return mapping, buckets


# ────────────────────────────────────────────────────────────────────────────────
def save_mapping(mapping: dict, path: str):
    """Save mapping to JSON (keys need to be converted to strings)."""
    # Convert tuple keys → string for JSON compatibility
    serialisable = {str(k): v for k, v in mapping.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nMapping saved to '{path}' ({len(mapping)} configurations).")


# ────────────────────────────────────────────────────────────────────────────────
def print_bucket_report(buckets):
    """Pretty print every bucket’s contents and summary statistics."""
    print("\n" + "=" * 78)
    print("BUCKET REPORT (greedy, 32 buckets)\n")

    for idx, bucket in enumerate(buckets):
        pct_sum = sum(entry[2] for entry in bucket)
        cfg_count = len(bucket)
        print(f"Bucket {idx:02d} – {pct_sum:6.2f}% of positions "
              f"across {cfg_count} unique configs")
        print("-" * 78)
        print(f"{'Queen?':<9} | {'Pieces':<6} | {'Pawns':<5} | "
              f"{'Count':<7} | {'%':>6}")
        print("-" * 78)
        for (has_q, pcs, pwn), cnt, pct in bucket:
            qtxt = "Q" if has_q else "-"
            print(f"{qtxt:<9} | {pcs:<6} | {pwn:<5} | {cnt:<7} | {pct:6.2f}")
        print("-" * 78 + "\n")


# ────────────────────────────────────────────────────────────────────────────────
def main():
    sorted_configs, total = scan_positions()
    mapping, buckets = greedy_bucket(
        sorted_configs, total_positions=total, num_buckets=NUM_BUCKETS
    )
    save_mapping(mapping, MAPPING_JSON)
    print_bucket_report(buckets)


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
