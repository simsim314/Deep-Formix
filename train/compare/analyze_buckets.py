#!/usr/bin/env python3
"""
analyze_buckets.py - Reads 'bucket_mapping.json' and systematically
searches for "holes" in the dataset.

For each hole, it finds the closest existing neighbor, searching first for
fewer pieces, and if that fails, searching for more pieces.
"""

import json
import ast
from collections import defaultdict

def find_holes_with_two_way_context(filepath="bucket_mapping_fixed.json"):
    """
    Reads the mapping file and reports on missing configurations and their
    closest existing neighbors using a two-way search.
    """
    print("")
    print("=" * 80)
    print("      DATASET HOLE & BI-DIRECTIONAL NEIGHBOR ANALYSIS      ")
    print("=" * 80)

    try:
        with open(filepath, 'r') as f:
            raw_mapping = json.load(f)
    except FileNotFoundError:
        print(f"\n[ERROR] The file '{filepath}' was not found in this directory.")
        return
    except json.JSONDecodeError:
        print(f"\n[ERROR] The file '{filepath}' is not a valid JSON file.")
        return

    # --- Data Processing for Efficient Lookup ---
    print("Parsing configurations and creating lookup tables...")
    
    existing_configs_set = set()
    configs_by_key = defaultdict(list)

    for key_str, bucket_idx in raw_mapping.items():
        try:
            config_tuple = ast.literal_eval(key_str)
            existing_configs_set.add(config_tuple)
            
            has_queen, piece_count, pawn_count = config_tuple
            lookup_key = (pawn_count, has_queen)
            configs_by_key[lookup_key].append((piece_count, bucket_idx))
            
        except (ValueError, SyntaxError):
            continue

    # Sort each neighbor list by piece_count descending. This is crucial.
    for key in configs_by_key:
        configs_by_key[key].sort(key=lambda x: x[0], reverse=True)
        
    print(f"Found {len(existing_configs_set)} unique configurations. Starting analysis...\n")

    # --- Hole Analysis ---
    for pawn_count in range(17):
        for has_queen in [True, False]:
            
            queen_status_str = "Has Queen" if has_queen else "No Queen"
            
            if has_queen:
                theoretical_max_pieces = 32 - pawn_count
            else:
                theoretical_max_pieces = 30 - pawn_count
            
            print(f"--- Analyzing: {pawn_count} Pawns, {queen_status_str} (Max Pieces: {theoretical_max_pieces}) ---")
            
            # Get the pre-sorted list of neighbors for this category
            neighbors = configs_by_key.get((pawn_count, has_queen), [])
            
            # Iterate downwards from the theoretical max
            for piece_count in range(max(0, theoretical_max_pieces), -1, -1):
                key_to_check = (has_queen, piece_count, pawn_count)
                
                if key_to_check in existing_configs_set:
                    continue # This is not a hole, skip it

                # --- This is a HOLE. Now find the closest neighbor. ---
                print(f"  [HOLE FOUND] Config: (Queen={has_queen}, Pieces={piece_count}, Pawns={pawn_count})")
                
                # 1. Try to find the closest neighbor with FEWER pieces.
                closest_smaller = None
                for neighbor_pieces, neighbor_bucket in neighbors:
                    if neighbor_pieces < piece_count:
                        closest_smaller = (neighbor_pieces, neighbor_bucket)
                        break # Found it, stop searching

                if closest_smaller:
                    # Success: found a smaller neighbor.
                    print(f"    -> Closest Existing (↓): {closest_smaller[0]} pieces in Bucket {closest_smaller[1]}")
                else:
                    # 2. Failure: No smaller neighbor exists. Try to find the closest with MORE pieces.
                    # Since the 'neighbors' list is sorted descending, the last element is the smallest existing piece count.
                    closest_larger = None
                    if neighbors and neighbors[-1][0] > piece_count:
                        closest_larger = neighbors[-1]
                    
                    if closest_larger:
                        print(f"    -> Closest Existing (↑): {closest_larger[0]} pieces in Bucket {closest_larger[1]}")
                    else:
                        # This case means the entire category is empty.
                        print(f"    -> No configurations found for this entire pawn/queen category.")

            print("") # Blank line for readability

    print("="*80)
    print("Hole analysis complete.")
    print("="*80)

if __name__ == "__main__":
    find_holes_with_two_way_context()
