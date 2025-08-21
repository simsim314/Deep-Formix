#!/usr/bin/env python3
"""
fix_buckets.py - Reads 'bucket_mapping.json', fills all theoretical
holes by assigning them the bucket of their closest neighbor, and saves
the result to a new file, 'bucket_mapping_fixed.json'.
"""

import json
import ast
from collections import defaultdict

def fix_holes_and_create_new_mapping(
    input_filepath="bucket_mapping.json", 
    output_filepath="bucket_mapping_fixed.json"
):
    """
    Reads the input mapping, fills holes, and saves to the output file.
    """
    print("")
    print("=" * 80)
    print("      BUCKET MAP HOLE-FILLING UTILITY      ")
    print("=" * 80)

    try:
        with open(input_filepath, 'r') as f:
            raw_mapping = json.load(f)
    except FileNotFoundError:
        print(f"\n[ERROR] Input file '{input_filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"\n[ERROR] Input file '{input_filepath}' is not a valid JSON file.")
        return

    # --- Data Processing for Efficient Lookup ---
    print(f"Reading and parsing '{input_filepath}'...")
    
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

    for key in configs_by_key:
        configs_by_key[key].sort(key=lambda x: x[0], reverse=True)
        
    print(f"Found {len(existing_configs_set)} unique configurations. Starting hole-filling process...\n")
    
    # Start with a copy of the original map. We will add to this.
    fixed_mapping = raw_mapping.copy()
    holes_fixed_count = 0

    # --- Hole-Finding and Fixing Loop ---
    for pawn_count in range(17):
        for has_queen in [True, False]:
            if has_queen:
                theoretical_max_pieces = 32 - pawn_count
            else:
                theoretical_max_pieces = 30 - pawn_count
            
            neighbors = configs_by_key.get((pawn_count, has_queen), [])
            
            for piece_count in range(max(0, theoretical_max_pieces), -1, -1):
                key_to_check = (has_queen, piece_count, pawn_count)
                
                if key_to_check in existing_configs_set:
                    continue # Not a hole

                # --- This is a HOLE. Find its bucket. ---
                bucket_to_assign = None
                
                # 1. Search for closest smaller neighbor
                for neighbor_pieces, neighbor_bucket in neighbors:
                    if neighbor_pieces < piece_count:
                        bucket_to_assign = neighbor_bucket
                        break
                
                # 2. If no smaller, search for closest larger neighbor
                if bucket_to_assign is None and neighbors:
                    # The last element is the smallest existing piece count.
                    # If it's larger than our hole, it's the closest upward neighbor.
                    if neighbors[-1][0] > piece_count:
                        bucket_to_assign = neighbors[-1][1]

                # 3. If a bucket was found, add the fix to our map.
                if bucket_to_assign is not None:
                    hole_key_str = str(key_to_check)
                    fixed_mapping[hole_key_str] = bucket_to_assign
                    holes_fixed_count += 1
    
    # --- Save the Final Result ---
    print("\nProcess complete. Saving new file...")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(fixed_mapping, f, indent=2)
    except IOError as e:
        print(f"\n[ERROR] Could not write to output file '{output_filepath}': {e}")
        return

    print("-" * 80)
    print("SUMMARY:")
    print(f"  - Original Configurations: {len(raw_mapping)}")
    print(f"  - Holes Filled:            {holes_fixed_count}")
    print(f"  - Total Configurations:    {len(fixed_mapping)}")
    print(f"\nSuccessfully created new file: '{output_filepath}'")
    print("=" * 80)


if __name__ == "__main__":
    fix_holes_and_create_new_mapping()
