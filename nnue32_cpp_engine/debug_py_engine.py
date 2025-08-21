#!/usr/bin/env python3
import sys
import chess
from nnue32_engine import NNUE32Engine

# --- CONFIGURATION ---
NNUE_CHECKPOINT = "nnuew_resnet_step_01105000.pt"
MAPPING_PATH = "bucket_mapping.json"

def analyze_position(fen):
    """
    Analyzes all legal moves from a FEN and prints detailed evaluations.
    """
    print("Python Engine: Loading model...", file=sys.stderr)
    engine = NNUE32Engine(NNUE_CHECKPOINT, MAPPING_PATH)
    print("Python Engine: Model loaded.", file=sys.stderr)
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        print("No legal moves in this position.")
        return

    print(f"info string Analyzing {len(legal_moves)} legal moves for FEN: {fen}")
    
    move_evals = []
    for move in legal_moves:
        board.push(move)
        
        # Get detailed evaluation for the position after the move
        # Note: The evaluation is from the opponent's perspective
        cp, layer_sums = engine.get_detailed_eval(board)
        
        # We want our score, so we negate the opponent's score
        our_cp = -cp
        
        print(f"info string move {move.uci()} score cp {our_cp} layers {' '.join(f'{s:.6f}' for s in layer_sums)}")
        
        move_evals.append((move.uci(), our_cp))
        board.pop()

    # Sort and print the best move found
    best_move = max(move_evals, key=lambda item: item[1])
    print(f"\nBest move found: {best_move[0]} with score {best_move[1]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_py_engine.py \"<FEN_STRING>\"", file=sys.stderr)
        sys.exit(1)
    
    fen_string = sys.argv[1]
    try:
        chess.Board(fen_string)
    except ValueError:
        print(f"Error: Invalid FEN string provided: \"{fen_string}\"", file=sys.stderr)
        sys.exit(1)
        
    analyze_position(fen_string)
