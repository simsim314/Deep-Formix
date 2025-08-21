import torch
import numpy as np
import json
import ast
import sys
import chess
from model import StackedNNUE

# --- CONFIGURATION ---
CHECKPOINT_PATH = "nnuew_resnet_step_01105000.pt"
MAPPING_PATH = "bucket_mapping.json"

# --- HELPER FUNCTIONS ---
def get_features_from_fen(fen):
    board = chess.Board(fen)
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    piece_idx = np.full(64, -1, dtype=np.int64)
    for sq_idx in range(64):
        piece = board.piece_at(sq_idx)
        if piece:
            piece_idx[sq_idx] = piece_to_index[piece.symbol()]
            
    side_flag = np.array([0 if board.turn == chess.WHITE else 1], dtype=np.int64)
    
    ep_file_idx = -1
    if board.ep_square:
        ep_file_idx = chess.square_file(board.ep_square)
    ep_file = np.array([ep_file_idx], dtype=np.int64)
    
    castle_ms = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ], dtype=np.float32)

    fifty_a = np.array([min(board.halfmove_clock / 100.0, 1.0)], dtype=np.float32)
    
    return (torch.from_numpy(piece_idx).unsqueeze(0), torch.from_numpy(side_flag),
            torch.from_numpy(ep_file), torch.from_numpy(castle_ms).unsqueeze(0), torch.from_numpy(fifty_a))

def get_bucket_from_fen(fen):
    board = chess.Board(fen)
    has_queen = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
    
    piece_count = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                  len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                  len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                  len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
                  
    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
    
    key = (has_queen, piece_count, pawn_count)
    with open(MAPPING_PATH, 'r') as f:
        raw_map = json.load(f)
    mapping = {ast.literal_eval(k): v for k, v in raw_map.items()}
    bucket = mapping.get(key)
    if bucket is None:
        raise ValueError(f"Could not find bucket for FEN: {fen} with key: {key}")
    return bucket

# --- MAIN DEBUGGING LOGIC ---
def run_test_for_fen(fen):
    model = StackedNNUE()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    piece_idx, side_flag, ep_file, castle_ms, fifty_a = get_features_from_fen(fen)
    bucket_idx = get_bucket_from_fen(fen)
    descriptor_index = torch.tensor([bucket_idx], dtype=torch.long)
    
    print(f"PY: Using Bucket Index: {bucket_idx}")
    
    embedding_module = model.embeddings[bucket_idx]
    network_module = model.networks[bucket_idx]
    
    print("--- PY: FORWARD PASS ---")
    x_emb = model._compute_x_emb(embedding_module, piece_idx, side_flag, ep_file, castle_ms, fifty_a)
    print(f"PY: SUM(x_emb)            = {x_emb.sum().item():.6f}")
    x = network_module.fc1(x_emb)
    print(f"PY: SUM(fc1 output)       = {x.sum().item():.6f}")
    x = network_module.gelu(x)
    x = network_module.ln1(x)
    print(f"PY: SUM(after gelu+ln1)   = {x.sum().item():.6f}")
    x = network_module.fc2(x)
    print(f"PY: SUM(fc2 output)       = {x.sum().item():.6f}")
    x = network_module.gelu(x)
    x = network_module.ln2(x)
    print(f"PY: SUM(after gelu+ln2)   = {x.sum().item():.6f}")
    for i, block in enumerate(network_module.blocks):
        x = block(x)
        print(f"PY: SUM(after block {i:2d})   = {x.sum().item():.6f}")
    x = network_module.gelu(x)
    logits = network_module.fc_out(x)
    print(f"PY: SUM(logits)           = {logits.sum().item():.6f}")
    p_win = (logits.softmax(-1) * model.bins).sum(-1, keepdim=True)
    print(f"PY: FINAL p(win)          = {p_win.item():.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_positions.py \"<FEN_STRING>\"")
        sys.exit(1)
    run_test_for_fen(sys.argv[1])
