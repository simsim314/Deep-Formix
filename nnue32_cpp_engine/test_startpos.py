import torch, numpy as np, json, ast
from model import StackedNNUE

CHECKPOINT_PATH = "nnuew_resnet_step_01105000.pt"
MAPPING_PATH = "bucket_mapping.json"
STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def get_piece_features_from_fen(fen):
    parts = fen.split(' ')
    board_str, turn = parts[0], parts[1]
    piece_to_index = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    piece_idx = np.full(64, -1, dtype=np.int64)
    rank, file = 7, 0
    for char in board_str:
        if char == '/':
            rank -= 1; file = 0
        elif char.isdigit():
            file += int(char)
        else:
            sq = rank * 8 + file; piece_idx[sq] = piece_to_index[char]; file += 1
    side_flag = np.array([0 if turn == 'w' else 1], dtype=np.int64)
    ep_file = np.array([-1], dtype=np.int64)
    castle_ms = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    fifty_a = np.array([0.0], dtype=np.float32)
    return (torch.from_numpy(piece_idx).unsqueeze(0), torch.from_numpy(side_flag),
            torch.from_numpy(ep_file), torch.from_numpy(castle_ms).unsqueeze(0), torch.from_numpy(fifty_a))

def get_startpos_bucket():
    key = (True, 14, 16)
    with open(MAPPING_PATH, 'r') as f:
        raw_map = json.load(f)
    mapping = {ast.literal_eval(k): v for k, v in raw_map.items()}
    return mapping.get(key, -1)

def main():
    print("--- PY: Loading Model ---")
    model = StackedNNUE()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("\n--- PY: Preparing Startpos Input ---")
    piece_idx, side_flag, ep_file, castle_ms, fifty_a = get_piece_features_from_fen(STARTPOS_FEN)
    bucket_idx = get_startpos_bucket()
    descriptor_index = torch.tensor([bucket_idx], dtype=torch.long)
    print(f"PY: Using Bucket Index: {bucket_idx}")
    embedding_module = model.embeddings[bucket_idx]
    network_module = model.networks[bucket_idx]
    print("\n--- PY: FORWARD PASS ---")
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
    main()
