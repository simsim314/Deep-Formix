#!/usr/bin/env python3
import torch, numpy as np, json, ast
from model import StackedNNUE
CHECKPOINT_PATH = "nnuew_resnet_step_01105000.pt"
MAPPING_PATH = "bucket_mapping.json"
MODEL_OUTPUT_PATH = "nnuew_resnet.bin"
MAPPING_OUTPUT_PATH = "mapping.bin"
def get_model_tensors(model):
    state_dict = model.state_dict()
    tensors = []
    for i in range(32):
        prefix_emb = f'embeddings.{i}.'
        tensors.extend([state_dict[prefix_emb + n] for n in ['W_white_piece', 'W_black_piece', 'W_white_castle', 'W_black_castle', 'W_white_ep', 'W_black_ep', 'W_white_fifty', 'W_black_fifty']])
        prefix_net = f'networks.{i}.'
        tensors.extend([state_dict[prefix_net + n] for n in ['fc1.weight', 'fc1.bias', 'ln1.weight', 'ln1.bias', 'fc2.weight', 'fc2.bias', 'ln2.weight', 'ln2.bias']])
        for j in range(12):
            prefix_block = prefix_net + f'blocks.{j}.'
            tensors.extend([state_dict[prefix_block + n] for n in ['fc1.weight', 'fc1.bias', 'ln1.weight', 'ln1.bias', 'fc2.weight', 'fc2.bias', 'ln2.weight', 'ln2.bias']])
        tensors.extend([state_dict[prefix_net + 'fc_out.weight'], state_dict[prefix_net + 'fc_out.bias']])
    return tensors
def export_model():
    print(f"Loading model from '{CHECKPOINT_PATH}'..."); model = StackedNNUE(); checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu')); model.load_state_dict(checkpoint['model']); model.eval()
    tensors = get_model_tensors(model)
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        for t in tensors: f.write(t.detach().cpu().numpy().astype(np.float32).tobytes())
    print(f"Successfully wrote {len(tensors)} tensors to '{MODEL_OUTPUT_PATH}'.")
def create_map_key(config_tuple):
    has_queen, piece_count, pawn_count = config_tuple
    return np.uint64((int(has_queen) << 56) | (piece_count << 48) | (pawn_count << 40))
def export_mapping():
    print(f"Loading mapping from '{MAPPING_PATH}'...");
    with open(MAPPING_PATH, 'r') as f: raw_map = json.load(f)
    parsed_map = {ast.literal_eval(k): v for k, v in raw_map.items()}
    with open(MAPPING_OUTPUT_PATH, 'wb') as f:
        for config_tuple, bucket_idx in parsed_map.items(): f.write(create_map_key(config_tuple).tobytes()); f.write(np.int32(bucket_idx).tobytes())
    print(f"Successfully wrote {len(parsed_map)} entries to '{MAPPING_OUTPUT_PATH}'.")
if __name__ == '__main__':
    export_model(); print("-" * 50); export_mapping()
