#!/usr/bin/env python3
import sys
import chess
import torch
import numpy as np
import json
import ast
from pathlib import Path

try:
    from model import StackedNNUE
except ImportError:
    print("FATAL: Could not import StackedNNUE from model.py.", file=sys.stderr)
    sys.exit(1)

class NNUE32Engine:
    def __init__(self, checkpoint_path, mapping_path="bucket_mapping.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Suppress the "loading" message when used by the debug script
        if __name__ == "__main__":
             print(f"Engine: Loading model on {self.device}...", file=sys.stderr)

        self.net = StackedNNUE().to(self.device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.net.load_state_dict(checkpoint["model"])
            self.net.eval()
        except Exception as e:
            print(f"FATAL: Failed to load checkpoint '{checkpoint_path}': {e}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(mapping_path, 'r') as f:
                raw_map = json.load(f)
            self.bucket_map = {ast.literal_eval(k): v for k, v in raw_map.items()}
        except Exception as e:
            print(f"FATAL: Failed to load bucket mapping '{mapping_path}': {e}", file=sys.stderr)
            sys.exit(1)

        self.piece_to_index = {p: i for i, p in enumerate('PNBRQKpnbrqk')}
        if __name__ == "__main__":
            print("Engine: Model loaded successfully!", file=sys.stderr)

    def _calculate_descriptor_index(self, board: chess.Board) -> int:
        has_queen = bool(board.pieces(chess.QUEEN, chess.WHITE) or board.pieces(chess.QUEEN, chess.BLACK))
        piece_count = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)) + \
                      len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                      len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                      len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
        key = (has_queen, piece_count, pawn_count)
        return self.bucket_map.get(key, 0)

    def vectorize_board(self, board: chess.Board):
        """
        Converts a chess.Board object to a batch of tensors for the model.
        Shapes must be correct for a batch size of 1.
        """
        # Piece Index Tensor: Shape [1, 64]
        piece_idx = np.full(64, -1, dtype=np.int64)
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                piece_idx[sq] = self.piece_to_index[p.symbol()]
        pieces = torch.from_numpy(piece_idx).unsqueeze(0).to(self.device)

        # Side to Move Flag: Shape [1]
        side_flag = torch.tensor([int(board.turn)], dtype=torch.long).to(self.device)

        # En-Passant File: Shape [1]
        ep_file_idx = chess.square_file(board.ep_square) if board.ep_square else -1
        ep_file = torch.tensor([ep_file_idx], dtype=torch.long).to(self.device)

        # Castling Rights Mask: Shape [1, 4]
        castle_ms = np.array([
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)
        ], dtype=np.float32)
        castles = torch.from_numpy(castle_ms).unsqueeze(0).to(self.device)

        # 50-Move Rule Alpha: Shape [1]
        fifty_a_val = min(board.halfmove_clock / 100.0, 1.0)
        fifty_a = torch.tensor([fifty_a_val], dtype=torch.float32).to(self.device)
        
        # Descriptor Index: Shape [1]
        descriptor_index = torch.tensor([self._calculate_descriptor_index(board)], dtype=torch.long).to(self.device)

        return pieces, side_flag, ep_file, castles, fifty_a, descriptor_index

    def get_detailed_eval(self, board: chess.Board):
        """
        Performs a forward pass and captures intermediate layer sums.
        """
        activation_sums = {}
        hooks = []

        def get_sum_hook(name):
            def hook(model, input, output):
                activation_sums[name] = output.detach().sum().item()
            return hook

        descriptor_index_val = self._calculate_descriptor_index(board)
        network_module = self.net.networks[descriptor_index_val]
        embedding_module = self.net.embeddings[descriptor_index_val]

        # Register hooks
        hooks.append(network_module.fc1.register_forward_hook(get_sum_hook('fc1_pre')))
        hooks.append(network_module.ln1.register_forward_hook(get_sum_hook('ln1_post')))
        hooks.append(network_module.fc2.register_forward_hook(get_sum_hook('fc2_pre')))
        hooks.append(network_module.ln2.register_forward_hook(get_sum_hook('ln2_post')))
        for i, block in enumerate(network_module.blocks):
            hooks.append(block.register_forward_hook(get_sum_hook(f'block_{i}_post')))
        hooks.append(network_module.fc_out.register_forward_hook(get_sum_hook('fc_out_pre')))

        # Perform forward pass
        pieces, sides, eps, castles, fifties, descr = self.vectorize_board(board)
        
        with torch.no_grad():
            x_emb = self.net._compute_x_emb(embedding_module, pieces, sides, eps, castles, fifties)
            activation_sums['x_emb_sum'] = x_emb.sum().item()
            logits, p_win_tensor = self.net(pieces, sides, eps, castles, fifties, descr)
            p_win = p_win_tensor.item()
        
        for h in hooks: h.remove()

        # Assemble sums in the correct order to match C++
        ordered_sums = [
            activation_sums.get('x_emb_sum', 0.0),
            activation_sums.get('fc1_pre', 0.0),
            activation_sums.get('ln1_post', 0.0),
            activation_sums.get('fc2_pre', 0.0),
            activation_sums.get('ln2_post', 0.0),
        ]
        ordered_sums.extend([activation_sums.get(f'block_{i}_post', 0.0) for i in range(12)])
        ordered_sums.append(activation_sums.get('fc_out_pre', 0.0))
        ordered_sums.append(p_win)
        
        # Calculate final score
        if p_win >= 0.999: cp = 10000
        elif p_win <= 0.001: cp = -10000
        else: cp = int(290.68 * np.tan(3.096 * (p_win - 0.5)))
        
        return cp, ordered_sums
