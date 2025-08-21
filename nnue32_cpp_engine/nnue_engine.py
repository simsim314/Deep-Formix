#!/usr/bin/env python3
"""
Simple UCI chess engine using the trained NNUE network
"""

import sys
import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path

# Import the network class from your training script
from train_nnue import SideAwareNNUE

class NNUEEngine:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...", file=sys.stderr)
        
        # Load the network
        self.net = SideAwareNNUE().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.net.eval()
        
        # Piece mapping (same as in your preprocessing)
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        print("Model loaded successfully!", file=sys.stderr)
    
    def vectorize_position(self, board):
        """Convert board position to network input format"""
        # Piece array
        piece_array = np.full(64, -1, dtype=np.int8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_array[square] = self.piece_to_index[piece.symbol()]
        
        # En passant vector
        ep_vector = np.zeros(8, dtype=np.int8)
        if board.ep_square is not None:
            file = chess.square_file(board.ep_square)
            ep_vector[file] = 1
        
        # Castling rights
        castling_vector = np.array([
            [int(board.has_kingside_castling_rights(chess.WHITE))],
            [int(board.has_queenside_castling_rights(chess.WHITE))],
            [int(board.has_kingside_castling_rights(chess.BLACK))],
            [int(board.has_queenside_castling_rights(chess.BLACK))]
        ], dtype=np.int8)
        
        # Side to move
        side_vector = np.array([int(board.turn)], dtype=np.int8)
        
        # Fifty-move rule
        a = min(board.halfmove_clock / 100.0, 1.0)
        fifty_vector = np.array([a, 1.0 - a], dtype=np.float32)
        
        return piece_array, ep_vector, castling_vector, side_vector, fifty_vector
    
    def evaluate_position(self, board):
        """Evaluate a single position"""
        vec = self.vectorize_position(board)
        piece_arr, ep_vec, castle_vec, side_vec, fifty_vec = vec
        
        # Convert to tensors and add batch dimension
        pieces = torch.from_numpy(piece_arr.astype(np.int64)).unsqueeze(0).to(self.device)
        sides = torch.tensor([int(side_vec[0])], dtype=torch.long).to(self.device)
        eps = torch.tensor([np.argmax(ep_vec) if ep_vec.sum() else -1], dtype=torch.long).to(self.device)
        castles = torch.from_numpy(castle_vec.astype(np.float32)).reshape(1, 4).to(self.device)
        fifties = torch.tensor([float(fifty_vec[0])], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            _, p_win = self.net(pieces, sides, eps, castles, fifties)
            win_prob = p_win.item()
        
        # Convert win probability to centipawns (rough approximation)
        # 0.5 = 0 cp, 1.0 = +inf, 0.0 = -inf
        if win_prob >= 0.999:
            cp = 10000
        elif win_prob <= 0.001:
            cp = -10000
        else:
            cp = int(400 * np.log10(win_prob / (1 - win_prob)))
        
        return win_prob, cp
        
    def search_best_move(self, board, print_analysis=True):
        """Search for the best move by evaluating all legal moves"""
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        move_evals = []
        
        for move in legal_moves:
            # Make the move
            board.push(move)
            
            # Evaluate position after move (from opponent's perspective)
            win_prob, cp = self.evaluate_position(board)
            
            # Since we evaluated after our move, we need to invert the evaluation
            our_win_prob = 1 - win_prob
            our_cp = -cp
            
            move_evals.append((move, our_win_prob, our_cp))
            
            # Undo the move
            board.pop()
        
        # Sort by win probability (highest first)
        move_evals.sort(key=lambda x: x[1], reverse=True)
        
        if print_analysis:
            # Print ALL moves in sorted order
            for move, win_prob, cp in move_evals:
                print(f"Move: {move.uci():6} Win%: {win_prob*100:5.1f}% CP: {cp:+5d}", file=sys.stderr)
        
        best_move, best_win_prob, best_cp = move_evals[0]
        
        if print_analysis:
            print(f"\nBest move: {best_move.uci()} with Win%: {best_win_prob*100:.1f}% CP: {best_cp:+d}", file=sys.stderr)
            
            # Show the board after the best move
            board.push(best_move)
            print(f"\nBoard after {best_move.uci()}:", file=sys.stderr)
            print(board, file=sys.stderr)
            board.pop()
        
        return best_move
    
def uci_loop(engine):
    """Main UCI protocol loop"""
    board = chess.Board()
    
    while True:
        try:
            command = input().strip()
        except EOFError:
            break
            
        if command == "uci":
            print("id name NNUE Chess Engine")
            print("id author Your Name")
            print("uciok")
            
        elif command == "isready":
            print("readyok")
            
        elif command.startswith("position"):
            parts = command.split()
            
            if "startpos" in parts:
                board = chess.Board()
                moves_idx = parts.index("moves") if "moves" in parts else None
            elif "fen" in parts:
                fen_idx = parts.index("fen")
                fen_parts = []
                i = fen_idx + 1
                while i < len(parts) and parts[i] != "moves":
                    fen_parts.append(parts[i])
                    i += 1
                board = chess.Board(" ".join(fen_parts))
                moves_idx = parts.index("moves") if "moves" in parts else None
            
            if moves_idx is not None:
                for move_uci in parts[moves_idx + 1:]:
                    board.push(chess.Move.from_uci(move_uci))
                    
        elif command.startswith("go"):
            # Simple implementation - just find best move
            print(f"\nCurrent position:", file=sys.stderr)
            print(board, file=sys.stderr)
            print(f"\nAnalyzing moves...", file=sys.stderr)
            
            best_move = engine.search_best_move(board, print_analysis=True)
            
            if best_move:
                # Apply the best move to the board
                board.push(best_move)
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove (none)")
                
        elif command.startswith("move "):
            # Handle manual move input
            move_uci = command.split()[1]
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    print(f"Move {move_uci} applied", file=sys.stderr)
                    
                    # Automatically analyze and make engine's move
                    print(f"\nCurrent position after {move_uci}:", file=sys.stderr)
                    print(board, file=sys.stderr)
                    print(f"\nAnalyzing moves...", file=sys.stderr)
                    
                    best_move = engine.search_best_move(board, print_analysis=True)
                    if best_move:
                        board.push(best_move)
                        print(f"bestmove {best_move.uci()}")
                else:
                    print(f"Illegal move: {move_uci}", file=sys.stderr)
            except:
                print(f"Invalid move format: {move_uci}", file=sys.stderr)
                
        elif command == "quit":
            break
            
        elif command == "d":
            # Display current board (non-standard but useful)
            print(board, file=sys.stderr)
            win_prob, cp = engine.evaluate_position(board)
            print(f"Evaluation: Win%: {win_prob*100:.1f}% CP: {cp:+d}", file=sys.stderr)
def main():
    checkpoint_path = "checkpoints/attempt3_nnue_epoch007.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file {checkpoint_path} not found!", file=sys.stderr)
        sys.exit(1)
    
    engine = NNUEEngine(checkpoint_path)
    uci_loop(engine)

if __name__ == "__main__":
    main()
