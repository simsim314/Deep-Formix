#include "estimator.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <vector>

template<typename T>
void read_tensor(std::ifstream& file, T& matrix) {
    file.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(float));
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
}

template<int N>
void layer_norm(Eigen::Vector<float, N>& x, const Eigen::VectorXf& weight, const Eigen::VectorXf& bias) {
    float mean = x.mean();
    float rsqrt = 1.0f / std::sqrt((x.array() - mean).square().mean() + 1e-5f);
    x = ((x.array() - mean) * rsqrt) * weight.array() + bias.array();
}

ResidualBlock::ResidualBlock() {
    fc1_w.resize(32, 32); fc2_w.resize(32, 32);
    fc1_b.resize(32); fc2_b.resize(32);
    ln1_w.resize(32); ln1_b.resize(32);
    ln2_w.resize(32); ln2_b.resize(32);
}

MLPHead::MLPHead() {
    fc1_w.resize(256, 2240); fc1_b.resize(256);
    ln1_w.resize(256); ln1_b.resize(256);
    fc2_w.resize(32, 256); fc2_b.resize(32);
    ln2_w.resize(32); ln2_b.resize(32);
    fc_out_w.resize(51, 32); fc_out_b.resize(51);
}

BucketEmbedding::BucketEmbedding() {
    W_white_piece.resize(64 * 12, 32);
    W_black_piece.resize(64 * 12, 32);
    W_white_castle.resize(4, 32);
    W_black_castle.resize(4, 32);
    W_white_ep.resize(8, 32);
    W_black_ep.resize(8, 32);
    W_white_fifty.resize(2, 32);
    W_black_fifty.resize(2, 32);
}

void ResidualBlock::forward(Eigen::Vector<float, 32>& x) const {
    Eigen::Vector<float, 32> y = x;
    x = (fc1_w * x + fc1_b);
    x = x.unaryExpr(&gelu);
    layer_norm<32>(x, ln1_w, ln1_b);
    x = (fc2_w * x + fc2_b);
    x = x.unaryExpr(&gelu);
    layer_norm<32>(x, ln2_w, ln2_b);
    x += y;
}

Eigen::Vector<float, 51> MLPHead::forward(const Eigen::Vector<float, 2240>& x_emb) const {
    Eigen::Vector<float, 256> x1 = fc1_w * x_emb + fc1_b;
    x1 = x1.unaryExpr(&gelu);
    layer_norm<256>(x1, ln1_w, ln1_b);
    Eigen::Vector<float, 32> x2 = fc2_w * x1 + fc2_b;
    x2 = x2.unaryExpr(&gelu);
    layer_norm<32>(x2, ln2_w, ln2_b);
    for (const auto& block : blocks) {
        block.forward(x2);
    }
    x2 = x2.unaryExpr(&gelu);
    return fc_out_w * x2 + fc_out_b;
}

NNUE_Estimator::NNUE_Estimator(const std::string& model_path, const std::string& mapping_path) {
    for (int i = 0; i < 32; ++i) {
        embeddings[i] = std::make_unique<BucketEmbedding>();
        networks[i] = std::make_unique<MLPHead>();
    }
    bins.setLinSpaced(51, 0.0f, 1.0f);
    load_parameters(model_path);
    load_mapping(mapping_path);
}

void NNUE_Estimator::load_mapping(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open mapping file: " + path);
    uint64_t key;
    int32_t value;
    while (file.read(reinterpret_cast<char*>(&key), sizeof(key)) && file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        bucket_map[key] = value;
    }
}

void NNUE_Estimator::load_parameters(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open model file: " + path);
    for (int i = 0; i < 32; ++i) {
        auto& emb = *embeddings[i];
        read_tensor(file, emb.W_white_piece);
        read_tensor(file, emb.W_black_piece);
        read_tensor(file, emb.W_white_castle);
        read_tensor(file, emb.W_black_castle);
        read_tensor(file, emb.W_white_ep);
        read_tensor(file, emb.W_black_ep);
        read_tensor(file, emb.W_white_fifty);
        read_tensor(file, emb.W_black_fifty);

        auto& net = *networks[i];
        read_tensor(file, net.fc1_w); read_tensor(file, net.fc1_b);
        read_tensor(file, net.ln1_w); read_tensor(file, net.ln1_b);
        read_tensor(file, net.fc2_w); read_tensor(file, net.fc2_b);
        read_tensor(file, net.ln2_w); read_tensor(file, net.ln2_b);
        for (auto& block : net.blocks) {
            read_tensor(file, block.fc1_w); read_tensor(file, block.fc1_b);
            read_tensor(file, block.ln1_w); read_tensor(file, block.ln1_b);
            read_tensor(file, block.fc2_w); read_tensor(file, block.fc2_b);
            read_tensor(file, block.ln2_w); read_tensor(file, block.ln2_b);
        }
        read_tensor(file, net.fc_out_w); read_tensor(file, net.fc_out_b);
    }
}

uint64_t NNUE_Estimator::get_map_key(const chess::Board& board) const {
    bool has_queen = (board.pieces(chess::PieceType::QUEEN, chess::Color::WHITE).getBits() | board.pieces(chess::PieceType::QUEEN, chess::Color::BLACK).getBits()) != 0;
    int piece_count = board.pieces(chess::PieceType::ROOK).count()
                    + board.pieces(chess::PieceType::KNIGHT).count()
                    + board.pieces(chess::PieceType::BISHOP).count()
                    + board.pieces(chess::PieceType::QUEEN).count();
    int pawn_count = board.pieces(chess::PieceType::PAWN).count();
    return (static_cast<uint64_t>(has_queen) << 56) | (static_cast<uint64_t>(piece_count) << 48) | (static_cast<uint64_t>(pawn_count) << 40);
}

int NNUE_Estimator::get_descriptor_index(const chess::Board& board) const {
    auto it = bucket_map.find(get_map_key(board));
    return (it != bucket_map.end()) ? it->second : 0;
}

// FINAL CORRECTED LOGIC
void NNUE_Estimator::compute_x_emb(Eigen::Vector<float, 2240>& x_emb, const chess::Board& board) const {
    const int bucket_idx = get_descriptor_index(board);
    const bool is_white_turn = board.sideToMove() == chess::Color::WHITE;
    
    const auto& emb = *embeddings[bucket_idx];

    // *** THE FIX IS HERE ***
    // This logic is INVERTED to match the behavior of the "ground truth" Python script.
    // The Python script sends side_flag=1 for White's turn, which causes the model
    // to select the black embeddings. We replicate that exact behavior here.
    const auto& piece_table = is_white_turn ? emb.W_black_piece : emb.W_white_piece;
    const auto& castle_table = is_white_turn ? emb.W_black_castle : emb.W_white_castle;
    const auto& ep_table = is_white_turn ? emb.W_black_ep : emb.W_white_ep;
    const auto& fifty_table = is_white_turn ? emb.W_black_fifty : emb.W_white_fifty;

    auto pieces_vec = x_emb.head<2048>();
    pieces_vec.setZero();
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        auto piece = board.at(chess::Square(sq_idx));
        if (piece.type() != chess::PieceType::NONE) {
            int piece_idx = static_cast<int>(piece.type()) + (piece.color() == chess::Color::BLACK ? 6 : 0);
            pieces_vec.segment<32>(sq_idx * 32) = piece_table.row(sq_idx * 12 + piece_idx).transpose();
        }
    }

    auto castle_vec = x_emb.segment<128>(2048);
    auto ep_vec = x_emb.segment<32>(2176);
    auto fifty_vec = x_emb.tail<32>();
    
    ep_vec.setZero();
    
    auto cr = board.castlingRights();
    Eigen::Vector<float, 4> castle_rights;
    castle_rights << cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE),
                     cr.has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE),
                     cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE),
                     cr.has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    
    for(int i = 0; i < 4; ++i) {
        castle_vec.segment<32>(i * 32) = castle_rights[i] * castle_table.row(i).transpose();
    }
    
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        int file_idx = static_cast<int>(board.enpassantSq().file());
        ep_vec = ep_table.row(file_idx).transpose();
    }
    
    float fifty_a = std::min(static_cast<float>(board.halfMoveClock()) / 100.0f, 1.0f);
    fifty_vec = (1.0f - fifty_a) * fifty_table.row(0).transpose() + fifty_a * fifty_table.row(1).transpose();
}

int NNUE_Estimator::evaluate(const chess::Board& board) const {
    Eigen::Vector<float, 2240> x_emb;
    compute_x_emb(x_emb, board);
    int bucket_idx = get_descriptor_index(board);
    Eigen::Vector<float, 51> logits = networks[bucket_idx]->forward(x_emb);
    logits = (logits.array() - logits.maxCoeff()).exp();
    logits /= logits.sum();
    float p_win = logits.dot(bins);
    if (p_win > 0.999f) return 10000;
    if (p_win < 0.001f) return -10000;
    return static_cast<int>(290.68f * std::tan(3.096f * (p_win - 0.5f)));
}

DetailedEvaluation NNUE_Estimator::evaluate_detailed(const chess::Board& board) const {
    DetailedEvaluation result;
    std::vector<float>& sums = result.layer_sums;
    Eigen::Vector<float, 2240> x_emb;
    compute_x_emb(x_emb, board);
    sums.push_back(x_emb.sum());
    int bucket_idx = get_descriptor_index(board);
    const auto& network = *networks[bucket_idx];
    Eigen::Vector<float, 256> x1_pre_act = network.fc1_w * x_emb + network.fc1_b;
    sums.push_back(x1_pre_act.sum());
    Eigen::Vector<float, 256> x1 = x1_pre_act.unaryExpr(&gelu);
    layer_norm<256>(x1, network.ln1_w, network.ln1_b);
    sums.push_back(x1.sum());
    Eigen::Vector<float, 32> x2_pre_act = network.fc2_w * x1 + network.fc2_b;
    sums.push_back(x2_pre_act.sum());
    Eigen::Vector<float, 32> x2 = x2_pre_act.unaryExpr(&gelu);
    layer_norm<32>(x2, network.ln2_w, network.ln2_b);
    sums.push_back(x2.sum());
    for (int i = 0; i < 12; ++i) {
        network.blocks[i].forward(x2);
        sums.push_back(x2.sum());
    }
    x2 = x2.unaryExpr(&gelu);
    Eigen::Vector<float, 51> logits = network.fc_out_w * x2 + network.fc_out_b;
    sums.push_back(logits.sum());
    logits = (logits.array() - logits.maxCoeff()).exp();
    logits /= logits.sum();
    float p_win = logits.dot(bins);
    sums.push_back(p_win);
    if (p_win > 0.999f) result.score = 10000;
    else if (p_win < 0.001f) result.score = -10000;
    else result.score = static_cast<int>(290.68f * std::tan(3.096f * (p_win - 0.5f)));
    return result;
}
