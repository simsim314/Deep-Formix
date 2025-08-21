#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include "chess.hpp"

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
class NNUE_Estimator;

struct DetailedEvaluation {
    int score;
    std::vector<float> layer_sums;
};

struct ResidualBlock {
    RowMajorMatrixXf fc1_w, fc2_w;
    Eigen::VectorXf fc1_b, fc2_b, ln1_w, ln1_b, ln2_w, ln2_b;
    ResidualBlock();
    void forward(Eigen::Vector<float, 32>& x) const;
};

struct MLPHead {
    RowMajorMatrixXf fc1_w, fc2_w, fc_out_w;
    Eigen::VectorXf fc1_b, ln1_w, ln1_b, fc2_b, ln2_w, ln2_b, fc_out_b;
    std::array<ResidualBlock, 12> blocks;
    MLPHead();
    Eigen::Vector<float, 51> forward(const Eigen::Vector<float, 2240>& x_emb) const;
};

// CORRECTED STRUCTURE: Matches Python's BucketEmbedding
struct BucketEmbedding {
    RowMajorMatrixXf W_white_piece;
    RowMajorMatrixXf W_black_piece;
    RowMajorMatrixXf W_white_castle;
    RowMajorMatrixXf W_black_castle;
    RowMajorMatrixXf W_white_ep;
    RowMajorMatrixXf W_black_ep;
    RowMajorMatrixXf W_white_fifty;
    RowMajorMatrixXf W_black_fifty;
    BucketEmbedding();
};

class NNUE_Estimator {
public:
    NNUE_Estimator(const std::string& model_path, const std::string& mapping_path);
    int evaluate(const chess::Board& board) const;
    DetailedEvaluation evaluate_detailed(const chess::Board& board) const;
    int get_descriptor_index_public(const chess::Board& board) const { return get_descriptor_index(board); }
    const MLPHead& get_network_public(int idx) const { return *networks[idx]; }
    const Eigen::Vector<float, 51>& get_bins_public() const { return bins; }
    void compute_x_emb_public(Eigen::Vector<float, 2240>& x_emb, const chess::Board& board) const { compute_x_emb(x_emb, board); }
private:
    // CORRECTED STRUCTURE: One list of embeddings, not two.
    std::array<std::unique_ptr<BucketEmbedding>, 32> embeddings;
    std::array<std::unique_ptr<MLPHead>, 32> networks;
    std::unordered_map<uint64_t, int32_t> bucket_map;
    Eigen::Vector<float, 51> bins;

    void load_parameters(const std::string& path);
    void load_mapping(const std::string& path);
    uint64_t get_map_key(const chess::Board& board) const;
    int get_descriptor_index(const chess::Board& board) const;
    void compute_x_emb(Eigen::Vector<float, 2240>& x_emb, const chess::Board& board) const;
};
#endif
