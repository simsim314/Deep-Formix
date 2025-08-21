#include "estimator.hpp"
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision

// --- MAIN DEBUGGING LOGIC ---
int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "--- CPP: Loading Model ---" << std::endl;
    NNUE_Estimator estimator("nnuew_resnet.bin", "mapping.bin");
    
    std::cout << "\n--- CPP: Preparing Startpos Input ---" << std::endl;
    chess::Board board; // Defaults to startpos
    
    int bucket_idx = estimator.get_descriptor_index_public(board);
    std::cout << "CPP: Using Bucket Index: " << bucket_idx << std::endl;
    
    const auto& network = estimator.get_network_public(bucket_idx);
    
    std::cout << "\n--- CPP: FORWARD PASS ---" << std::endl;
    
    // 1. Compute Embedding
    Eigen::Vector<float, 2240> x_emb;
    estimator.compute_x_emb_public(x_emb, board);
    std::cout << "CPP: SUM(x_emb)            = " << x_emb.sum() << std::endl;
    
    // 2. First Linear Layer (fc1)
    Eigen::Vector<float, 256> x1 = (network.fc1_w * x_emb + network.fc1_b).unaryExpr(&gelu);
    std::cout << "CPP: SUM(fc1 output)       = " << x1.sum() << std::endl;

    // 3. First Activation + Norm
    layer_norm<256>(x1, network.ln1_w, network.ln1_b);
    std::cout << "CPP: SUM(after gelu+ln1)   = " << x1.sum() << std::endl;

    // 4. Second Linear Layer (fc2)
    Eigen::Vector<float, 32> x2 = (network.fc2_w * x1 + network.fc2_b).unaryExpr(&gelu);
    std::cout << "CPP: SUM(fc2 output)       = " << x2.sum() << std::endl;
    
    // 5. Second Activation + Norm
    layer_norm<32>(x2, network.ln2_w, network.ln2_b);
    std::cout << "CPP: SUM(after gelu+ln2)   = " << x2.sum() << std::endl;

    // 6. Residual Blocks
    for (int i = 0; i < 12; ++i) {
        network.blocks[i].forward(x2);
        std::cout << "CPP: SUM(after block " << std::setw(2) << i << ")   = " << x2.sum() << std::endl;
    }
    
    // 7. Final Activation + Output Layer
    Eigen::Vector<float, 51> logits = network.fc_out_w * x2.unaryExpr(&gelu) + network.fc_out_b;
    std::cout << "CPP: SUM(logits)           = " << logits.sum() << std::endl;
    
    // 8. Final Win Probability
    logits = (logits.array() - logits.maxCoeff()).exp();
    logits /= logits.sum();
    float p_win = logits.dot(estimator.get_bins_public());
    std::cout << "CPP: FINAL p(win)          = " << p_win << std::endl;
    
    return 0;
}
