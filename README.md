# Deep-Formix Chess Engine

**Deep-Formix** is a high-performance, GPU-accelerated chess engine written in C++ and CUDA. It features a novel NNUE (Efficiently Updatable Neural Network) evaluation system that utilizes 32 distinct models ("buckets"). The specific model used for evaluation is determined by the **total piece count, pawn count, and the presence of queens on the board**.

Remarkably, the neural network evaluation, **without performing any search**, plays at a level only up to **40 ELO points below Stockfish running a 1-ply search**. This demonstrates the profound positional understanding captured by the deep neural network architecture.

---

## The Core Idea: Searchless, High-Level Play

The central philosophy of deep-formix is inspired by recent advancements in ["searchless" chess engines](https://github.com/google-deepmind/searchless_chess), which aim to achieve Grandmaster-level play through pure pattern recognition rather than deep, brute-force calculation.

Instead of relying on a multi-ply search to reach a quiet (quiescent) position, deep-formix uses a very deep neural network to evaluate any given position instantly. The goal is to emulate the intuition of a top player: to "feel" the best move without exhaustive calculation.

Our network architecture is a **12-layer deep NNUE with 32x32 layers and skip connections**. [Skip connections](https://medium.com/@iamchaudukien/skip-connection-and-explanation-of-resnet-b32fe84ba32e) (the core feature of ResNet architectures) are crucial for training such deep networks effectively, as they help prevent the vanishing gradient problem and allow for a more direct flow of information across layers. This deep structure allows the engine to learn complex positional nuances that shallower networks might miss.

## Foundational Technologies

This engine was born from the synthesis of two specialized open-source concepts:

1.  **High-Speed Move Generation:** The engine's core move generation is built upon the principles of a [`gpu perft` repository](https://www.chessprogramming.org/Perft). Perft (Performance Test) is a method to verify a move generator's correctness by counting all possible moves to a certain depth. Using a GPU-accelerated perft design ensures the move generator is not only extremely fast but also rigorously validated.

2.  **GPU Chess Engine Framework:** The project's structure is influenced by public [`GPU Chess Engine` repositories](https://github.com/dkozykowski/Chess-Engine-GPU), which provide a framework for running engine logic on a GPU using CUDA. We have replaced the traditional evaluation function in such frameworks with our unique deep NNUE.

## Key Features

*   **GPU-Accelerated:** Core search and evaluation are implemented in C++/CUDA to leverage modern GPU hardware for massive parallelism.
*   **Deep NNUE with Skip Connections:** A 12-layer deep network allows for a sophisticated understanding of chess positions without requiring a deep search.
*   **NNUE32 Buckets:** The engine uses 32 separate neural network models, selected based on the number of pieces, pawns, and the presence of queens, to provide expert evaluation for any phase of the game.
*   **UCI Compatible:** Implements the Universal Chess Interface (UCI) for compatibility with standard chess GUIs and tournament software.

## License

The code authored for the deep-formix project is licensed under the **MIT License**.

However, this project was initially influenced by repositories like "GPU Chess Engine," which may be licensed under the GNU General Public License (GPL). While the original code has been almost entirely rewritten, its derivative status is unclear. If this project is legally considered a derivative work, it would consequently fall under the **GNU GPL**.
