# Self-Pruning Neural Network (CIFAR-10)

This repository contains an implementation of a dynamic, "Self-Pruning" neural network designed to address real-world constraints in memory and computational budgets. Instead of traditional post-training compression, this model features a built-in mechanism to identify and remove its own weakest connections during the training process.

##  Key Features
- **Custom `PrunableLinear` Layer:** A ground-up implementation that replaces standard linear layers with gated weight mechanisms.
- **Dynamic Gating:** Uses learnable "gate scores" and a Sigmoid transformation to act as binary-like switches (0 to 1) for every weight connection.
- **Sparsity Regularization:** A custom training loop that optimizes a joint loss function: 
  `Total Loss = Classification Loss + λ * Sparsity Loss (L1 Norm)`.
- **Resource Efficient:** Optimized for deployment on memory-constrained devices and CPU-only inference.

##  Experimental Results
The model was tested on the CIFAR-10 dataset using an MLP architecture. The results demonstrate a clear trade-off between network density and classification performance controlled by the hyperparameter `λ`.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 0.00005 | 53.67% | 59.37% |
| **0.00010** | **52.02%** | **81.32%** |
| 0.00020 | 47.39% | 93.48% |

> **Conclusion:** At the optimal λ of 0.0001, the network achieved **81.32% sparsity**—effectively removing over 80% of its connections—while maintaining stable performance, proving that the architecture can successfully identify and preserve critical pathways.

##  Technical Implementation
- **Framework:** PyTorch, NumPy
- **Architecture:** Feed-forward Neural Network (MLP)
- **Gate Logic:** `gates = sigmoid(5.0 * gate_scores)`
- **Optimization:** Adam Optimizer for simultaneous weight and gate-score updates.

##  Gate Distribution
The success of the pruning mechanism is evidenced by a bimodal distribution of gate values:
1. **The Zero Spike:** A large majority of gates sit at exactly 0.0, representing pruned weights.
2. **The Survivor Cluster:** A secondary cluster of active gates representing the essential weights preserved for inference.

---
*Developed for the Tredence Studio AI Engineering Case Study (2026).*
