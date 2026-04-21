# Project Report: The Self-Pruning Neural Network

## 1. Executive Summary
This project introduces a "Self-Pruning" neural network designed to address the challenges of deploying large models in environments with strict memory and computational budgets. Unlike traditional post-training pruning, this model features a built-in mechanism that identifies and removes its own weakest connections during the training process, adapting its architecture on the fly.

## 2. Problem Statement
In real-world AI deployment, memory and computational constraints often limit the use of sophisticated models. The objective was to design a neural network that learns which of its own weights are unnecessary through a gated connectivity mechanism and custom regularization.

## 3. Technical Implementation

### 3.1 Custom Prunable Linear Layer
The core of the architecture is a custom `PrunableLinear` layer. Each weight is associated with a learnable "gate" score. These scores are passed through a Sigmoid function with a temperature multiplier to create sharp, binary-like gates that multiply the weights during the forward pass.

### 3.2 Sparsity Regularization Loss
To encourage the network to prune itself, a custom loss function was implemented:
**Total Loss = Classification Loss + λ * Sparsity Loss**

The Sparsity Loss is defined as the L1 norm of all active gates. The L1 norm (sum of absolute values) is known to encourage sparsity by driving non-essential gate values to exactly zero.

## 4. Experimental Results (CIFAR-10)
The model was evaluated across various λ values to analyze the trade-off between accuracy and sparsity.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| 0.00005 | 53.67% | 59.37% |
| **0.00010** | **52.02%** | **81.32%** |
| 0.00020 | 47.39% | 93.48% |

**Key Finding:** At the optimal λ of 0.0001, the network removed **81.32%** of its parameters while maintaining a stable accuracy of **52.02%**, demonstrating high efficiency with minimal performance degradation.  

# Sparsity Plot 
![Gateway distribution](https://raw.githubusercontent.com/AmalFrancisOlakengil/Tredence_casestudy/refs/heads/main/gate_value_distribution.jpg)

## 5. Analysis of Gate Distribution
The final state of the model shows a distinct bimodal distribution:
* **Pruned Spike:** A large spike at zero represents successfully pruned connections.
* **Survivor Cluster:** A secondary cluster represents the critical weights essential for maintaining classification performance.

## 6. Conclusion
The implementation successfully demonstrates a "Builder Mindset" by creating a self-optimizing system capable of significant compression (81% reduction) while remaining functional on a standard image classification task. This approach is highly suitable for deployment in resource-constrained AI systems.

---
**Author:** AI Engineering Intern Candidate
**Date:** April 2026
