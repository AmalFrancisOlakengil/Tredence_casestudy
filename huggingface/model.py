import torch
import torch.nn as nn
import numpy as np

class PruningNet(nn.Module):
    def __init__(self):
        super(PruningNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_gates(self):
        # Helper to collect all gate values for loss calculation and reporting
        all_gates = [
            torch.sigmoid(self.fc1.gate_scores),
            torch.sigmoid(self.fc2.gate_scores),
            torch.sigmoid(self.fc3.gate_scores)
        ]
        return all_gates

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.0) # Start with gates open

    def forward(self, x):
        # Sigmoid to constrain gates between 0 and 1
        gates = torch.sigmoid(5.0 * self.gate_scores)
        pruned_weights = self.weight * gates
        return nn.functional.linear(x, pruned_weights, self.bias)