import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class SparseActivationLayer(nn.Module):
    def __init__(self, threshold: float = 0.1):
        super(SparseActivationLayer, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        mask = (x.abs() > self.threshold).float()

        sparsity = 100 * (1.0 - mask.mean().item())
        print(f"Sparsity: {sparsity:.2f}%")

        return x * mask

class SimpleFeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, threshold: float = 0.1):
        super(SimpleFeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sparse_activation1 = SparseActivationLayer(threshold)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sparse_activation2 = SparseActivationLayer(threshold)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.sparse_activation1(x)
        x = self.fc2(x)
        x = self.sparse_activation2(x)
        return x

def run_inference():
    input_dim = 128
    hidden_dim = 256
    output_dim = 10
    batch_size = 32

    model = SimpleFeedForwardNN(input_dim, hidden_dim, output_dim, threshold=0.1)
    model.eval()
    inputs = torch.randn(batch_size, input_dim)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    end_time = time.time()

    print(f"Inference Time: {end_time - start_time:.6f} seconds")
    print(f"Output Shape: {outputs.shape}")

if __name__ == "__main__": 
    run_inference()