"""
Layer Cache Simulation Demo

Purpose:
    This script demonstrates a simplified layer-level cache management system
    for transformer inference. It shows how intermediate results can be cached
    and dynamically reused across forward passes.

What this code demonstrates:
    1. Layer Cache Management: Caching layer outputs for potential reuse
    2. Dynamic Cache Decision: Randomly deciding whether to use cached results
    3. Memory Optimization: Using detach() to prevent gradient tracking

Important Notes:
    - This is NOT a correct KV Cache implementation for autoregressive generation
    - Real KV Cache stores K/V projections and appends new tokens, not full layer outputs
    - The "DeepSpeed" name is misleading - this doesn't demonstrate actual DeepSpeed
      features like tensor parallelism, pipeline parallelism, or ZeRO optimization
    - This is primarily for educational purposes to illustrate cache management concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, cache=None):
        if cache is not None:
            attn_output, _ = self.attention(x, cache, cache)
        else:
            attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class DeepSpeedSimulator:
    def __init__(self, model, num_layers, embed_dim, num_heads):
        self.num_layers = num_layers

        # Create transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])

        # Initialize layer caches
        self.layer_caches = [None for _ in range(num_layers)]

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            use_cache = random.choice([True, False])
            if use_cache and self.layer_caches[i] is not None:
                x = layer(x, cache=self.layer_caches[i])
                print(f"Layer {i}: Using cache")
            else:
                x = layer(x)
                self.layer_caches[i] = x.detach() # Update cache, no gradient tracking
                print(f"Layer {i}: No cache used")
        return x

class DummyModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, seq_length):
        super(DummyModel, self).__init__()
        self.embed = nn.Linear(100, embed_dim)
        self.simulator = DeepSpeedSimulator(self, num_layers, embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, 10)
        self.seq_length = seq_length

    def forward(self, x):
        # Assume x is of shape (batch_size, seq_length, feature_dim)
        x = self.embed(x)
        x = self.simulator.forward(x)

        # Simple pooling and output projection
        x = torch.mean(x, dim=1)
        x = self.out_proj(x)
        return x

def test_deep_speed_simulator():
    batch_size = 4
    seq_length = 50
    embed_dim = 128
    num_heads = 8
    num_layers = 6

    # Random input
    x = torch.randn(batch_size, seq_length, 100)

    # Instantiate and run the model
    model = DummyModel(embed_dim, num_heads, num_layers, seq_length)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        for step in range(3):
            print(f"=== Forward pass {step + 1} ===")
            output = model(x)
            print()
    end_time = time.time()
    print(f"Model output: {output}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print("Output shape:", output.shape)
    return output

if __name__ == "__main__":
    test_deep_speed_simulator()