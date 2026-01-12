import torch
from torch import nn
import torch.nn.functional as F
import time
import numpy as np

class StandardAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(StandardAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.scale = embed_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        Q = self.q_proj(x) # [batch_size, seq_length, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        return output

class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(FlashAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.scale = embed_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        Q = self.q_proj(x) # [batch_size, seq_length, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.embed_dim).transpose(1, 2)

        # Flash Attention implementation
        attn_output = self.flash_attention(Q, K, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(attn_output)
        return output

    def flash_attention(self, Q, K, V):
        # This is a simplified version of Flash Attention
        # In practice, Flash Attention uses more complex memory-efficient algorithms
        # Here we just simulate it with standard attention for demonstration purposes
        #
        # Key Techniques in Real Flash Attention:
        # 1. Tiling/Blocking: Processes Q, K, V in small blocks that fit in fast SRAM 
        #    (on-chip memory), avoiding slow HBM (GPU main memory) reads/writes
        #
        # 2. Online Softmax: Computes softmax incrementally without storing the full 
        #    N×N matrix using the "online softmax trick" (tracking running max and sum)
        #
        # 3. Kernel Fusion: Fuses matmul + softmax + matmul into a single GPU kernel, 
        #    reducing memory bandwidth bottleneck
        #
        # 4. Recomputation: During backward pass, recomputes attention instead of 
        #    storing it (trading compute for memory)
        #
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)
        return attn_output

def run_attention_comparison():
    torch.manual_seed(42)
    batch_size = 4
    seq_length = 64
    embed_dim = 128
    num_heads = 8

    x = torch.randn(batch_size, seq_length, embed_dim)

    standard_attn = StandardAttention(embed_dim, num_heads)
    flash_attn = FlashAttention(embed_dim, num_heads)

    # Warm-up
    for _ in range(10):
        _ = standard_attn(x)
        _ = flash_attn(x)

    # Measure Standard Attention
    start_time = time.time()
    for _ in range(100):
        _ = standard_attn(x)
    standard_time = time.time() - start_time

    # Measure Flash Attention
    start_time = time.time()
    for _ in range(100):
        _ = flash_attn(x)
    flash_time = time.time() - start_time

    mse = torch.mean((standard_attn(x) - flash_attn(x)) ** 2).item()
    print(f"Mean Squared Error between Standard and Flash Attention: {mse}")

    print(f"Standard Attention Time: {standard_time:.4f} seconds")
    print(f"Flash Attention Time: {flash_time:.4f} seconds")

if __name__ == "__main__":
    run_attention_comparison()
