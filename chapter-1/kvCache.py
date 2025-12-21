import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from typing import List, Tuple

class SimpleDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(SimpleDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, kv_cache = None, use_cache: bool = True) -> torch.Tensor:
        if kv_cache is not None:
            # Use cached key and value tensors for efficient decoding
            k, v = kv_cache
            x_attn, _ = self.self_attn(self.norm1(x), k, v, need_weights=False)
        else:
            # Compute self-attention normally
            x_attn, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)

        x = x + x_attn
        x = self.ffn(self.norm2(x))
        
        if use_cache:
            # Update kv_cache with new key and value tensors
            return x, (x.clone().detach(), x.clone().detach())

        return x, None

class KVCacheManager:
    def __init__(self, max_cache_size: int = 64):
        self.cache : List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.token_labels : List[str] = [] # To store labels for each token in the cache
        self.max_cache_size = max_cache_size

    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cache:
            return None
        
        k = torch.cat([item[0] for item in self.cache], dim=1)  # Concatenate along sequence length
        v = torch.cat([item[1] for item in self.cache], dim=1)  # Concatenate along sequence length
        return (k, v) # shape of k or v: (batch_size, total_sequence_length, embed_dim)

    def update_cache(self, new_kv: Tuple[torch.Tensor, torch.Tensor], tokens : List[str], current_round : int):
        self.cache.append(new_kv)
        self.token_labels += [f"Round{current_round}"] * new_kv[0].size(1)  # Assuming new_kv[0] shape is (batch_size, seq_len, embed_dim)

        if len(self.token_labels) > self.max_cache_size:
            keep_mask = torch.tensor(
                [label == f"Round{current_round}" for label in self.token_labels], dtype=torch.bool
            )
            self.cache = [(k[:, keep_mask, :], v[:, keep_mask, :]) for k, v in self.cache[-1]] # Keep only current round tokens
            self.token_labels = [label for label in self.token_labels if label == f"Round{current_round}"] # Update labels accordingly

def generate_tokens(prompt : str, vocab : List[str], num_tokens: int = 5) -> List[str]:
    return [random.choice(vocab) for _ in range(num_tokens)]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
decoder = SimpleDecoderBlock(embed_dim=64, num_heads=4).to(device)
kv_cache_manager = KVCacheManager(max_cache_size=30)
vocab = list(string.ascii_lowercase)  # Example vocabulary

for round_id in range(1, 6):
    prompt = f"[Round {round_id}] User Input: write an function"
    tokens = generate_tokens(prompt, vocab)
    print(f"Round {round_id} generated tokens: {' '.join(tokens)}")

    # Simulate token embeddings
    token_tensors = torch.stack([torch.randn(64) for _ in tokens]).unsqueeze(0).to(device)  # shape: (1, seq_len, embed_dim)

    # Retrieve kv_cache and decode
    kv_cache = kv_cache_manager.get_cache()
    output, new_kv = decoder(token_tensors, kv_cache=kv_cache, use_cache=True)

    if new_kv is not None:
        kv_cache_manager.update_cache(new_kv, tokens, current_round=round_id)

    summary = ''.join(random.choices(string.ascii_lowercase, k=10))
    print(f"Round {round_id} summary: {summary}")

print("\n=== Final KV Cache State ===")
print(f"Current token number in cache: {len(kv_cache_manager.token_labels)}")
print(f"Round labels in cache: {kv_cache_manager.token_labels}")