import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

"""
Key Ideas of RMSNorm:

1. Standard LayerNorm uses (x - mean) / std. RMSNorm argues that "centering" (subtracting the mean)
   is not important for LLM activation distributions, and actually adds computational overhead.
   Simply doing Scaling to stabilize values is enough. This makes computation faster on GPUs.

2. This structure ensures that the gradients in the residual stream are very clean and can flow
   directly to the bottom layers. This is the key reason why modern large models can stack
   dozens of layers without collapsing.
"""

# ==========================================
# 1. Manual RMSNorm Implementation (Core component of Qwen/Llama)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # RMSNorm typically only has a scaling parameter gamma (weight), no bias term beta
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Formula: x / sqrt(mean(x^2) + eps)
        # The mean is computed over the last dimension (dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # First normalize, then multiply by the learnable scaling parameter
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# ==========================================
# 2. Qwen-style Block (Pre-LN + RMSNorm)
# ==========================================
class QwenStyleBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Use RMSNorm instead of LayerNorm
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN layer (Qwen actually uses SwiGLU, but for simplicity we keep the original structure with GELU)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(), # Modern LLMs commonly use GELU or Swish, which are better than ReLU
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        
        # --- Key difference: Pre-LN structure ---
        
        # 1. Attention sub-layer
        # Formula: x = x + Attention(Norm(x))
        residual = x
        x_norm = self.ln1(x) # Norm first
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = residual + attn_output # Then residual connection

        # 2. FFN sub-layer
        # Formula: x = x + FFN(Norm(x))
        residual = x
        x_norm = self.ln2(x) # Norm first
        ffn_output = self.ffn(x_norm)
        x = residual + ffn_output # Then residual connection

        return x

# ==========================================
# (Below is your data loading and training code, almost unchanged, only replaced the Block)
# ==========================================

class DummyTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=32, embed_dim=128):
        self.num_samples = num_samples
        self.data = []
        for _ in range(num_samples):
            label = random.randint(0, 1)
            # Simple simulation: label 0 data skews negative, label 1 data skews positive
            base = -1.0 if label == 0 else 1.0
            feature = torch.randn(seq_length, embed_dim) + base
            self.data.append((feature, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_hidden_dim=256, num_classes=2, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()
        
        # [Modification] Using QwenStyleBlock here
        self.transformer_block = QwenStyleBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        
        # For stability in classification tasks, usually add a Norm layer before output (Final Norm)
        self.final_norm = RMSNorm(embed_dim)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.transformer_block(x)
        x = self.final_norm(x) # Qwen/Llama typically have a Final RMSNorm before the output layer
        
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits

# --- Run Test ---
if __name__ == "__main__":
    device = torch.device("cpu")
    # Slightly increase dimensions for demonstration purposes
    model = SimpleTransformerClassifier(embed_dim=128, num_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # AdamW is usually better than Adam
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(DummyTextDataset(), batch_size=32, shuffle=True)

    print("Starting training Qwen-style (RMSNorm + Pre-LN) model...")
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        correct = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / (len(train_loader.dataset))
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")