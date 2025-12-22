import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

class LegalQADataset(Dataset):
    def __init__(self, num_samples=500):
        self.data = []

        for _ in range(num_samples):
            label = random.randint(0, 1) # Binary classification
            q_vec = torch.randn(64) + label * 0.5  # Simple pattern based on label
            a_vec = torch.randn(64) + label * 0.5  # Simple pattern based on label
            self.data.append((q_vec, a_vec, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a, label = self.data[idx]
        return q, a, torch.tensor(label)

class LoRAModule(nn.Module):
    def __init__(self, input_dim, out_dim, rank=4, alpha=16):
        super(LoRAModule, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank

        # Original weight matrix (frozen during training)
        self.weight = nn.Parameter(torch.randn(out_dim, input_dim), requires_grad=False)

        # Trainable low-rank adaptation matrices, shaped (rank, input_dim) and (out_dim, rank)
        self.lora_A = nn.Parameter(torch.randn(rank, input_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x):
        # x: (batch_size, input_dim)
        original_output = F.linear(x, self.weight) # (batch_size, out_dim)
        lora_output = F.linear(x, self.lora_A) # (batch_size, rank)
        lora_output = F.linear(lora_output, self.lora_B) * self.scaling # (batch_size, out_dim)
        return original_output + lora_output

class QWQBlockWithLoRA(nn.Module):
    def __init__(self, input_dim=64):
        super(QWQBlockWithLoRA, self).__init__()
        self.q_proj = LoRAModule(input_dim, input_dim)
        self.k_proj = LoRAModule(input_dim, input_dim)
        self.v_proj = LoRAModule(input_dim, input_dim)
        self.out_proj = LoRAModule(input_dim, input_dim)
        self.ffn = nn.Sequential(
            LoRAModule(input_dim, input_dim * 2),
            nn.ReLU(),
            LoRAModule(input_dim * 2, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, q, k, v):
        # Self-attention
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        attn_scores = torch.matmul(q_proj, k_proj.T) / (q_proj.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_proj)
        #attn_output = self.out_proj(attn_output)

        # Add & Norm
        x = self.norm1(q + attn_output)
        x = self.norm2(self.ffn(x) + x)

        return x

class LegalQAClassifierWithLoRA(nn.Module):
    def __init__(self):
        super(LegalQAClassifierWithLoRA, self).__init__()
        self.encoder = QWQBlockWithLoRA()
        self.classifier = nn.Linear(64, 2)  # Binary classification

    def forward(self, q, a):
        q_encoded = self.encoder(q, q, q)
        a_encoded = self.encoder(a, a, a)
        combined = q_encoded * a_encoded # Element-wise multiplication
        return self.classifier(combined)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = LegalQAClassifierWithLoRA().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss()

loader = DataLoader(LegalQADataset(), batch_size=32, shuffle=True)

model.train()
for epoch in range(3):
    total_loss = 0
    correct = 0
    for batch in loader:
        q_batch, a_batch, labels = batch
        q_batch, a_batch, labels = q_batch.to(device), a_batch.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(q_batch, a_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    acc = correct / len(loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, Accuracy: {acc:.4f}")