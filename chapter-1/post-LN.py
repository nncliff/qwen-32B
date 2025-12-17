import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

class DummyTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=32, embed_dim=128):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.data = []

        for _ in range(num_samples):
            label = random.randint(0, 1)

            base = -1.0 if label == 0 else 1.0
            feature = torch.randn(seq_length, embed_dim) + base
            
            self.data.append((feature, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class PostLayerNormTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(PostLayerNormTransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        
        # Post-LayerNorm: Attention -> Residual -> LayerNorm
        # 1. Calculate Attention on x (not normalized x)
        attn_output, _ = self.attention(x, x, x)
        
        # 2. Add Residual Connection
        x = x + attn_output
        
        # 3. Apply LayerNorm AFTER the addition
        x = self.ln1(x)

        # Post-LayerNorm: FFN -> Residual -> LayerNorm
        # 1. Calculate FFN on x
        ffn_output = self.ffn(x)
        
        # 2. Add Residual Connection
        x = x + ffn_output
        
        # 3. Apply LayerNorm AFTER the addition
        x = self.ln2(x)

        return x

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, ff_hidden_dim=256, num_classes=2, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__()
        # Use the new Post-LN block
        self.transformer_block = PostLayerNormTransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        x = self.transformer_block(x)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_length)
        x = self.pool(x).squeeze(-1)  # Global average pooling (batch_size, embed_dim)
        logits = self.classifier(x)
        return logits

device = torch.device("cpu")
model = SimpleTransformerClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_loader = DataLoader(DummyTextDataset(), batch_size=32, shuffle=True)

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
