import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import random

class DummyAudioDataset(Dataset):
    def __init__(self, num_samples=300, sample_length=16000, num_classes=10):
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.num_classes = num_classes
        self.data = []

        for _ in range(num_samples):
            label = random.randint(0, num_classes - 1)
            waveform = torch.randn(sample_length) + label * 0.1  # Simple pattern based on label
            self.data.append((waveform, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class Top2Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Top2Router, self).__init__()
        self.num_experts = num_experts
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: (batch_size, input_dim) -> logits: (batch_size, num_experts)
        logits = self.fc(x)
        probabilities = F.softmax(logits, dim=-1)
        top2_vals, top2_indices = torch.topk(probabilities, 2, dim=-1)
        return top2_indices, top2_vals

class Top2MoE(nn.Module):
    def __init__(self, input_dim, expert_hidden_dim, num_experts=6):
        super(Top2MoE, self).__init__()
        self.num_experts = num_experts
        self.router = Top2Router(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch_size, input_dim)
        top2_indices, top2_vals = self.router(x)
        output = torch.zeros_like(x)

        for i in range(2):  # For top-2 experts
            expert_indices = top2_indices[:, i]
            expert_weights = top2_vals[:, i].unsqueeze(-1)

            for j in range(self.num_experts):
                mask = (expert_indices == j)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[j](expert_input)
                    output[mask] += expert_output * expert_weights[mask]

        return output

class MoEAudioClassifier(nn.Module):
    def __init__(self, in_channels=1, conv_dim=64, moe_dim=128, num_classes=10):
        super(MoEAudioClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(conv_dim, moe_dim)
        self.moe = Top2MoE(moe_dim, moe_dim*2, num_experts=6)
        self.classifier = nn.Linear(moe_dim, num_classes)

    def forward(self, x):
        # x: (batch_size, time_steps) -> (batch_size, 1, time_steps)
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.conv(x) # (batch_size, conv_dim, reduced_time_steps)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.proj(x)
        x = self.moe(x)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = DummyAudioDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MoEAudioClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    total_loss = 0
    correct = 0
    for batch in dataloader:
        waveforms, labels = batch
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)  # Add channel dimension
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/len(dataset):.4f}")