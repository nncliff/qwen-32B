import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.k = max(1, num_experts // 4)  # Select top-k experts, if k=1, it becomes Top-1 MoE

        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.ReLU(),
            nn.Linear(expert_dim, input_dim),
        ) for _ in range(num_experts)])
        
        self.classifier = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # the shape of x: 
        logits = self.classifier(x)
        topk_scores = F.softmax(logits, dim=1)

        topk_scores, topk_indices = torch.topk(topk_scores, k=self.k, dim=1)
        output = torch.zeros_like(x)

        for i in range(self.k):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_scores[:, i].unsqueeze(1)

            expert_mask = torch.zeros(x.shape[0], self.num_experts, dtype=torch.bool, device=x.device)
            expert_mask.scatter_(1, expert_idx.unsqueeze(1), True)

            for j, expert in enumerate(self.experts):
                mask = expert_mask[:, j]

                if mask.any():
                    expert_input = x[mask]
                    expert_out = expert(expert_input)
                    output[mask] += expert_out * expert_weight[mask]

        return output

class MoEClassifier(nn.Module):
    def __init__(self, input_dim=512, moe_hidden=1024, num_experts=8, num_classes=10):
        super(MoEClassifier, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove the final classification layer
        self.moe = MixtureOfExperts(input_dim, moe_hidden, num_experts)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.moe(x)
        return self.classifier(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = MoEClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(5):
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += predicted.eq(labels).sum().item()

    print(f'Epoch [{epoch+1}/5], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/len(train_loader.dataset):.2f}%')