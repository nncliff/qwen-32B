from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch
import torch.nn as nn
import random

def generated_dummy_graph(label):
    num_nodes = random.randint(10, 20)
    x = torch.randn(num_nodes, 16) + label
    edge_index = torch.tensor([[i, (i+1)%num_nodes] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label]))
    return data

def build_dataset(num_samples=300):
    dataset = []
    for _ in range(num_samples):
        label = random.randint(0, 1)
        graph = generated_dummy_graph(label)
        dataset.append(graph)
    return dataset

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        output = x.div(keep_prob) * binary_tensor
        return output

class ResidualGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=2, drop_path_prob=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gat_conv = GATConv(in_dim, out_dim // heads, heads=heads, concat=True)
        self.linear_proj = nn.Linear(in_dim, out_dim)
        self.drop_path = DropPath(drop_path_prob)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        identity = self.linear_proj(x)
        x = self.norm(x)
        out = self.gat_conv(x, edge_index)
        out = self.drop_path(out)
        out = identity + out  # Residual connection
        out = self.activation(out)
        return out

class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.block1 = ResidualGATBlock(in_dim, hidden_dim, drop_path_prob=0.2)
        self.block2 = ResidualGATBlock(hidden_dim, hidden_dim, drop_path_prob=0.2)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out

device = torch.device("cpu")
model = GATClassifier(in_dim=16, hidden_dim=64, out_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

dataset = build_dataset()
loader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(10):
    total_loss = 0.0
    correct = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == batch.y).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")