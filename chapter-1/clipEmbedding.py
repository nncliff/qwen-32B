import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim = 128
hidden_dim = 256
vocab_size = 10000
sequence_length = 30
batch_size = 64
num_epochs = 5
clip_grad_norm = 1.0

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, sequence_length)
        embedded = self.embedding(x)        # (batch_size, sequence_length, embedding_dim)
        output, _ = self.rnn(embedded)      # (batch_size, sequence_length, hidden_dim)
        logits = self.fc(output[:, -1, :])  # Use the last time step
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def generate_dummy_data(batch_size, sequence_length, vocab_size):
    input = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
    target = torch.randint(0, vocab_size, (batch_size,), dtype=torch.long)
    return input, target

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _ in range(100):  # Assume 100 batches per epoch
        inputs, targets = generate_dummy_data(batch_size, sequence_length, vocab_size)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval() # Evaluation
sample_input, _ = generate_dummy_data(1, sequence_length, vocab_size)
sample_input = sample_input.to(device)
with torch.no_grad():
    sample_output = model(sample_input)
    predicted_token = torch.argmax(sample_output, dim=1).item()
    print("Sample input:", sample_input.cpu().numpy()) # Print input sequence
    print("Predicted token:", predicted_token)