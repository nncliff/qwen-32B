import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class PatchExtractor(nn.Module):
    def __init__(self, patch_size, stride):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape

        # Explanation of tensor.unfold():
        # tensor.unfold(dimension, size, step) creates a sliding window view along a dimension.
        # Using step = patch_size means non-overlapping patches
        # If you used step < patch_size, patches would overlap
        # The stride parameter in __init__ is currently unused — you could use it as the step value for overlapping patches
        # 
        # Extract patches, resulting shape: (batch_size, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # Explanation of .contiguous().view():
        # After unfold(), the tensor may not be stored in contiguous memory (elements aren't laid out sequentially in memory). This happens because unfold creates a view with modified strides rather than copying data.
        #
        # .contiguous() creates a copy with elements in proper sequential memory order, which is required before calling .view().
        #
        # .view(batch_size, channels, -1, patch_size, patch_size)
        # Reshapes the tensor without changing data. The -1 means "infer this dimension automatically."
        #
        # Before: (batch, channels, num_patches_h, num_patches_w, patch_size, patch_size)
        # After: (batch, channels, num_patches, patch_size, patch_size)
        # Where num_patches = num_patches_h × num_patches_w
        # 
        # Alternative: Use .reshape() which handles non-contiguous tensors automatically:
        # patches = patches.reshape(batch_size, channels, -1, self.patch_size, self.patch_size)
        # 
        # Reshape to (batch_size, channels, num_patches, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)

        # Permute to (batch_size, num_patches, channels, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (batch_size, num_patches, channels, patch_size, patch_size)
        # Final reshape to (batch_size, num_patches, channels * patch_size * patch_size)
        patches = patches.contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)

        return patches  # (batch_size, num_patches, channels * patch_size * patch_size)

class PatchEmbedder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super(PatchEmbedder, self).__init__()
        self.fc1 = nn.Linear(in_dim, embed_dim * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        # x: (batch_size, num_patches, in_dim)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)  # (batch_size, num_patches, embed_dim)

class TextEmbedder(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=128):
        super(TextEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)  # (batch_size, seq_length, embed_dim)


# Example contrastive loss function using mean squared error as similarity metric
def contrastive_loss(vision_embeddings, text_embeddings):
    # Calculate mean squared error loss between vision and text embeddings
    return torch.mean((vision_embeddings - text_embeddings) ** 2)

class PatchMappingModel(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchMappingModel, self).__init__()
        self.patch_extractor = PatchExtractor(patch_size, patch_size)
        self.patch_embedder = PatchEmbedder(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        patches = self.patch_extractor(x)
        embeddings = self.patch_embedder(patches)
        return embeddings

def test():
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    patch_size = 16
    embed_dim = 128
    vocab_size = 50

    image_input = torch.randn(batch_size, channels, height, width)
    image_embedder = PatchMappingModel(patch_size, channels, embed_dim)
    image_embeddings = image_embedder(image_input)

    text_input = torch.rand(batch_size, vocab_size).long()
    text_embedder = TextEmbedder(vocab_size, embed_dim)
    text_embeddings = text_embedder(text_input)

    vision_mean = image_embeddings.mean(dim=1)  # (batch_size, embed_dim)
    text_mean = text_embeddings.mean(dim=1)     # (batch_size, embed_dim)
    loss = contrastive_loss(vision_mean, text_mean)

    print("Loss:", loss.item())
    print("Image Embeddings Shape:", image_embeddings.shape)
    print("Text Embeddings Shape:", text_embeddings.shape)

    # backpropagation example (only for demonstration; in practice, use an optimizer)
    optimizer = torch.optim.Adam(list(image_embedder.parameters()) + list(text_embedder.parameters()), lr=1e-3)
    num_steps = 5
    for step in range(num_steps):
        optimizer.zero_grad()
        image_embeddings = image_embedder(image_input)
        vision_mean = image_embeddings.mean(dim=1)
        text_embeddings = text_embedder(text_input)
        text_mean = text_embeddings.mean(dim=1)
        loss = contrastive_loss(vision_mean, text_mean)
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    test()