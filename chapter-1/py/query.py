import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import string
from typing import List, Dict
import hashlib

def generate_fake_document(num_paragraphs: int = 5, tokens_per_paragraph: int = 10, dim: int = 128) -> str:
    """Generates a fake document with random sentences."""
    document = []
    for _ in range(num_paragraphs):
        base = torch.randn(dim) # shape: (dim,)

        # broadcast to create a paragraph
        paragraph = base + 0.01 * torch.randn(tokens_per_paragraph, dim) # shape: (tokens_per_paragraph, dim)
        
        document.append(paragraph)
    
    return document # List of tensors representing paragraphs of shape (tokens_per_paragraph, dim)

class HashIndex:
    def __init__(self, key_vectors: List[torch.Tensor], num_buckets: int = 16):
        self.buckets: Dict[int, List[Tuple[int, torch.Tensor]]] = {i: [] for i in range(num_buckets)}
        self.num_buckets = num_buckets
        
        for idx, paragraph in enumerate(key_vectors):
            # Hash is based on the mean vector of the paragraph
            key = paragraph.mean(dim=0)  # the shape of key: (dim,)

            bucket_id = self._hash_vector(key) # shape: ()
            self.buckets[bucket_id].append((idx, key))

    def _hash_vector(self, vector: torch.Tensor) -> int:
        """Hashes a vector to a bucket ID."""
        h = hashlib.md5(vector[:3].detach().numpy().tobytes()).hexdigest()  # Use first 3 elements for hashing
        return int(h, 16) % self.num_buckets

    def search(self, query_vector: torch.Tensor, top_k: int = 3) -> List[int]:
        """Searches for the top_k closest key vectors to the query_vector."""
        bucket_id = self._hash_vector(query_vector)
        candidates = self.buckets[bucket_id]
        scores = []
        
        for idx, vector in candidates:
            # cosine similarity is expecting 2D tensors (batch_size, dim)
            # .item() to get scalar value from tensor
            dist = F.cosine_similarity(query_vector.unsqueeze(0), vector.unsqueeze(0)).item()
            scores.append((idx, dist))
        
        # Get top_k closest
        scores.sort(key=lambda x: x[1])  # Sort by similarity
        return [idx for idx, _ in scores[:top_k]]

class LongDocQA(nn.Module):
    def __init__(self, dim: int = 128):
        super(LongDocQA, self).__init__()

        self.query_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )

        self.answer_decoder = nn.GRU(input_size=dim, hidden_size=dim, batch_first=True)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query_vector: torch.Tensor, context_vectors: List[torch.Tensor]) -> torch.Tensor:
        input_seq = torch.stack(context_vectors, dim=0).unsqueeze(0)  # shape: (1, num_contexts, dim)
        _, hidden = self.answer_decoder(input_seq)  # shape: (1, 1, dim)
        response = self.output_proj(hidden.squeeze(0))  # shape: (1, dim)
        return response

# Construct fake document and index
torch.manual_seed(42)
dim = 64
doc = generate_fake_document(num_paragraphs=30, tokens_per_paragraph=8, dim=dim)
#index = HashIndex(doc_summary_vector, num_buckets=8)
index = HashIndex(doc, num_buckets=8)

# Simulate multiple turns of user queries
model = LongDocQA(dim=dim)
model.eval()

for i in range(1, 6):
    # Simulate a user query (main topic and paragraph are relevant)
    base_para = random.choice(doc)
    query_vector = base_para.mean(dim=0) + 0.02 * torch.randn(dim)  # shape: (dim,)
    query_encoded = model.query_encoder(query_vector)  # shape: (dim,)

    # Search for relevant paragraphs
    top_indices = index.search(query_encoded, top_k=3)
    selected_paragraphs = [doc[idx].mean(dim=0) for idx in top_indices]  # List of tensors of shape (dim,)

    # Generate answer
    with torch.no_grad():
        answer_vector = model(query_encoded, selected_paragraphs)  # shape: (dim,)

    keywords = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    print(f"Turn {i}: Top paragraphs indices: {top_indices}")
    print(f"Generate answer summary (norm): {answer_vector.norm().item():.4f} with Answer: {keywords}\n")
