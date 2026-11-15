import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input: each is (batch, context, attention_dim)
        k = self.key_gen(embedded)
        q = self.query_gen(embedded)
        v = self.value_gen(embedded)

        # 2. Scores: (batch, context, context), scaled by sqrt(attention_dim)
        scores = q @ k.transpose(1, 2) / (attention_dim_sqrt := k.shape[-1] ** 0.5)

        # 3. Causal mask: lower-triangular, future positions -> -inf
        context_length = embedded.shape[1]
        mask = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax over the last dim, so each row sums to 1
        scores = torch.softmax(scores, dim=2)

        # 5. Weighted sum of values, rounded to 4 decimals
        return torch.round(scores @ v, decimals=4)