from numpy import dtypes
from math import sqrt

import torch
from einops import einsum, rearrange


class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features, out_features, dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype, device=device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = sqrt(2.0 / (self.out_features + self.in_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.in_features
        ), f"Expected input shape to be ({x.shape[:-1]}, {self.in_features}), but got {x.shape}"
        # return einsum(x, self.weight, "b i, o i -> b o")
        return x @ self.weight.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dtype=None, device=None):
        super().__init__()
        self.num_embeddings =  num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.weight[x]
