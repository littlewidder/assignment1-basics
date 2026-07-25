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

class RMSNorm(torch.nn.Module):
    def __init__(self, d_m, eps: float = 1e-5, dtype=None, device=None):
        super().__init__()
        self.d_m = d_m
        self.eps = eps
        self.dtype=dtype
        self.device=device
        self.g = torch.nn.Parameter(torch.ones((d_m), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        assert x.shape[-1] == self.d_m, "dim does not match"
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x/rms*self.g).to(in_dtype)

class SwiGLU(torch.nn.Module):
    d_m: int
    d_ff:int
    weight: torch.Tensor

    def __init__(self, d_m, d_ff, dtype=None, device=None):
        super().__init__()
        self.d_m=d_m
        self.d_ff = d_ff
        self.w1 = torch.nn.Parameter( torch.empty((self.d_ff, d_m), dtype=dtype, device=device))
        self.w2=torch.nn.Parameter(torch.empty((self.d_ff, d_m), dtype=dtype, device=device))
        self.w3 = torch.nn.Parameter(torch.empty(( d_m, self.d_ff), dtype=dtype, device=device))
        self.reset_parameters()

    def reset_parameters(self):
        std:float = (2/(self.d_m+self.d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1, std = std, mean=0.0, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w2, std = std, mean=0.0, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.w3, std = std, mean=0.0, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_m, "input dim does not match"
        y1 = x@self.w1.T
        # s = y1/(1+torch.exp(-y1))
        s = y1*torch.sigmoid(y1)
        return (s*(x@self.w2.T))@self.w3.T
