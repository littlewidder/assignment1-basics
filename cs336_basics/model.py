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


class SwiGLU2(torch.nn.Module):
    w1: Linear
    w2:Linear
    w3:Linear

    def __init__(self, d_m, d_ff, dtype=None, device=None):
        super().__init__()
        self.w1 = Linear(d_m, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_m, d_ff, dtype=dtype, device=device)
        self.w3 = Linear(d_ff, d_m,  dtype=dtype, device=device)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.w1(x)
        return self.w3((y*torch.sigmoid(y))*self.w2(x))

class RoPE(torch.nn.Module):
    def __init__(self, d_m, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.d_m = d_m
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.register_buffers(max_seq_len, d_m, theta, device, dtype)

    def register_buffers(self, max_seq_len, d_m, theta, device, dtype):
        pos = torch.arange(max_seq_len, device=device, dtype=dtype)
        k = torch.arange(d_m//2, device=device, dtype=dtype)
        self.thetas = 1/(theta**((2.0*k)/d_m)) # thetas for one vector, one position, all pairs of dimensions
        self.register_buffer("cos_t", torch.cos(pos.unsqueeze(-1)*self.thetas), persistent=False)
        self.register_buffer("sin_t", torch.sin(pos.unsqueeze(-1)*self.thetas), persistent = False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_m, "input dim does not match"
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # y1 = x1*torch.cos(positions.unsqueeze(-1)*self.thetas) - x2*torch.sin(positions.unsqueeze(-1)*self.thetas)
        # y2 = x1*torch.sin(positions.unsqueeze(-1)*self.thetas) +x2*torch.cos(positions.unsqueeze(-1)*self.thetas)
        y1 = x1*self.cos_t[positions] - x2*self.sin_t[positions]
        y2 = x1*self.sin_t[positions] + x2*self.cos_t[positions]
        return torch.stack([y1, y2], dim=-1).flatten(-2)
