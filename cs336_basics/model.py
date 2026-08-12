from typing import Optional
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

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    y= x-x.max(dim=dim, keepdim=True)[0]
    e = torch.exp(y)
    return e/e.sum(dim=dim, keepdim=True)

class Attention1(torch.nn.Module):
    def __init__(self, dtype=None, device=None):
        super().__init__()
        self.dtype = dtype
        self.device = device

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor]):
        d_k = k.shape[-1]
        scores = (q@k.mT)/(d_k**0.5)
        if mask is not None:
            # add_mask = torch.where(mask, 0.0, float("-inf"))
            # scores = scores + add_mask
            scores.masked_fill_(~mask, float("-inf"))
        return softmax(scores, dim=-1)@v

class MHA(torch.nn.Module):
    def __init__(self, d_m: int, n_head: int, dtype = None, device=None):
        super().__init__()
        self.d_m = d_m
        self.n_heads = n_head
        self.Wq = torch.nn.Parameter(torch.empty((d_m, d_m), device=device, dtype=dtype))
        self.Wk = torch.nn.Parameter(torch.empty((d_m, d_m), device=device, dtype=dtype))
        self.Wv = torch.nn.Parameter(torch.empty((d_m, d_m), device=device, dtype=dtype))
        self.Wo = torch.nn.Parameter(torch.empty((d_m, d_m), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        d_head = self.d_m//self.n_heads
        q = (x@self.Wq.mT).unflatten(-1, (self.n_heads, d_head)).transpose(-3, -2)
        k = (x@self.Wk.mT).unflatten(-1, (self.n_heads, d_head)).transpose(-3, -2)
        v = (x@self.Wv.mT).unflatten(-1, (self.n_heads, d_head)).transpose(-3, -2)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        a = Attention1()(k, v, q, mask)                       # (..., n_head, seq, d_head)
        a = a.transpose(-3, -2).flatten(-2)                   # (..., seq, d_m)
        return a@self.Wo.mT

class MHA_full(torch.nn.Module):
    def __init__(self, d_m, n_head, max_seq_len, theta, dtype = None, device=None):
        super().__init__()
        self.d_m = d_m
        self.d_head = d_m//n_head
        self.n_head = n_head
        self.max_seq_leg = max_seq_len
        self.Lq = Linear(d_m, d_m, dtype, device)
        self.Lk = Linear(d_m, d_m, dtype, device)
        self.Lv = Linear(d_m, d_m, dtype, device)
        self.Lo = Linear(d_m, d_m, dtype, device)
        self.rope = RoPE(self.d_head, theta, max_seq_len, device, dtype)
        self.device = device

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        seq = x.shape[-2]
        q = self.Lq(x).unflatten(-1, (self.n_head, -1)).transpose(-3, -2)
        v = self.Lv(x).unflatten(-1, (self.n_head, -1)).transpose(-3, -2)
        k = self.Lk(x).unflatten(-1, (self.n_head, -1)).transpose(-3, -2)
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=self.device))
        q1 = self.rope(q, positions)
        k1 = self.rope(k, positions)
        a = Attention1()(k1, v, q1, mask)
        return self.Lo(a.transpose(-3, -2).flatten(-2))

class MyTransformer(torch.nn.Module):
    def __init__(self, d_m, n_head, max_seq_len, theta, d_ff):
        super().__init__()
        self.mha = MHA_full(d_m, n_head, max_seq_len, theta)
        self.norm1 = RMSNorm(d_m)
        self.ffn = SwiGLU2(d_m, d_ff)
        self.norm2 = RMSNorm(d_m)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[-2], device = x.device)
        y = self.mha(self.norm1(x), positions) + x
        return y+ self.ffn(self.norm2(y))

class FullTransformer(torch.nn.Module):
    def __init__(self, vocab_size, layers, context_len, d_m, num_heads, theta, d_ff,  dtype=None, device=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([MyTransformer(d_m, num_heads, context_len, theta, d_ff) for i in range(layers)])
        # for i in range(layers):
        #     self.layers[i] = MyTransformer(d_m, num_heads, context_len, theta, d_ff)
        self.embedding = Embedding(vocab_size, d_m)
        self.norm = RMSNorm(d_m)
        self.lm_head = Linear(d_m, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        activation = self.embedding(x)
        for layer in self.layers:
            activation = layer(activation)
        activation = self.norm(activation)
        return self.lm_head(activation)
