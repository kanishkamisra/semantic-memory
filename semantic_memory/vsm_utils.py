"""Utils specifically for Vector Space Models"""
import torch


def cosine(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sims = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sims


def jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    numerator = torch.minimum(a.unsqueeze(1), b.unsqueeze(0)).sum(2)
    denominator = torch.maximum(a.unsqueeze(1), b.unsqueeze(0)).sum(2)
    # numerator = torch.min(a, b).sum(1)
    # denominator = torch.max(a, b).sum(1)
    return numerator / denominator
