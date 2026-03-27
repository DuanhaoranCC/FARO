"""
Utility functions: reproducibility helpers and tensor operations.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def triu_flatten(M: torch.Tensor, K: int) -> torch.Tensor:
    """
    Extract the upper-triangular (offset=1) elements of a square matrix.

    Args:
        M: Tensor of shape (K, K) or (B, K, K).
        K: Matrix side length.

    Returns:
        Tensor of shape (K*(K-1)//2,) or (B, K*(K-1)//2).
    """
    idx = torch.triu_indices(K, K, offset=1, device=M.device)
    if M.dim() == 3:
        return M[:, idx[0], idx[1]]
    elif M.dim() == 2:
        return M[idx[0], idx[1]]
    else:
        raise ValueError(f"Expected M.dim() in {{2, 3}}, got {M.dim()}")
