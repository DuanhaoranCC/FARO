"""
Data loading for supported bot-detection datasets.

Supported datasets
------------------
- twibot-20   : 70 / 20 / 10 % random split
- Cresci-15   : pre-split index files
- MGTAB       : 70 / 20 / 10 % shuffled split (seed=42)
"""

import os
import argparse

import numpy as np
import torch
from torch_geometric.data import Data


def load_data(args: argparse.Namespace) -> Data:
    """
    Load graph data for the requested dataset and move it to the target device.

    Side-effect: sets ``args.num_relations`` from the edge_type tensor.

    Returns
    -------
    torch_geometric.data.Data
        Graph object with attributes:
        x, edge_index, edge_type, y, num_total,
        train_idx, val_idx, test_idx
    """
    path = os.path.join(args.base_dir, args.dataset)

    edge_index = torch.load(os.path.join(path, "edge_index.pt"))
    edge_type  = torch.load(os.path.join(path, "edge_type.pt"))
    args.num_relations = int(edge_type.max().item()) + 1

    if args.dataset == "twibot-20":
        data = _load_twibot20(path, edge_index, edge_type)
    elif args.dataset == "Cresci-15":
        data = _load_cresci15(path, edge_index, edge_type)
    else:
        data = _load_mgtab(path, edge_index, edge_type)

    data.num_total = data.x.size(0)
    return data.to(args.device)


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _load_twibot20(path: str, edge_index: torch.Tensor,
                   edge_type: torch.Tensor) -> Data:
    x = torch.cat([
        torch.load(os.path.join(path, "num_properties_tensor.pt")),
        torch.load(os.path.join(path, "tweets_tensor.pt")),
        torch.load(os.path.join(path, "cat_properties_tensor.pt")),
        torch.load(os.path.join(path, "des_tensor.pt")),
    ], dim=1)

    y      = torch.load(os.path.join(path, "label.pt")).long()
    full_y = torch.full((x.size(0),), -1, dtype=torch.long)
    full_y[: len(y)] = y

    idx    = np.arange(int(y.size(0)))
    tr, va = int(0.7 * len(idx)), int(0.9 * len(idx))

    data           = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=full_y)
    data.train_idx = idx[:tr].tolist()
    data.val_idx   = idx[tr:va].tolist()
    data.test_idx  = idx[va:].tolist()
    return data


def _load_cresci15(path: str, edge_index: torch.Tensor,
                   edge_type: torch.Tensor) -> Data:
    x = torch.cat([
        torch.load(os.path.join(path, "cat_properties_tensor.pt")),
        torch.load(os.path.join(path, "num_properties_tensor.pt")),
        torch.load(os.path.join(path, "des_tensor.pt")),
        torch.load(os.path.join(path, "tweets_tensor.pt")),
    ], dim=1)

    y      = torch.load(os.path.join(path, "label.pt")).long()
    full_y = torch.full((x.size(0),), -1, dtype=torch.long)
    full_y[: len(y)] = y

    to_list = lambda t: t.tolist() if isinstance(t, (torch.Tensor, np.ndarray)) else list(t)

    data           = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=full_y)
    data.train_idx = to_list(torch.load(os.path.join(path, "train_idx.pt")))
    data.val_idx   = to_list(torch.load(os.path.join(path, "val_idx.pt")))
    data.test_idx  = to_list(torch.load(os.path.join(path, "test_idx.pt")))
    return data


def _load_mgtab(path: str, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> Data:
    x      = torch.load(os.path.join(path, "features.pt"))
    y      = torch.load(os.path.join(path, "label.pt")).long()
    full_y = torch.full((x.size(0),), -1, dtype=torch.long)
    full_y[: len(y)] = y

    idx = np.arange(int(y.size(0)))
    np.random.RandomState(42).shuffle(idx)
    tr, va = int(0.7 * len(idx)), int(0.9 * len(idx))

    data           = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=full_y)
    data.train_idx = idx[:tr].tolist()
    data.val_idx   = idx[tr:va].tolist()
    data.test_idx  = idx[va:].tolist()
    return data
