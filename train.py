"""
Training and evaluation routines.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data

from models import FCN_SC_v5
from utils import set_seed


# ---------------------------------------------------------------------------
# Prototype initialisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def initialize_prototypes(model: FCN_SC_v5, data: Data, device: torch.device) -> None:
    """
    Initialise M_human and M_bot from the mean coupling matrices of
    training nodes belonging to each class.
    """
    model.eval()
    train_idx_t = torch.tensor(data.train_idx, device=device)
    y_train     = data.y[train_idx_t]
    human_idx   = train_idx_t[y_train == 0]
    bot_idx     = train_idx_t[y_train == 1]

    group_reprs = model.group_encoder(
        data.x, data.edge_index, data.edge_type, data.num_total
    )
    C = model.coupling_module.compute_coupling_matrix(group_reprs)

    if human_idx.numel() > 0:
        model.coupling_module.M_human.data.copy_(C[human_idx].mean(0))
    if bot_idx.numel() > 0:
        model.coupling_module.M_bot.data.copy_(C[bot_idx].mean(0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_split(model: FCN_SC_v5, data: Data,
               idx: List[int]) -> Tuple[float, float, float, float]:
    """
    Evaluate the model on a given index split.

    Returns
    -------
    accuracy, f1, recall, precision : floats
    """
    model.eval()
    idx_t      = torch.tensor(idx, dtype=torch.long, device=data.x.device)
    logits, *_ = model(data)
    pred       = logits[idx_t].argmax(dim=1).cpu().numpy()
    true       = data.y[idx_t].cpu().numpy()
    return (
        accuracy_score(true, pred),
        f1_score(true,        pred, average="binary", pos_label=1),
        recall_score(true,    pred, average="binary", pos_label=1),
        precision_score(true, pred, average="binary", pos_label=1),
    )


# ---------------------------------------------------------------------------
# Single-seed training loop
# ---------------------------------------------------------------------------

def run_one_seed(
    seed: int,
    data: Data,
    args: argparse.Namespace,
) -> Tuple[float, float, float, float, float, float]:
    """
    Train and evaluate the model for one random seed.

    Returns
    -------
    best_val_f1, best_val_acc, te_acc, te_f1, te_rec, te_pre
    """
    set_seed(seed)

    model = FCN_SC_v5(args).to(args.device)
    initialize_prototypes(model, data, args.device)

    optimizer        = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion        = nn.CrossEntropyLoss()
    val_labels       = data.y[data.val_idx]
    best_val_f1      = 0.0
    best_val_acc     = 0.0
    best_state       = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        logits, lph, lpb, lsh, lsb, l_sc = model(
            data, train_idx=data.train_idx, labels=data.y
        )
        loss_cls = criterion(logits[data.train_idx], data.y[data.train_idx])
        loss     = (
            loss_cls
            + args.lambda_proto  * (lph + lpb)
            + args.lambda_sep    * (lsh + lsb)
            + args.lambda_supcon * l_sc
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits, *_ = model(data)
            v_pred       = v_logits[data.val_idx].argmax(dim=1)
            v_f1         = f1_score(val_labels.cpu(), v_pred.cpu())
            v_acc        = accuracy_score(val_labels.cpu(), v_pred.cpu())

        if v_f1 > best_val_f1:
            best_val_f1      = v_f1
            best_val_acc     = v_acc
            patience_counter = 0
            best_state       = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"  [Seed {seed}] Ep {epoch:04d} | "
                f"L_cls:{loss_cls.item():.4f} "
                f"L_ph:{lph.item():.4f} L_pb:{lpb.item():.4f} | "
                f"L_sh:{lsh.item():.4f} L_sb:{lsb.item():.4f} | "
                f"L_sc:{l_sc.item():.4f} | "
                f"Val F1:{v_f1:.4f}"
            )

        if patience_counter > args.patience:
            print(f"  [Seed {seed}] Early stopping at epoch {epoch}.")
            break

    model.load_state_dict({k: v.to(args.device) for k, v in best_state.items()})
    te_acc, te_f1, te_rec, te_pre = eval_split(model, data, data.test_idx)
    return best_val_f1, best_val_acc, te_acc, te_f1, te_rec, te_pre
