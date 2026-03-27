"""
Entry point for training the FCN-SC bot-detection model.

Dataset-specific hyperparameter defaults are loaded automatically from
configs.py.  Any argument passed explicitly on the command line takes
precedence over those defaults.

Example
-------
# Use twibot-20 defaults:
    python main.py --dataset twibot-20

# Use MGTAB defaults, override learning rate:
    python main.py --dataset MGTAB --lr 0.01

# Multi-seed run on Cresci-15:
    python main.py --dataset Cresci-15 --seeds 0 1 2 3 4
"""

import argparse
import sys

import numpy as np
import torch

from configs import DATASET_DEFAULTS, SUPPORTED_DATASETS
from dataset import load_data
from train import run_one_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FCN-SC: Feature Coupling Network with Supervised Contrastive Loss"
    )

    # --- Infrastructure ---
    parser.add_argument("--dataset",    type=str, default="twibot-20",
                        choices=SUPPORTED_DATASETS,
                        help="Target dataset (default: twibot-20)")
    parser.add_argument("--base_dir",  type=str, default="../bd_dataset/",
                        help="Root directory containing dataset folders")
    parser.add_argument("--gpu",       type=int, default=0,
                        help="GPU index; uses CPU when CUDA is unavailable")
    parser.add_argument("--seeds",     type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                        help="List of random seeds for multi-run evaluation")

    # --- Architecture ---
    parser.add_argument("--group_dim",  type=int, default=32,
                        help="Dimension of each group representation")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension of the MLP classifier")

    # --- Training schedule ---
    parser.add_argument("--epochs",   type=int, default=500)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience (epochs)")

    # --- Loss weights & hyperparameters ---
    # Defaults below are intentionally None so we can detect which flags were
    # explicitly provided and apply dataset defaults for the rest.
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--weight_decay",  type=float, default=None)
    parser.add_argument("--lambda_proto",  type=float, default=None,
                        help="Weight for prototype alignment loss")
    parser.add_argument("--lambda_sep",    type=float, default=None,
                        help="Weight for prototype separation loss")
    parser.add_argument("--margin",        type=float, default=None,
                        help="Margin for separation hinge loss")
    parser.add_argument("--lambda_supcon", type=float, default=None,
                        help="Weight for supervised contrastive loss")
    parser.add_argument("--tau",           type=float, default=None,
                        help="Temperature for contrastive loss")
    parser.add_argument("--n_sup",         type=int,   default=None,
                        help="Number of training nodes sampled per epoch for SupCon")

    return parser


def _apply_dataset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill in None-valued hyperparameters with dataset-specific defaults.
    Explicitly provided CLI arguments are never overwritten.
    """
    defaults = DATASET_DEFAULTS[args.dataset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    args   = _apply_dataset_defaults(args)
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # --- Print configuration ---
    print("=" * 60)
    print(f"Dataset        : {args.dataset}")
    print(f"Device         : {args.device}")
    print(f"Seeds          : {args.seeds}")
    print(f"lr             : {args.lr}")
    print(f"weight_decay   : {args.weight_decay}")
    print(f"lambda_proto   : {args.lambda_proto}")
    print(f"lambda_sep     : {args.lambda_sep}")
    print(f"margin         : {args.margin}")
    print(f"lambda_supcon  : {args.lambda_supcon}")
    print(f"tau            : {args.tau}")
    print(f"n_sup          : {args.n_sup}")
    print("=" * 60)

    # --- Load data ---
    data = load_data(args)

    K        = (4 if args.dataset in ("twibot-20", "Cresci-15") else 3) + args.num_relations
    triu_dim = K * (K - 1) // 2
    clf_dim  = 2 * triu_dim
    print(f"Nodes          : {data.num_total}")
    print(f"K              : {K}  |  clf_input : {clf_dim}")
    print(f"Train / Val / Test : "
          f"{len(data.train_idx)} / {len(data.val_idx)} / {len(data.test_idx)}")
    print()

    # --- Multi-seed training ---
    val_f1_list  = []
    val_acc_list = []
    te_acc_list  = []
    te_f1_list   = []
    te_rec_list  = []
    te_pre_list  = []

    for seed in args.seeds:
        print("─" * 55)
        print(f"Running seed {seed} ...")
        best_val_f1, best_val_acc, te_acc, te_f1, te_rec, te_pre = \
            run_one_seed(seed, data, args)

        val_f1_list.append(best_val_f1)
        val_acc_list.append(best_val_acc)
        te_acc_list.append(te_acc)
        te_f1_list.append(te_f1)
        te_rec_list.append(te_rec)
        te_pre_list.append(te_pre)

        print(f"  [Seed {seed}] Best Val F1: {best_val_f1:.4f} | "
              f"Test Acc: {te_acc:.4f} | Test F1: {te_f1:.4f} | "
              f"Rec: {te_rec:.4f} | Pre: {te_pre:.4f}")

    # --- Aggregate results ---
    print("\n" + "★" * 60)
    print(f"FINAL RESULTS ({args.dataset})  —  {len(args.seeds)} seed(s)")
    print("★" * 60)

    print("\nPer-seed results:")
    for i, seed in enumerate(args.seeds):
        print(f"  Seed {seed:>5}: "
              f"val_f1={val_f1_list[i]:.4f}  "
              f"test_acc={te_acc_list[i]:.4f}  "
              f"test_f1={te_f1_list[i]:.4f}  "
              f"rec={te_rec_list[i]:.4f}  "
              f"pre={te_pre_list[i]:.4f}")

    print("\nMean ± Std:")
    print(f"  Val  F1   : {np.mean(val_f1_list):.4f} ± {np.std(val_f1_list):.4f}")
    print(f"  Test Acc  : {np.mean(te_acc_list):.4f} ± {np.std(te_acc_list):.4f}")
    print(f"  Test F1   : {np.mean(te_f1_list):.4f} ± {np.std(te_f1_list):.4f}")
    print(f"  Recall    : {np.mean(te_rec_list):.4f} ± {np.std(te_rec_list):.4f}")
    print(f"  Precision : {np.mean(te_pre_list):.4f} ± {np.std(te_pre_list):.4f}")


if __name__ == "__main__":
    main()
