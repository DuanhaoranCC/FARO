# FCN-SC: Feature Coupling Network with Supervised Contrastive Loss

Bot detection on social graphs using dual-prototype coupling matrices and supervised contrastive learning.

## Project Structure

```
.
├── configs.py    # Dataset-specific default hyperparameters
├── dataset.py    # Data loading for each supported dataset
├── models.py     # GroupFeatureEncoder, DualPrototypeCouplingModule, FCN_SC_v5
├── train.py      # Training loop, prototype initialisation, evaluation
└── main.py       # Entry point (argument parsing, multi-seed runner)
```

## Supported Datasets

| Dataset     | Split strategy              |
|-------------|----------------------------|
| `twibot-20` | 70 / 20 / 10 % random      |
| `Cresci-15` | Pre-defined index files     |
| `MGTAB`     | 70 / 20 / 10 % (seed=42)   |

Each dataset has its own default hyperparameters in `configs.py`.

## Usage

```bash
# Use dataset defaults
python main.py --dataset twibot-20
python main.py --dataset MGTAB
python main.py --dataset Cresci-15

# Override individual hyperparameters
python main.py --dataset MGTAB --lr 0.01 --tau 0.3

# Multi-seed evaluation
python main.py --dataset twibot-20 --seeds 0 1 2 3 4

# Custom data directory and GPU
python main.py --dataset twibot-20 --base_dir /data/datasets/ --gpu 1
```

## Arguments

| Argument         | Description                                      | Default (per dataset) |
|------------------|--------------------------------------------------|-----------------------|
| `--dataset`      | Dataset name                                     | `twibot-20`           |
| `--base_dir`     | Root directory of datasets                       | `../bd_dataset/`      |
| `--gpu`          | GPU index                                        | `0`                   |
| `--seeds`        | Random seeds for multi-run evaluation            | `[0]`                 |
| `--epochs`       | Maximum training epochs                          | `500`                 |
| `--patience`     | Early stopping patience                          | `30`                  |
| `--group_dim`    | Group representation dimension                   | `32`                  |
| `--hidden_dim`   | MLP hidden dimension                             | `64`                  |
| `--lr`           | Learning rate                                    | dataset default       |
| `--weight_decay` | AdamW weight decay                               | dataset default       |
| `--lambda_proto` | Prototype alignment loss weight                  | dataset default       |
| `--lambda_sep`   | Prototype separation loss weight                 | dataset default       |
| `--margin`       | Separation hinge margin                          | dataset default       |
| `--lambda_supcon`| Supervised contrastive loss weight               | dataset default       |
| `--tau`          | Contrastive loss temperature                     | dataset default       |
| `--n_sup`        | Training nodes sampled per epoch for SupCon      | dataset default       |

## Dataset Defaults

| Hyperparameter   | twibot-20 | MGTAB  | Cresci-15 |
|------------------|-----------|--------|-----------|
| `lr`             | 0.02      | 0.03   | 0.003     |
| `weight_decay`   | 1e-3      | 1e-5   | 1e-5      |
| `lambda_proto`   | 0.3       | 0.8    | 1.7       |
| `lambda_sep`     | 0.2       | 0.2    | 0.4       |
| `margin`         | 0.6       | 1.2    | 1.5       |
| `lambda_supcon`  | 0.1       | 0.5    | 1.2       |
| `tau`            | 0.07      | 0.5    | 0.3       |
| `n_sup`          | 512       | 512    | 512       |

## Requirements

```
torch
torch_geometric
scikit-learn
numpy
```
