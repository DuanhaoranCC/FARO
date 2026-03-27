"""
Default hyperparameters for each supported dataset.

When --dataset is specified, the corresponding defaults are applied first.
Any explicitly passed CLI argument will override these defaults.
"""

DATASET_DEFAULTS = {
    "twibot-20": {
        "lr":            0.02,
        "weight_decay":  1e-3,
        "lambda_proto":  0.3,
        "lambda_sep":    0.2,
        "margin":        0.6,
        "lambda_supcon": 0.1,
        "tau":           0.07,
        "n_sup":         512,
    },
    "MGTAB": {
        "lr":            0.03,
        "weight_decay":  1e-5,
        "lambda_proto":  0.8,
        "lambda_sep":    0.2,
        "margin":        1.2,
        "lambda_supcon": 0.5,
        "tau":           0.5,
        "n_sup":         512,
    },
    "Cresci-15": {
        "lr":            0.003,
        "weight_decay":  1e-5,
        "lambda_proto":  1.7,
        "lambda_sep":    0.4,
        "margin":        1.5,
        "lambda_supcon": 1.2,
        "tau":           0.3,
        "n_sup":         512,
    },
}

SUPPORTED_DATASETS = list(DATASET_DEFAULTS.keys())
