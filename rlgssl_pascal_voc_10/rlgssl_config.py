import torch

### Experimental Configuration Structure ###


config = {
    "root": {
        "key": "experiment_type",
        "values": ["rlgssl"],
        "default": None,
    },
    "common": {
        "dataset": "pascal_voc_10",
        "learning_rate": 1e-3,
        "patience": 100,
        "epochs": 512,
        "steps_per_epoch": 50,
        "train_batch_size": 8,
        "validation_batch_size": 8,
        "unlabeled_batch_size": 8,
        "accumulate": 4,
        "clip": 1.0,
        "autocast_type": 'float32',
        "unlabeled_workers": 4,
        "validation_workers": 0,
        "train_workers": 4,
        "unlabeled_fraction": 0.99,
        "iterations": 1,
        "num_classes": 10,
        "dropout": 0.0
    },
    "check_unique": True,
    "repetitions": 1,
}