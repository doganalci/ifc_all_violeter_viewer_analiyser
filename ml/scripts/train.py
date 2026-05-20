"""Train the GAT baseline.

Usage:
    python -m scripts.train --dataset-root ~/Desktop/codex1 --epochs 50

The dataset root must contain `violation_pool.sqlite` and an
`ifc_models/` tree populated by codex1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `data/`, `model/`, `train/` importable when run via `python scripts/train.py`.
_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML))
sys.path.insert(0, str(_ML.parent))

from paths import data_home, ml_runs_dir
from train.config import TrainConfig
from train.loop import run_training


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a GAT for IFC violation detection.")
    p.add_argument("--dataset-root", default=None, help="Varsayılan: IFC_DATA_HOME")
    p.add_argument("--cache-root", default="./data/cache")
    p.add_argument("--include-baselines", action="store_true")
    p.add_argument("--model", choices=["gat", "hetero_gat"], default="gat")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default=None, help="Varsayılan: IFC_DATA_HOME/ml_runs")
    p.add_argument("--run-name", default=None)
    a = p.parse_args()
    ds = a.dataset_root or str(data_home())
    run_dir = a.run_dir or str(ml_runs_dir())
    return TrainConfig(
        dataset_root=ds,
        cache_root=a.cache_root,
        include_baselines=a.include_baselines,
        model=a.model,
        hidden_dim=a.hidden_dim,
        heads=a.heads,
        dropout=a.dropout,
        epochs=a.epochs,
        lr=a.lr,
        weight_decay=a.weight_decay,
        pos_weight=a.pos_weight,
        patience=a.patience,
        threshold=a.threshold,
        val_frac=a.val_frac,
        test_frac=a.test_frac,
        split_seed=a.split_seed,
        device=a.device,
        seed=a.seed,
        run_dir=run_dir,
        run_name=a.run_name,
    )


if __name__ == "__main__":
    run_training(_parse())
