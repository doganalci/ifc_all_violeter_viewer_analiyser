"""Re-evaluate a saved checkpoint on the test split.

Usage:
    python -m scripts.evaluate --dataset-root ~/Desktop/codex1 \\
        --checkpoint runs/run_XXX/best.pt --config runs/run_XXX/config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML.parent))

from paths import data_home
from ml.data import IFCViolationDataset, split_by_baseline
from ml.model import GATNodeClassifier, HeteroGATNodeClassifier
from ml.train.config import TrainConfig
from ml.train.metrics import evaluate_predictions


def _build_model(cfg: TrainConfig, in_dim: int, num_edge_types: int):
    if cfg.model == "hetero_gat":
        return HeteroGATNodeClassifier(
            in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=num_edge_types, heads=cfg.heads, dropout=cfg.dropout,
        )
    return GATNodeClassifier(
        in_dim=in_dim, hidden_dim=cfg.hidden_dim,
        num_edge_types=num_edge_types, edge_emb_dim=cfg.edge_emb_dim,
        heads=cfg.heads, dropout=cfg.dropout,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=None, help="Varsayılan: IFC_DATA_HOME")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="path to runs/<id>/config.json")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--threshold", type=float, default=None)
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    threshold = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()

    ds = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
        use_rule_oracle=getattr(cfg, "use_rule_oracle", False),
        mask_numeric_features=getattr(cfg, "mask_numeric_features", False),
        mask_pset_features=getattr(cfg, "mask_pset_features", False),
        mask_type_features=getattr(cfg, "mask_type_features", False),
    )
    baseline_ids = [ds[i].baseline_id for i in range(len(ds))]
    splits = split_by_baseline(
        baseline_ids, val_frac=cfg.val_frac,
        test_frac=cfg.test_frac, seed=cfg.split_seed,
    )
    idx = {"train": splits.train, "val": splits.val, "test": splits.test}[a.split]
    if not idx:
        print(f"[evaluate] {a.split} split is empty")
        return
    loader = DataLoader(ds[idx], batch_size=4, shuffle=False)

    model = _build_model(cfg, ds.feature_dim, ds.num_edge_types).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    y_all, p_all, s_all, d_all = [], [], [], []
    c_all: list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_all.append(batch.y.cpu().numpy())
            p_all.append((probs >= threshold).astype(np.int64))
            s_all.append(probs)
            d_all.append(batch.decoy_mask.cpu().numpy())
            for cats in batch.categories:
                c_all.extend(cats)

    res = evaluate_predictions(
        np.concatenate(y_all),
        np.concatenate(p_all),
        np.concatenate(d_all),
        categories=c_all,
        y_score=np.concatenate(s_all),
    )
    print(json.dumps(res.to_dict(), indent=2))
    print()
    print("Confusion matrix:")
    print(res.confusion_matrix_pretty())


if __name__ == "__main__":
    main()
