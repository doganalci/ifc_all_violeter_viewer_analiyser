"""Emit per-IFC predictions in a format that the viewer can overlay.

Output (one JSON per IFC):

    {
      "model_version": "gat_v1",
      "ifc_id": "<uuid>",
      "predictions": [{"guid": "...", "score": 0.87, "predicted": 1}, ...],
      "metrics": {"precision": 0.83, "recall": 0.79, "decoy_fpr": 0.12}
    }

Usage:
    python -m scripts.export_predictions --dataset-root ~/Desktop/codex1 \\
        --checkpoint runs/run_XXX/best.pt --config runs/run_XXX/config.json \\
        --out-dir predictions/run_XXX
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--model-version", default="gat_v1")
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    threshold = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()
    out_dir = Path(a.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
    )
    baseline_ids = [ds[i].baseline_id for i in range(len(ds))]
    splits = split_by_baseline(
        baseline_ids, val_frac=cfg.val_frac,
        test_frac=cfg.test_frac, seed=cfg.split_seed,
    )
    if a.split == "all":
        idx = list(range(len(ds)))
    else:
        idx = {"train": splits.train, "val": splits.val, "test": splits.test}[a.split]

    model = _build_model(cfg, ds.feature_dim, ds.num_edge_types).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        for i in idx:
            data = ds[i].to(device)
            logits = model(data.x, data.edge_index, data.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(np.int64)
            y = data.y.cpu().numpy()
            decoy = data.decoy_mask.cpu().numpy()
            res = evaluate_predictions(y, preds, decoy, categories=data.categories)

            payload = {
                "model_version": a.model_version,
                "ifc_id": data.ifc_id,
                "baseline_id": data.baseline_id,
                "threshold": threshold,
                "predictions": [
                    {"guid": guid, "score": float(probs[j]), "predicted": int(preds[j])}
                    for j, guid in enumerate(data.node_ids)
                ],
                "metrics": res.to_dict(),
            }
            (out_dir / f"{data.ifc_id}.predictions.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    print(f"[export] wrote {len(idx)} prediction files to {out_dir}")


if __name__ == "__main__":
    main()
