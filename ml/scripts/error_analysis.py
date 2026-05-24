"""Eğitilmiş modelin hatalarını ayrıntılı dökümler.

Çıktılar:
  * confusion_matrix.txt          ASCII confusion matrix + tüm metrikler
  * metrics.json                  Tüm sayısal metrikler (AUC/MCC/per-cat dahil)
  * threshold_sweep.csv           21 eşikte P/R/F1 (doğru eşiği seçmek için)
  * false_positives.csv           Model "ihlal" dedi ama gerçekte değil
  * false_negatives.csv           Gerçek ihlal ama model kaçırdı
  * per_ifc.csv                   IFC başına TP/FP/FN/TN + recall

Kullanım:
    python ml/scripts/error_analysis.py \\
        --checkpoint ml_runs/<id>/best.pt \\
        --config ml_runs/<id>/config.json \\
        --split test \\
        --out-dir error_reports/<id>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML))
sys.path.insert(0, str(_ML.parent))

from paths import data_home
from data import IFCViolationDataset, split_by_baseline
from model import GATNodeClassifier, HeteroGATNodeClassifier
from train.config import TrainConfig
from train.metrics import evaluate_predictions


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
    p = argparse.ArgumentParser(description="Detaylı hata analizi raporu üret.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="runs/<id>/config.json")
    p.add_argument("--dataset-root", default=None, help="Varsayılan: IFC_DATA_HOME")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-n", type=int, default=200,
                   help="false_positives/negatives CSV'de en yüksek skorlu N satır")
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    threshold = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
    )
    if a.split == "all":
        idx = list(range(len(ds)))
    else:
        splits = split_by_baseline(
            ds.baseline_ids, val_frac=cfg.val_frac,
            test_frac=cfg.test_frac, seed=cfg.split_seed,
        )
        idx = {"train": splits.train, "val": splits.val, "test": splits.test}[a.split]

    if not idx:
        print(f"⚠️  '{a.split}' split'i boş.")
        return

    loader = DataLoader(ds[idx], batch_size=1, shuffle=False)
    model = _build_model(cfg, ds.feature_dim, ds.num_edge_types).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    # Tüm node'ları topla — hangi IFC'den geldiğini de izle
    y_all, p_all, s_all, d_all = [], [], [], []
    c_all: list[str | None] = []
    ifc_ids: list[str] = []
    node_guids: list[str] = []
    node_types: list[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            ifc_id = ds[idx[batch_idx]].ifc_id
            batch_on = batch.to(device)
            logits = model(batch_on.x, batch_on.edge_index, batch_on.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(np.int64)

            y = batch.y.cpu().numpy()
            d = batch.decoy_mask.cpu().numpy()
            cats = list(batch.categories[0]) if batch.categories else [None] * len(y)
            guids = list(getattr(batch, "node_ids", [[]])[0]) if hasattr(batch, "node_ids") else [""] * len(y)
            ntypes = list(getattr(batch, "node_types", [[]])[0]) if hasattr(batch, "node_types") else [""] * len(y)

            y_all.append(y); p_all.append(preds); s_all.append(probs); d_all.append(d)
            c_all.extend(cats)
            ifc_ids.extend([ifc_id] * len(y))
            node_guids.extend(guids if len(guids) == len(y) else [""] * len(y))
            node_types.extend(ntypes if len(ntypes) == len(y) else [""] * len(y))

    y_arr = np.concatenate(y_all)
    p_arr = np.concatenate(p_all)
    s_arr = np.concatenate(s_all)
    d_arr = np.concatenate(d_all)

    # === Aggregate metrikler =====================================
    res = evaluate_predictions(
        y_arr.astype(int), p_arr, d_arr,
        categories=c_all, y_score=s_arr,
    )

    (out / "metrics.json").write_text(
        json.dumps(res.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cm_text = (
        f"# Hata Analizi — split={a.split}, threshold={threshold:.2f}\n"
        f"# Toplam node: {len(y_arr)}  IFC: {len(set(ifc_ids))}\n\n"
        + res.confusion_matrix_pretty() + "\n\n"
        "Per-Category Precision/Recall/F1:\n"
    )
    for cat in sorted(set(res.per_category_recall) | set(res.per_category_precision)):
        cm_text += (
            f"  {cat:<32s}  "
            f"P={res.per_category_precision.get(cat, 0):.3f}  "
            f"R={res.per_category_recall.get(cat, 0):.3f}  "
            f"F1={res.per_category_f1.get(cat, 0):.3f}  "
            f"(n={res.per_category_support.get(cat, 0)})\n"
        )
    (out / "confusion_matrix.txt").write_text(cm_text, encoding="utf-8")

    # === Threshold sweep CSV =====================================
    with (out / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        for row in res.threshold_sweep:
            w.writerow([row["threshold"], row["precision"], row["recall"],
                        row["f1"], row["tp"], row["fp"], row["fn"]])

    # === False positives CSV =====================================
    fp_mask = p_arr.astype(bool) & ~y_arr.astype(bool)
    fp_rows = []
    for i in np.where(fp_mask)[0]:
        fp_rows.append({
            "ifc_id": ifc_ids[i],
            "node_guid": node_guids[i],
            "node_type": node_types[i],
            "score": float(s_arr[i]),
            "is_decoy": bool(d_arr[i]),
        })
    fp_rows.sort(key=lambda r: -r["score"])
    _write_csv(out / "false_positives.csv", fp_rows[: a.top_n])

    # === False negatives CSV =====================================
    fn_mask = ~p_arr.astype(bool) & y_arr.astype(bool)
    fn_rows = []
    for i in np.where(fn_mask)[0]:
        fn_rows.append({
            "ifc_id": ifc_ids[i],
            "node_guid": node_guids[i],
            "node_type": node_types[i],
            "category": c_all[i] or "",
            "score": float(s_arr[i]),
        })
    fn_rows.sort(key=lambda r: r["score"])
    _write_csv(out / "false_negatives.csv", fn_rows[: a.top_n])

    # === Per-IFC breakdown =======================================
    per_ifc: dict[str, dict] = {}
    for i, iid in enumerate(ifc_ids):
        b = per_ifc.setdefault(iid, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
        yt, yp = bool(y_arr[i]), bool(p_arr[i])
        if yt and yp: b["tp"] += 1
        elif yt and not yp: b["fn"] += 1
        elif not yt and yp: b["fp"] += 1
        else: b["tn"] += 1
        b["n"] += 1

    per_rows = []
    for iid, b in per_ifc.items():
        rec = b["tp"] / (b["tp"] + b["fn"]) if (b["tp"] + b["fn"]) else 0.0
        prec = b["tp"] / (b["tp"] + b["fp"]) if (b["tp"] + b["fp"]) else 0.0
        per_rows.append({
            "ifc_id": iid,
            "nodes": b["n"],
            "tp": b["tp"], "fp": b["fp"], "fn": b["fn"], "tn": b["tn"],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if (prec + rec) else 0.0,
        })
    per_rows.sort(key=lambda r: (r["recall"], -r["fn"]))
    _write_csv(out / "per_ifc.csv", per_rows)

    print(f"✅ Rapor: {out}")
    print()
    print(cm_text)
    print(f"   {len(fp_rows)} FP, {len(fn_rows)} FN  (top {a.top_n} CSV'ye yazıldı)")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
