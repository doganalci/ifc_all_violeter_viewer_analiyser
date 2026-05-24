"""FP denetimi: modelin 'yanlış alarm'ları gerçekten yanlış mı?

GAT Test'te precision düşük çıkıyor (çok FP). Soru: bu FP node'lar
gerçekten sorunsuz mu (model halüsinasyon), yoksa geometrik olarak
ihlal ama bizim injection etiketimizde olmayan node'lar mı (model
haklı, etiket eksik)?

Bu script her FP node'u kural-oracle'dan (ml.data.rule_oracle) geçirir:
  * FP + oracle "ihlal" diyor → model HAKLI, etiket eksikti
    (baseline-native ihlal — closed-world sorununun kanıtı)
  * FP + oracle "temiz" diyor → model HALÜSİNASYON (gerçek yanlış alarm)

Çıktı:
  fp_audit.csv      her FP: ifc, guid, type, score, oracle_violation?, kategori
  fp_audit_summary.json

Kullanım:
    python ml/scripts/audit_fp.py \\
        --checkpoint ml_runs/<id>/best.pt \\
        --config ml_runs/<id>/config.json \\
        --split test --out-dir fp_audit/<id>
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
sys.path.insert(0, str(_ML.parent))

from paths import data_home
from ml.data import IFCViolationDataset, split_by_baseline
from ml.data.graph_loader import load_sample
from ml.data.rule_oracle import find_violations
from ml.data.sqlite_reader import DatasetReader
from ml.model import GATNodeClassifier, HeteroGATNodeClassifier
from ml.train.config import TrainConfig


def _build_model(cfg, in_dim, num_edge_types):
    if cfg.model == "hetero_gat":
        return HeteroGATNodeClassifier(
            in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=num_edge_types, heads=cfg.heads, dropout=cfg.dropout)
    return GATNodeClassifier(
        in_dim=in_dim, hidden_dim=cfg.hidden_dim,
        num_edge_types=num_edge_types, edge_emb_dim=cfg.edge_emb_dim,
        heads=cfg.heads, dropout=cfg.dropout)


def main() -> None:
    p = argparse.ArgumentParser(description="FP node'ları oracle ile denetle")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    threshold = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = IFCViolationDataset(
        root=cfg.cache_root, dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
        use_rule_oracle=getattr(cfg, "use_rule_oracle", False),
        mask_numeric_features=getattr(cfg, "mask_numeric_features", False),
        mask_pset_features=getattr(cfg, "mask_pset_features", False),
        mask_type_features=getattr(cfg, "mask_type_features", False),
    )
    splits = split_by_baseline(ds.baseline_ids, val_frac=cfg.val_frac,
                               test_frac=cfg.test_frac, seed=cfg.split_seed)
    if a.split == "all":
        idx = list(range(len(ds)))
    else:
        idx = {"train": splits.train, "val": splits.val, "test": splits.test}[a.split]
    if not idx:
        print(f"⚠️  '{a.split}' boş."); return

    model = _build_model(cfg, ds.feature_dim, ds.num_edge_types).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    # IFC'lerin graph'larından node attribute'larına erişmek için
    # DatasetReader ile graph_path'leri çöz, oracle'ı orijinal geometride çalıştır.
    reader = DatasetReader(cfg.dataset_root)
    # ifc_id -> graph_path haritası
    id_to_graph: dict[str, str] = {}
    for e in reader.list_models(kind="violated", status="ok") + \
             reader.list_models(kind="violated", status="partial"):
        if e.graph_path:
            id_to_graph[e.id] = str(e.graph_path)
    reader.close()

    loader = DataLoader(ds[idx], batch_size=1, shuffle=False)

    fp_rows = []
    n_fp = 0
    n_fp_oracle_real = 0   # FP ama oracle ihlal diyor → model haklı
    n_fp_halluc = 0        # FP ve oracle temiz → halüsinasyon

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ifc_id = ds[idx[bi]].ifc_id
            b = batch.to(device)
            logits = model(b.x, b.edge_index, b.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(bool)
            y = batch.y.cpu().numpy().astype(bool)
            node_ids = batch.node_ids[0] if hasattr(batch, "node_ids") else []

            # Bu IFC'nin oracle ihlallerini orijinal graph'tan al
            oracle_hits: dict = {}
            gpath = id_to_graph.get(ifc_id)
            if gpath and Path(gpath).exists():
                try:
                    sample = load_sample(gpath, None, ifc_id=ifc_id)
                    oracle_hits = find_violations(sample.graph)
                except Exception:
                    oracle_hits = {}

            for i in range(len(y)):
                if preds[i] and not y[i]:    # FP
                    n_fp += 1
                    guid = node_ids[i] if i < len(node_ids) else ""
                    oracle_info = oracle_hits.get(guid)
                    is_oracle_real = oracle_info is not None
                    if is_oracle_real:
                        n_fp_oracle_real += 1
                    else:
                        n_fp_halluc += 1
                    fp_rows.append({
                        "ifc_id": ifc_id,
                        "node_guid": guid,
                        "score": round(float(probs[i]), 4),
                        "oracle_says_violation": is_oracle_real,
                        "oracle_category": (oracle_info or {}).get("category", ""),
                        "oracle_rule": (oracle_info or {}).get("rule", ""),
                    })

    # CSV
    fp_rows.sort(key=lambda r: (not r["oracle_says_violation"], -r["score"]))
    if fp_rows:
        with (out / "fp_audit.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
            w.writeheader()
            w.writerows(fp_rows)

    pct_real = (n_fp_oracle_real / n_fp * 100) if n_fp else 0.0
    summary = {
        "split": a.split,
        "threshold": threshold,
        "n_false_positive": n_fp,
        "fp_oracle_confirms_violation": n_fp_oracle_real,
        "fp_hallucination": n_fp_halluc,
        "pct_fp_actually_real": round(pct_real, 1),
        "verdict": (
            "🟢 FP'lerin ÇOĞU gerçek geometrik ihlal — model etiketsiz "
            "ihlalleri buluyor (closed-world sorunu kanıtı, model aslında iyi)"
            if pct_real >= 60 else
            "🟡 FP'lerin bir kısmı gerçek — karışık"
            if pct_real >= 30 else
            "🔴 FP'lerin çoğu halüsinasyon — model aşırı işaretliyor"
        ),
    }
    (out / "fp_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"Toplam FP (yanlış alarm): {n_fp}")
    print(f"  ✓ Oracle 'gerçek ihlal' diyor: {n_fp_oracle_real} "
          f"({pct_real:.1f}%) → model HAKLI, etiket eksikti")
    print(f"  ✗ Oracle 'temiz' diyor:        {n_fp_halluc} "
          f"({100-pct_real:.1f}%) → halüsinasyon")
    print()
    print(summary["verdict"])
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
