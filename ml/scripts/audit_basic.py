"""FP kırılımı: 157 yanlış alarm nereden geliyor? (basic_inject için)

GAT Test'te precision düşük. Bu script FP'leri ŞUNA göre kırar:
  * Node tipi (Door / Column / diğer)
  * basic_inject etiketinde 'compliant' (hard negative) mi yoksa
    etiketsiz (önemsiz yapı düğümü) mü?

Böylece 'model neyi karıştırıyor' netleşir:
  - Çok 'compliant Door' FP → kapı genişlik eşiğini öğrenememiş
  - Çok 'Column' FP → kolon yakınlık ölçemiyor (mesafe feature'ı yok)
  - Çok 'önemsiz' FP → duvar/döşeme uyduruyor (beklenmez)

Kullanım:
    python ml/scripts/audit_basic.py --checkpoint ml_runs/<id>/best.pt \\
        --config ml_runs/<id>/config.json --split test --out-dir audit/<id>
"""
from __future__ import annotations

import argparse
import collections
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
from ml.data.sqlite_reader import DatasetReader
from ml.model import GATNodeClassifier, HeteroGATNodeClassifier
from ml.train.config import TrainConfig


def _build(cfg, in_dim, ne):
    if cfg.model == "hetero_gat":
        return HeteroGATNodeClassifier(in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=ne, heads=cfg.heads, dropout=cfg.dropout)
    return GATNodeClassifier(in_dim=in_dim, hidden_dim=cfg.hidden_dim,
        num_edge_types=ne, edge_emb_dim=cfg.edge_emb_dim, heads=cfg.heads,
        dropout=cfg.dropout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    thr = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

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
    idx = {"train": splits.train, "val": splits.val, "test": splits.test,
           "all": list(range(len(ds)))}[a.split]
    if not idx:
        print("split boş"); return

    model = _build(cfg, ds.feature_dim, ds.num_edge_types).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device))
    model.eval()

    # ifc_id -> labels.json yolu (basic_inject status'leri için)
    reader = DatasetReader(cfg.dataset_root)
    id_to_lab = {}
    for st_ in ("ok", "partial"):
        for e in reader.list_models(kind="violated", status=st_):
            if e.labels_path and e.labels_path.exists():
                id_to_lab[e.id] = str(e.labels_path)
    reader.close()

    def _node_status(ifc_id, guid):
        """basic_inject etiketinden node'un durumu: applied/compliant/none."""
        lp = id_to_lab.get(ifc_id)
        if not lp:
            return "none"
        try:
            doc = json.loads(Path(lp).read_text(encoding="utf-8"))
        except Exception:
            return "none"
        for l in doc.get("labels", []):
            if l.get("ifc_global_id") == guid:
                return l.get("status", "none")  # applied | compliant
        return "none"

    loader = DataLoader(ds[idx], batch_size=1, shuffle=False)
    # FP kırılımı: (node_type, status) -> sayı
    fp_break = collections.Counter()
    tp_break = collections.Counter()
    fp_rows = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ifc_id = ds[idx[bi]].ifc_id
            b = batch.to(device)
            probs = torch.sigmoid(model(b.x, b.edge_index, b.edge_type)).cpu().numpy()
            preds = (probs >= thr).astype(bool)
            y = batch.y.cpu().numpy().astype(bool)
            nids = batch.node_ids[0] if hasattr(batch, "node_ids") else []
            ntypes = (batch.node_types[0] if hasattr(batch, "node_types")
                      else [""] * len(y))
            for i in range(len(y)):
                guid = nids[i] if i < len(nids) else ""
                nt = ntypes[i] if i < len(ntypes) else "?"
                if preds[i] and not y[i]:   # FP
                    stt = _node_status(ifc_id, guid)
                    bucket = "ADDED_COLUMN" if nt in ("IfcColumn",) else nt
                    fp_break[(bucket, stt)] += 1
                    fp_rows.append({"ifc": ifc_id, "guid": guid, "type": nt,
                                    "basic_status": stt, "score": round(float(probs[i]), 3)})
                elif preds[i] and y[i]:     # TP
                    bucket = "ADDED_COLUMN" if nt in ("IfcColumn",) else nt
                    tp_break[bucket] += 1

    # Rapor
    lines = ["# FP KIRILIMI (yanlış alarmlar nereden?)\n"]
    total_fp = sum(fp_break.values())
    lines.append(f"Toplam FP: {total_fp}\n")
    lines.append("(node_tipi, basic_status) → sayı:")
    for (nt, stt), n in sorted(fp_break.items(), key=lambda kv: -kv[1]):
        tag = {"compliant": "HARD-NEGATİF (uyumlu değişiklik)",
               "applied": "?? (ihlal etiketli ama FP — tuhaf)",
               "none": "önemsiz (etiketsiz yapı düğümü)"}.get(stt, stt)
        lines.append(f"  {nt:<20s} {stt:<10s} {n:>5d}   [{tag}]")
    lines.append("\nTP kırılımı (doğru yakalanan ihlaller):")
    for nt, n in sorted(tp_break.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {nt:<20s} {n:>5d}")

    # Yorum
    col_fp = sum(n for (nt, _), n in fp_break.items() if nt == "ADDED_COLUMN")
    door_compliant_fp = fp_break.get(("IfcDoor", "compliant"), 0)
    trivial_fp = sum(n for (_, stt), n in fp_break.items() if stt == "none")
    lines.append("\n--- TEŞHİS ---")
    if total_fp:
        lines.append(f"Kolon FP: {col_fp} ({col_fp*100//max(total_fp,1)}%) "
                     "→ yüksekse: kolon-kapı mesafesi feature'ı yok, model "
                     "tüm kolonları işaretliyor.")
        lines.append(f"Uyumlu kapı FP: {door_compliant_fp} → yüksekse: kapı "
                     "genişlik eşiğini (0.90) tam öğrenememiş.")
        lines.append(f"Önemsiz (yapı) FP: {trivial_fp} → yüksekse: duvar/döşeme "
                     "uyduruyor (beklenmez).")

    report = "\n".join(lines)
    (out / "fp_breakdown.txt").write_text(report, encoding="utf-8")
    if fp_rows:
        with (out / "fp_rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fp_rows[0].keys()))
            w.writeheader(); w.writerows(fp_rows)
    print(report)
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
