"""Anomali tespiti — normal baseline'larla eğit, ihlalleri (görmeden) yakala.

Akış:
  1. Bir dataset paketinin BASELINE'larıyla Graph AutoEncoder eğit
     (sadece normal; 'ihlal' etiketi GÖRÜLMEZ).
  2. Aynı paketin VIOLATED IFC'lerini skorla.
  3. Doğrula: ihlal node'ları (ground truth), normal node'lardan daha
     yüksek anomali skoru alıyor mu? → ROC-AUC.
     AUC yüksekse (>0.8): anomali tespiti ihlalleri ayırt ediyor →
     görülmemiş ihlal tiplerini de yakalayabilir.

ÖNEMLİ: Burada feature MASKELENMEZ — anomali, sayısal sapmadan (dar kapı)
beslenir. Kapı genişliği feature olarak GEREKLİ.

Kullanım:
    python ml/scripts/anomaly.py --tag basic2+1_baseline_v09_22 \\
        --epochs 100 --out-dir anomaly_reports/v09
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
from ml.data.graph_loader import load_sample
from ml.data.pyg_dataset import sample_to_data
from ml.data.sqlite_reader import DatasetReader
from ml.model.gae import GraphAutoEncoder
from violation_pool import storage


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = y_true.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    p = argparse.ArgumentParser(description="GAE anomali tespiti — normal'le eğit")
    p.add_argument("--tag", required=True, help="Dataset paketi (baseline+violated)")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    root = a.dataset_root or str(data_home())
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Baseline (normal) ve violated id'leri tag'den al ---------------
    base_ids = set(storage.ifc_ids_for_tags([a.tag], kind="baseline"))
    vio_ids = set(storage.ifc_ids_for_tags([a.tag], kind="violated"))
    reader = DatasetReader(root)

    def _load(entry):
        if not (entry.graph_path and entry.graph_path.exists()):
            return None
        return load_sample(entry.graph_path,
                           entry.labels_path if entry.labels_path
                           and entry.labels_path.exists() else None,
                           ifc_id=entry.id)

    base_samples, vio_samples = [], []
    for e in reader.list_models(kind="baseline", status="ok"):
        if e.id in base_ids:
            s = _load(e)
            if s: base_samples.append(s)
    for st_ in ("ok", "partial"):
        for e in reader.list_models(kind="violated", status=st_):
            if e.id in vio_ids:
                s = _load(e)
                if s: vio_samples.append(s)
    reader.close()

    if not base_samples:
        print(f"⚠️ '{a.tag}' için baseline graph yok."); return
    print(f"[anomaly] {len(base_samples)} normal baseline ile eğitim, "
          f"{len(vio_samples)} violated ile test.")

    # --- Veri (feature MASKELENMEZ) ------------------------------------
    train_data = [sample_to_data(s) for s in base_samples]
    in_dim = train_data[0].x.shape[1]
    n_edge = int(max((d.edge_type.max().item() if d.edge_type.numel() else 0)
                     for d in train_data)) + 1
    n_edge = max(n_edge, 7)
    loader = DataLoader(train_data, batch_size=8, shuffle=True)

    model = GraphAutoEncoder(in_dim, hidden_dim=a.hidden, latent_dim=a.latent,
                             num_edge_types=n_edge).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)

    # --- Eğitim (sadece normal reconstruct) ----------------------------
    model.train()
    for ep in range(1, a.epochs + 1):
        tot = 0.0
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            x_hat = model(b.x, b.edge_index, b.edge_type)
            loss = ((b.x - x_hat) ** 2).mean()
            loss.backward(); opt.step()
            tot += float(loss) * b.num_graphs
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  recon_loss={tot/len(train_data):.5f}")

    torch.save(model.state_dict(), out / "gae.pt")

    # --- Doğrulama: violated node skorları vs ground truth -------------
    model.eval()
    all_scores, all_y = [], []
    per_ifc = []
    for s in vio_samples:
        d = sample_to_data(s).to(device)
        score = model.anomaly_score(d.x, d.edge_index, d.edge_type).cpu().numpy()
        y = d.y.cpu().numpy()
        all_scores.append(score); all_y.append(y)
        if y.sum() > 0:
            pos_mean = float(score[y.astype(bool)].mean())
            neg_mean = float(score[~y.astype(bool)].mean()) if (~y.astype(bool)).any() else 0.0
            per_ifc.append({"ifc": s.ifc_id, "pos_score": round(pos_mean, 5),
                            "neg_score": round(neg_mean, 5)})

    scores = np.concatenate(all_scores); ys = np.concatenate(all_y)
    auc = _auc(ys, scores)
    # Eşik: normal node skorlarının 95. persentili
    thr = float(np.percentile(scores[ys == 0], 95)) if (ys == 0).any() else 0.0
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (ys == 1)).sum()); fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum()); tn = int(((pred == 0) & (ys == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    summary = {
        "tag": a.tag,
        "n_baseline_train": len(base_samples),
        "n_violated_test": len(vio_samples),
        "anomaly_auc": round(auc, 4),
        "threshold_p95": round(thr, 5),
        "precision_at_p95": round(prec, 3),
        "recall_at_p95": round(rec, 3),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "verdict": (
            f"🟢 AUC={auc:.3f} — anomali skoru ihlalleri AYIRT EDİYOR. "
            "Görülmemiş ihlal tiplerini yakalama umudu yüksek."
            if auc >= 0.8 else
            f"🟡 AUC={auc:.3f} — kısmi ayrım."
            if auc >= 0.65 else
            f"🔴 AUC={auc:.3f} — anomali skoru ihlali ayırt edemiyor. "
            "Saf reconstruction bu iş için zayıf; feature/mimari değişmeli."
        ),
    }
    (out / "anomaly_summary.json").write_text(
        json.dumps({**summary, "per_ifc_sample": per_ifc[:20]},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Anomali AUC: {auc:.4f}  (1.0=mükemmel ayrım, 0.5=rastgele)")
    print(f"p95 eşiğinde: precision={prec:.3f} recall={rec:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print("\n" + summary["verdict"])
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
