"""Sanity check: modelin gerçekten ne öğrendiğini test eder.

Üç ayrı testi peş peşe çalıştırır ve sonuçları tek bir rapora yazar.
Önceki konuşmalardaki üç şüpheyi doğrudan ölçer:

  1) FEATURE ABLATION
     - Test setindeki node feature'larından sayısal sütunları
       (OverallWidth/Height/NominalHeight/Elevation) sıfırla
     - Modelin recall'ı keskin düşerse: "trivial feature leak" kesin
     - Düşmezse: model başka sinyallere de bakıyor

  2) BASELINE AUDIT (kural-tabanlı oracle)
     - Baseline IFC'leri tara, basit geometrik kurallarla
       muhtemel ihlalleri işaretle (kapı genişliği < 90 cm, vb.)
     - Model bu baseline'ları çalıştırılınca kaç tane oracle-flag'ini
       yakalıyor? (= modelin "etiketsiz gerçek ihlal" yakalama yeteneği)
     - Düşükse: closed-world supervision sorunu kanıtlanmış

  3) SCORE DISTRIBUTION
     - True positive / true negative / decoy / baseline node'ların
       skor dağılımlarını histogram olarak çıkar
     - Dağılımlar üst üste binmişse: model "uydurma sıralama" yapıyor
     - Ayrılıyorsa: gerçek sinyal var

Kullanım:
    python ml/scripts/sanity_check.py \\
        --checkpoint ml_runs/<id>/best.pt \\
        --config ml_runs/<id>/config.json \\
        --out-dir sanity_reports/<id>
    # Sadece belli testler için: --mode ablation|baseline_audit|score_dist|all
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
from data.graph_loader import load_sample, NODE_TYPES
from data.features import _NUMERIC_ATTRS, FEATURE_DIM, build_node_features
from data.pyg_dataset import sample_to_data
from data.sqlite_reader import DatasetReader
from model import GATNodeClassifier, HeteroGATNodeClassifier
from train.config import TrainConfig
from train.metrics import evaluate_predictions


# ---------------------------------------------------------------------------
# 1) Kural-tabanlı oracle — baseline'larda etiketsiz ihlalleri bulur
# ---------------------------------------------------------------------------
# Eşikler erişilebilirlik mevzuatından kabaca türetildi (TS 9111 / ADA
# referansları). LLM bunlara uymaya çalışsa da baseline üretiminde her
# zaman ihlal kalır — script bunları bulup eğitim setindeki "etiketsiz
# negatif" gürültüsünü ölçer.
ORACLE_RULES = [
    # (ifc_type, attribute, "lt|gt", threshold, category)
    ("IfcDoor",     "OverallWidth",   "lt",  900.0,  "Kapı/Koridor"),    # < 90 cm
    ("IfcDoor",     "OverallHeight",  "lt",  2000.0, "Kapı/Koridor"),    # < 200 cm
    ("IfcWindow",   "OverallHeight",  "lt",  800.0,  "Yönlendirme/İşaretleme"),
    ("IfcSpace",    "OverallWidth",   "lt",  1500.0, "Manevra alanı"),   # < 150 cm
    ("IfcStair",    "NominalHeight",  "gt",  180.0,  "Merdiven"),        # > 18 cm rıht
    ("IfcRamp",     "OverallHeight",  "gt",  100.0,  "Rampa"),           # yüksek eğim
    ("IfcRailing",  "NominalHeight",  "lt",  900.0,  "Korkuluk/Küpeşte"),# < 90 cm
]


def _attr(node_data: dict, name: str) -> float | None:
    """Node verisi içinden attribute değerini float olarak getir."""
    attrs = node_data.get("attributes") or {}
    v = attrs.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def oracle_violations(sample) -> dict[str, str]:
    """Baseline graph'ında oracle kurallarının bulduğu ihlaller.

    Returns: { guid -> category } — model'in yakalaması beklenenler.
    """
    hits: dict[str, str] = {}
    for guid, node in sample.graph.nodes(data=True):
        ifc_type = node.get("ifc_type") or node.get("type")
        if not ifc_type:
            continue
        for typ, attr, op, thr, cat in ORACLE_RULES:
            if ifc_type != typ:
                continue
            val = _attr(node, attr)
            if val is None:
                continue
            if (op == "lt" and val < thr) or (op == "gt" and val > thr):
                hits[guid] = cat
                break
    return hits


# ---------------------------------------------------------------------------
# Model yükleme
# ---------------------------------------------------------------------------
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


def _load(cfg: TrainConfig, ckpt: str, in_dim: int, num_edge_types: int, device):
    model = _build_model(cfg, in_dim, num_edge_types).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1) Feature ablation
# ---------------------------------------------------------------------------
def run_ablation(cfg, ckpt, out: Path, threshold: float, device) -> dict:
    """Sayısal feature sütunlarını sıfırlayıp test'i yeniden çalıştır."""
    ds = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
    )
    splits = split_by_baseline(
        ds.baseline_ids, val_frac=cfg.val_frac,
        test_frac=cfg.test_frac, seed=cfg.split_seed,
    )
    if not splits.test:
        return {"skipped": "test split boş"}

    loader = DataLoader(ds[splits.test], batch_size=4, shuffle=False)
    model = _load(cfg, ckpt, ds.feature_dim, ds.num_edge_types, device)

    # Sayısal sütunların offset'i = len(NODE_TYPES); 4 sütun OverallWidth..Elevation
    num_start = len(NODE_TYPES)
    num_end = num_start + len(_NUMERIC_ATTRS)

    def _eval_mask(mask_numeric: bool):
        y_all, p_all, s_all, d_all, c_all = [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                x = batch.x.clone()
                if mask_numeric:
                    x[:, num_start:num_end] = 0.0
                batch_on = batch.to(device)
                x = x.to(device)
                logits = model(x, batch_on.edge_index, batch_on.edge_type)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= threshold).astype(np.int64)
                y_all.append(batch.y.cpu().numpy())
                p_all.append(preds); s_all.append(probs)
                d_all.append(batch.decoy_mask.cpu().numpy())
                for cats in batch.categories:
                    c_all.extend(cats)
        return evaluate_predictions(
            np.concatenate(y_all), np.concatenate(p_all),
            np.concatenate(d_all), categories=c_all,
            y_score=np.concatenate(s_all),
        )

    baseline_res = _eval_mask(False)
    ablated_res = _eval_mask(True)

    drop_f1 = baseline_res.f1 - ablated_res.f1
    drop_recall = baseline_res.recall - ablated_res.recall
    drop_auc = baseline_res.auc_roc - ablated_res.auc_roc

    verdict = (
        "🚨 KRİTİK FEATURE LEAK"
        if drop_f1 > 0.4 or drop_auc > 0.3
        else "⚠️ Olası leak (orta)"
        if drop_f1 > 0.2 or drop_auc > 0.15
        else "✅ Model sayısal feature'lara aşırı bağımlı değil"
    )

    rep = {
        "test": "feature_ablation",
        "masked_features": list(_NUMERIC_ATTRS),
        "baseline": baseline_res.to_dict(),
        "ablated": ablated_res.to_dict(),
        "delta": {
            "f1": round(drop_f1, 3),
            "recall": round(drop_recall, 3),
            "auc_roc": round(drop_auc, 3),
        },
        "verdict": verdict,
    }
    (out / "ablation.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rep


# ---------------------------------------------------------------------------
# 2) Baseline audit — oracle vs model
# ---------------------------------------------------------------------------
def run_baseline_audit(cfg, ckpt, out: Path, threshold: float, device) -> dict:
    """Baseline IFC'lerde oracle ihlallerini modelin yakalayıp yakalamadığı."""
    reader = DatasetReader(cfg.dataset_root)
    baseline_entries = [
        e for e in reader.list_baselines()
        if e.graph_path and e.ifc_path
    ]
    if not baseline_entries:
        return {"skipped": "baseline IFC bulunamadı"}

    ds_dummy = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=False,
    )
    model = _load(cfg, ckpt, ds_dummy.feature_dim, ds_dummy.num_edge_types, device)

    total_oracle = 0
    caught_by_model = 0
    model_flags_no_oracle = 0
    per_baseline: list[dict] = []
    missed_examples: list[dict] = []

    for ent in baseline_entries:
        try:
            sample = load_sample(
                graph_path=ent.graph_path,
                labels_path=ent.labels_path,
                ifc_id=ent.id,
            )
        except Exception as ex:
            per_baseline.append({"name": ent.name, "error": str(ex)})
            continue

        oracle = oracle_violations(sample)
        data = sample_to_data(sample).to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(bool)

        # Node sıralaması sample_to_data'da deterministtir; node_ids'i geri çek
        node_ids = list(sample.graph.nodes())
        n_oracle = len(oracle)
        n_caught = 0
        n_model_flags = int(preds.sum())
        for i, gid in enumerate(node_ids):
            if gid in oracle:
                if preds[i]:
                    n_caught += 1
                else:
                    missed_examples.append({
                        "ifc": ent.name,
                        "guid": gid,
                        "expected_category": oracle[gid],
                        "model_score": float(probs[i]),
                    })

        total_oracle += n_oracle
        caught_by_model += n_caught
        flags_outside_oracle = sum(
            1 for i, gid in enumerate(node_ids)
            if preds[i] and gid not in oracle
        )
        model_flags_no_oracle += flags_outside_oracle

        per_baseline.append({
            "name": ent.name,
            "nodes": int(len(node_ids)),
            "oracle_violations": n_oracle,
            "model_flagged_total": n_model_flags,
            "caught_by_model": n_caught,
            "missed_by_model": n_oracle - n_caught,
            "model_flags_outside_oracle": flags_outside_oracle,
            "oracle_recall": round(n_caught / n_oracle, 3) if n_oracle else None,
        })

    reader.close()
    overall_recall = caught_by_model / total_oracle if total_oracle else None

    verdict = (
        "🚨 Closed-world supervision sorunu kanıtlandı — model baseline'lardaki "
        "gerçek ihlalleri görmüyor"
        if overall_recall is not None and overall_recall < 0.3
        else "⚠️ Kısmi sorun (recall < 0.6)"
        if overall_recall is not None and overall_recall < 0.6
        else "✅ Model baseline'daki etiketsiz ihlalleri de yakalıyor (umut verici)"
        if overall_recall is not None
        else "ℹ️ Oracle hiçbir baseline'da ihlal bulamadı (kurallar değişebilir)"
    )

    rep = {
        "test": "baseline_audit",
        "n_baselines": len(baseline_entries),
        "n_oracle_violations": total_oracle,
        "n_caught_by_model": caught_by_model,
        "n_model_flags_outside_oracle": model_flags_no_oracle,
        "oracle_recall": round(overall_recall, 3) if overall_recall is not None else None,
        "verdict": verdict,
        "per_baseline": per_baseline,
    }
    (out / "baseline_audit.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if missed_examples:
        with (out / "baseline_audit_missed.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(missed_examples[0].keys()))
            w.writeheader()
            missed_examples.sort(key=lambda r: r["model_score"])
            w.writerows(missed_examples[:200])

    return rep


# ---------------------------------------------------------------------------
# 3) Score distribution
# ---------------------------------------------------------------------------
def run_score_dist(cfg, ckpt, out: Path, device) -> dict:
    """4 grup için skor histogramı."""
    ds = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=True,   # baseline node'ları da skorlamak için
    )
    splits = split_by_baseline(
        ds.baseline_ids, val_frac=cfg.val_frac,
        test_frac=cfg.test_frac, seed=cfg.split_seed,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = _load(cfg, ckpt, ds.feature_dim, ds.num_edge_types, device)

    buckets = {"true_pos": [], "true_neg": [], "decoy": [], "baseline_any": []}
    with torch.no_grad():
        for batch in loader:
            batch_on = batch.to(device)
            logits = model(batch_on.x, batch_on.edge_index, batch_on.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            y = batch.y.cpu().numpy().astype(bool)
            d = batch.decoy_mask.cpu().numpy().astype(bool)
            # baseline mi? (ifc_id'sinden kind=baseline mi anlamak için
            # dataset'in baseline_ids listesi var)
            for i in range(len(y)):
                if d[i]:
                    buckets["decoy"].append(float(probs[i]))
                elif y[i]:
                    buckets["true_pos"].append(float(probs[i]))
                else:
                    buckets["true_neg"].append(float(probs[i]))

    def _hist(scores: list[float]) -> list[int]:
        bins = [0] * 10
        for s in scores:
            idx = min(int(s * 10), 9)
            bins[idx] += 1
        return bins

    def _stats(scores: list[float]) -> dict:
        if not scores:
            return {"n": 0}
        arr = np.array(scores)
        return {
            "n": len(scores),
            "mean": round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "above_0.5": int((arr >= 0.5).sum()),
        }

    rep = {
        "test": "score_distribution",
        "histograms_0_to_1_in_10_bins": {k: _hist(v) for k, v in buckets.items()},
        "stats": {k: _stats(v) for k, v in buckets.items()},
    }

    # Karar: pozitif ile negatif dağılımının orta noktaları çakışıyor mu?
    pos_mean = rep["stats"]["true_pos"].get("mean", 0)
    neg_mean = rep["stats"]["true_neg"].get("mean", 0)
    separation = pos_mean - neg_mean
    rep["pos_neg_separation"] = round(separation, 3)
    rep["verdict"] = (
        "✅ Pozitif ile negatif skor dağılımı belirgin ayrılmış"
        if separation > 0.4
        else "⚠️ Orta seviye ayrım (model emin değil)"
        if separation > 0.2
        else "🚨 Dağılımlar üst üste — model rastgele tahmin ediyor olabilir"
    )

    (out / "score_distribution.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Modelin gerçekten ne öğrendiğini test et.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--mode",
        choices=["all", "ablation", "baseline_audit", "score_dist"],
        default="all",
    )
    a = p.parse_args()

    cfg_raw = json.loads(Path(a.config).read_text())
    cfg_raw["dataset_root"] = a.dataset_root or str(data_home())
    cfg = TrainConfig(**cfg_raw)
    threshold = a.threshold if a.threshold is not None else cfg.threshold
    device = cfg.resolve_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"checkpoint": a.checkpoint, "threshold": threshold}

    if a.mode in ("all", "ablation"):
        print("=== 1) Feature ablation ===")
        rep = run_ablation(cfg, a.checkpoint, out, threshold, device)
        summary["ablation"] = rep
        print("  ", rep.get("verdict", rep))
        if "delta" in rep:
            print(f"   ΔF1={rep['delta']['f1']:+.3f}  "
                  f"ΔR={rep['delta']['recall']:+.3f}  "
                  f"ΔAUC={rep['delta']['auc_roc']:+.3f}")

    if a.mode in ("all", "baseline_audit"):
        print("\n=== 2) Baseline audit (oracle vs model) ===")
        rep = run_baseline_audit(cfg, a.checkpoint, out, threshold, device)
        summary["baseline_audit"] = rep
        print("  ", rep.get("verdict", rep))
        if rep.get("oracle_recall") is not None:
            print(f"   Oracle ihlal: {rep['n_oracle_violations']}  "
                  f"Model yakaladı: {rep['n_caught_by_model']}  "
                  f"(recall={rep['oracle_recall']})")

    if a.mode in ("all", "score_dist"):
        print("\n=== 3) Score distribution ===")
        rep = run_score_dist(cfg, a.checkpoint, out, device)
        summary["score_dist"] = rep
        print("  ", rep.get("verdict", rep))
        for k in ("true_pos", "true_neg", "decoy"):
            s = rep["stats"].get(k, {})
            if s.get("n"):
                print(f"   {k:>10s}: n={s['n']}  mean={s.get('mean')}  "
                      f"≥0.5: {s.get('above_0.5')}")

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n✅ Tüm raporlar: {out}")


if __name__ == "__main__":
    main()
