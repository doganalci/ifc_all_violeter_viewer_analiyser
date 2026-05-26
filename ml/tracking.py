"""Deney & veri kümesi defteri — data klasöründe Excel olarak.

İki dosya üretir (data_home() altında):

  * **experiments.xlsx** — her eğitim çalışması bir satır: kullanılan paket(ler)
    (ana baseline), model hiperparametreleri, veri büyüklüğü, train/val/test
    dengesi, ve train/val/test metrikleri (F1/P/R/MCC/AUC + confusion).
  * **datasets.xlsx** — iki sayfa:
        - 'paketler': her dataset paketi (ana baseline) için baseline/violated
          sayıları.
        - 'ornekler': her IFC örneği — hangi baseline'dan ve hangi ana
          baseline'dan (paket) üretildiği dahil tam soy ağacı.

Excel yazımı openpyxl gerektirir; yoksa otomatik CSV'ye düşer (aynı isim,
.csv uzantısı) ki hiçbir bilgi kaybolmasın.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from paths import data_home


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #

def _pkg_from_name(nm: str | None) -> str:
    """Dosya/model adından paket (ana baseline) adını türet."""
    nm = nm or ""
    nm = re.sub(r"_violated\d+.*$", "", nm)
    nm = re.sub(r"^(basicinj|tametiket|llminj)_", "", nm)
    nm = re.sub(r"_\d+$", "", nm)
    return nm


def _write_table(rows: list[dict], path_xlsx: Path, *, sheet: str = "Sheet1",
                 append: bool = False) -> Path:
    """rows'u Excel'e yaz. openpyxl yoksa CSV'ye düş. Yazılan yolu döndür."""
    import pandas as pd
    df_new = pd.DataFrame(rows)
    try:
        if append and path_xlsx.exists():
            old = pd.read_excel(path_xlsx, sheet_name=sheet)
            df = pd.concat([old, df_new], ignore_index=True)
        else:
            df = df_new
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name=sheet, index=False)
        return path_xlsx
    except Exception:
        # openpyxl yok / Excel kilitli → CSV'ye düş
        csv_path = path_xlsx.with_suffix(".csv")
        if append and csv_path.exists():
            old = pd.read_csv(csv_path)
            df = pd.concat([old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(csv_path, index=False)
        return csv_path


def _ifc_id_to_pkg(dataset_root: str) -> tuple[dict[str, str], dict[str, dict]]:
    """ifc_id → paket adı, ve ifc_id → entry dict döndür.

    Paket: kaydın kendi dataset_tag'ı; boşsa parent baseline'ın paketi;
    o da yoksa addan türetilir.
    """
    from ml.data.sqlite_reader import DatasetReader
    entries: dict[str, Any] = {}
    with DatasetReader(dataset_root) as rd:
        for status in ("ok", "partial"):
            for e in rd.list_models(status=status):
                entries[e.id] = e
    base_pkg: dict[str, str] = {}
    for e in entries.values():
        if e.kind == "baseline":
            base_pkg[e.id] = e.dataset_tag or _pkg_from_name(e.name) or "(isimsiz)"

    def pkg_of(e) -> str:
        if e.dataset_tag:
            return e.dataset_tag
        if e.parent_id and e.parent_id in base_pkg:
            return base_pkg[e.parent_id]
        return _pkg_from_name(e.name) or "(isimsiz)"

    id_pkg = {eid: pkg_of(e) for eid, e in entries.items()}
    id_entry = {eid: e for eid, e in entries.items()}
    return id_pkg, id_entry


# --------------------------------------------------------------------------- #
# 1) Eğitim defteri
# --------------------------------------------------------------------------- #

def record_run(run_dir: str | Path, *, home: Path | None = None) -> Path | None:
    """Bir eğitim çalışmasını experiments.xlsx'e bir satır olarak ekle.

    run_dir altında config.json + summary.json bekler. Hata olursa None
    döndürür (eğitim akışını asla bozmaz).
    """
    try:
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "config.json").read_text())
        summ = json.loads((run_dir / "summary.json").read_text())
    except Exception:
        return None

    home = home or data_home()
    ids = summ.get("ifc_ids", {}) or {}
    dataset_root = cfg.get("dataset_root") or str(home)

    # Kullanılan paketler (ana baseline'lar)
    pkgs_used: list[str] = []
    try:
        id_pkg, _ = _ifc_id_to_pkg(dataset_root)
        used = set()
        for split in ("train", "val", "test"):
            for i in ids.get(split, []) or []:
                if i in id_pkg:
                    used.add(id_pkg[i])
        pkgs_used = sorted(used)
    except Exception:
        pkgs_used = []

    def _m(split: str, key: str):
        d = summ.get(split) or {}
        return d.get(key)

    def _cm(split: str, key: str):
        d = (summ.get(split) or {}).get("confusion") or {}
        return d.get(key)

    n_train = len(ids.get("train") or [])
    n_val = len(ids.get("val") or [])
    n_test = len(ids.get("test") or [])
    test_tp = _cm("test", "tp")
    test_fp = _cm("test", "fp")
    test_fn = _cm("test", "fn")
    test_tn = _cm("test", "tn")
    test_prec = _m("test", "precision")
    test_rec = _m("test", "recall")

    row = {
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run": run_dir.name,
        "paketler (ana baseline)": ", ".join(pkgs_used) if pkgs_used else "?",
        # Model hiperparametreleri
        "model": cfg.get("model"),
        "hidden_dim": cfg.get("hidden_dim"),
        "heads": cfg.get("heads"),
        "dropout": cfg.get("dropout"),
        "edge_emb_dim": cfg.get("edge_emb_dim"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "epochs(ayar)": cfg.get("epochs"),
        "best_epoch": summ.get("best_epoch"),
        "patience": cfg.get("patience"),
        "pos_weight": cfg.get("pos_weight"),
        "erken_durdu": summ.get("stopped_early"),
        "device": cfg.get("device"),
        # Maskeler / augment
        "mask_numeric": cfg.get("mask_numeric_features"),
        "mask_pset": cfg.get("mask_pset_features"),
        "mask_type": cfg.get("mask_type_features"),
        "rule_oracle": cfg.get("use_rule_oracle"),
        "include_baselines": cfg.get("include_baselines"),
        # Veri büyüklüğü / denge
        "IFC_toplam": n_train + n_val + n_test,
        "IFC_train": n_train,
        "IFC_val": n_val,
        "IFC_test": n_test,
        "val_frac": cfg.get("val_frac"),
        "test_frac": cfg.get("test_frac"),
        "split_seed": cfg.get("split_seed"),
        "threshold": cfg.get("threshold"),
        # Metrikler
        "train_F1": _m("train", "f1"),
        "val_F1": _m("val", "f1"),
        "test_F1": _m("test", "f1"),
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_MCC": _m("test", "mcc"),
        "test_AUC": _m("test", "auc_roc"),
        "test_bal_acc": _m("test", "balanced_accuracy"),
        "test_decoy_fpr": _m("test", "decoy_fpr"),
        "test_TP": test_tp,
        "test_FP": test_fp,
        "test_FN": test_fn,
        "test_TN": test_tn,
    }
    home.mkdir(parents=True, exist_ok=True)
    try:
        return _write_table([row], home / "experiments.xlsx",
                            sheet="deneyler", append=True)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 1b) İşlem günlüğü — yaptığımız her işlem bir satır
# --------------------------------------------------------------------------- #

def log_operation(operation: str, *, paket: str = "", adet=None,
                  ozet: str = "", parametreler: str = "",
                  home: Path | None = None) -> Path | None:
    """operations_log.xlsx'e bir işlem satırı ekle (üretim/test/eğitim vb.).

    Best-effort — hata olursa None döndürür, akışı bozmaz.
    """
    home = home or data_home()
    home.mkdir(parents=True, exist_ok=True)
    row = {
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "islem": operation,
        "paket (ana baseline)": paket,
        "adet": adet if adet is not None else "",
        "ozet": ozet,
        "parametreler": parametreler,
    }
    try:
        return _write_table([row], home / "operations_log.xlsx",
                            sheet="islemler", append=True)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 2) Veri kümesi defteri
# --------------------------------------------------------------------------- #

def rebuild_dataset_registry(dataset_root: str, *,
                             home: Path | None = None) -> dict[str, Path]:
    """datasets.xlsx'i DB'den baştan üret (paketler + ornekler sayfaları).

    Döndürür: {'paketler': path, 'ornekler': path} (CSV fallback'te .csv).
    """
    home = home or data_home()
    home.mkdir(parents=True, exist_ok=True)
    id_pkg, id_entry = _ifc_id_to_pkg(dataset_root)

    # Örnekler (her IFC bir satır) + soy ağacı
    sample_rows: list[dict] = []
    pkg_stats: dict[str, dict] = {}
    for eid, e in id_entry.items():
        pkg = id_pkg.get(eid, "(isimsiz)")
        parent = id_entry.get(e.parent_id) if e.parent_id else None
        sample_rows.append({
            "ana_baseline (paket)": pkg,
            "ornek": e.name,
            "tur": e.kind,                      # baseline | violated | imported
            "ornek_id": eid[:8],
            "baseline (parent)": (parent.name if parent else ""),
            "baseline_id": (e.parent_id or "")[:8],
            "status": e.status,
            "graph_var": bool(e.graph_path and e.graph_path.exists()),
            "ifc_yolu": str(e.ifc_path) if e.ifc_path else "",
        })
        s = pkg_stats.setdefault(pkg, {"baseline": 0, "violated": 0,
                                       "imported": 0})
        if e.kind in s:
            s[e.kind] += 1

    pkg_rows = [{
        "ana_baseline (paket)": pkg,
        "baseline": s["baseline"],
        "violated (ihlal)": s["violated"],
        "imported": s["imported"],
        "toplam": s["baseline"] + s["violated"] + s["imported"],
    } for pkg, s in sorted(pkg_stats.items())]

    out: dict[str, Path] = {}
    # İki sayfayı tek workbook'a yazmayı dene; olmazsa ayrı CSV'ler
    xlsx = home / "datasets.xlsx"
    try:
        import pandas as pd
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
            pd.DataFrame(pkg_rows).to_excel(xw, sheet_name="paketler", index=False)
            pd.DataFrame(sample_rows).to_excel(xw, sheet_name="ornekler", index=False)
        out["paketler"] = xlsx
        out["ornekler"] = xlsx
    except Exception:
        out["paketler"] = _write_table(pkg_rows, home / "datasets_paketler.xlsx",
                                       sheet="paketler")
        out["ornekler"] = _write_table(sample_rows, home / "datasets_ornekler.xlsx",
                                       sheet="ornekler")
    return out
