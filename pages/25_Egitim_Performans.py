"""Sayfa 25 — Eğitim Performans.

Tüm geçmiş GAT eğitim run'larını tek yerde görüntüle, karşılaştır,
detay incele. Sayfa 24 (eğitim) sadece son birkaç run'ı gösterir;
buraya gelip kapsamlı analiz yaparsın.

İçerik:
  1. Filtreler (dataset, model tipi, tarih aralığı)
  2. Özet (toplam run, en iyi/ortalama F1)
  3. Tüm run'lar tablosu (zengin sütunlar)
  4. Karşılaştırma (≥2 run seç → yan yana metrikler)
  5. Detay paneli (1 run seç → summary + history + maskeler + sil)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


st.set_page_config(page_title="Eğitim Performans",
                   layout="wide", page_icon="📈")
st.title("📈 Eğitim Performans")
st.caption(
    "Tüm geçmiş GAT eğitim run'ları — filtrele, karşılaştır, detay incele. "
    "Yeni eğitim için **🧠 Sayfa 24 — GAT Eğitim**."
)


# --- Run'ları yükle ------------------------------------------------------
@st.cache_data(ttl=5, show_spinner=False)
def _load_runs() -> list[dict]:
    rr = Path("runs")
    if not rr.exists():
        return []
    out = []
    for rd in sorted(rr.iterdir(), reverse=True):
        if not rd.is_dir():
            continue
        m, s, c = {}, {}, {}
        for fname, holder in (("meta.json", m), ("summary.json", s),
                              ("config.json", c)):
            p = rd / fname
            if p.exists():
                try:
                    holder.update(json.loads(p.read_text(encoding="utf-8")))
                except Exception:
                    pass
        # Birleşik flat dict
        mm = m.get("metrics") or {}
        tr = mm.get("train") or {}
        va = mm.get("val") or {}
        te = mm.get("test") or (m.get("test_metrics") or {})
        out.append({
            "run": rd.name,
            "run_dir": str(rd),
            "dataset_slug": (m.get("dataset") or {}).get("slug"),
            "dataset_name": (m.get("dataset") or {}).get("name") or "—",
            "model": (m.get("model") or {}).get("type") or c.get("model")
                     or "?",
            "hidden_dim": (m.get("model") or {}).get("hidden_dim")
                           or c.get("hidden_dim"),
            "heads": (m.get("model") or {}).get("heads") or c.get("heads"),
            "dropout": (m.get("model") or {}).get("dropout")
                        or c.get("dropout"),
            "lr": (m.get("optim") or {}).get("lr") or c.get("lr"),
            "epochs": (m.get("optim") or {}).get("epochs")
                       or c.get("epochs"),
            "best_epoch": (m.get("best") or {}).get("epoch")
                           or s.get("best_epoch"),
            "train_f1": tr.get("f1"),
            "val_f1": va.get("f1"),
            "test_f1": te.get("f1"),
            "test_precision": te.get("precision"),
            "test_recall": te.get("recall"),
            "test_auc": te.get("auc_roc"),
            "test_mcc": te.get("mcc"),
            "test_decoy_fpr": te.get("decoy_fpr"),
            "duration_s": m.get("duration_s"),
            "created_at": (m.get("created_at") or "")[:19],
            "masks": m.get("masks") or {},
            "split": m.get("split") or {},
            "_meta": m,
            "_summary": s,
            "_config": c,
        })
    return out


runs = _load_runs()
if not runs:
    st.info(
        "Henüz hiç eğitim çalıştırılmamış. **🧠 Sayfa 24 — GAT Eğitim** "
        "ile başla."
    )
    st.stop()


# --- 1. Filtreler --------------------------------------------------------
st.subheader("Filtreler")
fc1, fc2, fc3 = st.columns(3)
with fc1:
    _ds = sorted({r["dataset_name"] for r in runs if r["dataset_name"]})
    sel_ds = st.multiselect("📚 Dataset", options=_ds, default=_ds,
                             help="Hangi dataset'le eğitilen run'lar.")
with fc2:
    _models = sorted({r["model"] for r in runs if r["model"]})
    sel_models = st.multiselect("🧠 Model tipi", options=_models,
                                  default=_models)
with fc3:
    only_complete = st.checkbox(
        "Sadece tamamlanan (test F1 var)",
        value=False,
        help="Eski/iptal edilmiş run'ları filtreler.",
    )

filt = [
    r for r in runs
    if r["dataset_name"] in set(sel_ds)
    and r["model"] in set(sel_models)
    and (not only_complete or r.get("test_f1") is not None)
]


# --- 2. Özet -------------------------------------------------------------
st.subheader(f"Özet ({len(filt)} run)")
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Toplam run", len(filt))
_test_f1s = [r["test_f1"] for r in filt if r.get("test_f1") is not None]
if _test_f1s:
    sc2.metric("En iyi Test F1", f"{max(_test_f1s):.3f}")
    sc3.metric("Ortalama Test F1", f"{sum(_test_f1s) / len(_test_f1s):.3f}")
sc4.metric("Toplam süre",
            f"{sum((r['duration_s'] or 0) for r in filt):.0f}s")


# --- 3. Tüm run'lar tablosu ----------------------------------------------
st.subheader("📋 Run listesi")
table_rows = []
for r in filt:
    masks = r.get("masks") or {}
    mtxt = ",".join(k for k, v in masks.items() if v) or "—"
    table_rows.append({
        "Run": r["run"],
        "Dataset": r["dataset_name"],
        "Model": r["model"],
        "Train F1": r["train_f1"],
        "Val F1": r["val_f1"],
        "Test F1": r["test_f1"],
        "Test AUC": r["test_auc"],
        "Test P": r["test_precision"],
        "Test R": r["test_recall"],
        "MCC": r["test_mcc"],
        "Best ep": r["best_epoch"],
        "Maskeler": mtxt,
        "LR": r["lr"],
        "Dropout": r["dropout"],
        "Hidden": r["hidden_dim"],
        "Süre s": r["duration_s"],
        "Oluşturma": r["created_at"],
    })
if not table_rows:
    st.caption("Filtreyle eşleşen run yok.")
    st.stop()
df = pd.DataFrame(table_rows)
st.dataframe(df, hide_index=True, use_container_width=True)


# --- 4. Karşılaştırma ----------------------------------------------------
st.subheader("📊 Karşılaştırma (≥2 run seç)")
sel_runs_names = st.multiselect(
    "Karşılaştırılacak run'lar",
    options=[r["run"] for r in filt],
    default=[],
)
if len(sel_runs_names) >= 2:
    sel_runs_data = [r for r in filt if r["run"] in set(sel_runs_names)]
    cmp_rows = []
    for split_name in ("Train", "Val", "Test"):
        row = {"Split": split_name}
        for r in sel_runs_data:
            key = {"Train": "train_f1", "Val": "val_f1",
                   "Test": "test_f1"}[split_name]
            row[r["run"]] = r.get(key)
        cmp_rows.append(row)
    # Aux: test AUC, precision, recall ek satırlar
    for label, key in (("Test AUC", "test_auc"),
                       ("Test Precision", "test_precision"),
                       ("Test Recall", "test_recall"),
                       ("Test MCC", "test_mcc")):
        row = {"Split": label}
        for r in sel_runs_data:
            row[r["run"]] = r.get(key)
        cmp_rows.append(row)
    st.dataframe(pd.DataFrame(cmp_rows), hide_index=True,
                 use_container_width=True)

    # Konfig farkı
    st.markdown("**Konfigürasyon farkları**")
    cfg_rows = []
    for label, key in (("Model", "model"), ("Dataset", "dataset_name"),
                       ("LR", "lr"), ("Dropout", "dropout"),
                       ("Hidden", "hidden_dim"), ("Heads", "heads"),
                       ("Best ep", "best_epoch"),
                       ("Maskeler", "masks")):
        row = {"Param": label}
        for r in sel_runs_data:
            v = r.get(key)
            if key == "masks":
                v = ",".join(k for k, b in (v or {}).items() if b) or "—"
            row[r["run"]] = v
        cfg_rows.append(row)
    st.dataframe(pd.DataFrame(cfg_rows), hide_index=True,
                 use_container_width=True)
else:
    st.caption("En az 2 run seç → side-by-side metrik + konfig tablosu.")


# --- 5. Detay paneli -----------------------------------------------------
st.subheader("🔍 Run detayı")
sel_one = st.selectbox(
    "Run seç",
    options=[r["run"] for r in filt],
    key="perf_detail_run",
)
detail = next((r for r in filt if r["run"] == sel_one), None)
if detail:
    meta = detail["_meta"]
    summary = detail["_summary"]

    dc1, dc2 = st.columns([3, 1])
    with dc1:
        st.markdown(f"### `{detail['run']}`")
        st.caption(
            f"Dataset: **{detail['dataset_name']}** · "
            f"Model: **{detail['model']}** · "
            f"Oluşturma: {detail['created_at']} · "
            f"Süre: {detail['duration_s'] or '—'}s"
        )

    with dc2:
        _confirm = st.checkbox("Sil onayı", key=f"del_confirm_{sel_one}")
        if st.button("🗑️ Run klasörünü sil", type="secondary",
                     disabled=not _confirm, key=f"del_btn_{sel_one}",
                     use_container_width=True):
            import shutil
            try:
                shutil.rmtree(detail["run_dir"])
                st.success(f"`{sel_one}` silindi.")
                _load_runs.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Silinemedi: {e}")

    # Train/Val/Test metrik tablosu (sayfa 24 ile aynı formatta)
    st.markdown("**📊 Train / Val / Test metrikleri**")
    metric_keys = [
        ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. Acc"),
        ("auc_roc", "AUC ROC"), ("auc_pr", "AUC PR"),
        ("mcc", "MCC"), ("decoy_fpr", "Decoy FPR"),
    ]
    rows = []
    metrics = meta.get("metrics") or {}
    for sp in ("train", "val", "test"):
        d = metrics.get(sp)
        if not d:
            continue
        rows.append({
            "Split": sp.capitalize(),
            **{label: (round(d[k], 4) if k in d else None)
               for k, label in metric_keys},
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

    # Konfig + maskeler + split bilgileri
    info_cols = st.columns(3)
    with info_cols[0]:
        st.markdown("**🎛️ Model + Optim**")
        st.json({"model": meta.get("model"),
                 "optim": meta.get("optim")}, expanded=False)
    with info_cols[1]:
        st.markdown("**🎭 Maskeler**")
        st.json(meta.get("masks") or {}, expanded=True)
    with info_cols[2]:
        st.markdown("**🔀 Split**")
        st.json(meta.get("split") or {}, expanded=True)

    # History — her metrik ayrı küçük chart (eğri net görünsün)
    history = summary.get("history") or []
    if history:
        st.markdown("**📈 Eğitim eğrileri (epoch bazında)**")
        hdf = pd.DataFrame(history)
        if "epoch" in hdf.columns:
            hdf = hdf.set_index("epoch")
        _grid = [
            ("Loss (train)", ["loss"]),
            ("F1 (val)", ["f1"]),
            ("Accuracy (val)", ["accuracy"]),
            ("Balanced Accuracy (val)", ["balanced_accuracy"]),
            ("Precision (val)", ["precision"]),
            ("Recall (val)", ["recall"]),
            ("AUC ROC (val)", ["auc_roc"]),
            ("AUC PR (val)", ["auc_pr"]),
            ("MCC (val)", ["mcc"]),
            ("Decoy FPR (val)", ["decoy_fpr"]),
        ]
        for i in range(0, len(_grid), 2):
            cols = st.columns(2)
            for j, (title, ck) in enumerate(_grid[i:i + 2]):
                if not all(c in hdf.columns for c in ck):
                    cols[j].caption(f"{title}  _(history'de yok — eski run)_")
                    continue
                cols[j].caption(title)
                cols[j].line_chart(hdf[ck], height=160)

    # Dataset paket detayı
    _ds = meta.get("dataset") or {}
    if _ds.get("tags"):
        with st.expander(f"📚 Dataset paketleri ({len(_ds['tags'])})"):
            for tag in _ds["tags"]:
                st.markdown(f"- `{tag}`")
            st.caption(
                f"slug: `{_ds.get('slug', '—')}` · "
                f"n_ifc_ids: {_ds.get('n_ifc_ids', '—')}"
            )

    # Run klasörü dosyaları
    with st.expander(f"📁 Run klasörü: `{detail['run_dir']}`"):
        rd = Path(detail["run_dir"])
        for f in sorted(rd.iterdir()):
            if f.is_file():
                _sz = f.stat().st_size / 1024
                st.caption(f"  • `{f.name}` ({_sz:.1f} KB)")
