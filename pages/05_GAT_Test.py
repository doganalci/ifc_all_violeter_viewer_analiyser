"""Tab: eğitilmiş modeli toplu test et.

Page 03 tek IFC inceleme için; bu sayfa **bütün bir split üzerinde**
ya da elle seçilen IFC seti üzerinde eğitilmiş modeli koşturur ve:

  * Aggregate metrikler (P/R/F1/decoy_fpr) tüm setin üstünde
  * Her IFC için ayrı satır: kaç TP / FP / FN, decoy aldanma, F1
  * Kategori bazında recall (hangi ihlal kuralında zayıf?)
  * En kötü 5 IFC için drill-down (page 03 link'i ile)

run_dir/summary.json'da kayıtlı `ifc_ids` (train/val/test'e hangi GUID'ler
düştü) okunur — yani test, modelin **eğitimde hiç görmediği** verilerin
üstünde koşar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    folder_browser, get_dataset_root, list_entries, reader_for, set_dataset_root,
    violated_children,
)
from ml.data.pyg_dataset import sample_to_data
from ml.data.graph_loader import load_sample
from ml.train.config import TrainConfig
from ml.train.metrics import evaluate_predictions


# ---- Sidebar (lighter — model picker on the page itself) --------------------
st.set_page_config(page_title="GAT Test", layout="wide", page_icon="🧪")
st.sidebar.title("IFC Graph Analysis")
current_root = get_dataset_root()
root = st.sidebar.text_input("Dataset klasörü", value=current_root)
if root != current_root:
    set_dataset_root(root)
with st.sidebar.expander("📁 Klasör seç (gez)"):
    picked = folder_browser(start=root, key="test_browser")
    if picked:
        set_dataset_root(picked)
        st.rerun()


st.title("🧪 GAT Test")
st.caption("Eğitilmiş bir modeli istediğin set üzerinde toplu olarak değerlendir.")

# ---- Pick a training run ---------------------------------------------------
run_root = Path("runs")
runs = sorted([p for p in run_root.iterdir() if p.is_dir()
               and (p / "best.pt").exists()]) if run_root.exists() else []
if not runs:
    st.info("Henüz eğitilmiş model yok. **GAT Eğitim** sayfasından başlat.")
    st.stop()

ra, rb = st.columns([3, 2])
with ra:
    run = st.selectbox("Run", options=runs, format_func=lambda p: p.name)
with rb:
    threshold = st.slider("Karar eşiği", 0.0, 1.0, 0.5, 0.05)

cfg_path = run / "config.json"
ckpt_path = run / "best.pt"
summary_path = run / "summary.json"
if not cfg_path.exists():
    st.error("config.json yok.")
    st.stop()
cfg = TrainConfig(**json.loads(cfg_path.read_text()))
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

# ---- Pick what to test on --------------------------------------------------
st.subheader("1. Hangi veride test edeceğiz?")

split_map = summary.get("ifc_ids", {})
default_set = "test"
choice_options = ["test (eğitimde görmedi)", "val", "train",
                  "paket (ana baseline) seç", "baseline seç", "elle seç"]
if not split_map.get("test"):
    default_set = "paket (ana baseline) seç"
    choice_options = ["paket (ana baseline) seç", "baseline seç", "elle seç"]
choice = st.radio("Set", options=choice_options, horizontal=True,
                  index=choice_options.index({
                      "test": "test (eğitimde görmedi)",
                      "val": "val",
                      "train": "train",
                  }.get(default_set, default_set)) if default_set in (
                      "test", "val", "train") else 0)


def _pkg_from_name(nm: str | None) -> str:
    import re as _re
    nm = nm or ""
    nm = _re.sub(r"_violated\d+.*$", "", nm)              # _violatedN son ekini at
    nm = _re.sub(r"^(basicinj|tametiket|llminj)_", "", nm)  # yöntem ön ekini at
    nm = _re.sub(r"_\d+$", "", nm)                        # _00044 sayısını at
    return nm


if choice == "test (eğitimde görmedi)":
    selected_ids: list[str] = list(split_map.get("test") or [])
elif choice == "val":
    selected_ids = list(split_map.get("val") or [])
elif choice == "train":
    selected_ids = list(split_map.get("train") or [])
elif choice == "paket (ana baseline) seç":
    # Bir PAKET (ana baseline = dataset_tag) seç → o paketteki TÜM ihlaller.
    # Eğitimde görülmemiş YENİ bir paketle gerçek genelleme testi için ideal.
    all_violated = [e for e in list_entries(root, kind="violated") if e["graph_ok"]]
    all_baselines = list_entries(root, kind="baseline", require_graph=False)
    # Ana baseline (parent) → paket adı. Violated kendi dataset_tag'ı boşsa
    # paketi PARENT baseline'dan çözüyoruz (Tam Etiketleme'de violated tag'siz
    # kalabiliyor → '(etiketsiz)' görünmesini bu engelliyor).
    _base_pkg = {b["id"]: (b.get("dataset_tag")
                           or _pkg_from_name(b.get("name")) or "(isimsiz)")
                 for b in all_baselines}

    def _pkg_of(e: dict) -> str:
        if e.get("dataset_tag"):
            return e["dataset_tag"]
        pid = e.get("parent_id")
        if pid and pid in _base_pkg:
            return _base_pkg[pid]
        return _pkg_from_name(e.get("name")) or "(isimsiz)"

    pkg_map: dict[str, list[dict]] = {}
    for e in all_violated:
        pkg_map.setdefault(_pkg_of(e), []).append(e)
    if not pkg_map:
        st.warning("Graph'lı ihlal içeren paket yok. Önce ihlal üret.")
        st.stop()
    pkgs = sorted(pkg_map, key=lambda p: -len(pkg_map[p]))
    # Eğitimde kullanılan paketi işaretle (train+val split'inden türet).
    _train_ids = set(split_map.get("train") or []) | set(split_map.get("val") or [])
    _has_split = bool(_train_ids)

    def _badge(p: str) -> str:
        if not _has_split:
            return ""   # bu run split bilgisi tutmuyor → emin değiliz, iddia etme
        return ("  ⚠️ eğitimde kullanıldı"
                if any(e["id"] in _train_ids for e in pkg_map[p])
                else "  ✅ yeni (görülmedi)")

    psel = st.selectbox(
        "📦 Paket (ana baseline)",
        options=pkgs,
        format_func=lambda p: f"{p} · {len(pkg_map[p])} ihlal{_badge(p)}",
    )
    selected_ids = [e["id"] for e in pkg_map.get(psel, [])]
    _seen = _has_split and any(e["id"] in _train_ids for e in pkg_map.get(psel, []))
    if not _has_split:
        st.info("Bu run, hangi IFC'lerin eğitimde kullanıldığını kaydetmemiş; "
                "'yeni mi' bilgisini gösteremiyorum. Modelin görmediğinden emin "
                "olmak için tamamen yeni isimli bir paket üret ve onu seç.")
    elif _seen:
        st.warning(
            "Bu paketin bir kısmı **eğitimde kullanıldı** — gerçek genelleme "
            "ölçümü için modelin hiç görmediği YENİ bir paket üret ve onu seç.")
    else:
        st.success(f"`{psel}` paketi modelin **hiç görmediği** veri → "
                   f"gerçek genelleme testi ({len(selected_ids)} ihlal).")
elif choice == "baseline seç":
    # Bir baseline seç → ondan üretilen TÜM ihlalleri test et
    baselines = [e for e in list_entries(root, kind="baseline", require_graph=False)]
    bvio = {b["id"]: [k for k in violated_children(root, b["id"]) if k["graph_ok"]]
            for b in baselines}
    baselines.sort(key=lambda b: (-len(bvio.get(b["id"], [])), b["name"]))
    baselines = [b for b in baselines if bvio.get(b["id"])]
    if not baselines:
        st.warning("İhlali olan (graph'lı) baseline yok. Önce ihlal üret.")
        st.stop()
    bsel = st.selectbox(
        "Baseline seç",
        options=[b["id"] for b in baselines],
        format_func=lambda i: next(
            f"{b['name']} · {i[:8]} · {len(bvio[i])} ihlal"
            for b in baselines if b["id"] == i),
    )
    selected_ids = [k["id"] for k in bvio.get(bsel, [])]
    st.caption(f"`{next(b['name'] for b in baselines if b['id'] == bsel)}` "
               f"baseline'ından {len(selected_ids)} ihlal test edilecek.")
else:
    all_violated = [e for e in list_entries(root, kind="violated") if e["graph_ok"]]
    selected = st.multiselect(
        "IFC seç",
        options=[e["id"] for e in all_violated],
        format_func=lambda i: next(e["name"] + " · " + i[:8]
                                   for e in all_violated if e["id"] == i),
    )
    selected_ids = selected

if not selected_ids:
    st.warning("Test edilecek IFC seçili değil.")
    st.stop()

st.caption(f"{len(selected_ids)} IFC test edilecek.")

# ---- Resolve entries → file paths ------------------------------------------
all_entries = {e["id"]: e for e in list_entries(root, kind=None) if e["graph_ok"]}
test_entries = [all_entries[i] for i in selected_ids if i in all_entries]
missing = [i for i in selected_ids if i not in all_entries]
if missing:
    st.warning(
        f"{len(missing)} IFC bu dataset klasöründe bulunamadı (yeni klasör mü açtın?). "
        f"İlk birkaçı: {[m[:8] for m in missing[:3]]}"
    )

if not test_entries:
    st.error("Eşleşen IFC kalmadı.")
    st.stop()

# ---- Run inference ---------------------------------------------------------
go = st.button("🧪 Testi çalıştır", type="primary")
if not go:
    st.stop()

try:
    import torch
except ImportError:
    st.error("PyTorch yüklü değil.")
    st.stop()


def _build_model(cfg, in_dim, num_edge_types):
    from ml.model.gat import GATNodeClassifier
    from ml.model.hetero_gat import HeteroGATNodeClassifier
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


device = cfg.resolve_device()
progress = st.progress(0.0, text="Modeli yüklüyorum...")

# Build one Data to infer feature_dim / num_edge_types.
first_sample = load_sample(test_entries[0]["graph_path"],
                           test_entries[0].get("labels_path"),
                           ifc_id=test_entries[0]["id"])
first_data = sample_to_data(first_sample)
num_edge_types = int(first_data.edge_type.max().item()) + 1 if first_data.edge_type.numel() else 7

model = _build_model(cfg, first_data.x.shape[1], max(num_edge_types, 7)).to(device)
model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
model.eval()

rows: list[dict] = []
per_ifc_payload: list[dict] = []
y_all, p_all, d_all = [], [], []
cat_all: list = []

with torch.no_grad():
    for k, ent in enumerate(test_entries):
        sample = load_sample(ent["graph_path"], ent.get("labels_path"),
                             ifc_id=ent["id"])
        data = sample_to_data(sample).to(device)
        logits = model(data.x, data.edge_index, data.edge_type)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(np.int64)
        y = data.y.cpu().numpy()
        decoy = data.decoy_mask.cpu().numpy()
        res = evaluate_predictions(
            y, preds, decoy,
            categories=data.categories,
            y_score=probs,
        )
        rows.append({
            "ifc_id": ent["id"][:8],
            "name": ent["name"],
            "kind": ent["kind"],
            "nodes": int(len(y)),
            "positives": int(res.n_positive),
            "decoys": int(res.n_decoys),
            "TP": int((preds & y).sum()),
            "FP": int(res.n_predicted_positive - (preds & y).sum()),
            "FN": int(res.n_positive - (preds & y).sum()),
            "precision": round(res.precision, 3),
            "recall": round(res.recall, 3),
            "f1": round(res.f1, 3),
            "decoy_fpr": round(res.decoy_fpr, 3),
        })
        per_ifc_payload.append({
            "ifc_id": ent["id"],
            "name": ent["name"],
            "metrics": res.to_dict(),
        })
        y_all.append(y)
        p_all.append(preds)
        d_all.append(decoy)
        cat_all.extend(data.categories)
        progress.progress((k + 1) / len(test_entries),
                          text=f"{k + 1}/{len(test_entries)}  ·  {ent['name']}")

progress.empty()

# ---- Aggregate -------------------------------------------------------------
s_all_concat = []
for r in rows:
    pass  # skor toplamı aşağıda her IFC için ayrıca tutuluyor
# y_score aggregate için per-IFC döngüsünde toplayalım
# (zaten preds threshold ile alındı; AUC için raw probs lazım)
# Bu yüzden toplu evaluate'te y_score=None — AUC olmaz.
# Bunun yerine her IFC'de ayrı ayrı AUC hesaplandı (res.auc_roc satır altında).

agg = evaluate_predictions(
    np.concatenate(y_all),
    np.concatenate(p_all),
    np.concatenate(d_all),
    categories=cat_all,
)

st.subheader("Toplu sonuç")
c1, c2, c3, c4 = st.columns(4)
c1.metric("F1", f"{agg.f1:.3f}")
c2.metric("Precision", f"{agg.precision:.3f}")
c3.metric("Recall", f"{agg.recall:.3f}")
c4.metric("Decoy FPR", f"{agg.decoy_fpr:.3f}",
          help="Decoy node'ların kaçını yanlışlıkla ihlal saydı.")
c5, c6, c7, c8 = st.columns(4)
c5.metric("Balanced Acc", f"{agg.balanced_accuracy:.3f}",
          help="(TPR + TNR) / 2 — sınıf dengesizliğine sağlam.")
c6.metric("MCC", f"{agg.mcc:+.3f}",
          help="Matthews correlation. -1..+1. 0 = rastgele.")
c7.metric("Accuracy", f"{agg.accuracy:.3f}")
c8.metric("Pozitif oran",
          f"{(agg.n_positive / max(agg.n_positive + agg.tn + agg.fp, 1)):.1%}",
          help="Test setindeki gerçek pozitif oranı (baseline).")
st.caption(
    f"Pozitif örnek: {agg.n_positive} · Tahmin pozitif: {agg.n_predicted_positive} "
    f"· Decoy: {agg.n_decoys}"
)

# ---- Confusion Matrix -----------------------------------------------------
st.subheader("Confusion Matrix")
cm_df = pd.DataFrame(
    [[agg.tn, agg.fp], [agg.fn, agg.tp]],
    index=["Gerçek: değil", "Gerçek: ihlal"],
    columns=["Tahmin: değil", "Tahmin: ihlal"],
)
cm_col1, cm_col2 = st.columns([1, 2])
with cm_col1:
    st.dataframe(cm_df, use_container_width=True)
with cm_col2:
    st.caption(
        f"**TP={agg.tp}**: doğru bilinen ihlaller  · "
        f"**FN={agg.fn}**: kaçırılan gerçek ihlaller (recall'u düşürür)  · "
        f"**FP={agg.fp}**: yanlış alarm (precision'ı düşürür)  · "
        f"**TN={agg.tn}**: doğru reddedilen normaller"
    )

if agg.per_category_recall:
    st.subheader("Kategori Bazında Performans")
    cat_rows = []
    for c in sorted(set(agg.per_category_recall) | set(agg.per_category_precision)):
        sup = int(agg.per_category_support.get(c, 0))
        rec = agg.per_category_recall.get(c, 0.0)
        tp_c = round(rec * sup)
        fn_c = sup - tp_c
        cat_rows.append({
            "kategori": c,
            "precision": agg.per_category_precision.get(c, 0.0),
            "recall": rec,
            "f1": agg.per_category_f1.get(c, 0.0),
            "TP": tp_c,
            "FN": fn_c,
            "support": sup,
        })
    cat_df = pd.DataFrame(cat_rows).sort_values("f1")
    st.dataframe(cat_df, hide_index=True, use_container_width=True)

    st.markdown("**TP / FN dağılımı (kaç pozitif yakaladık vs kaçırdık)**")
    st.bar_chart(cat_df[["kategori", "TP", "FN"]].set_index("kategori"),
                 color=["#22c55e", "#ef4444"])

    st.markdown("**Precision / Recall / F1**")
    st.bar_chart(cat_df.set_index("kategori")[["precision", "recall", "f1"]])

    weak = cat_df[cat_df["f1"] < 0.7]
    if not weak.empty:
        with st.expander(f"⚠️ Zayıf kategoriler ({len(weak)})", expanded=True):
            for _, row in weak.iterrows():
                st.warning(
                    f"**{row['kategori']}** — F1={row['f1']:.2f} · "
                    f"{row['TP']}/{row['support']} yakalandı, {row['FN']} kaçırıldı."
                )

# ---- Per-IFC table ---------------------------------------------------------
st.subheader("Her IFC için sonuç")
df = pd.DataFrame(rows).sort_values("f1")
st.dataframe(df, hide_index=True, use_container_width=True)

with st.expander("❌ En kötü 5 IFC"):
    for r in df.head(5).to_dict("records"):
        st.markdown(
            f"• **{r['name']}** (`{r['ifc_id']}`) — "
            f"f1={r['f1']}  ·  TP/FP/FN = {r['TP']}/{r['FP']}/{r['FN']}  "
            f"·  decoy_fpr={r['decoy_fpr']}"
        )

# ---- PDF rapor + işlem günlüğü --------------------------------------------
st.subheader("📄 Test raporu (PDF)")
import datetime as _dt
_set_label = choice
_meta_lines = [
    f"Run: {run.name}",
    f"Tarih: {_dt.datetime.now():%Y-%m-%d %H:%M}",
    f"Test seti: {_set_label}",
    f"Karar eşiği: {threshold}",
    f"Test edilen IFC: {len(test_entries)}",
    f"Pozitif örnek: {agg.n_positive}   Tahmin pozitif: {agg.n_predicted_positive}",
    f"Decoy: {agg.n_decoys}",
]
_pc = {"f1": agg.per_category_f1, "precision": agg.per_category_precision,
       "recall": agg.per_category_recall, "support": agg.per_category_support}
if st.button("📄 PDF rapor üret"):
    try:
        from ml.report import build_eval_report
        from ml.tracking import log_operation
        _stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _pdf = build_eval_report(
            Path("predictions") / f"test_{run.name}_{_stamp}.pdf",
            title=f"GAT Test Raporu — {run.name}",
            meta_lines=_meta_lines, agg=agg.to_dict(),
            per_ifc_rows=df.to_dict("records"), per_category=_pc,
        )
        if _pdf:
            st.success(f"Üretildi: {_pdf}")
            with open(_pdf, "rb") as f:
                st.download_button("📥 PDF indir", f, file_name=Path(_pdf).name,
                                   mime="application/pdf")
            log_operation("gat_test", paket=str(_set_label),
                          adet=len(test_entries),
                          ozet=f"F1={agg.f1:.3f} P={agg.precision:.3f} "
                               f"R={agg.recall:.3f}",
                          parametreler=f"run={run.name} threshold={threshold}")
        else:
            st.error("Rapor üretilemedi (matplotlib kurulu mu?).")
    except Exception as e:
        st.error(f"Hata: {e}")

# ---- Export ----------------------------------------------------------------
st.subheader("Sonuçları dışa aktar")
out_dir = Path("predictions") / run.name
if st.button("📤 JSON'ları yaz (viewer'a beslemek için)"):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Re-run a quick pass to write per-IFC predictions in viewer format.
    with torch.no_grad():
        for ent in test_entries:
            sample = load_sample(ent["graph_path"], ent.get("labels_path"),
                                 ifc_id=ent["id"])
            data = sample_to_data(sample).to(device)
            logits = model(data.x, data.edge_index, data.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            payload = {
                "model_version": run.name,
                "ifc_id": ent["id"],
                "threshold": threshold,
                "predictions": [
                    {"guid": guid, "score": float(probs[j]), "predicted": int(preds[j])}
                    for j, guid in enumerate(data.node_ids)
                ],
            }
            (out_dir / f"{ent['id']}.predictions.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
    st.success(f"Yazıldı: {out_dir} ({len(test_entries)} dosya)")
