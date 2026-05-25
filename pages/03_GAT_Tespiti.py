"""Tab: GAT inference + visual review of predictions.

Workflow:
  1. Pick a trained checkpoint (anything under ./runs/).
  2. We rebuild the model with the matching config, run the selected
     IFC through it, and threshold the probabilities.
  3. The graph + IFC views colour each node with its prediction.
     Predictions are layered on top of the ground truth so you can
     visually inspect TP / FP / FN / decoy fooling.
  4. Metrics from this single IFC are displayed alongside.

If no checkpoint exists yet, the page links the user to the training
command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import load_sample_for, sidebar_config
from ml.data.pyg_dataset import sample_to_data
from ml.train.config import TrainConfig
from ml.train.metrics import evaluate_predictions
from ml.viz.graph_view import interactive_agraph, static_plotly
from ml.viz.ifc3d import build_figure, extract_meshes


@st.cache_data(show_spinner="IFC tessellate ediliyor...")
def _cached_meshes(ifc_path: str, _mtime: float):
    return extract_meshes(ifc_path)


def _mtime_safe(p: str) -> float:
    import os
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _list_runs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted([p for p in run_dir.iterdir()
                   if p.is_dir() and (p / "best.pt").exists()])


def _load_model(cfg: TrainConfig, in_dim: int, num_edge_types: int):
    # Lazy: avoid pulling torch on pages that don't need it.
    import torch
    from ml.model.gat import GATNodeClassifier
    from ml.model.hetero_gat import HeteroGATNodeClassifier

    if cfg.model == "hetero_gat":
        m = HeteroGATNodeClassifier(
            in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=num_edge_types, heads=cfg.heads, dropout=cfg.dropout,
        )
    else:
        m = GATNodeClassifier(
            in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=num_edge_types, edge_emb_dim=cfg.edge_emb_dim,
            heads=cfg.heads, dropout=cfg.dropout,
        )
    return m


st.set_page_config(page_title="GAT Tespiti", layout="wide", page_icon="🤖")
entry = sidebar_config()

st.title("🤖 GAT Tespiti")
st.caption("Eğitilmiş Graph Attention Network ile node-level ihlal tespiti.")

if entry is None:
    st.stop()

# --- Üstte: Baseline'dan üretilen ihlalleri seç (sidebar'a bağlı değil) ----
from ml.app.state import (
    list_entries as _list_entries, violated_children as _violated_children,
    get_dataset_root as _gdr,
)
_root0 = _gdr()
with st.container():
    st.markdown("### 📦 Baseline → ihlal seç (tespit için)")
    # Baseline'ı graph şartı OLMADAN listele: tespit violated child üzerinde
    # yapılır (baseline'ın kendi graph'ı gerekmez). require_graph=True iken
    # graph'sız baseline'lar gizleniyordu → ihlali olan baseline görünmüyordu.
    _bl = _list_entries(_root0, kind="baseline", require_graph=False)
    _bl_vio = {b["id"]: [k for k in _violated_children(_root0, b["id"])
                         if k.get("graph_ok")] for b in _bl}
    # En çok (tespit edilebilir) ihlali olan baseline üstte; ihlalsizleri en sonda
    _bl.sort(key=lambda b: (-len(_bl_vio.get(b["id"], [])), b["name"]))
    bc = st.columns([2, 3])
    with bc[0]:
        _bi = st.selectbox(
            "Baseline",
            options=[None] + list(range(len(_bl))),
            format_func=lambda i: (
                "— sidebar seçimini kullan —" if i is None
                else f"{_bl[i]['name']} · {_bl[i]['id'][:8]} "
                     f"· {len(_bl_vio.get(_bl[i]['id'], []))} ihlal"),
            key="det_baseline",
        )
    if _bi is not None:
        kids = _bl_vio.get(_bl[_bi]["id"], [])
        with bc[1]:
            if kids:
                _ki = st.selectbox(
                    f"İhlal ({len(kids)} adet)",
                    options=list(range(len(kids))),
                    format_func=lambda i: f"[{i+1}/{len(kids)}] {kids[i]['name']} · {kids[i]['id'][:8]}",
                    key="det_violated",
                )
                entry = kids[_ki]   # tespit bu violated üzerinde yapılır
            else:
                st.warning("Bu baseline'dan graph'lı ihlal yok.")
    st.divider()

run_root = Path("runs")
runs = _list_runs(run_root)
if not runs:
    st.info(
        "Henüz eğitilmiş model yok. Önce şunu çalıştır:\n\n"
        "```bash\n"
        "python -m scripts.train --dataset-root <codex1_path>\n"
        "```"
    )
    st.stop()

col_run, col_th = st.columns([3, 1])
with col_run:
    run = st.selectbox("Eğitim run", options=runs, format_func=lambda p: p.name)
with col_th:
    threshold = st.slider("Eşik", 0.0, 1.0, 0.5, 0.05)

cfg_path = run / "config.json"
ckpt_path = run / "best.pt"
if not cfg_path.exists():
    st.error("config.json bulunamadı — bu run düzgün kaydedilmemiş.")
    st.stop()

cfg = TrainConfig(**json.loads(cfg_path.read_text()))

# Show training summary if available.
summary_p = run / "summary.json"
if summary_p.exists():
    summary = json.loads(summary_p.read_text())
    with st.expander("📊 Eğitim özeti", expanded=False):
        st.write(f"best_val_f1: **{summary.get('best_val_f1', 0):.3f}** "
                 f"@ epoch {summary.get('best_epoch')}")
        if summary.get("test"):
            t = summary["test"]
            st.write(f"test F1={t['f1']:.3f} · "
                     f"P={t['precision']:.3f} · R={t['recall']:.3f} · "
                     f"decoy_fpr={t['decoy_fpr']:.3f}")

# Build PyG Data for the selected IFC.
sample = load_sample_for(entry)
g = sample.graph
data = sample_to_data(sample)

# Lazy torch import + inference.
try:
    import torch
except ImportError:
    st.error("PyTorch yüklü değil — `pip install torch torch_geometric`.")
    st.stop()

with st.spinner("Model çalıştırılıyor..."):
    device = cfg.resolve_device()
    model = _load_model(cfg, data.x.shape[1], int(data.edge_type.max().item()) + 1 if data.edge_type.numel() else 7)
    state = torch.load(str(ckpt_path), map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        st.error(f"Checkpoint yüklenirken hata: {e}")
        st.stop()
    model.to(device).eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device),
                       data.edge_type.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

preds = (probs >= threshold).astype(np.int64)
y_true = data.y.cpu().numpy()
decoy_mask = data.decoy_mask.cpu().numpy()
res = evaluate_predictions(
    y_true, preds, decoy_mask,
    categories=data.categories,
    y_score=probs,
)

# ---- Per-IFC metrics --------------------------------------------------------
r1 = st.columns(4)
r1[0].metric("F1", f"{res.f1:.3f}")
r1[1].metric("Precision", f"{res.precision:.3f}")
r1[2].metric("Recall", f"{res.recall:.3f}")
r1[3].metric("Accuracy", f"{res.accuracy:.3f}")
r2 = st.columns(4)
r2[0].metric("Balanced Acc", f"{res.balanced_accuracy:.3f}",
             help="(TPR + TNR) / 2 — sınıf dengesizliğine sağlam.")
r2[1].metric("MCC", f"{res.mcc:+.3f}",
             help="Matthews correlation. -1..+1.")
r2[2].metric("AUC-ROC", f"{res.auc_roc:.3f}",
             help="Eşikten bağımsız sıralama gücü.")
r2[3].metric("Decoy FPR", f"{res.decoy_fpr:.3f}",
             help="Decoy node'lardan kaçı yanlışlıkla ihlal işaretlenmiş.")

# Confusion matrix
with st.expander("📐 Confusion Matrix", expanded=False):
    import pandas as _pd
    cm_df = _pd.DataFrame(
        [[res.tn, res.fp], [res.fn, res.tp]],
        index=["Gerçek: değil", "Gerçek: ihlal"],
        columns=["Tahmin: değil", "Tahmin: ihlal"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.caption(
        f"TP={res.tp}  ·  FN={res.fn}  ·  FP={res.fp}  ·  TN={res.tn}"
    )

if res.per_category_recall:
    with st.expander("📂 Kategori başına P / R / F1"):
        import pandas as _pd
        rows = []
        for c in sorted(set(res.per_category_recall) | set(res.per_category_precision)):
            rows.append({
                "kategori": c,
                "precision": res.per_category_precision.get(c, 0.0),
                "recall": res.per_category_recall.get(c, 0.0),
                "f1": res.per_category_f1.get(c, 0.0),
                "support": res.per_category_support.get(c, 0),
            })
        st.dataframe(_pd.DataFrame(rows).sort_values("f1"),
                     hide_index=True, use_container_width=True)

# ---- Three-way comparison: Baseline | Injected | Model Prediction ----------
st.divider()
st.subheader("🔍 Baseline · İhlal Edilmiş · Model Tahmini")
st.caption(
    "Üç sütun, her birinde **üstte IFC 3D** ve **altında grafik**. "
    "Soldan sağa: ham baseline (temiz) → bizim enjekte ettiğimiz ihlaller "
    "(kırmızı) → modelin tahmini (yeşil)."
)

from ml.app.state import (
    entry_by_id, get_dataset_root, get_selected_node, set_selected_node,
)
import os as _os

node_ids = data.node_ids
predicted_guids = {node_ids[i] for i, p in enumerate(preds) if p == 1}
true_guids = {node_ids[i] for i, y in enumerate(y_true) if y == 1}
decoy_guids = set(sample.decoy_guids)

# Baseline entry (varsa). entry["parent_id"] enjekte edilmiş IFC'nin
# kaynak baseline'ına işaret eder.
baseline_entry = None
baseline_sample = None
if entry.get("parent_id"):
    baseline_entry = entry_by_id(get_dataset_root(), entry["parent_id"])
    if baseline_entry and baseline_entry.get("graph_ok"):
        baseline_sample = load_sample_for(baseline_entry)

# Seçili node — graph'lardan birine tıklandığında IFC 3D'lerde mavi vurgu.
selected_guid = get_selected_node()
all_guids: set[str] = set(g.nodes)
if baseline_sample:
    all_guids |= set(baseline_sample.graph.nodes)
if selected_guid not in all_guids:
    selected_guid = None

# Seçili node bilgi paneli
if selected_guid:
    info_cols = st.columns([5, 1])
    with info_cols[0]:
        nd = (g.nodes.get(selected_guid)
              or (baseline_sample.graph.nodes.get(selected_guid) if baseline_sample else None)
              or {})
        ifc_type = nd.get("ifc_type") or nd.get("type") or "?"
        name = (nd.get("attributes") or {}).get("Name", "")
        st.info(f"🔵 Seçili: **{ifc_type}** · {selected_guid}  ·  {name}")
    with info_cols[1]:
        if st.button("✕ Seçimi temizle", use_container_width=True):
            set_selected_node(None)
            st.rerun()

cols = st.columns(3)


def _render_3d(slot, title: str, ifc_path: str | None,
               highlights_red: set[str], highlights_green: set[str],
               highlights_amber: set[str] = set(),
               sel: str | None = None, key_suffix: str = ""):
    """IFC 3D'yi sütun içinde render eder (plotly native — slot.X yeterli)."""
    with slot:
        st.markdown(f"**{title}**")
        if not ifc_path or not _os.path.exists(ifc_path):
            st.caption("IFC dosyası bulunamadı.")
            return
        try:
            meshes = _cached_meshes(ifc_path, _mtime_safe(ifc_path))
        except Exception as e:
            st.error(f"IFC açılamadı: {e}")
            return
        if not meshes:
            st.caption("Boş mesh.")
            return
        fig = build_figure(
            meshes,
            violation_guids=highlights_red,
            decoy_guids=highlights_amber,
            path_guids=highlights_green,
            selected_guid=sel,        # 🔵 mavi vurgu
            height=460,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key_suffix}")


def _render_graph(slot, title: str, graph,
                  highlights_red: set[str], highlights_green: set[str],
                  highlights_amber: set[str] = set(),
                  sel: str | None = None, key_suffix: str = "") -> str | None:
    """Interaktif grafiği sütun içinde render eder.

    `interactive_agraph` özel bir Streamlit component (vis-network); top-level
    çağrıldığında sütun bağlamını algılamaz ve tüm genişliğe yayılır. Bu
    yüzden `with slot:` bloğunun *içinde* çağırmak şart.
    """
    with slot:
        st.markdown(f"**{title}**")
        if graph is None:
            st.caption("Grafik yok.")
            return None
        clicked = interactive_agraph(
            graph,
            violation_guids=highlights_red,
            decoy_guids=highlights_amber,
            path_guids=highlights_green,
            selected_guid=sel,
            height=400,
            key=f"graph_{key_suffix}",
        )
        return clicked
    return None


# Sütun 1 — BASELINE (temiz, vurgu yok)
baseline_ifc = baseline_entry.get("ifc_path") if baseline_entry else None
baseline_graph = baseline_sample.graph if baseline_sample else None
_render_3d(cols[0], "🧱 Baseline IFC (ham)",
           baseline_ifc, set(), set(),
           sel=selected_guid, key_suffix="baseline")
clicked_base = _render_graph(cols[0], "Baseline grafik",
                             baseline_graph, set(), set(),
                             sel=selected_guid, key_suffix="baseline")
if baseline_entry is None:
    cols[0].caption("Bu kayıt için baseline yok (parent_id boş veya imports).")

# Sütun 2 — INJECTED / GROUND TRUTH (kırmızı = gerçek enjekte ihlaller, sarı = decoy)
_render_3d(cols[1], "💥 İhlal Edilmiş IFC (gerçek)",
           entry["ifc_path"], true_guids, set(), decoy_guids,
           sel=selected_guid, key_suffix="violated")
clicked_vio = _render_graph(cols[1], "İhlal grafiği (kırmızı=gerçek, sarı=decoy)",
                            g, true_guids, set(), decoy_guids,
                            sel=selected_guid, key_suffix="violated")

# Sütun 3 — MODEL PREDICTION (yeşil = tahmin)
_render_3d(cols[2], "🤖 Model Tahmini",
           entry["ifc_path"], set(), predicted_guids,
           sel=selected_guid, key_suffix="pred")
clicked_pred = _render_graph(cols[2], "Tahmin grafiği (yeşil=GAT)",
                             g, set(), predicted_guids,
                             sel=selected_guid, key_suffix="pred")

# Tıklamayı yakala — herhangi bir grafikten gelen yeni seçimi state'e yaz.
for clk in (clicked_base, clicked_vio, clicked_pred):
    if clk and clk != selected_guid:
        set_selected_node(clk)
        st.rerun()

st.caption(
    "💡 **Tıkla / sürükle:** Grafik node'larını sürükleyebilir, tıklayarak "
    "ilgili IFC elemanını üç 3D görünümde de **mavi** vurgulu görebilirsin. "
    "Renk legend: yeşil+kırmızı çakışma=TP, sadece yeşil=FP, sadece kırmızı=FN, sarı=decoy."
)

# ---- Confusion table --------------------------------------------------------
st.divider()
st.subheader("Hata analizi")
tp = sorted(predicted_guids & true_guids)
fp = sorted(predicted_guids - true_guids - decoy_guids)
fn = sorted(true_guids - predicted_guids)
decoy_fp = sorted(predicted_guids & decoy_guids)

cols = st.columns(4)
cols[0].metric("TP", len(tp))
cols[1].metric("FP", len(fp))
cols[2].metric("FN", len(fn))
cols[3].metric("Decoy → FP", len(decoy_fp))

# ---- Eleman (IFC tipi) bazında confusion matrix ----------------------------
st.markdown("### IFC Eleman Tipine Göre Confusion Matrix")
st.caption(
    "Hangi tipte eleman (Door, Stair, Wall, …) için modelin nerede başarılı "
    "ve nerede zorlandığı. TP+FN sütununa bakarak modelin **hangi tipte** "
    "ihlal yakaladığını veya kaçırdığını görebilirsin."
)

import pandas as _pd

def _type_of(guid: str) -> str:
    nd = g.nodes.get(guid, {})
    return nd.get("ifc_type") or nd.get("type") or "?"

# Tüm IFC tiplerini topla
all_types = sorted({_type_of(guid) for guid in node_ids})
type_rows = []
for t in all_types:
    nodes_t = {guid for guid in node_ids if _type_of(guid) == t}
    tp_t = len(set(tp) & nodes_t)
    fp_t = len(set(fp) & nodes_t)
    fn_t = len(set(fn) & nodes_t)
    decoy_fp_t = len(set(decoy_fp) & nodes_t)
    pos_t = tp_t + fn_t       # gerçek pozitif sayısı
    pred_t = tp_t + fp_t      # tahmin edilen pozitif sayısı
    tn_t = len(nodes_t) - tp_t - fp_t - fn_t
    recall_t = (tp_t / pos_t) if pos_t else None
    prec_t = (tp_t / pred_t) if pred_t else None
    f1_t = (2 * prec_t * recall_t / (prec_t + recall_t)
            if (prec_t and recall_t) else None)
    type_rows.append({
        "IFC tipi": t,
        "Toplam": len(nodes_t),
        "TP": tp_t,
        "FP": fp_t,
        "FN": fn_t,
        "TN": tn_t,
        "Decoy→FP": decoy_fp_t,
        "Precision": round(prec_t, 3) if prec_t is not None else None,
        "Recall": round(recall_t, 3) if recall_t is not None else None,
        "F1": round(f1_t, 3) if f1_t is not None else None,
    })

type_df = _pd.DataFrame(type_rows)
# Sadece eğitim/test sinyali olan satırları öncele (TP+FN+FP>0)
type_df["_has_signal"] = (type_df["TP"] + type_df["FN"] + type_df["FP"]) > 0
type_df = type_df.sort_values(
    ["_has_signal", "F1", "FN"], ascending=[False, True, False]
).drop(columns="_has_signal")
st.dataframe(type_df, hide_index=True, use_container_width=True)

# TP/FN/FP stacked bar — sinyali olan tipler için
signal_df = type_df[(type_df["TP"] + type_df["FN"] + type_df["FP"]) > 0]
if not signal_df.empty:
    st.markdown("**Tip bazında dağılım** (yeşil=TP, kırmızı=FN, turuncu=FP)")
    chart_df = signal_df[["IFC tipi", "TP", "FN", "FP"]].set_index("IFC tipi")
    st.bar_chart(chart_df, color=["#22c55e", "#ef4444", "#f97316"])

# ---- Liste expandlar -------------------------------------------------------
with st.expander(f"❌ FN — kaçırılan {len(fn)} ihlal (tıkla → mavi vurgula)"):
    for guid in fn:
        nd = g.nodes[guid]
        c1, c2 = st.columns([5, 1])
        c1.write(f"• **{nd.get('ifc_type', '?')}** · `{guid[:12]}…` · "
                 f"{(nd.get('attributes') or {}).get('Name', '')}")
        if c2.button("👁", key=f"fn_{guid}", help="Bu node'u seç"):
            set_selected_node(guid)
            st.rerun()

with st.expander(f"⚠️ FP — yanlış alarm {len(fp)} (tıkla → mavi vurgula)"):
    for guid in fp:
        nd = g.nodes[guid]
        c1, c2 = st.columns([5, 1])
        c1.write(f"• **{nd.get('ifc_type', '?')}** · `{guid[:12]}…` · "
                 f"{(nd.get('attributes') or {}).get('Name', '')}")
        if c2.button("👁", key=f"fp_{guid}", help="Bu node'u seç"):
            set_selected_node(guid)
            st.rerun()
