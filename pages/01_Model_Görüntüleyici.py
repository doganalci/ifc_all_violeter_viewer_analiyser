"""Tab: side-by-side IFC 3D + interactive graph.

Layout adapts to what's selected:

  * **baseline / imported**  → 2 paneller: IFC + graph
  * **violated** with a parent baseline → **4 paneller (2×2)**:
        ┌─────────────────┬──────────────────┐
        │  Baseline IFC   │  Violated IFC    │
        ├─────────────────┼──────────────────┤
        │  Baseline Graph │  Violated Graph  │
        └─────────────────┴──────────────────┘

Her panelin başlığında bir **⛶ Büyüt** tik kutusu var: işaretlersen
panel tam ekrana yayılır (diğerleri saklanır), kaldırınca grid'e döner.

Cross-highlight: violated graph'ta bir node'a tıklarsan hem violated
IFC hem de baseline tarafı (GUID baseline'da varsa) aynı elemanı
mavi vurguluyor — modifiye edilmiş eleman karşılığını anında görmek
için.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    entry_by_id, get_dataset_root, get_selected_node, labels_summary,
    load_sample_for, set_selected_node, sidebar_config,
)
from ml.viz.graph_view import interactive_agraph
from ml.viz.ifc3d import build_figure, extract_meshes


# ---- Cached IFC tessellation ------------------------------------------------

@st.cache_data(show_spinner="IFC tessellate ediliyor...")
def _cached_meshes(ifc_path: str, _mtime: float):
    return extract_meshes(ifc_path)


def _mtime_safe(p: str | None) -> float:
    import os
    if not p:
        return 0.0
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _safe_meshes(ifc_path: str | None) -> list | None:
    if not ifc_path or not Path(ifc_path).exists():
        return None
    try:
        return _cached_meshes(ifc_path, _mtime_safe(ifc_path))
    except ImportError as e:
        st.error(f"ifcopenshell kullanılamıyor: {e}")
        return None
    except Exception as e:
        st.error(f"IFC açılamadı ({Path(ifc_path).name}): {e}")
        return None


# ---- Panel framework --------------------------------------------------------

def _panel_header(title: str, idx: int) -> bool:
    """Render the title row with a 'büyüt' checkbox; return its state."""
    head = st.columns([6, 1])
    with head[0]:
        st.markdown(f"### {title}")
    with head[1]:
        return st.checkbox("⛶ Büyüt", value=False, key=f"max_{idx}",
                           help="Bu paneli tam genişlikte göster")


def _render_ifc(meshes, *, violation_guids, decoy_guids,
                selected_guid, height, key_suffix=""):
    if meshes is None:
        st.info("IFC dosyası bulunamadı ya da ifcopenshell yok.")
        return
    fig = build_figure(
        meshes,
        violation_guids=violation_guids,
        decoy_guids=decoy_guids,
        selected_guid=selected_guid,
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key_suffix}")


def _render_graph(sample, *, violation_guids, decoy_guids,
                  selected_guid, height, key_suffix=""):
    if sample is None:
        st.info("Bu model için graph.json üretilmemiş.")
        return None
    clicked = interactive_agraph(
        sample.graph,
        violation_guids=violation_guids,
        decoy_guids=decoy_guids,
        selected_guid=selected_guid,
        height=height,
        key=f"graph_{key_suffix}",
    )
    return clicked


# ---- Page -------------------------------------------------------------------

st.set_page_config(page_title="Model Görüntüleyici", layout="wide", page_icon="🧱")
entry = sidebar_config()

st.title("🧱 Model Görüntüleyici")
st.caption(
    "Violated bir model seçince baseline ile **yan yana 4-panel** "
    "karşılaştırma görürsün. Her panelin üstündeki **⛶ Büyüt** ile o "
    "paneli tam genişliğe geçirebilirsin."
)

if entry is None:
    st.stop()

root = get_dataset_root()

# Selected sample (may be None if graph not generated yet)
sample = load_sample_for(entry)

# Resolve the comparison partner.
partner_entry: dict | None = None
partner_sample = None
if entry["kind"] == "violated" and entry.get("parent_id"):
    partner_entry = entry_by_id(root, entry["parent_id"])
    if partner_entry:
        partner_sample = load_sample_for(partner_entry)

# ---- Top controls -----------------------------------------------------------
ctrl_a, ctrl_b = st.columns([3, 2])
with ctrl_a:
    overlay = st.multiselect(
        "Vurgu katmanları",
        options=["İhlaller", "Decoys"],
        default=["İhlaller", "Decoys"],
        help="Violated tarafında işaretli elemanları kırmızı/sarı boyar.",
    )
with ctrl_b:
    if entry["kind"] == "violated" and partner_entry is None:
        st.warning("Bu violated modelin baseline'ı listede yok; sadece violated görüntüleniyor.")
    elif entry["kind"] == "violated":
        st.success(f"Baseline ile karşılaştırılıyor: `{partner_entry['id'][:8]}`")
    elif entry["kind"] == "baseline":
        # Allow opening any one of its violated children alongside.
        from app.state import violated_children
        kids = violated_children(root, entry["id"])
        if kids:
            kid_idx = st.selectbox(
                "Yan yana göstermek için violated seç (opsiyonel)",
                options=[None] + list(range(len(kids))),
                format_func=lambda i: "— yok —" if i is None
                                       else f"{kids[i]['name']} · {kids[i]['id'][:8]}",
            )
            if kid_idx is not None:
                partner_entry = kids[kid_idx]
                partner_sample = load_sample_for(partner_entry)

# Determine which sample defines the violation/decoy sets (always
# from the violated side — baseline has none of these labels).
violated_sample = None
baseline_sample = None
violated_entry: dict | None = None
baseline_entry: dict | None = None
if entry["kind"] == "violated":
    violated_sample, violated_entry = sample, entry
    baseline_sample, baseline_entry = partner_sample, partner_entry
elif entry["kind"] == "baseline":
    baseline_sample, baseline_entry = sample, entry
    violated_sample, violated_entry = partner_sample, partner_entry
else:  # imported — treat as standalone
    baseline_sample, baseline_entry = sample, entry

if violated_sample is not None:
    violation_guids = {gid for gid, y in violated_sample.y.items() if y == 1}
    decoy_guids = violated_sample.decoy_guids
else:
    violation_guids = set()
    decoy_guids = set()

vio_show = violation_guids if "İhlaller" in overlay else set()
dec_show = decoy_guids if "Decoys" in overlay else set()

# ---- Selected node + panels -------------------------------------------------
current = get_selected_node()
all_guids: set[str] = set()
if violated_sample:
    all_guids.update(violated_sample.graph.nodes)
if baseline_sample:
    all_guids.update(baseline_sample.graph.nodes)
if current not in all_guids:
    current = None

# Build panel descriptors.
PANEL_HEIGHT_GRID = 460
PANEL_HEIGHT_FULL = 760

baseline_meshes = _safe_meshes(baseline_entry["ifc_path"]) if baseline_entry else None
violated_meshes = _safe_meshes(violated_entry["ifc_path"]) if violated_entry else None

is_pair = (violated_entry is not None) and (baseline_entry is not None)


def panel_baseline_ifc(height: int):
    _render_ifc(baseline_meshes,
                violation_guids=set(),
                decoy_guids=set(),
                selected_guid=current,
                height=height, key_suffix="baseline_ifc")


def panel_violated_ifc(height: int):
    _render_ifc(violated_meshes,
                violation_guids=vio_show,
                decoy_guids=dec_show,
                selected_guid=current,
                height=height, key_suffix="violated_ifc")


def panel_baseline_graph(height: int):
    clicked = _render_graph(baseline_sample,
                            violation_guids=set(),
                            decoy_guids=set(),
                            selected_guid=current,
                            height=height, key_suffix="baseline_graph")
    if clicked and clicked != current:
        set_selected_node(clicked)
        st.rerun()


def panel_violated_graph(height: int):
    clicked = _render_graph(violated_sample,
                            violation_guids=vio_show,
                            decoy_guids=dec_show,
                            selected_guid=current,
                            height=height, key_suffix="violated_graph")
    if clicked and clicked != current:
        set_selected_node(clicked)
        st.rerun()


# Pick the panel set based on layout.
if is_pair:
    panels = [
        ("Baseline · IFC 3D", panel_baseline_ifc),
        ("Violated · IFC 3D", panel_violated_ifc),
        ("Baseline · Graph",  panel_baseline_graph),
        ("Violated · Graph",  panel_violated_graph),
    ]
elif violated_entry is not None:
    panels = [
        ("Violated · IFC 3D", panel_violated_ifc),
        ("Violated · Graph",  panel_violated_graph),
    ]
else:
    panels = [
        ("IFC 3D", panel_baseline_ifc),
        ("Graph",  panel_baseline_graph),
    ]

# Resolve maximize: the first checked panel wins.
max_idx = next(
    (i for i in range(len(panels))
     if st.session_state.get(f"max_{i}", False)),
    None,
)

st.divider()

if max_idx is not None:
    title, render_fn = panels[max_idx]
    _panel_header(title, max_idx)
    render_fn(PANEL_HEIGHT_FULL)
    st.caption("Tik'i kaldırarak grid görünüme dön.")
else:
    if len(panels) == 4:
        row1 = st.columns(2)
        row2 = st.columns(2)
        slots = [row1[0], row1[1], row2[0], row2[1]]
    else:
        slots = st.columns(len(panels))
    for i, slot in enumerate(slots):
        with slot:
            title, render_fn = panels[i]
            _panel_header(title, i)
            render_fn(PANEL_HEIGHT_GRID)

# ---- Inspector --------------------------------------------------------------
if current:
    primary_g = (violated_sample or baseline_sample).graph
    if current not in primary_g.nodes and baseline_sample is not None:
        primary_g = baseline_sample.graph
    if current in primary_g.nodes:
        nd = primary_g.nodes[current]
        st.divider()
        st.subheader(f"🔎 {nd.get('ifc_type', '?')} — {current}")
        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown("**Attributes**")
            st.json(nd.get("attributes") or {}, expanded=False)
        with info_cols[1]:
            st.markdown("**Psets**")
            st.json(nd.get("psets") or {}, expanded=False)
        with info_cols[2]:
            st.markdown("**Etiketler**")
            is_v = current in violation_guids
            is_d = current in decoy_guids
            st.write("• İhlal mi?", "✅ Evet" if is_v else "—")
            st.write("• Decoy mi?", "🪤 Evet" if is_d else "—")
            if violated_entry:
                doc = labels_summary(violated_entry) or {}
                match = [l for l in doc.get("labels", [])
                         if l.get("ifc_global_id") == current]
                if match:
                    lab = match[0]
                    if lab.get("attribute"):
                        st.write(f"`{lab['attribute']}`: "
                                 f"{lab.get('value_before')} → "
                                 f"{lab.get('value_after')}")
                    with st.expander("Tam etiket"):
                        st.json(lab, expanded=False)
