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


def _render_ifc(meshes, *, violation_guids, decoy_guids, normal_guids=None,
                selected_guid, height, key_suffix=""):
    if meshes is None:
        st.info("IFC dosyası bulunamadı ya da ifcopenshell yok.")
        return
    fig = build_figure(
        meshes,
        violation_guids=violation_guids,
        decoy_guids=decoy_guids,
        normal_guids=normal_guids,
        selected_guid=selected_guid,
        height=height,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key_suffix}")


def _render_graph(sample, *, violation_guids, decoy_guids, normal_guids=None,
                  selected_guid, height, key_suffix=""):
    if sample is None:
        st.info("Bu model için graph.json üretilmemiş.")
        return None
    clicked = interactive_agraph(
        sample.graph,
        violation_guids=violation_guids,
        decoy_guids=decoy_guids,
        normal_guids=normal_guids,
        selected_guid=selected_guid,
        height=height,
        key=f"graph_{key_suffix}",
    )
    return clicked


# ---- Page -------------------------------------------------------------------

st.set_page_config(page_title="Model Görüntüleyici", layout="wide", page_icon="🧱")
entry = sidebar_config()

st.title("🧱 Model Görüntüleyici")

if entry is None:
    st.stop()

root = get_dataset_root()

# --- Üstte: Baseline'dan ihlalleri gez (sidebar Tür'üne bağlı değil) -------
from ml.app.state import list_entries as _list_entries
st.markdown("### 📦 Baseline'dan ihlalleri gez")
st.caption(
    "Bir baseline seç → ondan üretilen TÜM ihlalleri ◀▶ ile sırayla incele. "
    "Solda baseline (temiz), sağda o ihlal. (Tek model incelemek için "
    "soldaki sidebar'ı kullan.)"
)
# Baseline'ı graph şartı OLMADAN listele: baseline'ın 3D'si IFC'den gelir,
# ihlal listesi de baseline'ın kendi graph'ına ihtiyaç duymaz. (require_graph
# True iken graph'sız sentetik baseline'lar gizleniyordu → liste boş kalıyordu.)
from ml.app.state import violated_children as _vchildren
_baselines = _list_entries(root, kind="baseline", require_graph=False)
# Sadece en az bir ihlali olan baseline'ları öne çıkar (gezilebilir olanlar);
# yine de hiç ihlali olmayanları da en sona ekle ki baseline 3D'si görülebilsin.
_vio_counts = {b["id"]: len(_vchildren(root, b["id"])) for b in _baselines}
_baselines.sort(key=lambda b: (-_vio_counts.get(b["id"], 0), b["name"]))
_bopts = [None] + list(range(len(_baselines)))
_bsel = st.selectbox(
    "Baseline seç (ihlallerini gezmek için)",
    options=_bopts,
    index=0,
    format_func=lambda i: (
        "— sidebar seçimini kullan —" if i is None
        else f"{_baselines[i]['name']} · {_baselines[i]['id'][:8]} "
             f"· {_vio_counts.get(_baselines[i]['id'], 0)} ihlal"),
    key="mv_baseline_browse",
)
if _bsel is not None:
    # Baseline gezgini moduna geç — entry'yi seçilen baseline yap
    entry = _baselines[_bsel]
    # Seçilen baseline'dan üretilen ihlalleri altta liste olarak göster
    _kids = _vchildren(root, entry["id"])
    if _kids:
        import pandas as _pd
        st.markdown(f"**📋 `{entry['name']}` baseline'ından üretilen "
                    f"{len(_kids)} ihlal:**")
        st.dataframe(
            _pd.DataFrame([{
                "#": i + 1,
                "ihlal dosyası": k["name"],
                "id": k["id"][:8],
                "graph": "✓" if k.get("graph_ok") else "—",
            } for i, k in enumerate(_kids)]),
            hide_index=True, use_container_width=True, height=min(280, 60 + 35 * len(_kids)),
        )
        st.caption("Aşağıdan ◀▶ ile sırayla gez ya da açılır listeden seç.")
    else:
        st.info("Bu baseline'dan henüz ihlal üretilmemiş.")

st.divider()
st.caption(
    "Violated bir model seçince baseline ile **yan yana 4-panel** "
    "karşılaştırma görürsün. Her panelin üstündeki **⛶ Büyüt** ile o "
    "paneli tam genişliğe geçirebilirsin."
)

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
        options=["İhlaller", "Decoys", "Normal (ihlal olmayan etiketli)"],
        default=["İhlaller", "Decoys"],
        help="Violated tarafında işaretli elemanları boyar: ihlal=kırmızı, "
             "decoy=sarı, normal (uyumlu/clean etiketli)=yeşil. 'Normal' "
             "katmanı tam etiketlemede tüm yapının açıkça 'ihlal değil' "
             "işaretlendiğini görmeni sağlar.",
    )
with ctrl_b:
    if entry["kind"] == "violated" and partner_entry is None:
        st.warning("Bu violated modelin baseline'ı listede yok; sadece violated görüntüleniyor.")
    elif entry["kind"] == "violated":
        st.success(f"Baseline ile karşılaştırılıyor: `{partner_entry['id'][:8]}`")
    elif entry["kind"] == "baseline":
        # Baseline seçilince: ondan üretilen TÜM ihlalleri gez (prev/next).
        from ml.app.state import violated_children
        kids = violated_children(root, entry["id"])
        if kids:
            st.caption(f"🔢 Bu baseline'dan **{len(kids)}** ihlal üretilmiş — "
                       "sırayla gez:")
            _key = f"viobrowse_{entry['id']}"
            cur = st.session_state.get(_key, 0)
            cur = max(0, min(cur, len(kids) - 1))
            nav = st.columns([1, 1, 3, 1])
            if nav[0].button("◀ Önceki", use_container_width=True,
                             disabled=cur == 0, key=f"{_key}_prev"):
                st.session_state[_key] = cur - 1
                st.rerun()
            if nav[1].button("Sonraki ▶", use_container_width=True,
                             disabled=cur >= len(kids) - 1, key=f"{_key}_next"):
                st.session_state[_key] = cur + 1
                st.rerun()
            with nav[2]:
                picked = st.selectbox(
                    "İhlal seç",
                    options=list(range(len(kids))),
                    index=cur,
                    format_func=lambda i: f"[{i+1}/{len(kids)}] {kids[i]['name']} · {kids[i]['id'][:8]}",
                    key=f"{_key}_sel",
                    label_visibility="collapsed",
                )
                if picked != cur:
                    st.session_state[_key] = picked
                    st.rerun()
            nav[3].metric("Sıra", f"{cur+1}/{len(kids)}")
            partner_entry = kids[cur]
            partner_sample = load_sample_for(partner_entry)
        else:
            st.info("Bu baseline'dan henüz ihlal üretilmemiş.")

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

# Normal = etikette açıkça 'ihlal değil' (compliant / clean) işaretli node'lar.
# İhlal ve decoy hariç tutulur; labels.json'dan okunur.
normal_guids: set[str] = set()
if violated_entry is not None:
    _doc = labels_summary(violated_entry) or {}
    for _l in _doc.get("labels", []):
        _g = _l.get("ifc_global_id")
        _stt = (_l.get("status") or "").lower()
        if _g and _stt in ("compliant", "clean") and not _l.get("is_decoy"):
            normal_guids.add(_g)
    normal_guids -= violation_guids
    normal_guids -= decoy_guids

vio_show = violation_guids if "İhlaller" in overlay else set()
dec_show = decoy_guids if "Decoys" in overlay else set()
nor_show = normal_guids if "Normal (ihlal olmayan etiketli)" in overlay else set()

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
                normal_guids=nor_show,
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
                            normal_guids=nor_show,
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
