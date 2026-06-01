"""Tab: room-to-room routing.

Pick two IfcSpaces; we compute three routes:

    * shortest (BFS, minimum hop count)
    * widest (max-of-min door width — bottleneck path)
    * accessible (weighted: narrow doors penalised heavily)

Each route is drawn on both the graph view and the IFC 3D view in a
distinct colour, plus a table summarising door widths so you can see
which option fails the 0.9 m accessibility threshold.

There's also a 'nearest exit' query that finds the best route from
one room to the closest external door (`IsExternal=True`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.analysis import (
    accessible_path, list_rooms, shortest_path, widest_path,
)
from ml.analysis.pathfind import ACCESSIBLE_WIDTH_M, nearest_exit
from ml.app.state import labels_summary, load_sample_for, sidebar_config
from ml.viz.graph_view import static_plotly
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


def _path_color_legend():
    st.caption(
        "🟢 birinci yol  ·  🟣 ikinci yol  ·  "
        f"erişilebilir kapı eşiği: {ACCESSIBLE_WIDTH_M:.2f} m"
    )


st.set_page_config(page_title="Statik Analiz", layout="wide", page_icon="🧭")
entry = sidebar_config()

st.title("🧭 Statik Analiz — Oda Arası Yol Bulma")
st.caption("İki oda seç, en kısa / en geniş / en erişilebilir yolu hem graph hem IFC üzerinde gör.")

if entry is None:
    st.stop()

sample = load_sample_for(entry)
g = sample.graph
rooms = list_rooms(g)

if len(rooms) < 2:
    st.warning("Bu modelde yol bulmak için yeterli IfcSpace yok.")
    st.stop()

room_label = {r["guid"]: f"{r['name']}  ·  z={r['elevation']}" for r in rooms}

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    src = st.selectbox("Başlangıç odası", options=[r["guid"] for r in rooms],
                       format_func=lambda g: room_label[g], key="path_src")
with col_b:
    dst_options = [r["guid"] for r in rooms if r["guid"] != src]
    dst = st.selectbox("Hedef oda", options=dst_options,
                       format_func=lambda g: room_label[g], key="path_dst")
with col_c:
    mode = st.radio(
        "Görüntülenecek yollar",
        options=["en kısa + en erişilebilir", "en kısa + en geniş",
                 "en erişilebilir + en geniş", "üçü ayrı ayrı"],
        index=0,
    )

go = st.button("Yolları hesapla", type="primary")

st.divider()

if not go and "path_results" not in st.session_state:
    st.info("Yolları hesaplamak için yukarıdaki butona bas.")
    st.stop()

if go:
    results = {
        "en kısa": shortest_path(g, src, dst),
        "en erişilebilir": accessible_path(g, src, dst),
        "en geniş": widest_path(g, src, dst),
    }
    st.session_state["path_results"] = results
    st.session_state["path_src"] = src
    st.session_state["path_dst"] = dst

results = st.session_state["path_results"]

# Metric table
mcols = st.columns(3)
for i, (name, res) in enumerate(results.items()):
    with mcols[i]:
        st.metric(f"{name}", f"{res.metric:.2f}" if res.metric is not None else "—",
                  help=res.detail)
        if not res.found:
            st.caption(f"❌ {res.detail}")
        else:
            widths = res.door_widths
            if widths:
                bottleneck = min(widths)
                ok = bottleneck >= ACCESSIBLE_WIDTH_M
                st.caption(f"darboğaz: {bottleneck:.2f} m  {'✅' if ok else '⚠️'}")

# Determine which two paths to draw on the views.
if mode == "üçü ayrı ayrı":
    chosen_pairs = [("en kısa", "en erişilebilir"),
                    ("en kısa", "en geniş"),
                    ("en erişilebilir", "en geniş")]
else:
    a, b = mode.split(" + ")
    chosen_pairs = [(a, b)]

st.divider()
_path_color_legend()

# Cached IFC meshes once (independent of which paths we draw).
meshes = None
if entry["ifc_path"] and Path(entry["ifc_path"]).exists():
    try:
        meshes = _cached_meshes(entry["ifc_path"], _mtime_safe(entry["ifc_path"]))
    except ImportError as e:
        st.error(f"ifcopenshell kullanılamıyor: {e}")
    except Exception as e:
        st.error(f"IFC açılamadı: {e}")

for primary_name, secondary_name in chosen_pairs:
    primary = results[primary_name]
    secondary = results[secondary_name]
    st.subheader(f"{primary_name} (yeşil) vs {secondary_name} (mor)")

    left, right = st.columns(2)
    with left:
        st.markdown("**Graph**")
        fig = static_plotly(
            g,
            path_guids=set(primary.nodes) if primary.found else set(),
            path_edges=primary.edges if primary.found else [],
            path_alt_guids=set(secondary.nodes) if secondary.found else set(),
            path_alt_edges=secondary.edges if secondary.found else [],
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**IFC 3D**")
        if not meshes:
            st.caption("IFC görüntülenemiyor.")
        else:
            fig = build_figure(
                meshes,
                path_guids=set(primary.nodes) if primary.found else set(),
                path_alt_guids=set(secondary.nodes) if secondary.found else set(),
                height=560,
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# Nearest-exit query
st.subheader("🚪 En yakın çıkışa yol")
exit_cols = st.columns([2, 2, 1])
with exit_cols[0]:
    exit_src = st.selectbox(
        "Hangi odadan?", options=[r["guid"] for r in rooms],
        format_func=lambda g: room_label[g], key="exit_src",
    )
with exit_cols[1]:
    exit_variant = st.selectbox(
        "Varyant", options=["accessible", "shortest", "widest"],
        index=0, key="exit_variant",
    )
with exit_cols[2]:
    go_exit = st.button("Bul", key="exit_go")

if go_exit:
    res, door = nearest_exit(g, exit_src, variant=exit_variant)
    if not res.found:
        st.error(res.detail)
    else:
        st.success(f"Çıkış kapısı: {door[:8]}  ·  {res.detail}")
        left, right = st.columns(2)
        with left:
            fig = static_plotly(
                g, path_guids=set(res.nodes), path_edges=res.edges, height=540,
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            if meshes:
                fig = build_figure(meshes, path_guids=set(res.nodes), height=540)
                st.plotly_chart(fig, use_container_width=True)
