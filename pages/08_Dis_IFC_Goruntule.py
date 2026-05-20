"""Dış IFC görüntüleyici — dataset dışında olan tek IFC veya bir klasördeki tüm
IFC'leri hızlıca 3D + graph olarak gösterir.

Burada IFC'ler **kalıcı olarak kaydedilmez**. Eğer kaydedip ihlal enjekte etmek
istersen → sayfa 1 'IFC Stüdyo' → 'Gerçek IFC içe aktar' tabını kullan
(yüklenen dosya `data/ifc_models/imports/` altında kaydedilir, graph üretilir,
sonra ihlal enjeksiyonu için seçilebilir).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from viewer.ifc3d import ifc_to_figure
from viewer.graph_view import graph_to_figure
from violation_pool import ifc_graph


st.set_page_config(page_title="Dış IFC Görüntüle", layout="wide", page_icon="📂")
st.title("📂 Dış IFC Görüntüle")
st.caption(
    "Dataset'in dışındaki tek bir IFC dosyasını veya bir klasördeki tüm "
    "IFC'leri hızlıca incele. Kalıcı kayıt **yok** — kaydetmek + ihlal "
    "enjekte etmek istersen sayfa 1 'IFC Stüdyo → Gerçek IFC içe aktar'."
)


# ------------------------------------------------------------------
# Cache yardımcıları
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _ifc_fig(path_str: str, mtime: float):
    return ifc_to_figure(path_str)


@st.cache_resource(show_spinner=False)
def _graph_fig(path_str: str, mtime: float):
    g = ifc_graph.build_graph(path_str)
    return graph_to_figure(g)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _render_ifc(path: Path) -> None:
    st.markdown(f"### `{path.name}`")
    st.caption(str(path))
    c3d, cgr = st.tabs(["3D", "Graph"])
    with c3d:
        try:
            with st.spinner("Tessellation..."):
                fig = _ifc_fig(str(path), _mtime(path))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"3D çizilemedi: {e}")
    with cgr:
        try:
            with st.spinner("Graph üretiliyor..."):
                fig = _graph_fig(str(path), _mtime(path))
            st.plotly_chart(
                fig, use_container_width=True,
                config={"scrollZoom": True, "displaylogo": False},
            )
        except Exception as e:
            st.error(f"Graph üretilemedi: {e}")


# ------------------------------------------------------------------
# UI: iki tab
# ------------------------------------------------------------------
tab_upload, tab_folder = st.tabs(
    ["🔼 Tek dosya yükle", "🗂️ Klasördeki tüm IFC'ler"]
)

# ---- Tab 1: tek dosya upload --------------------------------------
with tab_upload:
    uf = st.file_uploader("IFC dosyası seç", type=["ifc"], key="single_upload")
    if uf is not None:
        with tempfile.NamedTemporaryFile(
            suffix=".ifc", delete=False, prefix="adhoc_"
        ) as t:
            t.write(uf.getbuffer())
            tmp_path = Path(t.name)
        try:
            _render_ifc(tmp_path)
        finally:
            # tempfile'ı kapat ama silme — Streamlit re-run'da cache hâlâ
            # ona referans verebilir; OS rastgele bir gün temizler.
            pass
    else:
        st.info("Yukarıdan bir `.ifc` dosyası seç.")

# ---- Tab 2: klasör tarama -----------------------------------------
with tab_folder:
    default_dir = st.session_state.get("adhoc_dir", str(Path.home() / "Desktop"))
    folder = st.text_input(
        "Klasör yolu",
        value=default_dir,
        help="Bu klasör (ve alt klasörler) içindeki tüm `.ifc` dosyaları listelenir.",
    )
    recursive = st.checkbox("Alt klasörleri de tara", value=True)

    base = Path(folder).expanduser()
    if not base.exists():
        st.warning(f"Klasör bulunamadı: `{base}`")
    elif not base.is_dir():
        st.warning(f"Bu bir klasör değil: `{base}`")
    else:
        st.session_state["adhoc_dir"] = str(base)
        pattern = "**/*.ifc" if recursive else "*.ifc"
        files = sorted(base.glob(pattern))
        st.caption(f"{len(files)} IFC dosyası bulundu.")

        if not files:
            st.info("Bu klasörde IFC dosyası yok.")
        else:
            labels = [
                f"{f.relative_to(base)}  ·  {f.stat().st_size // 1024} KB"
                for f in files
            ]
            idx = st.selectbox(
                "Görüntülenecek dosya",
                options=list(range(len(files))),
                format_func=lambda i: labels[i],
            )
            _render_ifc(files[idx])
