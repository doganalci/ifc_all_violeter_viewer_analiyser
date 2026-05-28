"""IFC Görüntüleyici — 3 sütun (ana baseline | baseline | ihlalli) × 2 satır (3D + graph).

Akış:
  * Üstte ana baseline (paket) seçilir → tüm paket aktif olur.
  * Col 1 = **Ana baseline** (paketin ilk baseline'ı, sabit referans).
  * Col 2 = **Baseline** — paket içinden bir baseline (◀/▶/dropdown).
  * Col 3 = **İhlalli** — seçili baseline'dan üretilen ihlalli IFC (◀/▶/dropdown).
  * Üst sıra: 3D IFC render (Plotly Mesh3d).
  * Alt sıra: interaktif graph (streamlit-agraph: node sürükle, tıkla).
  * Graph'ta tıklama → üç sütunda da aynı GUID parlatılır (cross-highlight).

Boş veri:
  * Baseline yoksa sayfa durur.
  * İhlalli yoksa col 3 = "📭 Veri üretilmemiş".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    get_dataset_root, get_selected_node, labels_summary,
    list_entries, load_sample_for, set_selected_node, violated_children,
)
from ml.viz.graph_view import interactive_agraph
from ml.viz.ifc3d import build_figure, extract_meshes
from violation_pool import storage


def _ifc_prompt_panel(entry: dict | None, key: str) -> None:
    """Verilen IFC için LLM prompt + tasarım planı expander'ları çiz."""
    if entry is None:
        return
    rec = storage.get_ifc_model(entry["id"]) or {}
    user_prompt = rec.get("prompt") or ""
    llm_model = rec.get("llm_model") or ""
    params = rec.get("params") or {}
    if isinstance(params, str):
        import json as _json
        try:
            params = _json.loads(params)
        except Exception:
            params = {}
    design = params.get("design_plan") or {}
    kind_label = params.get("kind_label") or ""

    if not (user_prompt or design):
        # LLM'siz üretilmiş (eski prosedürel) — yine de bilgi göster
        with st.expander(f"ℹ️ Üretim bilgisi · {key}", expanded=False):
            st.write(f"**Model:** `{llm_model or 'synth (LLM yok)'}`")
            st.write(f"**Tür:** `{kind_label or '—'}`")
            if params:
                st.json(params, expanded=False)
        return

    with st.expander(f"🤖 LLM prompt · {key}", expanded=False):
        meta_cols = st.columns([1, 1, 2])
        meta_cols[0].metric("Model", llm_model or "?")
        meta_cols[1].metric("Tür", kind_label or "baseline")
        if design:
            meta_cols[2].caption(
                f"🏛️ Tasarım: **{design.get('n_storeys', '?')} kat** · "
                f"**{design.get('n_rooms_per_floor', '?')} oda/kat** · "
                f"layout=`{design.get('layout', '?')}`"
            )
        if user_prompt:
            st.markdown("**User prompt** (LLM'e gönderilen):")
            st.code(user_prompt, language="text")
        if design.get("system_prompt"):
            st.markdown("**System prompt** (LLM rolü):")
            st.code(design["system_prompt"], language="text")
        if design.get("rationale"):
            st.markdown(f"**LLM rationale:** _{design['rationale']}_")
        if design.get("raw_llm_response"):
            st.markdown("**LLM ham yanıtı:**")
            st.code(design["raw_llm_response"], language="json")


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


def _safe_meshes(ifc_path: str | None):
    if not ifc_path or not Path(ifc_path).exists():
        return None
    try:
        return _cached_meshes(ifc_path, _mtime_safe(ifc_path))
    except ImportError as e:
        st.error(f"ifcopenshell yok: {e}")
        return None
    except Exception as e:
        st.error(f"IFC açılamadı ({Path(ifc_path).name}): {e}")
        return None


# ---- Page ------------------------------------------------------------------

st.set_page_config(page_title="IFC Görüntüleyici", layout="wide", page_icon="🔍")
st.title("🔍 IFC Görüntüleyici")
st.caption(
    "**Ana baseline** (paketin temsilcisi) · **Baseline** (paket içinden seçili biri) · "
    "**İhlalli** (seçili baseline'dan türeyen). Graph'ta bir node'a tıklarsan "
    "üç sütunda da aynı eleman vurgulanır."
)

root = get_dataset_root()
if not root:
    st.error("Dataset kökü tanımlı değil.")
    st.stop()

# --- Paket (ana baseline adı) seçimi -------------------------------------
all_baselines = list_entries(root, kind="baseline", require_graph=False)
if not all_baselines:
    st.warning(
        "Henüz baseline yok. Önce **🏠 Baseline Üretimi** sayfasıyla üret."
    )
    st.stop()


def _pkg_of(b: dict) -> str:
    tag = b.get("dataset_tag")
    if tag:
        return tag
    return re.sub(r"_\d+$", "", b.get("name") or "(isimsiz)") or "(isimsiz)"


packages = sorted({_pkg_of(b) for b in all_baselines})
pkg_counts = {p: sum(1 for b in all_baselines if _pkg_of(b) == p) for p in packages}

sel_pkg = st.selectbox(
    "📦 Ana baseline (paket)",
    options=packages,
    format_func=lambda p: f"{p}  ·  {pkg_counts.get(p, 0)} baseline",
    key="mv6_pkg",
)

pkg_baselines_all = sorted(
    [b for b in all_baselines if _pkg_of(b) == sel_pkg],
    key=lambda b: b.get("name") or "",
)
if not pkg_baselines_all:
    st.warning("Bu pakette baseline yok.")
    st.stop()

# Ana baseline = parent_id NULL olan baseline (LLM hiyerarşik üretimde).
# Fallback: parent_id'ye bakmadan paketin ilki (eski prosedürel üretim için).
_ana_candidates = [b for b in pkg_baselines_all if not b.get("parent_id")]
if _ana_candidates:
    ana_entry = _ana_candidates[0]
    # Varyantlar = ana'nın doğrudan çocukları
    pkg_baselines = [b for b in pkg_baselines_all
                     if b.get("parent_id") == ana_entry["id"]]
    if not pkg_baselines:
        # Hiyerarşi var ama varyant yok → col 2 = ana'nın kendisi
        pkg_baselines = [ana_entry]
    _hierarchy_mode = "llm"
else:
    # Eski prosedürel paket: hiyerarşi yok, hepsi peer.
    ana_entry = pkg_baselines_all[0]
    pkg_baselines = pkg_baselines_all
    _hierarchy_mode = "flat"

st.caption(
    f"🏛️ Ana baseline: `{ana_entry['name']}` · "
    f"{len(pkg_baselines)} varyant · "
    f"hiyerarşi: **{'LLM (parent_id ile)' if _hierarchy_mode == 'llm' else 'düz (eski prosedürel)'}**"
)

# --- Baseline seçici (col 2) ---------------------------------------------
st.markdown("#### Baseline (paket içi seçim)")
b_key = f"mv6_baseline_idx::{sel_pkg}"
cur_b = max(0, min(st.session_state.get(b_key, 0), len(pkg_baselines) - 1))

bcols = st.columns([1, 1, 6, 1])
if bcols[0].button("◀", key=f"{b_key}_prev", disabled=cur_b == 0,
                   help="Önceki baseline"):
    st.session_state[b_key] = cur_b - 1
    st.rerun()
if bcols[1].button("▶", key=f"{b_key}_next",
                   disabled=cur_b >= len(pkg_baselines) - 1,
                   help="Sonraki baseline"):
    st.session_state[b_key] = cur_b + 1
    st.rerun()
with bcols[2]:
    picked = st.selectbox(
        "Baseline",
        options=list(range(len(pkg_baselines))),
        index=cur_b,
        format_func=lambda i: (
            f"[{i + 1}/{len(pkg_baselines)}] "
            f"{pkg_baselines[i]['name']}  ·  {pkg_baselines[i]['id'][:8]}"
        ),
        key=f"{b_key}_sel",
        label_visibility="collapsed",
    )
    if picked != cur_b:
        st.session_state[b_key] = picked
        st.rerun()
bcols[3].metric("Sıra", f"{cur_b + 1}/{len(pkg_baselines)}")

baseline_entry = pkg_baselines[cur_b]

# --- İhlalli seçici (col 3) — seçili baseline'a göre filtrelenmiş --------
kids = violated_children(root, baseline_entry["id"])
st.markdown(
    f"#### İhlalli (`{baseline_entry['name']}` baseline'ından üretilen — "
    f"{len(kids)} adet)"
)
violated_entry: dict | None = None
if kids:
    v_key = f"mv6_violated_idx::{baseline_entry['id']}"
    cur_v = max(0, min(st.session_state.get(v_key, 0), len(kids) - 1))
    vcols = st.columns([1, 1, 6, 1])
    if vcols[0].button("◀", key=f"{v_key}_prev", disabled=cur_v == 0,
                       help="Önceki ihlalli"):
        st.session_state[v_key] = cur_v - 1
        st.rerun()
    if vcols[1].button("▶", key=f"{v_key}_next",
                       disabled=cur_v >= len(kids) - 1,
                       help="Sonraki ihlalli"):
        st.session_state[v_key] = cur_v + 1
        st.rerun()
    with vcols[2]:
        picked_v = st.selectbox(
            "İhlalli",
            options=list(range(len(kids))),
            index=cur_v,
            format_func=lambda i: (
                f"[{i + 1}/{len(kids)}] "
                f"{kids[i]['name']}  ·  {kids[i]['id'][:8]}"
            ),
            key=f"{v_key}_sel",
            label_visibility="collapsed",
        )
        if picked_v != cur_v:
            st.session_state[v_key] = picked_v
            st.rerun()
    vcols[3].metric("Sıra", f"{cur_v + 1}/{len(kids)}")
    violated_entry = kids[cur_v]
else:
    st.info("📭 Bu baseline'dan henüz ihlal üretilmemiş — col 3 boş gözükecek.")

# --- Etiket katmanları ---------------------------------------------------
st.markdown("#### Etiket katmanları")
ovc = st.columns([2, 2, 3, 3])
ov_vio = ovc[0].checkbox("🔴 İhlal", value=True, key="mv6_ov_vio")
ov_decoy = ovc[1].checkbox("🟡 Decoy", value=True, key="mv6_ov_decoy")
ov_normal = ovc[2].checkbox("🟢 Etiketli-uygun (clean)", value=False,
                            key="mv6_ov_normal")
if ovc[3].button("🧹 Seçimi temizle (cross-highlight)",
                 use_container_width=True):
    set_selected_node(None)
    st.rerun()

# --- Sample'ları yükle ---------------------------------------------------
ana_sample = load_sample_for(ana_entry)
baseline_sample = load_sample_for(baseline_entry)
violated_sample = load_sample_for(violated_entry) if violated_entry else None

# İhlal / decoy / normal kümeleri (sadece ihlalli sütunu için anlamlı)
if violated_sample:
    vio_set = {gid for gid, y in violated_sample.y.items() if y == 1}
    decoy_set = set(violated_sample.decoy_guids)
else:
    vio_set, decoy_set = set(), set()

normal_set: set[str] = set()
if violated_entry:
    doc = labels_summary(violated_entry) or {}
    for lab in doc.get("labels", []):
        g = lab.get("ifc_global_id")
        stt = (lab.get("status") or "").lower()
        if g and stt in ("compliant", "clean") and not lab.get("is_decoy"):
            normal_set.add(g)
    normal_set -= vio_set
    normal_set -= decoy_set

vio_show = vio_set if ov_vio else set()
dec_show = decoy_set if ov_decoy else set()
nor_show = normal_set if ov_normal else set()

# --- Seçili node (cross-highlight) ---------------------------------------
current = get_selected_node()
all_guids: set[str] = set()
for s in (ana_sample, baseline_sample, violated_sample):
    if s:
        all_guids.update(s.graph.nodes)
if current not in all_guids:
    current = None

# --- IFC mesh'leri -------------------------------------------------------
ana_meshes = _safe_meshes(ana_entry.get("ifc_path"))
baseline_meshes = _safe_meshes(baseline_entry.get("ifc_path"))
violated_meshes = _safe_meshes(violated_entry["ifc_path"]) if violated_entry else None

PANEL_H_3D = 380
PANEL_H_GR = 380


def _render_ifc(col, title: str, meshes, *,
                vio: set = set(), decoy: set = set(), normal: set = set(),
                key: str) -> None:
    with col:
        st.markdown(f"##### {title}")
        if meshes is None:
            st.info("📭 Veri üretilmemiş")
            return
        fig = build_figure(
            meshes, violation_guids=vio, decoy_guids=decoy,
            normal_guids=normal, selected_guid=current, height=PANEL_H_3D,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key}")


def _render_graph(col, title: str, sample, *,
                  vio: set = set(), decoy: set = set(), normal: set = set(),
                  key: str) -> None:
    with col:
        st.markdown(f"##### Graph · {title}")
        if sample is None:
            st.info("📭 Veri üretilmemiş")
            return
        clicked = interactive_agraph(
            sample.graph,
            violation_guids=vio, decoy_guids=decoy, normal_guids=normal,
            selected_guid=current, height=PANEL_H_GR, key=f"graph_{key}",
        )
        if clicked and clicked != current:
            set_selected_node(clicked)
            st.rerun()


# --- 3D row --------------------------------------------------------------
st.divider()
st.markdown("### 🧱 3D Görselleştirme")
c1, c2, c3 = st.columns(3)
_render_ifc(c1, f"Ana baseline · `{ana_entry['name']}`", ana_meshes,
            key="ana_ifc")
_render_ifc(c2, f"Baseline · `{baseline_entry['name']}`", baseline_meshes,
            key="bsl_ifc")
_render_ifc(c3,
            f"İhlalli · `{violated_entry['name']}`" if violated_entry
            else "İhlalli",
            violated_meshes, vio=vio_show, decoy=dec_show, normal=nor_show,
            key="vio_ifc")

# --- Graph row -----------------------------------------------------------
st.markdown("### 🕸️ Graph (node tıkla → vurgula · sürükle → yeniden düzenle)")
g1, g2, g3 = st.columns(3)
_render_graph(g1, "Ana baseline", ana_sample, key="ana_g")
_render_graph(g2, "Baseline", baseline_sample, key="bsl_g")
_render_graph(g3, "İhlalli", violated_sample,
              vio=vio_show, decoy=dec_show, normal=nor_show, key="vio_g")

# --- LLM prompt expander'ları (her sütun için) --------------------------
st.markdown("### 🤖 Üretim prompt'ları")
p1, p2, p3 = st.columns(3)
with p1:
    _ifc_prompt_panel(ana_entry, key="ana baseline")
with p2:
    _ifc_prompt_panel(baseline_entry, key="baseline")
with p3:
    _ifc_prompt_panel(violated_entry, key="ihlalli")

# --- Inspector -----------------------------------------------------------
if current:
    src_g = None
    for s in (violated_sample, baseline_sample, ana_sample):
        if s and current in s.graph.nodes:
            src_g = s.graph
            break
    if src_g is not None:
        nd = src_g.nodes[current]
        st.divider()
        st.subheader(f"🔎 {nd.get('ifc_type', '?')} — `{current}`")
        ic = st.columns(3)
        with ic[0]:
            st.markdown("**Attributes**")
            st.json(nd.get("attributes") or {}, expanded=False)
        with ic[1]:
            st.markdown("**Psets**")
            st.json(nd.get("psets") or {}, expanded=False)
        with ic[2]:
            st.markdown("**Etiketler**")
            is_v = current in vio_set
            is_d = current in decoy_set
            st.write("• İhlal mi?", "✅ Evet" if is_v else "—")
            st.write("• Decoy mi?", "🪤 Evet" if is_d else "—")
            if violated_entry:
                doc = labels_summary(violated_entry) or {}
                match = [l for l in doc.get("labels", [])
                         if l.get("ifc_global_id") == current]
                if match:
                    lab = match[0]
                    if lab.get("attribute"):
                        st.write(
                            f"`{lab['attribute']}`: "
                            f"{lab.get('before') if 'before' in lab else lab.get('value_before')}"
                            f" → "
                            f"{lab.get('after') if 'after' in lab else lab.get('value_after')}"
                        )
                    with st.expander("Tam etiket"):
                        st.json(lab, expanded=False)
