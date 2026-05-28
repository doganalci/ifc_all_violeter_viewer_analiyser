"""IFC Görüntüleyici — 3 sütun (Ana baseline | İhlalli | Tahmin) × 2 satır (3D + graph).

Akış:
  * Üstte paket (ana baseline) seçilir → ana baseline col 1'de sabit.
  * Col 1 = **Ana baseline** (paketin parent_id=None baseline'ı, sabit).
  * Col 2 = **İhlalli** — ana'ya transitively bağlı tüm violateds. ◀/▶.
  * Col 3 = **Tahmin** — eğitilmiş GAT modelinin İhlalli üzerinde tahmini.
    Model seçici + eşik slider + "Tahmin Yap" butonu.
  * Üst sıra: 3D IFC render (Plotly Mesh3d).
  * Alt sıra: interaktif graph (streamlit-agraph: node sürükle, tıkla).
  * Graph'ta tıklama → üç sütunda da aynı GUID mavi vurgulanır.
  * Her sütunun başlığında ⛶ Büyüt: o sütun (3D + graph) tam genişlik,
    diğer ikisi saklı. Tekrar bas → grid'e döner.

Boş veri:
  * Ana baseline yoksa: "Gösterilecek veri yok"
  * Bu ana'ya ait İhlalli yoksa: col 2 = "Gösterilecek veri yok"
  * Tahmin yapılmadıysa: col 3 = "Gösterilecek veri yok"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    entry_by_id, get_dataset_root, get_selected_node, labels_summary,
    list_entries, load_sample_for, set_selected_node, violated_children,
)
from ml.viz.graph_view import interactive_agraph
from ml.viz.ifc3d import build_figure, extract_meshes
from violation_pool import storage


# ---- IFC tessellation cache ------------------------------------------------

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


# ---- Model inference helpers (legacy/03'ten) -------------------------------

def _list_runs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted([p for p in run_dir.iterdir()
                   if p.is_dir() and (p / "best.pt").exists()])


def _load_model(cfg, in_dim: int, num_edge_types: int):
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


def _run_inference(sample, run_path: Path, threshold: float):
    """İhlalli sample üzerinde GAT inference; (predicted_guids, info) döner."""
    from ml.train.config import TrainConfig
    from ml.data.graph_loader import sample_to_data
    import torch

    cfg_path = run_path / "config.json"
    ckpt_path = run_path / "best.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise RuntimeError("Run klasöründe config.json veya best.pt yok.")
    cfg = TrainConfig(**json.loads(cfg_path.read_text()))
    data = sample_to_data(sample)
    device = cfg.resolve_device()
    nt = int(data.edge_type.max().item()) + 1 if data.edge_type.numel() else 7
    model = _load_model(cfg, data.x.shape[1], nt)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device),
                       data.edge_type.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)
    predicted = {data.node_ids[i] for i, p in enumerate(preds) if p == 1}
    return predicted, {
        "n_predicted": int(preds.sum()),
        "threshold": threshold,
        "run_name": run_path.name,
    }


def _all_violateds_under_ana(root: str, ana_id: str) -> list[dict]:
    """Ana baseline'a transitively bağlı tüm violated'ları getir.

    Yapı: ana → varyant baseline → violated. Ana'ya doğrudan + varyantlara
    bağlı tüm violated'ları topla (varyantlar gizli, hepsi tek listede).
    """
    direct = violated_children(root, ana_id)
    # Ana'nın çocuk baseline'ları (varyantlar)
    all_bls = list_entries(root, kind="baseline", require_graph=False)
    variants = [b for b in all_bls if b.get("parent_id") == ana_id]
    indirect = []
    for v in variants:
        indirect.extend(violated_children(root, v["id"]))
    out = direct + indirect
    return sorted(out, key=lambda x: x.get("name") or "")


# ---- Page ------------------------------------------------------------------

st.set_page_config(page_title="IFC Görüntüleyici", layout="wide", page_icon="🔍")
st.title("🔍 IFC Görüntüleyici")
st.caption(
    "**Ana baseline** · **İhlalli** · **Tahmin** (GAT modeli). "
    "Graph'ta tıkla → üç sütunda da vurgula. Sütun başlığında **⛶ Büyüt** ile "
    "ekranı kapla."
)

root = get_dataset_root()
if not root:
    st.error("Dataset kökü tanımlı değil.")
    st.stop()

# --- Paket seçimi -------------------------------------------------------
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
pkg_counts = {p: sum(1 for b in all_baselines if _pkg_of(b) == p)
              for p in packages}

sel_pkg = st.selectbox(
    "📦 Ana baseline (paket)",
    options=packages,
    format_func=lambda p: f"{p}  ·  {pkg_counts.get(p, 0)} baseline",
    key="mv_pkg",
)

pkg_baselines = [b for b in all_baselines if _pkg_of(b) == sel_pkg]
_ana_candidates = [b for b in pkg_baselines if not b.get("parent_id")]
if _ana_candidates:
    ana_entry = _ana_candidates[0]
    _hier_mode = "LLM (parent_id ile)"
else:
    ana_entry = sorted(pkg_baselines, key=lambda b: b.get("name") or "")[0]
    _hier_mode = "düz (eski prosedürel)"

st.caption(
    f"🏛️ Ana baseline: `{ana_entry['name']}` · hiyerarşi: **{_hier_mode}**"
)

# --- İhlalli seçici -----------------------------------------------------
all_violateds = _all_violateds_under_ana(root, ana_entry["id"])
violated_entry: dict | None = None
st.markdown(f"#### İhlalli ({len(all_violateds)} adet — ana'ya transitively bağlı)")
if all_violateds:
    v_key = f"mv_violated::{ana_entry['id']}"
    cur_v = max(0, min(st.session_state.get(v_key, 0), len(all_violateds) - 1))
    vcols = st.columns([1, 1, 6, 1])
    if vcols[0].button("◀", key=f"{v_key}_p", disabled=cur_v == 0):
        st.session_state[v_key] = cur_v - 1
        st.rerun()
    if vcols[1].button("▶", key=f"{v_key}_n",
                       disabled=cur_v >= len(all_violateds) - 1):
        st.session_state[v_key] = cur_v + 1
        st.rerun()
    with vcols[2]:
        picked = st.selectbox(
            "İhlalli", options=list(range(len(all_violateds))),
            index=cur_v,
            format_func=lambda i: (
                f"[{i + 1}/{len(all_violateds)}] "
                f"{all_violateds[i]['name']} · "
                f"{all_violateds[i]['id'][:8]}"),
            key=f"{v_key}_sel", label_visibility="collapsed",
        )
        if picked != cur_v:
            st.session_state[v_key] = picked
            st.rerun()
    vcols[3].metric("Sıra", f"{cur_v + 1}/{len(all_violateds)}")
    violated_entry = all_violateds[cur_v]
else:
    st.info("📭 Bu pakette ihlalli IFC yok.")

# --- Tahmin (model seçimi + inference) -----------------------------------
st.markdown("#### 🤖 Tahmin")
predicted_guids: set[str] = set()
pred_info: dict = {}

run_root = Path("runs")
runs = _list_runs(run_root)
mc = st.columns([3, 1, 1])
with mc[0]:
    if runs:
        sel_run_idx = st.selectbox(
            "Eğitim run (GAT modeli)",
            options=list(range(len(runs))),
            format_func=lambda i: runs[i].name,
            key="mv_run",
        )
        sel_run = runs[sel_run_idx]
    else:
        st.warning("Henüz eğitilmiş model yok. Tahmin yapılamaz.")
        sel_run = None
with mc[1]:
    threshold = st.slider("Eşik", 0.0, 1.0, 0.5, 0.05, key="mv_th")
with mc[2]:
    do_predict = st.button("🤖 Tahmin Yap", type="primary",
                           disabled=(sel_run is None or violated_entry is None),
                           use_container_width=True)

# Cache son tahmini session state'te tut (ihlalli + run + eşik aynı kalırsa)
pred_cache_key = f"mv_pred::{violated_entry['id'] if violated_entry else ''}"
if do_predict and violated_entry and sel_run:
    vio_sample = load_sample_for(violated_entry)
    if vio_sample is None:
        st.error("İhlalli IFC'nin graph'ı yok — sample yüklenemedi.")
    else:
        try:
            with st.spinner("Model çalıştırılıyor..."):
                pg, info = _run_inference(vio_sample, sel_run, threshold)
            st.session_state[pred_cache_key] = {
                "predicted": list(pg),
                "info": info,
                "run": sel_run.name,
                "threshold": threshold,
            }
        except Exception as e:
            st.error(f"Tahmin hatası: {e}")

cached = st.session_state.get(pred_cache_key)
if cached:
    predicted_guids = set(cached.get("predicted", []))
    pred_info = cached.get("info", {})
    st.caption(
        f"✓ Son tahmin: **{cached.get('run')}** · eşik {cached.get('threshold')} "
        f"· {pred_info.get('n_predicted', 0)} ihlal tahmini"
    )

# --- Etiket katmanları ---------------------------------------------------
st.markdown("#### Etiket katmanları")
ovc = st.columns([2, 2, 3, 3])
ov_vio = ovc[0].checkbox("🔴 Gerçek ihlal", value=True, key="mv_ov_vio")
ov_decoy = ovc[1].checkbox("🟡 Decoy", value=True, key="mv_ov_decoy")
ov_normal = ovc[2].checkbox("🟢 Etiketli-uygun (clean)", value=False,
                            key="mv_ov_normal")
if ovc[3].button("🧹 Seçimi temizle (cross-highlight)",
                 use_container_width=True):
    set_selected_node(None)
    st.rerun()

# --- Büyütme kontrolü ---------------------------------------------------
st.markdown("#### Görünüm")
bc = st.columns(4)
mxd = st.session_state.get("mv_maximized", None)  # None | "ana" | "vio" | "pred"
if bc[0].button("⛶ Ana baseline büyüt", use_container_width=True,
                disabled=(mxd == "ana")):
    st.session_state["mv_maximized"] = "ana"
    st.rerun()
if bc[1].button("⛶ İhlalli büyüt", use_container_width=True,
                disabled=(mxd == "vio")):
    st.session_state["mv_maximized"] = "vio"
    st.rerun()
if bc[2].button("⛶ Tahmin büyüt", use_container_width=True,
                disabled=(mxd == "pred")):
    st.session_state["mv_maximized"] = "pred"
    st.rerun()
if bc[3].button("🔲 Küçült (3 sütun)", use_container_width=True,
                disabled=(mxd is None)):
    st.session_state["mv_maximized"] = None
    st.rerun()

# --- Sample'ları yükle --------------------------------------------------
ana_sample = load_sample_for(ana_entry)
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

# --- Seçili node (cross-highlight) --------------------------------------
current = get_selected_node()
all_guids: set[str] = set()
for s in (ana_sample, violated_sample):
    if s:
        all_guids.update(s.graph.nodes)
if current not in all_guids:
    current = None

# --- IFC mesh'leri ------------------------------------------------------
ana_meshes = _safe_meshes(ana_entry.get("ifc_path"))
violated_meshes = _safe_meshes(violated_entry["ifc_path"]) if violated_entry else None
# Tahmin sütunu için: ihlalli'nin IFC + graph'ı kullanılır, ama vurgu predicted_guids
pred_meshes = violated_meshes
pred_sample = violated_sample

PANEL_H_3D_GRID = 380
PANEL_H_3D_FULL = 720
PANEL_H_GR_GRID = 380
PANEL_H_GR_FULL = 600


def _render_ifc(col, title: str, meshes, *,
                vio: set = set(), decoy: set = set(), normal: set = set(),
                pred: set = set(), height: int = PANEL_H_3D_GRID,
                key: str) -> None:
    with col:
        st.markdown(f"##### {title}")
        if meshes is None:
            st.info("📭 Gösterilecek veri yok")
            return
        # Tahmin sütununda predicted_guids "yeşil" yerine biz ona ihlal rengi
        # (kırmızı) gösteriyoruz; bu daha sezgisel: "model bunu ihlal sayıyor".
        # build_figure violation_guids=kırmızı/parlak; pred için onu kullanıyoruz.
        merged_vio = vio | pred
        fig = build_figure(
            meshes, violation_guids=merged_vio, decoy_guids=decoy,
            normal_guids=normal, selected_guid=current, height=height,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key}")


def _render_graph(col, title: str, sample, *,
                  vio: set = set(), decoy: set = set(), normal: set = set(),
                  pred: set = set(), height: int = PANEL_H_GR_GRID,
                  key: str) -> None:
    with col:
        st.markdown(f"##### Graph · {title}")
        if sample is None:
            st.info("📭 Gösterilecek veri yok")
            return
        merged_vio = vio | pred
        clicked = interactive_agraph(
            sample.graph,
            violation_guids=merged_vio, decoy_guids=decoy,
            normal_guids=normal, selected_guid=current,
            height=height, key=f"graph_{key}",
        )
        if clicked and clicked != current:
            set_selected_node(clicked)
            st.rerun()


# --- Render based on maximized state -----------------------------------
st.divider()
mxd = st.session_state.get("mv_maximized", None)

if mxd == "ana":
    st.markdown("### 🏛️ Ana baseline (büyütülmüş)")
    _render_ifc(st, f"Ana baseline · `{ana_entry['name']}`", ana_meshes,
                height=PANEL_H_3D_FULL, key="ana_full")
    _render_graph(st, "Ana baseline", ana_sample,
                  height=PANEL_H_GR_FULL, key="ana_full_g")
elif mxd == "vio":
    st.markdown("### 💥 İhlalli (büyütülmüş)")
    _render_ifc(st,
                f"İhlalli · `{violated_entry['name'] if violated_entry else '—'}`",
                violated_meshes, vio=vio_show, decoy=dec_show, normal=nor_show,
                height=PANEL_H_3D_FULL, key="vio_full")
    _render_graph(st, "İhlalli", violated_sample,
                  vio=vio_show, decoy=dec_show, normal=nor_show,
                  height=PANEL_H_GR_FULL, key="vio_full_g")
elif mxd == "pred":
    st.markdown("### 🤖 Tahmin (büyütülmüş)")
    title = (f"Tahmin · run=`{cached.get('run')}`" if cached
             else "Tahmin (henüz çalıştırılmadı)")
    _render_ifc(st, title, pred_meshes,
                pred=predicted_guids,
                height=PANEL_H_3D_FULL, key="pred_full")
    _render_graph(st, title, pred_sample,
                  pred=predicted_guids,
                  height=PANEL_H_GR_FULL, key="pred_full_g")
else:
    # --- 3D row (3 sütun) ----------------------------------------------
    st.markdown("### 🧱 3D Görselleştirme")
    c1, c2, c3 = st.columns(3)
    _render_ifc(c1, f"Ana baseline · `{ana_entry['name']}`", ana_meshes,
                key="ana_ifc")
    _render_ifc(c2,
                f"İhlalli · `{violated_entry['name']}`" if violated_entry
                else "İhlalli",
                violated_meshes, vio=vio_show, decoy=dec_show, normal=nor_show,
                key="vio_ifc")
    pred_title = (f"Tahmin · `{cached.get('run')}` · eşik {cached.get('threshold')}"
                  if cached else "Tahmin")
    _render_ifc(c3, pred_title, pred_meshes,
                pred=predicted_guids if cached else set(),
                key="pred_ifc")

    # --- Graph row -----------------------------------------------------
    st.markdown("### 🕸️ Graph (node tıkla → vurgula · sürükle → düzenle)")
    g1, g2, g3 = st.columns(3)
    _render_graph(g1, "Ana baseline", ana_sample, key="ana_g")
    _render_graph(g2, "İhlalli", violated_sample,
                  vio=vio_show, decoy=dec_show, normal=nor_show, key="vio_g")
    _render_graph(g3, pred_title, pred_sample,
                  pred=predicted_guids if cached else set(), key="pred_g")

# --- Inspector ----------------------------------------------------------
if current:
    src_g = None
    for s in (violated_sample, ana_sample):
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
            is_p = current in predicted_guids
            st.write("• Gerçek ihlal?", "✅" if is_v else "—")
            st.write("• Decoy?", "🪤" if is_d else "—")
            st.write("• Model tahmini ihlal?", "🤖✅" if is_p else "—")
            if violated_entry:
                doc = labels_summary(violated_entry) or {}
                match = [l for l in doc.get("labels", [])
                         if l.get("ifc_global_id") == current]
                if match:
                    lab = match[0]
                    if lab.get("attribute"):
                        bef = lab.get("before") or lab.get("value_before")
                        aft = lab.get("after") or lab.get("value_after")
                        st.write(f"`{lab['attribute']}`: {bef} → {aft}")
                    with st.expander("Tam etiket"):
                        st.json(lab, expanded=False)
