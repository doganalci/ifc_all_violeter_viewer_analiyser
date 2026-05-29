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
import os
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
        # Pure LLM IFC'lerde tipik: parse OK ama geometry kernel patlar.
        err = str(e)
        if any(s in err.lower() for s in (
                "variant", "out of range", "step parsing", "geometry")):
            st.error(
                f"❌ Geometri çıkarılamadı: `{Path(ifc_path).name}`\n\n"
                f"Hata: `{err[:180]}`\n\n"
                "**Bu pure-LLM IFC'lerinde tipiktir** — dosya parse oluyor "
                "ama geometry kernel iç tutarsızlık görüyor (yanlış variant "
                "indexi, eksik placement, bozuk representation vs.). Hibrit "
                "(sayfa 10) IFC'leri tessellate olur. Aşağıda entity sayıları "
                "fallback olarak görünür."
            )
        else:
            st.error(f"IFC açılamadı ({Path(ifc_path).name}): {e}")
        return None


def _ifc_entity_summary(ifc_path: str | None) -> dict | None:
    """3D çıkmazsa fallback için entity sayıları + örnek isimler."""
    if not ifc_path or not Path(ifc_path).exists():
        return None
    try:
        import ifcopenshell
        f = ifcopenshell.open(str(ifc_path))
    except Exception:
        return None
    out = {
        "Walls (IfcWall+StandardCase)":
            len(f.by_type("IfcWall")) + len(f.by_type("IfcWallStandardCase")),
        "Doors (IfcDoor)": len(f.by_type("IfcDoor")),
        "Windows (IfcWindow)": len(f.by_type("IfcWindow")),
        "Spaces (IfcSpace)": len(f.by_type("IfcSpace")),
        "Storey (IfcBuildingStorey)": len(f.by_type("IfcBuildingStorey")),
        "Slabs (IfcSlab)": len(f.by_type("IfcSlab")),
    }
    # İlk birkaç entity adı
    sample_names = []
    for cls in ("IfcDoor", "IfcWall", "IfcSpace"):
        for e in f.by_type(cls)[:3]:
            nm = getattr(e, "Name", None) or "(isimsiz)"
            sample_names.append(f"{cls}: {nm}")
    return {"counts": out, "samples": sample_names}


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

    Sıralama dışarıdan uygulanır (_apply_sort).
    """
    direct = violated_children(root, ana_id)
    all_bls = list_entries(root, kind="baseline", require_graph=False)
    variants = [b for b in all_bls if b.get("parent_id") == ana_id]
    indirect = []
    for v in variants:
        indirect.extend(violated_children(root, v["id"]))
    return direct + indirect


# ---- Sıralama yardımcıları --------------------------------------------------

SORT_OPTIONS = {
    "📅 Üretim zamanı (yeni → eski)": "time_desc",
    "📅 Üretim zamanı (eski → yeni)": "time_asc",
    "🔤 Alfabetik (A → Z)": "alpha_asc",
    "🔤 Alfabetik (Z → A)": "alpha_desc",
}


def _mtime_of(p: str | None) -> float:
    """Dosya değişiklik zamanı (epoch). Yoksa 0."""
    if not p:
        return 0.0
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _entry_mtime(e: dict) -> float:
    """Bir IFC entry'sinin temsili mtime'ı (IFC > graph > labels)."""
    for k in ("ifc_path", "graph_path", "labels_path", "meta_path"):
        t = _mtime_of(e.get(k))
        if t > 0:
            return t
    return 0.0


def _apply_sort(entries: list[dict], sort_key: str) -> list[dict]:
    """Verilen IFC entry listesini sort_key'e göre sırala."""
    if sort_key == "time_desc":
        return sorted(entries, key=lambda e: -_entry_mtime(e))
    if sort_key == "time_asc":
        return sorted(entries, key=_entry_mtime)
    if sort_key == "alpha_desc":
        return sorted(entries, key=lambda e: (e.get("name") or "").lower(),
                      reverse=True)
    # default alpha_asc
    return sorted(entries, key=lambda e: (e.get("name") or "").lower())


def _sort_packages(packages: list[str], all_baselines: list[dict],
                    pkg_of_fn, sort_key: str) -> list[str]:
    """Paket adlarını sort_key'e göre sırala.

    Zaman tabanlı sıralama: paketin en yeni baseline'ının mtime'ı.
    """
    if sort_key in ("time_desc", "time_asc"):
        def _pkg_latest(pkg: str) -> float:
            return max(
                (_entry_mtime(b) for b in all_baselines
                 if pkg_of_fn(b) == pkg),
                default=0.0,
            )
        rev = (sort_key == "time_desc")
        return sorted(packages, key=_pkg_latest, reverse=rev)
    rev = (sort_key == "alpha_desc")
    return sorted(packages, key=lambda p: p.lower(), reverse=rev)


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


# --- Sıralama (tüm listelere uygulanır) ----------------------------------
sc1, sc2 = st.columns([3, 1])
with sc1:
    _sort_label = st.selectbox(
        "🔀 Sıralama",
        options=list(SORT_OPTIONS.keys()),
        index=0,
        help="Paketleri, baseline'ları ve ihlallileri bu sıraya göre listele. "
             "Default: yeni üretilenler üstte.",
        key="mv_sort",
    )
with sc2:
    if st.button("🔄 Yenile (cache'i atla)", use_container_width=True,
                 help="DB yeniden okunsun (yeni üretilen bir şey görünmüyorsa)"):
        st.cache_data.clear()
        st.rerun()
sort_key = SORT_OPTIONS[_sort_label]

packages_set = {_pkg_of(b) for b in all_baselines}
pkg_counts = {p: sum(1 for b in all_baselines if _pkg_of(b) == p)
              for p in packages_set}
packages = _sort_packages(list(packages_set), all_baselines, _pkg_of, sort_key)

sel_pkg = st.selectbox(
    "📦 Ana baseline (paket)",
    options=packages,
    format_func=lambda p: f"{p}  ·  {pkg_counts.get(p, 0)} baseline",
    key="mv_pkg",
)

pkg_baselines = _apply_sort(
    [b for b in all_baselines if _pkg_of(b) == sel_pkg], sort_key,
)
_ana_candidates = [b for b in pkg_baselines if not b.get("parent_id")]
if _ana_candidates:
    # Sort'a göre seç: yeni → eski ise en yeni ana baseline col 1'de
    ana_entry = _ana_candidates[0]
    _hier_mode = "LLM (parent_id ile)"
else:
    ana_entry = pkg_baselines[0]
    _hier_mode = "düz (eski prosedürel)"

st.caption(
    f"🏛️ Ana baseline: `{ana_entry['name']}` · hiyerarşi: **{_hier_mode}**"
)

# --- İhlalli seçici -----------------------------------------------------
all_violateds = _apply_sort(
    _all_violateds_under_ana(root, ana_entry["id"]), sort_key,
)
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
# State formatı: None | "<col>_both" | "<col>_3d" | "<col>_graph"
# col ∈ {"ana", "vio", "pred"}
st.markdown("#### Görünüm")
mxd = st.session_state.get("mv_maximized", None)
if mxd is None:
    bc = st.columns(3)
    if bc[0].button("⛶ Ana baseline büyüt (3D + graph)", use_container_width=True):
        st.session_state["mv_maximized"] = "ana_both"
        st.rerun()
    if bc[1].button("⛶ İhlalli büyüt (3D + graph)", use_container_width=True):
        st.session_state["mv_maximized"] = "vio_both"
        st.rerun()
    if bc[2].button("⛶ Tahmin büyüt (3D + graph)", use_container_width=True):
        st.session_state["mv_maximized"] = "pred_both"
        st.rerun()
else:
    col_name, mode = mxd.split("_", 1)
    bc = st.columns(4)
    if mode == "both":
        if bc[0].button("⛶ Sadece 3D büyüt", use_container_width=True):
            st.session_state["mv_maximized"] = f"{col_name}_3d"
            st.rerun()
        if bc[1].button("⛶ Sadece Graph büyüt", use_container_width=True):
            st.session_state["mv_maximized"] = f"{col_name}_graph"
            st.rerun()
        if bc[2].button("🔲 Küçült (3 sütun grid)", use_container_width=True,
                        type="primary"):
            st.session_state["mv_maximized"] = None
            st.rerun()
    else:
        if bc[0].button("⬜ Stack moduna dön (3D + graph)",
                        use_container_width=True):
            st.session_state["mv_maximized"] = f"{col_name}_both"
            st.rerun()
        if bc[1].button("🔲 Küçült (3 sütun grid)", use_container_width=True,
                        type="primary"):
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

# Panel boyutları
PANEL_H_3D_GRID = 380
PANEL_H_GR_GRID = 380
# "both" mod — iki panel stacked, biraz büyük
PANEL_H_3D_BOTH = 520
PANEL_H_GR_BOTH = 480
# Tekli panel modu — çok büyük
PANEL_H_3D_FULL = 760
PANEL_H_GR_FULL = 700


def _panel_max_btn(target_state: str, btn_key: str) -> None:
    """Panel başlığındaki ⛶/🔲 buton — büyütme toggle.

    target_state: bu butonun maksimize ettiği state ("ana_3d", "vio_graph" vs.)
    """
    cur = st.session_state.get("mv_maximized")
    label = "🔲" if cur == target_state else "⛶"
    if st.button(label, key=btn_key,
                 help=("Bu paneli küçült" if cur == target_state
                       else "Bu paneli tek başına büyüt")):
        st.session_state["mv_maximized"] = (
            None if cur == target_state else target_state)
        st.rerun()


def _render_ifc(col, title: str, meshes, *,
                vio: set = set(), decoy: set = set(), normal: set = set(),
                pred: set = set(), height: int = PANEL_H_3D_GRID,
                key: str, max_state: str | None = None,
                fallback_ifc_path: str | None = None) -> None:
    with col:
        # Başlık + büyütme butonu
        if max_state:
            hc = st.columns([7, 1])
            hc[0].markdown(f"##### {title}")
            with hc[1]:
                _panel_max_btn(max_state, f"maxbtn_{key}")
        else:
            st.markdown(f"##### {title}")
        if meshes is None:
            # Geometri çıkmadı — fallback: entity summary
            if fallback_ifc_path:
                summary = _ifc_entity_summary(fallback_ifc_path)
                if summary:
                    st.warning(
                        "📊 3D çizilemedi → entity özeti (fallback)"
                    )
                    st.json(summary["counts"], expanded=True)
                    if summary["samples"]:
                        with st.expander("İlk entity'ler"):
                            for s in summary["samples"]:
                                st.code(s, language="text")
                    return
            st.info("📭 Gösterilecek veri yok")
            return
        merged_vio = vio | pred
        fig = build_figure(
            meshes, violation_guids=merged_vio, decoy_guids=decoy,
            normal_guids=normal, selected_guid=current, height=height,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ifc_{key}")


def _render_graph(col, title: str, sample, *,
                  vio: set = set(), decoy: set = set(), normal: set = set(),
                  pred: set = set(), height: int = PANEL_H_GR_GRID,
                  key: str, max_state: str | None = None,
                  ifc_path: str | None = None,
                  graph_path: str | None = None) -> None:
    with col:
        # Başlık + büyütme butonu
        if max_state:
            hc = st.columns([7, 1])
            hc[0].markdown(f"##### Graph · {title}")
            with hc[1]:
                _panel_max_btn(max_state, f"maxbtn_{key}")
        else:
            st.markdown(f"##### Graph · {title}")
        if sample is None:
            st.info("📭 Gösterilecek veri yok")
            # IFC var ama graph yoksa "Graph üret" butonu
            if ifc_path and graph_path:
                if st.button("🔧 Graph'ı yeniden üret",
                             key=f"rebuild_{key}",
                             help="ifc_graph.build_and_save'i tekrar çalıştır "
                                  "(yeni fix dahil)."):
                    try:
                        from violation_pool import ifc_graph
                        ifc_graph.build_and_save(ifc_path, graph_path)
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Graph üretim hatası: {e}")
            return
        n_nodes = len(sample.graph.nodes)
        # Diagnostic: eğer çok az node var ama IFC zengin → pure-LLM eksik
        # GUID/RelAggregates sorunu olabilir.
        if n_nodes < 5 and ifc_path:
            summary = _ifc_entity_summary(ifc_path)
            if summary:
                total_entities = sum(summary["counts"].values())
                if total_entities > n_nodes + 3:
                    st.caption(
                        f"ℹ️ Graph {n_nodes} node ama IFC {total_entities} "
                        "entity içeriyor. **Pure-LLM eksik GlobalId/IfcRel\\*** "
                        "sorunu — graph yapısı zayıf. 🔧 ile yeniden üret."
                    )
                    if st.button("🔧 Graph'ı yeniden üret",
                                 key=f"rebuild_sparse_{key}",
                                 help="ifc_graph fix'i ile orphan entity'leri "
                                      "de node olarak ekle"):
                        try:
                            from violation_pool import ifc_graph as _ig
                            _ig.build_and_save(ifc_path, graph_path)
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Graph üretim hatası: {e}")
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

# Pre-compute tahmin başlığı
pred_title = (f"Tahmin · `{cached.get('run')}` · eşik {cached.get('threshold')}"
              if cached else "Tahmin")
vio_title = (f"İhlalli · `{violated_entry['name']}`" if violated_entry
             else "İhlalli")

# Sütun argümanları — render fonksiyonlarına gönderilecek
_col_args = {
    "ana": {
        "title": f"Ana baseline · `{ana_entry['name']}`",
        "ifc_meshes": ana_meshes, "sample": ana_sample,
        "ifc_path": ana_entry.get("ifc_path"),
        "graph_path": ana_entry.get("graph_path"),
        "vio": set(), "decoy": set(), "normal": set(),
        "pred": set(),
    },
    "vio": {
        "title": vio_title,
        "ifc_meshes": violated_meshes, "sample": violated_sample,
        "ifc_path": violated_entry["ifc_path"] if violated_entry else None,
        "graph_path": violated_entry["graph_path"] if violated_entry else None,
        "vio": vio_show, "decoy": dec_show, "normal": nor_show,
        "pred": set(),
    },
    "pred": {
        "title": pred_title,
        "ifc_meshes": pred_meshes, "sample": pred_sample,
        "ifc_path": violated_entry["ifc_path"] if violated_entry else None,
        "graph_path": violated_entry["graph_path"] if violated_entry else None,
        "vio": set(), "decoy": set(), "normal": set(),
        "pred": predicted_guids if cached else set(),
    },
}


def _render_col_panel(c: dict, *, panel_key: str, mode: str) -> None:
    """Tek bir sütunun (col_name) ilgili modda 3D + graph render."""
    if mode == "both":
        _render_ifc(st, c["title"], c["ifc_meshes"],
                    vio=c["vio"], decoy=c["decoy"], normal=c["normal"],
                    pred=c["pred"], height=PANEL_H_3D_BOTH,
                    key=f"{panel_key}_3d", max_state=f"{panel_key}_3d",
                    fallback_ifc_path=c.get("ifc_path"))
        _render_graph(st, c["title"], c["sample"],
                      vio=c["vio"], decoy=c["decoy"], normal=c["normal"],
                      pred=c["pred"], height=PANEL_H_GR_BOTH,
                      key=f"{panel_key}_g", max_state=f"{panel_key}_graph",
                      ifc_path=c.get("ifc_path"),
                      graph_path=c.get("graph_path"))
    elif mode == "3d":
        _render_ifc(st, c["title"], c["ifc_meshes"],
                    vio=c["vio"], decoy=c["decoy"], normal=c["normal"],
                    pred=c["pred"], height=PANEL_H_3D_FULL,
                    key=f"{panel_key}_3d_full",
                    fallback_ifc_path=c.get("ifc_path"))
    elif mode == "graph":
        _render_graph(st, c["title"], c["sample"],
                      vio=c["vio"], decoy=c["decoy"], normal=c["normal"],
                      pred=c["pred"], height=PANEL_H_GR_FULL,
                      key=f"{panel_key}_g_full",
                      ifc_path=c.get("ifc_path"),
                      graph_path=c.get("graph_path"))


if mxd is not None:
    col_name, mode = mxd.split("_", 1)
    panel_title = {"ana": "🏛️ Ana baseline",
                   "vio": "💥 İhlalli",
                   "pred": "🤖 Tahmin"}.get(col_name, col_name)
    mode_label = {"both": "3D + Graph", "3d": "sadece 3D",
                  "graph": "sadece Graph"}.get(mode, mode)
    st.markdown(f"### {panel_title} ({mode_label})")
    _render_col_panel(_col_args[col_name], panel_key=col_name, mode=mode)
else:
    # 3D row
    st.markdown("### 🧱 3D Görselleştirme")
    c1, c2, c3 = st.columns(3)
    _render_ifc(c1, _col_args["ana"]["title"], _col_args["ana"]["ifc_meshes"],
                key="ana_ifc_grid", max_state="ana_3d",
                fallback_ifc_path=_col_args["ana"].get("ifc_path"))
    _render_ifc(c2, _col_args["vio"]["title"], _col_args["vio"]["ifc_meshes"],
                vio=vio_show, decoy=dec_show, normal=nor_show,
                key="vio_ifc_grid", max_state="vio_3d",
                fallback_ifc_path=_col_args["vio"].get("ifc_path"))
    _render_ifc(c3, _col_args["pred"]["title"],
                _col_args["pred"]["ifc_meshes"],
                pred=predicted_guids if cached else set(),
                key="pred_ifc_grid", max_state="pred_3d",
                fallback_ifc_path=_col_args["pred"].get("ifc_path"))

    # Graph row
    st.markdown("### 🕸️ Graph (node tıkla → vurgula · sürükle → düzenle)")
    g1, g2, g3 = st.columns(3)
    _render_graph(g1, _col_args["ana"]["title"], ana_sample,
                  key="ana_g_grid", max_state="ana_graph",
                  ifc_path=_col_args["ana"].get("ifc_path"),
                  graph_path=_col_args["ana"].get("graph_path"))
    _render_graph(g2, _col_args["vio"]["title"], violated_sample,
                  vio=vio_show, decoy=dec_show, normal=nor_show,
                  key="vio_g_grid", max_state="vio_graph",
                  ifc_path=_col_args["vio"].get("ifc_path"),
                  graph_path=_col_args["vio"].get("graph_path"))
    _render_graph(g3, _col_args["pred"]["title"], pred_sample,
                  pred=predicted_guids if cached else set(),
                  key="pred_g_grid", max_state="pred_graph",
                  ifc_path=_col_args["pred"].get("ifc_path"),
                  graph_path=_col_args["pred"].get("graph_path"))

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
