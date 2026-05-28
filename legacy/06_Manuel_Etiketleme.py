"""Manuel etiketleme: altın standart test seti üretme.

Akış:
  1. Sidebar'dan dataset root + listeden bir IFC seç
  2. Sol: IFC 3D (mavi = aktif node) · Sağ: tıklanabilir grafik
  3. Altta aktif node paneli:
       ✓ İhlal (kategori + şiddet)  ·  ✗ Değil  ·  ? Bilmiyorum
       not yaz, "Kaydet & Sıradakine geç" butonu
  4. Tüm kararlar `<ifc>.manual_labels.json` dosyasına yazılır.

Otomatik etiket sistemi etkilenmez; manuel set ayrı bir altın test seti
olarak `ml.scripts.error_analysis` / `evaluate` ile kullanılabilir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    entry_by_id, folder_browser, get_dataset_root, get_selected_node,
    list_entries, load_sample_for, set_dataset_root, set_selected_node,
)
from ml.data import manual_labels as ml_labels
from ml.viz.graph_view import interactive_agraph
from ml.viz.ifc3d import build_figure, extract_meshes


st.set_page_config(page_title="Manuel Etiketleme", layout="wide", page_icon="🏷️")
st.title("🏷️ Manuel Etiketleme — Altın Standart Test Seti")
st.caption(
    "Her node için **gerçek karar**: ihlal mi, değil mi, bilmiyorum mu. "
    "Otomatik etiketlerin kapsamadığı bağlamı yakalamak için. Bu dosyalar "
    "eğitim setine girmez — sadece test'te kullanılır."
)

# ---- Sidebar: dataset root ------------------------------------------------
st.sidebar.title("IFC")
st.sidebar.caption("Aynı dataset root — etiketler IFC yanına yazılır.")
current_root = get_dataset_root()
root = st.sidebar.text_input("Dataset klasörü", value=current_root)
if root != current_root:
    set_dataset_root(root)
with st.sidebar.expander("📁 Klasör seç (gez)"):
    picked = folder_browser(start=root, key="manual_browser")
    if picked:
        set_dataset_root(picked)
        st.rerun()

if not Path(root).expanduser().exists():
    st.error(f"Klasör yok: {root}")
    st.stop()

# ---- IFC seçimi -----------------------------------------------------------
entries = list_entries(root, require_graph=True)
if not entries:
    st.warning("Bu klasörde graph.json'lı IFC bulunamadı.")
    st.stop()

# Etiketleme ilerlemesini hesapla
def _progress(entry: dict) -> tuple[int, int]:
    if not entry.get("ifc_path"):
        return 0, 0
    doc = ml_labels.load(entry["ifc_path"])
    return len(doc.get("labels", {})), 0  # total bilinmiyor (graph load gerek)

st.subheader("1. IFC seç")
ifc_cols = st.columns([4, 1])
with ifc_cols[0]:
    options = [(e["id"], e) for e in entries]
    selected_idx = st.selectbox(
        "IFC",
        options=list(range(len(options))),
        format_func=lambda i: (
            f"[{options[i][1]['kind']}] {options[i][1]['name']} · "
            f"{options[i][1]['id'][:8]} · "
            f"{_progress(options[i][1])[0]} etiket"
        ),
    )
with ifc_cols[1]:
    if st.button("🎯 Yalnız etiketlenmiş", help="Sadece manuel etiketi olanları göster"):
        st.session_state["only_annotated"] = not st.session_state.get("only_annotated", False)
        st.rerun()

entry = options[selected_idx][1]
sample = load_sample_for(entry)
if sample is None:
    st.warning("Bu IFC'nin graph.json'ı yok.")
    st.stop()

g = sample.graph
all_guids = list(g.nodes)
doc = ml_labels.load(entry["ifc_path"])
labels = dict(doc.get("labels", {}))

st.subheader("2. Etiketleme")

# İlerleme metriği
prog_cols = st.columns(5)
n_total = len(all_guids)
n_done = len(labels)
n_vio = sum(1 for v in labels.values() if v.get("verdict") == ml_labels.VERDICT_VIOLATION)
n_not = sum(1 for v in labels.values() if v.get("verdict") == ml_labels.VERDICT_NOT)
n_unk = sum(1 for v in labels.values() if v.get("verdict") == ml_labels.VERDICT_UNKNOWN)
prog_cols[0].metric("Toplam node", n_total)
prog_cols[1].metric("Etiketlenmiş", f"{n_done} / {n_total}")
prog_cols[2].metric("✓ İhlal", n_vio)
prog_cols[3].metric("✗ Değil", n_not)
prog_cols[4].metric("? Bilmiyorum", n_unk)

if n_total > 0:
    st.progress(n_done / n_total, text=f"{n_done * 100 // max(n_total, 1)}% etiketlendi")

# ---- 3-panel görüntü -------------------------------------------------------
selected_guid = get_selected_node()
if selected_guid not in all_guids:
    # Otomatik: ilk etiketlenmemiş node'u seç
    for guid in all_guids:
        if guid not in labels:
            selected_guid = guid
            set_selected_node(guid)
            break

# Sürdür / sıçra
nav = st.columns([1, 1, 1, 1, 3])
with nav[0]:
    if st.button("⏮ İlk", use_container_width=True):
        set_selected_node(all_guids[0] if all_guids else None)
        st.rerun()
with nav[1]:
    if st.button("◀ Önceki", use_container_width=True) and selected_guid in all_guids:
        i = all_guids.index(selected_guid)
        set_selected_node(all_guids[max(0, i - 1)])
        st.rerun()
with nav[2]:
    if st.button("Sıradaki ▶", use_container_width=True) and selected_guid in all_guids:
        i = all_guids.index(selected_guid)
        set_selected_node(all_guids[min(len(all_guids) - 1, i + 1)])
        st.rerun()
with nav[3]:
    if st.button("⏭ Etiketsiz", use_container_width=True,
                 help="Bir sonraki henüz etiketlenmemiş node'a atla"):
        try:
            i = all_guids.index(selected_guid) if selected_guid in all_guids else -1
        except ValueError:
            i = -1
        for k in range(i + 1, len(all_guids)):
            if all_guids[k] not in labels:
                set_selected_node(all_guids[k])
                st.rerun()
        st.info("Tüm node'lar etiketli! 🎉")
with nav[4]:
    if selected_guid:
        st.caption(f"Aktif: `{selected_guid}`")

# IFC 3D + graph yan yana
view_cols = st.columns([3, 2])
with view_cols[0]:
    st.markdown("**🧱 IFC 3D — mavi = aktif node**")
    try:
        meshes = extract_meshes(entry["ifc_path"]) if entry.get("ifc_path") else None
    except Exception as e:
        st.error(f"IFC açılamadı: {e}")
        meshes = None
    if meshes:
        # Mevcut etiketlere göre renk: kırmızı = violation, yeşil = not_vio
        vio_set = {g for g, v in labels.items() if v.get("verdict") == ml_labels.VERDICT_VIOLATION}
        notvio_set = {g for g, v in labels.items() if v.get("verdict") == ml_labels.VERDICT_NOT}
        fig = build_figure(
            meshes,
            violation_guids=vio_set,
            path_guids=notvio_set,    # 'path' rengi = yeşil
            selected_guid=selected_guid,
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True, key="manual_ifc")

with view_cols[1]:
    st.markdown("**Grafik — tıklayarak node seç**")
    vio_set = {g for g, v in labels.items() if v.get("verdict") == ml_labels.VERDICT_VIOLATION}
    notvio_set = {g for g, v in labels.items() if v.get("verdict") == ml_labels.VERDICT_NOT}
    clicked = interactive_agraph(
        g,
        violation_guids=vio_set,
        path_guids=notvio_set,
        selected_guid=selected_guid,
        height=520,
        key="manual_graph",
    )
    if clicked and clicked != selected_guid:
        set_selected_node(clicked)
        st.rerun()

# ---- Aktif node paneli ----------------------------------------------------
if not selected_guid or selected_guid not in g.nodes:
    st.info("Bir node seç (üst butonlar veya grafiğe tıkla).")
    st.stop()

st.divider()
nd = g.nodes[selected_guid]
ifc_type = nd.get("ifc_type") or nd.get("type") or "?"
attrs = nd.get("attributes") or {}
name = attrs.get("Name", "")
existing = labels.get(selected_guid, {})

st.subheader(f"3. Karar: {ifc_type} · {name or selected_guid[:12]}")

info_cols = st.columns([2, 3])
with info_cols[0]:
    st.markdown("**Attributes**")
    st.json(attrs, expanded=False)
    st.markdown("**Psets**")
    st.json(nd.get("psets") or {}, expanded=False)

with info_cols[1]:
    # Mevcut karar
    if existing:
        st.caption(f"Önceki karar: **{existing.get('verdict', '?')}** "
                   f"{('· ' + existing.get('category', '')) if existing.get('category') else ''}")

    # Verdict radio
    verdict = st.radio(
        "Karar",
        options=[ml_labels.VERDICT_VIOLATION, ml_labels.VERDICT_NOT, ml_labels.VERDICT_UNKNOWN],
        format_func=lambda v: {"violation": "✓ İhlal",
                                "not_violation": "✗ İhlal değil",
                                "unknown": "? Bilmiyorum"}[v],
        index=([ml_labels.VERDICT_VIOLATION, ml_labels.VERDICT_NOT,
                ml_labels.VERDICT_UNKNOWN]
               .index(existing.get("verdict", ml_labels.VERDICT_NOT))
               if existing else 1),
        horizontal=True,
    )

    category = None
    severity = None
    if verdict == ml_labels.VERDICT_VIOLATION:
        cat_cols = st.columns(2)
        with cat_cols[0]:
            category = st.selectbox(
                "Kategori",
                options=ml_labels.VIOLATION_CATEGORIES,
                index=(ml_labels.VIOLATION_CATEGORIES.index(existing.get("category"))
                       if existing.get("category") in ml_labels.VIOLATION_CATEGORIES else 0),
            )
        with cat_cols[1]:
            severity = st.selectbox(
                "Şiddet",
                options=ml_labels.SEVERITIES,
                index=(ml_labels.SEVERITIES.index(existing.get("severity"))
                       if existing.get("severity") in ml_labels.SEVERITIES else 2),
            )

    note = st.text_area("Not (gerekçe / bağlam)", value=existing.get("note", ""),
                         height=80, placeholder="Neden bu karar? Bağlam ne?")

    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("💾 Kaydet ve sıradakine geç",
                     type="primary", use_container_width=True):
            ml_labels.upsert_node(
                entry["ifc_path"], entry["id"], selected_guid,
                verdict, category=category, severity=severity, note=note,
            )
            # Sıradaki etiketsiz'e atla
            try:
                i = all_guids.index(selected_guid)
            except ValueError:
                i = -1
            done = {**labels, selected_guid: {"verdict": verdict}}
            next_guid = None
            for k in range(i + 1, len(all_guids)):
                if all_guids[k] not in done:
                    next_guid = all_guids[k]
                    break
            if next_guid is None:
                # baştan ara
                for k in range(0, i):
                    if all_guids[k] not in done:
                        next_guid = all_guids[k]
                        break
            if next_guid:
                set_selected_node(next_guid)
            st.success(f"✅ Kaydedildi: {verdict}")
            st.rerun()
    with btn_cols[1]:
        if st.button("💾 Sadece kaydet", use_container_width=True):
            ml_labels.upsert_node(
                entry["ifc_path"], entry["id"], selected_guid,
                verdict, category=category, severity=severity, note=note,
            )
            st.success("Kaydedildi.")
            st.rerun()
    with btn_cols[2]:
        if existing and st.button("🗑 Bu node'un etiketini sil",
                                   use_container_width=True):
            ml_labels.delete_node(entry["ifc_path"], entry["id"], selected_guid)
            st.success("Silindi.")
            st.rerun()

# ---- Dosya bilgisi --------------------------------------------------------
with st.expander("📁 Dosya bilgisi"):
    p = ml_labels.manual_labels_path(entry["ifc_path"])
    st.code(str(p))
    if p.exists():
        st.caption(f"Boyut: {p.stat().st_size} bayt")
        st.caption(f"Annotator: {doc.get('annotator', '?')}")
        st.caption(f"Son güncelleme: {doc.get('updated_at', '—')}")
    else:
        st.caption("Henüz dosya yok — ilk kayıtla oluşacak.")
