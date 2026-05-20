"""Streamlit entry point.

Run with:
    streamlit run app.py

Streamlit auto-discovers `pages/` for the multipage layout. This file
is the landing page and explains the workflow.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.state import sidebar_config, labels_summary


st.set_page_config(
    page_title="IFC Graph Analysis",
    page_icon="🏗️",
    layout="wide",
)

entry = sidebar_config()

st.title("🏗️ IFC Graph Analysis")
st.markdown(
    """
codex1'in ürettiği TS 9111 / TS ISO 21542 erişilebilirlik ihlal
dataset'i üzerinde **graph tabanlı analiz** ve **derin öğrenme**.

Sol menüden bir codex1 dataset klasörü ve bir IFC modeli seç, sonra
yukarıdaki sayfalardan birini aç:

| Sayfa | İçerik |
|---|---|
| **🧱 Model Görüntüleyici** | IFC 3D + interaktif graph yan yana; violated seçince baseline da otomatik açılıp 4-panel karşılaştırma olur. Her panel için **⛶ Büyüt** tik kutusu. |
| **🧭 Statik Analiz** | İki oda seç, aralarındaki **en kısa**, **en geniş** ve **en erişilebilir** yolları bul. Üç yol hem graph hem IFC üzerinde farklı renklerle çizilir. |
| **🤖 GAT Tespiti** | Tek bir IFC üzerinde eğitilmiş modelin tahminlerini görsel olarak incele (TP/FP/FN/decoy aldanma). |
| **🏋️ GAT Eğitim** | Hiperparam formu + dataset/pool filtresi + train/val/test slider'ları + canlı progress ile **tek tıkla eğitim**. |
| **🧪 GAT Test** | Eğitilmiş bir run'ı seçtiğin split (train/val/test/manuel) üzerinde toplu test et — aggregate metrikler + her IFC için ayrı satır. |
    """
)

if entry is None:
    st.warning(
        "Devam etmek için sol menüden geçerli bir **codex1 dataset root** "
        "girip listeden bir model seç."
    )
    st.stop()

st.divider()

cols = st.columns(3)
with cols[0]:
    st.subheader("Seçili model")
    st.code(entry["id"])
    st.caption(f"📐 tür: `{entry['kind']}`")
    if entry["parent_id"]:
        st.caption(f"🧬 baseline: `{entry['parent_id']}`")
    st.caption(f"📄 {Path(entry['ifc_path']).name if entry['ifc_path'] else '—'}")

with cols[1]:
    st.subheader("Dosyalar")
    st.text(f"IFC:    {entry['ifc_path']}")
    st.text(f"Graph:  {entry['graph_path']}")
    st.text(f"Labels: {entry.get('labels_path') or '—'}")

with cols[2]:
    st.subheader("Label özeti")
    doc = labels_summary(entry)
    if doc is None:
        st.caption("Bu IFC için label dosyası yok (baseline mi?).")
    else:
        sm = doc.get("summary", {})
        st.metric("uygulanan ihlal", sm.get("applied", 0))
        st.metric("decoy", sm.get("decoys", 0))
        st.caption(f"talep edilen: {sm.get('requested', 0)} · "
                   f"atlanan: {sm.get('skipped', 0)}")

st.divider()
st.caption("Çalıştırma: `streamlit run app.py`  ·  "
           "Env değişkeni `IFC_DATASET_ROOT` ile dataset root öntanımlı.")
