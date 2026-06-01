"""Sayfa 22 — RAG Operations.

Bağımsız RAG yönetim paneli (legacy/99'a hiç dokunmaz):
  1. Collection listesi + chunk sayısı
  2. Yeni collection oluştur
  3. PDF / TXT / MD yükle → seçili collection'a ingest
  4. Hızlı retrieve testi (debug)
  5. Collection sil (bozuk index'i kurtarmak için)

Sayfa 21 (RAG İhlal Üretimi) bu sayfada hazırlanan collection'ları kullanır.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from violation_pool import rag
from violation_pool.config import settings


st.set_page_config(page_title="RAG Operations", layout="wide", page_icon="📚")
st.title("📚 RAG Operations")
st.caption(
    "ChromaDB collection yönetimi: oluştur, doküman ingest et, test sorgusu "
    "çalıştır, gerekirse sil. Sayfa 21 buradaki collection'ları kullanır."
)

if not settings.openai_api_key:
    st.error("⚠️ `OPENAI_API_KEY` tanımlı değil.")
    st.stop()

st.caption(f"📁 ChromaDB dizini: `{settings.vectorstore_dir}`")


# --- 1. Mevcut collection'lar --------------------------------------------
st.subheader("1. Mevcut collection'lar")

@st.cache_data(ttl=5, show_spinner=False)
def _collections_table() -> pd.DataFrame:
    rows = []
    for name in rag.list_collections():
        try:
            info = rag.collection_info(name)
            rows.append({
                "collection": name,
                "chunk": info.get("count", 0),
                "embedding": (info.get("metadata") or {}).get(
                    "embedding_model", "?"),
            })
        except Exception as e:
            rows.append({"collection": name, "chunk": "ERR",
                         "embedding": str(e)[:60]})
    return pd.DataFrame(rows)

cols_df = _collections_table()
if cols_df.empty:
    st.info("Henüz hiç collection yok. Aşağıdan yeni bir tane oluştur.")
else:
    st.dataframe(cols_df, hide_index=True, use_container_width=True)


# --- 2. Yeni collection oluştur ------------------------------------------
st.subheader("2. Yeni collection oluştur")
nc1, nc2 = st.columns([3, 1])
with nc1:
    new_name = st.text_input(
        "Collection adı",
        placeholder="ör. ts9111_v2",
        help="Sadece harf/rakam/altçizgi/tire. Türkçe karakter kullanma.",
    )
with nc2:
    if st.button("➕ Oluştur", use_container_width=True,
                 disabled=not new_name.strip()):
        try:
            rag.create_collection(new_name.strip())
            st.success(f"`{new_name.strip()}` oluşturuldu.")
            _collections_table.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Oluşturulamadı: {e}")


# --- 3. Doküman ingest ---------------------------------------------------
st.subheader("3. Doküman yükle & ingest et")

existing = rag.list_collections()
if not existing:
    st.info("Önce bir collection oluştur.")
else:
    ic1, ic2 = st.columns([2, 3])
    with ic1:
        target_coll = st.selectbox(
            "Hedef collection",
            options=existing,
            help="Yüklenen dosyalar bu collection'a eklenir.",
        )
        try:
            cur_count = rag.collection_info(target_coll).get("count", 0)
            st.caption(f"Mevcut chunk: **{cur_count}**")
        except Exception as e:
            cur_count = 0
            st.caption(f"Info hatası: {e}")
    with ic2:
        uploaded = st.file_uploader(
            "PDF / TXT / MD dosyaları",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="PDF için sayfa bazlı çıkarım yapılır; TXT/MD tek parça okunur.",
        )

    if uploaded:
        st.caption(f"📥 {len(uploaded)} dosya seçildi: " +
                   ", ".join(f.name for f in uploaded))
        if st.button("📚 Ingest et", type="primary",
                     use_container_width=True):
            t0 = time.time()
            try:
                with tempfile.TemporaryDirectory() as td:
                    tdp = Path(td)
                    paths = []
                    for uf in uploaded:
                        p = tdp / uf.name
                        p.write_bytes(uf.getbuffer())
                        paths.append(p)
                    with st.spinner(
                        f"{len(paths)} dosya işleniyor, "
                        f"chunk + embed üretiliyor..."
                    ):
                        info = rag.ingest_documents(target_coll, paths)
            except Exception as e:
                st.exception(e)
                st.stop()
            dt = time.time() - t0
            st.success(
                f"✓ {info['chunks_added']} chunk eklendi "
                f"({len(info['documents'])} doküman) · {dt:.1f}s"
            )
            _collections_table.clear()
            st.json(info, expanded=False)


# --- 4. Hızlı retrieve testi ---------------------------------------------
st.subheader("4. Retrieve testi (debug)")
existing = rag.list_collections()
if not existing:
    st.caption("Önce collection oluştur ve ingest et.")
else:
    tc1, tc2, tc3 = st.columns([2, 3, 1])
    with tc1:
        test_coll = st.selectbox(
            "Test collection", options=existing, key="test_coll",
        )
    with tc2:
        test_q = st.text_input(
            "Sorgu",
            value="kapı net geçiş genişliği minimum",
            help="Bu sorguyla collection'dan top-k chunk çekilir.",
        )
    with tc3:
        test_k = st.number_input("k", min_value=1, max_value=20, value=5)

    if st.button("🔍 Sorgula", disabled=not test_q.strip()):
        try:
            with st.spinner("Embedding + arama..."):
                hits = rag.retrieve(test_coll, test_q.strip(), k=int(test_k))
        except Exception as e:
            st.exception(e)
            hits = []
        if not hits:
            st.warning("0 sonuç. Collection boş ya da sorguyla eşleşme yok.")
        else:
            st.success(f"{len(hits)} chunk bulundu.")
            for i, h in enumerate(hits, 1):
                meta = h.get("metadata") or {}
                with st.expander(
                    f"#{i} · {meta.get('document', '?')} "
                    f"p{meta.get('page', '?')} c{meta.get('chunk', '?')}",
                    expanded=(i == 1),
                ):
                    st.write(h.get("text", "")[:2000])


# --- 5. Tehlikeli zone: collection sil ------------------------------------
st.divider()
with st.expander("⚠️ Collection sil (bozuk index'i kurtarmak için)"):
    st.caption(
        "ChromaDB segmenti bozulduğunda (örn. `Error loading hnsw index`) "
        "collection'ı silip yeniden oluşturmak gerekebilir. **Geri "
        "alınamaz** — chunk'lar kaybolur."
    )
    existing = rag.list_collections()
    if not existing:
        st.caption("Silinecek collection yok.")
    else:
        dc1, dc2 = st.columns([3, 1])
        with dc1:
            del_name = st.selectbox(
                "Silinecek collection", options=existing, key="del_coll",
            )
        with dc2:
            confirm = st.checkbox("Eminim", key="confirm_del")
        if st.button("🗑️ Sil", disabled=not confirm,
                     type="secondary"):
            try:
                rag.delete_collection(del_name)
                st.success(f"`{del_name}` silindi.")
                _collections_table.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Silinemedi: {e}")
                st.caption(
                    "ChromaDB segmenti tamamen bozuksa Streamlit'i kapat, "
                    f"`{settings.vectorstore_dir}` dizinindeki collection "
                    "klasörünü elle sil, sonra Streamlit'i tekrar başlat."
                )

st.divider()
st.info(
    "💡 **Akış:** Bu sayfada bir collection oluştur → TS 9111 / TS ISO 21542 "
    "PDF'lerini ingest et → Sayfa 21'e geç → aynı collection'ı seç → ihlal "
    "üret. Legacy/99'a hiç ihtiyaç yok."
)
