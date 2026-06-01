"""Ana sayfa (landing) — mod bilgi + hızlı dataset geçişi.

app.py içinde st.navigation tarafından default page olarak kayıtlıdır.
Burada SADECE içerik üretiriz; sayfa kararlarını app.py verir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from paths import data_home
from ml.app.state import folder_browser, get_dataset_root, set_dataset_root


st.set_page_config(page_title="IFC İhlal Analizi", layout="wide", page_icon="🏠")

st.title("🏠 IFC İhlal Analizi")
st.caption(
    "Sentetik IFC üretimi, ihlal enjeksiyonu, eğitim, tespit ve raporlama. "
    "Sistem iki paralel modda çalışır."
)

_mode = st.session_state.get("app_mode", "new")

# ─── Mod bilgisi + geçiş düğmesi ─────────────────────────────────────────────
if _mode == "new":
    st.success("🤖 **LLM-Tabanlı Sistem** (aktif mod — yapım aşamasında)")
    st.caption("LLM (GPT-xx) ile ihlal üretimi + LLM ile ihlal tespiti.")
    if st.button("📦 Eski sürüme git (kural-tabanlı sistem)",
                 use_container_width=True, type="primary"):
        st.session_state["app_mode"] = "old"
        st.rerun()
else:
    st.warning("📦 **Eski Sürüm — Kural-Tabanlı Sistem** (tez deneyleri)")
    st.caption("Sentetik üretim, kural-tabanlı enjeksiyon, GAT eğitim/tespit, "
               "raporlar. Sol menüden istediğin sayfayı aç.")
    if st.button("← Yeni sürüme dön (LLM)",
                 use_container_width=True, type="primary"):
        st.session_state["app_mode"] = "new"
        st.rerun()

st.divider()

# ─── Dataset klasörü hızlı geçişi ────────────────────────────────────────────
st.subheader("📂 Dataset klasörü")
st.caption("Eski/yeni veri kümeleri arası anında geçiş. Boş bir klasör seçince "
           "sıfırdan başlanır → yavaşlık biter.")
cur = get_dataset_root()
qc1, qc2 = st.columns([4, 1])
new_root = qc1.text_input(
    "Mevcut klasör", value=cur, label_visibility="collapsed",
    help="Tam yol. Var olan bir klasörü gir; içinde violation_pool.sqlite "
         "yoksa ilk açılışta otomatik kurulur.",
)
if qc2.button("✅ Geçiş yap", use_container_width=True):
    if Path(new_root).expanduser().exists():
        set_dataset_root(new_root)
        st.success(f"Geçildi: `{new_root}`")
        st.rerun()
    else:
        st.error(f"Klasör bulunamadı: {new_root}")
with st.expander("📁 Klasör gez & seç"):
    p = folder_browser(start=new_root, key="home_browser")
    if p:
        set_dataset_root(p)
        st.rerun()
st.caption(f"💡 Varsayılan (`IFC_DATA_HOME`): `{data_home()}`")

st.divider()

# ─── Aktif modun ipuçları ────────────────────────────────────────────────────
if _mode == "new":
    st.info(
        "🚧 **Yeni LLM Sayfaları** sırayla eklenecek (sol menüde görünecek):\n\n"
        "- 20 — LLM ile ihlal üretimi (GPT-xx)\n"
        "- 21 — LLM-tabanlı ihlal tespiti\n"
        "- 22 — LLM ile kural yorumlama / oracle\n"
        "- 23 — LLM ile açıklama / etiket\n\n"
        "Şimdilik landing burası. Eski sayfaları görmek için yukarıdan "
        "**📦 Eski sürüme git** düğmesine bas."
    )
else:
    st.info(
        "📦 **Eski sürüm sayfaları** sol menüde:\n\n"
        "- 🏗️ Sentetik Üretim · 🚪 Basic Injection · ✅ Tam Etiketleme · "
        "🚪 Kapı Erişilebilirliği\n"
        "- 🎓 GAT Eğitim · 🧪 GAT Test · 🤖 GAT Tespiti\n"
        "- 🧱 Model Görüntüleyici · 📋 Statik Analiz · ✏️ Manuel Etiketleme\n"
        "- 📒 Kayıtlar (deney + işlem + PDF) · 📚 Codex Eski LLM Havuzu"
    )
