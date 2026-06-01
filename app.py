"""IFC İhlal Analizi — entry point.

st.navigation ile programatik sayfa yönetimi:
  * mod = "new" → sadece ana sayfa + yeni LLM sayfaları (sol menüde).
  * mod = "old" → ana sayfa + tüm eski (kural-tabanlı) sayfalar.

Eski sayfalar legacy/ klasöründe; pages/ yalnızca YENİ LLM çalışmaları için.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


_MODE_KEY = "app_mode"
if _MODE_KEY not in st.session_state:
    st.session_state[_MODE_KEY] = "new"

_mode = st.session_state[_MODE_KEY]

_home = st.Page("home.py", title="🏠 Ana Sayfa", default=True)

if _mode == "old":
    # Eski kural-tabanlı sistem — tüm sayfalar görünür
    nav: dict[str, list] = {
        "Ana": [_home],
        "📦 Eski — Veri & enjeksiyon": [
            st.Page("legacy/07_Sentetik_Uretim.py",
                    title="🏗️ Sentetik Üretim"),
            st.Page("legacy/08_Basic_Injection.py",
                    title="🚪 Basic Injection (kural)"),
            st.Page("legacy/09_Tam_Etiketleme.py",
                    title="✅ Tam Etiketleme"),
            st.Page("legacy/11_Kapi_Erisilebilirligi.py",
                    title="🚪 Kapı Erişilebilirliği Deneyi"),
        ],
        "📦 Eski — GAT": [
            st.Page("legacy/04_GAT_Egitim.py", title="🎓 GAT Eğitim"),
            st.Page("legacy/05_GAT_Test.py", title="🧪 GAT Test"),
            st.Page("legacy/03_GAT_Tespiti.py", title="🤖 GAT Tespiti"),
        ],
        "📦 Eski — Görselleştirme & kayıt": [
            st.Page("legacy/01_Model_Görüntüleyici.py",
                    title="🧱 Model Görüntüleyici"),
            st.Page("legacy/02_Statik_Analiz.py", title="📋 Statik Analiz"),
            st.Page("legacy/06_Manuel_Etiketleme.py",
                    title="✏️ Manuel Etiketleme"),
            st.Page("legacy/10_Kayitlar.py", title="📒 Kayıtlar"),
            st.Page("legacy/99_Codex_LLM_Havuzu.py",
                    title="📚 Codex Eski LLM Havuzu"),
        ],
    }
else:
    # Yeni LLM-tabanlı sistem — sayfalar pages/ altına eklendikçe burada görünecek
    nav = {
        "Ana": [_home],
    }
    # pages/ içindeki yeni LLM sayfaları otomatik tara — sırayla eklenecek
    _pages_dir = _ROOT / "pages"
    if _pages_dir.exists():
        _llm_pages = []
        for f in sorted(_pages_dir.glob("*.py")):
            _llm_pages.append(st.Page(f"pages/{f.name}", title=f.stem.replace("_", " ")))
        if _llm_pages:
            nav["🤖 LLM Sistemi"] = _llm_pages

pg = st.navigation(nav)
pg.run()
