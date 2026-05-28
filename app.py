"""Ana sayfa — IFC ihlal analizi mod seçici + hızlı dataset geçişi.

Sistem iki paralel modda çalışır:
  * 🔧 Kural-tabanlı: deterministik enjeksiyon, GAT eğitim/tespit (tamamlandı).
  * 🤖 LLM-tabanlı: LLM ile çeşitlilik, açıklama, oracle, tespit (yapım aşamasında).

Eski codex1 LLM havuzu/RAG/fine-tune arayüzü `📚 Codex Eski LLM Havuzu` sayfasında
durur — eski verilere erişim için.
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
    "Sentetik IFC üretimi, ihlal enjeksiyonu, GAT-tabanlı tespit ve raporlama. "
    "Sistem iki modda çalışır: kural-tabanlı (tamamlandı, sürekli erişilebilir) "
    "ve LLM-tabanlı (geliştirilmekte)."
)

# ─── Dataset klasörü — hızlı geçiş ───────────────────────────────────────────
st.subheader("📂 Dataset klasörü")
st.caption("Eski/yeni veri kümeleri arasında anında geçiş için klasör değiştir. "
           "Boş bir klasör seçince sıfırdan yeni dataset gibi açılır.")
cur = get_dataset_root()
qc1, qc2 = st.columns([4, 1])
new_root = qc1.text_input(
    "Mevcut klasör", value=cur, label_visibility="collapsed",
    help="Tam yol. Klasör yoksa hata verir; var olan bir klasörü gir.",
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
st.caption(f"💡 Varsayılan klasör (`IFC_DATA_HOME`): `{data_home()}`")

st.divider()

# ─── Mod seçici — kural vs LLM ───────────────────────────────────────────────
mc1, mc2 = st.columns(2, gap="large")

with mc1:
    st.markdown("## 🔧 Kural-Tabanlı Sistem")
    st.caption(
        "Deterministik enjeksiyon, hard-negative üretimi, GAT eğitim/tespit. "
        "Tez deneylerimizin tamamlandığı pipeline. Aşağıdaki sayfalardan "
        "istediğin zaman geri dönebilirsin."
    )
    st.markdown("**Veri üretimi**")
    st.page_link("pages/07_Sentetik_Uretim.py", label="🏗️ Sentetik Üretim")
    st.page_link("pages/08_Basic_Injection.py", label="🚪 Basic Injection (kural)")
    st.page_link("pages/09_Tam_Etiketleme.py", label="✅ Tam Etiketleme")
    st.page_link("pages/11_Kapi_Erisilebilirligi.py",
                 label="🚪 Kapı Erişilebilirliği Deneyi")
    st.markdown("**Model & analiz**")
    st.page_link("pages/04_GAT_Egitim.py", label="🎓 GAT Eğitim")
    st.page_link("pages/05_GAT_Test.py", label="🧪 GAT Test")
    st.page_link("pages/03_GAT_Tespiti.py", label="🤖 GAT Tespiti")
    st.markdown("**Görselleştirme & etiket**")
    st.page_link("pages/01_Model_Görüntüleyici.py", label="🧱 Model Görüntüleyici")
    st.page_link("pages/02_Statik_Analiz.py", label="📋 Statik Analiz")
    st.page_link("pages/06_Manuel_Etiketleme.py", label="✏️ Manuel Etiketleme")
    st.markdown("**Raporlar & defterler**")
    st.page_link("pages/10_Kayitlar.py", label="📒 Kayıtlar (deney + işlem + PDF)")

with mc2:
    st.markdown("## 🤖 LLM-Tabanlı Sistem")
    st.caption(
        "LLM ile çeşitli ihlal üretimi, LLM-tabanlı tespit, kural yorumlama "
        "(oracle) ve doğal dil açıklamaları. **Yapım aşamasında** — sayfalar "
        "sırayla ekleniyor."
    )
    st.info(
        "🚧 Yeni LLM sayfaları (20–23 aralığında) sırayla eklenecek:\n\n"
        "- 20 — LLM ile çeşitli ihlal üretimi\n"
        "- 21 — LLM-tabanlı tespit modeli\n"
        "- 22 — LLM ile kural yorumlama / oracle\n"
        "- 23 — LLM ile açıklama / etiket"
    )
    st.markdown("**Mevcut (eski) LLM altyapısı**")
    st.page_link("pages/99_Codex_LLM_Havuzu.py",
                 label="📚 Codex Eski LLM Havuzu (havuz / RAG / FT / enjeksiyon)")
    st.caption("Codex1'in orijinal LLM tabanlı havuz oluşturma ve fine-tune "
               "arayüzü — eski verilerle çalışmak için durur.")

st.divider()
st.caption(
    "İpucu: yavaşlık yaşıyorsan üstteki **Dataset klasörü** kutusunu boş/yeni "
    "bir klasöre çevir; eski 50k IFC'lik klasör artık yüklenmez. İstediğin "
    "zaman eski yolu yapıştırıp geri dönersin."
)
