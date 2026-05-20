"""Ana giriş — IFC All Violeter Viewer & Analyser.

Bu dosya Streamlit multipage uygulamasının landing sayfası. Asıl çalışan
modüller `pages/` altında ayrı sayfalar olarak yüklenir:

  1. Havuz Oluşturma & IFC Stüdyo  (codex1 çekirdeği)
  2. Dataset Görüntüleyici          (read-only browser)
  3. Model Görüntüleyici            (tekil IFC + graph)
  4. Statik Analiz                  (path-finding)
  5. GAT Eğitim
  6. GAT Tespiti
  7. GAT Test

Buradan sol menüden istediğin sayfaya geç.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

import paths  # secrets + data_home yan etkili olarak yüklenir


st.set_page_config(
    page_title="IFC Violeter — Ana Sayfa",
    page_icon="🏛️",
    layout="wide",
)


# ---- Sistem durumu --------------------------------------------------------

def _status_table() -> list[tuple[str, str, bool]]:
    s = paths.status()
    rows: list[tuple[str, str, bool]] = []
    rows.append(("Program kökü", s["program_root"], True))
    rows.append(("Parent klasör", s["parent_root"], True))
    rows.append(("Veri klasörü", s["data_home"], Path(s["data_home"]).exists()))
    rows.append((
        "secrets.txt",
        s["secrets_file"] or "BULUNAMADI  →  parent klasöre kopyala",
        s["secrets_file"] is not None,
    ))
    rows.append((
        "OPENAI_API_KEY",
        "tanımlı" if s["openai_api_key"] == "set" else "EKSİK (secrets.txt içine yaz)",
        s["openai_api_key"] == "set",
    ))
    db = paths.db_path()
    rows.append((
        "violation_pool.sqlite",
        str(db) if db.exists() else f"henüz yok  ({db})",
        db.exists(),
    ))
    return rows


def _ifc_counts() -> dict[str, int]:
    base = paths.ifc_models_dir()
    # folder adları (column adı != folder adı bazı türler için)
    return {
        kind: sum(1 for _ in (base / kind).glob("*.ifc"))
        for kind in ("baseline", "baseline_uploaded", "violated", "imports")
    }


# ---- UI -------------------------------------------------------------------

st.title("🏛️ IFC All Violeter Viewer & Analyser")
st.caption(
    "Erişilebilirlik ihlali dataset üretimi + viewer + GAT analizinin "
    "tek çatı altında birleştirilmiş hali."
)

st.divider()

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Sistem durumu")
    for label, value, ok in _status_table():
        icon = "✅" if ok else "⚠️"
        st.markdown(f"{icon} **{label}** — `{value}`")

    if not paths.secrets_ok():
        st.warning(
            "OPENAI_API_KEY tanımlı değil. Parent klasöre `secrets.txt` "
            "oluştur (örnek: `secrets.txt.example`). LLM gerektiren sayfalar "
            "(Havuz Oluşturma, IFC Stüdyo) o ana kadar çalışmaz."
        )

with col_right:
    st.subheader("Dataset özeti")
    if paths.db_path().exists():
        counts = _ifc_counts()
        c1, c2 = st.columns(2)
        c1.metric("📐 Baseline (LLM)", counts["baseline"])
        c2.metric("📦 Baseline (yüklenen)", counts["baseline_uploaded"])
        c1.metric("⚠️ Violated", counts["violated"])
        c2.metric("📥 Imports", counts["imports"])
    else:
        st.info("Henüz veri yok. Sol menüden **Havuz Oluşturma & IFC Stüdyo** "
                "sayfasına gidip ilk havuzunu üret.")

st.divider()

st.subheader("Sayfalar")
st.markdown(
    """
| # | Sayfa | Ne işe yarar |
|---|---|---|
| 1 | **Havuz Oluşturma & IFC Stüdyo** | LLM/RAG/FT ile ihlal kuralı havuzu üret, IFC üret + ihlal enjekte et |
| 2 | **Dataset Görüntüleyici** | Tüm dataset'i read-only gez (havuz + IFC + etiket + 3D + graph) |
| 3 | **Model Görüntüleyici** | Tekil IFC modelini 3D + graph yan yana, click-to-cross-highlight |
| 4 | **Statik Analiz** | İki oda arasında en kısa / en geniş / en erişilebilir yol |
| 5 | **GAT Eğitim** | Hetero-GAT eğit (uygulama içi) |
| 6 | **GAT Tespiti** | Eğitilmiş checkpoint'le tek-model tahmini + label diff |
| 7 | **GAT Test** | Birden çok IFC üzerinde toplu test |
| 8 | **Dış IFC Görüntüle** | Dataset dışındaki tek IFC veya klasördeki tüm IFC'leri hızlıca 3D + graph olarak gör (kalıcı kayıt yok) |
"""
)

st.divider()

with st.expander("📦 Kurulum / yerleşim hatırlatması", expanded=False):
    st.code(
        """ifc_claude_code_directory/                       ← parent (kalıcı)
├── ifc_all_violeter_viewer_analiyser/           ← repo (silinip clone'lanabilir)
├── secrets.txt                                  ← API key (parent'ta, kalıcı)
└── data/                                        ← veri (parent'ta, kalıcı)
    ├── violation_pool.sqlite
    ├── ifc_models/{baseline,violated,imports}/
    ├── exports/  docs/  vectorstore/  runs/  logs/""",
        language=None,
    )
    st.markdown(
        "Repo'yu `git pull` ettiğinde veya tümüyle silip yeniden "
        "clone'ladığında **data/** ve **secrets.txt** parent klasörde "
        "kaldığı için etkilenmez."
    )

st.caption(f"Çalışma dizini: `{os.getcwd()}`")
