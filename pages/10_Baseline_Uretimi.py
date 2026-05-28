"""LLM ile Baseline Üretimi — hiyerarşik (ana baseline + N varyant).

Akış:
  1. Paket adı + bina tarifi (template prompt) gir.
  2. "Üret" → LLM gpt-4o ile (a) ana baseline planını + (b) N varyant
     planını üretir; her biri ayrı LLM çağrısı, kendi prompt'uyla.
  3. Tasarım planları prosedürel motora verilir → geçerli IFC4 dosyaları.
  4. Hepsi DB'ye baseline olarak kayıt; ana parent_id=None, varyantlar
     parent_id=ana.id.

İhlal eklenmez — sadece IFC üretimi. Prompt'lar her IFC için DB'ye saklanır
ve görüntüleyici (sayfa 15) içinde "🤖 LLM prompt'u" expander ile gözükür.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from violation_pool import storage
from violation_pool.config import settings
from llm.baseline_pipeline import run_baseline_pipeline


st.set_page_config(page_title="Baseline Üretimi", layout="wide", page_icon="🏠")
st.title("🏠 LLM ile Baseline Üretimi")
st.caption(
    "Bir bina tarifi yaz → LLM **ana baseline** + N **varyant** üretir. "
    "Her IFC kendi LLM çağrısıyla, kendi prompt'u DB'ye saklı. "
    "İhlal eklenmez (sonraki adım)."
)

# OPENAI key kontrolü
if not settings.openai_api_key:
    st.error(
        "⚠️ `OPENAI_API_KEY` tanımlı değil. Data klasöründeki `.env` "
        "dosyana ekle ve Streamlit'i yeniden başlat."
    )
    st.stop()


# --- 1. Paket adı --------------------------------------------------------
st.subheader("1. Paket (ana baseline) adı")
c_name, c_help = st.columns([2, 1])
with c_name:
    ana_ad = st.text_input(
        "📛 Paket adı (dataset_tag)",
        value="ofis_v1",
        help="Bu paketin etiketi. Ana baseline'ın dosya adı: "
             "`<paket>_ana_<seed>.ifc`. Varyantlar: `<paket>_<seed>.ifc`.",
    )
with c_help:
    st.caption("Sadece harf/rakam/_/-/+ kullan. "
               "Diğerleri otomatik _'a çevrilir.")

ana_ad_clean = re.sub(r"[^A-Za-z0-9_\-+]", "_", ana_ad.strip()) or "paket"
if ana_ad_clean != ana_ad:
    st.caption(f"⚠️ Temizlenmiş: `{ana_ad_clean}`")

try:
    existing = storage.list_dataset_tags()
except Exception as e:
    existing = []
    st.warning(f"DB okunamadı: {e}")
existing_names = {t["tag"] for t in existing}
if ana_ad_clean in existing_names:
    m = next(t for t in existing if t["tag"] == ana_ad_clean)
    st.warning(
        f"ℹ️ `{ana_ad_clean}` paketi zaten var "
        f"({m.get('baseline', 0)} baseline + {m.get('violated', 0)} violated). "
        "Yeni IFC'ler bu pakete EKLENECEK — eskisi silinmez. "
        "Birden fazla ana baseline çıkarsa görüntüleyici ilkini gösterir."
    )


# --- 2. Bina tarifi (template prompt) -----------------------------------
st.subheader("2. Bina tarifi (template prompt)")
default_prompt = (
    "Küçük ölçekli bir ofis binası.\n"
    "- 2-3 oda + merkezi koridor\n"
    "- 1-2 kat\n"
    "- Her odanın koridora bir kapısı olsun\n"
    "- Erişilebilirlik açısından kapı ve koridor boyutları mevzuata uygun"
)
user_template = st.text_area(
    "💬 Bina tarifi",
    value=default_prompt, height=180,
    help="Bina tipi, oda sayısı, kat sayısı, layout istekleri vs. LLM "
         "bunu okuyup TASARIM PARAMETRELERİ (JSON) üretir. "
         "Prosedürel motor bunları geçerli IFC'ye çevirir."
)


# --- 3. LLM modeli + üretim params --------------------------------------
st.subheader("3. LLM ve üretim parametreleri")
mc1, mc2 = st.columns([2, 1])
with mc1:
    model = st.text_input(
        "🤖 GPT modeli", value="gpt-4o",
        help="Default: gpt-4o. İstediğin modeli yaz: gpt-4o-mini, gpt-4, "
             "gpt-5, gpt-5-mini, gpt-4-turbo vb.",
    )
with mc2:
    st.caption(
        "Öneriler:\n"
        "- `gpt-4o-mini` (ucuz, hızlı)\n"
        "- `gpt-4o` (denge)\n"
        "- `gpt-4` (klasik)\n"
        "- `gpt-5`, `gpt-5-mini` (yeni)"
    )

p1, p2 = st.columns(2)
with p1:
    variants = st.number_input(
        "🔁 Varyant sayısı (ana hariç)",
        min_value=1, max_value=50, value=5,
        help="Ana baseline'a EK olarak üretilecek varyant baseline sayısı. "
             "Her varyant ayrı LLM çağrısı ile üretilir.",
    )
with p2:
    seed_start = st.number_input(
        "🎲 Tohum başlangıcı",
        min_value=0, max_value=99999, value=0,
        help="Ana baseline = seed_start, varyantlar = seed_start+1, +2, ...",
    )

mc = st.columns(3)
mc[0].metric("Ana baseline", 1)
mc[1].metric("Varyant", int(variants))
mc[2].metric("Toplam LLM çağrısı", int(variants) + 1)

# Sistem prompt önizleme
with st.expander("🔬 Sistem promptu (LLM rolü) — önizleme", expanded=False):
    from llm.design_planner import SYSTEM_PROMPT, VARIATION_INSTRUCTION
    st.markdown("**Ana baseline için system prompt:**")
    st.code(SYSTEM_PROMPT, language="text")
    st.markdown("**Varyant baseline için EK (ana özetiyle birlikte):**")
    st.code(VARIATION_INSTRUCTION, language="text")


# --- 4. Çalıştır --------------------------------------------------------
st.subheader("4. Üret")
if st.button("🏠 LLM ile baseline'ları üret", type="primary",
             use_container_width=True):
    bar = st.progress(0.0, text="başlatılıyor...")
    log_slot = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, label):
        bar.progress(done / total, text=f"{done}/{total} · {label}")
        logs.append(f"  • {label}")
        if len(logs) % 2 == 0 or done == total:
            log_slot.code("\n".join(logs[-15:]))

    try:
        res = run_baseline_pipeline(
            user_template=user_template,
            dataset_tag=ana_ad_clean,
            variants=int(variants),
            seed_start=int(seed_start),
            model=model.strip() or "gpt-4o",
            progress_cb=_cb,
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar.empty()
    dt = time.time() - t0

    n_ok = (1 if res.get("ana_id") else 0) + len(res.get("variant_ids", []))
    st.success(
        f"🎉 Bitti — {dt:.1f}s · paket `{ana_ad_clean}` · "
        f"{n_ok} IFC ({'ana ✓' if res.get('ana_id') else 'ana ❌'} + "
        f"{len(res.get('variant_ids', []))} varyant)"
    )

    if res.get("errors"):
        with st.expander(f"⚠️ {len(res['errors'])} hata — detay",
                         expanded=False):
            for e in res["errors"]:
                st.code(e, language="text")

    # Ana baseline detayı
    if res.get("ana_plan"):
        ap = res["ana_plan"]
        st.markdown("### 🏛️ Ana baseline")
        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown(
                f"**Tasarım:** {ap.n_storeys} kat · "
                f"{ap.n_rooms_per_floor} oda/kat · "
                f"`{ap.layout}` layout · "
                f"kat yüksekliği {ap.storey_height} m"
            )
            st.markdown(f"**LLM rationale:** _{ap.rationale}_")
            if res.get("ana_path"):
                st.caption(f"📁 `{Path(res['ana_path']).name}`")
        with cols[1]:
            with st.expander("🤖 User prompt (LLM'e gönderilen)",
                             expanded=False):
                st.code(ap.user_prompt, language="text")
            with st.expander("📤 LLM ham yanıtı"):
                st.code(ap.raw_llm_response, language="json")

    # Varyant özet tablosu
    if res.get("variant_plans"):
        st.markdown("### 🔀 Varyantlar")
        rows = []
        for i, vp in enumerate(res["variant_plans"]):
            rows.append({
                "#": i + 1,
                "Dosya": (Path(res["variant_paths"][i]).name
                          if i < len(res["variant_paths"]) else "—"),
                "Kat": vp.n_storeys,
                "Oda/Kat": vp.n_rooms_per_floor,
                "Layout": vp.layout,
                "Rationale": (vp.rationale[:60] + "…"
                              if len(vp.rationale) > 60 else vp.rationale),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        with st.expander("🤖 Varyant prompt'larını gör", expanded=False):
            for i, vp in enumerate(res["variant_plans"]):
                st.markdown(f"**Varyant {i+1}** ({vp.model}):")
                st.code(vp.user_prompt, language="text")
                st.caption(f"LLM yanıtı: `{vp.raw_llm_response[:120]}…`")

    st.caption(
        "Sonraki adım: **🔍 IFC Görüntüleyici** sayfasına geç → bu paketi "
        "seç → ana baseline ve varyantları yan yana incele. "
        "İhlal eklemesini birlikte planlayacağız."
    )


# --- Mevcut paketler ----------------------------------------------------
st.divider()
st.subheader("📦 Mevcut paketler")
if existing:
    df = pd.DataFrame(existing)[["tag", "baseline", "violated", "total"]]
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.caption("Henüz paket yok.")
