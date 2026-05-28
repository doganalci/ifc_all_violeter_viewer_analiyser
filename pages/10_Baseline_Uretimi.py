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
from llm.pricing import known_models
from ml.data.synth_baseline_v2 import SynthParamsV2


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
    _model_options = known_models() + ["✏️ Özel (elle yaz)"]
    _model_pick = st.selectbox(
        "🤖 GPT modeli",
        options=_model_options,
        index=0,
        help="Bilinen modeller için fiyat tablosu var → maliyet tam hesaplanır. "
             "Özel modelde fiyat gpt-4o varsayılır (tahmin etiketi).",
    )
    if _model_pick == "✏️ Özel (elle yaz)":
        model = st.text_input("Özel model adı", value="gpt-4o",
                              key="custom_model")
    else:
        model = _model_pick
with mc2:
    from llm.pricing import estimate_cost
    _est = estimate_cost(model, 1_000_000, 1_000_000)
    st.caption(
        f"💵 **{model}** fiyat:\n"
        f"- Input: ${_est['input_usd']:.2f} / 1M token\n"
        f"- Output: ${_est['output_usd']:.2f} / 1M token\n"
        + (f"- ⚠️ Bilinmeyen model — gpt-4o varsayım"
           if _est['is_estimate'] else "")
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

# Geometri parametreleri (sliders) — LLM kat/oda/layout seçiyor; oda/koridor
# fiziksel boyutu prosedürel motor bu aralıktan örnek alır.
with st.expander("📐 Geometri parametreleri (oda + koridor + kapı boyut aralığı)",
                 expanded=False):
    st.caption(
        "Prosedürel motor (synth_baseline_v2) bu aralıklardan rastgele örnek "
        "alarak her IFC'nin oda/koridor/kapı boyutlarını belirler. LLM bunlara "
        "karışmaz — sadece n_storeys, n_rooms, layout seçer."
    )
    gc1, gc2 = st.columns(2)
    with gc1:
        st.markdown("**Oda (m)**")
        room_w = st.slider("Oda genişliği (en)",
                           min_value=2.0, max_value=8.0, value=(3.0, 5.5),
                           step=0.1, key="g_rw")
        room_l = st.slider("Oda boyu",
                           min_value=2.0, max_value=10.0, value=(3.5, 6.0),
                           step=0.1, key="g_rl")
        st.markdown("**Kapı (m)**")
        door_w = st.slider("Kapı genişliği",
                           min_value=0.90, max_value=1.50, value=(0.95, 1.20),
                           step=0.05, key="g_dw",
                           help="≥ 0.90 m (TS 9111). Min eşik altına çekersen "
                                "baseline mevzuata aykırı çıkabilir.")
        door_h = st.slider("Kapı yüksekliği",
                           min_value=2.00, max_value=2.50, value=(2.10, 2.30),
                           step=0.05, key="g_dh",
                           help="≥ 2.00 m (TS 9111).")
    with gc2:
        st.markdown("**Koridor (m)**")
        corridor_w = st.slider("Koridor genişliği",
                               min_value=1.20, max_value=3.00, value=(1.40, 2.20),
                               step=0.1, key="g_cw",
                               help="≥ 1.20 m (TS 9111). Eşik altına çekersen "
                                    "baseline mevzuata aykırı çıkabilir.")
        corridor_l = st.slider("Koridor uzunluğu",
                               min_value=2.0, max_value=10.0, value=(3.0, 6.0),
                               step=0.1, key="g_cl")
        st.markdown("**Genel**")
        wall_t = st.slider("Duvar kalınlığı",
                           min_value=0.10, max_value=0.40, value=0.20,
                           step=0.05, key="g_wt")

    # Uyarılar
    if door_w[0] < 0.90 or door_h[0] < 2.00 or corridor_w[0] < 1.20:
        st.warning(
            "⚠️ Minimum değerlerden bazıları TS 9111 eşiğinin altında — "
            "üretilen baseline'lar mevzuata aykırı olabilir."
        )

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

    # UI'dan slider'larla gelen geometri aralıkları → SynthParamsV2
    g_params = SynthParamsV2(
        room_w_min=float(room_w[0]), room_w_max=float(room_w[1]),
        room_l_min=float(room_l[0]), room_l_max=float(room_l[1]),
        corridor_w_min=float(corridor_w[0]),
        corridor_w_max=float(corridor_w[1]),
        corridor_l_min=float(corridor_l[0]),
        corridor_l_max=float(corridor_l[1]),
        door_w_min=float(door_w[0]), door_w_max=float(door_w[1]),
        door_h_min=float(door_h[0]), door_h_max=float(door_h[1]),
        wall_thickness=float(wall_t),
    )

    try:
        res = run_baseline_pipeline(
            user_template=user_template,
            dataset_tag=ana_ad_clean,
            variants=int(variants),
            seed_start=int(seed_start),
            model=model.strip() or "gpt-4o",
            params=g_params,
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

    # LLM ölçüm özeti
    totals = res.get("totals", {})
    if totals.get("n_calls", 0):
        st.markdown("#### 📊 LLM kullanım özeti")
        mc = st.columns(5)
        mc[0].metric("LLM çağrısı", totals.get("n_calls", 0))
        mc[1].metric("Toplam token",
                     f"{totals.get('total_tokens', 0):,}".replace(",", "."))
        mc[2].metric("Prompt token",
                     f"{totals.get('prompt_tokens', 0):,}".replace(",", "."))
        mc[3].metric("Süre (LLM)", f"{totals.get('duration_s', 0):.1f}s")
        mc[4].metric("Maliyet (≈)",
                     f"${totals.get('cost_usd', 0):.4f}")
        st.caption(
            f"📒 Detay defter: `data/llm_generations.xlsx` (her LLM çağrısı bir satır)"
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
            st.markdown(
                f"**Ölçüm:** {ap.total_tokens:,} token · "
                f"{ap.duration_s:.2f}s · ${ap.cost_usd:.5f} "
                f"({ap.cost_matched}"
                f"{' · tahmin' if ap.cost_is_estimate else ''})"
            )
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
                "Token": vp.total_tokens,
                "Süre (s)": round(vp.duration_s, 2),
                "USD": f"{vp.cost_usd:.5f}",
                "Rationale": (vp.rationale[:50] + "…"
                              if len(vp.rationale) > 50 else vp.rationale),
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
