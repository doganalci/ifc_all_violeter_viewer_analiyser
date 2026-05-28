"""LLM ile Kapı İhlal Üretimi — GPT (gpt-4o vb.) ile çeşitli senaryolar.

Bir baseline paketindeki kapılara LLM-bazlı genişlik/yükseklik değişikliği
enjekte eder. İhlaller + uyumlu (hard-negative) değişiklikler birlikte üretilir.
Etiket KURALLA ölçülür (eşik kontrolü) → ground truth dürüst.

Çıktı: yeni bir 'llmgen' paketi (Tam Etiketleme formatında, her node etiketli).
Eski sürümdeki GAT Eğitim sayfasıyla doğrudan eğitilebilir.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from violation_pool import storage
from violation_pool.config import settings
from llm.batch import run_llm_door_batch
from llm.door_inject import LLMDoorParams, MIN_DOOR_HEIGHT_M, MIN_DOOR_WIDTH_M


st.set_page_config(page_title="LLM Kapı İhlal Üretimi", layout="wide",
                   page_icon="🤖")
st.title("🤖 LLM ile Kapı İhlal Üretimi")
st.caption(
    "Sentetik baseline'lara GPT ile kapı genişliği veya yüksekliği ihlali "
    "enjekte eder. Hard-negative (uyumlu değişiklik) de üretir. "
    "Etiket yine KURALLA ölçülür → ground truth dürüst kalır."
)
st.info(
    f"📏 Kurallar: Kapı net genişliği ≥ **{MIN_DOOR_WIDTH_M*100:.0f} cm** · "
    f"Kapı net yüksekliği ≥ **{MIN_DOOR_HEIGHT_M*100:.0f} cm**. "
    "Altı = ihlal · Üstü ama değiştirilmiş = hard-negative."
)

if not settings.openai_api_key:
    st.error("⚠️ `OPENAI_API_KEY` tanımlı değil. Data klasöründeki `.env` "
             "dosyana ekle ve Streamlit'i yeniden başlat.")
    st.stop()

# --- Paket ----------------------------------------------------------------
st.subheader("1. Baseline paketi")
try:
    tags = storage.list_dataset_tags()
except Exception as e:
    st.error(f"DB okunamadı: {e}"); st.stop()
base_tags = [t for t in tags if t.get("baseline", 0) > 0]
if not base_tags:
    st.warning("Baseline içeren paket yok. **Eski sürüme git** → 🏗️ Sentetik "
               "Üretim ile baseline üret, sonra burayı çalıştır.")
    st.stop()
st.dataframe(pd.DataFrame(base_tags)[["tag", "baseline", "violated", "total"]],
             hide_index=True, use_container_width=True)
tag = st.selectbox("📦 Paket", [t["tag"] for t in base_tags])

# --- Model + parametreler -------------------------------------------------
st.subheader("2. Model ve parametreler")
mc1, mc2 = st.columns([2, 1])
with mc1:
    model = st.text_input(
        "🤖 GPT modeli", value="gpt-4o",
        help="Default: gpt-4o. İstediğin modeli yaz: gpt-4o-mini, gpt-4, "
             "gpt-5, gpt-5-mini, gpt-4-turbo vb. (Geçersiz model adı API "
             "tarafından reddedilir.)",
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
    variants = st.number_input("🔁 Baseline başına varyant", 1, 50, 5)
    seed_start = st.number_input("Tohum başlangıç", 0, 99999, 3000)
with p2:
    n_vio_target = st.slider("🎯 Hedef ihlal / IFC", 0, 10, 2,
                             help="LLM'e hint olarak gönderilir.")
    n_comp_target = st.slider("🎯 Hedef uyumlu (hard-negative) / IFC", 0, 10, 2)

n_baseline = next((t["baseline"] for t in base_tags if t["tag"] == tag), 0)
total_ifc = int(n_baseline) * int(variants)
mc = st.columns(3)
mc[0].metric("✨ Üretilecek ihlalli IFC", total_ifc)
mc[1].metric("Tahmini LLM çağrısı", total_ifc)
mc[2].metric("Hedef toplam ihlal", total_ifc * int(n_vio_target))

params = LLMDoorParams(
    target_n_violations=int(n_vio_target),
    target_n_compliant=int(n_comp_target),
)

# --- Çalıştır --------------------------------------------------------------
st.subheader("3. Çalıştır (üret + tam etiketle)")
if st.button("🤖 LLM üret + tam etiketle", type="primary",
             use_container_width=True):
    bar = st.progress(0.0, text="başlatılıyor...")
    log_slot = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, stem):
        bar.progress(done / total, text=f"{done}/{total} · {stem}")
        logs.append(f"  ✓ {stem}")
        if len(logs) % 3 == 0 or done == total:
            log_slot.code("\n".join(logs[-15:]))

    try:
        res = run_llm_door_batch(
            tag, variants=int(variants), seed_start=int(seed_start),
            params=params, model=model.strip() or "gpt-4o",
            full_label=True, method_label="llmgen", progress_cb=_cb,
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar.empty()
    dt = time.time() - t0
    st.success(f"🎉 Bitti — {dt:.1f}s · model={model}")
    m = st.columns(4)
    m[0].metric("İhlalli IFC", res["ok"])
    m[1].metric("İhlal (y=1)", res["violations"])
    m[2].metric("Hard negatif (y=0)", res["hard_negatives"])
    m[3].metric("Kesin-temiz node", res.get("clean_labeled", 0),
                help="Tam etiketleme ile açıkça 'ihlal değil' işaretlenen "
                     "geri kalan yapı node'ları")

    if res["ok"] == 0:
        st.error(
            f"🔴 Hiç ihlalli IFC üretilemedi ({res.get('err', 0)} hata). "
            "Yaygın sebep: model adı geçersiz, API key revoke, rate-limit, "
            "veya LLM JSON bozuk döndürdü."
        )
        for it in [x for x in res.get("items", []) if "error" in x][:5]:
            st.code(f"{it['stem']}\n  → {it['error']}", language="text")
    gerrs = res.get("graph_errors", [])
    if gerrs:
        st.warning(f"⚠️ {len(gerrs)} graph hatası.")
        for ge in gerrs[:5]:
            st.caption(f"   • {ge}")

    # LLM hata sayacı (her ihlalli IFC'de _error varsa)
    llm_errs = [it for it in res.get("items", [])
                if it.get("llm_error")]
    if llm_errs:
        st.warning(f"LLM hata/bos plan: {len(llm_errs)} IFC için "
                   "(yine de üretildi ama az/no değişiklik).")

    if res["ok"] > 0:
        st.caption(
            f"Sonraki: **📦 Eski sürüme git → 🎓 GAT Eğitim** → paket "
            f"`{tag}` seç → eğit. Bağımsız test için farklı tohumla ikinci "
            "bir LLM paketi üret."
        )
