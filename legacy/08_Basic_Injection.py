"""Basic Injection — kuralsal, LLM'siz, sadece KAPI + KOLON.

Hızlı (API yok), deterministik. İki ihlal tipi:
  1. Kapı genişliği: bazıları <90cm (ihlal), bazıları ≥90cm (uygun=hard negative)
  2. Kolon: kapı önüne; yakın+hatta (ihlal) veya uzak/yana (uygun=hard negative)

Hard negative'ler kritik: model 'değişti=ihlal' diye ezberleyemez,
gerçek geometrik kuralı öğrenmek zorunda kalır.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from violation_pool import storage
from violation_pool.basic_inject import (
    BasicParams, MIN_DOOR_WIDTH_M, DOOR_CLEARANCE_M, run_basic_batch,
)

st.set_page_config(page_title="Basic Injection", layout="wide", page_icon="🚪")
st.title("🚪 Basic Injection — Kapı + Kolon (kuralsal, LLM'siz)")
st.caption(
    "Sentetik baseline'lara **deterministik** ihlal enjekte eder. API çağrısı "
    "yok → anlık hızlı, rate-limit yok. Sadece iki tip: kapı genişliği + "
    "kapı önü kolon. **Uyumlu değişiklikler** (hard negative) de üretir → "
    "model 'değişti=ihlal' diye ezberleyemez."
)

st.info(
    f"📏 Kurallar: Kapı net genişliği ≥ **{MIN_DOOR_WIDTH_M*100:.0f} cm** · "
    f"Kapı önü serbest mesafe ≥ **{DOOR_CLEARANCE_M*100:.0f} cm** "
    f"(kolon bundan yakın + geçiş hattındaysa → engel/ihlal)."
)

# --- Dataset seçimi --------------------------------------------------------
st.subheader("1. Hedef paket")
try:
    tags = storage.list_dataset_tags()
except Exception as e:
    st.error(f"DB okunamadı: {e}")
    st.stop()
baseline_tags = [t for t in tags if t.get("baseline", 0) > 0]
if not baseline_tags:
    st.warning("Baseline içeren dataset paketi yok. Önce **Sentetik Üretim** ile üret.")
    st.stop()

import pandas as pd
st.dataframe(pd.DataFrame(baseline_tags)[["tag", "baseline", "violated", "total"]],
             hide_index=True, use_container_width=True)
tag = st.selectbox("📦 Dataset paketi", [t["tag"] for t in baseline_tags])

# --- Parametreler ----------------------------------------------------------
st.subheader("2. Parametreler")

mode = st.radio(
    "Üretim modu",
    ["⚡ Kural tabanlı (hızlı, deterministik)", "🤖 GPT destekli (çeşitli, yavaş)"],
    horizontal=True,
    help="Kural: anlık, API yok. GPT: LLM plan önerir (daha çeşitli senaryo) "
         "ama etiket YİNE kuralla ölçülür (ground truth dürüst kalır). "
         "GPT modu API çağrısı yapar → yavaş + rate-limit riski.",
)
use_gpt = mode.startswith("🤖")
gpt_model = "gpt-4o-mini"
if use_gpt:
    gpt_model = st.text_input("GPT modeli", value="gpt-4o-mini")

c1, c2, c3 = st.columns(3)
with c1:
    variants = st.number_input("🔁 Baseline başına varyant", 1, 50, 5)
    seed_start = st.number_input("Tohum başlangıç", 0, 99999, 1000)
with c2:
    door_mod = st.slider("Değiştirilecek kapı oranı", 0.0, 1.0, 0.6, 0.1)
    door_vio = st.slider("Dar (ihlal) kapı oranı", 0.0, 1.0, 0.5, 0.1,
                          help="Geri kalanı uyumlu ama değiştirilmiş = hard negative")
with c3:
    col_ratio = st.slider("Kolon konan kapı oranı", 0.0, 1.0, 0.6, 0.1)
    col_block = st.slider("Engelleyici kolon oranı", 0.0, 1.0, 0.5, 0.1,
                          help="Geri kalanı uzak/yana = hard negative")

params = BasicParams(
    door_modify_ratio=door_mod, door_violation_ratio=door_vio,
    column_ratio=col_ratio, column_block_ratio=col_block,
)

n_baseline = next((t["baseline"] for t in baseline_tags if t["tag"] == tag), 0)
st.metric("✨ Üretilecek toplam violated IFC", n_baseline * int(variants),
          help=f"{n_baseline} baseline × {int(variants)} varyant")

# --- Çalıştır --------------------------------------------------------------
st.subheader("3. Çalıştır")
if st.button("🚪 Basic injection başlat", type="primary"):
    bar = st.progress(0.0, text="başlatılıyor...")
    log = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, stem):
        bar.progress(done / total, text=f"{done}/{total} · {stem}")
        logs.append(f"  ✓ {stem}")
        if len(logs) % 5 == 0 or done == total:
            log.code("\n".join(logs[-20:]))

    try:
        res = run_basic_batch(
            tag, variants=int(variants), seed_start=int(seed_start),
            params=params, register_in_db=True, progress_cb=_cb,
            use_gpt=use_gpt, model=gpt_model,
            method_label="basicinj",
        )
    except Exception as e:
        st.exception(e)
        st.stop()
    bar.empty()
    dt = time.time() - t0
    st.success(
        f"🎉 Bitti — {dt:.1f}s (LLM yok, anlık). "
        f"**{res['ok']}** IFC üretildi, {res['err']} hata."
    )
    m = st.columns(4)
    m[0].metric("Toplam violated IFC", res["ok"])
    m[1].metric("Graph'lı (eğitilebilir)", res.get("with_graph", 0),
                help="Eğitim için graph.json gerekli — bu sayı düşükse sorun var")
    m[2].metric("İhlal etiketi (y=1)", res["violations"])
    m[3].metric("Hard negative (y=0)", res["hard_negatives"],
                help="Değişti ama uyumlu — model bunları ihlal SANMAMALI")

    # Graph üretim hataları (eğitilebilir 0 ise kritik)
    gerrs = res.get("graph_errors", [])
    if gerrs:
        st.error(
            f"⚠️ {len(gerrs)} IFC'de graph üretimi başarısız → bunlar EĞİTİLEMEZ. "
            "İlk birkaç hata:"
        )
        for ge in gerrs[:5]:
            st.caption(f"   • {ge}")
    st.caption(
        "Sonraki: **GAT Eğitim** → bu paketi seç → eğit. İhlal/hard-negative "
        "dengesi sayesinde model gerçek kuralı öğrenmek zorunda."
    )
