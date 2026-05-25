"""Tam Etiketleme — baseline'dan ihlalli üret + HER node'u açık etiketle.

Basic Injection ile aynı (kapı + kolon ihlali) ama farkı: üretilen
violated IFC'lerde **her node'un** kesin etiketi yazılır:
  * Enjekte edilen dar kapı / engelleyici kolon → ihlal (y=1)
  * Değiştirilen ama uyumlu kapı / uzak kolon   → hard negative (y=0)
  * Geri kalan TÜM node'lar (duvar, döşeme, dokunulmamış kapı, …)
    → 'clean' (y=0)  ← baseline garantili temiz olduğu için KESİN

Sonuç: closed-world tam geçerli, eksiksiz etiketli eğitim seti. Model
hem ihlali hem ihlal-olmayanı kesin örneklerle öğrenir.
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

st.set_page_config(page_title="Tam Etiketleme", layout="wide", page_icon="✅")
st.title("✅ Tam Etiketleme — Üret + Her Node'u Etiketle")
st.caption(
    "Basic Injection ile aynı (kapı + kolon) ama üretilen violated "
    "dosyalarda **HER node'un** kesin etiketi yazılır: ihlaller (y=1), "
    "uyumlu değişiklikler ve tüm yapı (y=0). Baseline garantili temiz "
    "olduğu için 'geri kalan hepsi temiz' demek güvenli."
)
st.info(
    f"📏 Kurallar: Kapı ≥ {MIN_DOOR_WIDTH_M*100:.0f} cm · "
    f"Kapı önü serbest ≥ {DOOR_CLEARANCE_M*100:.0f} cm. "
    "Etiketler kuralla ölçülür (ground truth dürüst)."
)

# --- Paket ----------------------------------------------------------------
st.subheader("1. Baseline paketi")
try:
    tags = storage.list_dataset_tags()
except Exception as e:
    st.error(f"DB okunamadı: {e}"); st.stop()
base_tags = [t for t in tags if t.get("baseline", 0) > 0]
if not base_tags:
    st.warning("Baseline içeren paket yok. Önce Sentetik Üretim ile üret.")
    st.stop()
import pandas as pd
st.dataframe(pd.DataFrame(base_tags)[["tag", "baseline", "violated", "total"]],
             hide_index=True, use_container_width=True)
tag = st.selectbox("📦 Paket", [t["tag"] for t in base_tags])

# --- Parametreler (Basic Injection ile aynı) ------------------------------
st.subheader("2. Parametreler")
mode = st.radio("Üretim modu",
                ["⚡ Kural tabanlı (hızlı)", "🤖 GPT destekli (çeşitli)"],
                horizontal=True)
use_gpt = mode.startswith("🤖")
gpt_model = st.text_input("GPT modeli", "gpt-4o-mini") if use_gpt else "gpt-4o-mini"

c1, c2, c3 = st.columns(3)
with c1:
    variants = st.number_input("🔁 Baseline başına varyant", 1, 50, 5)
    seed_start = st.number_input("Tohum başlangıç", 0, 99999, 2000)
with c2:
    door_mod = st.slider("Değiştirilecek kapı oranı", 0.0, 1.0, 0.6, 0.1)
    door_vio = st.slider("Dar (ihlal) kapı oranı", 0.0, 1.0, 0.5, 0.1)
with c3:
    col_ratio = st.slider("Kolon konan kapı oranı", 0.0, 1.0, 0.6, 0.1)
    col_block = st.slider("Engelleyici kolon oranı", 0.0, 1.0, 0.5, 0.1)

params = BasicParams(
    door_modify_ratio=door_mod, door_violation_ratio=door_vio,
    column_ratio=col_ratio, column_block_ratio=col_block,
)
n_baseline = next((t["baseline"] for t in base_tags if t["tag"] == tag), 0)
st.metric("✨ Üretilecek violated IFC", n_baseline * int(variants))

# --- Çalıştır --------------------------------------------------------------
st.subheader("3. Çalıştır (üret + tam etiketle)")
if st.button("✅ Tam etiketli dataset üret", type="primary"):
    bar = st.progress(0.0, text="başlatılıyor...")
    log = st.empty(); logs: list[str] = []; t0 = time.time()

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
            full_label=True,   # ← HER node etiketlenir
            method_label="tametiket",
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar.empty()
    st.success(f"🎉 Bitti — {time.time()-t0:.1f}s. {res['ok']} violated IFC.")
    m = st.columns(4)
    m[0].metric("Violated IFC", res["ok"])
    m[1].metric("İhlal (y=1)", res["violations"])
    m[2].metric("Hard negatif (y=0)", res["hard_negatives"])
    m[3].metric("Kesin-temiz node (y=0)", res.get("clean_labeled", 0),
                help="Tam etiketleme ile açıkça 'temiz' işaretlenen yapı node'ları")

    # 0 üretildiyse NEDENİNİ göster — sessiz başarısızlığı kır
    if res["ok"] == 0:
        st.error(
            f"🔴 HİÇ violated üretilemedi! ({res.get('err', 0)} hata). "
            "İlk hataları aşağıda gör — inject_basic bu baseline'larda "
            "çalışmıyor demektir."
        )
        _errs = [it for it in res.get("items", []) if "error" in it]
        for it in _errs[:5]:
            st.code(f"{it['stem']}\n  → {it['error']}", language="text")
        if not _errs:
            st.warning(
                "Hata kaydı yok ama 0 üretildi — muhtemelen bu pakette "
                "geçerli (status=ok) baseline yok ya da hepsi graph'sız. "
                "Sentetik Üretim ile YENİ baseline üret, sonra burayı çalıştır."
            )
    gerrs = res.get("graph_errors", [])
    if gerrs:
        st.error(f"⚠️ {len(gerrs)} IFC'de graph hatası (eğitilemez):")
        for ge in gerrs[:5]:
            st.code(ge, language="text")
    if res["ok"] > 0:
        st.caption(
            "Sonraki: GAT Eğitim → bu paketi seç → eğit. Her node kesin etiketli."
        )

with st.expander("📖 Basic Injection'dan farkı"):
    st.markdown("""
| | Basic Injection | Tam Etiketleme |
|---|---|---|
| Üretim | kapı + kolon ihlali | **aynı** |
| Etiket | sadece değişen node'lar (ihlal + hard-neg) | **HER node** (+ tüm yapı 'clean') |
| Closed-world | etiketsiz=negatif (varsayım) | **her node açık** (varsayım yok) |

Tam etiketleme, modele "bu duvar/döşeme/dokunulmamış kapı KESİN normal"
sinyalini de açıkça verir. Baseline garantili compliant olduğundan bu
güvenli. Sonuç: daha zengin negatif → daha iyi precision.
""")
