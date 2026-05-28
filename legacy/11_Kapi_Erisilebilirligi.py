"""Kapı Erişilebilirliği Deneyi — tek sekmede uçtan uca.

Bu sayfa TEK tıkla:
  1) Kapı-zengin sentetik baseline'lar üretir (oda/koridor boyutları çeşitli,
     boş duvarlara ek kapılar).
  2) Bu baseline'lara SADECE KAPI ihlali enjekte eder (kolon yok):
       * kapı genişliği < 0.90 m  → ihlal
       * kapı yüksekliği < 2.00 m → ihlal
     Uyumlu ama değiştirilmiş kapılar = hard negative.
  3) Tam etiketleme yapar (her node açıkça etiketli).

Girdi: ana baseline'dan kaç baseline + her baseline'dan kaç tam-etiketli
ihlalli IFC. Hepsi tek pakette toplanır.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from paths import ifc_models_dir
from ml.data.synth_baseline import SynthParams, generate_batch
from violation_pool import storage
from violation_pool.basic_inject import (
    BasicParams, MIN_DOOR_WIDTH_M, MIN_DOOR_HEIGHT_M, run_basic_batch,
)

st.set_page_config(page_title="Kapı Erişilebilirliği Deneyi",
                   layout="wide", page_icon="🚪")
st.title("🚪 Kapı Erişilebilirliği Deneyi (tek sekme)")
st.caption(
    "Kapı-zengin baseline üret → SADECE kapı ihlali (genişlik veya yükseklik) "
    "enjekte et → tam etiketle. Hepsi tek tıkla, tek pakette."
)
st.info(
    f"📏 Kurallar: Kapı net genişliği ≥ **{MIN_DOOR_WIDTH_M*100:.0f} cm** · "
    f"Kapı net yüksekliği ≥ **{MIN_DOOR_HEIGHT_M*100:.0f} cm**. "
    "Bu eşiklerin altı ihlal; üstü ama değiştirilmiş = hard negative. Kolon yok."
)

# --- Paket adı -------------------------------------------------------------
st.subheader("1. Paket (ana baseline) adı")
_PREFIX = "kapi_deney"
try:
    _tags = {t["tag"] for t in storage.list_dataset_tags()}
except Exception:
    _tags = set()
_v = 1
while f"{_PREFIX}_v{_v:02d}" in _tags:
    _v += 1
tag = st.text_input("📦 Dataset adı", value=f"{_PREFIX}_v{_v:02d}")
safe_tag = "".join(c if c.isalnum() or c in "-_+" else "_" for c in tag.strip()) or f"{_PREFIX}_v{_v:02d}"

# --- Miktarlar -------------------------------------------------------------
st.subheader("2. Miktar")
m1, m2, m3, m4 = st.columns(4)
with m1:
    n_baseline = st.number_input("🏗️ Üretilecek baseline", 1, 2000, 100, 1,
                                 help="Ana baseline'dan kaç baseline üretilecek")
with m2:
    variants = st.number_input("🔁 Baseline başına ihlalli IFC", 1, 100, 10, 1,
                               help="Her baseline'dan kaç tam-etiketli ihlal")
with m3:
    base_seed = st.number_input("Baseline tohum başlangıç", 0, 999999, 0, 1)
with m4:
    inj_seed = st.number_input("İhlal tohum başlangıç", 0, 999999, 5000, 1)
st.metric("✨ Üretilecek toplam ihlalli IFC", int(n_baseline) * int(variants))

# --- Kapı / yapı parametreleri --------------------------------------------
st.subheader("3. Kapı & yapı parametreleri")
p1, p2 = st.columns(2)
with p1:
    st.markdown("**Yapı çeşitliliği**")
    extra_doors = st.slider("Ek kapı sayısı (boş duvarlara)", 0, 5, 4,
                            help="3 temel kapıya ek. 4 → toplam ~7 kapı.")
    rw = st.slider("Oda genişliği aralığı (m)", 2.5, 8.0, (3.0, 5.5), 0.1)
    rl = st.slider("Oda uzunluğu aralığı (m)", 2.5, 8.0, (3.5, 6.0), 0.1)
    cw = st.slider("Koridor genişliği aralığı (m)", 1.2, 3.0, (1.4, 2.4), 0.05)
with p2:
    st.markdown("**İhlal kurgusu**")
    dim_mode_label = st.radio(
        "Hangi boyut ihlal edilsin?",
        ["İkisi (genişlik+yükseklik)", "Sadece genişlik", "Sadece yükseklik"],
        help="'İkisi' → her kapı için rastgele genişlik VEYA yükseklik.",
    )
    dim_mode = {"İkisi (genişlik+yükseklik)": "both",
                "Sadece genişlik": "width",
                "Sadece yükseklik": "height"}[dim_mode_label]
    door_mod = st.slider("Değiştirilecek kapı oranı", 0.0, 1.0, 0.6, 0.1)
    door_vio = st.slider("İhlal (uygunsuz) kapı oranı", 0.0, 1.0, 0.5, 0.1,
                         help="Gerisi uyumlu ama değiştirilmiş = hard negative")

synth_params = SynthParams(
    room_w_min=rw[0], room_w_max=rw[1],
    room_l_min=rl[0], room_l_max=rl[1],
    corridor_w_min=cw[0], corridor_w_max=cw[1],
    extra_doors=int(extra_doors),
)
basic_params = BasicParams(
    door_modify_ratio=door_mod, door_violation_ratio=door_vio,
    door_only=True, door_dim_mode=dim_mode,
)

# --- Çalıştır --------------------------------------------------------------
st.subheader("4. Çalıştır (üret + ihlal + tam etiketle)")
if st.button("🚪 Deneyi üret", type="primary", use_container_width=True):
    out_dir = ifc_models_dir() / "baseline" / safe_tag
    t0 = time.time()

    # 1) Baseline üretimi
    st.markdown("**Adım 1/2 — Baseline üretimi**")
    bar1 = st.progress(0.0, text="baseline üretiliyor...")

    def _cb1(i, total, info):
        bar1.progress(i / total, text=f"baseline {i}/{total}")

    try:
        bres = generate_batch(
            int(n_baseline), out_dir, seed_start=int(base_seed),
            params=synth_params, register_in_db=True,
            dataset_tag=safe_tag, progress_cb=_cb1,
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar1.empty()
    st.success(f"✅ {len(bres)} baseline üretildi.")

    # 2) Sadece-kapı ihlali + tam etiketleme
    st.markdown("**Adım 2/2 — Kapı ihlali enjeksiyonu + tam etiketleme**")
    bar2 = st.progress(0.0, text="ihlal üretiliyor...")
    log = st.empty(); logs: list[str] = []

    def _cb2(done, total, stem):
        bar2.progress(done / total, text=f"{done}/{total} · {stem}")
        logs.append(f"  ✓ {stem}")
        if len(logs) % 5 == 0 or done == total:
            log.code("\n".join(logs[-15:]))

    try:
        res = run_basic_batch(
            safe_tag, variants=int(variants), seed_start=int(inj_seed),
            params=basic_params, register_in_db=True, progress_cb=_cb2,
            use_gpt=False, full_label=True, method_label="kapideney",
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar2.empty()

    st.success(f"🎉 Bitti — {time.time()-t0:.1f}s. {res['ok']} ihlalli IFC.")
    c = st.columns(4)
    c[0].metric("İhlalli IFC", res["ok"])
    c[1].metric("İhlal (y=1)", res["violations"])
    c[2].metric("Hard negatif (y=0)", res["hard_negatives"])
    c[3].metric("Kesin-temiz node (y=0)", res.get("clean_labeled", 0))

    if res["ok"] == 0:
        st.error(f"🔴 Hiç ihlalli IFC üretilemedi ({res.get('err', 0)} hata).")
        for it in [x for x in res.get("items", []) if "error" in x][:5]:
            st.code(f"{it['stem']}\n  → {it['error']}")
    else:
        st.caption(
            f"Sonraki: **GAT Eğitim** → paket `{safe_tag}` seç → eğit. "
            "Bağımsız test için ayrı isim/tohumla bir paket daha üret, "
            "**GAT Test → paket seç** ile test et."
        )
