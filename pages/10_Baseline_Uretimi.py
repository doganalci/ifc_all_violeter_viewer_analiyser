"""Baseline Üretimi — sentetik, kuralsal (LLM'siz).

Tek bir "ana baseline" adı altında N tane varyant IFC üretir. Her varyant
aynı şemada (2 oda + 1 koridor veya 3 oda + 1 koridor) ama farklı boyut
örneklemesiyle. Tüm dosyalar aynı dataset_tag ile DB'ye baseline olarak
kayıt edilir → sonraki adımda bu paketten ihlalli IFC'ler üretilecek.

Bu sayfa yalnızca IFC oluşturur — ihlal eklenmez.
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
from ml.data.synth_baseline_v2 import SynthParamsV2, generate_batch_v2


st.set_page_config(page_title="Baseline Üretimi", layout="wide", page_icon="🏠")
st.title("🏠 Baseline Üretimi")
st.caption(
    "Tek bir **ana baseline adı** altında N adet varyant IFC üretir "
    "(LLM yok, kuralsal). Bu paket sonraki adımda ihlal enjeksiyonuna girer."
)

st.info(
    "📏 Tüm baseline'lar mevzuata uygun aralıklarda üretilir "
    "(kapı genişliği ≥ 0.95 m, yükseklik ≥ 2.10 m, koridor ≥ 1.40 m). "
    "İhlal yok — bu pakete sonraki adımda ihlal enjekte edilecek."
)


# --- Ana baseline adı -----------------------------------------------------
st.subheader("1. Ana baseline adı")
c_name, c_help = st.columns([2, 1])
with c_name:
    ana_ad = st.text_input(
        "📛 Ana baseline adı (dataset_tag)",
        value="ana_baseline_v1",
        help="Bu paketin etiketi. Üretilen tüm IFC'ler bu adın altında "
             "gruplanır ve dosya adları bununla başlar. Örn: ofis_v1 → "
             "ofis_v1_00000.ifc, ofis_v1_00001.ifc, ...",
    )
with c_help:
    st.caption("Sadece harf/rakam/_/-/+ kullan. Boşluk ve özel karakter "
               "otomatik _'a çevrilir.")

ana_ad_clean = re.sub(r"[^A-Za-z0-9_\-+]", "_", ana_ad.strip()) or "ana_baseline"
if ana_ad_clean != ana_ad:
    st.caption(f"⚠️ Temizlenmiş ad: `{ana_ad_clean}`")

# Var olan paketlerde aynı tag var mı?
try:
    existing_tags = storage.list_dataset_tags()
except Exception as e:
    existing_tags = []
    st.warning(f"DB okunamadı: {e}")
existing_tag_names = {t["tag"] for t in existing_tags}
if ana_ad_clean in existing_tag_names:
    matched = next(t for t in existing_tags if t["tag"] == ana_ad_clean)
    st.warning(
        f"ℹ️ `{ana_ad_clean}` paketi zaten var "
        f"({matched.get('baseline', 0)} baseline + {matched.get('violated', 0)} "
        f"violated). Yeni varyantlar bu pakete EKLENECEK (silinmez)."
    )


# --- Tasarım -------------------------------------------------------------
st.subheader("2. Tasarım")
layout_sec = st.radio(
    "Yerleşim",
    options=["2 oda + 1 koridor", "3 oda + 1 koridor"],
    horizontal=True,
)
n_rooms = 2 if layout_sec.startswith("2") else 3

with st.expander("ℹ️ Bu yerleşim ne demek?"):
    st.markdown("""
- **2 oda + 1 koridor**: Klasik şema. Koridor ortada, oda batı ve doğusunda.
  Zemin katta koridorun güney duvarında dış giriş kapısı.
- **3 oda + 1 koridor**: Yukarısı + 3. oda koridorun **kuzey** kanadında.
  Dış giriş güney duvarında.

Tüm varyantlarda oda/koridor/kapı boyutları **rastgele** ama mevzuata uygun
aralıklarda örneklenir (her tohumda farklı). Dış cephelere pencere otomatik
eklenir.
""")


# --- Üretim parametreleri ------------------------------------------------
st.subheader("3. Üretim")
c1, c2 = st.columns(2)
with c1:
    n_variants = st.number_input(
        "🔁 Varyant sayısı (üretilecek IFC)",
        min_value=1, max_value=500, value=10,
        help="Bu ana baseline adı altında kaç tane farklı IFC üretelim?",
    )
with c2:
    seed_start = st.number_input(
        "🎲 Tohum başlangıcı",
        min_value=0, max_value=99999, value=0,
        help="Tekrar üretilebilirlik için. Aynı tohum + aynı ad → aynı IFC.",
    )

st.metric("✨ Üretilecek IFC", int(n_variants))
st.caption(
    f"📁 Çıkış: `data/ifc_models/baseline/{ana_ad_clean}_{seed_start:05d}.ifc` "
    f"… `{ana_ad_clean}_{seed_start + int(n_variants) - 1:05d}.ifc`"
)


# --- Çalıştır ------------------------------------------------------------
st.subheader("4. Üret")
if st.button("🏠 Baseline'ları üret", type="primary", use_container_width=True):
    params = SynthParamsV2(
        n_rooms_per_floor=n_rooms,
        n_storeys=1,
        layout="straight",
    )
    bar = st.progress(0.0, text="başlatılıyor...")
    log_slot = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, info):
        bar.progress(done / total,
                     text=f"{done}/{total} · {Path(info['ifc_path']).name}")
        logs.append(f"  ✓ {Path(info['ifc_path']).name}")
        if len(logs) % 3 == 0 or done == total:
            log_slot.code("\n".join(logs[-15:]))

    out_dir = settings.ifc_dir / "baseline"
    try:
        results = generate_batch_v2(
            n=int(n_variants),
            out_dir=out_dir,
            seed_start=int(seed_start),
            params=params,
            dataset_tag=ana_ad_clean,
            progress_cb=_cb,
        )
    except Exception as e:
        st.exception(e); st.stop()
    bar.empty()
    dt = time.time() - t0

    n_ok = len([r for r in results if r.get("ifc_path")])
    st.success(f"🎉 Bitti — {dt:.1f}s · paket `{ana_ad_clean}` · {n_ok} baseline.")
    m = st.columns(3)
    m[0].metric("Üretilen", n_ok)
    m[1].metric("Yerleşim", f"{n_rooms} oda + koridor")
    m[2].metric("Tohum aralığı",
                f"{int(seed_start)}-{int(seed_start) + n_ok - 1}")

    # Üretilen dosyaların özet tablosu
    rows = []
    for r in results:
        spec_meta = r.get("spec", {}).get("_meta", {})
        rows.append({
            "Dosya": Path(r["ifc_path"]).name,
            "Tohum": spec_meta.get("seed"),
            "Oda/Kat": spec_meta.get("n_rooms_per_floor"),
            "Kapı genişlikleri (cm)": ", ".join(
                f"{v:.0f}" for v in
                (spec_meta.get("door_widths_cm") or {}).values()
            ),
        })
    if rows:
        with st.expander(f"📋 Üretilen {len(rows)} IFC — detay",
                         expanded=False):
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

    st.caption(
        "Sonraki adım: bu pakete LLM + RAG ile **ihlal enjeksiyonu** "
        "yapılacak (henüz hazır değil — birlikte planlıyoruz)."
    )


# --- Mevcut paketler -----------------------------------------------------
st.divider()
st.subheader("📦 Mevcut paketler (durum kontrolü)")
if existing_tags:
    df = pd.DataFrame(existing_tags)[["tag", "baseline", "violated", "total"]]
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.caption("Henüz paket yok.")
