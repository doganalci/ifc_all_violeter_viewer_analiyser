"""Sentetik baseline üretim — 2 odalı + koridor, mevzuata uygun.

Kullanım: parametreleri ayarla, sayıyı seç, **Üret** bas. Her IFC IFC_DATA_HOME/
ifc_models/baseline/ altına yazılır ve DB'ye baseline olarak kaydedilir.
Sonra normal pipeline (üst sekme 'Üretim & Pipeline') ile bunlara violation
enjekte edebilirsin.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from paths import ifc_models_dir
from ml.data.synth_baseline import (
    MIN_CORRIDOR_WIDTH_MM, MIN_DOOR_HEIGHT_MM, MIN_DOOR_WIDTH_MM,
    SynthParams, generate_batch, make_spec,
)


st.set_page_config(page_title="Sentetik Üretim", layout="wide", page_icon="🏗️")
st.title("🏗️ Sentetik Baseline Üretim")
st.caption(
    "**Mevzuata uygun** 2-oda + koridor IFC'leri parametrik üret. Bu baseline'lar "
    "garantili clean (gizli ihlal yok) — closed-world supervision varsayımı "
    "bunlar üzerinde DOĞRU olur. Üretildikten sonra codex1 pipeline'ı ile bunlara "
    "violation enjekte ederek temiz bir eğitim seti elde edersin."
)

st.info(
    f"💡 Garanti edilen eşikler: "
    f"Kapı genişliği ≥ {MIN_DOOR_WIDTH_MM/10:.0f} cm  ·  "
    f"Kapı yüksekliği ≥ {MIN_DOOR_HEIGHT_MM/10:.0f} cm  ·  "
    f"Koridor ≥ {MIN_CORRIDOR_WIDTH_MM/10:.0f} cm. "
    "Parametre uzayında bu eşiklerin **altına** inilmez."
)

st.subheader("1. Parametre uzayı")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Oda boyutu (m)**")
    rw = st.slider("Oda genişliği aralığı", 2.5, 8.0, (3.0, 5.5), 0.1)
    rl = st.slider("Oda uzunluğu aralığı", 2.5, 8.0, (3.5, 6.0), 0.1)
    st.markdown("**Koridor (m)**")
    cw = st.slider("Koridor genişliği aralığı (alt 1.2)", 1.2, 3.0, (1.4, 2.2), 0.05)
    cl = st.slider("Koridor uzunluğu aralığı", 1.5, 6.0, (2.0, 5.0), 0.1)
with c2:
    st.markdown("**Kapılar (m)**")
    dw = st.slider("Kapı genişliği aralığı (alt 0.90)", 0.90, 1.50, (0.95, 1.20), 0.05)
    dh = st.slider("Kapı yüksekliği aralığı (alt 2.00)", 2.00, 2.50, (2.10, 2.30), 0.05)
    storey_height = st.number_input("Kat yüksekliği (m)", 2.4, 5.0, 3.0, 0.1)
    wall_thickness = st.number_input("Duvar kalınlığı (m)", 0.10, 0.40, 0.20, 0.01)
    add_windows = st.checkbox("Pencere ekle (görsellik)", value=True)

params = SynthParams(
    room_w_min=rw[0], room_w_max=rw[1],
    room_l_min=rl[0], room_l_max=rl[1],
    corridor_w_min=cw[0], corridor_w_max=cw[1],
    corridor_l_min=cl[0], corridor_l_max=cl[1],
    door_w_min=dw[0], door_w_max=dw[1],
    door_h_min=dh[0], door_h_max=dh[1],
    storey_height=storey_height,
    wall_thickness=wall_thickness,
    add_windows=add_windows,
)

# ---- Önizleme: tek spec ----------------------------------------------------
st.subheader("2. Önizleme (rastgele 1 örnek)")
pv_cols = st.columns([1, 3])
with pv_cols[0]:
    preview_seed = st.number_input("Önizleme seed", 0, 99999, 42, 1)
    if st.button("🔄 Önizleme yenile", use_container_width=True):
        st.rerun()
with pv_cols[1]:
    preview = make_spec(int(preview_seed), params=params)
    st.json(preview, expanded=False)

# ---- Üretim ---------------------------------------------------------------
st.subheader("3. Toplu üretim")

# Dataset etiketi — her üretim ayrı bir 'paket' (klasör + DB tag)
# Default: 'basic2+1_baseline' + auto-incrementing version
import datetime as _dt
import re as _re
from violation_pool import storage as _storage

_BASE_PREFIX = "basic2+1_baseline"
try:
    _existing_tags = {t["tag"] for t in _storage.list_dataset_tags()}
except Exception:
    _existing_tags = set()
# v01, v02, ... boş ilk numarayı bul
_next_v = 1
while f"{_BASE_PREFIX}_v{_next_v:02d}" in _existing_tags:
    _next_v += 1
_default_tag = f"{_BASE_PREFIX}_v{_next_v:02d}"

tag_cols = st.columns([3, 2])
with tag_cols[0]:
    dataset_tag = st.text_input(
        "📦 Dataset adı (etiket)",
        value=_default_tag,
        help="Bu üretim ayrı bir klasöre yazılır + DB'de bu etiketle "
             "işaretlenir. Eğitim ve enjeksiyon sayfalarında dataset "
             "seçerken bu isim görünür. Tekrar aynı isim verirsen yan yana "
             "eklenir (üzerine yazmaz).",
    )
with tag_cols[1]:
    safe_tag = "".join(c if c.isalnum() or c in "-_+" else "_" for c in dataset_tag.strip()) or _default_tag
    st.caption(f"🗂 Klasör: `baseline/{safe_tag}/`")
    if _existing_tags:
        with st.popover("📋 Mevcut paketler", use_container_width=True):
            for t in sorted(_existing_tags):
                st.caption(f"• `{t}`")

gen_cols = st.columns([2, 1, 1, 2])
with gen_cols[0]:
    n = st.number_input("Üretilecek IFC sayısı", 1, 1000, 50, 1)
with gen_cols[1]:
    seed_start = st.number_input("Seed başlangıç", 0, 999999, 0, 1)
with gen_cols[2]:
    register_db = st.checkbox("DB'ye kaydet", value=True,
                               help="Baseline olarak SQLite'a yaz")
with gen_cols[3]:
    out_dir = st.text_input("Çıkış klasörü",
                              value=str(ifc_models_dir() / "baseline" / safe_tag))

if st.button("🏗️ Üret", type="primary", use_container_width=True):
    progress = st.progress(0.0, text="başlatılıyor...")
    log = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(i, total, info):
        progress.progress(i / total, text=f"{i}/{total}  ·  {Path(info['ifc_path']).name}")
        logs.append(f"  ✓ [{i}/{total}] seed={info['spec']['_meta']['seed']}  "
                    f"corridor={info['spec']['_meta']['corridor_width_cm']}cm")
        log.code("\n".join(logs[-30:]))

    try:
        results = generate_batch(
            int(n), out_dir,
            seed_start=int(seed_start),
            params=params,
            register_in_db=register_db,
            dataset_tag=safe_tag,
            progress_cb=_cb,
        )
    except Exception as e:
        st.exception(e)
        st.stop()
    progress.empty()
    dt = time.time() - t0
    st.success(f"✅ {len(results)} baseline üretildi ({dt:.1f} s)")
    st.caption(f"📁 {out_dir}")
    if register_db:
        st.caption("📊 DB'ye baseline olarak kaydedildi — codex1 pipeline'ında "
                   "üst sekmede 'Üretim' → bu baseline'lara violation enjekte edebilirsin.")
    with st.expander(f"📋 Üretilen dosyalar ({len(results)})"):
        for r in results:
            st.caption(f"`{Path(r['ifc_path']).name}` · seed={r['spec']['_meta']['seed']} "
                       f"· id={r['ifc_id'][:8]}")
