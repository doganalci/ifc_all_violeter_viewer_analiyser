"""Tam Etiketleme — baseline'ları KESİN temiz olarak işaretle.

Mantık: Sentetik baseline'lar garantili compliant (eşik-altı parametre
seçilemez). O halde her node'unu **kesin 'ihlal değil'** olarak
işaretleyebiliriz. Bu:
  * Closed-world varsayımını GEÇERLİ kılar (varsayım değil, bilinen gerçek)
  * Modele bol GÜVENİLİR NEGATİF verir → over-flagging azalır → precision ↑

Bu sayfa seçilen paketin baseline'larına 'hepsi temiz' labels.json yazar
ve eğitime 'kesin negatif' olarak girmeye hazırlar. Violated IFC'ler
zaten ihlalleri işaretli; bu ikisi birlikte tam etiketli set olur.

Sonra GAT Eğitim'de **'Baseline'ları dahil et'** açılırsa bu temiz
baseline'lar negatif örnek olarak eğitime katılır.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from violation_pool import storage
from ml.data.graph_loader import load_graph

st.set_page_config(page_title="Tam Etiketleme", layout="wide", page_icon="✅")
st.title("✅ Tam Etiketleme — Baseline = Kesin Temiz")
st.caption(
    "Sentetik baseline'lar garantili compliant. Tüm node'larını **kesin "
    "'ihlal değil'** olarak işaretle → modele güvenilir negatif örnek. "
    "Violated IFC'lerle birlikte **tam etiketli** (her node'un kesin "
    "kararı belli) bir eğitim seti olur."
)

st.info(
    "💡 Neden işe yarar: Model 'normal neye benzer'i bol örnekle öğrenir → "
    "şüpheli görmediği şeye 'ihlal' deme eğilimi azalır → **precision artar** "
    "(daha az yanlış alarm). Recall'ı düşürmeden FP'yi kırpar."
)

# --- Paket seçimi ----------------------------------------------------------
st.subheader("1. Baseline paketi seç")
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

# --- Çalıştır --------------------------------------------------------------
st.subheader("2. Baseline'ları 'kesin temiz' etiketle")
st.caption(
    "Her baseline için labels.json yazılır: tüm node'lar `status='clean'`, "
    "`is_violation=False`. Eğitimde y=0 (kesin negatif) olur."
)

if st.button("✅ Baseline'ları tam-temiz etiketle", type="primary"):
    base_ids = storage.ifc_ids_for_tags([tag], kind="baseline")
    baselines = [storage.get_ifc_model(i) for i in base_ids]
    baselines = [b for b in baselines if b and b.get("status") == "ok"]
    if not baselines:
        st.error("Bu pakette geçerli baseline yok.")
        st.stop()

    bar = st.progress(0.0, text="başlatılıyor...")
    log = st.empty()
    logs: list[str] = []
    total_nodes = 0
    n_ok = 0
    t0 = time.time()

    for k, b in enumerate(baselines):
        gp = b.get("graph_path")
        if not gp or not Path(gp).exists():
            logs.append(f"  ⚠️ {Path(b['file_path']).stem}: graph yok, atlandı")
            continue
        try:
            g = load_graph(gp)
            nodes = list(g.nodes())
            doc = {
                "violated_id": b["id"],
                "baseline_id": b["id"],
                "source": "full_clean_label",
                "all_clean": True,
                "labels": [
                    {
                        "ifc_global_id": n,
                        "category": "",
                        "severity": "uygun",
                        "status": "clean",       # eğitimde y=0
                        "is_violation": False,
                        "is_decoy": False,
                        "attribute": None,
                        "evidence": "Baseline garantili compliant — kesin temiz",
                    }
                    for n in nodes
                ],
            }
            # labels.json'u baseline'ın yanına yaz
            lab_path = Path(gp).with_name(Path(gp).name.replace(".graph.json",
                                                                ".labels.json"))
            lab_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            # DB'de labels_path güncelle (yoksa)
            try:
                with storage._conn() as c:
                    c.execute("UPDATE ifc_models SET labels_path=? WHERE id=?",
                              (str(lab_path), b["id"]))
            except Exception:
                pass
            total_nodes += len(nodes)
            n_ok += 1
            logs.append(f"  ✓ {Path(b['file_path']).stem}: {len(nodes)} node temiz")
        except Exception as e:
            logs.append(f"  ✗ {Path(b['file_path']).stem}: {e}")
        bar.progress((k + 1) / len(baselines))
        if len(logs) % 5 == 0 or k == len(baselines) - 1:
            log.code("\n".join(logs[-20:]))

    bar.empty()
    st.success(
        f"🎉 {n_ok} baseline tam-temiz etiketlendi ({time.time()-t0:.1f}s). "
        f"Toplam **{total_nodes}** node kesin negatif (y=0) olarak hazır."
    )
    st.caption(
        "Sonraki: **GAT Eğitim** → aynı paketi seç → **'Baseline'ları dahil "
        "et'** kutusunu İŞARETLE. Bu temiz baseline'lar negatif örnek olarak "
        "eğitime girer, precision'ı yükseltir."
    )

# --- Bilgi -----------------------------------------------------------------
with st.expander("📖 Bu nasıl yardımcı olur?"):
    st.markdown("""
**Sorun:** Şu an eğitim sadece violated IFC'lerle yapılınca, modelin
gördüğü tek 'normal' örnek o binaların ihlalsiz node'ları. Yeterince
çeşitli normal görmeyince, şüpheli her şeye 'ihlal' demeye eğilimli →
düşük precision (çok yanlış alarm).

**Çözüm:** Garantili temiz baseline'ları da kesin-negatif olarak ekle.
Model 'normal bina neye benzer'i çok daha geniş örnekle öğrenir →
değişmiş-ama-uyumlu kapıları, uzak kolonları daha iyi 'normal' der →
**precision artar, recall korunur.**

**Closed-world geçerliliği:** Normalde 'etiketsiz = negatif' bir
VARSAYIM (riskli). Ama sentetik baseline'da gerçekten ihlal YOK, o yüzden
'hepsi temiz' demek bir varsayım değil, **bilinen gerçek**. Bu yüzden bu
veride güvenli.
""")
