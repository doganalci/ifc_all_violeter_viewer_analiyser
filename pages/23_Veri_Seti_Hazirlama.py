"""Sayfa 23 — Veri Seti Hazırlama.

Birden fazla paket etiketini (dataset_tag) tek bir adlandırılmış
'veri seti' olarak kaydet. Eğitim sayfası bu kaydedilmiş dataset'i
seçince ilgili tag'leri otomatik yükler.

Akış:
  1. Mevcut veri setleri tablosu
  2. Yeni veri seti formu (ad + paket multiselect + not)
  3. Detay paneli: seçilen dataset'in paketleri (baseline/violated/model)
  4. Düzenle (paket ekle/çıkar) + sil
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from services import datasets as ds
from violation_pool import storage


st.set_page_config(page_title="Veri Seti Hazırlama",
                   layout="wide", page_icon="📚")
st.title("📚 Veri Seti Hazırlama")
st.caption(
    "Birden fazla paketi tek bir adlandırılmış veri setine birleştir. "
    "Eğitim sayfası bu setleri tek tıkla yükler; sonradan paket "
    "ekleyebilir / çıkarabilirsin."
)


# --- Paket envanteri (yardımcı) ------------------------------------------
@st.cache_data(ttl=10, show_spinner=False)
def _all_tags() -> list[dict]:
    return storage.list_dataset_tags()


all_tags = _all_tags()
tag_index = {t["tag"]: t for t in all_tags}


def _composition_table(tags: list[str]) -> pd.DataFrame:
    rows = []
    for t in tags:
        info = tag_index.get(t)
        if not info:
            rows.append({
                "Paket": t, "Baseline": "—", "Violated": "—",
                "Toplam": "—", "Not": "⚠️ DB'de yok",
            })
        else:
            rows.append({
                "Paket": t,
                "Baseline": info.get("baseline", 0),
                "Violated": info.get("violated", 0),
                "Toplam": info.get("total", 0),
                "Not": "",
            })
    return pd.DataFrame(rows)


# --- 1. Mevcut veri setleri ----------------------------------------------
st.subheader("1. Kayıtlı veri setleri")
existing = ds.list_datasets()

if not existing:
    st.info("Henüz veri seti yok. Aşağıdan yeni bir tane oluştur.")
else:
    rows = []
    for d in existing:
        comp = _composition_table(d.get("tags") or [])
        b_sum = comp["Baseline"].apply(
            lambda x: int(x) if isinstance(x, (int, float)) else 0).sum()
        v_sum = comp["Violated"].apply(
            lambda x: int(x) if isinstance(x, (int, float)) else 0).sum()
        rows.append({
            "Slug": d.get("slug"),
            "Ad": d.get("name"),
            "Paket #": len(d.get("tags") or []),
            "Baseline ∑": int(b_sum),
            "Violated ∑": int(v_sum),
            "Oluşturma": (d.get("created_at") or "")[:19],
            "Güncel": (d.get("updated_at") or "")[:19],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 use_container_width=True)


# --- 2. Yeni veri seti ---------------------------------------------------
st.subheader("2. Yeni veri seti")
nc1, nc2 = st.columns([1, 2])
with nc1:
    new_name = st.text_input(
        "Ad", placeholder="ör. training_v1",
        help="Boşluk/özel karakterler '_' ile slugify edilir.",
    )
    new_notes = st.text_area("Not (opsiyonel)", height=100,
                             placeholder="örn. ilk eğitim deneyi, ...")
with nc2:
    new_tags = st.multiselect(
        "📦 Paketler",
        options=[t["tag"] for t in all_tags],
        format_func=lambda t: (
            f"{t} · base={tag_index[t].get('baseline', 0)} · "
            f"vio={tag_index[t].get('violated', 0)}"
        ),
        help="Bu paketlerdeki TÜM baseline + violated IFC'ler veri sete "
             "dahil olur.",
    )
    if new_tags:
        st.caption(f"✓ {len(new_tags)} paket seçili.")

if st.button("➕ Kaydet", type="primary",
             disabled=not (new_name.strip() and new_tags)):
    try:
        doc = ds.save_dataset(new_name, new_tags, new_notes)
        st.success(f"✓ `{doc['slug']}` oluşturuldu "
                   f"({len(doc['tags'])} paket).")
        st.rerun()
    except FileExistsError as e:
        st.error(f"⚠️ {e}")
    except Exception as e:
        st.error(f"Kaydedilemedi: {e}")


# --- 3. Detay / Düzenle / Sil --------------------------------------------
if existing:
    st.subheader("3. Detay / düzenle")
    slug_map = {d["slug"]: d for d in existing}
    sel = st.selectbox(
        "Veri seti seç",
        options=list(slug_map.keys()),
        format_func=lambda s: f"{slug_map[s]['name']} (`{s}`)",
    )
    doc = slug_map[sel]

    dc1, dc2 = st.columns([2, 1])
    with dc1:
        st.markdown(f"### {doc['name']}")
        st.caption(
            f"slug `{doc['slug']}` · oluşturma {doc.get('created_at', '?')} · "
            f"güncel {doc.get('updated_at', '?')}"
        )
        if doc.get("notes"):
            st.markdown(f"> _{doc['notes']}_")
        st.markdown(f"**Kompozisyon** ({len(doc.get('tags') or [])} paket):")
        comp = _composition_table(doc.get("tags") or [])
        st.dataframe(comp, hide_index=True, use_container_width=True)

        # Örnek IFC isimleri (her paketten ilk birkaç)
        with st.expander("📋 Örnek IFC isimleri (her paketten ilk 3)"):
            for t in doc.get("tags") or []:
                try:
                    ids = storage.ifc_ids_for_tags([t], kind=None)[:6]
                    names = []
                    for i in ids:
                        m = storage.get_ifc_model(i)
                        if m:
                            kind = m.get("kind", "?")
                            names.append(f"  • [{kind}] {m.get('name')}")
                    st.markdown(f"**`{t}`**")
                    st.code("\n".join(names) or "(boş)", language="text")
                except Exception as e:
                    st.caption(f"`{t}` okunamadı: {e}")

    with dc2:
        st.markdown("### 📝 Düzenle")
        new_tag_list = st.multiselect(
            "Paketler",
            options=[t["tag"] for t in all_tags],
            default=[t for t in (doc.get("tags") or [])
                     if t in tag_index],
            key=f"edit_tags_{sel}",
            format_func=lambda t: (
                f"{t} · b={tag_index[t].get('baseline', 0)}"
                f"·v={tag_index[t].get('violated', 0)}"
            ),
        )
        edit_notes = st.text_area(
            "Not", value=doc.get("notes") or "", height=80,
            key=f"edit_notes_{sel}",
        )
        bc1, bc2 = st.columns(2)
        if bc1.button("💾 Güncelle", use_container_width=True,
                      key=f"upd_{sel}"):
            try:
                ds.update_dataset(sel, tags=new_tag_list, notes=edit_notes)
                st.success("✓ Güncellendi.")
                st.rerun()
            except Exception as e:
                st.error(f"Güncelleme hatası: {e}")
        with bc2:
            confirm = st.checkbox("Sil onayı", key=f"del_confirm_{sel}")
            if st.button("🗑️ Sil", type="secondary",
                         use_container_width=True,
                         disabled=not confirm, key=f"del_{sel}"):
                try:
                    ds.delete_dataset(sel)
                    st.success(f"`{sel}` silindi.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Silinemedi: {e}")


st.divider()
st.info(
    "💡 **Eğitim sayfası entegrasyonu:** Eğitim sayfası üstünde "
    "'📚 Kayıtlı veri seti' seçtiğinde bu paketlerin tag'leri otomatik "
    "yüklenir. Kayıtlı dataset seçmeden manuel tag seçimi de eskisi gibi "
    "çalışır."
)
