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
from llm.excel_log import read_llm_totals
from llm.pricing import estimate_cost, known_models
from ml.data.synth_baseline_v2 import SynthParamsV2


def _fmt_int(n: int | float) -> str:
    """1234 → '1.234' (Türkçe binlik ayraç)."""
    return f"{int(n):,}".replace(",", ".")


def _render_persistent_counter(model_name: str) -> None:
    """Üst kısımda sabit token + maliyet sayaç widget'ı."""
    totals = read_llm_totals() or {
        "n_calls": 0, "total_tokens": 0,
        "total_cost_usd": 0.0, "total_duration_s": 0.0,
    }
    session_t = st.session_state.get("session_llm_totals", {
        "n_calls": 0, "tokens": 0, "cost": 0.0, "duration": 0.0,
    })
    mc = st.columns(4)
    mc[0].metric(
        "📒 Toplam üretim (tüm zamanlar)",
        _fmt_int(totals.get("n_calls", 0)),
        delta=(f"+{session_t['n_calls']} bu oturum"
               if session_t["n_calls"] else None),
    )
    mc[1].metric(
        "Token (toplam)",
        _fmt_int(totals.get("total_tokens", 0)),
        delta=(f"+{_fmt_int(session_t['tokens'])} bu oturum"
               if session_t["tokens"] else None),
    )
    mc[2].metric(
        "Maliyet ≈ (toplam)",
        f"${totals.get('total_cost_usd', 0):.4f}",
        delta=(f"+${session_t['cost']:.4f}"
               if session_t["cost"] > 0 else None),
    )
    mc[3].metric(
        "Süre (toplam, LLM)",
        f"{totals.get('total_duration_s', 0):.0f}s",
        delta=(f"+{session_t['duration']:.1f}s"
               if session_t["duration"] else None),
    )


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

# Sabit sayaç — sayfa başında her zaman görünür
_render_persistent_counter("placeholder")
try:
    from paths import data_home as _dh
    _excel_path = _dh() / "llm_generations.xlsx"
    st.caption(
        f"📒 `{_excel_path}` — her LLM çağrısı için bir satır kalıcı. "
        "**Bu klasör repo dışındadır → git pull/push'tan etkilenmez** "
        "(`IFC_DATA_HOME` env var ile yer değiştirilebilir)."
    )
except Exception:
    st.caption("📒 llm_generations.xlsx — her LLM çağrısı kalıcı kayıt")
st.divider()


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
    "- Her odanın koridora bir kapısı olsun\n"
    "- Erişilebilirlik açısından kapı ve koridor boyutları mevzuata uygun"
)
user_template = st.text_area(
    "💬 Bina tarifi",
    value=default_prompt, height=140,
    help="Bina tipi, işlev, stil özellikleri vs. **Sayılar (oda, salon, "
         "koridor, kat) aşağıdaki parametrelerden gelir** — buraya yazma. "
         "LLM bu metni + parametreleri okuyup TASARIM PARAMETRELERİ (JSON) "
         "üretir."
)

# --- Zorunlu sayısal kısıtlar (prompt'a dinamik enjekte) ----------------
st.markdown(
    "**🔢 Zorunlu sayılar** (LLM prompt'una otomatik eklenir + kod tarafında zorlanır)"
)
ncc = st.columns(4)
with ncc[0]:
    n_oda_in = st.number_input(
        "🚪 Oda sayısı (kat başına)", min_value=1, max_value=4, value=2,
        help="Salon + oda toplamı 2-4 olmalı. Salon dahil değil.",
    )
with ncc[1]:
    n_salon_in = st.number_input(
        "🛋️ Salon sayısı (kat başına)", min_value=0, max_value=3, value=1,
        help="0 = sadece odalar. ≥1 = ilk N büyük oda 'Salon' olarak adlandırılır.",
    )
with ncc[2]:
    n_kor_in = st.number_input(
        "🚶 Koridor sayısı (kat başına)", min_value=1, max_value=2, value=1,
        help="1 = straight (merkez koridor). 2 = lshape (iki perpendiküler).",
    )
with ncc[3]:
    n_kat_in = st.number_input(
        "🏢 Kat sayısı", min_value=1, max_value=3, value=1,
        help="Default 1 (tek kat). 2-3 katlı bina için artır. "
             "Her kat aynı şemayı (oda/salon/koridor) tekrar eder.",
    )

# Oda başına kapı sayısı — range slider
n_doors_range = st.slider(
    "🚪 Oda başına kapı sayısı (her odanın kaç kapısı olacak)",
    min_value=1, max_value=4, value=(1, 2), step=1,
    help="1. kapı koridora bakar (zorunlu). Ek kapılar dış cephe duvarlarına "
         "konur. LLM her oda için bu aralıkta kendi sayısını seçer; min=max "
         "yaparsan tüm odalar aynı sayıda olur.",
)

# --- Gelişmiş ayarlar (LLM'e tercih aralıkları olarak iletilir) ---------
with st.expander(
    "🔧 Gelişmiş ayarlar — boyut aralıkları (LLM'e tercih olarak iletilir)",
    expanded=False,
):
    st.caption(
        "Buradaki aralıklar **prompt'a 'tercih edilen aralık' olarak eklenir**. "
        "LLM bunları okur ve gerektiğinde uyar; zorunlu değil. Default değerler "
        "makul (mevzuata uygun). Çok dar bir aralık verirsen LLM o aralıkta "
        "kalmaya çalışır."
    )
    use_advanced = st.checkbox(
        "Gelişmiş ayarları LLM prompt'una ekle",
        value=False,
        help="Kapalıysa: LLM tamamen serbest karar verir. Açıksa: aralıklar "
             "prompt'a tercih olarak girer.",
    )

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Oda boyutu (m)**")
        adv_room_w = st.slider(
            "Oda genişliği (en)",
            min_value=2.0, max_value=8.0, value=(3.0, 5.5), step=0.1,
            key="adv_rw",
        )
        adv_room_l = st.slider(
            "Oda boyu",
            min_value=2.0, max_value=10.0, value=(3.5, 6.0), step=0.1,
            key="adv_rl",
        )
        st.markdown("**Kapı (m)**")
        adv_door_w = st.slider(
            "Kapı genişliği",
            min_value=0.90, max_value=1.50, value=(0.95, 1.20), step=0.05,
            key="adv_dw",
            help="≥ 0.90 m (TS 9111).",
        )
        adv_door_h = st.slider(
            "Kapı yüksekliği",
            min_value=2.00, max_value=2.50, value=(2.10, 2.30), step=0.05,
            key="adv_dh",
            help="≥ 2.00 m (TS 9111).",
        )
    with ac2:
        st.markdown("**Koridor (m)**")
        adv_corridor_w = st.slider(
            "Koridor genişliği",
            min_value=1.20, max_value=3.00, value=(1.40, 2.20), step=0.1,
            key="adv_cw",
            help="≥ 1.20 m (TS 9111).",
        )
        adv_corridor_l = st.slider(
            "Koridor uzunluğu",
            min_value=2.0, max_value=10.0, value=(3.0, 6.0), step=0.1,
            key="adv_cl",
        )
        st.markdown("**Yapı (m)**")
        adv_wall_t = st.slider(
            "Duvar kalınlığı",
            min_value=0.10, max_value=0.40, value=0.20, step=0.05,
            key="adv_wt",
        )
        adv_storey_h = st.slider(
            "Kat yüksekliği",
            min_value=2.70, max_value=3.50, value=(2.80, 3.20), step=0.05,
            key="adv_sh",
        )

    if use_advanced:
        # Mevzuat uyarısı
        if (adv_door_w[0] < 0.90 or adv_door_h[0] < 2.00
                or adv_corridor_w[0] < 1.20):
            st.warning(
                "⚠️ Bazı min değerler TS 9111 eşiğinin altında. "
                "LLM yine eşik altına inmemeli ama prompt'a 'tercih' "
                "olarak böyle gönderiyorsun."
            )
        st.caption(
            "✓ Bu aralıklar prompt'a **TERCİH EDİLEN ARALIKLAR** olarak girecek."
        )

# Doğrulama
n_total_rooms = int(n_oda_in) + int(n_salon_in)
if n_total_rooms < 2 or n_total_rooms > 4:
    st.error(
        f"⚠️ Oda + salon toplamı 2-4 arası olmalı. Şu an: {n_total_rooms}."
    )
    constraints_valid = False
else:
    st.caption(
        f"✓ {int(n_kat_in)} kat × ({int(n_oda_in)} oda + {int(n_salon_in)} salon "
        f"+ {int(n_kor_in)} koridor) · "
        f"layout `{'lshape' if int(n_kor_in) == 2 else 'straight'}` zorlanacak"
    )
    constraints_valid = True


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

mc = st.columns(4)
mc[0].metric("Ana baseline", 1)
mc[1].metric("Varyant", int(variants))
n_calls_est = int(variants) + 1
mc[2].metric("Toplam LLM çağrısı", n_calls_est)
# Tahmini maliyet — kabaca ~2000 input + ~700 output / çağrı
_est_per_call = estimate_cost(model, 2000, 700)
_est_total = _est_per_call["total_usd"] * n_calls_est
mc[3].metric(
    "🧮 Tahmini maliyet",
    f"${_est_total:.4f}",
    help=f"~{2700 * n_calls_est:,} token · {model} fiyatı"
         f"{' (tahmin)' if _est_per_call['is_estimate'] else ''}",
)

st.info(
    "💡 Bu sayfada **tüm boyut kararlarını LLM verir** (kat yüksekliği, duvar "
    "kalınlığı, koridor en/boy, **her odanın ayrı en/boyu**, iç + dış kapı "
    "boyutları). Boyutları etkilemek istersen bina tarifi prompt'una yaz "
    "(örn. \"odalar yaklaşık 4-5 m\", \"geniş koridor (2 m+)\")."
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
             use_container_width=True, disabled=not constraints_valid):
    bar = st.progress(0.0, text="başlatılıyor...")
    log_slot = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, label):
        bar.progress(done / total, text=f"{done}/{total} · {label}")
        logs.append(f"  • {label}")
        if len(logs) % 2 == 0 or done == total:
            log_slot.code("\n".join(logs[-15:]))

    # LLM tüm boyutları belirleyecek — params sadece fallback aralıkları içerir.
    g_params = SynthParamsV2()

    try:
        res = run_baseline_pipeline(
            user_template=user_template,
            dataset_tag=ana_ad_clean,
            variants=int(variants),
            seed_start=int(seed_start),
            model=model.strip() or "gpt-4o",
            params=g_params,
            constraints={
                "n_rooms": int(n_oda_in),
                "n_salons": int(n_salon_in),
                "n_corridors": int(n_kor_in),
                "n_storeys": int(n_kat_in),
                "n_doors_min": int(n_doors_range[0]),
                "n_doors_max": int(n_doors_range[1]),
                "prefs": (
                    {
                        "room_w": list(adv_room_w),
                        "room_l": list(adv_room_l),
                        "corridor_w": list(adv_corridor_w),
                        "corridor_l": list(adv_corridor_l),
                        "door_w": list(adv_door_w),
                        "door_h": list(adv_door_h),
                        "wall_t": float(adv_wall_t),
                        "storey_h": list(adv_storey_h),
                    }
                    if use_advanced else None
                ),
            },
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

    # Sürekli sayaç delta'sı için session state güncelle
    _t = res.get("totals", {})
    s = st.session_state.setdefault(
        "session_llm_totals",
        {"n_calls": 0, "tokens": 0, "cost": 0.0, "duration": 0.0},
    )
    s["n_calls"] += int(_t.get("n_calls", 0))
    s["tokens"] += int(_t.get("total_tokens", 0))
    s["cost"] += float(_t.get("cost_usd", 0.0))
    s["duration"] += float(_t.get("duration_s", 0.0))

    # Parametre denetimi — LLM ne karar verdi, fiilen ne çizildi
    with st.expander("🔬 Parametre denetimi (UI zorla + LLM kararı + motor çıktısı)",
                     expanded=False):
        st.markdown("**UI'dan zorlanan kısıtlar:**")
        sent_audit = {
            "🚪 Oda sayısı": int(n_oda_in),
            "🛋️ Salon sayısı": int(n_salon_in),
            "🚶 Koridor sayısı": int(n_kor_in),
            "🏢 Kat sayısı": int(n_kat_in),
            "🚪 Oda başına kapı aralığı": list(n_doors_range),
            "Toplam oda+salon": n_total_rooms,
            "Layout (zorlanan)": (
                "lshape" if int(n_kor_in) == 2 else "straight"),
        }
        if use_advanced:
            sent_audit["🔧 Gelişmiş tercihler (prompt'a)"] = {
                "oda_genislik": list(adv_room_w),
                "oda_boy": list(adv_room_l),
                "koridor_genislik": list(adv_corridor_w),
                "koridor_uzunluk": list(adv_corridor_l),
                "kapi_genislik": list(adv_door_w),
                "kapi_yukseklik": list(adv_door_h),
                "duvar_kalinlik": float(adv_wall_t),
                "kat_yukseklik": list(adv_storey_h),
            }
        st.json(sent_audit, expanded=True)

        st.markdown("**Her IFC için LLM'in seçtiği boyutlar:**")
        ifc_rows = []
        all_plans = []
        if res.get("ana_plan"):
            all_plans.append(("ana_baseline", res["ana_plan"]))
        for i, vp in enumerate(res.get("variant_plans", [])):
            all_plans.append((f"varyant_{i+1}", vp))
        for label, plan in all_plans:
            room_sizes = "; ".join(
                f"{r.get('role','?')}={r.get('width',0):.2f}×{r.get('length',0):.2f}"
                f"[{r.get('n_doors', 1)}k]"
                for r in (plan.rooms or [])
            )[:140]
            ifc_rows.append({
                "IFC": label,
                "Kat yük. (m)": round(plan.storey_height, 2),
                "Duvar (m)": round(plan.wall_thickness, 2),
                "Koridor (m)": f"{plan.corridor_width:.2f}×{plan.corridor_length:.2f}",
                "İç kapı (m)": f"{plan.interior_door_w:.2f}×{plan.interior_door_h:.2f}",
                "Dış kapı (m)": f"{plan.entrance_door_w:.2f}×{plan.entrance_door_h:.2f}",
                "Odalar": room_sizes,
            })
        if ifc_rows:
            st.dataframe(pd.DataFrame(ifc_rows), hide_index=True,
                         use_container_width=True)
        st.caption(
            "💡 Bütün boyutlar LLM'den geliyor. 'Kat=1', layout, oda+salon "
            "sayısı UI'dan zorlu kalıyor."
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
