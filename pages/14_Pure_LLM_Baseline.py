"""Sayfa 14 — Tamamen LLM ile IFC Üretimi (deneysel).

LLM IFC4 text'ini DOĞRUDAN yazar. Prosedürel motor yok. UI parametreleri
(oda/salon/koridor/kat/kapı sayıları + tercih boyutları) Sayfa 10'la AYNI —
prompt'a ZORUNLU + TERCİH bloğu olarak girer. LLM bunları okur ve uymaya
çalışır (zorla edemediğimiz için kontrol sonrası rapor edilir).

Geçerli IFC'ler Sayfa 15'te görülebilir (paket içinde "pure_llm" olarak).
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

from llm.excel_log import log_llm_generation, read_llm_totals
from llm.pricing import estimate_cost, known_models
from llm.pure_llm_baseline import generate_pure_llm_ifc, register_pure_llm_baseline
from violation_pool import storage
from violation_pool.config import settings


def _fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


st.set_page_config(page_title="Pure LLM Baseline", layout="wide",
                   page_icon="🧪")
st.title("🧪 Tamamen LLM ile IFC Üretimi (deneysel)")

st.warning(
    "⚠️ **Deneysel mod**: LLM IFC4 text'ini DOĞRUDAN yazar — prosedürel motor "
    "yok. Başarı oranı düşük (%20-50), maliyet yüksek. UI parametreleri "
    "prompt'a ZORUNLU/TERCİH bloğu olarak girer; LLM uymaya çalışır ama "
    "kod tarafında zorlayamayız (kontrol sonrası rapor edilir). "
    "**Üretim için sayfa 10 (hibrit) önerilir.**"
)

if not settings.openai_api_key:
    st.error("⚠️ `OPENAI_API_KEY` tanımlı değil.")
    st.stop()

# --- Sürekli sayaç ------------------------------------------------------
totals = read_llm_totals() or {
    "n_calls": 0, "total_tokens": 0,
    "total_cost_usd": 0.0, "total_duration_s": 0.0,
}
session_t = st.session_state.get("session_llm_totals", {
    "n_calls": 0, "tokens": 0, "cost": 0.0, "duration": 0.0,
})
mc = st.columns(4)
mc[0].metric("📒 Toplam üretim", _fmt_int(totals["n_calls"]),
             delta=(f"+{session_t['n_calls']} bu oturum"
                    if session_t["n_calls"] else None))
mc[1].metric("Token (toplam)", _fmt_int(totals["total_tokens"]),
             delta=(f"+{_fmt_int(session_t['tokens'])} bu oturum"
                    if session_t["tokens"] else None))
mc[2].metric("Maliyet ≈ (toplam)",
             f"${totals['total_cost_usd']:.4f}",
             delta=(f"+${session_t['cost']:.4f}"
                    if session_t["cost"] > 0 else None))
mc[3].metric("Süre (toplam, LLM)",
             f"{totals['total_duration_s']:.0f}s",
             delta=(f"+{session_t['duration']:.1f}s"
                    if session_t["duration"] else None))
st.caption("📒 Sayfa 10 ile aynı `llm_generations.xlsx` — repo dışı.")
st.divider()


# --- 1. Paket adı -------------------------------------------------------
st.subheader("1. Paket (ana baseline) adı")
c_name, c_help = st.columns([2, 1])
with c_name:
    ana_ad = st.text_input(
        "📛 Paket adı (dataset_tag)",
        value="pure_llm_v1",
        help="Üretilen IFC'lerin dosya adı prefix'i ve DB etiketi. "
             "Sayfa 15'te bu paket altında görünür.",
    )
with c_help:
    st.caption("Sadece harf/rakam/_/-/+ kullan.")

ana_ad_clean = re.sub(r"[^A-Za-z0-9_\-+]", "_", ana_ad.strip()) or "pure_llm"
if ana_ad_clean != ana_ad:
    st.caption(f"⚠️ Temizlenmiş: `{ana_ad_clean}`")

try:
    existing = storage.list_dataset_tags()
except Exception:
    existing = []
existing_names = {t["tag"] for t in existing}
if ana_ad_clean in existing_names:
    m = next(t for t in existing if t["tag"] == ana_ad_clean)
    st.warning(
        f"ℹ️ `{ana_ad_clean}` paketi zaten var "
        f"({m.get('baseline', 0)} baseline + {m.get('violated', 0)} violated). "
        "Yeni IFC'ler bu pakete EKLENECEK."
    )

# --- 2. Bina tarifi -----------------------------------------------------
st.subheader("2. Bina tarifi (template prompt)")
default_prompt = (
    "Küçük ölçekli bir ofis binası IFC dosyası üret.\n"
    "- Her odanın koridora bir kapısı olsun\n"
    "- Erişilebilirlik mevzuatına uygun (kapı ≥0.90 m, koridor ≥1.20 m)\n"
    "- Birim METRE, IFC4 schema"
)
user_template = st.text_area(
    "💬 Bina tarifi",
    value=default_prompt, height=160,
    help="LLM bunu okuyup IFC4 text'i yazar. Sayılar (oda/salon/koridor/"
         "kat) aşağıdaki parametrelerden gelir — buraya yazma.",
)

# --- Zorunlu sayısal kısıtlar ------------------------------------------
st.markdown("**🔢 Zorunlu sayılar** (LLM prompt'una otomatik eklenir)")
ncc = st.columns(4)
with ncc[0]:
    n_oda_in = st.number_input(
        "🚪 Oda sayısı", min_value=1, max_value=4, value=2,
        help="Salon hariç oda sayısı.",
    )
with ncc[1]:
    n_salon_in = st.number_input(
        "🛋️ Salon sayısı", min_value=0, max_value=3, value=1,
    )
with ncc[2]:
    n_kor_in = st.number_input(
        "🚶 Koridor sayısı", min_value=1, max_value=2, value=1,
    )
with ncc[3]:
    n_kat_in = st.number_input(
        "🏢 Kat sayısı", min_value=1, max_value=3, value=1,
    )

n_doors_range = st.slider(
    "🚪 Oda başına kapı sayısı (her odanın kaç kapısı olacak)",
    min_value=1, max_value=4, value=(1, 2), step=1,
)

# Doğrulama
n_total_rooms = int(n_oda_in) + int(n_salon_in)
if n_total_rooms < 2 or n_total_rooms > 4:
    st.error(f"⚠️ Oda + salon toplamı 2-4 arası olmalı. Şu an: {n_total_rooms}.")
    constraints_valid = False
else:
    st.caption(
        f"✓ {int(n_kat_in)} kat × ({int(n_oda_in)} oda + {int(n_salon_in)} "
        f"salon + {int(n_kor_in)} koridor)"
    )
    constraints_valid = True


# --- Gelişmiş ayarlar --------------------------------------------------
with st.expander(
    "🔧 Gelişmiş ayarlar — boyut aralıkları (LLM'e tercih olarak iletilir)",
    expanded=False,
):
    use_advanced = st.checkbox(
        "Gelişmiş ayarları prompt'a ekle", value=False, key="pure_use_adv",
    )
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Oda boyutu (m)**")
        adv_room_w = st.slider("Oda genişliği", 2.0, 8.0, (3.0, 5.5), 0.1, key="p_rw")
        adv_room_l = st.slider("Oda boyu", 2.0, 10.0, (3.5, 6.0), 0.1, key="p_rl")
        st.markdown("**Kapı (m)**")
        adv_door_w = st.slider("Kapı genişliği", 0.90, 1.50, (0.95, 1.20), 0.05, key="p_dw")
        adv_door_h = st.slider("Kapı yüksekliği", 2.00, 2.50, (2.10, 2.30), 0.05, key="p_dh")
    with ac2:
        st.markdown("**Koridor (m)**")
        adv_corridor_w = st.slider("Koridor genişliği", 1.20, 3.00, (1.40, 2.20), 0.1, key="p_cw")
        adv_corridor_l = st.slider("Koridor uzunluğu", 2.0, 10.0, (3.0, 6.0), 0.1, key="p_cl")
        st.markdown("**Yapı (m)**")
        adv_wall_t = st.slider("Duvar kalınlığı", 0.10, 0.40, 0.20, 0.05, key="p_wt")
        adv_storey_h = st.slider("Kat yüksekliği", 2.70, 3.50, (2.80, 3.20), 0.05, key="p_sh")
    if use_advanced:
        st.caption("✓ Bu aralıklar prompt'a TERCİH bloğu olarak girecek.")


# --- 3. Model + parametreler -------------------------------------------
st.subheader("3. Model ve üretim parametreleri")
mc = st.columns([2, 1, 1])
with mc[0]:
    _opts = known_models() + ["✏️ Özel (elle yaz)"]
    _default_idx = _opts.index("gpt-5") if "gpt-5" in _opts else 0
    _pick = st.selectbox(
        "🤖 GPT modeli (güçlü modeller önerilir)",
        options=_opts, index=_default_idx,
        help="Pure LLM IFC zor — gpt-5, gpt-4-turbo gibi güçlü modeller "
             "daha iyi sonuç verir.",
    )
    if _pick == "✏️ Özel (elle yaz)":
        model = st.text_input("Model adı", value="gpt-5", key="pure_custom_model")
    else:
        model = _pick
with mc[1]:
    n_attempts = st.number_input(
        "🔁 Deneme sayısı", min_value=1, max_value=20, value=3,
        help="Her deneme ayrı LLM çağrısı. Başarısız olanlar da loglanır.",
    )
with mc[2]:
    seed_start = st.number_input(
        "🎲 Tohum başlangıcı", min_value=0, max_value=99999, value=0,
        help="Dosya adında kullanılır: {paket}_pure_{seed:05d}.ifc",
    )

# Tahmini maliyet
_est = estimate_cost(model, 1500, 5500)
_total_est = _est["total_usd"] * int(n_attempts)
metrics = st.columns(3)
metrics[0].metric("Toplam LLM çağrısı", int(n_attempts))
metrics[1].metric("🧮 Tahmini maliyet", f"${_total_est:.4f}",
                  help=f"~{(1500 + 5500) * n_attempts:,} token · {model}")
metrics[2].metric("⏱️ Tahmini süre",
                  f"~{int(n_attempts) * 20}s",
                  help="Pure LLM IFC üretimi yavaş — IFC4 sentaksı uzun.")


# --- 4. Çalıştır --------------------------------------------------------
st.subheader("4. Çalıştır")
if st.button("🧪 Pure LLM ile üret", type="primary",
             use_container_width=True, disabled=not constraints_valid):
    constraints_dict = {
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
    }

    out_dir = settings.ifc_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = st.progress(0.0, text="başlatılıyor...")
    results: list[dict] = []
    t0 = time.time()

    for i in range(int(n_attempts)):
        seed = int(seed_start) + i
        stem = f"{ana_ad_clean}_pure_{seed:05d}"
        # Çakışan ad için ek
        n = 1
        while (out_dir / f"{stem}.ifc").exists():
            stem = f"{ana_ad_clean}_pure_{seed:05d}_{n}"
            n += 1
        out_path = out_dir / f"{stem}.ifc"

        try:
            r = generate_pure_llm_ifc(
                user_template, out_path, model=model,
                constraints=constraints_dict,
            )
        except Exception as e:
            results.append({
                "n": i + 1, "stem": stem, "valid": False,
                "error": str(e)[:200], "tokens": 0, "cost_usd": 0.0,
                "duration_s": 0.0, "n_walls": 0, "n_doors": 0,
                "n_spaces": 0, "n_windows": 0,
            })
            bar.progress((i + 1) / n_attempts,
                         text=f"{i+1}/{n_attempts} · LLM HATA")
            continue

        # Geçerliyse DB'ye baseline olarak kaydet (sayfa 15 görür)
        ifc_id = None
        if r.valid:
            ifc_id = register_pure_llm_baseline(
                r, dataset_tag=ana_ad_clean,
                user_prompt=user_template,
                constraints=constraints_dict,
            )

        # Excel log
        log_llm_generation(
            paket=ana_ad_clean, ifc_name=stem,
            kind="pure_llm_baseline",
            model=model, user_prompt=r.user_prompt,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            duration_s=r.duration_s, cost_usd=r.cost_usd,
            design_summary=(f"walls={r.n_walls} doors={r.n_doors} "
                            f"spaces={r.n_spaces} windows={r.n_windows}"),
            rationale="pure_llm: LLM doğrudan IFC text üretti",
            parametreler={
                "mode": "pure_llm", "model": model,
                "seed": seed, "constraints": constraints_dict,
            },
            tasarim_plan=None,
            status="ok" if r.valid else "error",
            error=r.parse_error or "",
        )

        results.append({
            "n": i + 1, "stem": stem, "valid": r.valid,
            "error": r.parse_error,
            "tokens": r.prompt_tokens + r.completion_tokens,
            "cost_usd": r.cost_usd, "duration_s": r.duration_s,
            "n_walls": r.n_walls, "n_doors": r.n_doors,
            "n_spaces": r.n_spaces, "n_windows": r.n_windows,
            "ifc_path": r.ifc_path, "ifc_id": ifc_id,
            "raw_text_path": str(out_path),
        })
        bar.progress((i + 1) / n_attempts,
                     text=f"{i+1}/{n_attempts} · "
                          f"{'✓ valid + DB' if (r.valid and ifc_id) else ('✓ valid' if r.valid else '❌ invalid')}")

    bar.empty()
    dt = time.time() - t0

    # Session counter'ı güncelle
    total_tokens = sum(r["tokens"] for r in results)
    total_cost = sum(r["cost_usd"] for r in results)
    total_dur = sum(r["duration_s"] for r in results)
    s = st.session_state.setdefault(
        "session_llm_totals",
        {"n_calls": 0, "tokens": 0, "cost": 0.0, "duration": 0.0},
    )
    s["n_calls"] += len(results)
    s["tokens"] += int(total_tokens)
    s["cost"] += float(total_cost)
    s["duration"] += float(total_dur)

    # --- Özet ---
    n_ok = sum(1 for r in results if r["valid"])
    n_fail = len(results) - n_ok
    pct = (100 * n_ok / len(results)) if results else 0
    if n_ok == 0:
        st.error(f"🔴 Hiçbir IFC geçerli değil ({n_fail} hata).")
    elif n_ok < len(results):
        st.warning(f"⚠️ Başarı: {n_ok}/{len(results)} (%{pct:.0f}) · {dt:.1f}s")
    else:
        st.success(
            f"🎉 Hepsi geçerli — {n_ok}/{len(results)} (%100) · {dt:.1f}s"
        )

    m = st.columns(5)
    m[0].metric("Geçerli IFC", n_ok)
    m[1].metric("Geçersiz", n_fail)
    m[2].metric("Başarı %", f"{pct:.0f}%")
    m[3].metric("Toplam token", _fmt_int(total_tokens))
    m[4].metric("Toplam maliyet", f"${total_cost:.4f}")

    # Constraint check tablosu
    st.markdown("### 📐 Beklenen vs Gerçekleşen")
    expected_spaces = n_total_rooms + int(n_kor_in) * int(n_kat_in)
    cc_rows = []
    for r in results:
        if not r["valid"]:
            continue
        cc_rows.append({
            "IFC": r["stem"],
            "Beklenen IfcSpace": expected_spaces,
            "Gerçek IfcSpace": r["n_spaces"],
            "Eşleşti mi?": "✓" if r["n_spaces"] == expected_spaces else "≠",
            "IfcWall": r["n_walls"],
            "IfcDoor": r["n_doors"],
            "IfcWindow": r["n_windows"],
        })
    if cc_rows:
        st.dataframe(pd.DataFrame(cc_rows), hide_index=True,
                     use_container_width=True)
        n_match = sum(1 for r in cc_rows if "✓" in r["Eşleşti mi?"])
        st.caption(
            f"💡 {n_match}/{len(cc_rows)} geçerli IFC kullanıcı oda/koridor "
            "sayısına uydu. LLM kısıtları her zaman dinlemiyor — pure-LLM'in "
            "yapısal zayıflığı."
        )

    # Detay tablo
    with st.expander("📋 Tüm denemelerin detayı", expanded=False):
        df = pd.DataFrame(results)
        display_cols = ["n", "stem", "valid", "tokens", "cost_usd",
                        "duration_s", "n_walls", "n_doors", "n_spaces",
                        "n_windows", "error"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], hide_index=True,
                     use_container_width=True)

    # Başarısız denemeler
    if n_fail > 0:
        with st.expander(f"❌ {n_fail} başarısız deneme — detay"):
            for r in results:
                if not r["valid"]:
                    st.markdown(f"**{r['stem']}**")
                    st.code(r.get("error", "?") or "?", language="text")
                    st.caption(f"Ham .ifc kalıcı: `{r['raw_text_path']}`")

    if n_ok > 0:
        st.caption(
            f"✅ {n_ok} geçerli IFC `{ana_ad_clean}` paketine eklendi → "
            "**🔍 IFC Görüntüleyici** sayfasından incele."
        )


# --- Kapanış notu -------------------------------------------------------
st.divider()
st.info(
    "💡 **Sonuç**: pure-LLM IFC üretimi başarı oranı düşük ve maliyetli. "
    "Geçerli olsa bile kullanıcı oda/koridor sayısına her zaman uymaz "
    "(LLM'i kod tarafında zorlamak imkansız çünkü IFC text'i biz yazmıyoruz). "
    "Dataset çeşitlilik kaynağı olarak kullanılabilir; ana baseline üretim "
    "için **sayfa 10 (hibrit)** daha güvenilir."
)
