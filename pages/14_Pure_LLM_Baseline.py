"""Sayfa 14 — Tamamen LLM ile IFC Üretimi (deneysel).

LLM IFC4 text'ini DOĞRUDAN yazar. Prosedürel motor yok. Bu sayfanın amacı:
pure-LLM IFC üretiminin sınırlarını göstermek ve dataset çeşitlilik için
kullanılabilirliğini test etmek.

Sayfa 10'daki hibrit yaklaşımın aksine burada:
  * LLM tüm IFC'yi text olarak yazar
  * Token maliyeti yüksek (~6000-10000 / IFC)
  * Başarı oranı düşük (parse hatası, boş dosya, geometri bozuk)
  * Süre yüksek (10-30s / IFC)

Sonuç → tek başına baseline üretim için **mantıklı değil**, ama varyasyon
kaynağı olarak değerlendirilebilir.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from llm.excel_log import log_llm_generation, read_llm_totals
from llm.pricing import estimate_cost, known_models
from llm.pure_llm_baseline import generate_pure_llm_ifc
from violation_pool.config import settings


st.set_page_config(page_title="Pure LLM Baseline", layout="wide",
                   page_icon="🧪")
st.title("🧪 Tamamen LLM ile IFC Üretimi (deneysel)")

st.warning(
    "⚠️ **Deneysel mod**: LLM IFC4 text'ini DOĞRUDAN yazar — prosedürel motor "
    "yok. Başarı oranı düşük (%20-50), maliyet yüksek (~6000+ token / IFC). "
    "Amaç: pure-LLM yaklaşımının sınırlarını göstermek. **Üretim için sayfa "
    "10 (hibrit) önerilir.**"
)

if not settings.openai_api_key:
    st.error("⚠️ `OPENAI_API_KEY` tanımlı değil.")
    st.stop()

# Sürekli sayaç (sayfa 10 ile aynı veri tabanı)
totals = read_llm_totals() or {
    "n_calls": 0, "total_tokens": 0,
    "total_cost_usd": 0.0, "total_duration_s": 0.0,
}
mc = st.columns(4)
mc[0].metric("📒 Tüm LLM çağrıları", f"{totals['n_calls']:,}".replace(",", "."))
mc[1].metric("Token (toplam)",
             f"{totals['total_tokens']:,}".replace(",", "."))
mc[2].metric("Maliyet ≈ (toplam)", f"${totals['total_cost_usd']:.4f}")
mc[3].metric("Süre (toplam, LLM)", f"{totals['total_duration_s']:.0f}s")
st.divider()


# --- 1. Bina tarifi -----------------------------------------------------
st.subheader("1. Bina tarifi")
default_prompt = (
    "Küçük bir ofis binası IFC dosyası üret:\n"
    "- 3 oda (her biri ~4×4 m)\n"
    "- 1 merkezi koridor (5 m × 1.5 m)\n"
    "- Her odanın koridora bir kapısı (0.95 m × 2.10 m)\n"
    "- 1 ana giriş kapısı (1.10 m × 2.20 m)\n"
    "- Erişilebilirlik mevzuatına uygun (kapı ≥0.90 m, koridor ≥1.20 m)"
)
user_prompt = st.text_area(
    "💬 Bina tarifi (LLM'e direkt iletilir)",
    value=default_prompt, height=200,
    help="LLM bunu okuyup IFC4 text'i üretir. Detaylı tarif daha iyi sonuç verir.",
)


# --- 2. Model + parametreler --------------------------------------------
st.subheader("2. Model ve parametreler")
mc = st.columns([2, 1, 1])
with mc[0]:
    _opts = known_models() + ["✏️ Özel (elle yaz)"]
    # Default: gpt-5 (güçlü model, pure-LLM için daha iyi şans)
    _default_idx = _opts.index("gpt-5") if "gpt-5" in _opts else 0
    _pick = st.selectbox(
        "🤖 GPT modeli (güçlü modeller önerilir)",
        options=_opts, index=_default_idx,
        help="Pure LLM IFC zor — gpt-5, gpt-4-turbo gibi güçlü modeller "
             "daha yüksek başarı oranı verir.",
    )
    if _pick == "✏️ Özel (elle yaz)":
        model = st.text_input("Model", value="gpt-5", key="custom_pure")
    else:
        model = _pick
with mc[1]:
    n_attempts = st.number_input(
        "🔁 Deneme sayısı",
        min_value=1, max_value=10, value=3,
        help="Her deneme ayrı LLM çağrısı. Başarısız olanlar da loglanır.",
    )
with mc[2]:
    # Tahmini maliyet
    _est = estimate_cost(model, 1500, 5500)  # gerçekçi: 5500 output
    _total_est = _est["total_usd"] * int(n_attempts)
    st.metric("🧮 Tahmini maliyet",
              f"${_total_est:.4f}",
              help=f"~{(1500 + 5500) * n_attempts:,} token · {model}")


# --- 3. Çalıştır --------------------------------------------------------
st.subheader("3. Çalıştır")
if st.button("🧪 Pure LLM ile üret", type="primary", use_container_width=True):
    out_dir = settings.ifc_dir / "pure_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    bar = st.progress(0.0, text="başlatılıyor...")
    results: list[dict] = []
    t0 = time.time()

    for i in range(int(n_attempts)):
        stem = f"pure_llm_{model.replace('-', '_')}_{uuid.uuid4().hex[:8]}"
        out_path = out_dir / f"{stem}.ifc"
        try:
            r = generate_pure_llm_ifc(user_prompt, out_path, model=model)
        except Exception as e:
            results.append({
                "n": i + 1, "stem": stem, "valid": False,
                "error": str(e)[:200],
                "tokens": 0, "cost_usd": 0.0, "duration_s": 0.0,
                "n_walls": 0, "n_doors": 0, "n_spaces": 0, "n_windows": 0,
            })
            bar.progress((i + 1) / n_attempts,
                         text=f"{i+1}/{n_attempts} · LLM HATA")
            continue

        # Excel'e logla (başarı/başarısız fark etmez)
        log_llm_generation(
            paket="pure_llm", ifc_name=stem, kind="pure_llm_baseline",
            model=model, user_prompt=user_prompt,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            duration_s=r.duration_s, cost_usd=r.cost_usd,
            design_summary=(f"walls={r.n_walls} doors={r.n_doors} "
                            f"spaces={r.n_spaces} windows={r.n_windows}"),
            rationale="pure_llm: LLM doğrudan IFC text üretti",
            parametreler={"mode": "pure_llm", "model": model,
                          "n_attempts": int(n_attempts)},
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
            "ifc_path": r.ifc_path,
            "raw_text_path": str(out_path),  # her durumda kalıcı
        })
        bar.progress((i + 1) / n_attempts,
                     text=f"{i+1}/{n_attempts} · {'✓ valid' if r.valid else '❌ invalid'}")

    bar.empty()
    dt = time.time() - t0

    # --- Özet ---
    n_ok = sum(1 for r in results if r["valid"])
    n_fail = len(results) - n_ok
    pct = (100 * n_ok / len(results)) if results else 0
    if n_ok == 0:
        st.error(f"🔴 Hiçbir IFC geçerli değil ({n_fail} hata).")
    elif n_ok < len(results):
        st.warning(f"⚠️ Başarı: {n_ok}/{len(results)} (%{pct:.0f}) · {dt:.1f}s")
    else:
        st.success(f"🎉 Hepsi geçerli — {n_ok}/{len(results)} (%100) · {dt:.1f}s")

    m = st.columns(5)
    m[0].metric("Geçerli IFC", n_ok)
    m[1].metric("Geçersiz", n_fail)
    m[2].metric("Başarı %", f"{pct:.0f}%")
    m[3].metric("Toplam token",
                f"{sum(r['tokens'] for r in results):,}".replace(",", "."))
    m[4].metric("Toplam maliyet",
                f"${sum(r['cost_usd'] for r in results):.4f}")

    # Detay tablo
    st.markdown("### 📋 Deneme sonuçları")
    df = pd.DataFrame(results)
    if not df.empty:
        display_cols = ["n", "stem", "valid", "tokens", "cost_usd",
                        "duration_s", "n_walls", "n_doors", "n_spaces",
                        "n_windows", "error"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], hide_index=True,
                     use_container_width=True)

    # Başarılı örnekler
    valid_ones = [r for r in results if r["valid"]]
    if valid_ones:
        st.markdown("### 📄 Geçerli IFC örnekleri")
        for r in valid_ones[:3]:
            with st.expander(
                f"✓ `{r['stem']}.ifc` — {r['n_walls']} duvar · "
                f"{r['n_doors']} kapı · {r['n_spaces']} oda · "
                f"{r['n_windows']} pencere", expanded=False,
            ):
                st.caption(f"📁 `{r['ifc_path']}`")
                # İlk 40 satır preview
                try:
                    content = Path(r["raw_text_path"]).read_text(encoding="utf-8")
                    head = "\n".join(content.split("\n")[:40])
                    st.code(head, language="text")
                    st.caption(f"_(toplam {len(content.splitlines())} satır)_")
                except Exception as e:
                    st.error(f"Dosya okunamadı: {e}")

    # Başarısız denemeler
    if n_fail > 0:
        st.markdown("### ❌ Başarısız denemeler (parse hataları)")
        for r in results:
            if not r["valid"]:
                with st.expander(f"❌ {r['stem']}"):
                    st.code(r.get("error", "?") or "?", language="text")
                    st.caption(
                        f"Ham yanıt yine kayıtlı: `{r['raw_text_path']}`"
                    )


# --- Kapanış notu -------------------------------------------------------
st.divider()
st.info(
    "💡 **Sonuç (genel gözlem)**: pure-LLM IFC üretimi başarı oranı düşük "
    "ve maliyetli — IFC4 sentaksı katı (her referans tutarlı olmalı, GUID "
    "formatı, koordinat ağacı). Güçlü modeller (gpt-5+) daha iyi sonuç "
    "verir ama yine de hibrit yaklaşıma göre dezavantajlı.\n\n"
    "**Mantıklı kullanım**: bu sayfadan üretilen geçerli IFC'leri dataset "
    "çeşitlilik kaynağı olarak değerlendir (\"sıra dışı\" örnekler). Ana "
    "baseline üretim için **sayfa 10 (hibrit)** daha güvenilir."
)
