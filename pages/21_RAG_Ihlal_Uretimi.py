"""Sayfa 21 — RAG ile İhlal Üretimi.

Mevcut RAG corpus'undan (ChromaDB) TS 9111 / TS ISO 21542 kurallarını
çekip LLM ile somut ihlal cümleleri üretir, baseline IFC'lere uygular.

Akış:
  1. Paket seç (baseline'lar olan)
  2. ChromaDB collection seç (önceden ingest edilmiş)
  3. Kategori multiselect (16 default hepsi)
  4. Baseline başına ihlal sayısı + decoy oranı
  5. Model seç + maliyet tahmini
  6. Üret → RAG retrieve → generate_rag → inject_violations zinciri
  7. Sonuç: başarı, ihlal, decoy istatistikleri
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from llm.excel_log import read_llm_totals
from llm.pricing import estimate_cost, known_models
from llm.violation_pipeline import run_rag_violation_pipeline
from violation_pool import rag, storage
from violation_pool.config import settings


# OPTIMIZED_PROMPT'taki 16 kategori (TS 9111 / TS ISO 21542)
CATEGORIES = [
    "Yaya erişimi", "Giriş", "Kapı/Koridor", "Rampa", "Merdiven",
    "Korkuluk/Küpeşte", "Asansör", "Tuvalet/Banyo", "Mutfak",
    "Otopark", "Uyarı yüzeyi", "Yönlendirme/İşaretleme",
    "Görsel/Kontrast", "Aydınlatma", "Manevra alanı", "Eşik/Kot farkı",
]

# IFC'den doğrudan geometrik/mekânsal olarak doğrulanabilir kategoriler.
# Diğerleri (Aydınlatma, Kontrast, Uyarı yüzeyi vb.) genelde IFC'de bulunmaz.
DEFAULT_CATEGORIES = [
    "Giriş", "Kapı/Koridor", "Rampa", "Merdiven",
    "Tuvalet/Banyo", "Mutfak", "Manevra alanı", "Eşik/Kot farkı",
]


def _fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", ".")


st.set_page_config(page_title="RAG İhlal Üretimi", layout="wide", page_icon="🤖")
st.title("🤖 RAG ile İhlal Üretimi")
st.caption(
    "Mevcut RAG corpus'undan TS 9111 / TS ISO 21542 kurallarını çek → "
    "LLM ile somut ihlal cümleleri üret → baseline IFC'lere uygula. "
    "Etiket KURALLA ölçülür (ground truth dürüst)."
)

if not settings.openai_api_key:
    st.error("⚠️ `OPENAI_API_KEY` tanımlı değil.")
    st.stop()

# Sürekli sayaç (sayfa 10 + 14 + 21 paylaşımlı)
totals = read_llm_totals() or {
    "n_calls": 0, "total_tokens": 0,
    "total_cost_usd": 0.0, "total_duration_s": 0.0,
}
mc = st.columns(4)
mc[0].metric("📒 Toplam üretim", _fmt_int(totals["n_calls"]))
mc[1].metric("Token (toplam)", _fmt_int(totals["total_tokens"]))
mc[2].metric("Maliyet ≈ (toplam)", f"${totals['total_cost_usd']:.4f}")
mc[3].metric("Süre (toplam, LLM)", f"{totals['total_duration_s']:.0f}s")
st.divider()


# --- 1. RAG Collection ----------------------------------------------------
st.subheader("1. RAG kaynağı (ChromaDB collection)")
try:
    collections = rag.list_collections()
except Exception as e:
    st.error(f"ChromaDB hatası: {e}")
    st.stop()

if not collections:
    st.warning(
        "📭 ChromaDB'de hiç collection yok. **Önce doküman ingest et:**\n\n"
        "Sol menüden **📚 22 RAG Operations** → collection oluştur → PDF "
        "yükle → ingest et. Sonra buraya dön."
    )
    st.stop()

cc = st.columns([2, 2])
with cc[0]:
    collection_name = st.selectbox(
        "📚 Collection",
        options=collections,
        help="Önceden ingest edilmiş RAG corpus. Hangi corpus'u kullanırsan "
             "LLM oradan kural çeker.",
    )
with cc[1]:
    try:
        info = rag.collection_info(collection_name) or {}
        n_chunks = info.get("count", 0)
        st.caption(
            f"📊 Collection durumu:\n"
            f"- **{n_chunks}** chunk\n"
            f"- Embedding: `text-embedding-3-small`\n"
        )
    except Exception as e:
        st.caption(f"Info alınamadı: {e}")
        n_chunks = 0

if n_chunks == 0:
    st.warning(
        f"`{collection_name}` collection boş. **📚 22 RAG Operations** "
        "sayfasından bu collection'a PDF ingest et."
    )


# --- 2. Baseline paketi ---------------------------------------------------
st.subheader("2. Baseline paketi (hedef)")
try:
    all_tags = storage.list_dataset_tags()
except Exception as e:
    st.error(f"DB hatası: {e}")
    st.stop()
base_tags = [t for t in all_tags if t.get("baseline", 0) > 0]
if not base_tags:
    st.warning(
        "Baseline içeren paket yok. Önce **🏠 Sayfa 10** ile baseline üret."
    )
    st.stop()
st.dataframe(
    pd.DataFrame(base_tags)[["tag", "baseline", "violated", "total"]],
    hide_index=True, use_container_width=True,
)
sel_tag = st.selectbox(
    "📦 Paket",
    options=[t["tag"] for t in base_tags],
    help="Bu paketin tüm baseline'larına ihlal enjekte edilecek. "
         "Üretilen ihlalliler aynı paket altında DB'de saklanır.",
)


# --- 3. Kategori seçimi ---------------------------------------------------
st.subheader("3. İhlal kategorileri (TS 9111 / TS ISO 21542)")
selected_cats = st.multiselect(
    "Hangi kategorilerde ihlal üretilsin?",
    options=CATEGORIES,
    default=DEFAULT_CATEGORIES,
    help="LLM havuz üretirken bu kategorilerden ihlal cümleleri seçer. "
         "Default: IFC'den doğrudan ölçülebilir 8 kategori. "
         "Aydınlatma / kontrast / uyarı yüzeyi gibi alanları üretmek "
         "istiyorsan elle ekle.",
)
if not selected_cats:
    st.error("⚠️ En az 1 kategori seçilmelidir.")
    st.stop()
st.caption(f"✓ {len(selected_cats)} / {len(CATEGORIES)} kategori seçili.")


# --- 4. Üretim parametreleri ----------------------------------------------
st.subheader("4. Üretim parametreleri")
pa, pb = st.columns(2)
with pa:
    violated_per_baseline = st.slider(
        "📦 İhlalli IFC / baseline",
        min_value=1, max_value=20, value=1,
        help="Her baseline'dan kaç farklı ihlalli IFC üretilsin? "
             "Aynı baseline'a birden fazla variant üretilirse her "
             "seferinde havuzdan farklı ihlal kombinasyonu seçilir; "
             "dosya adı _violated1, _violated2 olarak artar.",
    )
    n_per_ifc = st.slider(
        "🎯 İhlal / IFC dosyası",
        min_value=1, max_value=15, value=3,
        help="Her ihlalli IFC dosyasının içine kaç gerçek ihlal enjekte edilecek.",
    )
    rag_k = st.slider(
        "📚 RAG k (chunk sayısı)",
        min_value=4, max_value=20, value=8,
        help="LLM'e bağlam olarak verilecek doküman chunk sayısı.",
    )
with pb:
    decoy_ratio = st.slider(
        "🪤 Decoy oranı (sahte etiket — IFC değişmez)",
        min_value=0.0, max_value=1.0, value=0.20, step=0.05,
        help="İhlal sayısının yüzdesi kadar SAHTE (decoy) etiket eklenir. "
             "IFC modifiye edilmez. Modelin yanlış pozitif yapmasını "
             "ölçmek için.",
    )
    compliant_ratio = st.slider(
        "🟢 Uyumlu ekleme oranı (gerçek kolon — kural bozulmaz)",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="İhlal sayısının yüzdesi kadar KURAL BOZMAYAN kolon eklenir "
             "(IFC GERÇEKTEN değişir; etiket=compliant). Modelin "
             "'kolon görünce ihlal de' kestirmesini engellemek için "
             "negatif eğitim örneği.",
    )

n_baselines = next(
    (t["baseline"] for t in base_tags if t["tag"] == sel_tag), 0
)
expected_violated_ifcs = int(n_baselines) * int(violated_per_baseline)
expected_violations = int(n_per_ifc) * expected_violated_ifcs


# --- 5. Model -------------------------------------------------------------
st.subheader("5. LLM modeli")


@st.cache_data(ttl=30, show_spinner=False)
def _ollama_models(endpoint: str) -> list[str]:
    """Ollama API'den yüklü modelleri çek. Sunucu kapalıysa [] döner."""
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(f"{endpoint.rstrip('/v1')}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as r:
            data = _json.loads(r.read())
        return sorted(m.get("name", "") for m in data.get("models", []))
    except Exception:
        return []


_DEFAULT_LOCAL_ENDPOINT = "http://localhost:11434/v1"

with st.expander("⚙️ Gelişmiş — local endpoint override"):
    custom_endpoint = st.text_input(
        "Local LLM endpoint URL",
        value=_DEFAULT_LOCAL_ENDPOINT,
        help="Ollama default: http://localhost:11434/v1  ·  "
             "vLLM: http://localhost:8000/v1  ·  "
             "LM Studio: http://localhost:1234/v1. "
             "Aşağıdaki dropdown'da 🏠 modellerini buradan listeler.",
    )

_ollama_list = _ollama_models(custom_endpoint)
_openai_models = known_models()

# Birleşik dropdown — ☁️ OpenAI · 🏠 Local · ✏️ Custom
_unified_opts: list[str] = []
_unified_opts += [f"☁️ {m}" for m in _openai_models]
_unified_opts += [f"🏠 {m}" for m in _ollama_list]
_unified_opts += ["✏️ Özel (elle yaz)"]

# Default seçim: önce ucuz local varsa onu öner, yoksa gpt-4o, yoksa ilki.
_pref = next((o for o in _unified_opts if "qwen2.5" in o.lower()), None)
if not _pref:
    _pref = next((o for o in _unified_opts if o == "☁️ gpt-4o"), None)
_default_idx = _unified_opts.index(_pref) if _pref else 0

mc = st.columns([2, 1, 1])
with mc[0]:
    _pick = st.selectbox(
        "🤖 Model",
        options=_unified_opts, index=_default_idx,
        help=("☁️ = OpenAI (maliyetli)  ·  "
              "🏠 = Ollama / local (ücretsiz, daha yavaş)  ·  "
              "✏️ = elle yaz (özel deployment)"),
    )
    if _pick.startswith("☁️ "):
        model = _pick[2:].strip()
        use_local = False
        llm_base_url = None
    elif _pick.startswith("🏠 "):
        model = _pick[2:].strip()
        use_local = True
        llm_base_url = custom_endpoint.strip() or _DEFAULT_LOCAL_ENDPOINT
    else:
        # Özel — kullanıcı modeli ve (opsiyonel) endpoint'i kendi belirler
        model = st.text_input("Model adı", value="gpt-4o",
                              key="rag_custom_model")
        _custom_local = st.checkbox(
            "Bu model local endpoint'e gitsin",
            value=False, key="rag_custom_is_local",
            help="Açarsan yukarıdaki Gelişmiş endpoint kullanılır.",
        )
        use_local = bool(_custom_local)
        llm_base_url = (custom_endpoint.strip()
                        if use_local else None)
    llm_api_key = "local" if use_local else None
    if not _ollama_list and _pick == _unified_opts[0]:
        st.caption(
            "💡 Local model görünmüyor mu? Ollama'yı kur ve model çek: "
            "`.\\scripts\\install_local_llm.ps1` (Windows) ya da "
            "`./scripts/install_local_llm.sh` (mac/linux)."
        )
with mc[1]:
    # Tahmini maliyet
    expected_compliant = int(round(expected_violations * compliant_ratio))
    n_inject_calls = expected_violations + expected_compliant
    if use_local:
        total_est = 0.0
        st.metric("🧮 Tahmini maliyet", "$0.0000",
                  help="Local LLM — ücretsiz")
    else:
        rag_call_cost = estimate_cost(model, 3000, 5000)["total_usd"]
        inject_call_cost = estimate_cost(model, 2000, 500)["total_usd"]
        total_est = rag_call_cost + n_inject_calls * inject_call_cost
        st.metric("🧮 Tahmini maliyet", f"${total_est:.4f}",
                  help=f"1 RAG havuz + {n_inject_calls} inject çağrı "
                       f"({expected_violations} ihlal + {expected_compliant} uyumlu)")
with mc[2]:
    st.metric("⏱️ Tahmini süre",
              f"~{5 + n_inject_calls * 2}s",
              help="RAG havuz ~5s + her inject ~2s")

st.markdown(
    f"**Beklenen sonuç** — {n_baselines} baseline × "
    f"{int(violated_per_baseline)} variant = "
    f"{expected_violated_ifcs} ihlalli IFC dosyası, her birinin içine "
    f"{int(n_per_ifc)} ihlal:"
)
sm = st.columns(5)
sm[0].metric("Baseline sayısı", n_baselines,
             help="Paketteki temiz IFC sayısı.")
sm[1].metric("Violated IFC (dosya)", expected_violated_ifcs,
             help=f"{n_baselines} baseline × "
                  f"{int(violated_per_baseline)} variant = "
                  f"{expected_violated_ifcs}")
sm[2].metric("Toplam ihlal (içerik)", expected_violations,
             help=f"{expected_violated_ifcs} dosya × {int(n_per_ifc)} ihlal "
                  f"= {expected_violations}")
sm[3].metric("Toplam decoy ≈",
             int(expected_violations * decoy_ratio),
             help="Sahte etiket (IFC değişmez).")
sm[4].metric("Toplam uyumlu ≈", expected_compliant,
             help="Kural bozmayan kolon ekleme (IFC GERÇEKTEN değişir, "
                  "etiket=compliant).")


# --- 6. Çalıştır ----------------------------------------------------------
st.subheader("6. Çalıştır")
disabled = (n_chunks == 0)
if st.button("🤖 RAG'dan ihlal üret + enjekte et",
             type="primary", use_container_width=True, disabled=disabled):
    bar = st.progress(0.0, text="başlatılıyor...")
    log_slot = st.empty()
    logs: list[str] = []
    t0 = time.time()

    def _cb(done, total, label):
        bar.progress(done / total, text=f"{done}/{total} · {label}")
        logs.append(f"  • {label}")
        if len(logs) % 2 == 0 or done == total:
            log_slot.code("\n".join(logs[-15:]))

    try:
        res = run_rag_violation_pipeline(
            dataset_tag=sel_tag,
            collection_name=collection_name,
            categories=(selected_cats
                        if len(selected_cats) < len(CATEGORIES) else None),
            n_violations_per_ifc=int(n_per_ifc),
            violated_per_baseline=int(violated_per_baseline),
            decoy_ratio=float(decoy_ratio),
            compliant_addition_ratio=float(compliant_ratio),
            rag_k=int(rag_k),
            model=model.strip() or "gpt-4o",
            llm_base_url=(llm_base_url.strip() if use_local else None),
            llm_api_key=(llm_api_key if use_local else None),
            progress_cb=_cb,
        )
    except RuntimeError as e:
        st.error(f"🔴 {e}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()
    bar.empty()
    dt = time.time() - t0

    # Session counter güncelle (token bilgisi eksik — yine de süre+1 çağrı)
    s = st.session_state.setdefault(
        "session_llm_totals",
        {"n_calls": 0, "tokens": 0, "cost": 0.0, "duration": 0.0},
    )
    s["n_calls"] += 1
    s["duration"] += float(res.get("duration_s", 0))

    # --- Sonuç özeti ---
    if res["ok"] == 0:
        st.error(f"🔴 Hiçbir IFC üretilemedi ({res['err']} hata).")
    elif res["err"] > 0:
        st.warning(
            f"⚠️ {res['ok']}/{res['ok'] + res['err']} IFC başarılı · "
            f"{dt:.1f}s"
        )
    else:
        st.success(
            f"🎉 Bitti — {res['ok']} ihlalli IFC üretildi · {dt:.1f}s"
        )

    m = st.columns(6)
    m[0].metric("İhlalli IFC", res["ok"])
    m[1].metric("Hata", res["err"])
    m[2].metric("Uygulanan ihlal", res["violations_applied"])
    m[3].metric("Decoy etiket", res["decoys"])
    m[4].metric("Uyumlu ekleme", res.get("compliant", 0))
    m[5].metric("RAG havuz", res["pool_size"])

    # İhlal başına detay
    if res.get("items"):
        st.markdown("### 📋 Üretilen IFC'ler")
        import json as _json

        def _read_applied(labels_path: str | None) -> tuple[list[str], int]:
            """labels.json'dan uygulanan ihlal başlıklarını + skip sayısını çek."""
            if not labels_path:
                return [], 0
            try:
                doc = _json.loads(Path(labels_path).read_text(encoding="utf-8"))
            except Exception:
                return [], 0
            titles, skipped = [], 0
            for l in doc.get("labels", []):
                if l.get("is_decoy"):
                    continue
                if l.get("status") == "applied":
                    titles.append(l.get("title") or "?")
                elif l.get("status") == "skipped":
                    skipped += 1
            return titles, skipped

        rows = []
        for it in res["items"]:
            if "error" in it:
                rows.append({
                    "Baseline": it["baseline"],
                    "Violated IFC": "—",
                    "Durum": f"❌ {it['error'][:80]}",
                    "Uygulanan": 0, "İstenen": int(n_per_ifc), "Atlanan": 0,
                    "İhlal başlıkları": "—",
                    "Decoy": 0,
                })
            else:
                summary = it.get("summary", {})
                applied_n = summary.get(
                    "n_violations_applied",
                    summary.get("applied", summary.get("n_applied", 0)))
                req = summary.get("requested", int(n_per_ifc))
                titles, skipped_n = _read_applied(it.get("labels_path"))
                title_str = "; ".join(titles[:3])
                if len(titles) > 3:
                    title_str += f"  …(+{len(titles) - 3})"
                rows.append({
                    "Baseline": it["baseline"],
                    "Violated IFC": it.get("violated", "?"),
                    "Durum": "✓ OK" if applied_n >= req else "⚠️ kısmi",
                    "Uygulanan": applied_n,
                    "İstenen": req,
                    "Atlanan": skipped_n,
                    "İhlal başlıkları": title_str or "—",
                    "Decoy": summary.get("n_decoys",
                                         summary.get("decoys", 0)),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        # "neden 5 dedim 2 oldu" tanısı
        n_partial = sum(1 for r in rows if r.get("Durum") == "⚠️ kısmi")
        if n_partial:
            st.info(
                f"ℹ️ {n_partial} IFC istenen sayıda ihlal uygulanamadan "
                "bitti. **Sebep:** havuzdaki bazı ihlaller bu baseline'a "
                "uygulanabilir hedef bulamadı (örn. baseline'da yeterli "
                "kapı/rampa yok). Sayfa 15'te ihlalli IFC'yi seçip "
                "'🔴 İhlaller' expander'ında **Atlanan ihlaller** "
                "bölümüne bak — LLM hangi nedenle pas geçtiği orada."
            )

    st.caption(
        f"📒 Defter güncel · Sonraki: **🔍 Sayfa 15 — IFC Görüntüleyici** → "
        f"`{sel_tag}` paketini seç → ihlalliler col 2'de görünür."
    )

    if res.get("ok", 0) > 0:
        if st.button("🔍 Görüntüleyicide göster",
                     type="primary", use_container_width=True,
                     key="rag_jump_viewer"):
            st.session_state["viewer_jump_pkg"] = sel_tag
            st.switch_page("pages/15_Ifc_Goruntuleyici.py")


# --- Kapanış --------------------------------------------------------------
st.divider()
st.info(
    "💡 **Bu sayfa nasıl çalışır:** RAG corpus'tan k=8 kural chunk çekilir "
    "→ LLM ile (OPTIMIZED_PROMPT şeması) sözel ihlal havuzu üretilir → "
    "her baseline için havuzdan N ihlal seçilir → `inject_violations` her "
    "ihlal için ayrı bir LLM çağrısıyla IFC eylem önerir (`_propose_edit`) "
    "→ IFC modifiye edilir → etiket kuralla ölçülür (örn. `kapı < 0.90 m "
    "→ is_violation=True`). Ground truth dürüstlüğü korunur — LLM ne "
    "önerirse önersin, etiket kurala bağlı.\n\n"
    "**Ön gereksinim:** ChromaDB'de TS 9111 / TS ISO 21542 dokümanı "
    "ingest edilmiş olmalı (📚 Sayfa 22 — RAG Operations)."
)
