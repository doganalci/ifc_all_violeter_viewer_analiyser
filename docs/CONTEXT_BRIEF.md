# Yeni Claude için Context Brief

> **Tek dosyadan tüm proje contextini al.** Bu belge geliştirme oturumunun özetini,
> hangi sayfanın ne yaptığını ve hangi dosyanın neyi içerdiğini özetler.

---

## 1. Proje Bir Cümlede

IFC4 formatındaki bina modellerinde TS 9111 / TS ISO 21542 erişilebilirlik
ihlallerini node-düzeyinde tespit eden GAT (Graph Attention Network) modelini
sentetik veri üzerinde eğiten ve test eden uçtan uca bir Streamlit + Python
pipeline'ı.

## 2. Ana Mimari Karar

**Hibrit yaklaşım:** LLM (GPT-4o/5) tasarım kararlarını verir, prosedürel motor
(ifcopenshell) geçerli IFC dosyasını yazar. Bu, saf LLM IFC üretimine göre
%100 validity ve 5-10x daha düşük token maliyeti sağlar. Pure-LLM yaklaşımı
sadece kıyas/demo için kullanılır (sayfa 14).

## 3. Veri Konumu

Tüm üretim verisi `paths.data_home()` ile çözümlenen klasörde, **repo dışında**:
1. `IFC_DATA_HOME` env var (en yüksek öncelik)
2. Komşu klasör `../ifc_desktop_doc_dataset/`
3. Desktop fallback

`.gitignore` ile `/data/` ignore edilmiş.

## 4. Aktif Sayfalar (`pages/`)

| Sayfa | Dosya | Amaç |
|---|---|---|
| 10 | `10_Baseline_Uretimi.py` | Hibrit baseline üretimi (LLM + motor). Ana üretim akışı. |
| 14 | `14_Pure_LLM_Baseline.py` | Pure-LLM IFC (deneysel kıyas). |
| 15 | `15_Ifc_Goruntuleyici.py` | 3 sütun viewer + GAT tahmin. |
| 20 | `20_LLM_Ihlal_Uretimi.py` | LLM-yönlendirmeli kapı ihlal enjeksiyonu. |

Eski (`legacy/`): 01-11 + 99. Yeni akış üretim için bunları kullanmıyor, ama
GAT eğitim hâlâ `legacy/04_GAT_Egitim.py` ile yapılıyor.

## 5. Modül Haritası

```
llm/                          ← LLM + maliyet
├── design_planner.py         ← LLM tasarım planı + token/cost ölçüm
├── baseline_pipeline.py      ← 1 ana + N varyant pipeline
├── pure_llm_baseline.py      ← Pure-LLM IFC üretici (sayfa 14)
├── door_inject.py            ← Kapı ihlal enjeksiyonu
├── batch.py                  ← İhlal batch runner
├── pricing.py                ← Model fiyat tablosu (USD/1M token)
└── excel_log.py              ← llm_generations.xlsx yazım/okuma

ml/                           ← ML + viz
├── data/
│   ├── synth_baseline_v2.py  ← Prosedürel motor (2-4 oda, lshape, multi-storey)
│   ├── graph_loader.py       ← .graph.json → PyG Data
│   └── ...
├── viz/
│   ├── ifc3d.py              ← Plotly 3D
│   └── graph_view.py         ← streamlit-agraph
├── model/
│   ├── gat.py                ← Homojen GAT
│   └── hetero_gat.py         ← Heterojen GAT (edge type-aware)
├── train/                    ← Eğitim, config, metrics
├── app/state.py              ← Streamlit state + DB readers
└── tracking.py               ← operations_log/datasets/experiments.xlsx

violation_pool/               ← codex1'den miras (üretim çekirdeği)
├── ifc_template.py           ← Spec → geçerli IFC4 (multi-storey)
├── ifc_graph.py              ← IFC → NetworkX (synthetic GUID fallback dahil)
├── ifc_inject.py             ← İhlal enjeksiyonu + LLM chat_with_retry
├── storage.py                ← SQLite DB (ifc_models, labels)
├── rag.py                    ← ChromaDB (RAG için, henüz tam aktif değil)
├── prompts.py                ← OPTIMIZED_PROMPT (RAG ihlal seçimi için)
├── llm.py                    ← Eski LLM havuz üreteci
└── config.py                 ← settings (OPENAI_API_KEY vs.)
```

## 6. Veri Dosyaları (kalıcı kayıt)

Her IFC için **dört dosya**:
- `{stem}.ifc` — IFC4 dosyası
- `{stem}.meta.json` — üretim meta verisi
- `{stem}.labels.json` — node etiketleri (closed-world full labeling)
- `{stem}.graph.json` — NetworkX node-link

Merkezi rapor:
- `violation_pool.sqlite` — SQL gerçek kaynağı
- `llm_generations.xlsx` — her LLM çağrısı (token, maliyet, parametreler)
- `operations_log.xlsx` — batch operasyonlar
- `datasets.xlsx` — paket + örnek defteri
- `experiments.xlsx` — GAT eğitim run kayıtları

## 7. Etiket Şeması (closed-world)

`labels.json` içindeki her label:
```json
{
  "ifc_global_id": "GUID",
  "category": "Kapı/LLM" | "Koridor" | ...,
  "severity": "kritik" | "uygun",
  "status": "applied"     // ihlal uygulanmış (y=1)
          | "compliant"   // değiştirilmiş ama eşik üstü (hard-negative, y=0)
          | "clean",      // hiç dokunulmamış (kesin uygun, y=0)
  "is_decoy": false,
  "attribute": "OverallWidth",
  "before": 0.95, "after": 0.80,
  "rule": "OverallWidth >= 0.90 m",
  "evidence": "..."
}
```

**Etiket dürüstlük garantisi:** LLM ne öneriyorsa öneirsin, `is_violation` flag
sadece eşik kontrolüyle (`< 0.90 → True`) belirlenir.

## 8. Tipik Akış (sıfırdan)

```
1. Sayfa 10 → paket adı + bina tarifi + parametreler → LLM ile 1+N IFC üret
   → data/ifc_models/baseline/{paket}_*.ifc + DB kayıt
   
2. Sayfa 20 → mevcut baseline paketinden ihlalli üret (kapı genişliği/yükseklik)
   → data/ifc_models/violated/llmgen_*.ifc + ihlal etiketli
   
3. Sayfa 15 → paket seç → 3 sütun (Ana | İhlalli | Tahmin) yan yana incele
   → Eğer eğitilmiş GAT varsa "Tahmin Yap" → predicted ihlaller kırmızı

4. (Eski) legacy/04_GAT_Egitim.py → paketten eğitim → runs/{run_name}/best.pt
```

## 9. Bu Dönem Yapılan Önemli Değişiklikler

- **Hibrit baseline pipeline** kuruldu (sayfa 10): kullanıcı parametreleri +
  LLM tasarım planı + prosedürel motor. Token + maliyet ölçümü.
- **Pure-LLM baseline** (sayfa 14): kıyas amaçlı, başarısızlık modlarını
  belgelemek için.
- **Sayfa 15 yeniden tasarımı:** 3 sütun layout (Ana | İhlalli | Tahmin),
  iki katmanlı büyütme (col_both / col_3d / col_graph), sıralama seçici,
  GAT inference, pure-LLM fallback.
- **Çok katlı + L-plan IFC** desteği (synth_baseline_v2 + ifc_template).
- **Per-room kapı sayısı** parametresi (1-4).
- **Excel raporları:** llm_generations.xlsx + cumulative counter +
  parametre/tasarım plan JSON sütunları.
- **Synthetic GUID fallback** (ifc_graph): pure-LLM IFC'lerinde eksik GUID'leri
  graph'a node olarak ekler.

## 10. Açık Görevler

Sıradaki büyük iş:
1. **RAG-tabanlı ihlal üretimi** — sayfa 20'yi yeniden tasarla. ChromaDB'ye
   TS 9111 ingest, sözel ihlal seçimi, kategori bazlı UI.
2. **GAT karşılaştırma deneyleri** — homojen vs heterojen, hibrit dataset vs
   karışık (hibrit + pure-LLM) dataset.
3. **Mevzuat genişletme** — şu an kapı + koridor. Eşik/kot, rampa, asansör
   kategorileri eklenmeli.

Detay: `NEXT_STEPS.md`.

## 11. Hızlı Erişim — "Hangi dosyaya bakayım?"

| Soru | Cevap |
|---|---|
| Bir IFC'nin maliyeti / parametresi? | `llm_generations.xlsx` (ifc_adi sütununda ara) |
| Bir paketin ana baseline'ı + varyantları? | `datasets.xlsx` (sheet `ornekler`) |
| Bir IFC'nin ground truth ihlalleri? | `{stem}.labels.json` |
| Bir IFC'nin yapısı / üretim parametresi? | `{stem}.meta.json` |
| Hangi UI sayfası ne yapıyordu? | `PIPELINE_DOCUMENTATION.md` §4 |
| Bu dönemde ne yaptık? | `THESIS_REPORT_TEMPLATE.md` |
| Sıradaki adım ne? | `NEXT_STEPS.md` |
| Terim ne anlama geliyor? | `GLOSSARY.md` |

---

**Önemli:** Pipeline değişikliği yapmadan önce mutlaka `paths.data_home()`'un
nereye baktığını kontrol et. Test verileri repo dışında olduğu için kod
güncellemesi data'yı etkilemez.
