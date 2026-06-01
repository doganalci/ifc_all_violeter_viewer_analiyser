# Pipeline Dokümantasyonu — Dosya, Klasör ve Deney Haritası

> Bu doküman BIM erişilebilirlik denetimi pipeline'ında hangi rapor
> dosyasının neyi içerdiğini, hangi UI sayfasının ne yaptığını ve hangi
> deneyin nereye bakılarak değerlendirildiğini açıklar.

---

## 1. Veri Klasör Yapısı

Tüm veri **repo dışında** tutuluyor (`paths.data_home()` ile çözümlenir).

Öncelik sırası:
1. `IFC_DATA_HOME` env var
2. Komşu klasör (`../ifc_desktop_doc_dataset/`)
3. Desktop fallback

```
data_home/
├── ifc_models/
│   ├── baseline/
│   │   ├── ofis_v1_ana_00000.ifc        ← hibrit ana baseline
│   │   ├── ofis_v1_00001.ifc            ← hibrit varyant
│   │   ├── pure_llm_v2_pure_00000.ifc   ← pure LLM IFC
│   │   ├── *.graph.json                 ← her IFC'nin graph'ı
│   │   ├── *.labels.json                ← her IFC'nin etiketleri
│   │   └── *.meta.json                  ← her IFC'nin meta verisi
│   └── violated/                    # İhlal enjekte edilmiş IFC'ler
├── docs/                            # RAG için kaynak doküman (PDF/MD)
├── vectorstore/                     # ChromaDB persistent
├── exports/                         # Manuel export'lar
├── ml_runs/                         # GAT model checkpoint'leri (varsa)
├── violation_pool.sqlite            # MERKEZİ İNDEKS DB
├── llm_generations.xlsx             # HER LLM ÇAĞRISI SATIR SATIR
├── operations_log.xlsx              # TÜM OPERASYON LOGU
├── datasets.xlsx                    # PAKET + ÖRNEK DEFTERI
├── experiments.xlsx                 # EĞİTİM ÇALIŞMA KAYITLARI
└── .env                             # OPENAI_API_KEY vs.
```

Eğitilmiş GAT modelleri ayrıca proje kökündeki **`runs/`** klasöründe.

---

## 2. Excel/CSV Rapor Dosyaları

### 2.1 `llm_generations.xlsx` ← LLM kullanım analizi

**Sheet:** `uretim`. Her LLM çağrısı için bir satır.

| Sütun | Açıklama |
|---|---|
| `zaman` | Tarih + saat |
| `paket` | dataset_tag |
| `ifc_adi` | Dosya adı |
| `tur` | `ana_baseline` / `variant_baseline` / `pure_llm_baseline` |
| `model` | gpt-4o, gpt-5, vs. |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | OpenAI usage |
| `sure_s` | Çağrı süresi |
| `maliyet_usd` | Fiyat tablosundan |
| `tasarim_ozeti` | Kısa tasarım açıklaması |
| `rationale` | LLM gerekçesi |
| `user_prompt` | Tam prompt (max 1500 char) |
| `parametreler_json` | UI constraints + model + seed (JSON) |
| `tasarim_plan_json` | LLM tüm çıktısı (JSON) |
| `status` | `ok` / `error` |
| `error` | Hata mesajı |

**Yararlı sorgular:**
- gpt-4o vs gpt-5 maliyet → group by `model`
- Pure-LLM başarı oranı → `tur == "pure_llm_baseline"` group by `status`

### 2.2 `operations_log.xlsx` ← Batch akış izlemek

**Sheet:** `islemler`. Her büyük operasyon (üretim batch, eğitim, vs.) bir
satır. Token tek tek değil aggregate.

### 2.3 `datasets.xlsx` ← Paket içeriği

DB'den otomatik üretiliyor. İki sheet:
- `paketler`: paket × baseline/violated/total
- `ornekler`: her IFC bir satır, parent_id ile bağlantı

### 2.4 `experiments.xlsx` ← GAT eğitim kayıtları

Sadece eski sayfa 04 ile `record_run()` çağrıldıysa dolu. Yeni dönem
eğitimleri eklemeli.

### 2.5 `violation_pool.sqlite` ← Merkezi gerçek

SQLite veritabanı. Excel'ler bunun türevleri.

**Önemli tablolar:**
- `ifc_models`: id, kind, name, parent_id, dataset_tag, file paths,
  llm_model, prompt, params (JSON), status, created_at
- `ifc_violation_labels`: node-düzeyi etiketler

**Hiyerarşi:** parent_id ile bağlanır:
- Ana baseline: parent_id = NULL
- Varyant baseline: parent_id = ana.id
- Violated: parent_id = baseline.id

---

## 3. Per-IFC Dosya Üçlüsü (.meta + .labels + .graph)

Her `xxx.ifc` dosyasının yanında üç sidecar:

### 3.1 `*.meta.json`

IFC üretim meta verisi: `ifc_id`, `kind`, `source`, `seed`, `spec_meta`
(motor seçimleri, room_sizes, door_widths vs.).

### 3.2 `*.labels.json`

ML eğitimi için node etiketleri. Her label:
```json
{
  "ifc_global_id": "GUID",
  "category": "Kapı/LLM",
  "severity": "kritik" | "uygun",
  "status": "applied" | "compliant" | "clean",
  "is_decoy": false,
  "attribute": "OverallWidth",
  "before": 0.95, "after": 0.80,
  "rule": "OverallWidth >= 0.90 m",
  "evidence": "..."
}
```

### 3.3 `*.graph.json`

NetworkX node-link formatı. GAT'in girdisi.

Edge tipleri: aggregates, contains, bounds, voids, fills, connects,
co_bounds_space.

---

## 4. UI Sayfaları (Streamlit)

### 4.1 Sayfa 10 — 🏠 LLM ile Baseline Üretimi (HİBRİT)

**Konum:** `pages/10_Baseline_Uretimi.py`

**Akış:** Kullanıcı parametreleri + bina tarifi → LLM tasarım planı JSON →
prosedürel motor IFC çizer → hiyerarşik baseline paketi (1 ana + N varyant).

**UI parametreleri:**
- Paket adı (`dataset_tag`)
- Bina tarifi (text_area)
- Zorunlu sayılar: 🚪 oda, 🛋️ salon, 🚶 koridor, 🏢 kat
- 🚪 Oda başına kapı sayısı (range slider 1-4)
- 🔧 Gelişmiş ayarlar (opsiyonel): boyut aralıkları
- 🤖 GPT modeli + varyant sayısı + seed

**Çıktı:** `baseline/{paket}_ana_*.ifc` + `{paket}_*.ifc`. DB'ye `kind=baseline`,
hiyerarşik kayıt. Her LLM çağrısı `llm_generations.xlsx`'e.

### 4.2 Sayfa 14 — 🧪 Tamamen LLM ile IFC Üretimi (DENEYSEL)

**Konum:** `pages/14_Pure_LLM_Baseline.py`

**Akış:** LLM'den **doğrudan IFC4 text'i** alır (prosedürel motor yok).

**UI parametreleri:** Sayfa 10 ile birebir aynı (karşılaştırma için).

**Çıktı:** Aynı baseline klasörü, `_pure_` prefix'i ile. DB'ye `kind=baseline`,
parent_id=None.

**Önemli:** Tipik olarak (a) geometry kernel çökmesi (b) eksik GUID → graph
orphan (c) eksik IfcRel*. Sayfa 15'te bu görünür.

### 4.3 Sayfa 15 — 🔍 IFC Görüntüleyici (3D + Graph + Tahmin)

**Konum:** `pages/15_Ifc_Goruntuleyici.py`

**Layout:**
```
Ana baseline 3D | İhlalli 3D | Tahmin 3D
Ana baseline gr | İhlalli gr | Tahmin gr
```

**Özellikler:**
- 🔀 Sıralama (yeni→eski default)
- 🔄 Yenile (cache temizleme)
- 📦 Paket selectbox
- İhlalli ◀▶ navigasyon
- 🤖 Tahmin: GAT run + eşik + buton (inference)
- 🔴🟡🟢 Etiket katmanları
- ⛶ İki katmanlı büyütme (col_both → col_3d / col_graph)
- Cross-highlight: graph tıkla → 6 panelde vurgu
- Pure-LLM fallback: entity sayıları + graph yeniden üret butonu

### 4.4 Sayfa 20 — 🤖 LLM ile Kapı İhlal Üretimi

**Konum:** `pages/20_LLM_Ihlal_Uretimi.py`

**Akış:** Baseline paketteki kapılara LLM-yönlendirmeli değişiklik. Etiket
kuralla ölçülür.

**Sıradaki iş:** RAG-tabanlı yeniden tasarım (`NEXT_STEPS.md` P1).

### 4.5 Legacy Sayfalar

`legacy/` klasöründe eski sürüm sayfalar (01-11, 99). Yeni akış için sadece
GAT eğitim (`04_GAT_Egitim.py`) ve test (`05_GAT_Test.py`) hâlâ kullanılır.

---

## 5. Modül Haritası — UI → Kod

| UI | Çağırdığı Modüller |
|---|---|
| Sayfa 10 | `llm/baseline_pipeline.py` → `llm/design_planner.py` → `ml/data/synth_baseline_v2.py` → `violation_pool/ifc_template.py` → `violation_pool/ifc_graph.py` → `violation_pool/storage.py` |
| Sayfa 14 | `llm/pure_llm_baseline.py` → `violation_pool/ifc_inject._chat_with_retry` |
| Sayfa 15 | `ml/viz/ifc3d.py` + `ml/viz/graph_view.py` + `ml/app/state.py` + `ml/data/graph_loader.py` + `ml/model/gat.py` |
| Sayfa 20 | `llm/door_inject.py` + `llm/batch.py` |

**Ortak yardımcılar:**
- `llm/pricing.py` — model fiyat tablosu
- `llm/excel_log.py` — `llm_generations.xlsx`
- `ml/tracking.py` — `operations_log/datasets/experiments.xlsx`
- `paths.py` — `data_home()`
- `violation_pool/config.py` — `settings`

---

## 6. Yapılan Deneyler

| # | Deney | UI | Ölçüt | Sonuç |
|---|---|---|---|---|
| E1 | Hibrit baseline | Sayfa 10 | Validity, token, hız | %100, ~$0.007/IFC, 2-3s |
| E2 | Pure-LLM IFC | Sayfa 14 | Parse, geo, graph | gpt-4o: %25, gpt-5: %60 |
| E3 | LLM kapı ihlal | Sayfa 20 | İhlal/hard-neg dengesi | Kural-bazlı dürüst etiket |
| E4 | Görsel kıyas | Sayfa 15 | Pure vs hibrit kalite | Hibrit 50-200 node, pure 3-10 |
| E5 | GAT inference | Sayfa 15 | F1, P/R, AUC | Önceki modelden |
| E6 | Maliyet/model kıyas | xlsx analiz | $/IFC, başarı | gpt-4o-mini ucuz, gpt-5 kaliteli |

---

## 7. Hızlı Karar Ağacı

| Soru | Cevap |
|---|---|
| Üretim parametreleri / maliyet detayı? | `llm_generations.xlsx` |
| Paket bazında üretim sayısı? | `datasets.xlsx` |
| IFC sidecar verisi? | `*.meta.json` + `*.labels.json` + `*.graph.json` |
| Hiyerarşi (ana-varyant-violated)? | `violation_pool.sqlite` `ifc_models.parent_id` |
| Operasyon geçmişi? | `operations_log.xlsx` |
| GAT eğitim kayıtları? | `experiments.xlsx` + `runs/*/best.pt` |
| Hangi UI ne yapıyor? | Bu doküman §4 + §5 |
| Hangi deney nereye işliyor? | Bu doküman §6 |

---

## 8. Mimari Kararlar

1. **Hibrit > Pure LLM** (E2 ile gösterildi).
2. **Etiketler kural-bazlı ölçülür** (LLM-as-judge bias'tan kaçınma).
3. **Closed-world supervision** (baseline'lar garantili temiz).
4. **Hiyerarşik baseline** (parent_id ile).
5. **Repo dışı veri** (IFC_DATA_HOME).
6. **Excel + SQLite** (insan-okur + sorgulanabilir).

---

İlgili dokümanlar:
- `CONTEXT_BRIEF.md` — yeni Claude için tek dosya brief
- `NEXT_STEPS.md` — sıradaki adımlar
- `GLOSSARY.md` — terim sözlüğü
- `THESIS_REPORT_TEMPLATE.md` — tez raporu taslağı
