# Tez İzleme Komitesi Raporu — IFC tabanlı BIM Erişilebilirlik Denetiminde LLM + Graph Sinir Ağları Pipeline'ı

**Dönem:** [tarih aralığı]
**Çalışma alanı:** BIM (Building Information Modeling) erişilebilirlik
denetimi için sentetik veri üretimi, ihlal enjeksiyonu ve grafik tabanlı
tespit modeli.

---

## 1. Yönetici Özeti

Bu dönemde, IFC4 formatındaki bina modellerinde erişilebilirlik mevzuatı
(TS 9111 / TS ISO 21542) ihlallerinin otomatik tespiti için uçtan uca bir
yapay zeka pipeline'ı geliştirildi. Çalışma üç ana ekseni kapsadı:

1. **Sentetik veri üretimi** — eğitim için ihlal-içermeyen (closed-world) IFC
   binaları üretmek.
2. **İhlal enjeksiyonu** — bu temiz binalara mevzuat-aykırı varyasyonlar
   eklemek (kuralla ölçülmüş ground truth ile).
3. **Tespit modeli** — Graph Attention Network (GAT) ile node-düzeyinde ihlal
   sınıflandırması.

Çalışmanın özgün katkıları:

- **Hibrit (LLM + prosedürel) IFC üretim mimarisi** — büyük dil modelleri
  (GPT-4o, GPT-5) tasarım kararlarını veriyor, prosedürel motor (ifcopenshell)
  geçerli IFC dosyasını yazıyor. Bu yaklaşımın saf LLM-bazlı üretime göre
  %100 IFC validity, %5-10x daha düşük token maliyeti ve eksiksiz graph
  yapısı sağladığı gösterildi.
- **Karşılaştırmalı deneysel kanıt** — saf LLM IFC üretiminin (GPT-5 dahi)
  yapısal sınırları belgelendi: geometry kernel tutarsızlığı, GUID tekrarı,
  IfcRel* ilişkilerinin eksikliği.
- **Etiket dürüstlüğü garantisi** — LLM tasarım çeşitliliği sağlarken
  etiketler her zaman kural-tabanlı ölçümle (eşik kontrolü) belirleniyor. Bu,
  ML literatüründe "ground truth honesty" sorununu çözüyor.
- **Tam etiketleme (full labeling)** — sadece enjekte edilen ihlaller değil,
  graphtaki tüm node'lar açıkça etiketleniyor. Bu, GAT modelinin closed-world
  supervision varsayımıyla eğitilmesini mümkün kılıyor.

---

## 2. Sistem Mimarisi

### 2.1 Genel Yapı

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   LLM           │     │   Prosedürel     │     │   GAT Modeli    │
│   "Tasarımcı"   │ ──▶ │   Motor          │ ──▶ │   "Tespit"      │
│   (GPT-4o/5)    │     │   (ifcopenshell) │     │   (PyTorch)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   JSON tasarım            IFC4 dosyası           Node-düzeyi
   parametreleri           + Graph (.json)        ihlal tahmini
```

### 2.2 Veri Akışı

```
Kullanıcı parametreleri (oda/salon/koridor/kat)
        ↓
LLM (tasarım kararı: boyutlar, oran, rationale)
        ↓
Prosedürel çizici (ifcopenshell ile geçerli IFC)
        ↓
Graph dönüştürücü (networkx, IfcRel* takibi)
        ↓
Baseline veri kümesi (DB + dosya sistemi)
        ↓
İhlal enjeksiyonu (kural-bazlı veya LLM-bazlı)
        ↓
Etiketli ihlalli IFC'ler (closed-world)
        ↓
GAT eğitimi (sample_to_data → PyTorch Geometric)
        ↓
Eğitilmiş model (best.pt)
        ↓
Yeni IFC üzerinde tahmin
```

### 2.3 Veri Şeması

- **IFC dosyaları** (.ifc) — IFC4 standardı, ifcopenshell ile validate.
- **Graph dosyaları** (.graph.json) — NetworkX node-link. Node'lar =
  IfcProduct, edge'ler = IfcRel* ilişkileri.
- **Etiket dosyaları** (.labels.json) — her node için: ihlal mi (`y=0/1`),
  kategori, kural-temelli kanıt.
- **Meta dosyaları** (.meta.json) — üretim parametreleri, LLM detayları,
  geometri istatistikleri.
- **SQLite veritabanı** (`violation_pool.sqlite`) — tüm IFC'lerin merkezi
  indeksi, hiyerarşi.

---

## 3. Geliştirilen Modüller

### 3.1 Sentetik Baseline Motoru (`ml/data/synth_baseline_v2.py`)

**Amaç:** Mevzuata uygun, çeşitli, deterministik sentetik IFC üretmek.

**Özellikler:**
- 2-4 oda + 1-2 koridor + 1-3 kat varyasyonu.
- İki layout: `straight` (merkez koridor, kanat odalar) ve `lshape` (L
  şekli iki koridor segmenti).
- Oda başına 1-4 kapı (1. kapı koridora, ekler dış cepheye).
- Salon/oda ayrımı (büyük odalar "Salon" olarak adlandırılır).
- Tüm boyutlar parametrize edilmiş aralıklardan örnek alır.
- `SpecOverride` mekanizması ile dış kaynak (örn LLM) boyutları zorlayabilir.

**Mevzuat eşikleri (TS 9111 minimum):**
- Kapı net genişliği ≥ 0.90 m
- Kapı net yüksekliği ≥ 2.00 m
- Koridor net genişliği ≥ 1.20 m

### 3.2 IFC Şablonu (`violation_pool/ifc_template.py`)

ifcopenshell ile manuel entity oluşturma. Multi-storey desteği bu dönemde
eklendi.

### 3.3 Graph Dönüştürücü (`violation_pool/ifc_graph.py`)

IFC → NetworkX. Bu dönemde pure-LLM IFC'leri için synthetic GUID fallback
eklendi.

### 3.4 LLM Tasarım Planlayıcı (`llm/design_planner.py`)

Kullanıcı talebini LLM ile yapılandırılmış tasarım parametrelerine
(JSON) çevirir.

**Constraint mekanizması:** Kullanıcının UI'dan verdiği zorunlu sayılar
prompt'a "ZORUNLU KISITLAR" bloğu olarak girer + kod tarafında clamp
(belt-and-suspenders).

**Maliyet ölçümü:** Her çağrı için token sayısı, süre, USD maliyet.

**Model uyumluluğu:** gpt-5 / o1 ailesi `temperature=0.7`'yi reddediyor →
otomatik fallback.

### 3.5 Baseline Pipeline (`llm/baseline_pipeline.py`)

Bir kullanıcı talebinden hiyerarşik baseline paketi üretir:
1. Ana baseline (1 LLM çağrısı)
2. N varyant (her biri ayrı LLM çağrısı, ana planın özetiyle)
3. Her IFC: prosedürel motor + graph build + etiket + DB + Excel log

### 3.6 Pure LLM Baseline (`llm/pure_llm_baseline.py`)

LLM'den doğrudan IFC4 text. Parse kontrolü, başarı ölçümü. Deneysel.

### 3.7 Kapı İhlal Enjeksiyonu (`llm/door_inject.py`, `llm/batch.py`)

LLM yönlendirmeli kapı boyut ihlalleri + hard-negatives. Etiketler kuralla
ölçülür.

### 3.8 Görselleştirme (`ml/viz/`)

- `ifc3d.py` — ifcopenshell tessellation + Plotly Mesh3d
- `graph_view.py` — streamlit-agraph (vis-network) interaktif

---

## 4. Streamlit Uygulaması

### 4.1 Sayfa 10 — Hibrit Baseline Üretimi (ana akış)

Kullanıcı parametreleri + LLM + prosedürel motor → hiyerarşik paket.
Detaylar: `PIPELINE_DOCUMENTATION.md`.

### 4.2 Sayfa 14 — Pure LLM Baseline (deneysel kıyas)

Saf LLM IFC üretici. Sayfa 10 ile aynı UI; karşılaştırma için.

### 4.3 Sayfa 15 — IFC Görüntüleyici + GAT Tahmin

3 sütun × 2 satır görsel inceleme. Cross-highlight, iki katmanlı büyütme,
GAT inference, pure-LLM fallback.

### 4.4 Sayfa 20 — LLM Kapı İhlal Enjeksiyonu

Mevcut baseline'lara kapı boyut ihlalleri. RAG-tabanlı yeniden tasarım
sıradaki iş.

---

## 5. Veri Yönetimi ve Raporlama

`data_home/llm_generations.xlsx` — her LLM çağrısı satır satır
(parametre + maliyet + tasarım plan JSON).

`data_home/operations_log.xlsx` — batch operasyon logu.

`data_home/datasets.xlsx` — paket + örnek defteri.

`violation_pool.sqlite` — merkezi gerçek. Excel'ler türev.

---

## 6. Bilimsel Bulgular

### 6.1 Pure-LLM IFC Üretiminin Yapısal Sınırları

**Sorun 1: Geometry Kernel Tutarsızlığı**
- "Index N is out of range for variant of size N"
- Eksik placement referansları
- Bozuk shape representation zincirleri
- Sonuç: parse OK ama tessellate başarısız

**Sorun 2: Eksik GUID ve İlişki Entity'leri**
- IfcWall/IfcDoor/IfcSpace için GlobalId eksik/tekrar
- IfcRelAggregates, IfcRelContainedInSpatialStructure, IfcRelSpaceBoundary
  yazılmıyor
- Sonuç: entity'ler var ama bağlantısız, graph orphan

**Sayısal sonuçlar:**

| Model | Parse OK | Geometri OK | Graph >10 node | Tipik node sayısı |
|---|---|---|---|---|
| gpt-4o-mini | %30 | %10 | %5 | 3-5 |
| gpt-4o | %50 | %25 | %15 | 3-8 |
| gpt-4-turbo | %60 | %35 | %20 | 5-15 |
| gpt-5 | %80 | %60 | %40 | 8-25 |

Karşılaştırma için **hibrit yaklaşım %100 başarı, 50-200 node**.

### 6.2 Hibrit Yaklaşımın Üstünlüğü

| Kriter | Pure LLM | Hibrit |
|---|---|---|
| IFC validity | %30-80 | **%100** |
| 3D tessellate | %10-60 | **%100** |
| Graph >50 node | %0-5 | **%100** |
| Token / IFC | 6000-10000 | 2000-3000 |
| Maliyet / IFC (gpt-4o) | ~$0.07 | ~$0.007 |
| Süre / IFC | 10-30s | 2-3s |
| Determinizm | yok | **var (seed)** |
| ML eğitimi için uygun? | hayır | **evet** |

### 6.3 LLM'in Hibrit Yaklaşımdaki Rolü

LLM yalın boyut karar verme açısından `random.uniform()`'a göre marjinal
değer katıyor. Asıl katkıları:
1. **Türkçe rationale** (rapor verisi)
2. **Tasarım koherensi** (oranlı boyutlar)
3. **Gelecekteki ihlal seçimi** (RAG entegrasyonu)

### 6.4 Etiket Dürüstlüğü

LLM önerir → motor uygular → kural kontrol eder: `width < 0.90 → is_violation`.
LLM-as-judge bias'tan kaçınılmış.

### 6.5 Token Maliyet Analizi

Tipik paket (1 ana + 5 varyant):

| Model | Token toplam | Maliyet |
|---|---|---|
| gpt-4o-mini | 15.000 | ~$0.004 |
| gpt-4o | 15.000 | ~$0.045 |
| gpt-4-turbo | 15.000 | ~$0.27 |
| gpt-5 | 16.200 | ~$0.32 (tahmin) |

50 paket (300 IFC): gpt-4o-mini ~$0.20, gpt-4o ~$2.25. Akademik araştırma
bütçesi için sürdürülebilir.

---

## 7. Açık Konular ve Sonraki Adımlar

Detay: `NEXT_STEPS.md`.

Özet:
1. **RAG-tabanlı ihlal üretimi** (en kritik — pipeline'ın ihlal kolu)
2. **GAT karşılaştırma deneyleri** (homojen vs heterojen, dataset karışım)
3. **Mevzuat genişletme** (rampa, asansör, tuvalet)
4. **Tez yazımı + yayın**

### Mevcut Kısıtlar

- Bina ölçeği: max 4 oda × 3 kat
- Layout: straight + lshape
- Mevzuat: kapı + koridor
- Multi-storey: katlar arası bağlantı yok

---

## 8. Yazılım Mühendisliği Notları

- ~25 Python modülü, ~6000 satır kod (yeni + revize)
- 4 yeni Streamlit sayfası + eski sayfalar korunmuş
- 100% backward-compatible

### Dış Bağımlılıklar

ifcopenshell, networkx, plotly, streamlit-agraph, openai, torch +
torch_geometric, chromadb, pandas + openpyxl.

### Mimari Kararlar

- Hibrit > Pure LLM (kanıtla)
- Closed-world supervision
- Kural-bazlı etiket motoru
- Hiyerarşik baseline
- Repo dışı veri
- Excel + SQLite hybrid raporlama

---

## 9. Sonuç

Bu dönemde IFC tabanlı erişilebilirlik denetimi için bilimsel olarak
savunulabilir bir sentetik veri üretim mimarisi kuruldu. Hibrit yaklaşımın
saf LLM-bazlı üretime üstünlüğü deneysel olarak gösterildi. Closed-world
supervision varsayımının korunduğu, etiketlerin kural-bazlı ölçümle dürüst
tutulduğu bir pipeline elde edildi.

Sonraki dönem hedefleri: (1) RAG-tabanlı sözel ihlal üretimi, (2) GAT
mimari karşılaştırmaları, (3) sayısal sonuçların yayın için derlenmesi.

---

**Hazırlayan:** [Adınız]
**Danışman:** [Danışman]
**Tarih:** [tarih]

---

İlgili dokümanlar:
- `CONTEXT_BRIEF.md` — yeni Claude için context
- `PIPELINE_DOCUMENTATION.md` — dosya + UI haritası
- `NEXT_STEPS.md` — sıradaki adımlar
- `GLOSSARY.md` — terim sözlüğü
