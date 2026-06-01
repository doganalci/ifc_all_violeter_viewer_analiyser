# Terim Sözlüğü

Pipeline'da geçen teknik terimler, kısa açıklamalar.

---

## Mevzuat ve Standartlar

**TS 9111**
Türk Standartları Enstitüsü — "Özürlüler ve hareket kısıtlılığı bulunan
kişiler için binalarda ulaşılabilirlik gerekleri" standardı. Pipeline'ın
varsayılan eşik kaynağı.

**TS ISO 21542**
"Bina yapımı — Yapılı çevrenin erişilebilirliği ve kullanılabilirliği"
uluslararası standardı. TS 9111 ile uyumlu, daha geniş kapsam.

**ADA (Americans with Disabilities Act)**
ABD erişilebilirlik standardı. Kapı genişliği 32 inch (~0.81 m), TS 9111'den
daha geniş kapsamlı bazı eşikler içerir. Pipeline'da opsiyonel referans.

**Mevzuat eşikleri (pipeline'da kullanılan)**
- Kapı net genişliği ≥ 0.90 m
- Kapı net yüksekliği ≥ 2.00 m
- Koridor net genişliği ≥ 1.20 m

---

## IFC (Industry Foundation Classes)

**IFC4**
Building Information Modeling için açık standart, ISO 16739. Pipeline'da
hep IFC4 schema kullanılıyor. Dosya formatı: ISO-10303-21 (STEP).

**IfcProject, IfcSite, IfcBuilding, IfcBuildingStorey**
Mekânsal hiyerarşi: Project en üst, sırayla Site → Building → Storey (kat).
Her biri zorunlu, IfcRelAggregates ile bağlanır.

**IfcSpace**
Bir odayı / mekânı temsil eden entity. IfcBuildingStorey ile IfcRelAggregates,
duvarlarla IfcRelSpaceBoundary üzerinden bağlanır.

**IfcWall / IfcWallStandardCase**
Duvar entity'leri. WallStandardCase düz/uniform duvarlar için kullanışlı
varyant. Pipeline'ın hibrit motoru bunu kullanır.

**IfcDoor / IfcWindow**
Açıklık entity'leri. `OverallWidth` ve `OverallHeight` attribute'ları
boyutları taşır → ihlal ölçümü buradan.

**IfcOpeningElement**
Duvardaki açıklığın geometrik tanımı. IfcRelVoidsElement ile Wall'a,
IfcRelFillsElement ile Door'a bağlanır.

**IfcRelAggregates / IfcRelContainedInSpatialStructure**
Hiyerarşik ilişkiler. Aggregates: spatial structure (Project→Site→...→Space).
ContainedInSpatialStructure: fiziksel elemanlar Storey'e (Wall, Door, vs).

**IfcRelSpaceBoundary**
Space ↔ duvar/eleman ilişkisi. Bir odanın hangi duvarlarla sınırlandığını
söyler.

**IfcRelVoidsElement / IfcRelFillsElement**
Duvar → Açıklık → Kapı zincirinin ilişki entity'leri.

**GlobalId (GUID)**
Her IFC entity'sinin 22-karakter base64 kimliği. Eşsiz olmalı; graph
builder bunu node ID olarak kullanır. **Pure-LLM IFC'lerinde sık eksik.**

**Pset (Property Set)**
Entity'lere bağlanan özellik kümeleri (örn `Pset_DoorCommon`). Pipeline
graph'a bu bilgiyi node attribute olarak kaydeder.

---

## ML / Graph

**GAT (Graph Attention Network)**
Veliçković ve diğerleri 2018, graph üzerinde attention mekanizması. Her
node'un komşularına ne kadar dikkat edeceğini öğrenir. Pipeline'da node-düzeyi
sınıflandırıcı (ihlal vs değil).

**Homojen vs Heterojen GAT**
- Homojen: tüm edge'ler aynı tip, basit
- Heterojen: edge tipi (contains, bounds, voids, vs.) ayrı embed edilir,
  edge type-aware

**PyTorch Geometric (PyG)**
Graph neural networks için PyTorch eklentisi. Pipeline `sample_to_data`
ile graph'ı `torch_geometric.data.Data` nesnesine çevirir.

**Node-düzeyi sınıflandırma**
Her node için ayrı tahmin (binary: ihlal mi?). Pipeline'ın hedef görevi.

**Closed-world supervision**
Eğitim varsayımı: etiketlenmemiş node = kesinlikle negatif (open-world: belki
de pozitif olabilir, bilmiyoruz). Tam etiketleme bu varsayımı korur.

**Hard negative**
Modelin yanlış pozitif yapma riski yüksek olan örnek. Pipeline'da:
"değiştirilmiş ama eşik üstü kalan" kapılar/koridorlar (örn 0.92 m kapı —
ihlal değil ama LLM dokunmuş, model kafası karışabilir).

**Decoy**
Yapay olarak yerleştirilen "tuzak" ihlaller (modelin ezberlememesi için).
Pipeline'da kullanılıyor ama yeni baseline akışında yok.

**F1, Precision, Recall**
Standart sınıflandırma metrikleri. Pipeline'da test sayfası bunları
hesaplar + Balanced Accuracy + MCC + AUC-ROC + decoy_fpr.

**Cross-highlight**
Bir görsel görünümde seçilen node'un diğer ilişkili görünümlerde de
vurgulanması. Pipeline'da sayfa 15: graph tıklaması → 6 panelde mavi
vurgu.

---

## LLM / RAG

**LLM (Large Language Model)**
GPT-4o, GPT-5 gibi büyük dil modelleri. Pipeline'da OpenAI Chat API
üzerinden kullanılıyor.

**Temperature**
LLM çıktısının rastgeleliğini kontrol eden parametre. Pipeline default 0.7
(tasarım çeşitliliği için). gpt-5 / o1 ailesi sadece default 1'i destekler
→ fallback otomatik.

**Token**
LLM'in işlediği "kelime parçası". OpenAI billing token başına. Pipeline her
çağrı için prompt_tokens + completion_tokens kaydeder.

**Response format JSON**
OpenAI'a "sadece JSON döndür" demek için kullanılan param. Pipeline tasarım
planı + ihlal listesi için kullanır → parse güvenli.

**RAG (Retrieval-Augmented Generation)**
LLM'e yanıt üretmeden önce ilgili doküman parçalarını veritabanından çekip
prompt'a ekleme. Pipeline'da ChromaDB tabanlı; TS 9111 kuralları için
planlanıyor (henüz tam aktif değil).

**ChromaDB**
Open-source vector database. Embed'leri sakla, semantic search yap.
Pipeline'da `data_home/vectorstore/` altında persistent.

**Embedding**
Metnin sayısal vektör temsili. OpenAI `text-embedding-3-small` kullanılıyor
(1536 boyut).

**Chunk**
RAG için bölünmüş doküman parçası. Pipeline'da 1200 karakter, 150 overlap.

**OPTIMIZED_PROMPT**
`violation_pool/prompts.py`'daki sözel ihlal üretimi için sistem prompt'u.
Rol tanımı + 16 kategori + JSON şema + örnekler. RAG ile birleştirilince
ihlal cümlesi üretir.

**Pure-LLM IFC**
LLM'in IFC text'ini doğrudan yazması. Pipeline'da deneysel; sınırları
kanıtlanmış (sayfa 14).

**Hibrit yaklaşım**
LLM tasarım kararları + prosedürel motor IFC üretimi. Pipeline'ın asıl
yöntemi (sayfa 10).

**Constraints block**
Kullanıcının UI'dan verdiği zorunlu sayıların prompt'a "ZORUNLU KISITLAR"
bloğu olarak eklenmesi. `_build_constraints_block` ile üretilir.

**Preferences block**
Gelişmiş ayarların prompt'a "TERCİH EDİLEN ARALIKLAR" bloğu olarak
eklenmesi (zorla değil, öneri).

---

## Pipeline-Spesifik Terimler

**Ana baseline**
Bir paketteki "ana / temsilci" baseline (DB'de parent_id=NULL). Sayfa 10'da
LLM tarafından template prompttan üretilir.

**Varyant baseline**
Ana baseline'a referansla üretilen, kendi LLM çağrısıyla varyasyon olarak
çıkan baseline (DB'de parent_id=ana.id).

**Violated (ihlalli)**
Bir baseline'a ihlal enjekte edilmiş türev (DB'de parent_id=baseline.id,
kind=violated).

**Paket / dataset_tag**
Bir grup baseline + violated'ı bir arada tutan etiket. Sayfa 15 ve eğitim
sayfaları paket bazında çalışır.

**Tam etiketleme (full labeling)**
İhlal enjekte edilmemiş node'ların da `status="clean"` ile etiketlenmesi.
Closed-world supervision için gerekli.

**Constraint check**
Sayfa 14'te LLM'in kullanıcı kısıtlarına ne kadar uyduğunun ölçümü
(beklenen IfcSpace sayısı vs gerçek).

**Synthetic GUID fallback**
`ifc_graph.build_graph`'in pure-LLM IFC'lerindeki eksik GlobalId'lere
sentetik kimlik atayıp graph'a node olarak eklemesi.

**method_label**
Üretim metoduna verilen dosya prefix'i: `llmgen` (LLM door inject), `pure`
(pure LLM), default `basic` (kuralsal).

---

## Dosya Türleri (özet)

| Uzantı | İçerik |
|---|---|
| `.ifc` | IFC4 dosyası (Step format) |
| `.graph.json` | NetworkX node-link |
| `.labels.json` | Node etiketleri |
| `.meta.json` | Üretim meta verisi |
| `.xlsx` | Excel raporları |
| `.sqlite` | Merkezi DB |
| `.pt` | PyTorch checkpoint |

---

**Not:** Türkçe akademik dil için "ihlal" yerine "uygunsuzluk" da geçer.
Pipeline kodunda "ihlal" tutarlı kullanılıyor.
