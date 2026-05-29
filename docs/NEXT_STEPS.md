# Sıradaki Adımlar (Önceliklendirilmiş)

Bu dönem hibrit baseline üretimi + viewer + pure-LLM kıyas tamamlandı.
Aşağıdaki çalışmalar **öncelik sırasına göre** sıralanmıştır.

---

## P1 — RAG-tabanlı İhlal Üretimi (Acil — Tez İçin Kritik)

**Amaç:** TS 9111 / TS ISO 21542 / ADA mevzuat metnini ChromaDB'ye ingest
ederek, LLM'in sözel ihlal seçip uygun IFC eylemine çevirmesi.

**Durum:** Altyapı parçaları mevcut, üst seviye akış yok.

**Yapılması gereken:**

1. **Corpus hazırla:** TS 9111 metni (Markdown veya PDF) `data_home/docs/`
   altına. ADA özetleri opsiyonel.
2. **Sayfa hazırla** (`pages/20_LLM_Ihlal_Uretimi.py` yeniden tasarımı):
   - Baseline paket seç
   - İhlal kategorileri (çoklu seçim): kapı, koridor, eşik/kot, rampa
   - Baseline başına Y ihlal + hard-negative
   - RAG k=8 chunk çek → LLM ile sözel ihlal seç → ifc_inject'e devret
3. **Pipeline** (`llm/violation_pipeline.py` yeni dosya):
   - rag.retrieve(query, k) → kural chunk'ları
   - generate_rag(OPTIMIZED_PROMPT, chunks) → JSON sözel ihlaller
   - inject_violations(ifc, violations) → modify_attribute / add_obstruction
   - Kuralla ölç → etiket
   - Tam etiketleme (graph'ın geri kalanı = clean)
4. **DB + Excel:** Her ihlal üretimi `llm_generations.xlsx`'e satır.
   `operations_log.xlsx`'e batch özet.

**Mevcut yararlı modüller:**
- `violation_pool/rag.py` (ingest + retrieve hazır)
- `violation_pool/llm.py:generate_rag` (RAG chunks + prompt → ihlal)
- `violation_pool/prompts.py:OPTIMIZED_PROMPT` (rol + 16 kategori + JSON şema)
- `violation_pool/ifc_inject.py:inject_violations` (sözel → IFC eylem)

**Tahmini kapsam:** ~400 satır yeni kod, 1-2 günlük iş.

**Tez katkısı:** Pipeline'ın "ihlal üretimi" kolu tamamlanmış olur. Sentetik
veri kalitesi gerçek mevzuata bağlanır.

---

## P2 — GAT Karşılaştırma Deneyleri

**Amaç:** Modelin başarımını farklı veri ve mimari koşullarda kıyaslamak.

**Deneyler:**

1. **Homojen GAT vs Heterojen GAT** (`ml/model/gat.py` vs `hetero_gat.py`)
   - Edge type bilgisi katkısı?
   - Aynı paket, aynı seed, sadece model değişiyor
2. **Hibrit-only dataset vs Hibrit + Pure-LLM karışık dataset**
   - Pure-LLM IFC'lerin model dayanıklılığına etkisi
   - Out-of-distribution sınama
3. **Eşik dağılımı analizi**
   - Hard-negative oranı (yakın ama eşik üstü) modelin precision'ını nasıl
     etkiliyor?

**Mevcut altyapı:**
- `legacy/04_GAT_Egitim.py` — eğitim sayfası
- `legacy/05_GAT_Test.py` — test sayfası
- `ml/train/metrics.py` — F1, P/R, AUC, MCC, decoy_fpr, per-category
- `experiments.xlsx` — `record_run()` ile her eğitim kaydedilir

**Tahmini kapsam:** ~50 satır yeni karşılaştırma scripti + 4-8 saat eğitim
+ rapor.

---

## P3 — Mevzuat Kapsamının Genişletilmesi

**Amaç:** Şu an sadece kapı + koridor. Diğer kategorileri eklemek.

**Kategoriler (TS 9111 + TS ISO 21542):**
1. Eşik / kot farkı (≤ 1.3 cm veya rampa)
2. Rampa eğimi (≤ %8 normal, ≤ %5 önerilen)
3. Asansör kabin boyutları
4. Tuvalet / WC manevra alanı
5. Korkuluk yüksekliği + sürekliliği
6. Aydınlatma seviyesi
7. Görsel kontrast
8. Yönlendirme / işaretleme

**Yapılması gereken:**
- `llm/door_inject.py`'in genişletilmesi VEYA kategori başına ayrı modül
- `synth_baseline_v2.py` motoruna rampa, asansör, tuvalet gibi yapısal
  eleman eklemesi
- Her kategori için ölçüm fonksiyonu (kural-bazlı `is_violation`)

**Tahmini kapsam:** Her kategori ~1 günlük iş. Tüm kategoriler ~2 hafta.

---

## P4 — Tez Yazımı + Yayın Hazırlığı

**Yapılması gereken:**

1. **Deney tablolarını derle:**
   - `llm_generations.xlsx` pivot: model × tur × başarı oranı
   - `experiments.xlsx`: F1 trajektorisi, en iyi modeller
2. **Ekran görüntüleri (sayfa 15):** baseline / ihlalli / tahmin yan yana
3. **Mimari diyagramı:** sayfa 10 akışı + sayfa 14 vs sayfa 10 kıyaslama
4. **Anketler / vaka çalışmaları (opsiyonel):** gerçek mimarlardan bir
   baseline alıp pipeline ile değerlendirme

---

## P5 — Geliştirme Borçları (Düşük Öncelik)

- **GAT modeli karşılaştırma sayfası** — `experiments.xlsx`'den otomatik
  tablo, en iyi N model, parametre sweep görselleştirme.
- **RAG ingest sayfası** — UI'dan PDF/MD yükleyip ChromaDB'ye ekle.
- **Pure-LLM IFC çeşitlilik analizi** — hibrit dataset'e karışım oranı vs
  başarım.
- **Export sayfası** — paketleri ZIP olarak dışa, başkasıyla paylaşmak için.
- **Multi-storey için merdiven/asansör elemanları** — şu an katlar boş,
  bağlantısız.

---

## Yapılmayanlar / Kapsam Dışı

- **Gerçek bina IFC'leri** — bu çalışma sentetik veri ile sınırlı (tez
  kapsamı). Gerçek BIM modelleri ileri çalışma.
- **3D viewer dışı geometrik analiz** — Brep / NURBS işlemleri yapılmıyor.
- **Multi-language (İngilizce arayüz)** — şu an sadece Türkçe.

---

**Strateji önerisi:** P1 (RAG ihlal) en kritik. Tez için pipeline'ın "ihlal"
kolu tamamlanmalı. P2 ve P3 paralel ilerleyebilir. P4 tez yazımına başlarken.
