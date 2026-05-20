# IFC Erişilebilirlik İhlal Dataset (`{zip_name}`)

**Üretim zamanı:** {generated_at}
**Dahil edilen türler:** {kinds}
**Graph dosyaları dahil:** {include_graph}

## İçerik

- **{n_examples}** adet IFC örneği
- **{n_labels}** adet etiketli ihlal kaydı
  - `applied`: {n_applied} (gerçek ihlaller)
  - `decoy`: {n_decoys} (yapay / yanıltıcı — modelin yanlış pozitif eğilimini ölçmek için)

## Zip Yapısı

```
examples/{{kind}}/<name>.ifc        # ham IFC dosyaları
labels/{{kind}}/<name>.labels.json  # ihlal etiketleri (aşağıdaki şema)
meta/{{kind}}/<name>.meta.json      # üretim metadata'sı (yöntem, model, tarih)
graphs/{{kind}}/<name>.graph.json   # NetworkX node-link grafik (varsa)
manifest.json                       # tüm örneklerin indeksi + istatistik
README.md                           # bu dosya
```

`{{kind}}` şu olabilir: `baseline` (temiz model), `violated` (ihlal enjekte edilmiş), `imports` (kullanıcı yüklemesi).

## Etiket Şeması

Her `<name>.labels.json` bir JSON dizisidir. Her kayıt aşağıdaki alanları içerir:

```json
{{
  "violation_id": "uuid",
  "ifc_global_id": "IFC entity GUID (hangi nesneye uygulandı)",
  "category": "16 kategori enum'undan biri (aşağıda)",
  "severity": "düşük | orta | yüksek | kritik",
  "title": "kısa başlık",
  "description": "tek cümlelik somut kural",
  "threshold": "sayısal eşik (örn. '< 90 cm') veya null",
  "attribute": "değiştirilen IFC özelliği (örn. 'OverallWidth')",
  "before": "enjeksiyon öncesi değer",
  "after": "enjeksiyon sonrası değer",
  "is_decoy": false,
  "status": "applied | failed | pending",
  "evidence": [
    {{
      "document": "kaynak PDF adı",
      "page": 12,
      "clause": "madde no",
      "snippet": "ilgili metin parçası"
    }}
  ]
}}
```

### `is_decoy` Hakkında

Decoy etiketler **kasten yanlış**tır — IFC modelinde aslında **bulunmayan** veya **uygulanmamış** ihlallerdir. Eğitilmiş modelin bu örneklere "ihlal değil" demesi beklenir. Decoy yanlış pozitif oranı (`decoy_fpr`) modelin güvenilirliğinin ana göstergesidir.

### 16 İhlal Kategorisi

`category` alanı aşağıdaki değerlerden birini alır:

| # | Kategori | Tipik konu |
|---|----------|-----------|
| 1 | Yaya erişimi | Yaya yolları, geçişler, kaldırım sürekliliği |
| 2 | Giriş | Bina ana girişi, otomatik kapı, eşik |
| 3 | Kapı/Koridor | Geçiş genişliği, kapı açılım açısı |
| 4 | Rampa | Eğim, uzunluk, sahanlık |
| 5 | Merdiven | Rıht, basamak, sahanlık, korkuluk |
| 6 | Korkuluk/Küpeşte | Yükseklik, çift seviye, kavrama formu |
| 7 | Asansör | Kabin boyutu, buton yüksekliği, kapı genişliği |
| 8 | Tuvalet/Banyo | Manevra alanı, tutamak, donanım yüksekliği |
| 9 | Mutfak | Tezgah yüksekliği, diz boşluğu |
| 10 | Otopark | Engelli park yeri, genişlik, konum |
| 11 | Uyarı yüzeyi | Hissedilebilir yüzeyler, tehlike işaretleri |
| 12 | Yönlendirme/İşaretleme | Yön levhaları, Braille, kontrast |
| 13 | Görsel/Kontrast | Renk kontrastı, kenar belirginliği |
| 14 | Aydınlatma | Lux seviyesi, gölgelenme |
| 15 | Manevra alanı | 150×150 cm dönüş, yan boşluk |
| 16 | Eşik/Kot farkı | İzin verilen kot farkı, eşik yüksekliği |

### Şiddet Seviyeleri

| Severity | Anlam |
|----------|-------|
| `düşük` | Konfor düşüşü, erişimi engellemez |
| `orta` | Bazı kullanıcılar için engelleyici |
| `yüksek` | Çoğu engelli kullanıcı için engelleyici |
| `kritik` | Mevzuata aykırı; düzeltilmeden kullanılamaz |

## İstatistik

### Kategori Dağılımı
{categories}

### Şiddet Dağılımı
{severities}

## Etiketleme Yöntemi

Etiketler aşağıdaki yöntemlerden biriyle üretilmiştir (her IFC'nin `meta.json`'unda `method` alanı vardır):

1. **naive** — Tek promt ile LLM'e doğrudan sor
2. **optimized** — İyileştirilmiş promt + JSON şema kontrolü
3. **rag** — Standart/yönetmelik PDF'lerinden RAG ile bağlam çekip LLM'e ver
4. **finetune** — Standart belgelerle fine-tune edilmiş LLM ile üret

Üretim hattı:
1. **Baseline IFC** parametrik olarak oluşturulur (LLM yalnızca JSON spec üretir; geometriyi `ifcopenshell` inşa eder).
2. **İhlal havuzu** seçilen yöntemle üretilir; her ihlal `evidence` zinciri (PDF madde alıntıları) ile gelir.
3. **Enjeksiyon adımı** ihlal kuralını ilgili IFC entity'sine uygular (`OverallWidth`'i 90→70 cm yapma, kolon ekleyerek engelleme, vb.).
4. Başarısız enjeksiyonlar `status='failed'` olarak işaretlenir; opsiyonel olarak havuzdan yedek ihlal denenir.
5. Decoy ihlaller havuza karıştırılır — kayıt edilen ama IFC'ye uygulanmayan.

## Eğitim İçin Kullanım İpuçları

- **Pozitif örnekler:** `status='applied' AND is_decoy=False` olan label kayıtlarındaki `ifc_global_id`'ler.
- **Negatif örnekler:** Aynı IFC'deki diğer tüm node'lar + decoy etiketler.
- **Train/val/test bölmesi:** `meta.json.baseline_id` üzerinden grupla — aynı baseline'dan türeyen IFC'ler aynı split'te kalmalı (data leakage'ı önlemek için).
- **Metrik:** `precision`, `recall`, `f1`, ve özellikle `decoy_fpr` (decoy'ları yanlışlıkla pozitif tahmin etme oranı).

## Lisans ve Atıf

İçerideki PDF kaynaklarının lisanslarına dikkat edin — bunlar `evidence` alanında alıntılanmıştır ama tam PDF'ler bu pakete dahil değildir. Üretilen IFC'ler ve etiketler için: kendi projenizde kullanırken bu repoya referans verin.

---

*Bu README otomatik olarak `dataset_export/prepare_dataset.py` tarafından oluşturulmuştur.*
