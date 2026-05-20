# Erişilebilirlik Dataset Viewer

[codex1](https://github.com/doganalci/codex1) projesinin ürettiği **erişilebilirlik
ihlal havuzu + ihlalli IFC dataset'ini** okuyup inceleyen, Streamlit tabanlı,
**read-only** bir görüntüleyici.

Veri tabanını yazmaz, dosyaları değiştirmez; sadece görselleştirir.

## Özellikler

- 📂 **Veri kaynağı seçici** — Sidebar'da iki mod:
  📦 *codex1 (komşu repo)* otomatik bulur, 📂 *Özel yol* ile başka dataset.
- 📜 **İhlal Havuzu tabı** — Run seç → meta header (method/llm/batch/token),
  filtreli Excel benzeri ihlal tablosu (kategori/şiddet/batch/arama),
  kategori-şiddet-batch dağılım grafikleri, satır seç → kanıt zinciri,
  `exports/` Excel'i tek tıkla indir.
- 📋 **Sidebar dosya listesi** — Kind filtresi + isim/id araması + scrollable
  liste. Tıklayınca detay paneli yüklenir.
- ◀▶ **Sıralı gezinme** — Detay başlığında `Önceki / 3 / 47 / Sonraki`.
  Sidebar'ın filtreli sırasını takip eder.
- 🗂️ **Dataset browser** — Tüm baseline / violated / imported IFC'leri filtrele
  (tür, status, havuz, etiket sayısı, isim arama). Tablo + "Detayı aç" butonu.
- 🧱 **3D viewer** — `ifcopenshell.geom` ile tessellate, plotly Mesh3d.
  Kırmızı = ihlal, sarı = decoy, soluk = normal.
- 🕸️ **Graph viewer** — NetworkX `graph.json`'u plotly 2D olarak çizer
  (pan + scroll zoom + embedded spring layout).
- 📐 **Baseline ↔ violated yan yana** — Violated IFC açıkken toggle ile 3D ve
  Graph sekmelerinde solda baseline + sağda violated.
- 🧬 **Türev listesi** — Baseline açıkken o baseline'dan türetilmiş tüm
  violated'lar tabloda; "→ Aç" ile hızlı geçiş.
- 🏷️ **Etiket tablosu + kanıt zinciri** — Her etiket için before/after,
  havuz kaydı, evidence dokümanları.
- 📄 **Meta + dosya indirme** — meta.json, params, prompt; tüm ürün
  dosyalarını indir.
- 📊 **Sidebar özet** — Havuz/IFC/etiket sayıları, toplam token.

## Yol Haritası (sıradaki)

- Etiket analitiği (kategori/severity/action dağılımı, cross-table)
- Baseline ↔ violated **graph diff** (eklenen node / değişen edge vurgusu)
- Decoy/gerçek tahmin oyunu (decoy fooling rate)
- HuggingFace datasets / Parquet / COCO benzeri export

## Kurulum

Viewer codex1 ile **aynı** bağımlılıkları kullanır (`streamlit`, `ifcopenshell`,
`plotly`, `networkx`, `pandas`). Codex1'in `violation-pool` conda env'i varsa
onu kullan; ayrı env açmaya gerek yok:

```bash
conda activate violation-pool
# Eksik bir paket çıkarsa:
pip install -r requirements.txt

cp .env.example .env  # VIEWER_DATASET_ROOT=../codex1
streamlit run app.py
```

Sıfırdan kurulum için:

```bash
conda create -n violation-pool python=3.11 -y
conda activate violation-pool
conda install -c conda-forge ifcopenshell  # geometri kernel
pip install -r requirements.txt
```

`VIEWER_DATASET_ROOT` codex1'in kök klasörünü gösterir (içinde
`violation_pool.sqlite` ve `ifc_models/` olmalı). Sol panelden de
çalışırken değiştirilebilir.

## Yapı

```
app.py                 # Streamlit giriş noktası
viewer/
  config.py            # DatasetPaths — dataset kök çözümleme
  db.py                # SQLite read-only sorgular (mode=ro)
  ifc3d.py             # IFC → plotly Mesh3d
  graph_view.py        # graph.json → plotly 2D
```

## Notlar

- Dataset SQLite `mode=ro` URI ile açılır; yanlışlıkla yazılamaz.
- Codex1'in `file_path / graph_path / meta_path / labels_path` alanları
  görece olabilir; `resolve_artifact` dataset köküne göre çözer.
- ifcopenshell'in geometry kernel'i (OpenCascade) yoksa 3D sekmesi açıklayıcı
  hata verir, ama browser/graph/etiket çalışmaya devam eder.
