# IFC All Violeter Viewer & Analyser

IFC tabanlı erişilebilirlik ihlal dataset'i üretme, görselleştirme,
GAT ile tespit modeli eğitme ve dışa aktarmayı tek çatı altında toplayan
bir Streamlit + Python projesi.

Üç ayrı geliştirme adımının birleştirilmiş halidir:
**codex1** (üretim çekirdeği) + **ifc_and_graph_viewer** (salt-okunur viewer)
+ **ifc_graph_analysis** (GAT eğitim & spatial analiz).

---

## Mimari: Kod / Veri Ayrımı

İki repo birbirinden bağımsız `git pull` ile güncellenir:

```
~/Desktop/doga_full_ifc_prog/                 ← parent (isim önemli değil)
├── ifc_all_violeter_viewer_analiyser/        ← PROGRAM (bu repo)
└── ifc_desktop_doc_dataset/                  ← VERİ (.env, PDF, IFC, model)
```

Program **veri klasörünü** şu sırayla bulur (ilk uyan kazanır):

1. `--data-home /abs/yol` CLI argümanı
2. `IFC_DATA_HOME` ortam değişkeni
3. **Kardeş klasör** araması: `<bu_repo>/../ifc_desktop_doc_dataset/`
4. `~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset/`

Standart yerleşimde **hiçbir ayar gerekmez**.

---

## Kurulum (Tek Sefer)

```bash
mkdir -p ~/Desktop/doga_full_ifc_prog && cd ~/Desktop/doga_full_ifc_prog \
 && git clone https://github.com/doganalci/ifc_desktop_doc_dataset.git \
 && git clone https://github.com/doganalci/ifc_all_violeter_viewer_analiyser.git \
 && cd ifc_all_violeter_viewer_analiyser \
 && conda activate violation-pool \
 && pip install -r requirements.txt \
 && cp .env.example ../ifc_desktop_doc_dataset/.env
```

Sonra `~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset/.env` dosyasını
açıp `OPENAI_API_KEY=...` satırını doldurun.

> **Torch / PyG:** GPU varsa `requirements.txt`'i kurmadan önce
> kendi CUDA wheel'inizi indirin (bkz. dosya başındaki yorum).

---

## Sonraki Çalıştırmalar (Silme YOK, sadece pull)

### Program güncellemesi
```bash
cd ~/Desktop/doga_full_ifc_prog/ifc_all_violeter_viewer_analiyser \
 && git pull \
 && pip install -r requirements.txt
```

### Veri senkronu (başka makinede ürettiğin veriyi çek / paylaş)
```bash
cd ~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset && git pull
# çalıştıktan sonra paylaşmak için:
cd ~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset \
 && git add -A && git commit -m "veri güncelleme" && git push
```

### Programı çalıştır
```bash
cd ~/Desktop/doga_full_ifc_prog/ifc_all_violeter_viewer_analiyser \
 && conda activate violation-pool && streamlit run app.py
```

---

## Repo Yapısı

```
ifc_all_violeter_viewer_analiyser/
├── app.py                       ← ANA Streamlit (üretim + dataset pipeline)
├── paths.py                     ← IFC_DATA_HOME çözümleme (tek kaynak)
├── violation_pool/              ← Çekirdek: LLM, RAG, IFC üretim/enjeksiyon
│   ├── config.py  storage.py  pipeline.py
│   ├── llm.py  prompts.py  rag.py  finetune.py
│   ├── ifc_gen.py  ifc_inject.py  ifc_template.py
│   ├── ifc_graph.py  ifc_viewer.py  graph_viewer.py
│   └── excel_export.py
├── viewer/                      ← Salt-okunur dataset viewer
│   ├── viewer_app.py  config.py  db.py  ifc3d.py  graph_view.py
├── ml/                          ← GAT eğitimi + statik analiz
│   ├── data/  model/  train/  analysis/  viz/  app/  scripts/
├── pages/                       ← Streamlit multipage (ml sayfaları)
│   ├── 01_Model_Görüntüleyici.py
│   ├── 02_Statik_Analiz.py
│   ├── 03_GAT_Tespiti.py
│   ├── 04_GAT_Egitim.py
│   └── 05_GAT_Test.py
├── dataset_export/              ← Eğitim dataset paketleyici (CLI)
│   ├── prepare_dataset.py
│   └── DATASET_README.md.tpl    ← Çıkan zip'e gömülecek README şablonu
├── requirements.txt
├── .env.example
└── README.md
```

---

## Çalıştırma Komutları

### 1) Ana üretim & pipeline UI'ı
```bash
streamlit run app.py
```
İçinde: ihlal havuzu üretimi, IFC studio, dataset pipeline, viewer
sekmeleri, ml sayfaları (Streamlit multipage olarak `pages/`).

### 2) Sadece salt-okunur viewer
```bash
streamlit run viewer/viewer_app.py
```

### 3) GAT eğitimi (CLI)
```bash
# Varsayılan: IFC_DATA_HOME altındaki veriyi kullan, ml_runs/ altına yaz
python ml/scripts/train.py --epochs 50

# Veya tam kontrol
python ml/scripts/train.py \
    --dataset-root ~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset \
    --model hetero_gat --epochs 100 --run-name deneme1
```

### 4) Eğitilmiş modeli değerlendir / tahmin üret
```bash
python ml/scripts/evaluate.py \
    --checkpoint ml_runs/<run_id>/best.pt \
    --config ml_runs/<run_id>/config.json

python ml/scripts/export_predictions.py \
    --checkpoint ml_runs/<run_id>/best.pt \
    --config ml_runs/<run_id>/config.json \
    --out-dir predictions/<run_id>
```

### 5) Eğitim için veri seti paketle (zip + README)
```bash
# Sadece ihlalli IFC'ler:
python dataset_export/prepare_dataset.py --out dataset.zip

# Hepsini (baseline + violated + imports) ve grafikleri dahil et:
python dataset_export/prepare_dataset.py \
    --out dataset_full.zip --kind all --include-baseline --include-graph
```

Çıkan zip içinde `examples/`, `labels/`, `meta/`, `manifest.json`,
ve etiketleme şemasını anlatan otomatik üretilmiş `README.md` bulunur.

---

## Veri Klasörü İçeriği

`IFC_DATA_HOME` (örn. `~/Desktop/doga_full_ifc_prog/ifc_desktop_doc_dataset`)
altındaki standart düzen:

```
ifc_desktop_doc_dataset/
├── .env                         ← API key'ler (gitignore!)
├── docs/                        ← Yüklenen PDF'ler (yönetmelik, standart)
├── vectorstore/                 ← Chroma persistent index
├── ifc_models/
│   ├── baseline/                ← Üretilen temiz IFC'ler
│   ├── violated/                ← İhlal enjekte edilmiş IFC'ler + .labels.json
│   └── imports/                 ← Kullanıcı yüklediği IFC'ler
├── exports/                     ← Excel + fine-tune JSONL
├── violation_pool.sqlite        ← Tüm metadata (runs, violations, evidence)
└── ml_runs/                     ← GAT checkpoint + config.json + metrik logları
```

---

## Etiket Şeması

`dataset_export/DATASET_README.md.tpl` içinde detaylı anlatılmıştır.
Özet: **16 kategori** (Yaya erişimi, Rampa, Merdiven, …, Manevra alanı),
**4 şiddet seviyesi** (düşük/orta/yüksek/kritik), **decoy** flag'i ile
yanıltıcı etiketler, ve her ihlal için PDF kaynaklı `evidence` zinciri.

---

## Önceki Repolar

Bu repo şu üç önceki çalışmanın birleştirilmiş ve genişletilmiş halidir:

- [codex1](https://github.com/doganalci/codex1) — üretim çekirdeği
- [ifc_and_graph_viewer](https://github.com/doganalci/ifc_and_graph_viewer) — viewer
- [ifc_graph_analysis](https://github.com/doganalci/ifc_graph_analysis) — GAT + pathfinding

Eski reponun READMEsi `legacy_docs/` altında referans olarak korunmuştur.
