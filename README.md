# IFC All Violeter Viewer & Analyser

Erişilebilirlik (TS 9111 / TS ISO 21542) ihlal dataset'inin **uçtan uca tek bir Streamlit
arayüzü** altında üretildiği, görselleştirildiği ve **GAT** ile analiz edildiği konsolide repo.

Üç ayrı geliştirme adımının birleştirilmiş hâlidir:

| Kaynak repo                | Karşılığı (bu repoda)                                        |
|---                         |---                                                            |
| `codex1`                   | `violation_pool/` çekirdeği + sayfa 1                         |
| `ifc_and_graph_viewer`     | `viewer/` modülü + sayfa 2                                    |
| `ifc_graph_analysis`       | `ml/` modülü + sayfa 3-7                                      |

---

## Yerleşim (kritik)

Kod, sırlar ve veri **üç ayrı yerde** durur. Kodu silip yeniden clone'lasan
bile sırların ve verin kaybolmaz:

```
ifc_claude_code_directory/                       ← parent (kullanıcı klasörü)
├── ifc_all_violeter_viewer_analiyser/           ← KOD (bu repo, silinebilir/yenilenir)
├── secrets.txt                                  ← SIRLAR (kalıcı, gitignored)
└── data/                                        ← VERİ (kalıcı, gitignored)
    ├── violation_pool.sqlite                    ← SQLite (havuzlar, IFC kaydı, token logu)
    ├── ifc_models/
    │   ├── baseline/                            ← LLM ile üretilmiş baseline IFC'ler
    │   ├── violated/                            ← İhlal enjekte edilmiş IFC'ler
    │   └── imports/                             ← Dışarıdan eklenenler
    ├── exports/                                 ← Excel çıktıları
    ├── docs/                                    ← Yüklenen PDF/TXT
    ├── vectorstore/                             ← Chroma vektör DB
    ├── runs/                                    ← GAT checkpoint'leri (best.pt vs)
    └── logs/
```

**Cross-platform**: tüm yollar `pathlib.Path` ile çözülür. macOS, Linux, Windows fark
etmez; mutlak yol hard-code edilmemiştir. Parent klasörün adı önemli değil — repo neredeyse
kardeş `data/` ve `secrets.txt` orada aranır.

### Override (opsiyonel)

| Env değişkeni       | Etki                                                   |
|---                  |---                                                     |
| `IFC_DATA_HOME`     | Veri klasörünü parent dışında bir yere yönlendir       |
| `IFC_SECRETS_FILE`  | `secrets.txt`'i alternatif bir yere yönlendir          |

---

## Kurulum (tek sefer)

```bash
# 1) Parent klasörü hazırla ve repo'yu clone'la
mkdir -p ~/Desktop/ifc_claude_code_directory
cd ~/Desktop/ifc_claude_code_directory
git clone https://github.com/doganalci/ifc_all_violeter_viewer_analiyser.git

# 2) Conda env (önerilen)
conda create -n violation-pool python=3.11 -y
conda activate violation-pool

# (GPU varsa torch'u önce kendi CUDA wheel'inle kur — bkz. requirements.txt yorumu)
pip install -r ifc_all_violeter_viewer_analiyser/requirements.txt
conda install -c conda-forge ifcopenshell    # geometry kernel

# 3) Sırları parent klasöre koy ve doldur
cp ifc_all_violeter_viewer_analiyser/secrets.txt.example secrets.txt
$EDITOR secrets.txt    # OPENAI_API_KEY=... yaz

# 4) Çalıştır
cd ifc_all_violeter_viewer_analiyser
streamlit run app.py
```

İlk çalıştırmada `data/` klasörü ve alt dizinleri otomatik oluşur. Ana sayfa
(`app.py`) sistem durumunu (secrets, data home, DB) tablo halinde gösterir.

### Sonraki çalıştırmalar — UI değişikliği sonrası

UI değiştirildiğinde repo silinip yeniden clone'lansa bile **veri ve sırlar
parent klasörde durduğu için etkilenmez**:

```bash
cd ~/Desktop/ifc_claude_code_directory
rm -rf ifc_all_violeter_viewer_analiyser
git clone https://github.com/doganalci/ifc_all_violeter_viewer_analiyser.git
cd ifc_all_violeter_viewer_analiyser
streamlit run app.py
```

Daha hafifi: `git pull` ile statik kodu güncelle (gitignore sayesinde
`data/` ve `secrets.txt` zaten korunur):

```bash
cd ifc_all_violeter_viewer_analiyser && git pull
```

---

## Sayfalar (sol menü)

| # | Sayfa                             | Ne yapar                                                                                  |
|---|---                                |---                                                                                        |
| 0 | **Ana Sayfa** (`app.py`)          | Sistem durumu (secrets, veri, DB), kısa istatistik, sayfa nav                              |
| 1 | **Havuz Oluşturma & IFC Stüdyo**  | LLM/RAG/Fine-tune ile ihlal havuzu üret; baseline IFC üret + ihlal enjekte et             |
| 2 | **Dataset Görüntüleyici**         | Read-only browser: havuz tablosu, IFC listesi, 3D + graph + etiket karşılaştırma          |
| 3 | **Model Görüntüleyici**           | Tekil IFC modeli 3D + interactive graph, click-to-cross-highlight, baseline↔violated 2×2 |
| 4 | **Statik Analiz**                 | İki oda arasında en kısa / en geniş / en erişilebilir yol                                 |
| 5 | **GAT Eğitim**                    | Hetero-GAT in-app eğitim, hiperparam formu, canlı epoch progress                          |
| 6 | **GAT Tespiti**                   | Tekil IFC üzerinde inference — TP/FP/FN/decoy aldanma                                     |
| 7 | **GAT Test**                      | Toplu test (train/val/test/manuel split) + her IFC için satır                             |

---

## Repo yapısı

```
ifc_all_violeter_viewer_analiyser/
├── app.py                          ← Streamlit giriş (landing/status)
├── paths.py                        ← Tek noktadan yol & secrets çözümleme
├── secrets.txt.example             ← Boş şablon (parent'a kopyalanacak)
├── requirements.txt
├── pages/                          ← Streamlit multipage UI
│   ├── 01_Havuz_Olusturma_ve_IFC_Studyo.py
│   ├── 02_Dataset_Goruntuleyici.py
│   ├── 03_Model_Goruntuleyici.py
│   ├── 04_Statik_Analiz.py
│   ├── 05_GAT_Egitim.py
│   ├── 06_GAT_Tespiti.py
│   └── 07_GAT_Test.py
├── violation_pool/                 ← codex1 çekirdeği
│   ├── config.py  storage.py  llm.py  prompts.py  pipeline.py
│   ├── rag.py  finetune.py  excel_export.py
│   ├── ifc_gen.py  ifc_inject.py  ifc_graph.py  ifc_template.py
│   └── ifc_viewer.py  graph_viewer.py
├── viewer/                         ← Read-only dataset viewer
│   └── config.py  db.py  ifc3d.py  graph_view.py
├── ml/                             ← GAT eğitim & inference
│   ├── app/state.py                ← paylaşımlı sidebar + cache
│   ├── data/                       ← dataset loader, splits, features
│   ├── model/                      ← gat.py (homogen), hetero_gat.py
│   ├── train/                      ← config, loop, metrics
│   ├── analysis/                   ← pathfind.py
│   ├── viz/                        ← ifc3d.py, graph_view.py
│   └── scripts/                    ← CLI: train, evaluate, explore, export
├── dataset_export/                 ← HF/Parquet/ZIP paketleyici (CLI)
└── legacy_docs/                    ← Eski README'ler (referans)
```

---

## Önemli notlar

- **secrets.txt** asla commit edilmez (`.gitignore`'da). Parent klasörde tut.
- **data/** asla commit edilmez. Dışarıyla paylaşmak istersen ayrı yöntem
  kullan (örn. zip + drive).
- **violation_pool.sqlite** veri klasöründe — SQLite read-only erişim
  Dataset Görüntüleyici'de mode=ro URI ile açılır.
- `ifcopenshell` geometry kernel'i (OpenCascade) yoksa 3D paneller hata
  verir; browser/graph/etiket çalışmaya devam eder. Kurulum:
  `conda install -c conda-forge ifcopenshell`.
- **Torch / PyG** GPU için kendi CUDA wheel'ini önce kur, sonra
  `requirements.txt`'i uygula. CPU için doğrudan kurulur.

---

## Hızlı CLI'lar (Streamlit dışı)

```bash
# Eğitim (terminalden, uzun run'lar için)
python -m ml.scripts.train --epochs 50

# Tahmin export (viewer ile uyumlu JSON)
python -m ml.scripts.export_predictions --run-dir ~/Desktop/.../data/runs/<id>

# Dataset paketle
python dataset_export/prepare_dataset.py --out dataset.zip
```
