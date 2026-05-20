# ifc_graph_analysis

Graph Attention Network (GAT) baseline for **TS 9111 / TS ISO 21542
accessibility violation detection** on IFC building models.

Trained on the dataset produced by
[`doganalci/codex1`](https://github.com/doganalci/codex1): parametric
IFC baselines with injected violations and decoy (honeypot) labels.

## Task

**Node-level binary classification** — for each IFC entity (door, wall,
slab, …) predict whether it has been modified by an accessibility
violation injection.

Ground truth comes from `<violated_ifc>.labels.json`:

| label                                              | y |
|----------------------------------------------------|---|
| `status='applied' AND is_decoy=False`              | 1 |
| everything else (incl. decoys, baselines)          | 0 |

Decoy nodes carry a *fake* violation label inside the dataset but the
underlying IFC is unmodified — the model must learn to ignore them.
We track this with **`decoy_fpr`** (rate at which decoys are predicted
positive) as a first-class metric alongside precision/recall/F1.

## Layout

```
app.py                 Streamlit entry — `streamlit run app.py`
app/state.py           sidebar + cached dataset accessors shared by pages
pages/
  01_Model_Görüntüleyici.py   IFC 3D + draggable graph, click-to-cross-highlight
  02_Statik_Analiz.py         shortest / widest / accessible room→room paths
  03_GAT_Tespiti.py           run a trained checkpoint, inspect predictions

viz/
  ifc3d.py             plotly Mesh3d builder with per-GUID overlays
  graph_view.py        streamlit-agraph (interactive) + plotly (static) renderers
analysis/
  pathfind.py          room-adjacency nav graph + 3 path variants + exit query

data/
  sqlite_reader.py     read-only access to codex1's violation_pool.sqlite
  graph_loader.py      graph.json + labels.json → (networkx, y, decoy_set)
  features.py          one-hot ifc_type + numeric attrs → node feature matrix
  pyg_dataset.py       torch_geometric.data.InMemoryDataset
  splits.py            train/val/test split, stratified by baseline_id
model/
  gat.py               2-layer GATv2 + sigmoid head
  hetero_gat.py        per-edge-type relational GAT (HeteroData variant)
train/
  config.py            hyperparameter dataclass
  metrics.py           P/R/F1, decoy_fpr, per-category recall
  loop.py              training/eval loop
scripts/
  train.py             python -m scripts.train --dataset-root ../codex1
  evaluate.py
  export_predictions.py  emit viewer-compatible JSON
```

## Quick start

```bash
# 1. Get a CUDA- or CPU-matched torch first.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Then everything else (Streamlit app + IFC viewer + PyG).
pip install -r requirements.txt

# 3a. Train a model.
python -m scripts.train --dataset-root ~/Desktop/codex1 --epochs 50

# 3b. Launch the analysis app (model viewer + path analysis + GAT inference).
IFC_DATASET_ROOT=~/Desktop/codex1 streamlit run app.py
```

## The app

`streamlit run app.py` opens a multipage UI:

* **🧱 Model Görüntüleyici** — seçili IFC modelini 3D ve interaktif graph olarak
  yan yana gösterir. Graph'ta bir node'a tıklarsan IFC'de aynı eleman
  parlar; düğümleri sürükleyerek graph'ı yeniden düzenleyebilirsin.
* **🧭 Statik Analiz** — iki oda seç, üç farklı strateji ile yol bul:
  en kısa (hop sayısı), en geniş (max-of-min kapı genişliği), en
  erişilebilir (dar kapıları ağır cezalandırılmış maliyet). Üç yol da
  hem graph hem IFC üzerinde farklı renklerle çizilir. Ayrıca bir
  odadan en yakın **dış çıkışa** giden yolu da hesaplar.
* **🤖 GAT Tespiti** — eğitilmiş bir checkpoint seç (`runs/<id>/best.pt`),
  modelin işaretlediği ihlal node'larını gerçek etiketlerle yan yana
  gör (TP / FP / FN / decoy-aldanma dökümlü).

## Data assumptions

The dataset root must look like:

```
<root>/
  violation_pool.sqlite
  ifc_models/
    baseline/<id>.graph.json
    violated/<id>.graph.json
    violated/<id>.labels.json
```

`graph.json` is `nx.node_link_data(g, edges="links")` over a
`MultiDiGraph`; node ids are IFC `GlobalId`s that line up 1:1 with
`labels.json[*].ifc_global_id`. See `codex1/violation_pool/ifc_graph.py`
for the schema and `codex1/violation_pool/ifc_inject.py` for label
generation (including decoy injection).

## Status

MVP scaffold + analysis app. Data path is end-to-end, the homogeneous
GAT trains with edge-type as an integer feature, and the Streamlit
multipage app supports inspecting models, computing accessible routes
between rooms, and reviewing GAT predictions on both the graph and the
IFC geometry.
