"""Tab: model eğitimi — tek tıkla, formdaki hiperparam'larla.

Akış:
  1. Sidebar'dan codex1 dataset root seçilir (her sayfada ortak).
  2. Dataset filtresi: hangi pool_run_id(ler)den gelen violated IFC'leri
     kullanacağız + opsiyonel olarak hangi baselinelar dahil edilsin.
  3. Split: train/val/test yüzdeleri (baseline_id'ye göre stratify).
  4. Hiperparametre formu (model, hidden_dim, heads, dropout, lr, ...).
  5. 'Eğitimi başlat' butonu → in-process eğitim; her epoch sonunda
     progress bar + loss/F1/decoy_fpr canlı güncellenir.
  6. Bitince summary + best.pt kaydedilen run_dir gösterilir; doğrudan
     test sayfasına yönlendirme linki çıkar.

Eğitim Streamlit script'inin içinde senkron koşar. Küçük graflarda
saniyeler sürer; uzun run'lar için CLI scripts/train.py de var.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import (
    folder_browser, get_dataset_root, list_entries, reader_for, set_dataset_root,
)
from ml.data.splits import SplitIndices, split_by_baseline
from ml.train.config import TrainConfig


st.set_page_config(page_title="GAT Eğitim", layout="wide", page_icon="🏋️")
st.title("🏋️ GAT Eğitim")

# GPU durum göstergesi
def _gpu_status():
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"🟢 GPU aktif: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        ver = getattr(torch, "__version__", "?")
        cpu_only = "+cpu" in ver or torch.version.cuda is None
        return False, (f"🔴 GPU YOK — CPU build (torch {ver}). "
                       "NVIDIA kartın varsa CUDA torch kur (aşağıda komut).")
    except Exception as e:
        return False, f"⚠️ torch yüklü değil: {e}"

_gpu_ok, _gpu_msg = _gpu_status()
if _gpu_ok:
    st.success(_gpu_msg)
else:
    st.warning(_gpu_msg)
    with st.expander("⚙️ GPU'yu nasıl aktif ederim? (NVIDIA)"):
        st.code(
            "# Mevcut CPU torch'u kaldır, CUDA build kur (CUDA 12.1 örneği):\n"
            "pip uninstall -y torch torch_geometric\n"
            "pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "pip install torch_geometric\n"
            "# Kontrol:\n"
            'python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"',
            language="powershell",
        )
        st.caption("CUDA sürümünü `nvidia-smi` ile öğren; cu118/cu121/cu124 "
                   "uygun olanı seç. Kurulumdan sonra Streamlit'i yeniden başlat.")

st.caption(
    "Hiperparametreleri ayarla, dataset filtresini seç, **Eğitimi başlat**'a bas. "
    "Eğitim bu sayfada koşar; her epoch sonunda canlı metrikler güncellenir."
)

# ---- Dataset root -----------------------------------------------------------
st.sidebar.title("IFC Graph Analysis")
st.sidebar.caption("Eğitim için aynı sidebar — dataset root paylaşımlı.")
current_root = get_dataset_root()
root = st.sidebar.text_input("Dataset klasörü", value=current_root)
if root != current_root:
    set_dataset_root(root)
with st.sidebar.expander("📁 Klasör seç (gez)"):
    picked = folder_browser(start=root, key="train_browser")
    if picked:
        from ml.app.state import set_dataset_root
        set_dataset_root(picked)
        st.rerun()

if not Path(root).expanduser().exists():
    st.error(f"Klasör yok: {root}")
    st.stop()
if not (Path(root).expanduser() / "violation_pool.sqlite").exists():
    st.error("violation_pool.sqlite bulunamadı.")
    st.stop()

# ---- Dataset filtering ------------------------------------------------------
st.subheader("1. Dataset seçimi")

# A) Dataset tag bazlı filtre (yeni: Sentetik Üretim ile gelen paketler)
from violation_pool import storage as _storage
all_tags = _storage.list_dataset_tags()  # [{tag, baseline, violated, total}, ...]

if all_tags:
    st.caption("**📦 Dataset etiketleri** — Sentetik Üretim ile veya pipeline'ın "
                "verdiği etiketlere göre paket seçimi. Çoklu seçilebilir.")

    # Kayıtlı veri seti yükleyici (sayfa 23 ile entegrasyon)
    try:
        from services import datasets as _ds
        _saved = _ds.list_datasets()
    except Exception:
        _saved = []
    if _saved:
        _saved_map = {d["slug"]: d for d in _saved}
        _opts = ["— (manuel seç)"] + list(_saved_map.keys())
        _pick = st.selectbox(
            "📚 Kayıtlı veri seti (opsiyonel)",
            options=_opts,
            format_func=lambda s: (
                "— (manuel seç)" if s.startswith("—")
                else f"{_saved_map[s]['name']} · "
                     f"{len(_saved_map[s].get('tags') or [])} paket"
            ),
            help="Sayfa 23'te oluşturulan kayıtlı veri setini seç → "
                 "paketler aşağıdaki multiselect'e otomatik dolar.",
        )
        if _pick != "— (manuel seç)":
            _ds_tags = _saved_map[_pick].get("tags") or []
            # Mevcut DB'de olmayan tag'leri sessizce at
            _ds_tags = [t for t in _ds_tags
                        if t in {x["tag"] for x in all_tags}]
            st.session_state["_chosen_tags_default"] = _ds_tags
            st.caption(
                f"✓ `{_pick}` yüklendi — {len(_ds_tags)} paket "
                "multiselect'e doldu."
            )

    _cols = ["tag", "baseline", "violated", "eğitilebilir", "total"]
    _cols = [c for c in _cols if c in pd.DataFrame(all_tags).columns]
    tags_df = pd.DataFrame(all_tags)[_cols]
    st.caption("💡 **eğitilebilir** = status ok/partial + graph'lı violated. "
               "Bu sütun 0 olan paketleri seçme (eski başarısız enjeksiyon).")
    tag_cols = st.columns([3, 2])
    with tag_cols[0]:
        st.dataframe(tags_df, hide_index=True, use_container_width=True)
    with tag_cols[1]:
        _default_tags = st.session_state.get(
            "_chosen_tags_default",
            [t["tag"] for t in all_tags],
        )
        chosen_tags = st.multiselect(
            "Dataset paket(ler)i",
            options=[t["tag"] for t in all_tags],
            default=_default_tags,
            help="Hepsini seçili bırakırsan tüm verilerle eğitir. "
                 "'Kayıtlı veri seti' seçtiysen oradan otomatik dolar.",
        )
        if not chosen_tags:
            st.error("En az bir dataset seç.")
            st.stop()
else:
    chosen_tags = None  # tag yoksa filtre uygulama
    st.caption("(Henüz dataset etiketi yok — tüm violated IFC'ler kullanılacak.)")

# B) Pool filtresi — SADECE codex1 LLM havuzları için, OPT-IN.
# Default boş = filtre yok (basic_inject violated'ları pool_run_id=None
# olduğu için tüm-pool-seçili default'u onları yanlışlıkla eliyordu).
pools = reader_for(root).list_pool_runs()
chosen_pools: list[str] | None = None
if pools:
    with st.expander("🔧 Pool_run filtresi (opsiyonel — codex1 LLM havuzu, "
                     "boş bırak = TÜM violated dahil)"):
        st.caption(
            "⚠️ Bu filtre sadece codex1 LLM-havuzlu violated'lara uygulanır. "
            "Basic Injection / kuralsal üretilenlerin pool'u yoktur — bu "
            "filtreyi BOŞ bırak ki onlar da dahil olsun. Sadece belirli bir "
            "LLM havuzuyla sınırlamak istersen seç."
        )
        pool_df = pd.DataFrame(pools)[["id", "name", "n_violated", "method",
                                        "llm_model", "created_at"]]
        pool_df = pool_df.rename(columns={"id": "pool_id", "n_violated": "#IFC"})
        st.dataframe(pool_df, hide_index=True, use_container_width=True)
        picked = st.multiselect(
            "Pool_run(lar) seç — BOŞ = filtre yok (önerilen)",
            options=[p["id"] for p in pools],
            default=[],   # boş default → filtre uygulanmaz
            format_func=lambda i: f"{next(p['name'] for p in pools if p['id']==i)} ({i[:8]})",
        )
        chosen_pools = picked or None

# Filtreleri uygula
all_violated = list_entries(root, kind="violated")
filtered = all_violated
if chosen_tags is not None:
    allowed = set(_storage.ifc_ids_for_tags(chosen_tags, kind="violated"))
    filtered = [e for e in filtered if e["id"] in allowed]
if chosen_pools is not None:
    filtered = [e for e in filtered if e.get("pool_run_id") in chosen_pools]
with_graph = [e for e in filtered if e.get("graph_ok")]
violated_entries = with_graph

diag = st.columns(3)
diag[0].metric("Toplam violated", len(all_violated))
diag[1].metric("Filtre sonrası", len(filtered))
diag[2].metric("Graph'lı (eğitilebilir)", len(violated_entries))

# Teşhis: filtre 0 verdiğinde nedenini sayıyla göster
if chosen_tags is not None and (not filtered or not violated_entries):
    with st.expander("🔬 Teşhis — neden boş?", expanded=True):
        st.caption(f"Seçili etiket(ler): {chosen_tags}")
        # ifc_ids_for_tags ne döndürdü?
        st.write(f"• `ifc_ids_for_tags` bu etiket(ler) için "
                 f"**{len(allowed)}** violated id döndürdü.")
        # list_entries içinde bu id'ler var mı?
        in_le = [e for e in all_violated if e["id"] in allowed]
        # DB tanılama: status + parent-bazlı violated sayısı
        try:
            import sqlite3 as _sq
            _db = str(Path(root).expanduser() / "violation_pool.sqlite")
            _c = _sq.connect(f"file:{_db}?mode=ro", uri=True)
            # 1) Seçili tag'lerin baseline'larından PARENT olan violated (tag'den bağımsız)
            _tph = ",".join("?" * len(chosen_tags))
            _pcount = _c.execute(
                f"""SELECT child.status, COUNT(*) FROM ifc_models child
                    JOIN ifc_models parent ON child.parent_id = parent.id
                    WHERE child.kind='violated' AND parent.dataset_tag IN ({_tph})
                    GROUP BY child.status""",
                list(chosen_tags),
            ).fetchall()
            _pstat = {r[0]: r[1] for r in _pcount}
            st.write(f"• Seçili paketin baseline'larından üretilmiş violated "
                     f"(parent üzerinden, status dağılımı): **{_pstat}**")
            # 2) allowed id'lerin status'ü (varsa)
            if allowed:
                _ph = ",".join("?" * len(allowed))
                _rows = _c.execute(
                    f"SELECT status, COUNT(*) FROM ifc_models WHERE id IN ({_ph}) "
                    "GROUP BY status", list(allowed)
                ).fetchall()
                st.write(f"• allowed id status: **{ {r[0]: r[1] for r in _rows} }**")
            _c.close()
            # Yorum
            _total_p = sum(_pstat.values())
            if _total_p > 0 and not allowed:
                st.error(
                    f"🔴 Bu baseline'lardan {_total_p} violated üretilmiş AMA "
                    "ifc_ids_for_tags onları bulamıyor → violated'ların kendi "
                    "dataset_tag'ı boş VE parent JOIN'i tutmuyor olabilir. "
                    "Düzeltme: aşağıdaki 'tag onar' komutu."
                )
            elif _total_p == 0:
                st.error(
                    "🔴 Bu paketin baseline'larından HİÇ violated üretilmemiş. "
                    "Tam Etiketleme/Basic Injection bu pakette çalışmamış ya da "
                    "hata almış. Sentetik Üretim sonrası Tam Etiketleme'yi bu "
                    "paket seçili olarak çalıştır."
                )
        except Exception as _e:
            st.caption(f"(DB tanılama atlandı: {_e})")
        st.write(f"• Bunların **{len(in_le)}** tanesi list_entries'te (status ok/partial).")
        with_g = [e for e in in_le if e.get("graph_ok")]
        st.write(f"• Bunların **{len(with_g)}** tanesinin graph.json dosyası mevcut.")
        # Graph'ı eksik olanlardan örnek
        no_g = [e for e in in_le if not e.get("graph_ok")]
        if no_g:
            st.warning(
                f"⚠️ {len(no_g)} violated IFC'nin graph.json'ı YOK/bulunamıyor → "
                "eğitilebilir değil. Basic Injection sırasında graph üretimi "
                "başarısız olmuş olabilir."
            )
            for e in no_g[:5]:
                st.caption(f"   • {e['name']} · {e['id'][:8]} · "
                           f"graph_path={e.get('graph_path')}")
        # list_entries'te HİÇ yoksa: tag eşleşmesi sorunu
        if allowed and not in_le:
            st.error(
                "🔴 ifc_ids_for_tags id döndürdü ama bunlar list_entries'te yok. "
                "Muhtemelen status != ok/partial. DB'de bu violated'ların "
                "status'ünü kontrol et."
            )
        if not allowed:
            st.error(
                "🔴 ifc_ids_for_tags bu etiket için HİÇ violated bulamadı. "
                "Violated IFC'lerin dataset_tag'ı (veya parent baseline'ın tag'ı) "
                "seçtiğinle eşleşmiyor. Tablodaki sayı parent üzerinden geliyor "
                "ama violated kaydının kendi tag'ı farklı olabilir."
            )

if not violated_entries:
    # Help the user figure out *why* the filter is empty.
    if not all_violated:
        st.error(
            "Bu dataset'te hiç violated IFC yok. codex1'in pipeline'ında "
            "**'Pool'dan violated üret'** adımını çalıştırman gerek."
        )
    elif not filtered:
        st.error(
            "Seçili dataset/pool filtreleri sonucu hiç violated IFC kalmadı. "
            "Üstteki dataset paketlerinden farklı bir kombinasyon seç."
        )
    else:
        # filtered var ama hiçbirinin graph'ı yok.
        without_graph = [e for e in filtered if not e.get("graph_ok")]
        st.error(
            f"Seçili filtrede {len(filtered)} violated IFC var ama hiçbirinin "
            f"`graph.json` dosyası yok. codex1'de IFC üretildikten sonra "
            f"graph generation adımının çalıştığından emin ol. "
            f"İlk birkaç eksik: {[e['id'][:8] for e in without_graph[:3]]}"
        )
    st.stop()

st.caption("**Augmentation & Feature Maskeleme**")
aug_cols = st.columns(2)
with aug_cols[0]:
    include_baselines = st.checkbox(
        "Baseline'ları da dahil et",
        value=False,
        help="Açarsan ihlal içermeyen modeller de eğitim sinyaline katılır.",
    )
    use_rule_oracle = st.checkbox(
        "📏 Kural-tabanlı oracle augment",
        value=True,
        help=(
            "Eğitim öncesi geometrik kurallarla (kapı <90 cm, korkuluk <90 cm) "
            "etiketsiz ihlalleri pozitif olarak ekler. Closed-world supervision "
            "sorununu çözer."
        ),
    )
with aug_cols[1]:
    st.caption("🔬 **Feature leak teşhisi** — gerçek F1'i görmek için maskele")
    mask_numeric_features = st.checkbox(
        "Sayısal attribute'ları gizle (OverallWidth/Height/...)",
        value=True,
        help="En güçlü leak: bu değerler injection/oracle eşiği olduğu için "
             "feature olarak görünce model etiketi okur.",
    )
    mask_pset_features = st.checkbox(
        "Pset bayraklarını gizle (DoorCommon, IsExternal, FireRating)",
        value=False,
        help="Bazı enjeksiyon tipleri Pset değiştirebilir.",
    )
    mask_type_features = st.checkbox(
        "IFC tipi one-hot gizle (Door/Stair/Column...)",
        value=False,
        help="add_obstruction yeni IfcColumn ekler — model 'yeni Column = "
             "ihlal' diye trivial pattern bulabilir, bu seçenekle önle.",
    )

# Cache yönetimi — kullanıcı doğru cache'i temizlediğine emin olsun
from pathlib import Path as _Path
import datetime as _dt
_cache_dir = _Path("./data/cache").expanduser().resolve()
with st.expander("🗑 Cache yönetimi (mevcut: " + str(_cache_dir) + ")"):
    if _cache_dir.exists():
        cache_files = sorted(_cache_dir.glob("**/*.pt"))
        if cache_files:
            for cf in cache_files:
                mt = _dt.datetime.fromtimestamp(cf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                st.caption(f"  📦 `{cf.name}` ({cf.stat().st_size/1024:.1f} KB · {mt})")
        else:
            st.caption("  (cache boş)")
        if st.button("🗑 Tüm cache'i sil",
                     help="Sonraki eğitimde dataset baştan işlenir — feature "
                          "değişikliği yaptıysan ŞART"):
            import shutil as _sh
            _sh.rmtree(_cache_dir, ignore_errors=True)
            st.success("Cache silindi.")
            st.rerun()
    else:
        st.caption("  (cache klasörü yok — ilk eğitimde oluşacak)")

with st.expander("📖 Oracle kural listesi"):
    from ml.data.rule_oracle import summarize_rules
    st.dataframe(pd.DataFrame(summarize_rules()),
                 hide_index=True, use_container_width=True)
    st.caption(
        "Eşikler TS 9111 / ADA referansları üzerinden kabaca türetilmiştir. "
        "Yeni kural eklemek için: `ml/data/rule_oracle.py` → `RULES` tuple'ına "
        "ekle (cache invalidate olur)."
    )

_mask_summary = []
if mask_numeric_features: _mask_summary.append("numeric")
if mask_pset_features: _mask_summary.append("psets")
if mask_type_features: _mask_summary.append("type")
st.success(
    f"✅ {len(violated_entries)} violated IFC seçildi"
    + (" (+ baseline'lar)" if include_baselines else "")
    + (" + oracle augment" if use_rule_oracle else "")
    + (f" · maskeli: {','.join(_mask_summary)}" if _mask_summary else "")
)

# ---- Split control ----------------------------------------------------------
st.subheader("2. Train / Val / Test bölünmesi")

split_cols = st.columns([2, 2, 2, 2])
with split_cols[0]:
    train_pct = st.slider("Train %", 30, 90, 70, 5)
with split_cols[1]:
    val_pct = st.slider("Val %", 0, 40, 15, 5)
with split_cols[2]:
    test_pct = max(0, 100 - train_pct - val_pct)
    st.metric("Test %", test_pct)
with split_cols[3]:
    split_seed = st.number_input("Split seed", 0, 9999, 0, 1)

if train_pct + val_pct > 100:
    st.error("Train + Val > 100% — geçersiz.")
    st.stop()

# Preview the split.
ids = [e["id"] for e in violated_entries]
baseline_ids = [e.get("parent_id") for e in violated_entries]
splits_preview = split_by_baseline(
    baseline_ids,
    val_frac=val_pct / 100.0,
    test_frac=test_pct / 100.0,
    seed=int(split_seed),
)

prev = st.columns(3)
prev[0].metric("Train IFC", len(splits_preview.train))
prev[1].metric("Val IFC", len(splits_preview.val))
prev[2].metric("Test IFC", len(splits_preview.test))

with st.expander("📋 Hangi IFC hangi split'te?", expanded=False):
    def _name_of(i: int) -> str:
        return f"{violated_entries[i]['name']}  ·  {violated_entries[i]['id'][:8]}"
    st.markdown("**Train:**")
    st.write("\n".join("• " + _name_of(i) for i in splits_preview.train) or "—")
    st.markdown("**Val:**")
    st.write("\n".join("• " + _name_of(i) for i in splits_preview.val) or "—")
    st.markdown("**Test:**")
    st.write("\n".join("• " + _name_of(i) for i in splits_preview.test) or "—")

# ---- Hyperparams ------------------------------------------------------------
st.subheader("3. Model & eğitim hiperparametreleri")

hp_a, hp_b, hp_c = st.columns(3)
with hp_a:
    model_type = st.selectbox("Model", options=["gat", "hetero_gat"], index=0,
                              help="hetero_gat = her edge tipine ayrı conv.")
    hidden_dim = st.select_slider("hidden_dim", options=[16, 32, 64, 128, 256], value=64)
    heads = st.slider("Attention heads", 1, 8, 4)
    edge_emb_dim = st.slider("Edge embedding dim", 4, 32, 8, 4)
    dropout = st.slider("Dropout", 0.0, 0.7, 0.3, 0.05)
with hp_b:
    epochs = st.number_input("Epoch sayısı", 1, 1000, 50)
    lr = st.select_slider("Learning rate",
                          options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                          value=5e-3)
    weight_decay = st.select_slider("Weight decay",
                                    options=[0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                    value=5e-4)
    patience = st.number_input("Early-stop patience", 1, 100, 10)
    threshold = st.slider("Karar eşiği", 0.1, 0.9, 0.5, 0.05)
with hp_c:
    pw_mode = st.radio("pos_weight", ["Otomatik (sınıf dengesinden)", "Manuel"],
                       index=0)
    pos_weight = (None if pw_mode.startswith("Otomatik")
                  else st.number_input("pos_weight değeri", 0.1, 100.0, 5.0, 0.5))
    device = st.selectbox("Cihaz", options=["auto", "cpu", "cuda"], index=0)
    seed = st.number_input("Seed", 0, 99999, 42)
    run_name = st.text_input("Run adı (boş = otomatik)",
                             value=f"ui_{time.strftime('%Y%m%d_%H%M%S')}")

# ---- Launch -----------------------------------------------------------------
st.subheader("4. Eğitimi başlat")

cfg = TrainConfig(
    dataset_root=root,
    cache_root="./data/cache",
    include_baselines=include_baselines,
    use_rule_oracle=use_rule_oracle,
    mask_numeric_features=mask_numeric_features,
    mask_pset_features=mask_pset_features,
    mask_type_features=mask_type_features,
    model=model_type,
    hidden_dim=int(hidden_dim),
    heads=int(heads),
    dropout=float(dropout),
    edge_emb_dim=int(edge_emb_dim),
    epochs=int(epochs),
    lr=float(lr),
    weight_decay=float(weight_decay),
    pos_weight=pos_weight,
    patience=int(patience),
    threshold=float(threshold),
    val_frac=val_pct / 100.0,
    test_frac=test_pct / 100.0,
    split_seed=int(split_seed),
    device=device,
    seed=int(seed),
    run_dir="./runs",
    run_name=run_name or None,
)

start = st.button("🚀 Eğitimi başlat", type="primary")

if start:
    # In-place imports so torch is only required when actually training.
    try:
        from ml.train.loop import run_training
        from ml.train.metrics import EvalResult  # noqa: F401
    except ImportError as e:
        st.error(f"PyTorch/PyG yüklü değil: {e}")
        st.stop()

    progress_slot = st.empty()
    metric_slot = st.empty()
    chart_slot = st.empty()
    # Açılabilir terminal — eğitim sırasındaki tüm stdout/stderr buraya yazılır.
    log_slot = st.expander("🖥️ Terminal (canlı)", expanded=True)
    log_buffer: list[str] = []
    log_show_lines = 400  # son N satırı göster (UI bunaltmasın)

    history_rows: list[dict] = []

    total_epochs = cfg.epochs

    def on_setup(info: dict):
        progress_slot.info(
            f"device={info['device']}  ·  feature_dim={info['feature_dim']}  ·  "
            f"split={info['splits']}  ·  pos_weight={info['pos_weight']:.2f}"
        )

    def on_epoch(epoch: int, train_loss: float, val_res):
        history_rows.append({
            "epoch": epoch,
            "loss": train_loss,
            "f1": val_res.f1,
            "precision": val_res.precision,
            "recall": val_res.recall,
            "balanced_acc": val_res.balanced_accuracy,
            "mcc": val_res.mcc,
            "auc_roc": val_res.auc_roc,
            "accuracy": val_res.accuracy,
            "decoy_fpr": val_res.decoy_fpr,
        })
        progress_slot.progress(epoch / total_epochs,
                               text=f"epoch {epoch}/{total_epochs}  ·  "
                                    f"loss={train_loss:.4f}  f1={val_res.f1:.3f}")
        with metric_slot.container():
            r1 = st.columns(4)
            r1[0].metric("F1 (val)", f"{val_res.f1:.3f}")
            r1[1].metric("Precision", f"{val_res.precision:.3f}")
            r1[2].metric("Recall", f"{val_res.recall:.3f}")
            r1[3].metric("Accuracy", f"{val_res.accuracy:.3f}")
            r2 = st.columns(4)
            r2[0].metric("Balanced Acc", f"{val_res.balanced_accuracy:.3f}",
                         help="(TPR + TNR) / 2 — sınıf dengesizliğine sağlam.")
            r2[1].metric("MCC", f"{val_res.mcc:+.3f}",
                         help="Matthews correlation. -1..+1. 0 = rastgele.")
            r2[2].metric("AUC-ROC", f"{val_res.auc_roc:.3f}",
                         help="Eşikten bağımsız sıralama gücü.")
            r2[3].metric("Decoy FPR", f"{val_res.decoy_fpr:.3f}",
                         help="Decoy node'ların kaçını yanlışlıkla ihlal saydı.")
        df = pd.DataFrame(history_rows).set_index("epoch")
        with chart_slot.container():
            c1, c2 = st.columns(2)
            c1.caption("Loss & F1 & AUC")
            c1.line_chart(df[["loss", "f1", "auc_roc"]])
            c2.caption("Bal_acc & MCC & Decoy FPR")
            c2.line_chart(df[["balanced_acc", "mcc", "decoy_fpr"]])

    log_placeholder = log_slot.empty()

    def _render_log() -> None:
        text = "\n".join(log_buffer[-log_show_lines:])
        log_placeholder.code(text or "(henüz çıktı yok)", language="text")

    def on_log(msg: str):
        for line in str(msg).rstrip("\n").split("\n"):
            log_buffer.append(line)
        _render_log()

    # stdout/stderr'i de yakala: torch_geometric / chromadb / PyG cache
    # gibi paketler print ile yazar; sade `on_log` sadece loop.py'nin
    # _log çağrılarını alır. Tee ile her ikisini de logla.
    import io, sys as _sys

    class _Tee(io.TextIOBase):
        def __init__(self, original):
            self._orig = original
            self._buf = ""

        def write(self, s):  # type: ignore[override]
            if not s:
                return 0
            try:
                self._orig.write(s)
            except Exception:
                pass
            self._buf += s
            if "\n" in self._buf:
                lines = self._buf.split("\n")
                self._buf = lines[-1]
                for line in lines[:-1]:
                    if line.strip():
                        log_buffer.append(line)
                _render_log()
            return len(s)

        def flush(self):
            try:
                self._orig.flush()
            except Exception:
                pass

    _orig_out, _orig_err = _sys.stdout, _sys.stderr

    with st.spinner("Eğitim çalışıyor..."):
        try:
            _sys.stdout = _Tee(_orig_out)
            _sys.stderr = _Tee(_orig_err)
            try:
                summary = run_training(
                    cfg,
                    on_setup=on_setup,
                    on_epoch_end=on_epoch,
                    on_log=on_log,
                    filter_ifc_ids=[e["id"] for e in violated_entries],
                )
            finally:
                _sys.stdout, _sys.stderr = _orig_out, _orig_err
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("✅ Eğitim tamamlandı.")
    st.write(f"En iyi val F1: **{summary['best_val_f1']:.3f}** @ epoch {summary['best_epoch']}")

    # Süreler
    _tim = summary.get("timing") or {}
    if _tim:
        st.markdown("**⏱ Süreler**")
        tcol = st.columns(4)
        tcol[0].metric("Toplam eğitim", f"{_tim.get('total_train_seconds', 0):.1f} s",
                       f"{_tim.get('epochs_run', 0)} epoch")
        tcol[1].metric("Epoch ortalama", f"{_tim.get('avg_epoch_seconds', 0):.2f} s")
        tcol[2].metric("Örnek başına (eğitim)", f"{_tim.get('per_sample_train_ms', 0):.1f} ms",
                       help="Bir epoch'ta IFC başına ortalama")
        tcol[3].metric("Test değerlendirme", f"{_tim.get('test_eval_seconds', 0):.2f} s",
                       f"{_tim.get('per_sample_test_ms', 0):.1f} ms/IFC")

    def _render_split_metrics(t: dict | None, split_label: str):
        if not t:
            st.info(f"{split_label} split boş ya da hesaplanmadı.")
            return
        r1 = st.columns(4)
        r1[0].metric("F1", f"{t['f1']:.3f}")
        r1[1].metric("Precision", f"{t['precision']:.3f}")
        r1[2].metric("Recall", f"{t['recall']:.3f}")
        r1[3].metric("Accuracy", f"{t['accuracy']:.3f}")
        r2 = st.columns(4)
        r2[0].metric("Balanced Acc", f"{t.get('balanced_accuracy', 0):.3f}")
        r2[1].metric("MCC", f"{t.get('mcc', 0):+.3f}")
        r2[2].metric("AUC-ROC", f"{t.get('auc_roc', 0):.3f}")
        r2[3].metric("Decoy FPR", f"{t['decoy_fpr']:.3f}")

        cm = t.get("confusion", {})
        if cm:
            st.markdown("**Confusion Matrix**")
            cm_df = pd.DataFrame(
                [[cm.get("tn", 0), cm.get("fp", 0)],
                 [cm.get("fn", 0), cm.get("tp", 0)]],
                index=["Gerçek: değil", "Gerçek: ihlal"],
                columns=["Tahmin: değil", "Tahmin: ihlal"],
            )
            cc1, cc2 = st.columns([1, 2])
            cc1.dataframe(cm_df, use_container_width=True)
            cc2.caption(
                f"**TP={cm.get('tp', 0)}** doğru yakalanan  ·  "
                f"**FN={cm.get('fn', 0)}** kaçırılan ihlal  ·  "
                f"**FP={cm.get('fp', 0)}** yanlış alarm  ·  "
                f"**TN={cm.get('tn', 0)}** doğru reddedilen normal"
            )

        per_p = t.get("per_category_precision", {})
        per_r = t.get("per_category_recall", {})
        per_f = t.get("per_category_f1", {})
        per_s = t.get("per_category_support", {})
        if per_r:
            st.markdown("**Kategori Bazında P / R / F1**")
            cat_rows = []
            for c in sorted(set(per_r) | set(per_p)):
                sup = int(per_s.get(c, 0))
                rec = per_r.get(c, 0.0)
                tp_c = round(rec * sup)
                fn_c = sup - tp_c
                cat_rows.append({
                    "kategori": c,
                    "precision": per_p.get(c, 0.0),
                    "recall": rec,
                    "f1": per_f.get(c, 0.0),
                    "TP": tp_c,
                    "FN": fn_c,
                    "support": sup,
                })
            cat_df = pd.DataFrame(cat_rows).sort_values("f1")
            st.dataframe(cat_df, hide_index=True, use_container_width=True)
            st.markdown("**TP / FN dağılımı**")
            st.bar_chart(cat_df[["kategori", "TP", "FN"]].set_index("kategori"),
                         color=["#22c55e", "#ef4444"])
            weak = cat_df[cat_df["f1"] < 0.7]
            if not weak.empty:
                with st.expander(f"⚠️ Zayıf kategoriler ({len(weak)})", expanded=False):
                    for _, row in weak.iterrows():
                        st.warning(
                            f"**{row['kategori']}** — F1={row['f1']:.2f} "
                            f"({row['TP']}/{row['support']} yakalandı, "
                            f"{row['FN']} kaçırıldı)"
                        )

    st.subheader("📊 Sonuçlar (en iyi model — best.pt)")
    # Train / Val / Test farkını anla — overfit / generalization sinyali
    def _f1(d): return d.get("f1", 0) if d else 0
    t_f1, v_f1, te_f1 = _f1(summary.get("train")), _f1(summary.get("val")), _f1(summary.get("test"))
    overview = st.columns(4)
    overview[0].metric("Train F1", f"{t_f1:.3f}")
    overview[1].metric("Val F1", f"{v_f1:.3f}",
                        delta=f"{v_f1 - t_f1:+.3f}",
                        delta_color="inverse",
                        help="Train'den çok düşükse overfit, çok yüksekse şüphe.")
    overview[2].metric("Test F1", f"{te_f1:.3f}",
                        delta=f"{te_f1 - v_f1:+.3f}",
                        delta_color="inverse",
                        help="Val'den ciddi farklıysa validation set güvenilir değil.")
    gap = t_f1 - te_f1
    overview[3].metric("Train - Test gap", f"{gap:+.3f}",
                        help="Generalization gap. >0.2 ise muhtemelen overfitting.")

    tab_tr, tab_va, tab_te = st.tabs([
        f"🟦 Train ({t_f1:.3f})", f"🟨 Val ({v_f1:.3f})", f"🟥 Test ({te_f1:.3f})"
    ])
    with tab_tr:
        _render_split_metrics(summary.get("train"), "Train")
    with tab_va:
        _render_split_metrics(summary.get("val"), "Val")
    with tab_te:
        _render_split_metrics(summary.get("test"), "Test")

    st.code(summary["run_dir"])
    st.caption(
        "🔍 Daha detaylı analiz için sol menüden **GAT Test** sayfasına geç, bu run'ı seç. "
        "CLI'de `python ml/scripts/error_analysis.py` ve `sanity_check.py` daha kapsamlı rapor üretir."
    )
