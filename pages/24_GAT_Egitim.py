"""Sayfa 24 — GAT Eğitim.

Sayfa 23'te oluşturulan veri seti seçilir → train/val/test split
ayarlanır → GAT veya HeteroGAT eğitilir → tüm run metadata (dataset,
split, hiperparametreler, metrikler, timestamp) `runs/<name>/` altına
yazılır. Sayfa 15 viewer aynı run'ları seçip tahmin yapar.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from ml.app.state import get_dataset_root
from ml.train.config import TrainConfig
from ml.train.loop import run_training
from services import datasets as ds
from violation_pool import storage


st.set_page_config(page_title="GAT Eğitim", layout="wide", page_icon="🧠")
st.title("🧠 GAT Eğitim")
st.caption(
    "Veri seti seç → split + hiperparametre ayarla → eğit. "
    "Tüm run metadata `runs/<name>/` altında saklanır; "
    "Sayfa 15 viewer aynı modeli tahmin için kullanır."
)


# --- 1. Veri seti --------------------------------------------------------
st.subheader("1. Veri seti")

@st.cache_data(ttl=10, show_spinner=False)
def _all_tags() -> list[dict]:
    return storage.list_dataset_tags()


all_tags = _all_tags()
tag_index = {t["tag"]: t for t in all_tags}
if not all_tags:
    st.error("Henüz hiç paket yok. Önce baseline + ihlal üret (sayfa 10, 21).")
    st.stop()

saved = ds.list_datasets()
ds_opts = ["— (manuel paket seç)"] + [d["slug"] for d in saved]
ds_map = {d["slug"]: d for d in saved}

dc1, dc2 = st.columns([2, 3])
with dc1:
    pick = st.selectbox(
        "📚 Kayıtlı veri seti",
        options=ds_opts,
        format_func=lambda s: (
            "— (manuel paket seç)" if s.startswith("—")
            else f"{ds_map[s]['name']} · "
                 f"{len(ds_map[s].get('tags') or [])} paket"
        ),
        help="Sayfa 23'te oluşturulan veri setlerinden seç → paketler "
             "otomatik dolar. Yoksa aşağıdan manuel seç.",
    )

with dc2:
    if pick.startswith("—"):
        _default_tags = [t["tag"] for t in all_tags]
        chosen_tags = st.multiselect(
            "📦 Paketler",
            options=[t["tag"] for t in all_tags],
            default=_default_tags,
            format_func=lambda t: (
                f"{t} · base={tag_index[t].get('baseline', 0)} · "
                f"vio={tag_index[t].get('violated', 0)}"
            ),
        )
        chosen_dataset_slug: str | None = None
        chosen_dataset_name: str | None = None
    else:
        chosen_dataset_slug = pick
        chosen_dataset_name = ds_map[pick]["name"]
        chosen_tags = [t for t in (ds_map[pick].get("tags") or [])
                       if t in tag_index]
        st.caption(
            f"✓ **{chosen_dataset_name}** · {len(chosen_tags)} paket "
            "(DB'de mevcut)"
        )
        with st.expander("Paket detayı"):
            for t in chosen_tags:
                info = tag_index[t]
                st.markdown(
                    f"- **`{t}`** · base={info.get('baseline', 0)} · "
                    f"vio={info.get('violated', 0)} · "
                    f"total={info.get('total', 0)}"
                )

if not chosen_tags:
    st.error("En az 1 paket seçilmeli.")
    st.stop()

# Filtre için IFC id'leri çek (sadece violated)
allowed_ids = storage.ifc_ids_for_tags(chosen_tags, kind="violated")
st.caption(
    f"📊 Eğitim havuzu: **{len(allowed_ids)}** violated IFC "
    f"({len(chosen_tags)} paketten)"
)
if len(allowed_ids) < 5:
    st.warning(
        "⚠️ Çok az veri (< 5 IFC). Anlamlı eğitim için en az 20-50 öneririz."
    )


# --- 2. Model -------------------------------------------------------------
st.subheader("2. Model")
mc1, mc2 = st.columns([1, 3])
with mc1:
    model_type = st.radio(
        "Model tipi",
        options=["gat", "hetero_gat"],
        format_func=lambda x: {"gat": "🧠 GAT (homojen)",
                               "hetero_gat": "🧠🧠 Hetero GAT "
                                             "(edge tipine duyarlı)"}[x],
        help="Hetero GAT IFC ilişkilerini (Aggregates, Contains, ...) "
             "ayrı tutar; daha kaliteli ama yavaş.",
    )
with mc2:
    st.caption(
        "**GAT**: tüm edge'ler tek tip · küçük dataset'te yeterli.  \n"
        "**Hetero GAT**: 7 edge tipi (Aggregates, Contains, Voids vb.) "
        "ayrı projeksiyon → daha iyi yapısal öğrenme, ~2× yavaş."
    )


# --- 3. Veri bölme (train/val/test) --------------------------------------
st.subheader("3. Veri bölme (split)")
st.caption(
    "Default %70/%20/%10 — küçük dataset'te val ve test'i %20/%20 yapmak "
    "daha sağlam ölçüm verir."
)
sc1, sc2, sc3 = st.columns(3)
with sc1:
    train_pct = st.slider("Train %", min_value=40, max_value=90,
                           value=70, step=5)
with sc2:
    max_val = 100 - train_pct - 5  # en az %5 test
    val_pct = st.slider("Val %", min_value=5, max_value=max(5, max_val),
                         value=min(20, max_val), step=5)
with sc3:
    test_pct = 100 - train_pct - val_pct
    st.metric("Test %", f"{test_pct}%",
              help="Otomatik = 100 - train - val")

if test_pct < 5:
    st.error("Test % en az 5 olmalı. Train veya Val'ı azalt.")
    st.stop()

_n = len(allowed_ids)
st.caption(
    f"≈ Train {int(_n * train_pct / 100)} · "
    f"Val {int(_n * val_pct / 100)} · "
    f"Test {int(_n * test_pct / 100)}"
)

split_seed = st.number_input("Split seed (tekrar üretilebilirlik)",
                              min_value=0, value=0, step=1)


# --- 4. Hiperparametreler ------------------------------------------------
st.subheader("4. Hiperparametreler")
with st.expander("⚙️ Detaylı ayarlar (default'lar eskisi gibi)",
                 expanded=False):
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        hidden_dim = st.number_input("hidden_dim", 16, 256, 64, step=16)
        heads = st.number_input("attention heads", 1, 16, 4)
        dropout = st.slider("dropout", 0.0, 0.7, 0.30, 0.05)
        edge_emb_dim = st.number_input("edge_emb_dim", 4, 64, 8, step=4,
                                        disabled=(model_type != "hetero_gat"),
                                        help="Sadece Hetero GAT'ta kullanılır.")
    with hc2:
        epochs = st.number_input("epochs", 5, 500, 50, step=5)
        lr = st.select_slider("learning rate",
                              options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3,
                                       5e-3, 1e-2, 3e-2],
                              value=5e-3,
                              format_func=lambda x: f"{x:.0e}")
        weight_decay = st.select_slider("weight_decay",
                                         options=[0.0, 1e-5, 1e-4, 5e-4,
                                                  1e-3, 5e-3],
                                         value=5e-4,
                                         format_func=lambda x: f"{x:.0e}")
        patience = st.number_input("early-stop patience", 0, 100, 10,
                                    help="0 → erken durdurma kapalı")
    with hc3:
        threshold = st.slider("classification threshold", 0.1, 0.9,
                               0.5, 0.05)
        seed = st.number_input("model seed", 0, 9999, 42)
        device = st.selectbox("device", options=["auto", "cuda", "cpu"],
                              index=0)
        include_baselines = st.checkbox(
            "Baseline IFC'leri eğitime kat",
            value=False,
            help="Default kapalı: sadece violated IFC'ler. Açarsan "
                 "baseline (ihlalsiz) IFC'ler de negatif örnek olur.",
        )

with st.expander("🎭 Etiket / özellik mask'leri (opsiyonel)",
                 expanded=False):
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        mask_num = st.checkbox(
            "Numeric features mask",
            help="OverallWidth/Height/NominalHeight/Elevation sıfırla. "
                 "Model bunlara bakmadan öğrensin (leak engelleme).",
        )
    with mc2:
        mask_pset = st.checkbox(
            "Pset/bayrak features mask",
            help="IsExternal/FireRating gibi pset bayraklarını sıfırla.",
        )
    with mc3:
        mask_type = st.checkbox(
            "IFC type one-hot mask",
            help="IfcDoor/IfcColumn one-hot sıfırla. add_obstruction "
                 "ihlali için 'IfcColumn = ihlal' leak'ini engeller.",
        )

# Run adı — default formatta dataset + model + timestamp.
# Kullanıcı istediği gibi değiştirebilir.
_ts = time.strftime("%Y%m%d_%H%M%S")
_data_token = (chosen_dataset_slug or "manual")
default_name = f"{model_type}_{_data_token}_{_ts}"
run_name = st.text_input(
    "Run adı (run klasör adı)",
    value=default_name,
    help="Default: `<model>_<dataset>_<timestamp>`. Sayfa 15 viewer "
         "dropdown'ında bu ad görünür. Eski adla aynı verirsen önceki "
         "run'ın üzerine yazılır.",
)


# --- 5. Eğit -------------------------------------------------------------
st.subheader("5. Eğit")


def _purge_cache(cache_dir: Path) -> tuple[bool, str]:
    """Cache klasörünü agresif sil ve doğrula.

    Windows'ta `shutil.rmtree(ignore_errors=True)` dosya kilitliyse
    sessizce skip eder ve cache hâlâ orada kalır → eğitim eski IFC
    listesini okur. Burada:
    1) Streamlit cache referanslarını temizle (varsa).
    2) gc.collect() ile torch tensor reference'larını serbest bırak.
    3) shutil.rmtree dene.
    4) Hâlâ varsa dosya bazında unlink dene.
    5) Sonuçu doğrula; başarısızsa kullanıcıya net mesaj ver.
    """
    import gc
    import shutil

    if not cache_dir.exists():
        return True, "Cache zaten yok."

    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    gc.collect()

    try:
        shutil.rmtree(cache_dir)
    except Exception:
        pass

    if cache_dir.exists():
        # Dosya bazında dene (rmtree Windows kilidinde takılmış olabilir).
        for p in sorted(cache_dir.rglob("*"), reverse=True):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            except Exception:
                pass
        try:
            cache_dir.rmdir()
        except Exception:
            pass

    # Son kontrol
    if cache_dir.exists():
        remaining = list(cache_dir.rglob("*.pt"))
        if remaining:
            return False, (
                f"⚠️ Cache silinemedi (Windows dosya kilidi olabilir). "
                f"Kalan {len(remaining)} `.pt` dosyası var. **Streamlit'i "
                f"komple kapat**, sonra şu klasörü File Explorer'dan elle sil:"
                f"\n\n`{cache_dir}`\n\nSonra Streamlit'i tekrar başlat."
            )
    return True, "✓ Cache silindi (yeniden başlatma gerekmez)."


# Cache yönetimi — eğitim artık dataset'e özel cache kullanıyor
# (data/cache/<slug>/processed/...), eski global cache rahatsızlık vermez.
# Yine de buradan tüm cache'leri silmek mümkün (disk temizliği için).
_cache_dir = Path("./data/cache").expanduser().resolve()
with st.expander(f"🗑 Cache yönetimi ({_cache_dir})"):
    st.caption(
        "💡 Eğitim artık her dataset için ayrı cache kullanıyor "
        "(`data/cache/<dataset_slug>/`). Yeni paket eklediğinde cache "
        "silmek **gerekmez**; sadece disk dolarsa eski olanları sil."
    )
    if _cache_dir.exists():
        _cf = sorted(_cache_dir.glob("**/*.pt"))
        if _cf:
            import datetime as _dt
            for cf in _cf:
                _mt = _dt.datetime.fromtimestamp(
                    cf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                _sz = cf.stat().st_size / 1024 / 1024
                _rel = cf.relative_to(_cache_dir)
                st.caption(f"📦 `{_rel}` · {_sz:.1f} MB · {_mt}")
            if st.button("🗑 Tüm cache'i sil",
                         help="Tüm dataset'lerin cache'i silinir; sonraki "
                              "eğitimde her biri yeniden üretilir.",
                         key="clear_cache_top"):
                ok, msg = _purge_cache(_cache_dir)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
        else:
            st.caption("(cache boş)")
    else:
        st.caption("(cache klasörü yok — ilk eğitimde oluşur)")

go = st.button("🚀 Eğitimi başlat",
               type="primary", use_container_width=True,
               disabled=(test_pct < 5 or not chosen_tags))

if go:
    # Cache'i dataset'e özel yapıyoruz → 'Filtre hiçbir IFC eşleştirmedi'
    # hatası ortadan kalkar. Her dataset/slug kendi cache klasörüne yazar,
    # global tek cache'e çakışmaz. Eski global cache (data/cache/processed/
    # ifc_violation.pt) dokunulmadan kalır — kullanıcı isterse expander'dan
    # silebilir, ama artık zorunlu değil.
    _cache_token = (chosen_dataset_slug or "manual")
    _per_dataset_cache = str(_cache_dir / _cache_token)

    cfg = TrainConfig(
        # Data
        dataset_root=get_dataset_root(),
        cache_root=_per_dataset_cache,
        include_baselines=bool(include_baselines),
        mask_numeric_features=bool(mask_num),
        mask_pset_features=bool(mask_pset),
        mask_type_features=bool(mask_type),
        val_frac=val_pct / 100.0,
        test_frac=test_pct / 100.0,
        split_seed=int(split_seed),
        # Model
        model=model_type,
        hidden_dim=int(hidden_dim),
        heads=int(heads),
        dropout=float(dropout),
        edge_emb_dim=int(edge_emb_dim),
        # Optim
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        patience=int(patience),
        threshold=float(threshold),
        # Runtime
        device=device,
        seed=int(seed),
        run_name=run_name.strip() or default_name,
    )

    # Progress UI
    bar = st.progress(0.0, text="başlatılıyor...")
    metric_slot = st.empty()
    log_slot = st.empty()
    history: list[dict] = []
    setup_info: dict = {}
    log_buf: list[str] = []

    def _on_setup(info: dict) -> None:
        setup_info.update(info)
        log_buf.append(
            f"[setup] n_total={info.get('n_total')} · "
            f"feat_dim={info.get('feature_dim')} · "
            f"device={info.get('device')} · "
            f"pos_weight={info.get('pos_weight')}"
        )
        sp = info.get("splits") or {}
        log_buf.append(
            f"[split] train={sp.get('train', 0)} · "
            f"val={sp.get('val', 0)} · test={sp.get('test', 0)}"
        )
        log_slot.code("\n".join(log_buf[-20:]), language="text")

    def _on_epoch(ep: int, loss: float, res) -> None:
        history.append({
            "epoch": ep,
            "loss": float(loss),
            "f1": float(getattr(res, "f1", 0)),
            "auc_roc": float(getattr(res, "auc_roc", 0)),
            "precision": float(getattr(res, "precision", 0)),
            "recall": float(getattr(res, "recall", 0)),
        })
        bar.progress(min(1.0, ep / max(epochs, 1)),
                     text=f"epoch {ep}/{epochs} · loss={loss:.4f} · "
                          f"val F1={getattr(res, 'f1', 0):.3f}")
        df = pd.DataFrame(history).set_index("epoch")
        with metric_slot.container():
            cc1, cc2 = st.columns(2)
            cc1.caption("Loss")
            cc1.line_chart(df[["loss"]])
            cc2.caption("Val F1 / AUC / Precision / Recall")
            cc2.line_chart(df[["f1", "auc_roc", "precision", "recall"]])

    def _on_log(msg: str) -> None:
        for line in str(msg).rstrip().split("\n"):
            if line.strip():
                log_buf.append(line)
        log_slot.code("\n".join(log_buf[-20:]), language="text")

    t0 = time.time()
    try:
        with st.spinner("Eğitim çalışıyor..."):
            summary = run_training(
                cfg,
                on_setup=_on_setup,
                on_epoch_end=_on_epoch,
                on_log=_on_log,
                filter_ifc_ids=allowed_ids,
            )
    except RuntimeError as e:
        _msg = str(e)
        if "Filtre hiçbir IFC eşleştirmedi" in _msg or "CACHE BAYAT" in _msg:
            st.error(
                "🔴 Cache bayat — yeni paketin IFC'leri eski cache'te yok. "
                "Aşağıdaki butonla cache'i sil, sonra tekrar başlat."
            )
            if st.button("🗑 Cache'i sil ve hazırla",
                         type="primary",
                         key="clear_cache_inline"):
                ok, msg = _purge_cache(_cache_dir)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
        else:
            st.exception(e)
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()
    bar.empty()
    dt = time.time() - t0

    # Run metadata zenginleştirme — dataset/paket bilgisi config.json
    # yanına ek olarak meta.json'a yazılır.
    run_dir = Path(summary["run_dir"])
    meta = {
        "run_name": cfg.run_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_s": round(dt, 2),
        "dataset": {
            "slug": chosen_dataset_slug,
            "name": chosen_dataset_name,
            "tags": chosen_tags,
            "n_ifc_ids": len(allowed_ids),
        },
        "split": {
            "train_pct": train_pct, "val_pct": val_pct,
            "test_pct": test_pct, "seed": int(split_seed),
            "n_train": len(summary.get("ifc_ids", {}).get("train", [])),
            "n_val":   len(summary.get("ifc_ids", {}).get("val", [])),
            "n_test":  len(summary.get("ifc_ids", {}).get("test", [])),
        },
        "model": {
            "type": model_type,
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "dropout": float(dropout),
            "edge_emb_dim": int(edge_emb_dim),
        },
        "optim": {
            "epochs": int(epochs), "lr": float(lr),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
        },
        "masks": {
            "numeric": bool(mask_num),
            "pset": bool(mask_pset),
            "type": bool(mask_type),
        },
        "best": {
            "epoch": summary.get("best_epoch"),
            "val_f1": summary.get("best_val_f1"),
        },
        "test_metrics": {
            k: float(getattr(summary.get("test"), k, 0))
            for k in ("precision", "recall", "f1", "auc_roc",
                      "balanced_acc", "mcc")
            if summary.get("test") is not None
        },
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # Sonuç kartı
    st.success(f"✓ Eğitim bitti · {dt:.1f}s")
    test = summary.get("test")
    val = summary.get("val")
    if test is not None:
        rc = st.columns(5)
        rc[0].metric("Test F1", f"{getattr(test, 'f1', 0):.3f}")
        rc[1].metric("Test AUC", f"{getattr(test, 'auc_roc', 0):.3f}")
        rc[2].metric("Test Precision",
                     f"{getattr(test, 'precision', 0):.3f}")
        rc[3].metric("Test Recall",
                     f"{getattr(test, 'recall', 0):.3f}")
        rc[4].metric("Best epoch",
                     f"{summary.get('best_epoch', '?')}")
    st.caption(
        f"📁 `{run_dir.name}` · {len(history)} epoch · "
        f"sonraki adım: **Sayfa 15** → bu run'ı seç → 'Tahmin Yap'"
    )


# --- 6. Geçmiş run'lar ---------------------------------------------------
st.divider()
st.subheader("📋 Geçmiş run'lar")
run_root = Path("runs")
if not run_root.exists():
    st.caption("Henüz hiç eğitim çalıştırılmamış.")
else:
    rows = []
    for rd in sorted(run_root.iterdir(), reverse=True):
        if not rd.is_dir():
            continue
        m = {}
        meta_p = rd / "meta.json"
        if meta_p.exists():
            try:
                m = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                m = {}
        s = {}
        sum_p = rd / "summary.json"
        if sum_p.exists():
            try:
                s = json.loads(sum_p.read_text(encoding="utf-8"))
            except Exception:
                pass
        rows.append({
            "Run": rd.name,
            "Dataset": (m.get("dataset") or {}).get("name") or "—",
            "Model": (m.get("model") or {}).get("type") or "?",
            "Test F1": (m.get("test_metrics") or {}).get("f1"),
            "Test AUC": (m.get("test_metrics") or {}).get("auc_roc"),
            "Best ep": (m.get("best") or {}).get("epoch")
                       or s.get("best_epoch"),
            "Süre s": (m.get("duration_s") or "—"),
            "Oluşturma": (m.get("created_at") or "—")[:19],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        st.caption(
            "💡 Sayfa 15 → tahmin sütununda bu run'lardan birini seç → "
            "'Tahmin Yap'."
        )
    else:
        st.caption("Henüz hiç eğitim çalıştırılmamış.")
