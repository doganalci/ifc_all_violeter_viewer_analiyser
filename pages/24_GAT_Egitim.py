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
from services.metric_colors import METRIC_COLORS, metric_colors, metric_label
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
        epochs = st.number_input(
            "epochs", 5, 1000, 150, step=5,
            help="Maksimum epoch sayısı. Default 150 — yavaş öğrenip "
                 "daha iyi yakınsasın.",
        )
        lr = st.select_slider(
            "learning rate",
            options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3,
                     5e-3, 1e-2, 3e-2],
            value=1e-3,
            format_func=lambda x: f"{x:.0e}",
            help="Düşük LR → yavaş ama stabil öğrenme. Erken val_F1 "
                 "platosu görüyorsan 3e-4 veya 1e-4 dene.",
        )
        weight_decay = st.select_slider(
            "weight_decay",
            options=[0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
            value=5e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        patience = st.number_input(
            "early-stop patience", 0, 200, 30,
            help="0 → erken durdurma kapalı. Default 30 — val_F1'in "
                 "yavaş yavaş iyileşmesine vakit tanı. Çok hızlı "
                 "duruyorsa 50-80'e çıkar.",
        )
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

# Run adı — default `<model>_<dataset>_<timestamp>`. Default sadece
# session'a 1 kez yazılır; kullanıcı yazdıktan sonra rerun'larda korunur.
# Dataset veya model değişirse default yenilenir (kullanıcının elle
# yazdığı son değer üzerine yazılır — istenen davranış).
_data_token = (chosen_dataset_slug or "manual")
_rn_sig = f"{model_type}::{_data_token}"
if st.session_state.get("_rn_sig") != _rn_sig:
    st.session_state["run_name_input"] = (
        f"{model_type}_{_data_token}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    st.session_state["_rn_sig"] = _rn_sig

run_name = st.text_input(
    "Run adı (run klasör adı)",
    key="run_name_input",
    help="Default: `<model>_<dataset>_<timestamp>`. Sayfa 15 viewer "
         "dropdown'ında bu ad görünür. İstediğin gibi değiştirebilirsin; "
         "dataset veya model değiştirirsen default yeniden oluşturulur. "
         "Eski adla aynı verirsen önceki run'ın üzerine yazılır.",
)
default_name = st.session_state["run_name_input"]


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
        # res EvalResult dataclass'ı veya dict olabilir; ikisini de yakala.
        def _val(k, default=0.0):
            try:
                if isinstance(res, dict):
                    return float(res.get(k, default))
                return float(getattr(res, k, default))
            except Exception:
                return default

        history.append({
            "epoch": ep,
            "loss": float(loss),
            "f1": _val("f1"),
            "precision": _val("precision"),
            "recall": _val("recall"),
            "accuracy": _val("accuracy"),
            "balanced_accuracy": _val("balanced_accuracy"),
            "mcc": _val("mcc"),
            "auc_roc": _val("auc_roc"),
            "auc_pr": _val("auc_pr"),
            "decoy_fpr": _val("decoy_fpr"),
        })
        bar.progress(min(1.0, ep / max(epochs, 1)),
                     text=f"epoch {ep}/{epochs} · loss={loss:.4f} · "
                          f"val F1={_val('f1'):.3f} · "
                          f"acc={_val('accuracy'):.3f}")
        df = pd.DataFrame(history).set_index("epoch")
        with metric_slot.container():
            # En üst — birbirine yakın metrik grupları tek chart'ta,
            # sabit renkler (alttaki ayrı chart'larla aynı palet).
            def _legend_md(cols: list[str]) -> str:
                bits = []
                for c in cols:
                    col = METRIC_COLORS.get(c, "#94a3b8")
                    bits.append(
                        f"<span style='display:inline-block;width:10px;"
                        f"height:10px;background:{col};margin-right:4px;"
                        f"border-radius:2px;'></span>{metric_label(c)}"
                    )
                return "&nbsp;&nbsp;".join(bits)

            top1, top2 = st.columns(2)
            _doğr_cols = [c for c in ("f1", "accuracy", "balanced_accuracy")
                           if c in df.columns]
            if _doğr_cols:
                top1.caption("Doğruluk genel")
                top1.line_chart(df[_doğr_cols], height=240,
                                color=metric_colors(_doğr_cols))
                top1.markdown(_legend_md(_doğr_cols),
                              unsafe_allow_html=True)
            _kalite_cols = [c for c in ("precision", "recall",
                                          "auc_roc", "auc_pr")
                             if c in df.columns]
            if _kalite_cols:
                top2.caption("Sınıflandırma kalitesi")
                top2.line_chart(df[_kalite_cols], height=240,
                                color=metric_colors(_kalite_cols))
                top2.markdown(_legend_md(_kalite_cols),
                              unsafe_allow_html=True)

            # Her metrik için ayrı chart — üstle aynı renkler.
            _grid = [
                ("Loss (train)", ["loss"]),
                ("F1 (val)", ["f1"]),
                ("Accuracy (val)", ["accuracy"]),
                ("Balanced Accuracy (val)", ["balanced_accuracy"]),
                ("Precision (val)", ["precision"]),
                ("Recall (val)", ["recall"]),
                ("AUC ROC (val)", ["auc_roc"]),
                ("AUC PR (val)", ["auc_pr"]),
                ("MCC (val)", ["mcc"]),
                ("Decoy FPR (val)", ["decoy_fpr"]),
            ]
            # 2 sütunlu grid
            for i in range(0, len(_grid), 2):
                cc = st.columns(2)
                for j, (title, cols) in enumerate(_grid[i:i + 2]):
                    if not all(c in df.columns for c in cols):
                        continue
                    cc[j].caption(title)
                    cc[j].line_chart(df[cols], height=160,
                                     color=metric_colors(cols))

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
        "metrics": {
            split: {
                key: float((summary.get(split) or {}).get(src, 0))
                for key, src in (
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("f1", "f1"),
                    ("accuracy", "accuracy"),
                    ("balanced_accuracy", "balanced_accuracy"),
                    ("auc_roc", "auc_roc"),
                    ("auc_pr", "auc_pr"),
                    ("mcc", "mcc"),
                    ("decoy_fpr", "decoy_fpr"),
                )
            }
            for split in ("train", "val", "test")
            if summary.get(split) is not None
        },
        # Eski 'test_metrics' alanı backwards-compat (sayfa 15 dropdown
        # rozeti hâlâ onu okuyor).
        "test_metrics": {
            k: float((summary.get("test") or {}).get(src, 0))
            for k, src in (
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "f1"),
                ("auc_roc", "auc_roc"),
                ("balanced_acc", "balanced_accuracy"),
                ("mcc", "mcc"),
            )
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
    train_res = summary.get("train")

    def _m(d, key, default=0.0) -> float:
        if d is None:
            return default
        try:
            return float(d[key])
        except Exception:
            return default

    # En kritik bilgi: train/val/test ayrı ayrı. Aralarında uçurum varsa
    # overfit; üçü de çok yüksekse leak şüphesi.
    st.markdown("### 📊 Train / Val / Test metrikleri")
    metric_keys = [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Bal. Acc"),
        ("auc_roc", "AUC ROC"),
        ("auc_pr", "AUC PR"),
        ("mcc", "MCC"),
        ("decoy_fpr", "Decoy FPR"),
    ]
    rows = []
    for split_name, split_d in (
        ("Train", train_res), ("Val", val), ("Test", test),
    ):
        if split_d is None:
            continue
        rows.append({
            "Split": split_name,
            **{label: round(_m(split_d, k), 4) for k, label in metric_keys},
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

    # Hızlı tanı: F1 farkına bakıp uyarı ver.
    if train_res is not None and test is not None:
        _tr_f1 = _m(train_res, "f1")
        _ts_f1 = _m(test, "f1")
        _gap = _tr_f1 - _ts_f1
        if _tr_f1 > 0.95 and _ts_f1 > 0.85:
            st.warning(
                "⚠️ Train ve Test F1 ikisi de çok yüksek. **Data leakage "
                "şüphesi**: 🎭 mask expander'ında 'IFC type one-hot mask' "
                "ve 'Numeric features mask' aç → tekrar eğit. Mask'lerle "
                "F1 0.5-0.7'ye düşerse o gerçek performans."
            )
        elif _gap > 0.15:
            st.warning(
                f"⚠️ Train F1 ({_tr_f1:.3f}) >> Test F1 ({_ts_f1:.3f}) — "
                "**overfit**. dropout artır (0.5+), hidden_dim azalt "
                "(64→32), weight_decay artır (5e-4→1e-3)."
            )

    # Confusion matrix'leri (varsa)
    cms = []
    for split_name, split_d in (
        ("Train", train_res), ("Val", val), ("Test", test),
    ):
        if split_d is None:
            continue
        cm = split_d.get("confusion") if isinstance(split_d, dict) else None
        if not cm:
            continue
        # cm formatı: {"tn":int, "fp":int, "fn":int, "tp":int}
        cms.append((split_name, cm))
    if cms:
        st.markdown("### 🧮 Confusion matrix'ler")
        cm_cols = st.columns(len(cms))
        for col, (name, cm) in zip(cm_cols, cms):
            with col:
                st.caption(f"**{name}**")
                tn = int(cm.get("tn", 0))
                fp = int(cm.get("fp", 0))
                fn = int(cm.get("fn", 0))
                tp = int(cm.get("tp", 0))
                df_cm = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Gerçek 0", "Gerçek 1"],
                    columns=["Tahmin 0", "Tahmin 1"],
                )
                st.dataframe(df_cm, use_container_width=True)
                _support = tn + fp + fn + tp
                _pos = tp + fn
                st.caption(
                    f"n={_support} · pozitif={_pos} "
                    f"({_pos / max(_support, 1) * 100:.1f}%)"
                )

    # Run klasörü + sonraki adım
    st.caption(
        f"📁 `{run_dir.name}` · best epoch={summary.get('best_epoch', '?')}"
        f" · {len(history)} epoch çalıştı · sonraki adım: **Sayfa 15** "
        "→ bu run'ı seç → 'Tahmin Yap'"
    )


# --- 6. Geçmiş run'lar (son 5 — tamamı için sayfa 25) -------------------
st.divider()
st.subheader("📋 Son eğitimler")
st.caption(
    "Sadece en son 5 run gösteriliyor. Tüm geçmiş + karşılaştırma + "
    "detay incelemesi için **📈 Sayfa 25 — Eğitim Performans**."
)
run_root = Path("runs")
if not run_root.exists():
    st.caption("Henüz hiç eğitim çalıştırılmamış.")
else:
    rows = []
    for rd in sorted(run_root.iterdir(), reverse=True)[:5]:
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
        _mm = m.get("metrics") or {}
        _tr = _mm.get("train") or {}
        _va = _mm.get("val") or {}
        _te = _mm.get("test") or (m.get("test_metrics") or {})
        rows.append({
            "Run": rd.name,
            "Dataset": (m.get("dataset") or {}).get("name") or "—",
            "Model": (m.get("model") or {}).get("type") or "?",
            "Train F1": _tr.get("f1"),
            "Val F1": _va.get("f1"),
            "Test F1": _te.get("f1"),
            "Test AUC": _te.get("auc_roc"),
            "Best ep": (m.get("best") or {}).get("epoch")
                       or s.get("best_epoch"),
            "Süre s": (m.get("duration_s") or "—"),
            "Oluşturma": (m.get("created_at") or "—")[:19],
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
