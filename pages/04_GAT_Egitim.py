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
st.subheader("1. Dataset & etiket seçimi")

pools = reader_for(root).list_pool_runs()

if not pools:
    st.warning("Bu dataset'te kaydedilmiş ihlal havuzu (pool_run) yok.")
    st.stop()

pool_df = pd.DataFrame(pools)[["id", "name", "n_violated", "method", "llm_model", "created_at"]]
pool_df = pool_df.rename(columns={"id": "pool_id", "n_violated": "#IFC"})

cols = st.columns([3, 2])
with cols[0]:
    st.caption("Mevcut ihlal havuzları (codex1'de üretilen label set'leri):")
    st.dataframe(pool_df, hide_index=True, use_container_width=True)
with cols[1]:
    chosen_pools = st.multiselect(
        "Kullanılacak pool_run(lar)",
        options=[p["id"] for p in pools],
        default=[p["id"] for p in pools],
        format_func=lambda i: f"{next(p['name'] for p in pools if p['id']==i)} ({i[:8]})",
    )
    if not chosen_pools:
        st.error("En az bir pool seç.")
        st.stop()

# Collect violated IFC entries belonging to chosen pools.
all_violated = list_entries(root, kind="violated")
in_pool = [e for e in all_violated if e.get("pool_run_id") in chosen_pools]
with_graph = [e for e in in_pool if e.get("graph_ok")]
violated_entries = with_graph

diag = st.columns(3)
diag[0].metric("Toplam violated", len(all_violated))
diag[1].metric("Seçili pool'da", len(in_pool))
diag[2].metric("Graph'lı (eğitilebilir)", len(violated_entries))

if not violated_entries:
    # Help the user figure out *why* the filter is empty.
    if not all_violated:
        st.error(
            "Bu dataset'te hiç violated IFC yok. codex1'in pipeline'ında "
            "**'Pool'dan violated üret'** adımını çalıştırman gerek."
        )
    elif not in_pool:
        st.error(
            "Seçili pool(lara) bağlı violated IFC yok. Üstteki tabloda "
            "`#IFC` sütunundaki sayılara bak; >0 olan bir pool seç."
        )
    else:
        # In-pool var ama hiçbirinin graph'ı yok.
        without_graph = [e for e in in_pool if not e.get("graph_ok")]
        st.error(
            f"Seçili pool'da {len(in_pool)} violated IFC var ama hiçbirinin "
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

    if summary.get("test"):
        t = summary["test"]
        st.subheader("📊 Test sonucu")
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

        # Confusion matrix
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

        # Per-category breakdown
        per_p = t.get("per_category_precision", {})
        per_r = t.get("per_category_recall", {})
        per_f = t.get("per_category_f1", {})
        per_s = t.get("per_category_support", {})
        if per_r:
            st.markdown("### Kategori Bazında Performans")

            # Sayısal tablo: P/R/F1/Support + türetilmiş TP/FN
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

            # Stacked bar: TP (yeşil) + FN (kırmızı) per kategori
            st.markdown("**TP / FN dağılımı (kaç pozitif yakaladık vs kaçırdık)**")
            stack_df = cat_df[["kategori", "TP", "FN"]].set_index("kategori")
            st.bar_chart(stack_df, color=["#22c55e", "#ef4444"])

            # Renk-kodlu metrik bar chart
            st.markdown("**Precision / Recall / F1 — kategoriler arası karşılaştırma**")
            metric_df = cat_df.set_index("kategori")[["precision", "recall", "f1"]]
            st.bar_chart(metric_df)

            # Uyarı: zayıf kategoriler
            weak = cat_df[cat_df["f1"] < 0.7]
            if not weak.empty:
                with st.expander(f"⚠️ Zayıf kategoriler ({len(weak)})", expanded=True):
                    for _, row in weak.iterrows():
                        st.warning(
                            f"**{row['kategori']}** — F1={row['f1']:.2f} "
                            f"(R={row['recall']:.2f}, P={row['precision']:.2f}) "
                            f"· {row['TP']}/{row['support']} yakalandı, "
                            f"{row['FN']} kaçırıldı."
                        )

    st.code(summary["run_dir"])
    st.caption(
        "🔍 Daha detaylı analiz için sol menüden **GAT Test** sayfasına geç, bu run'ı seç. "
        "CLI'de `python ml/scripts/error_analysis.py` ve `sanity_check.py` daha kapsamlı rapor üretir."
    )
