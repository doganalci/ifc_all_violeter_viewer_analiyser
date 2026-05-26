"""Eğitim sonrası detaylı PDF rapor.

build_report(run_dir) tek bir PDF üretir; içinde:
  * Başlık + özet + tüm hiperparametreler
  * Veri kümesi: paket(ler) (ana baseline), baseline/ihlal sayıları, train/val
    /test dağılımı, sınıf dengesi, örnek isimleri
  * Eğitim eğrileri: loss + val F1/P/R/AUC (epoch bazında)
  * Confusion matrix (train/val/test) ısı haritaları + metrik tablosu
  * Kategori bazında P/R/F1 (varsa)

matplotlib (requirements'ta) ile çizilir; tek dosya, harici bağımlılık yok.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from paths import data_home
from ml.tracking import _ifc_id_to_pkg


def _fig_text(pdf, title: str, lines: list[str]):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 dikey
    fig.text(0.07, 0.95, title, fontsize=16, weight="bold")
    y = 0.90
    for ln in lines:
        fig.text(0.07, y, ln, fontsize=9, family="monospace", va="top")
        y -= 0.022
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _table_fig(pdf, title: str, header: list[str], rows: list[list],
               note: str | None = None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=15, weight="bold")
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.78]); ax.axis("off")
    if rows:
        t = ax.table(cellText=rows, colLabels=header, loc="upper center",
                     cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
    if note:
        fig.text(0.07, 0.06, note, fontsize=8, style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def _confusion_fig(pdf, splits: dict[str, dict]):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, "Confusion Matrix (train / val / test)",
             fontsize=15, weight="bold")
    names = [s for s in ("train", "val", "test") if splits.get(s)]
    for i, name in enumerate(names):
        cm = (splits[name] or {}).get("confusion") or {}
        tn, fp = cm.get("tn", 0), cm.get("fp", 0)
        fn, tp = cm.get("fn", 0), cm.get("tp", 0)
        ax = fig.add_subplot(len(names), 1, i + 1)
        mat = [[tn, fp], [fn, tp]]
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Tahmin: değil", "Tahmin: ihlal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Gerçek: değil", "Gerçek: ihlal"])
        ax.set_title(f"{name}  (TP={tp} FP={fp} FN={fn} TN={tn})", fontsize=10)
        for (r, c), v in [((0, 0), tn), ((0, 1), fp), ((1, 0), fn), ((1, 1), tp)]:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=11,
                    color="black")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def _curves_fig(pdf, history: list[dict]):
    import matplotlib.pyplot as plt
    if not history:
        return
    ep = [h.get("epoch") for h in history]
    loss = [h.get("train_loss") for h in history]
    f1 = [(h.get("val") or {}).get("f1") for h in history]
    pr = [(h.get("val") or {}).get("precision") for h in history]
    rc = [(h.get("val") or {}).get("recall") for h in history]
    auc = [(h.get("val") or {}).get("auc_roc") for h in history]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, "Eğitim Eğrileri", fontsize=15, weight="bold")
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(ep, loss, marker="o", ms=3, color="#c0392b")
    ax1.set_title("Train Loss"); ax1.set_xlabel("epoch"); ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(ep, f1, label="val F1", marker="o", ms=3)
    ax2.plot(ep, pr, label="val Precision", marker="o", ms=3)
    ax2.plot(ep, rc, label="val Recall", marker="o", ms=3)
    ax2.plot(ep, auc, label="val AUC", marker="o", ms=3, ls="--")
    ax2.set_title("Validation Metrikleri"); ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1.02); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def _dist_fig(pdf, split_counts: dict[str, int], pkg_counts: dict[str, dict],
              test_cm: dict):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, "Veri Kümesi Dağılımı", fontsize=15, weight="bold")
    # train/val/test IFC sayıları
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(list(split_counts.keys()), list(split_counts.values()),
            color=["#2980b9", "#27ae60", "#e67e22"])
    ax1.set_title("IFC sayısı (train / val / test)")
    for i, v in enumerate(split_counts.values()):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    # test sınıf dengesi
    ax2 = fig.add_subplot(2, 1, 2)
    pos = (test_cm.get("tp", 0) + test_cm.get("fn", 0))
    neg = (test_cm.get("tn", 0) + test_cm.get("fp", 0))
    if pos + neg > 0:
        ax2.pie([pos, neg], labels=[f"ihlal ({pos})", f"ihlal değil ({neg})"],
                autopct="%1.1f%%", colors=["#e74c3c", "#95a5a6"])
        ax2.set_title("Test seti node sınıf dengesi")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def build_eval_report(out_path: str | Path, *, title: str,
                      meta_lines: list[str], agg: dict,
                      per_ifc_rows: list[dict] | None = None,
                      per_category: dict | None = None) -> Path | None:
    """GAT Test (toplu değerlendirme) için tek PDF rapor üret.

    agg: EvalResult.to_dict() benzeri (f1/precision/recall/.../confusion).
    per_ifc_rows: her IFC için {name,f1,TP,FP,FN,...} satırları (en kötüler).
    per_category: {'f1':{...},'precision':{...},'recall':{...},'support':{...}}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _g(k):
        return agg.get(k)

    try:
        with PdfPages(out_path) as pdf:
            _fig_text(pdf, title, meta_lines)
            mrows = [[
                "değerlendirme",
                f"{_g('f1'):.3f}" if _g('f1') is not None else "-",
                f"{_g('precision'):.3f}" if _g('precision') is not None else "-",
                f"{_g('recall'):.3f}" if _g('recall') is not None else "-",
                f"{_g('balanced_accuracy'):.3f}" if _g('balanced_accuracy') is not None else "-",
                f"{_g('mcc'):+.3f}" if _g('mcc') is not None else "-",
                f"{_g('auc_roc'):.3f}" if _g('auc_roc') is not None else "-",
                f"{_g('decoy_fpr'):.3f}" if _g('decoy_fpr') is not None else "-",
            ]]
            _table_fig(pdf, "Değerlendirme Metrikleri",
                       ["set", "F1", "P", "R", "Bal.Acc", "MCC", "AUC", "decoy_FPR"],
                       mrows)
            _confusion_fig(pdf, {"değerlendirme": {"confusion": agg.get("confusion") or {}}})
            if per_category and per_category.get("f1"):
                f1 = per_category.get("f1") or {}
                pr = per_category.get("precision") or {}
                rc = per_category.get("recall") or {}
                sup = per_category.get("support") or {}
                crows = [[c, f"{pr.get(c,0):.3f}", f"{rc.get(c,0):.3f}",
                          f"{f1.get(c,0):.3f}", sup.get(c, 0)] for c in sorted(f1)]
                _table_fig(pdf, "Kategori Bazında Performans",
                           ["kategori", "P", "R", "F1", "destek"], crows)
            if per_ifc_rows:
                rows = [[r.get("name", "")[:30], r.get("f1"), r.get("TP"),
                         r.get("FP"), r.get("FN")] for r in per_ifc_rows[:25]]
                _table_fig(pdf, "En kötü IFC'ler (F1 artan)",
                           ["IFC", "F1", "TP", "FP", "FN"], rows)
    except Exception:
        return None
    # data/reports arşiv kopyası
    try:
        rep_dir = data_home() / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(out_path, rep_dir / out_path.name)
    except Exception:
        pass
    return out_path


def build_report(run_dir: str | Path, dataset_root: str | None = None,
                 out_path: str | Path | None = None) -> Path | None:
    """run_dir'deki config+summary'den tek PDF rapor üret. Hata olursa None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return None

    run_dir = Path(run_dir)
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        summ = json.loads((run_dir / "summary.json").read_text())
    except Exception:
        return None

    dataset_root = dataset_root or cfg.get("dataset_root") or str(data_home())
    ids = summ.get("ifc_ids", {}) or {}

    # Veri kümesi bilgisi
    pkg_counts: dict[str, dict] = {}
    sample_names: dict[str, list[str]] = {"baseline": [], "violated": []}
    used_pkgs: list[str] = []
    try:
        id_pkg, id_entry = _ifc_id_to_pkg(dataset_root)
        used = set()
        for split in ("train", "val", "test"):
            for i in ids.get(split, []) or []:
                e = id_entry.get(i)
                if not e:
                    continue
                pkg = id_pkg.get(i, "(isimsiz)")
                used.add(pkg)
                s = pkg_counts.setdefault(pkg, {"baseline": 0, "violated": 0})
                if e.kind in s:
                    s[e.kind] += 1
                if len(sample_names.get(e.kind, [])) < 15:
                    sample_names.setdefault(e.kind, []).append(e.name)
        used_pkgs = sorted(used)
    except Exception:
        pass

    def _m(split, key):
        return (summ.get(split) or {}).get(key)

    tim = summ.get("timing") or {}
    out_path = Path(out_path) if out_path else (run_dir / "report.pdf")
    test_cm = (summ.get("test") or {}).get("confusion") or {}

    try:
        with PdfPages(out_path) as pdf:
            # 1) Başlık + hiperparametreler
            head = [
                f"Run: {run_dir.name}",
                f"Tarih: {datetime.now():%Y-%m-%d %H:%M}",
                f"Paket(ler) / ana baseline: {', '.join(used_pkgs) or '?'}",
                f"Cihaz: {cfg.get('device')}   best_epoch: {summ.get('best_epoch')}"
                f"   erken_durdu: {summ.get('stopped_early')}",
                "",
                "── HİPERPARAMETRELER ─────────────────────────────",
                f"model            : {cfg.get('model')}",
                f"hidden_dim       : {cfg.get('hidden_dim')}",
                f"heads            : {cfg.get('heads')}",
                f"dropout          : {cfg.get('dropout')}",
                f"edge_emb_dim     : {cfg.get('edge_emb_dim')}",
                f"lr               : {cfg.get('lr')}",
                f"weight_decay     : {cfg.get('weight_decay')}",
                f"epochs (ayar)    : {cfg.get('epochs')}",
                f"patience         : {cfg.get('patience')}",
                f"pos_weight       : {cfg.get('pos_weight')}",
                f"threshold        : {cfg.get('threshold')}",
                "",
                "── BÖLME / MASKELER ──────────────────────────────",
                f"val_frac/test_frac : {cfg.get('val_frac')} / {cfg.get('test_frac')}",
                f"split_seed         : {cfg.get('split_seed')}",
                f"include_baselines  : {cfg.get('include_baselines')}",
                f"mask_numeric/pset/type : {cfg.get('mask_numeric_features')}"
                f" / {cfg.get('mask_pset_features')} / {cfg.get('mask_type_features')}",
                f"rule_oracle        : {cfg.get('use_rule_oracle')}",
                "",
                "── SÜRELER ───────────────────────────────────────",
                f"toplam eğitim       : {tim.get('total_train_seconds')} s "
                f"({tim.get('epochs_run')} epoch)",
                f"epoch ortalama      : {tim.get('avg_epoch_seconds')} s",
                f"örnek başına eğitim : {tim.get('per_sample_train_ms')} ms/IFC (epoch içi)",
                f"test değerlendirme  : {tim.get('test_eval_seconds')} s "
                f"({tim.get('per_sample_test_ms')} ms/IFC)",
            ]
            _fig_text(pdf, "GAT Eğitim Raporu", head)

            # 2) Veri kümesi tablosu + dağılım
            pkg_rows = [[p, c["baseline"], c["violated"],
                         c["baseline"] + c["violated"]]
                        for p, c in sorted(pkg_counts.items())]
            _table_fig(pdf, "Veri Kümesi — Paketler (ana baseline)",
                       ["paket", "baseline", "ihlal", "toplam"], pkg_rows,
                       note="Her örneğin baseline/ana baseline soy ağacı için "
                            "data/datasets.xlsx → 'ornekler'.")
            split_counts = {
                "train": len(ids.get("train") or []),
                "val": len(ids.get("val") or []),
                "test": len(ids.get("test") or []),
            }
            _dist_fig(pdf, split_counts, pkg_counts, test_cm)

            # 3) Eğitim eğrileri
            _curves_fig(pdf, summ.get("history") or [])

            # 4) Metrik tablosu + confusion
            mrows = []
            for s in ("train", "val", "test"):
                if summ.get(s):
                    mrows.append([
                        s,
                        f"{_m(s,'f1'):.3f}" if _m(s, 'f1') is not None else "-",
                        f"{_m(s,'precision'):.3f}" if _m(s, 'precision') is not None else "-",
                        f"{_m(s,'recall'):.3f}" if _m(s, 'recall') is not None else "-",
                        f"{_m(s,'balanced_accuracy'):.3f}" if _m(s, 'balanced_accuracy') is not None else "-",
                        f"{_m(s,'mcc'):+.3f}" if _m(s, 'mcc') is not None else "-",
                        f"{_m(s,'auc_roc'):.3f}" if _m(s, 'auc_roc') is not None else "-",
                        f"{_m(s,'decoy_fpr'):.3f}" if _m(s, 'decoy_fpr') is not None else "-",
                    ])
            _table_fig(pdf, "Değerlendirme Metrikleri",
                       ["set", "F1", "P", "R", "Bal.Acc", "MCC", "AUC", "decoy_FPR"],
                       mrows)
            _confusion_fig(pdf, {"train": summ.get("train"),
                                 "val": summ.get("val"), "test": summ.get("test")})

            # 5) Kategori bazında P/R/F1 (varsa)
            per = (summ.get("test") or {}).get("per_category_f1") or {}
            if per:
                prec = (summ.get("test") or {}).get("per_category_precision") or {}
                rec = (summ.get("test") or {}).get("per_category_recall") or {}
                sup = (summ.get("test") or {}).get("per_category_support") or {}
                crows = [[c, f"{prec.get(c,0):.3f}", f"{rec.get(c,0):.3f}",
                          f"{per.get(c,0):.3f}", sup.get(c, 0)]
                         for c in sorted(per)]
                _table_fig(pdf, "Kategori Bazında Performans (test)",
                           ["kategori", "P", "R", "F1", "destek"], crows)

            # 6) Örnek isimleri
            ex_lines = ["── BASELINE örnekleri ──"] + \
                [f"  {n}" for n in sample_names.get("baseline", [])[:15]] + \
                ["", "── İHLAL örnekleri ──"] + \
                [f"  {n}" for n in sample_names.get("violated", [])[:15]]
            _fig_text(pdf, "Örnek İsimleri (soy ağacı)", ex_lines)
    except Exception:
        return None

    # data klasörüne de bir kopya arşivle
    try:
        rep_dir = data_home() / "reports"
        rep_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(out_path, rep_dir / f"{run_dir.name}_report.pdf")
    except Exception:
        pass
    return out_path
