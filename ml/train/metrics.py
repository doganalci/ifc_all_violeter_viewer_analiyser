"""Evaluation metrics for node-level violation detection.

Sınıflandırma performansını mümkün olduğunca eleştirel açıdan ölçer.
Confusion matrix, AUC, threshold sweep ve per-category breakdown
hepsi `EvalResult` içine yazılır — JSON'a serileştirilebilir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    # --- Ana metrikler --------------------------------------------------
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float          # (TPR + TNR) / 2 — sınıf dengesizliğine sağlam
    mcc: float                        # Matthews correlation coef — -1..+1
    auc_roc: float                    # ROC eğrisi altındaki alan (probs gerekli)
    auc_pr: float                     # Precision-Recall eğrisi altındaki alan
    decoy_fpr: float

    # --- Confusion matrix ----------------------------------------------
    tp: int
    fp: int
    fn: int
    tn: int
    n_positive: int
    n_predicted_positive: int
    n_decoys: int

    # --- Per-category ----------------------------------------------------
    per_category_recall: dict[str, float] = field(default_factory=dict)
    per_category_precision: dict[str, float] = field(default_factory=dict)
    per_category_f1: dict[str, float] = field(default_factory=dict)
    per_category_support: dict[str, int] = field(default_factory=dict)

    # --- Threshold sweep (opsiyonel; raw çıktıdan hesaplanır) ----------
    threshold_sweep: list[dict] = field(default_factory=list)

    # ----- Yardımcılar ---------------------------------------------------
    @property
    def tpr(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float:
        return _safe_div(self.fp, self.fp + self.tn)

    @property
    def tnr(self) -> float:
        return _safe_div(self.tn, self.tn + self.fp)

    def confusion_matrix(self) -> np.ndarray:
        """[[TN, FP], [FN, TP]] — sklearn ile uyumlu."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)

    def confusion_matrix_pretty(self) -> str:
        """İnsan-okur konsol çıktısı."""
        return (
            "                    pred=neg   pred=pos\n"
            f"  actual=neg      {self.tn:>9d}  {self.fp:>9d}\n"
            f"  actual=pos      {self.fn:>9d}  {self.tp:>9d}\n"
            f"  precision={self.precision:.3f}  recall={self.recall:.3f}  f1={self.f1:.3f}\n"
            f"  balanced_acc={self.balanced_accuracy:.3f}  mcc={self.mcc:+.3f}  "
            f"AUC-ROC={self.auc_roc:.3f}  AUC-PR={self.auc_pr:.3f}\n"
            f"  decoy_fpr={self.decoy_fpr:.3f}  ({self.n_decoys} decoys)"
        )

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "decoy_fpr": self.decoy_fpr,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "n_positive": self.n_positive,
            "n_predicted_positive": self.n_predicted_positive,
            "n_decoys": self.n_decoys,
            "per_category_recall": dict(self.per_category_recall),
            "per_category_precision": dict(self.per_category_precision),
            "per_category_f1": dict(self.per_category_f1),
            "per_category_support": dict(self.per_category_support),
            "threshold_sweep": list(self.threshold_sweep),
        }


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den <= 0:
        return 0.0
    return float(num / np.sqrt(den))


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC altında alan (Mann-Whitney U formu).

    Rank'ler ascending sıralamadan gelir: en düşük skor rank=1.
    Yüksek skorlu pozitiflerin rank'ı büyük olmalı (iyi AUC).
    """
    y_bool = y_true.astype(bool)
    n_pos = int(y_bool.sum())
    n_neg = int((~y_bool).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(y_score)             # ascending: en düşük skor başta
    y_sorted = y_bool[order]
    ranks = np.arange(1, len(y_sorted) + 1)
    rank_sum = float(ranks[y_sorted].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (PR eğrisi altında alan)."""
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    y = y_true[order].astype(int)
    cum_tp = np.cumsum(y)
    precision_at_k = cum_tp / np.arange(1, len(y) + 1)
    return float(precision_at_k[y == 1].mean()) if y.sum() > 0 else 0.0


def _threshold_sweep(
    y_true: np.ndarray, y_score: np.ndarray, n_points: int = 21
) -> list[dict]:
    """11/21 noktada precision-recall-f1 — eşik seçimini bilgilendirir."""
    y_bool = y_true.astype(bool)
    not_y = ~y_bool
    out = []
    for t in np.linspace(0.0, 1.0, n_points):
        pred = (y_score >= t).astype(bool)
        tp = int(np.logical_and(pred, y_bool).sum())
        fp = int(np.logical_and(pred, not_y).sum())
        fn = int(np.logical_and(~pred, y_bool).sum())
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        out.append({
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decoy_mask: np.ndarray,
    categories: list[str | None] | None = None,
    y_score: np.ndarray | None = None,
) -> EvalResult:
    """Tüm metrik paketini hesapla.

    Args:
        y_true:    [N] {0,1} ground truth.
        y_pred:    [N] {0,1} eşiklenmiş tahmin.
        decoy_mask: [N] bool — decoy node maskesi.
        categories: opsiyonel [N] kategori listesi.
        y_score:   [N] [0..1] olasılık skorları. Varsa AUC ve threshold
                   sweep hesaplanır; yoksa AUC=0 raporlanır.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    decoy_mask = decoy_mask.astype(bool)

    tp = int(np.logical_and(y_pred, y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    bal_acc = (tpr + tnr) / 2
    mcc = _mcc(tp, fp, fn, tn)

    auc_roc = _auc_roc(y_true.astype(int), y_score) if y_score is not None else 0.0
    auc_pr = _auc_pr(y_true.astype(int), y_score) if y_score is not None else 0.0
    sweep = _threshold_sweep(y_true.astype(int), y_score) if y_score is not None else []

    n_decoys = int(decoy_mask.sum())
    decoy_fpr = _safe_div(int(np.logical_and(decoy_mask, y_pred).sum()), n_decoys)

    per_cat_recall: dict[str, float] = {}
    per_cat_prec: dict[str, float] = {}
    per_cat_f1: dict[str, float] = {}
    per_cat_support: dict[str, int] = {}

    if categories is not None:
        # Pozitif örnekleri kategoriye göre topla (recall için)
        pos_buckets: dict[str, list[bool]] = {}
        for i, cat in enumerate(categories):
            if not y_true[i] or cat is None:
                continue
            pos_buckets.setdefault(cat, []).append(bool(y_pred[i]))
        # Tahmin edilen pozitifleri kategoriye göre topla (precision için)
        pred_buckets: dict[str, list[bool]] = {}
        for i, cat in enumerate(categories):
            if not y_pred[i] or cat is None:
                continue
            pred_buckets.setdefault(cat, []).append(bool(y_true[i]))

        cats_seen = set(pos_buckets) | set(pred_buckets)
        for c in sorted(cats_seen):
            r_vals = pos_buckets.get(c, [])
            p_vals = pred_buckets.get(c, [])
            r = float(np.mean(r_vals)) if r_vals else 0.0
            p = float(np.mean(p_vals)) if p_vals else 0.0
            per_cat_recall[c] = r
            per_cat_prec[c] = p
            per_cat_f1[c] = _safe_div(2 * p * r, p + r)
            per_cat_support[c] = len(r_vals)

    return EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        balanced_accuracy=bal_acc,
        mcc=mcc,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        decoy_fpr=decoy_fpr,
        tp=tp, fp=fp, fn=fn, tn=tn,
        n_positive=int(y_true.sum()),
        n_predicted_positive=int(y_pred.sum()),
        n_decoys=n_decoys,
        per_category_recall=per_cat_recall,
        per_category_precision=per_cat_prec,
        per_category_f1=per_cat_f1,
        per_category_support=per_cat_support,
        threshold_sweep=sweep,
    )
