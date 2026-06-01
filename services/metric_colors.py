"""Eğitim metrik chart'ları için sabit renk paleti.

Hem birleşik özet chart'larda hem ayrı küçük chart'larda aynı renk
kullanılır; kullanıcı aşağı kaydırdığında 'şu mavi olan F1'di' diye
karıştırmaz.
"""
from __future__ import annotations


METRIC_COLORS: dict[str, str] = {
    "loss":               "#ef4444",   # red
    "f1":                 "#3b82f6",   # blue
    "accuracy":           "#10b981",   # emerald
    "balanced_accuracy":  "#a855f7",   # purple
    "precision":          "#f59e0b",   # amber
    "recall":             "#ec4899",   # pink
    "auc_roc":            "#06b6d4",   # cyan
    "auc_pr":             "#84cc16",   # lime
    "mcc":                "#8b5cf6",   # violet
    "decoy_fpr":          "#64748b",   # slate
}


def metric_colors(keys) -> list[str]:
    """Verilen kolon adlarına karşılık renk listesi.

    Bilinmeyen anahtarlar için gri fallback.
    """
    return [METRIC_COLORS.get(k, "#94a3b8") for k in keys]


def metric_label(key: str) -> str:
    """İnsan-okur etiket (chart caption / legend için)."""
    return {
        "loss": "Loss",
        "f1": "F1",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Acc",
        "precision": "Precision",
        "recall": "Recall",
        "auc_roc": "AUC ROC",
        "auc_pr": "AUC PR",
        "mcc": "MCC",
        "decoy_fpr": "Decoy FPR",
    }.get(key, key)
