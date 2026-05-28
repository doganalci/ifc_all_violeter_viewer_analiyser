"""OpenAI model fiyatlandırma tablosu (USD / 1M token).

Resmi fiyatlardan derlenmiştir; bilinmeyen modeller için gpt-4o
varsayılan kullanılır ve UI'da "tahmin" olarak işaretlenir.

Versiyon ekli model adları (gpt-4o-2024-08-06 vs.) en uzun prefix eşleşmesi
ile bulunur.
"""
from __future__ import annotations


# (input_per_1M_usd, output_per_1M_usd)
_PRICING: dict[str, tuple[float, float]] = {
    # GPT-4 ailesi
    "gpt-4o":           (2.50, 10.00),
    "gpt-4o-mini":      (0.15,  0.60),
    "gpt-4-turbo":     (10.00, 30.00),
    "gpt-4-32k":       (60.00, 120.00),
    "gpt-4":           (30.00, 60.00),
    "gpt-3.5-turbo":    (0.50,  1.50),
    # GPT-5 ailesi (kabaca tahmin; resmi açıklanırsa güncellenir)
    "gpt-5-mini":       (0.50,  1.50),
    "gpt-5":           (10.00, 30.00),
    # OpenAI o-serisi (reasoning)
    "o1-mini":          (3.00, 12.00),
    "o1":              (15.00, 60.00),
}


def estimate_cost(model: str, prompt_tokens: int,
                  completion_tokens: int) -> dict:
    """Verilen token sayılarına göre USD maliyet tahmini.

    Returns:
        {
          "input_usd":     float,
          "output_usd":    float,
          "total_usd":     float,
          "is_estimate":   bool,    # True iff bilinmeyen model
          "matched":       str,     # eşleşen fiyat satırı
        }
    """
    base = (model or "").strip().lower()
    # En uzun prefix eşleşmesi (versiyon ekli adları yakalamak için)
    keys = sorted(_PRICING.keys(), key=lambda k: -len(k))
    matched = next((k for k in keys if base.startswith(k)), None)
    if matched:
        inp_rate, out_rate = _PRICING[matched]
        is_est = False
    else:
        inp_rate, out_rate = _PRICING["gpt-4o"]
        matched = "unknown→gpt-4o"
        is_est = True

    inp_usd = (max(0, prompt_tokens) / 1_000_000) * inp_rate
    out_usd = (max(0, completion_tokens) / 1_000_000) * out_rate
    return {
        "input_usd": round(inp_usd, 6),
        "output_usd": round(out_usd, 6),
        "total_usd": round(inp_usd + out_usd, 6),
        "is_estimate": is_est,
        "matched": matched,
    }


def known_models() -> list[str]:
    """UI'da gösterilecek bilinen model listesi (sıralı)."""
    return [
        "gpt-4o", "gpt-4o-mini",
        "gpt-4-turbo", "gpt-4",
        "gpt-5", "gpt-5-mini",
        "gpt-3.5-turbo",
        "o1", "o1-mini",
    ]
