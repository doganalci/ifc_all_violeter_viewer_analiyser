"""LLM üretim defterleri — her LLM çağrısı için bir Excel satırı.

Dosya: data_home() / llm_generations.xlsx (CSV fallback).
Sheet: 'uretim'.

Her satır:
    zaman | paket | ifc_adi | tur | model | prompt_tokens | completion_tokens
    | total_tokens | sure_s | maliyet_usd | tasarim_ozeti | rationale
    | user_prompt | status | error

Best-effort — yazma hatası akışı bozmaz, None döner.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path


def log_llm_generation(*,
                       paket: str,
                       ifc_name: str,
                       kind: str,
                       model: str,
                       user_prompt: str,
                       prompt_tokens: int = 0,
                       completion_tokens: int = 0,
                       duration_s: float = 0.0,
                       cost_usd: float = 0.0,
                       design_summary: str = "",
                       rationale: str = "",
                       status: str = "ok",
                       error: str = "") -> Path | None:
    """llm_generations.xlsx'e bir LLM çağrısı satırı ekler."""
    try:
        from ml.tracking import _write_table
        from paths import data_home
    except Exception:
        return None

    home = data_home()
    home.mkdir(parents=True, exist_ok=True)

    row = {
        "zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paket": paket,
        "ifc_adi": ifc_name,
        "tur": kind,                                  # ana_baseline | variant_baseline
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "sure_s": round(float(duration_s), 2),
        "maliyet_usd": round(float(cost_usd), 6),
        "tasarim_ozeti": design_summary[:300],
        "rationale": rationale[:300],
        "user_prompt": user_prompt[:1000],
        "status": status,
        "error": (error or "")[:300],
    }
    try:
        return _write_table([row], home / "llm_generations.xlsx",
                            sheet="uretim", append=True)
    except Exception:
        return None
