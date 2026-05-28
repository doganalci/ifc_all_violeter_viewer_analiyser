"""LLM üretim defterleri — her LLM çağrısı için bir Excel satırı.

Dosya: data_home() / llm_generations.xlsx (CSV fallback).
Sheet: 'uretim'.

Her satır:
    zaman | paket | ifc_adi | tur | model | prompt_tokens | completion_tokens
    | total_tokens | sure_s | maliyet_usd | tasarim_ozeti | rationale
    | user_prompt | parametreler_json | tasarim_plan_json | status | error

Best-effort — yazma hatası akışı bozmaz, None döner.
"""
from __future__ import annotations

import json
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
                       parametreler: dict | None = None,
                       tasarim_plan: dict | None = None,
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
        "tur": kind,
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "sure_s": round(float(duration_s), 2),
        "maliyet_usd": round(float(cost_usd), 6),
        "tasarim_ozeti": design_summary[:300],
        "rationale": rationale[:300],
        "user_prompt": user_prompt[:1500],
        "parametreler_json": (
            json.dumps(parametreler, ensure_ascii=False) if parametreler
            else ""
        )[:3000],
        "tasarim_plan_json": (
            json.dumps(tasarim_plan, ensure_ascii=False) if tasarim_plan
            else ""
        )[:3000],
        "status": status,
        "error": (error or "")[:300],
    }
    try:
        return _write_table([row], home / "llm_generations.xlsx",
                            sheet="uretim", append=True)
    except Exception:
        return None


def read_llm_totals(filter_paket: str | None = None) -> dict | None:
    """llm_generations.xlsx'i okuyup toplam istatistik döner.

    filter_paket verilirse sadece o paketin satırları sayılır.

    Returns: {n_calls, total_tokens, total_cost_usd, total_duration_s,
              n_ok, n_error}, ya da None (dosya yok / okunamadı).
    """
    try:
        from paths import data_home
        import pandas as pd
    except Exception:
        return None
    path = data_home() / "llm_generations.xlsx"
    if not path.exists():
        # CSV fallback de kontrol et
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                return None
        else:
            return None
    else:
        try:
            df = pd.read_excel(path, sheet_name="uretim")
        except Exception:
            return None
    if filter_paket:
        df = df[df.get("paket", "") == filter_paket]
    n = len(df)
    if n == 0:
        return {"n_calls": 0, "total_tokens": 0, "total_cost_usd": 0.0,
                "total_duration_s": 0.0, "n_ok": 0, "n_error": 0}

    def _sum(col):
        return float(df[col].sum()) if col in df.columns else 0.0

    return {
        "n_calls": int(n),
        "total_tokens": int(_sum("total_tokens")),
        "total_cost_usd": _sum("maliyet_usd"),
        "total_duration_s": _sum("sure_s"),
        "n_ok": int((df.get("status", "ok") == "ok").sum()),
        "n_error": int((df.get("status", "ok") == "error").sum()),
    }
