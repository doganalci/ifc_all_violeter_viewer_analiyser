"""Ortak OpenAI chat sarmalayıcısı.

İki problemi çözer:
1. Rate-limit / 5xx → exponential backoff retry.
2. Yeni / küçük modellerin (örn. gpt-5, o-serisi) `temperature`,
   `response_format`, `top_p`, `max_tokens` gibi parametreleri kabul
   etmemesi → BadRequest yerine kademeli olarak parametreyi düşür ve
   tekrar dene.

Önceden bu sarmalayıcı yoktu; gpt-4o-mini çalışıyor ama gpt-5 / o1 /
o3-mini gibi modeller `unsupported parameter: temperature` hatasıyla
patlıyordu.
"""
from __future__ import annotations

import time

# Parametreler "ciddiyet" sırasına göre düşürülür: önce model davranışını
# az etkileyenler, sonra cevap format/uzunluk kontrolü.
_DROP_ORDER = ("top_p", "temperature", "response_format",
               "max_tokens", "max_completion_tokens")


def safe_chat(client, *, model: str, messages: list,
              max_retries: int = 6, **kwargs):
    """OpenAI chat.completions.create için güvenli sarmalayıcı.

    - 429 / 5xx → exponential backoff (cap 30s).
    - 'unsupported parameter' / 'not supported' → ilgili kwarg'ı düşür,
      kalan parametrelerle yeniden dene. Tüm uyumsuzlar düşene kadar
      veya istek başarılı olana kadar devam eder.
    """
    delay = 2.0
    last_exc: Exception | None = None
    # max_retries hem rate-limit hem param-drop için ortak sayaç
    for _ in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            last_exc = e
            msg = str(e)
            low = msg.lower()
            status = getattr(e, "status_code", None)

            # Param uyumsuzluğu — kalıcı, drop edip retry
            if ("unsupported" in low or "not supported" in low
                    or "does not support" in low
                    or "invalid value for" in low):
                dropped = _pick_dropped_param(low, kwargs)
                if dropped:
                    kwargs.pop(dropped, None)
                    continue
                # özel durum: 'max_tokens' yeni modellerde
                # 'max_completion_tokens' olarak isteniyor
                if ("max_tokens" in kwargs
                        and "max_completion_tokens" in low):
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                raise

            # Rate-limit / 5xx — geçici, backoff retry
            is_rate = ("429" in msg or status == 429
                       or "rate_limit" in low)
            is_5xx = any(c in msg for c in ("500", "502", "503", "504"))
            if not (is_rate or is_5xx):
                raise
            wait = delay
            try:
                ra = getattr(getattr(e, "response", None), "headers", {}) or {}
                if "retry-after" in {k.lower() for k in ra.keys()}:
                    wait = float(next(v for k, v in ra.items()
                                      if k.lower() == "retry-after"))
            except Exception:
                pass
            time.sleep(min(wait, 30.0))
            delay = min(delay * 2, 30.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("safe_chat: retries exhausted without exception")


def _pick_dropped_param(low_msg: str, kwargs: dict) -> str | None:
    """Hata mesajında geçen ilk uyumsuz parametreyi bul."""
    for k in kwargs:
        if k.lower() in low_msg:
            return k
    # mesajda parametre adı belirsizse en zararsızdan başla
    for k in _DROP_ORDER:
        if k in kwargs:
            return k
    return None
