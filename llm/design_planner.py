"""LLM tasarım planlayıcı — bina parametreleri üretir (prosedürel çizici besler).

LLM IFC dosyasını DOĞRUDAN yazmaz (gpt-4o bile büyük IFC'leri tutarlı yazamıyor).
Bunun yerine LLM tasarım parametrelerini JSON olarak üretir → ml.data.synth_baseline_v2
prosedürel motoru bunu alır ve geçerli IFC4 dosyasına çevirir.

İki mod:
  * plan_ana_baseline   → kullanıcı template prompt'undan tek tasarım planı.
  * plan_variant_baseline → ana baseline'a varyasyon olarak tasarım planı
    (her varyant kendi LLM çağrısı, kendi tohum/varyasyon talimatıyla).

Her DesignPlan, üretimde kullanılan SİSTEM + USER prompt'unu sakla (DB + UI
için). Şeffaflık: her IFC için "bu IFC hangi prompt'la üretildi" görünür.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ml.data.synth_baseline_v2 import SpecOverride


# Sistem prompt — LLM'in rolü ve çıktı JSON şeması. Eski OPTIMIZED_PROMPT'tan
# ilham; sade tutuldu çünkü çıktı uzun değil (5 alan).
SYSTEM_PROMPT = """Sen bir mimar / BIM tasarımcısısın. Kullanıcının istediği
binanın TASARIM PARAMETRELERİNİ JSON olarak üretirsin. Bu parametreler
prosedürel bir IFC çizici motor tarafından gerçek IFC4 dosyasına dönüştürülür.

ZORUNLU JSON şeması (SADECE bu alanlar, ek alan ekleme):
{
  "n_storeys": <int 1-3>,
  "n_rooms_per_floor": <int 2-4>,
  "layout": "straight" | "lshape",
  "storey_height": <float 2.7-3.5>,
  "rationale": "<kısa Türkçe, neden bu seçim>"
}

KURALLAR:
- Sadece JSON döndür, açıklama veya kod bloğu yok.
- "straight": merkez koridor, oda kanatlar (N/S/E/W).
- "lshape": L şeklinde koridor (en fazla 3 oda; daha fazlasını straight yap).
- Bina küçük ölçekte (toplam oda ≤ 12). Büyük binalar şu an desteklenmiyor.
- Tüm değerler aralık dışında kalırsa motor klamplar; sen yine de aralıkta tut.
"""


VARIATION_INSTRUCTION = """

GÖREVİN: Ana baseline'a BENZER ama VARYASYONLU bir bina üret.

Ana baseline özeti:
{ana_summary}

Varyasyon rehberi:
- Aynı bina tipi/işlevi (ofis/konut/okul vs.) korunsun.
- Şu özelliklerden 1-2 tanesi farklı olsun:
    * oda sayısı (±1)
    * layout ("straight" ↔ "lshape")
    * kat sayısı (±1, sınırda kalmak şartıyla)
    * storey_height (küçük fark)
- Tohum: {seed} — bu tohum aynıysa aynı varyasyon dön.
"""


@dataclass
class DesignPlan:
    """LLM tasarım planı + kullanılan prompt'lar (şeffaflık için saklı)."""
    n_storeys: int
    n_rooms_per_floor: int
    layout: str
    storey_height: float
    rationale: str

    # Şeffaflık — bu IFC hangi prompt'la üretildi (DB'ye + UI'da expander).
    system_prompt: str = ""
    user_prompt: str = ""
    model: str = ""
    raw_llm_response: str = ""

    # Ölçüm — token sayımı, süre, USD maliyet
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    cost_is_estimate: bool = False
    cost_matched: str = ""

    def to_override(self) -> SpecOverride:
        return SpecOverride(
            n_storeys=int(self.n_storeys),
            n_rooms_per_floor=int(self.n_rooms_per_floor),
            layout=str(self.layout),
            storey_height=float(self.storey_height),
        )

    def design_summary(self) -> str:
        return (f"{self.n_storeys} kat · {self.n_rooms_per_floor} oda/kat · "
                f"{self.layout} · h={self.storey_height}m")

    def to_dict(self) -> dict:
        return {
            "n_storeys": self.n_storeys,
            "n_rooms_per_floor": self.n_rooms_per_floor,
            "layout": self.layout,
            "storey_height": self.storey_height,
            "rationale": self.rationale,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_llm_response": self.raw_llm_response,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_s": self.duration_s,
            "cost_usd": self.cost_usd,
            "cost_is_estimate": self.cost_is_estimate,
            "cost_matched": self.cost_matched,
        }


def plan_ana_baseline(user_template: str, *,
                      model: str = "gpt-4o", seed: int = 0) -> DesignPlan:
    """Ana baseline için tek LLM çağrısı.

    user_template: kullanıcının text_area'ya yazdığı bina tarifi
        (örn. "Küçük bir ofis binası, 2-3 oda + koridor, 100 m²").
    """
    sys = SYSTEM_PROMPT
    usr = (f"Bina tasarımı talebi (template):\n{user_template.strip()}\n\n"
           f"Tasarım tohumu: {seed}")
    return _call_llm(sys, usr, model)


def plan_variant_baseline(user_template: str, ana_plan: DesignPlan, *,
                          model: str = "gpt-4o", seed: int = 1) -> DesignPlan:
    """Ana baseline'a varyasyon olarak yeni baseline için LLM çağrısı."""
    ana_summary = json.dumps({
        "n_storeys": ana_plan.n_storeys,
        "n_rooms_per_floor": ana_plan.n_rooms_per_floor,
        "layout": ana_plan.layout,
        "storey_height": ana_plan.storey_height,
    }, ensure_ascii=False, indent=2)
    sys = SYSTEM_PROMPT + VARIATION_INSTRUCTION.format(
        ana_summary=ana_summary, seed=seed,
    )
    usr = (f"Bina tasarımı talebi (template):\n{user_template.strip()}\n\n"
           f"Bu, ana baseline'ın varyasyonu #{seed}'tir.")
    return _call_llm(sys, usr, model)


def _call_llm(system_prompt: str, user_prompt: str, model: str) -> DesignPlan:
    """OpenAI çağrısı + JSON parse. Token + süre + cost ölçer.

    Hata durumunda RuntimeError fırlatır (yine de süreyi ölçemediği için
    upstream catch eden taraf süre/maliyet için 0 alır).
    """
    import time
    from llm.pricing import estimate_cost
    from violation_pool.ifc_inject import _chat_with_retry

    t0 = time.time()
    resp = _chat_with_retry(
        model,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    duration = time.time() - t0
    content = resp.choices[0].message.content or "{}"

    # Token usage (OpenAI response.usage)
    usage = getattr(resp, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    tt = int(getattr(usage, "total_tokens", pt + ct) or (pt + ct))
    cost = estimate_cost(model, pt, ct)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM JSON döndürmedi: {e}. İçerik: {content[:200]}")

    return DesignPlan(
        n_storeys=int(data.get("n_storeys", 1)),
        n_rooms_per_floor=int(data.get("n_rooms_per_floor", 2)),
        layout=str(data.get("layout", "straight")),
        storey_height=float(data.get("storey_height", 3.0)),
        rationale=str(data.get("rationale", "")),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        raw_llm_response=content,
        prompt_tokens=pt,
        completion_tokens=ct,
        total_tokens=tt,
        duration_s=round(duration, 3),
        cost_usd=cost["total_usd"],
        cost_is_estimate=cost["is_estimate"],
        cost_matched=cost["matched"],
    )
