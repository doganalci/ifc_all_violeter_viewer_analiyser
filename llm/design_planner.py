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


SYSTEM_PROMPT = """Sen bir mimar / BIM tasarımcısısın. Kullanıcının istediği
binanın TÜM tasarım parametrelerini DETAYLI JSON olarak üretirsin. Bu plan
prosedürel motora verilecek ve geçerli IFC4 dosyasına çevrilecek.

NOT: Kullanıcı oda/salon/koridor sayısını ve layout'u zaten zorla belirledi
(prompt'taki "ZORUNLU KISITLAR" bloğuna bak). Sen sadece BOYUTLARI ve
RATIONALE'i belirle.

ZORUNLU JSON şeması (her alan zorunlu, ek alan ekleme):
{
  "storey_height":   <float 2.7-3.5>,        // metre, kat yüksekliği
  "wall_thickness":  <float 0.15-0.30>,      // metre, duvar kalınlığı
  "corridor": {
    "width":  <float 1.20-3.00>,             // ≥1.20 m mevzuat
    "length": <float 3.0-8.0>
  },
  "rooms": [                                  // tam (oda+salon) sayısı kadar
    {"role": "salon", "width": <float 2.5-7>, "length": <float 2.5-7>, "n_doors": <int 1-4>},
    {"role": "oda",   "width": <float 2.5-7>, "length": <float 2.5-7>, "n_doors": <int 1-4>}
  ],
  "interior_door": {
    "width":  <float 0.90-1.50>,             // ≥0.90 m mevzuat
    "height": <float 2.00-2.40>              // ≥2.00 m mevzuat
  },
  "entrance_door": {
    "width":  <float 0.90-1.50>,
    "height": <float 2.00-2.40>
  },
  "rationale": "<kısa Türkçe açıklama>"
}

KURALLAR:
- "rooms" listesi tam olarak (oda + salon) sayısında olmalı.
- İlk N tanesi role="salon" (N = kullanıcının verdiği salon sayısı), gerisi
  role="oda". Salonlar tipik olarak odalardan daha büyük (≥4 m en az).
- Her oda için "n_doors" kullanıcının verdiği [min, max] aralığında olmalı
  (constraint block'a bak). 1'inci kapı her zaman koridora bakar; ek kapılar
  dış cephe duvarlarına konur.
- Mevzuat eşikleri kesin: kapı ≥0.90 m, kapı yükseklik ≥2.00 m,
  koridor ≥1.20 m. Bunların altına inme.
- Sadece JSON döndür, açıklama veya kod bloğu yok.
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
    """LLM tasarım planı + kullanılan prompt'lar (şeffaflık için saklı).

    LLM artık tüm boyutları belirler (rooms, corridor, doors, wall thickness).
    """
    n_storeys: int
    n_rooms_per_floor: int
    layout: str
    storey_height: float
    rationale: str

    # LLM detayları — boyutlar
    wall_thickness: float = 0.20
    corridor_width: float = 1.50
    corridor_length: float = 4.0
    rooms: list[dict] = None         # [{"role":"salon","width":...,"length":...}, ...]
    interior_door_w: float = 0.95
    interior_door_h: float = 2.10
    entrance_door_w: float = 1.10
    entrance_door_h: float = 2.20

    # Şeffaflık
    system_prompt: str = ""
    user_prompt: str = ""
    model: str = ""
    raw_llm_response: str = ""

    # Ölçüm
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    cost_is_estimate: bool = False
    cost_matched: str = ""

    def __post_init__(self):
        if self.rooms is None:
            self.rooms = []

    def to_override(self) -> SpecOverride:
        return SpecOverride(
            n_storeys=int(self.n_storeys),
            n_rooms_per_floor=int(self.n_rooms_per_floor),
            layout=str(self.layout),
            storey_height=float(self.storey_height),
            wall_thickness=float(self.wall_thickness),
            corridor_width=float(self.corridor_width),
            corridor_length=float(self.corridor_length),
            rooms=list(self.rooms) if self.rooms else None,
            interior_door_w=float(self.interior_door_w),
            interior_door_h=float(self.interior_door_h),
            entrance_door_w=float(self.entrance_door_w),
            entrance_door_h=float(self.entrance_door_h),
        )

    def design_summary(self) -> str:
        return (
            f"{self.n_storeys} kat · {self.n_rooms_per_floor} oda/kat · "
            f"{self.layout} · h={self.storey_height}m · "
            f"koridor {self.corridor_width:.2f}×{self.corridor_length:.2f}m · "
            f"kapı {self.interior_door_w:.2f}×{self.interior_door_h:.2f}m"
        )

    def to_dict(self) -> dict:
        return {
            "n_storeys": self.n_storeys,
            "n_rooms_per_floor": self.n_rooms_per_floor,
            "layout": self.layout,
            "storey_height": self.storey_height,
            "wall_thickness": self.wall_thickness,
            "corridor_width": self.corridor_width,
            "corridor_length": self.corridor_length,
            "rooms": self.rooms,
            "interior_door_w": self.interior_door_w,
            "interior_door_h": self.interior_door_h,
            "entrance_door_w": self.entrance_door_w,
            "entrance_door_h": self.entrance_door_h,
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


def _build_constraints_block(constraints: dict | None) -> str:
    """Kullanıcının UI'dan verdiği zorunlu sayıları prompt'a struct'lı blok olarak çevir."""
    if not constraints:
        return ""
    parts = []
    n_oda = constraints.get("n_rooms")
    n_salon = constraints.get("n_salons")
    n_kor = constraints.get("n_corridors")
    n_kat = constraints.get("n_storeys")
    n_doors_min = constraints.get("n_doors_min")
    n_doors_max = constraints.get("n_doors_max")
    if n_kat is not None:
        parts.append(f"- Kat sayısı: {int(n_kat)}")
    if n_oda is not None:
        parts.append(f"- Oda sayısı (kat başına): {int(n_oda)}")
    if n_salon is not None:
        parts.append(f"- Salon sayısı (kat başına): {int(n_salon)}")
    if n_kor is not None:
        parts.append(f"- Koridor sayısı (kat başına): {int(n_kor)} "
                     f"({'straight' if int(n_kor) <= 1 else 'lshape'} layout demektir)")
    if n_doors_min is not None and n_doors_max is not None:
        parts.append(f"- Oda başına kapı sayısı: {int(n_doors_min)}-{int(n_doors_max)} arası "
                     f"(her oda için n_doors alanına kendi tercihin)")
    if not parts:
        return ""
    return (
        "\n\n⚠️ ZORUNLU KISITLAR (kullanıcı talebi, kesinlikle uy):\n"
        + "\n".join(parts)
        + "\n  → n_storeys = kullanıcının verdiği kat sayısı\n"
        + "  → n_rooms_per_floor = oda + salon toplamı\n"
        + "  → layout 'lshape' iff koridor sayısı 2; aksi 'straight'\n"
    )


def plan_ana_baseline(user_template: str, *,
                      constraints: dict | None = None,
                      model: str = "gpt-4o", seed: int = 0) -> DesignPlan:
    """Ana baseline için tek LLM çağrısı.

    Args:
        user_template: kullanıcının text_area'ya yazdığı bina tarifi.
        constraints: opsiyonel UI kısıtları {n_rooms, n_salons, n_corridors}.
            Prompt'a ZORUNLU blok olarak eklenir; baseline_pipeline ayrıca
            LLM çıktısını bu değerlere göre clamp eder.
    """
    sys = SYSTEM_PROMPT
    usr = (f"Bina tasarımı talebi (template):\n{user_template.strip()}\n\n"
           f"Tasarım tohumu: {seed}"
           f"{_build_constraints_block(constraints)}")
    return _call_llm(sys, usr, model)


def plan_variant_baseline(user_template: str, ana_plan: DesignPlan, *,
                          constraints: dict | None = None,
                          model: str = "gpt-4o", seed: int = 1) -> DesignPlan:
    """Ana baseline'a varyasyon olarak yeni baseline için LLM çağrısı.

    constraints verilirse varyantlarda da aynı ZORUNLU kısıt korunur
    (oda/salon/koridor sayıları sabit, varyasyon başka boyutlarda olur).
    """
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
           f"Bu, ana baseline'ın varyasyonu #{seed}'tir."
           f"{_build_constraints_block(constraints)}")
    return _call_llm(sys, usr, model)


def _call_llm(system_prompt: str, user_prompt: str, model: str) -> DesignPlan:
    """OpenAI çağrısı + JSON parse. Token + süre + cost ölçer.

    Hata durumunda RuntimeError fırlatır (yine de süreyi ölçemediği için
    upstream catch eden taraf süre/maliyet için 0 alır).

    Not: bazı modeller (gpt-5, o1 ailesi) custom temperature kabul etmiyor —
    sadece default (1). O hatada otomatik olarak temperature parametresini
    çıkarıp tekrar deniyoruz.
    """
    import time
    from llm.pricing import estimate_cost
    from violation_pool.ifc_inject import _chat_with_retry

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    response_format = {"type": "json_object"}

    t0 = time.time()
    try:
        resp = _chat_with_retry(
            model, messages,
            temperature=0.7,
            response_format=response_format,
        )
    except Exception as e:
        # gpt-5 / o1 ailesi: "does not support 'temperature'" — fallback
        msg = str(e).lower()
        if "temperature" in msg and ("unsupported" in msg or "does not support" in msg):
            resp = _chat_with_retry(
                model, messages,
                response_format=response_format,
            )
        else:
            raise
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

    # Yeni şema: corridor, rooms, interior_door, entrance_door
    cor = data.get("corridor") or {}
    int_door = data.get("interior_door") or {}
    ent_door = data.get("entrance_door") or {}
    rooms_raw = data.get("rooms") or []
    # Normalize: her oda için {role, width, length} olsun
    norm_rooms = []
    for r in rooms_raw:
        if isinstance(r, dict):
            norm_rooms.append({
                "role": str(r.get("role", "oda")),
                "width": float(r.get("width", r.get("w", 4.0))),
                "length": float(r.get("length", r.get("l", 4.0))),
                "n_doors": int(r.get("n_doors", 1) or 1),
            })

    return DesignPlan(
        # Üst seviye (constraints clamp eder)
        n_storeys=int(data.get("n_storeys", 1)),
        n_rooms_per_floor=int(data.get("n_rooms_per_floor", len(norm_rooms) or 2)),
        layout=str(data.get("layout", "straight")),
        storey_height=float(data.get("storey_height", 3.0)),
        # Boyutlar
        wall_thickness=float(data.get("wall_thickness", 0.20)),
        corridor_width=float(cor.get("width", 1.50)),
        corridor_length=float(cor.get("length", 4.0)),
        rooms=norm_rooms,
        interior_door_w=float(int_door.get("width", 0.95)),
        interior_door_h=float(int_door.get("height", 2.10)),
        entrance_door_w=float(ent_door.get("width", 1.10)),
        entrance_door_h=float(ent_door.get("height", 2.20)),
        rationale=str(data.get("rationale", "")),
        # Ölçüm
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
