"""Tamamen LLM-bazlı IFC üretici (deneysel, başarı oranı düşük).

LLM doğrudan IFC4 (.ifc) text'i üretir. Prosedürel motor yok.

Amaç: pure-LLM IFC üretiminin sınırlarını gösterme + dataset çeşitlilik
için kullanılabilirliğini test etme.

Tipik sonuçlar (gözlemsel):
  * gpt-4o ile: %20-40 invalid IFC (parse hatası)
  * gpt-4o ile başarılı dosyalar bile çoğunlukla geometri eksik / bozuk
  * gpt-5 ile: %40-70 başarı (umut)
  * Token maliyeti yüksek (~6000-10000 token/IFC)
  * Süre yüksek (10-30s/IFC)

Sonuç: tek başına baseline üretim için **mantıklı değil**; çeşitli ve
"sıra dışı" örnekler için yan kaynak olarak değerlendirilebilir.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


SYSTEM_PROMPT = """Sen bir BIM/IFC4 uzmanısın. Kullanıcının istediği binayı
**doğrudan IFC4 dosyası** olarak üretirsin. Çıktın geçerli ISO-10303-21
formatında olmalı.

GEREKLİ BÖLÜMLER (sırasıyla):
1. ISO-10303-21; HEADER; ... ENDSEC;
2. DATA;
   - IfcProject, IfcSite, IfcBuilding, IfcBuildingStorey (mekânsal yapı)
   - IfcUnitAssignment, IfcGeometricRepresentationContext (birim/bağlam)
   - IfcLocalPlacement + IfcCartesianPoint + IfcAxis2Placement3D (konumlandırma)
   - IfcWallStandardCase (her duvar bir entity)
   - IfcSpace (her oda)
   - IfcDoor, IfcWindow (açıklıklar; OverallWidth, OverallHeight ile)
   - IfcRelAggregates (Project→Site→Building→Storey ve Storey→Space)
   - IfcRelContainedInSpatialStructure (duvarlar+kapılar Storey'e)
3. ENDSEC; END-ISO-10303-21;

KURALLAR:
- Her #N referansı tutarlı olmalı (kullandığın her id'yi tanımla).
- GUID'ler 22 karakter base64 (örn '1xS_OYx5n0OuPxHRf3eAaB').
- Tüm koordinatlar IfcCartesianPoint olarak. Birim METRE.
- Bina küçük: 2-4 oda + 1 koridor, 1 kat.
- Mevzuata uygun: kapı genişlik ≥0.90 m, kapı yükseklik ≥2.00 m,
  koridor ≥1.20 m.

ÇIKTI: SADECE .ifc içeriği. Açıklama veya kod bloğu (```) yok.
İlk satır "ISO-10303-21;" olmalı, son satır "END-ISO-10303-21;".
"""


@dataclass
class PureLLMResult:
    """Pure-LLM IFC üretimi sonucu."""
    ifc_path: str | None
    valid: bool
    parse_error: str | None
    raw_response: str
    model: str
    user_prompt: str
    prompt_tokens: int
    completion_tokens: int
    duration_s: float
    cost_usd: float
    cost_is_estimate: bool
    cost_matched: str
    # Parse edilebilirse geometri sayıları
    n_doors: int = 0
    n_walls: int = 0
    n_spaces: int = 0
    n_windows: int = 0


def _strip_code_block(text: str) -> str:
    """LLM '```ifc\\n...\\n```' ile sararsa temizle."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines[1:])
    return t


def generate_pure_llm_ifc(user_prompt: str, out_path: Path,
                          model: str = "gpt-5") -> PureLLMResult:
    """LLM'den ham IFC text'i alıp dosyaya yaz + parse kontrolü.

    Hatalar (LLM çağrısı, JSON parse vs.) yukarı fırlar; geçersiz IFC
    (parse hatası) ise valid=False + parse_error ile dönülür (rapor için).
    """
    from llm.pricing import estimate_cost
    from violation_pool.ifc_inject import _chat_with_retry

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.time()
    try:
        resp = _chat_with_retry(model, messages, temperature=0.2)
    except Exception as e:
        msg = str(e).lower()
        if "temperature" in msg and ("unsupported" in msg
                                      or "does not support" in msg):
            resp = _chat_with_retry(model, messages)
        else:
            raise
    duration = time.time() - t0

    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = estimate_cost(model, pt, ct)

    # Kod bloğunu temizle ve dosyaya yaz (parse başarısız olsa da kalıcı)
    text = _strip_code_block(content)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    valid = False
    parse_err: str | None = None
    n_doors = n_walls = n_spaces = n_windows = 0
    try:
        import ifcopenshell
        f = ifcopenshell.open(str(out_path))
        valid = True
        n_doors = len(f.by_type("IfcDoor"))
        n_walls = (len(f.by_type("IfcWall"))
                   + len(f.by_type("IfcWallStandardCase")))
        n_spaces = len(f.by_type("IfcSpace"))
        n_windows = len(f.by_type("IfcWindow"))
        # Tamamen boşsa "valid ama anlamsız" — geçersiz say
        if n_walls == 0 and n_spaces == 0 and n_doors == 0:
            valid = False
            parse_err = "Parse OK ama hiç IfcWall/IfcSpace/IfcDoor yok"
    except Exception as e:
        parse_err = str(e)[:300]

    return PureLLMResult(
        ifc_path=str(out_path) if valid else None,
        valid=valid,
        parse_error=parse_err,
        raw_response=content,
        model=model,
        user_prompt=user_prompt,
        prompt_tokens=pt,
        completion_tokens=ct,
        duration_s=round(duration, 2),
        cost_usd=cost["total_usd"],
        cost_is_estimate=cost["is_estimate"],
        cost_matched=cost["matched"],
        n_doors=n_doors, n_walls=n_walls, n_spaces=n_spaces, n_windows=n_windows,
    )
