"""LLM ile kapı ihlal/uyumlu değişiklik enjeksiyonu.

LLM (gpt-4o vb.) baseline IFC'deki kapılara genişlik veya yükseklik değişikliği
önerir. Bazıları ihlal (eşik altı), bazıları hard-negative (uyumlu değişiklik).
Etiket KURALLA ölçülür → ground truth dürüst kalır.

Eşikler (TS 9111 / ADA):
  * Kapı net genişliği ≥ 0.90 m  → uygun; altı ihlal.
  * Kapı net yüksekliği ≥ 2.00 m → uygun; altı ihlal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ifcopenshell


# Kural eşikleri — etiket ölçümünde kullanılır
MIN_DOOR_WIDTH_M = 0.90
MIN_DOOR_HEIGHT_M = 2.00


@dataclass
class LLMDoorParams:
    target_n_violations: int = 2   # her IFC için hedef ihlal sayısı (LLM'e hint)
    target_n_compliant: int = 2    # uyumlu (hard-negative) değişiklik hedefi
    temperature: float = 0.7
    max_doors: int = 20            # IFC başına LLM'e gönderilecek maks kapı


_SYSTEM_PROMPT = """Sen bir bina erişilebilirliği test-senaryosu üreticisisin.
Verilen kapı listesindeki bazı kapıların GENİŞLİĞİNİ veya YÜKSEKLİĞİNİ değiştirip
gerçekçi test senaryoları üretirsin.

Standart eşikler (TS 9111 / ADA):
  * Kapı net GENİŞLİĞİ ≥ 0.90 m → uygun; altı İHLAL.
  * Kapı net YÜKSEKLİĞİ ≥ 2.00 m → uygun; altı İHLAL.

Hedef: yaklaşık {n_vio} İHLAL üreten ve {n_comp} UYUMLU değişiklik üreten plan
(uyumlu = değer değişti ama eşik üstünde kaldı = hard-negative).

Çeşitlilik için karışım kullan:
  * Bazı genişlik ihlali (ör. 0.70-0.85 m)
  * Bazı yükseklik ihlali (ör. 1.70-1.95 m)
  * Bazı uyumlu genişlik değişimi (ör. 0.92-1.20 m)
  * Bazı uyumlu yükseklik değişimi (ör. 2.05-2.30 m)

Sadece JSON döndür (başka açıklama yok):
{{
  "changes": [
    {{
      "guid": "<kapı_guid>",
      "attribute": "OverallWidth" | "OverallHeight",
      "new_value": <metre cinsinden float>,
      "intended": "violation" | "compliant",
      "rationale": "<kısa, Türkçe>"
    }}
  ]
}}
"""


def _ask_llm(doors_info: list[dict], model: str, params: LLMDoorParams,
             seed: int) -> dict:
    """LLM'den plan al. Hata olursa boş plan + hata mesajı."""
    from violation_pool.ifc_inject import _chat_with_retry
    sys_prompt = _SYSTEM_PROMPT.format(
        n_vio=params.target_n_violations,
        n_comp=params.target_n_compliant,
    )
    user = "Kapılar:\n" + json.dumps(doors_info, ensure_ascii=False, indent=2)
    try:
        resp = _chat_with_retry(
            model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}],
            temperature=params.temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"changes": [], "_error": f"json parse: {content[:120]}"}
    except Exception as e:
        return {"changes": [], "_error": str(e)}


def inject_doors_llm(baseline_ifc_path: str | Path,
                     out_ifc_path: str | Path, *,
                     seed: int = 0,
                     params: LLMDoorParams | None = None,
                     model: str = "gpt-4o") -> dict:
    """LLM ile kapı boyut ihlali enjekte et.

    Returns:
        { "ifc_path", "labels": [...], "summary": {...} }
    """
    p = params or LLMDoorParams()
    f = ifcopenshell.open(str(baseline_ifc_path))
    doors_all = list(f.by_type("IfcDoor"))
    doors = doors_all[: p.max_doors]
    out = Path(out_ifc_path); out.parent.mkdir(parents=True, exist_ok=True)

    if not doors:
        f.write(str(out))
        return {"ifc_path": str(out), "labels": [],
                "summary": {"n_doors": 0, "n_modified": 0,
                            "n_violations": 0, "n_compliant_changes": 0,
                            "llm_model": model, "llm_error": "no_doors"}}

    door_info = [{
        "guid": d.GlobalId,
        "name": getattr(d, "Name", "") or "",
        "width_m": round(float(getattr(d, "OverallWidth", 0) or 0), 3),
        "height_m": round(float(getattr(d, "OverallHeight", 0) or 0), 3),
    } for d in doors]

    plan = _ask_llm(door_info, model, p, seed)
    door_by_guid = {d.GlobalId: d for d in doors}
    labels: list[dict] = []
    seen_guids: set[str] = set()
    for ch in plan.get("changes", []) or []:
        try:
            guid = ch.get("guid")
            if not guid or guid in seen_guids:
                continue
            d = door_by_guid.get(guid)
            if d is None:
                continue
            attr = ch.get("attribute", "OverallWidth")
            if attr not in ("OverallWidth", "OverallHeight"):
                continue
            new_v = float(ch.get("new_value"))
            if not (0.30 <= new_v <= 3.50):     # akıl sağlığı sınırı
                continue
            before = float(getattr(d, attr, 0.0) or 0.0)
            new_v = round(new_v, 3)
            setattr(d, attr, new_v)
            seen_guids.add(guid)
            if attr == "OverallHeight":
                is_vio = new_v < MIN_DOOR_HEIGHT_M
                rule = f"OverallHeight >= {MIN_DOOR_HEIGHT_M} m"
                ev = (f"Kapı yüksekliği {new_v*100:.0f} cm "
                      f"({'< 200 → İHLAL' if is_vio else '≥ 200 → uygun'})")
            else:
                is_vio = new_v < MIN_DOOR_WIDTH_M
                rule = f"OverallWidth >= {MIN_DOOR_WIDTH_M} m"
                ev = (f"Kapı genişliği {new_v*100:.0f} cm "
                      f"({'< 90 → İHLAL' if is_vio else '≥ 90 → uygun'})")
            labels.append({
                "ifc_global_id": d.GlobalId,
                "category": "Kapı/LLM",
                "severity": "kritik" if is_vio else "uygun",
                "is_violation": bool(is_vio),
                "attribute": attr,
                "before": round(before, 3),
                "after": new_v,
                "rule": rule,
                "evidence": ev,
                "llm_rationale": str(ch.get("rationale", "")),
                "llm_intended": str(ch.get("intended", "")),
            })
        except (TypeError, ValueError, KeyError):
            continue

    f.write(str(out))
    n_vio = sum(1 for l in labels if l["is_violation"])
    return {
        "ifc_path": str(out),
        "labels": labels,
        "summary": {
            "n_doors": len(doors),
            "n_modified": len(labels),
            "n_violations": n_vio,
            "n_compliant_changes": len(labels) - n_vio,
            "llm_model": model,
            "llm_error": plan.get("_error"),
        },
    }
