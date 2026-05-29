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
                          model: str = "gpt-5",
                          constraints: dict | None = None) -> PureLLMResult:
    """LLM'den ham IFC text'i alıp dosyaya yaz + parse kontrolü.

    Args:
        user_prompt: kullanıcının bina tarifi.
        out_path: çıktı .ifc dosya yolu.
        model: OpenAI model adı.
        constraints: opsiyonel UI kısıtları {n_rooms, n_salons, n_corridors,
            n_storeys, n_doors_min/max, prefs} — design_planner ile aynı
            şema. Prompt'a ZORUNLU + TERCİH bloğu olarak eklenir.

    Hatalar (LLM çağrısı vs.) yukarı fırlar; geçersiz IFC (parse hatası)
    valid=False + parse_error ile dönülür.
    """
    from llm.design_planner import _build_constraints_block
    from llm.pricing import estimate_cost
    from violation_pool.ifc_inject import _chat_with_retry

    # Constraints'i prompt'a ekle
    constraints_block = _build_constraints_block(constraints)
    full_user_prompt = user_prompt.strip() + constraints_block

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_user_prompt},
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
        # NOT: parse OK ama geometri boş ise yine de valid sayıyoruz —
        # kullanıcı "LLM ne üretti" görmek istiyor; sadece uyarı veriyoruz.
        if n_walls == 0 and n_spaces == 0 and n_doors == 0:
            parse_err = ("Parse OK ama hiç IfcWall/IfcSpace/IfcDoor yok "
                         "(geometrisi eksik IFC)")
    except Exception as e:
        parse_err = f"ifcopenshell parse hatası: {str(e)[:280]}"

    return PureLLMResult(
        ifc_path=str(out_path) if valid else None,
        valid=valid,
        parse_error=parse_err,
        raw_response=content,
        model=model,
        user_prompt=full_user_prompt,
        prompt_tokens=pt,
        completion_tokens=ct,
        duration_s=round(duration, 2),
        cost_usd=cost["total_usd"],
        cost_is_estimate=cost["is_estimate"],
        cost_matched=cost["matched"],
        n_doors=n_doors, n_walls=n_walls, n_spaces=n_spaces, n_windows=n_windows,
    )


def register_pure_llm_baseline(result: PureLLMResult, *,
                                dataset_tag: str,
                                user_prompt: str,
                                constraints: dict | None = None
                                ) -> tuple[str | None, str | None]:
    """Geçerli pure-LLM IFC'sini codex1 storage'a baseline olarak yaz.

    Geçersizse (ifc_path None) (None, sebep) döner.
    Aksi halde (ifc_id, None) — DB kaydı başarılı.
    Hata varsa (None, hata_mesajı).

    Sayfa 15 (görüntüleyici) üretilen IFC'leri normal baseline olarak görür.
    """
    if not result.ifc_path:
        return None, "ifc_path yok (parse hatası)"
    import json
    import uuid as _uuid
    try:
        from violation_pool import ifc_graph, storage
    except Exception as e:
        return None, f"DB modülleri yok: {e}"

    ifc_id = str(_uuid.uuid4())
    ifc_p = Path(result.ifc_path)
    stem = ifc_p.parent / ifc_p.stem
    graph_warning = ""

    # Graph üretmeye çalış (başarısızsa geç — DB kaydı yine yapılır)
    graph_path = stem.parent / f"{stem.name}.graph.json"
    try:
        ifc_graph.build_and_save(str(ifc_p), str(graph_path))
        graph_path_str = str(graph_path)
    except Exception as e:
        graph_warning = f" (graph üretilemedi: {str(e)[:80]})"
        graph_path_str = None

    # Meta + boş labels
    meta = {
        "ifc_id": ifc_id, "kind": "baseline",
        "source": "pure_llm", "model": result.model,
        "n_walls": result.n_walls, "n_doors": result.n_doors,
        "n_spaces": result.n_spaces, "n_windows": result.n_windows,
        "duration_s": result.duration_s, "cost_usd": result.cost_usd,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.prompt_tokens + result.completion_tokens,
    }
    meta_path = stem.parent / f"{stem.name}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    labels_path = stem.parent / f"{stem.name}.labels.json"
    labels_path.write_text(json.dumps({
        "violated_id": ifc_id, "labels": [],
        "_meta": "pure_llm baseline — no violations by design",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        storage.create_ifc_model(
            id=ifc_id, kind="baseline",
            name=stem.name,
            parent_id=None,            # pure_llm'de hiyerarşi yok
            pool_run_id=None,
            params={
                "kind_label": "pure_llm_baseline",
                "model": result.model,
                "constraints": constraints or {},
                "geometry": {
                    "n_walls": result.n_walls, "n_doors": result.n_doors,
                    "n_spaces": result.n_spaces,
                    "n_windows": result.n_windows,
                },
                "llm_measurement": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "duration_s": result.duration_s,
                    "cost_usd": result.cost_usd,
                },
            },
            file_path=str(ifc_p),
            meta_path=str(meta_path),
            labels_path=str(labels_path),
            graph_path=graph_path_str,
            llm_model=result.model,
            prompt=user_prompt,
            status="ok", error=None,
            dataset_tag=dataset_tag,
        )
    except Exception as e:
        return None, f"DB kaydı başarısız: {str(e)[:200]}"

    # Defteri yenile (best-effort)
    try:
        from ml.tracking import rebuild_dataset_registry
        from paths import data_home
        rebuild_dataset_registry(str(data_home()))
    except Exception:
        pass
    return ifc_id, (graph_warning.strip() if graph_warning else None)
