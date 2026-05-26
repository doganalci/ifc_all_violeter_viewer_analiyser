"""Basic (kuralsal, LLM'siz) ihlal enjeksiyonu — sadece KAPI + KOLON.

Felsefe: Karmaşık LLM enjeksiyonu yerine deterministik, hızlı, sadece iki
tip ihlal üreten bir mod. Sentetik 2-oda+koridor baseline'larına uygulanır.

İKİ TİP:
  1. door_width — Bazı kapıların genişliğini değiştir.
     * < 0.90 m olursa → İHLAL (kapı dar, geçilemez)
     * ≥ 0.90 m olursa → İHLAL DEĞİL (hard negative: değişti ama uygun!)
       Model bunu 'değişti diye ihlal' sanmamalı.
  2. column_block — Kapıların önüne IfcColumn koy.
     * Kapıya yakın (< clearance) VE geçiş hattını kapatıyorsa → İHLAL
     * Uzakta VEYA yana kaçıksa → İHLAL DEĞİL (hard negative: kolon var
       ama geçişi engellemiyor)

Standart eşikler (TS 9111 / ADA, metre):
  * Kapı net genişliği ≥ 0.90 m
  * Kapı önü serbest yaklaşım ≥ 1.20 m (kolon bundan yakınsa ve hattı
    kapatıyorsa engel sayılır)

Birim: Sentetik baseline METRE kullanır; tüm hesap metre cinsinden.

Hiçbir başka tip (duvar/merdiven/rampa/korkuluk) işaretlenmez.
"""
from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import ifcopenshell
import ifcopenshell.util.placement


# --- Standart eşikler (METRE) ---------------------------------------------
MIN_DOOR_WIDTH_M = 0.90          # bunun altı ihlal
MIN_DOOR_HEIGHT_M = 2.00         # bunun altı ihlal (kapı net yüksekliği)
DOOR_CLEARANCE_M = 1.20          # kapı önü serbest mesafe
COLUMN_SIZE_M = 0.30             # kolon kenar uzunluğu


@dataclass
class BasicParams:
    # Kaç kapı değiştirilsin (oran)
    door_modify_ratio: float = 0.6
    # Değiştirilen kapıların kaçı ihlal (dar) olsun (gerisi compliant=hard negative)
    door_violation_ratio: float = 0.5
    # İhlalli dar kapı aralığı (m)
    narrow_min: float = 0.70
    narrow_max: float = 0.88
    # Uyumlu ama değiştirilmiş kapı aralığı (m) — hard negative
    compliant_min: float = 0.92
    compliant_max: float = 1.15
    # --- Kapı-odaklı deney: yükseklik ihlali + kolonları kapatma -----------
    # Sadece kapı ihlali üret (kolon enjeksiyonunu tamamen kapat)
    door_only: bool = False
    # Hangi boyut ihlal edilsin: "width" | "height" | "both"
    # "both" → her kapı için rastgele genişlik VEYA yükseklik
    door_dim_mode: str = "width"
    # İhlalli kısa kapı yüksekliği aralığı (m) — < 2.00
    short_min: float = 1.70
    short_max: float = 1.95
    # Uyumlu ama değiştirilmiş yükseklik aralığı (m) — hard negative, ≥ 2.00
    tall_min: float = 2.05
    tall_max: float = 2.30
    # Kaç kapıya kolon konsun (oran)
    column_ratio: float = 0.6
    # Konan kolonların kaçı engelleyici (yakın+hatta) olsun
    column_block_ratio: float = 0.5
    # Engelleyici kolon mesafesi (m) — kapı önünde, clearance içinde
    block_dist_min: float = 0.30
    block_dist_max: float = 0.90
    # Engellemeyen kolon mesafesi (m) — uzak
    far_dist_min: float = 1.60
    far_dist_max: float = 2.50
    # Engellemeyen kolon yan kaçıklığı (m) — kapı genişliği dışına
    side_offset: float = 1.20


def _door_world_xy(door) -> tuple[float, float]:
    m = ifcopenshell.util.placement.get_local_placement(door.ObjectPlacement)
    return float(m[0][3]), float(m[1][3])


def _door_facing(door) -> tuple[float, float]:
    """Kapının baktığı yön (placement X ekseni) — kabaca normal alıyoruz.

    Sentetik kapılar duvara dik yerleşmiş; placement matris X ekseni
    duvar boyu, Y ekseni normal olabilir. Basitlik: matris X yönünü
    'duvar boyu', ona dik (-y, x) yönünü 'geçiş normali' kabul et.
    """
    m = ifcopenshell.util.placement.get_local_placement(door.ObjectPlacement)
    ax = (float(m[0][0]), float(m[1][0]))   # local X yönü (duvar boyu)
    norm = math.hypot(*ax) or 1.0
    ax = (ax[0] / norm, ax[1] / norm)
    # geçiş normali = duvar boyuna dik
    return (-ax[1], ax[0])


def _gpt_plan(doors_info: list[dict], model: str, seed: int) -> dict | None:
    """LLM'den kapı+kolon değişiklik planı iste (çeşitlilik için).

    LLM SADECE plan önerir; ihlal etiketi yine kuralla ölçülür (ground
    truth dürüst kalır). Plan formatı:
      {"doors": {guid: new_width_m}, "columns": [{guid, distance_m, lateral_m}]}
    """
    try:
        from .ifc_inject import _chat_with_retry
        from ._json import _parse_json  # yoksa aşağıdaki fallback
    except Exception:
        from .ifc_inject import _chat_with_retry
        import json as _json
        def _parse_json(s):
            try:
                return _json.loads(s)
            except Exception:
                return {}
    sys_prompt = (
        "Sen bir erişilebilirlik test-senaryosu üreticisisin. Verilen "
        "kapıların bazılarının genişliğini değiştir ve bazı kapıların önüne "
        "kolon koy. ÇEŞİTLİLİK önemli: bazı değişiklikler standarda UYGUN "
        "kalsın (≥0.90 m kapı, ≥1.20 m kolon mesafesi), bazıları İHLAL olsun. "
        "Sadece JSON döndür:\n"
        '{"doors": {"<guid>": <yeni_genişlik_m>}, '
        '"columns": [{"guid": "<kapı_guid>", "distance_m": <m>, "lateral_m": <m>}]}'
    )
    user = "Kapılar:\n" + json.dumps(doors_info, ensure_ascii=False, indent=2)
    try:
        resp = _chat_with_retry(
            model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}],
            temperature=0.7, response_format={"type": "json_object"},
        )
        return _parse_json(resp.choices[0].message.content or "")
    except Exception as e:
        print(f"[basic_inject] GPT plan hatası: {e}")
        return None


def inject_basic(baseline_ifc_path: str | Path, out_ifc_path: str | Path,
                 *, seed: int = 0, params: BasicParams | None = None,
                 use_gpt: bool = False, model: str | None = None) -> dict:
    """Baseline IFC'ye kapı+kolon ihlali enjekte et.

    use_gpt=False (varsayılan): kuralsal, deterministik, hızlı.
    use_gpt=True: LLM plan önerir (çeşitlilik), etiket yine kuralla ölçülür.

    Returns:
        { "ifc_path", "labels": [...], "summary": {...} }
    """
    p = params or BasicParams()
    rng = random.Random(seed)
    f = ifcopenshell.open(str(baseline_ifc_path))

    owner = f.by_type("IfcOwnerHistory")[0] if f.by_type("IfcOwnerHistory") else None
    body_ctx = None
    for ctx in f.by_type("IfcGeometricRepresentationSubContext"):
        if ctx.ContextIdentifier == "Body":
            body_ctx = ctx
            break
    if body_ctx is None:
        subs = f.by_type("IfcGeometricRepresentationSubContext")
        body_ctx = subs[0] if subs else (f.by_type("IfcGeometricRepresentationContext") or [None])[0]
    storey = (f.by_type("IfcBuildingStorey") or [None])[0]

    doors = list(f.by_type("IfcDoor"))
    labels: list[dict] = []

    # GPT modunda planı al
    gpt = None
    if use_gpt and doors:
        info = [{"guid": d.GlobalId,
                 "width_m": round(float(getattr(d, "OverallWidth", 0.9) or 0.9), 3),
                 "name": getattr(d, "Name", "")}
                for d in doors]
        gpt = _gpt_plan(info, model or "gpt-4o-mini", seed)

    # --- 1) KAPI BOYUTU (genişlik ve/veya yükseklik) -------------------
    # _door_plan: guid -> (attribute, new_value)
    if gpt and isinstance(gpt.get("doors"), dict):
        # GPT planı: hangi kapı hangi genişlik (GPT modu sadece genişlik)
        gpt_doors = gpt["doors"]
        mod_doors = [d for d in doors if d.GlobalId in gpt_doors]
        _door_plan = {d.GlobalId: ("OverallWidth", float(gpt_doors[d.GlobalId]))
                      for d in mod_doors}
    else:
        n_mod = int(round(len(doors) * p.door_modify_ratio))
        mod_doors = rng.sample(doors, min(n_mod, len(doors))) if doors else []
        _door_plan = {}
        for d in mod_doors:
            # Hangi boyut? width / height / both(rastgele)
            if p.door_dim_mode == "height":
                dim = "height"
            elif p.door_dim_mode == "both":
                dim = "height" if rng.random() < 0.5 else "width"
            else:
                dim = "width"
            is_vio = rng.random() < p.door_violation_ratio
            if dim == "height":
                val = (rng.uniform(p.short_min, p.short_max) if is_vio
                       else rng.uniform(p.tall_min, p.tall_max))
                _door_plan[d.GlobalId] = ("OverallHeight", round(val, 3))
            else:
                val = (rng.uniform(p.narrow_min, p.narrow_max) if is_vio
                       else rng.uniform(p.compliant_min, p.compliant_max))
                _door_plan[d.GlobalId] = ("OverallWidth", round(val, 3))

    for d in mod_doors:
        attr, new_v = _door_plan.get(d.GlobalId, ("OverallWidth", None))
        if new_v is None:
            continue
        before = float(getattr(d, attr, 0.0) or 0.0)
        new_v = round(float(new_v), 3)
        setattr(d, attr, new_v)
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
            "category": "Kapı/Koridor",
            "severity": "kritik" if is_vio else "uygun",
            "is_violation": bool(is_vio),
            "attribute": attr,
            "before": round(before, 3),
            "after": new_v,
            "rule": rule,
            "evidence": ev,
        })

    # --- 2) KOLON YERLEŞTİRME (door_only ise atla) ---------------------
    if not p.door_only and storey is not None and body_ctx is not None:
        from .ifc_inject import _add_column_obstruction
        door_by_guid = {d.GlobalId: d for d in doors}
        # GPT planı varsa onu kullan; yoksa kuralsal
        col_specs: list[tuple] = []   # (door, dist, lateral)
        if gpt and isinstance(gpt.get("columns"), list):
            for c in gpt["columns"]:
                d = door_by_guid.get(c.get("guid"))
                if d is None:
                    continue
                try:
                    col_specs.append((d, float(c.get("distance_m", 0.5)),
                                      float(c.get("lateral_m", 0.0))))
                except (TypeError, ValueError):
                    continue
        else:
            n_col = int(round(len(doors) * p.column_ratio))
            col_doors = rng.sample(doors, min(n_col, len(doors))) if doors else []
            for d in col_doors:
                if rng.random() < p.column_block_ratio:
                    dist = rng.uniform(p.block_dist_min, p.block_dist_max)
                    lateral = rng.uniform(-0.15, 0.15)
                else:
                    if rng.random() < 0.5:
                        dist = rng.uniform(p.far_dist_min, p.far_dist_max)
                        lateral = rng.uniform(-0.15, 0.15)
                    else:
                        dist = rng.uniform(p.block_dist_min, p.block_dist_max)
                        lateral = (1 if rng.random() < 0.5 else -1) * p.side_offset
                col_specs.append((d, dist, lateral))

        for d, dist, lateral in col_specs:
            dx, dy = _door_world_xy(d)
            nx, ny = _door_facing(d)
            lx, ly = -ny, nx
            cx = dx + nx * dist + lx * lateral
            cy = dy + ny * dist + ly * lateral
            col = _add_column_obstruction(
                f, owner, body_ctx, storey, cx, cy,
                size_x=COLUMN_SIZE_M, size_y=COLUMN_SIZE_M,
                name=f"Column near {d.GlobalId[:8]}",
            )
            door_w = float(getattr(d, "OverallWidth", 0.9) or 0.9)
            # Engel kuralı: clearance içinde VE yan kaçıklık geçiş hattında
            in_front = dist < DOOR_CLEARANCE_M
            in_path = abs(lateral) < (door_w / 2 + COLUMN_SIZE_M / 2)
            is_vio = in_front and in_path
            labels.append({
                "ifc_global_id": col.GlobalId,
                "category": "Kapı/Koridor",
                "severity": "kritik" if is_vio else "uygun",
                "is_violation": bool(is_vio),
                "attribute": "[ADDED_COLUMN]",
                "before": None,
                "after": f"dist={dist:.2f}m lateral={lateral:.2f}m near {d.GlobalId[:8]}",
                "rule": f"kolon kapı önü ≥ {DOOR_CLEARANCE_M} m VEYA hattın dışında",
                "evidence": (f"Kolon kapıya {dist*100:.0f} cm, yan {abs(lateral)*100:.0f} cm "
                             + ("→ geçişi engelliyor → İHLAL" if is_vio
                                else "→ geçiş açık → uygun")),
            })

    out = Path(out_ifc_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(out))

    n_vio = sum(1 for l in labels if l["is_violation"])
    n_neg = len(labels) - n_vio
    return {
        "ifc_path": str(out),
        "labels": labels,
        "summary": {
            "n_doors": len(doors),
            "n_modified_doors": len(mod_doors),
            "n_columns": sum(1 for l in labels if l["attribute"] == "[ADDED_COLUMN]"),
            "n_violations": n_vio,
            "n_compliant_changes": n_neg,    # hard negatives
        },
    }


def _labels_to_doc(ifc_id: str, baseline_id: str, labels: list[dict]) -> dict:
    """Eğitim pipeline'ının okuyacağı labels.json formatı.

    İhlal (is_violation=True) → status='applied' (y=1).
    Uyumlu değişiklik (is_violation=False) → status='compliant' (y=0,
    ama node grafikte değişmiş geometriyle var = hard negative).
    """
    out_labels = []
    for l in labels:
        out_labels.append({
            "ifc_global_id": l["ifc_global_id"],
            "category": l["category"],
            "severity": l["severity"],
            "status": "applied" if l["is_violation"] else "compliant",
            "is_decoy": False,
            "attribute": l["attribute"],
            "before": l.get("before"),
            "after": l.get("after"),
            "rule": l.get("rule"),
            "evidence": l.get("evidence"),
        })
    return {
        "violated_id": ifc_id,
        "baseline_id": baseline_id,
        "source": "basic_inject",
        "labels": out_labels,
    }


def run_basic_batch(dataset_tag: str, *, variants: int = 5, seed_start: int = 1000,
                    params: BasicParams | None = None,
                    register_in_db: bool = True, progress_cb=None,
                    use_gpt: bool = False, model: str | None = None,
                    full_label: bool = False,
                    method_label: str = "basicinj") -> dict:
    """dataset_tag'li tüm baseline'lara basic ihlal enjekte et (LLM'siz).

    Her baseline için `variants` adet farklı varyant üretir.
    """
    from . import storage, ifc_graph
    from .config import settings

    # tag'li baseline'ları bul
    baseline_ids = storage.ifc_ids_for_tags([dataset_tag], kind="baseline")
    baselines = [storage.get_ifc_model(i) for i in baseline_ids]
    baselines = [b for b in baselines if b and b.get("status") == "ok"]
    if not baselines:
        raise RuntimeError(f"'{dataset_tag}' etiketli baseline bulunamadı.")

    out_dir = settings.ifc_dir / "violated"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"ok": 0, "err": 0, "violations": 0, "hard_negatives": 0, "items": []}
    total = len(baselines) * variants
    done = 0

    _mlabel = "".join(c if c.isalnum() or c in "-_" else "_"
                      for c in str(method_label).strip()) or "inj"
    for b in baselines:
        base_stem = Path(b["file_path"]).stem
        # İsim: <yöntem>_<baseline>_violatedN → nasıl üretildiği belli
        name_base = f"{_mlabel}_{base_stem}"
        for vi in range(variants):
            seed = seed_start + done
            out_id = str(uuid.uuid4())
            # İlk boş _violatedN
            n = 1
            while (out_dir / f"{name_base}_violated{n}.ifc").exists():
                n += 1
            stem = f"{name_base}_violated{n}"
            if (out_dir / f"{stem}.ifc").exists():
                stem = f"{name_base}_violated{n}_{out_id[:6]}"
            out_ifc = out_dir / f"{stem}.ifc"
            try:
                r = inject_basic(b["file_path"], out_ifc, seed=seed, params=params,
                                 use_gpt=use_gpt, model=model)
                # labels.json
                doc = _labels_to_doc(out_id, b["id"], r["labels"])
                lab_path = out_dir / f"{stem}.labels.json"
                lab_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                meta_path = out_dir / f"{stem}.meta.json"
                meta_path.write_text(json.dumps({
                    "ifc_id": out_id, "kind": "violated",
                    "baseline_id": b["id"], "source": "basic_inject",
                    "seed": seed, "summary": r["summary"],
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                # graph
                graph_path = None
                try:
                    gp = out_dir / f"{stem}.graph.json"
                    ifc_graph.build_and_save(str(out_ifc), str(gp))
                    graph_path = str(gp)
                except Exception as _ge:
                    graph_path = None
                    results.setdefault("graph_errors", []).append(
                        f"{stem}: {_ge}")

                # TAM ETİKETLEME: graph'taki HER node'a açık etiket.
                # İhlal/hard-negatif zaten doc'ta; geri kalan tüm node'lar
                # 'clean' (y=0). Baseline garantili temiz olduğundan güvenli.
                if full_label and graph_path:
                    try:
                        from ml.data.graph_loader import load_graph as _lg
                        g = _lg(graph_path)
                        labeled = {l["ifc_global_id"] for l in doc["labels"]}
                        n_clean = 0
                        for nid in g.nodes():
                            if nid in labeled:
                                continue
                            doc["labels"].append({
                                "ifc_global_id": nid, "category": "",
                                "severity": "uygun", "status": "clean",
                                "is_decoy": False, "attribute": None,
                                "before": None, "after": None,
                                "evidence": "Baseline temiz — kesin ihlal değil",
                            })
                            n_clean += 1
                        lab_path.write_text(json.dumps(doc, indent=2,
                                            ensure_ascii=False), encoding="utf-8")
                        results["clean_labeled"] = results.get("clean_labeled", 0) + n_clean
                    except Exception as _le:
                        results.setdefault("label_errors", []).append(f"{stem}: {_le}")
                # DB — sadece ihlal varsa 'ok', yoksa yine 'ok' (negatifler de
                # değerli, hard negative). status='ok' eğitim listesine girsin.
                if register_in_db:
                    mid = storage.create_ifc_model(
                        id=out_id, kind="violated",
                        name=f"{base_stem}.violated", parent_id=b["id"],
                        llm_model="basic_inject", prompt=None, pool_run_id=None,
                        params={"summary": r["summary"], "seed": seed},
                        file_path=str(out_ifc), meta_path=str(meta_path),
                        labels_path=str(lab_path), graph_path=graph_path,
                        status="ok", error=None,
                        dataset_tag=b.get("dataset_tag") or dataset_tag,
                    )
                    # ihlal etiketlerini DB'ye de yaz (görselleştirme için)
                    storage.add_ifc_labels(mid, [
                        {**lab} for lab in doc["labels"]
                    ])
                results["ok"] += 1
                if graph_path:
                    results["with_graph"] = results.get("with_graph", 0) + 1
                results["violations"] += r["summary"]["n_violations"]
                results["hard_negatives"] += r["summary"]["n_compliant_changes"]
                results["items"].append({"stem": stem, **r["summary"]})
            except Exception as e:
                results["err"] += 1
                results["items"].append({"stem": stem, "error": str(e)})
            done += 1
            if progress_cb:
                progress_cb(done, total, stem)
    # Veri kümesi defterini (data klasörü/datasets.xlsx) güncelle — best-effort.
    if register_in_db:
        try:
            from ml.tracking import rebuild_dataset_registry
            from paths import data_home
            rebuild_dataset_registry(str(data_home()))
        except Exception:
            pass
    return results
