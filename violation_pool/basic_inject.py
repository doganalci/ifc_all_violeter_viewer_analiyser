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


def inject_basic(baseline_ifc_path: str | Path, out_ifc_path: str | Path,
                 *, seed: int = 0, params: BasicParams | None = None) -> dict:
    """Baseline IFC'ye kuralsal kapı+kolon ihlali enjekte et.

    Returns:
        { "ifc_path", "labels": [ {ifc_global_id, category, severity,
          is_violation, attribute, before, after, rule, evidence}, ... ],
          "summary": {...} }
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

    # --- 1) KAPI GENİŞLİĞİ ---------------------------------------------
    n_mod = int(round(len(doors) * p.door_modify_ratio))
    mod_doors = rng.sample(doors, min(n_mod, len(doors))) if doors else []
    for d in mod_doors:
        before = float(getattr(d, "OverallWidth", 0.0) or 0.0)
        make_violation = rng.random() < p.door_violation_ratio
        if make_violation:
            new_w = round(rng.uniform(p.narrow_min, p.narrow_max), 3)
        else:
            new_w = round(rng.uniform(p.compliant_min, p.compliant_max), 3)
        d.OverallWidth = new_w
        is_vio = new_w < MIN_DOOR_WIDTH_M
        labels.append({
            "ifc_global_id": d.GlobalId,
            "category": "Kapı/Koridor",
            "severity": "kritik" if is_vio else "uygun",
            "is_violation": bool(is_vio),
            "attribute": "OverallWidth",
            "before": round(before, 3),
            "after": new_w,
            "rule": f"OverallWidth >= {MIN_DOOR_WIDTH_M} m",
            "evidence": f"Kapı genişliği {new_w*100:.0f} cm "
                        f"({'< 90 → İHLAL' if is_vio else '≥ 90 → uygun'})",
        })

    # --- 2) KOLON YERLEŞTİRME ------------------------------------------
    if storey is not None and body_ctx is not None:
        from .ifc_inject import _add_column_obstruction
        n_col = int(round(len(doors) * p.column_ratio))
        col_doors = rng.sample(doors, min(n_col, len(doors))) if doors else []
        for d in col_doors:
            dx, dy = _door_world_xy(d)
            nx, ny = _door_facing(d)
            blocking = rng.random() < p.column_block_ratio
            if blocking:
                dist = rng.uniform(p.block_dist_min, p.block_dist_max)
                lateral = rng.uniform(-0.15, 0.15)   # kapı hattında
            else:
                # ya uzak ya da yana kaçık
                if rng.random() < 0.5:
                    dist = rng.uniform(p.far_dist_min, p.far_dist_max)
                    lateral = rng.uniform(-0.15, 0.15)
                else:
                    dist = rng.uniform(p.block_dist_min, p.block_dist_max)
                    lateral = (1 if rng.random() < 0.5 else -1) * p.side_offset
            # duvar boyu yön (lateral için)
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
                    register_in_db: bool = True, progress_cb=None) -> dict:
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

    for b in baselines:
        base_stem = Path(b["file_path"]).stem
        for vi in range(variants):
            seed = seed_start + done
            out_id = str(uuid.uuid4())
            # İlk boş _violatedN
            n = 1
            while (out_dir / f"{base_stem}_violated{n}.ifc").exists():
                n += 1
            stem = f"{base_stem}_violated{n}"
            if (out_dir / f"{stem}.ifc").exists():
                stem = f"{base_stem}_violated{n}_{out_id[:6]}"
            out_ifc = out_dir / f"{stem}.ifc"
            try:
                r = inject_basic(b["file_path"], out_ifc, seed=seed, params=params)
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
                except Exception:
                    graph_path = None
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
                results["violations"] += r["summary"]["n_violations"]
                results["hard_negatives"] += r["summary"]["n_compliant_changes"]
                results["items"].append({"stem": stem, **r["summary"]})
            except Exception as e:
                results["err"] += 1
                results["items"].append({"stem": stem, "error": str(e)})
            done += 1
            if progress_cb:
                progress_cb(done, total, stem)
    return results
