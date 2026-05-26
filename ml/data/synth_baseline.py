"""Sentetik baseline IFC üretici — 2 oda + 1 koridor.

Amaç: codex1'in LLM-üretimli baseline'larında **etiketsiz gerçek ihlaller**
oluyordu (LLM mevzuat bilmeden değer üretiyor). Bu modül baseline'ı
**deterministik ve mevzuata uygun** parametrelerle üretir, böylece:

  * Closed-world supervision varsayımı baseline'lar için DOĞRU olur
    (gerçekten hiç ihlal yok).
  * Üzerine yapılan injection'lar tek pozitif sinyal kaynağı olur.
  * Model F1=1.000 alıyorsa bu artık 'gerçek' bir sayı sayılır
    (overfit / feature leak ayrı analiz konusu).

Layout (kuş bakışı):

    +-------+-----------+-------+
    |       |           |       |
    | Oda1  | Koridor   | Oda2  |
    |       |           |       |
    +---+---+---+---+---+---+---+
              ^
              dış giriş

Mevzuat eşikleri (TS 9111 / ADA özet):
  * Kapı net genişliği ≥ 0.90 m
  * Kapı net yüksekliği ≥ 2.00 m
  * Koridor net genişliği ≥ 1.20 m

CLI:
    python -m ml.scripts.gen_synth --n 50 --seed 0
"""
from __future__ import annotations

import json
import random
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Bu modül torch-bağımsız çalışabilmeli (sadece geometri); paths/storage
# erişimi için repo kökünü sys.path'e ekleyen daha üst seviye script çağrıları
# yeterli. Burada direkt import etmeyelim.


# --- Standart minimum eşikler (mm cinsinden — IFC default) ----------------
MIN_DOOR_WIDTH_MM = 900.0
MIN_DOOR_HEIGHT_MM = 2000.0
MIN_CORRIDOR_WIDTH_MM = 1200.0


@dataclass
class SynthParams:
    """Üretim parametre uzayı — her örnek bu aralıklardan örnek alır."""
    # m cinsinden
    room_w_min: float = 3.0
    room_w_max: float = 5.5
    room_l_min: float = 3.5
    room_l_max: float = 6.0
    corridor_w_min: float = 1.4    # ≥1.2 ama biraz pay
    corridor_w_max: float = 2.2
    corridor_l_min: float = 2.0
    corridor_l_max: float = 5.0
    door_w_min: float = 0.95       # ≥0.90 ama biraz pay
    door_w_max: float = 1.20
    door_h_min: float = 2.10       # ≥2.00 ama biraz pay
    door_h_max: float = 2.30
    storey_height: float = 3.0
    wall_thickness: float = 0.20
    # Pencere eklensin mi? (sadece görsellik için)
    add_windows: bool = True
    # Kapı-odaklı deney: oda/koridorun boş duvarlarına ek (uyumlu) kapılar.
    # 0 = sadece 3 temel kapı. 5'e kadar ek kapı eklenebilir.
    extra_doors: int = 0


def _u(rng: random.Random, lo: float, hi: float, *, step: float = 0.05) -> float:
    """[lo, hi] aralığından `step` granüle değer."""
    n = max(1, int(round((hi - lo) / step)))
    return round(lo + rng.randint(0, n) * step, 4)


def make_spec(seed: int, params: SynthParams | None = None,
              name: str | None = None) -> dict:
    """Tekrar üretilebilir bir 2-oda+koridor spec'i üret."""
    p = params or SynthParams()
    rng = random.Random(seed)

    w1 = _u(rng, p.room_w_min, p.room_w_max)
    l1 = _u(rng, p.room_l_min, p.room_l_max)
    w2 = _u(rng, p.room_w_min, p.room_w_max)
    l2 = _u(rng, p.room_l_min, p.room_l_max)
    cw = _u(rng, p.corridor_w_min, p.corridor_w_max)
    cl = _u(rng, p.corridor_l_min, p.corridor_l_max)

    # Koridoru oda boylarının ortasına hizala (y ekseninde)
    common_l = max(l1, l2, cl)
    # Oda1 solda, Koridor ortada, Oda2 sağda — hepsi y=0'dan başlar
    o1 = [0.0, (common_l - l1) / 2.0]
    oc = [w1, (common_l - cl) / 2.0]
    o2 = [w1 + cw, (common_l - l2) / 2.0]

    dw1 = _u(rng, p.door_w_min, p.door_w_max)
    dw2 = _u(rng, p.door_w_min, p.door_w_max)
    dwe = _u(rng, p.door_w_min, p.door_w_max)  # dış giriş
    dh = _u(rng, p.door_h_min, p.door_h_max)

    openings: list[dict] = [
        # Oda1 → Koridor (Oda1'in doğu duvarı)
        {"room": "Oda1", "side": "east", "type": "door",
         "width": dw1, "height": dh, "offset": max(0.1, (l1 - dw1) / 2.0),
         "is_exterior": False, "name": "Oda1-Koridor"},
        # Oda2 → Koridor (Oda2'nin batı duvarı)
        {"room": "Oda2", "side": "west", "type": "door",
         "width": dw2, "height": dh, "offset": max(0.1, (l2 - dw2) / 2.0),
         "is_exterior": False, "name": "Oda2-Koridor"},
        # Koridor → Dış (güney duvarı)
        {"room": "Koridor", "side": "south", "type": "door",
         "width": dwe, "height": dh, "offset": max(0.1, (cw - dwe) / 2.0),
         "is_exterior": True, "name": "Giris Kapisi"},
    ]
    if p.add_windows:
        # Oda1 batı, Oda2 doğu pencere
        openings.extend([
            {"room": "Oda1", "side": "west", "type": "window",
             "width": min(1.5, w1 - 0.2), "height": 1.4, "sill": 0.9,
             "offset": max(0.1, (l1 - 1.5) / 2.0), "is_exterior": True},
            {"room": "Oda2", "side": "east", "type": "window",
             "width": min(1.5, w2 - 0.2), "height": 1.4, "sill": 0.9,
             "offset": max(0.1, (l2 - 1.5) / 2.0), "is_exterior": True},
        ])

    # Kapı-odaklı deney: boş duvarlara ek uyumlu kapılar ekle. Her aday
    # (oda, kenar, duvar_uzunluğu); kapı duvara sığarsa eklenir. Hepsi
    # mevzuata uygun (≥0.90 m genişlik, ≥2.00 m yükseklik) — ihlal değil.
    if p.extra_doors and p.extra_doors > 0:
        candidates = [
            ("Oda1", "north", w1), ("Oda1", "south", w1),
            ("Oda2", "north", w2), ("Oda2", "south", w2),
            ("Koridor", "north", cw),
        ]
        added = 0
        for rname, side, wall_len in candidates:
            if added >= p.extra_doors:
                break
            dwx = _u(rng, p.door_w_min, p.door_w_max)
            if dwx > wall_len - 0.30:        # duvara sığmıyorsa atla
                continue
            openings.append({
                "room": rname, "side": side, "type": "door",
                "width": dwx, "height": _u(rng, p.door_h_min, p.door_h_max),
                "offset": max(0.1, (wall_len - dwx) / 2.0),
                "is_exterior": True, "name": f"{rname}-Kapi-{side}",
            })
            added += 1

    return {
        "name": name or f"synth-two-room-{seed:05d}",
        "storey_height": p.storey_height,
        "wall_thickness": p.wall_thickness,
        "rooms": [
            {"name": "Oda1",    "origin": o1, "size": [w1, l1]},
            {"name": "Koridor", "origin": oc, "size": [cw, cl]},
            {"name": "Oda2",    "origin": o2, "size": [w2, l2]},
        ],
        "openings": openings,
        "_meta": {
            "kind": "synth_two_room",
            "seed": seed,
            "corridor_width_cm": round(cw * 100, 1),
            "door_widths_cm": {
                "Oda1-Koridor": round(dw1 * 100, 1),
                "Oda2-Koridor": round(dw2 * 100, 1),
                "Giris Kapisi": round(dwe * 100, 1),
            },
            "compliant": True,  # Garantili — eşik altı parametre seçilemez
        },
    }


def generate(seed: int, out_path: str | Path,
             params: SynthParams | None = None,
             ifc_id: str | None = None,
             write_graph: bool = True,
             write_meta: bool = True,
             empty_labels: bool = True) -> dict:
    """Tek bir baseline IFC üret. Dosyaları yazıp özet döner.

    Returns:
        {"ifc_path": ..., "graph_path": ..., "labels_path": ...,
         "meta_path": ..., "spec": dict, "ifc_id": str}
    """
    from violation_pool.ifc_template import build_house_ifc

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = make_spec(seed, params=params, name=out.stem)
    build_house_ifc(spec, out)

    ifc_id = ifc_id or str(uuid.uuid4())
    base = out.with_suffix("")

    graph_path = None
    if write_graph:
        from violation_pool.ifc_graph import build_and_save
        gp = base.parent / f"{base.name}.graph.json"
        build_and_save(str(out), str(gp))
        graph_path = gp

    labels_path = None
    if empty_labels:
        lp = base.parent / f"{base.name}.labels.json"
        lp.write_text(json.dumps({
            "violated_id": ifc_id,
            "labels": [],
            "_meta": "synth baseline — no violations by design",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        labels_path = lp

    meta_path = None
    if write_meta:
        mp = base.parent / f"{base.name}.meta.json"
        mp.write_text(json.dumps({
            "ifc_id": ifc_id,
            "kind": "baseline",
            "source": "synth_two_room",
            "seed": seed,
            "spec_meta": spec.get("_meta", {}),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        meta_path = mp

    return {
        "ifc_path": str(out),
        "graph_path": str(graph_path) if graph_path else None,
        "labels_path": str(labels_path) if labels_path else None,
        "meta_path": str(meta_path) if meta_path else None,
        "spec": spec,
        "ifc_id": ifc_id,
    }


def generate_batch(n: int, out_dir: str | Path,
                   seed_start: int = 0,
                   params: SynthParams | None = None,
                   register_in_db: bool = True,
                   dataset_tag: str | None = None,
                   progress_cb=None) -> list[dict]:
    """N adet baseline üret. İsteğe bağlı olarak DB'ye baseline olarak kaydet.

    Args:
        dataset_tag: bu üretimin etiketi (örn 'synth_v1'). Eğitim sayfasında
            multi-select ile filtreleme bu etiket üzerinden yapılır.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dosya adı prefix'i: dataset etiketi (örn basic2+1_baseline_v04).
    # Böylece baseline + ondan türeyen violated izlenebilir adlar alır:
    #   basic2+1_baseline_v04_00000.ifc → ..._00000_violated1.ifc
    prefix = "synth_two_room"
    if dataset_tag:
        prefix = "".join(c if c.isalnum() or c in "-_+" else "_"
                         for c in str(dataset_tag).strip()) or prefix
    results: list[dict] = []
    for i in range(n):
        seed = seed_start + i
        ifc_path = out_dir / f"{prefix}_{seed:05d}.ifc"
        info = generate(seed, ifc_path, params=params)
        info["dataset_tag"] = dataset_tag
        if register_in_db:
            _register(info, dataset_tag=dataset_tag)
        results.append(info)
        if progress_cb:
            progress_cb(i + 1, n, info)
    # Veri kümesi defterini güncelle — best-effort (üretimi bozmaz).
    if register_in_db:
        try:
            from ml.tracking import rebuild_dataset_registry
            from paths import data_home
            rebuild_dataset_registry(str(data_home()))
        except Exception:
            pass
    return results


def _register(info: dict, dataset_tag: str | None = None) -> None:
    """codex1 storage'ına baseline olarak yaz."""
    try:
        from violation_pool import storage
        storage.create_ifc_model(
            id=info["ifc_id"],
            kind="baseline",
            name=Path(info["ifc_path"]).stem,
            parent_id=None,
            pool_run_id=None,
            params=info["spec"].get("_meta", {}),
            file_path=info["ifc_path"],
            meta_path=info.get("meta_path"),
            labels_path=info.get("labels_path"),
            graph_path=info.get("graph_path"),
            llm_model="synth",
            prompt="synthetic 2-room compliant",
            status="ok",
            error=None,
            dataset_tag=dataset_tag,
        )
    except Exception as e:
        print(f"[synth] DB kaydı atlandı: {e}")
