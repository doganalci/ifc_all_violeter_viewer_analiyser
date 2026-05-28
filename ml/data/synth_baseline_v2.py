"""Sentetik baseline üretici v2 — geniş varyasyon uzayı.

v1'in (`synth_baseline.py`) 2-oda + koridor şemasının üzerine inşa edilmiş;
LLM tasarımcı + prosedürel çizici pipeline'ı için tasarlandı.

Genişletmeler:
  * 2-4 oda (merkez koridordan dalan, N/S/E/W kanatlar)
  * Düz veya L-plan koridor
  * 1-3 kat (her kat bağımsız üretilir, dış cephede pencere)
  * Dış cephe pencereleri (görsellik için, etiket dışı)

Tüm üretilen baselin'ler **garantili mevzuata uygun** — parametre aralıkları
eşik üstünde tutulur (kapı genişliği ≥0.95, yükseklik ≥2.10, koridor ≥1.40).

LLM tasarım planı bu modülün `SpecOverride` arayüzünü doldurur (oda say.,
boyutlar, layout, kat sayısı); make_spec_v2 onu prosedürel olarak çizilebilir
spec dict'e çevirir.

CLI:
    python -m ml.scripts.gen_synth_v2 --n 20 --seed 0 --tag synth_v2_demo
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path


# Standart minimum eşikler (m cinsinden — IFC dünyasında uniform metre)
MIN_DOOR_WIDTH = 0.90
MIN_DOOR_HEIGHT = 2.00
MIN_CORRIDOR_WIDTH = 1.20


@dataclass
class SynthParamsV2:
    """Geniş varyant uzayı parametre kümesi (v2)."""
    # Oda boyut aralıkları (m)
    room_w_min: float = 3.0
    room_w_max: float = 5.5
    room_l_min: float = 3.5
    room_l_max: float = 6.0
    # Koridor boyutu (m) — eşik 1.20 m, kompliant biraz pay
    corridor_w_min: float = 1.40
    corridor_w_max: float = 2.20
    corridor_l_min: float = 3.0
    corridor_l_max: float = 6.0
    # Kapı boyutu — eşik 0.90/2.00, kompliant biraz pay
    door_w_min: float = 0.95
    door_w_max: float = 1.20
    door_h_min: float = 2.10
    door_h_max: float = 2.30
    # Genel
    storey_height: float = 3.0
    wall_thickness: float = 0.20
    add_windows: bool = True

    # Yapı varyasyonları
    n_rooms_per_floor: int = 3        # 2-4 (corridor + N kanat)
    n_storeys: int = 1                # 1-3
    layout: str = "straight"          # "straight" | "lshape"
    n_salons: int = 0                 # 0-N: kaç oda "Salon" olarak adlandırılsın

    # LLM'in açıkça vereceği plan (varsa make_spec_v2 onu override kabul eder).
    # Şema: { "storeys": [{"name", "elevation", "layout", "n_rooms",
    #                       "corridor": {"w": .., "l": ..},
    #                       "rooms": [{"side": ..., "w": .., "l": ..}],
    #                       "doors": {...}}, ...] }
    plan_override: dict | None = None


@dataclass
class SpecOverride:
    """LLM'in döneceği yapısal plan (subset).

    Tüm alanlar opsiyonel; verilmeyenler SynthParamsV2 aralıklarından örnek alır.
    """
    n_storeys: int | None = None
    n_rooms_per_floor: int | None = None
    layout: str | None = None
    storey_height: float | None = None
    # İleri seviye: kat bazında rooms / sizes (LLM detaylı plan verirse)
    storeys: list[dict] | None = None


# ---------------------------------------------------------------------------
# Geometri yardımcıları
# ---------------------------------------------------------------------------

def _u(rng: random.Random, lo: float, hi: float, *, step: float = 0.05) -> float:
    """[lo, hi] aralığından step granüle bir değer."""
    if hi <= lo:
        return round(lo, 4)
    n = max(1, int(round((hi - lo) / step)))
    return round(lo + rng.randint(0, n) * step, 4)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Layout çizicileri — her biri (rooms, openings) döner. Yerel koordinatlar.
# ---------------------------------------------------------------------------

# Oda yerleşim sırası: 2 oda → W,E. 3 oda → W,E,N. 4 oda → W,E,N,S.
_STRAIGHT_SIDES_BY_COUNT = {
    2: ("west", "east"),
    3: ("west", "east", "north"),
    4: ("west", "east", "north", "south"),
}


def _room_name(idx: int, n_total: int, n_salons: int) -> str:
    """İlk n_salons odayı 'Salon' olarak adlandır, gerisini 'Oda_N'."""
    n_salons = max(0, min(n_salons, n_total))
    if idx < n_salons:
        return f"Salon_{idx + 1}" if n_salons > 1 else "Salon"
    oda_idx = idx - n_salons
    n_odas = n_total - n_salons
    return f"Oda_{oda_idx + 1}" if n_odas > 1 else "Oda"


def _draw_straight(rng: random.Random, p: SynthParamsV2, n_rooms: int,
                   is_ground: bool) -> tuple[list, list]:
    """Merkez koridor + 2-4 oda (N/S/E/W kanatlar)."""
    n_rooms = max(2, min(4, n_rooms))
    cw = _u(rng, p.corridor_w_min, p.corridor_w_max)
    cl = _u(rng, p.corridor_l_min, p.corridor_l_max)

    # Koridor: (0,0) → (cl, cw). x ekseni boyunca uzun.
    rooms_out = [{"name": "Koridor", "origin": [0.0, 0.0], "size": [cl, cw]}]
    openings: list[dict] = []

    sides = _STRAIGHT_SIDES_BY_COUNT[n_rooms]
    cor_cx, cor_cy = cl / 2.0, cw / 2.0

    # Hangi koridor kenarının dışarı çıkış için kullanılabileceğini izle
    occupied_sides: set[str] = set()

    for idx, side in enumerate(sides):
        rw = _u(rng, p.room_w_min, p.room_w_max)
        rl = _u(rng, p.room_l_min, p.room_l_max)

        # Yerleşim hesabı: oda koridorun ilgili kenarında dış yöne konur,
        # boyutu koridor kenarı boyunca paylaşılan duvarı kapsayacak biçimde
        # ortalanır.
        if side == "west":
            # batı kenar; oda x ekseninde -rw..0, oda kenar uzunluğu y → rl
            rl = max(rl, cw + 0.5)        # paylaşılan kenarı kapsasın
            origin = [-rw, cor_cy - rl / 2.0]
            size = [rw, rl]
            door_side = "east"
            wall_len = rl
        elif side == "east":
            rl = max(rl, cw + 0.5)
            origin = [cl, cor_cy - rl / 2.0]
            size = [rw, rl]
            door_side = "west"
            wall_len = rl
        elif side == "north":
            rw = max(rw, min(cl, 2.5))     # paylaşılan kenarı bir miktar kapsasın
            origin = [cor_cx - rw / 2.0, cw]
            size = [rw, rl]
            door_side = "south"
            wall_len = rw
        else:  # south
            rw = max(rw, min(cl, 2.5))
            origin = [cor_cx - rw / 2.0, -rl]
            size = [rw, rl]
            door_side = "north"
            wall_len = rw

        rname = _room_name(idx, n_rooms, p.n_salons)
        rooms_out.append({"name": rname, "origin": origin, "size": size})
        occupied_sides.add(side)

        dw = _u(rng, p.door_w_min, p.door_w_max)
        dh = _u(rng, p.door_h_min, p.door_h_max)
        offset = _clamp((wall_len - dw) / 2.0, 0.10, wall_len - dw - 0.10)
        openings.append({
            "room": rname, "side": door_side, "type": "door",
            "width": dw, "height": dh, "offset": offset,
            "is_exterior": False, "name": f"{rname}-Koridor",
        })

        # Dış pencere — odanın koridordan uzak iç-dış kenarı
        if p.add_windows:
            ext_side, ext_wall_len = {
                "west":  ("west",  rl),
                "east":  ("east",  rl),
                "north": ("north", rw),
                "south": ("south", rw),
            }[side]
            ww = min(1.5, ext_wall_len - 0.4)
            if ww >= 0.6:
                openings.append({
                    "room": rname, "side": ext_side, "type": "window",
                    "width": ww, "height": 1.4, "sill": 0.9,
                    "offset": max(0.2, (ext_wall_len - ww) / 2.0),
                    "is_exterior": True,
                })

    # Zemin katta dış giriş kapısı — koridorun boş kenarına
    if is_ground:
        for trial_side in ("south", "north", "east", "west"):
            if trial_side in occupied_sides:
                continue
            wall_len = cl if trial_side in ("south", "north") else cw
            dwe = _u(rng, p.door_w_min, p.door_w_max)
            if wall_len - dwe < 0.4:
                continue
            offset = _clamp((wall_len - dwe) / 2.0, 0.10, wall_len - dwe - 0.10)
            openings.append({
                "room": "Koridor", "side": trial_side, "type": "door",
                "width": dwe, "height": _u(rng, p.door_h_min, p.door_h_max),
                "offset": offset,
                "is_exterior": True, "name": "Giris Kapisi",
            })
            break

    return rooms_out, openings


def _draw_lshape(rng: random.Random, p: SynthParamsV2, n_rooms: int,
                 is_ground: bool) -> tuple[list, list]:
    """L-plan koridor: yatay + dikey segment, 2-3 oda kanat.

    Layout (kuş bakışı):
        +---------+
        | Oda_N   |
        +----+----+
             | Kor_V |
        +----+    +----+
        |  Kor_H       |   Oda_E
        +----+----+----+
        | Oda_S   |
        +---------+
    """
    n_rooms = max(2, min(3, n_rooms))     # L-plan'da entrance için pay
    cw = _u(rng, p.corridor_w_min, p.corridor_w_max)
    cl_h = _u(rng, p.corridor_l_min, p.corridor_l_max)   # yatay uzunluk
    cl_v = _u(rng, p.corridor_l_min, p.corridor_l_max)   # dikey uzunluk

    # Kor_H: x in [0, cl_h], y in [0, cw]
    # Kor_V: x in [cl_h - cw, cl_h], y in [cw, cw + cl_v]
    rooms_out = [
        {"name": "Koridor_H", "origin": [0.0, 0.0], "size": [cl_h, cw]},
        {"name": "Koridor_V", "origin": [cl_h - cw, cw], "size": [cw, cl_v]},
    ]
    openings: list[dict] = [
        # İki koridoru birbirine bağlayan iç kapı (Kor_H'nin kuzey duvarında)
        {"room": "Koridor_H", "side": "north", "type": "door",
         "width": min(cw - 0.2, 1.0), "height": _u(rng, p.door_h_min, p.door_h_max),
         "offset": cl_h - cw + 0.1,
         "is_exterior": False, "name": "Koridor-Bağlantı"},
    ]

    # Yerleşim slotları (giriş için south_H her zaman boş bırakılır)
    _slot_positions = ["west_H", "east_V", "north_V"][:n_rooms]
    slots = [(_room_name(i, n_rooms, p.n_salons), sl)
             for i, sl in enumerate(_slot_positions)]

    cor_h_cx, cor_h_cy = cl_h / 2.0, cw / 2.0
    cor_v_cx = cl_h - cw / 2.0
    cor_v_cy = cw + cl_v / 2.0

    for rname, slot in slots:
        rw = _u(rng, p.room_w_min, p.room_w_max)
        rl = _u(rng, p.room_l_min, p.room_l_max)
        dw = _u(rng, p.door_w_min, p.door_w_max)
        dh = _u(rng, p.door_h_min, p.door_h_max)

        if slot == "west_H":
            rl = max(rl, cw + 0.5)
            origin = [-rw, cor_h_cy - rl / 2.0]
            size = [rw, rl]
            door_side, wall_len = "east", rl
            ext_side = "west"
        elif slot == "east_V":
            rl = max(rl, cw + 0.5)
            origin = [cl_h, cor_v_cy - rl / 2.0]
            size = [rw, rl]
            door_side, wall_len = "west", rl
            ext_side = "east"
        else:  # north_V
            rw = max(rw, cw + 0.5)
            origin = [cor_v_cx - rw / 2.0, cw + cl_v]
            size = [rw, rl]
            door_side, wall_len = "south", rw
            ext_side = "north"

        rooms_out.append({"name": rname, "origin": origin, "size": size})
        offset = _clamp((wall_len - dw) / 2.0, 0.10, wall_len - dw - 0.10)
        openings.append({
            "room": rname, "side": door_side, "type": "door",
            "width": dw, "height": dh, "offset": offset,
            "is_exterior": False, "name": f"{rname}-Koridor",
        })
        if p.add_windows:
            ww = min(1.5, (rl if ext_side in ("west", "east") else rw) - 0.4)
            if ww >= 0.6:
                ext_wall_len = rl if ext_side in ("west", "east") else rw
                openings.append({
                    "room": rname, "side": ext_side, "type": "window",
                    "width": ww, "height": 1.4, "sill": 0.9,
                    "offset": max(0.2, (ext_wall_len - ww) / 2.0),
                    "is_exterior": True,
                })

    # Zemin katta dış giriş — Kor_H'nin güneyine
    if is_ground:
        dwe = _u(rng, p.door_w_min, p.door_w_max)
        offset = _clamp((cl_h - dwe) / 2.0, 0.10, cl_h - dwe - 0.10)
        openings.append({
            "room": "Koridor_H", "side": "south", "type": "door",
            "width": dwe, "height": _u(rng, p.door_h_min, p.door_h_max),
            "offset": offset,
            "is_exterior": True, "name": "Giris Kapisi",
        })

    return rooms_out, openings


# ---------------------------------------------------------------------------
# Spec üretimi
# ---------------------------------------------------------------------------

def make_spec_v2(seed: int, params: SynthParamsV2 | None = None,
                 name: str | None = None,
                 override: SpecOverride | None = None) -> dict:
    """Tekrar üretilebilir çok katlı bir spec üret.

    `override` LLM'den gelen ipuçlarını uygular (None ise tamamen rastgele).
    """
    p = params or SynthParamsV2()
    rng = random.Random(seed)

    # Override uygula
    n_storeys = (override.n_storeys if override and override.n_storeys
                 else p.n_storeys)
    n_storeys = max(1, min(3, int(n_storeys)))
    n_rooms = (override.n_rooms_per_floor if override and override.n_rooms_per_floor
               else p.n_rooms_per_floor)
    n_rooms = max(2, min(4, int(n_rooms)))
    layout = (override.layout if override and override.layout
              else p.layout)
    if layout not in ("straight", "lshape"):
        layout = "straight"
    storey_h = (override.storey_height if override and override.storey_height
                else p.storey_height)

    storeys_data: list[dict] = []
    for s_idx in range(n_storeys):
        # Her kat bağımsız üretilir → çeşitlilik. Aynı seed'ten türeyen
        # alt seed kullanılır.
        floor_seed = seed * 100 + s_idx
        floor_rng = random.Random(floor_seed)
        is_ground = (s_idx == 0)
        if layout == "lshape":
            rooms, openings = _draw_lshape(floor_rng, p, n_rooms, is_ground)
        else:
            rooms, openings = _draw_straight(floor_rng, p, n_rooms, is_ground)
        storey_name = f"Kat {s_idx}" if s_idx > 0 else "Zemin"
        storeys_data.append({
            "name": storey_name,
            "elevation": s_idx * storey_h,
            "rooms": rooms,
            "openings": openings,
        })

    door_widths_cm: dict[str, float] = {}
    for sdata in storeys_data:
        for op in sdata["openings"]:
            if op.get("type") != "door":
                continue
            key = f"{sdata['name']} · {op.get('name') or op['room']+'-'+op['side']}"
            door_widths_cm[key] = round(float(op["width"]) * 100, 1)

    return {
        "name": name or f"synth-v2-{seed:05d}",
        "storey_height": storey_h,
        "wall_thickness": p.wall_thickness,
        "storeys": storeys_data,
        "_meta": {
            "kind": "synth_v2",
            "seed": seed,
            "n_storeys": n_storeys,
            "n_rooms_per_floor": n_rooms,
            "layout": layout,
            "door_widths_cm": door_widths_cm,
            "compliant": True,
        },
    }


# ---------------------------------------------------------------------------
# Üretim API'leri
# ---------------------------------------------------------------------------

def generate_v2(seed: int, out_path: str | Path,
                params: SynthParamsV2 | None = None,
                override: SpecOverride | None = None,
                ifc_id: str | None = None,
                write_graph: bool = True,
                write_meta: bool = True,
                empty_labels: bool = True) -> dict:
    """Tek bir v2 baseline IFC üret."""
    from violation_pool.ifc_template import build_house_ifc

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = make_spec_v2(seed, params=params, override=override, name=out.stem)
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
            "_meta": "synth_v2 baseline — no violations by design",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        labels_path = lp

    meta_path = None
    if write_meta:
        mp = base.parent / f"{base.name}.meta.json"
        mp.write_text(json.dumps({
            "ifc_id": ifc_id,
            "kind": "baseline",
            "source": "synth_v2",
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


def generate_batch_v2(n: int, out_dir: str | Path,
                      seed_start: int = 0,
                      params: SynthParamsV2 | None = None,
                      overrides: list[SpecOverride] | None = None,
                      register_in_db: bool = True,
                      dataset_tag: str | None = None,
                      progress_cb=None) -> list[dict]:
    """N adet v2 baseline üret.

    `overrides` listesi verilirse, i'inci baseline i'inci override ile üretilir
    (LLM tasarım planı yığını). None ise her IFC tamamen rastgele.
    """
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "synth_v2"
    if dataset_tag:
        prefix = "".join(c if c.isalnum() or c in "-_+" else "_"
                         for c in str(dataset_tag).strip()) or prefix

    results: list[dict] = []
    for i in range(n):
        seed = seed_start + i
        ovr = overrides[i] if overrides and i < len(overrides) else None
        ifc_path = out_dir / f"{prefix}_{seed:05d}.ifc"
        info = generate_v2(seed, ifc_path, params=params, override=ovr)
        info["dataset_tag"] = dataset_tag
        if register_in_db:
            _register_v2(info, dataset_tag=dataset_tag)
        results.append(info)
        if progress_cb:
            progress_cb(i + 1, n, info)

    if register_in_db:
        try:
            from ml.tracking import rebuild_dataset_registry, log_operation
            from paths import data_home
            rebuild_dataset_registry(str(data_home()))
            log_operation(
                "sentetik_uretim_v2",
                paket=dataset_tag or "", adet=len(results),
                ozet=f"{len(results)} baseline (v2, çok katlı/L-plan destekli)",
                parametreler=f"seed_start={seed_start}",
            )
        except Exception:
            pass
    return results


def _register_v2(info: dict, dataset_tag: str | None = None) -> None:
    """codex1 storage'ına baseline olarak yaz."""
    try:
        from violation_pool import storage
        storage.create_ifc_model(
            id=info["ifc_id"],
            kind="baseline",
            name=Path(info["ifc_path"]).stem,
            parent_id=None, pool_run_id=None,
            params=info["spec"].get("_meta", {}),
            file_path=info["ifc_path"],
            meta_path=info.get("meta_path"),
            labels_path=info.get("labels_path"),
            graph_path=info.get("graph_path"),
            llm_model="synth_v2",
            prompt="synth_v2 multi-storey compliant",
            status="ok", error=None,
            dataset_tag=dataset_tag,
        )
    except Exception as e:
        print(f"[synth_v2] DB kaydı atlandı: {e}")
