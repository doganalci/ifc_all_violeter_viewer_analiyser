"""Adlandırılmış, paket-bazlı veri seti kompozisyonu.

Bir 'veri seti' bir dizi paket etiketinin (dataset_tag) adlandırılmış
toplamıdır. Diskte JSON olarak `<data_home>/datasets/<slug>.json`
yolunda tutulur. Eğitim ve sayfa 23 buradan okuyup yazar.

Schema:
{
  "slug": "training_v1",
  "name": "Training v1",
  "tags": ["pkg_a", "pkg_b"],
  "notes": "...",
  "created_at": "2026-...",
  "updated_at": "2026-..."
}
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# paths.py modülü kullanılabilir olmalı (repo kökü importable).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from paths import data_home


def _dir() -> Path:
    p = data_home() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "untitled"


def _path(slug: str) -> Path:
    return _dir() / f"{slug}.json"


def list_datasets() -> list[dict]:
    out: list[dict] = []
    for fp in sorted(_dir().glob("*.json")):
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def get_dataset(slug: str) -> dict | None:
    p = _path(slug)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_dataset(name: str, tags: list[str], notes: str = "",
                 *, overwrite: bool = False) -> dict:
    slug = slugify(name)
    p = _path(slug)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset '{slug}' zaten var. Düzenle veya başka ad seç."
        )
    doc = {
        "slug": slug,
        "name": name.strip(),
        "tags": list(dict.fromkeys(tags)),
        "notes": notes.strip(),
        "created_at": now,
        "updated_at": now,
    }
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return doc


def update_dataset(slug: str, *, name: str | None = None,
                   tags: list[str] | None = None,
                   notes: str | None = None) -> dict:
    doc = get_dataset(slug)
    if not doc:
        raise FileNotFoundError(slug)
    if name is not None:
        doc["name"] = name.strip()
    if tags is not None:
        doc["tags"] = list(dict.fromkeys(tags))
    if notes is not None:
        doc["notes"] = notes.strip()
    doc["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _path(slug).write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return doc


def delete_dataset(slug: str) -> None:
    p = _path(slug)
    if p.exists():
        p.unlink()
