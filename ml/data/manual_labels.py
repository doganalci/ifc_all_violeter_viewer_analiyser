"""Manuel (altın standart) etiket dosyaları — load / save / list.

Otomatik üretilen `<ifc>.labels.json` dosyaları yanına paralel olarak
`<ifc>.manual_labels.json` yazılır. İçinde her node için **üçlü karar**
saklanır:

  * verdict: "violation" | "not_violation" | "unknown"
  * category: "Kapı/Koridor" | ...   (sadece violation için)
  * severity: "düşük|orta|yüksek|kritik" (opsiyonel)
  * note: serbest metin

Bu dosyalar oto-etiketlerin üzerine yazmaz; eğitim eski label'larla
yapılır, **test/değerlendirme** manuel set üzerinden raporlanır.

Format:
{
  "ifc_id": "<uuid>",
  "annotator": "user@host",
  "created_at": "2026-...",
  "updated_at": "2026-...",
  "labels": {
    "<node_guid>": {
        "verdict": "violation",
        "category": "Kapı/Koridor",
        "severity": "yüksek",
        "note": "Bağlamdan dolayı eşik altında"
    },
    ...
  }
}
"""
from __future__ import annotations

import getpass
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Iterable

# 16 kategori — codex1 prompts.py ile aynı
VIOLATION_CATEGORIES = (
    "Yaya erişimi", "Giriş", "Kapı/Koridor", "Rampa", "Merdiven",
    "Korkuluk/Küpeşte", "Asansör", "Tuvalet/Banyo", "Mutfak", "Otopark",
    "Uyarı yüzeyi", "Yönlendirme/İşaretleme", "Görsel/Kontrast",
    "Aydınlatma", "Manevra alanı", "Eşik/Kot farkı",
)
SEVERITIES = ("düşük", "orta", "yüksek", "kritik")

VERDICT_VIOLATION = "violation"
VERDICT_NOT = "not_violation"
VERDICT_UNKNOWN = "unknown"
VERDICTS = (VERDICT_VIOLATION, VERDICT_NOT, VERDICT_UNKNOWN)


def manual_labels_path(ifc_path: str | Path) -> Path:
    """`<name>.ifc` → `<name>.manual_labels.json` (aynı klasörde)."""
    p = Path(ifc_path)
    return p.parent / f"{p.stem}.manual_labels.json"


def load(ifc_path: str | Path) -> dict:
    """Manuel label dosyasını yükle (yoksa boş şema döner)."""
    p = manual_labels_path(ifc_path)
    if not p.exists():
        return {
            "ifc_id": "",
            "annotator": _who(),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": "",
            "labels": {},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "ifc_id": "",
            "annotator": _who(),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": "",
            "labels": {},
        }


def save(ifc_path: str | Path, ifc_id: str, labels: dict[str, dict]) -> Path:
    """Manuel label setini diske yaz, dolu dosya yolunu döner."""
    doc = load(ifc_path)
    doc["ifc_id"] = ifc_id
    doc["updated_at"] = datetime.utcnow().isoformat() + "Z"
    if not doc.get("created_at"):
        doc["created_at"] = doc["updated_at"]
    if not doc.get("annotator"):
        doc["annotator"] = _who()
    doc["labels"] = labels
    p = manual_labels_path(ifc_path)
    p.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def upsert_node(ifc_path: str | Path, ifc_id: str, guid: str,
                verdict: str, *, category: str | None = None,
                severity: str | None = None, note: str = "") -> Path:
    """Tek bir node için karar yaz, dosyayı güncelle."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict invalid: {verdict}")
    doc = load(ifc_path)
    doc.setdefault("labels", {})
    entry: dict = {"verdict": verdict, "note": note or ""}
    if verdict == VERDICT_VIOLATION:
        if category:
            entry["category"] = category
        if severity:
            entry["severity"] = severity
    doc["labels"][guid] = entry
    return save(ifc_path, ifc_id, doc["labels"])


def delete_node(ifc_path: str | Path, ifc_id: str, guid: str) -> None:
    doc = load(ifc_path)
    if guid in doc.get("labels", {}):
        del doc["labels"][guid]
        save(ifc_path, ifc_id, doc["labels"])


def stats(doc: dict) -> dict:
    """{ "violation": N, "not_violation": M, "unknown": K, "total": T }"""
    out = {v: 0 for v in VERDICTS}
    for v in doc.get("labels", {}).values():
        out[v.get("verdict", "unknown")] = out.get(v.get("verdict", "unknown"), 0) + 1
    out["total"] = sum(out.values())
    return out


def list_annotated_ifcs(roots: Iterable[Path]) -> list[Path]:
    """Manuel etiketi olan tüm IFC'lerin yollarını dön (golden test seti)."""
    out: list[Path] = []
    for root in roots:
        if not root or not Path(root).exists():
            continue
        for ml in Path(root).rglob("*.manual_labels.json"):
            # Karşılık gelen .ifc'yi ara
            ifc = ml.with_name(ml.name.replace(".manual_labels.json", ".ifc"))
            if ifc.exists():
                out.append(ifc)
    return out


def _who() -> str:
    try:
        return f"{getpass.getuser()}@{socket.gethostname()}"
    except Exception:
        return "user"
