"""Kural-tabanlı erişilebilirlik oracle'ı — etiket augmentation.

Sorun: `ifc_inject.py` sadece bizim enjekte ettiğimiz ihlalleri etiketler.
Baseline IFC'lerde zaten var olan (parametrik üretim sırasında oluşan)
gerçek ihlaller etiketsiz kalır ve eğitimde **yanlış negatif** sinyale
dönüşür. Sonuç: model "etiketten kopya çeker" — enjeksiyon imzasını
tanır ama erişilebilirlik kurallarını öğrenmez.

Çözüm: Eğitim verisini oluştururken her IFC'nin grafiği geometrik
kurallardan geçirilir. Mevzuata aykırı node'lar **otomatik etiketlenir**
ve pozitif örnek olarak eğitime girer. Böylece model:
  * Bizim enjekte ettiğimiz ihlaller (semantik etiket)
  * Geometrik kurallarla bulunan baseline-native ihlaller (oracle etiketi)
ikisinden de öğrenir.

Kurallar bilinçli olarak basit ve **mevzuat-tipik** tutulmuştur (TS 9111,
ADA referansları). Amaç pseudo-label üretmek; perfect olması gerekmez.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx


@dataclass(frozen=True)
class Rule:
    ifc_type: str
    attribute: str
    op: str            # "lt" | "gt"
    threshold: float   # mm cinsinden (ifcopenshell varsayılanı)
    category: str
    severity: str = "yüksek"
    note: str = ""


# Erişilebilirlik kuralları (mm; ifcopenshell varsayılan birim).
# Yeni kural eklemek için: tuple sonuna ekle; eğitim cache'i invalidate olur.
RULES: tuple[Rule, ...] = (
    # Kapılar
    Rule("IfcDoor", "OverallWidth", "lt", 900.0, "Kapı/Koridor", "kritik",
         "Engelli erişimi için min 90 cm net açıklık"),
    Rule("IfcDoor", "OverallHeight", "lt", 2000.0, "Kapı/Koridor", "orta",
         "Min 200 cm geçiş yüksekliği"),

    # Pencereler — alçak / kontrast denetimi
    Rule("IfcWindow", "OverallHeight", "lt", 800.0, "Yönlendirme/İşaretleme", "düşük",
         "Çok kısa pencere — görüş hattı kesilebilir"),

    # Mekânlar — manevra alanı
    Rule("IfcSpace", "OverallWidth", "lt", 1500.0, "Manevra alanı", "kritik",
         "Tekerlekli sandalye için min 150x150 cm dönüş alanı"),
    Rule("IfcSpace", "OverallHeight", "lt", 2400.0, "Manevra alanı", "düşük",
         "Net tavan yüksekliği"),

    # Merdivenler
    Rule("IfcStair", "NominalHeight", "gt", 180.0, "Merdiven", "yüksek",
         "Max rıht yüksekliği 18 cm"),
    Rule("IfcStairFlight", "NominalHeight", "gt", 180.0, "Merdiven", "yüksek",
         "Max rıht yüksekliği 18 cm"),

    # Rampalar
    Rule("IfcRamp", "OverallHeight", "gt", 1500.0, "Rampa", "yüksek",
         "Tek seferde 1.5 m'den fazla yükselen rampa — sahanlık gerek"),

    # Korkuluklar
    Rule("IfcRailing", "NominalHeight", "lt", 900.0, "Korkuluk/Küpeşte", "kritik",
         "Min 90 cm güvenlik yüksekliği"),
    Rule("IfcRailing", "NominalHeight", "gt", 1100.0, "Korkuluk/Küpeşte", "orta",
         "Çok yüksek korkuluk — çocuk/engelli kavraması zor"),
)


def _attr_value(node_data: dict, name: str) -> float | None:
    attrs = node_data.get("attributes") or {}
    v = attrs.get(name)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_violations(graph: nx.MultiDiGraph) -> dict[str, dict]:
    """Grafiği geometrik kurallardan geçir, ihlal adaylarını döndür.

    Returns:
        { guid -> { "category": str, "severity": str, "rule": str, "value": float, "threshold": float } }
    """
    hits: dict[str, dict] = {}
    for guid, node in graph.nodes(data=True):
        ifc_type = node.get("ifc_type") or node.get("type")
        if not ifc_type:
            continue
        for r in RULES:
            if r.ifc_type != ifc_type:
                continue
            val = _attr_value(node, r.attribute)
            if val is None:
                continue
            triggered = (
                (r.op == "lt" and val < r.threshold)
                or (r.op == "gt" and val > r.threshold)
            )
            if not triggered:
                continue
            # Aynı node için birden fazla kural tetiklendiyse ilk eşleşmeyi tut
            # (önem sırası: RULES içindeki sıra).
            if guid not in hits:
                hits[guid] = {
                    "category": r.category,
                    "severity": r.severity,
                    "rule": f"{r.ifc_type}.{r.attribute} {r.op} {r.threshold}",
                    "value": val,
                    "threshold": r.threshold,
                }
            break
    return hits


def augment_sample(sample, *, replace_existing: bool = False) -> tuple[int, int]:
    """Bir Sample'ın label setini oracle kurallarıyla genişlet.

    Args:
        sample: ml.data.graph_loader.Sample objesi (mutate edilir)
        replace_existing: True ise mevcut etiket olsa bile üzerine yaz.

    Returns:
        (n_added, n_skipped_existing)
    """
    found = find_violations(sample.graph)
    n_added = 0
    n_skipped = 0
    for guid, info in found.items():
        already = sample.y.get(guid, 0) == 1
        if already and not replace_existing:
            n_skipped += 1
            continue
        sample.y[guid] = 1
        sample.category_per_pos[guid] = info["category"]
        if not already:
            n_added += 1
    return n_added, n_skipped


def summarize_rules() -> list[dict]:
    """UI'da kullanılabilir özet (kural listesi tablosu)."""
    return [
        {
            "ifc_type": r.ifc_type,
            "attribute": r.attribute,
            "op": r.op,
            "threshold_mm": r.threshold,
            "category": r.category,
            "severity": r.severity,
            "açıklama": r.note,
        }
        for r in RULES
    ]
