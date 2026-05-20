"""IFC ihlal datasetini taşınabilir bir .zip olarak paketler.

`IFC_DATA_HOME/ifc_models/{kind}/` altındaki üretilmiş IFC dosyalarını,
yanlarında duran `.labels.json` / `.meta.json` / `.graph.json` dosyaları
ile birlikte tek bir zip içine alır ve etiketleme şemasını anlatan bir
README ekler. Çıkan zip'i kullanıcı başka bir ortama (HuggingFace,
PyTorch eğitim ortamı, başka makinedeki bir araştırmacı) götürebilir.

Kullanım:
    python dataset_export/prepare_dataset.py --out dataset.zip
    python dataset_export/prepare_dataset.py --out v1.zip --kind violated
    python dataset_export/prepare_dataset.py --out full.zip --include-baseline --include-graph

Zip yapısı:
    examples/{kind}/<name>.ifc
    labels/{kind}/<name>.labels.json
    meta/{kind}/<name>.meta.json
    graphs/{kind}/<name>.graph.json   (--include-graph ise)
    manifest.json                       (tüm örneklerin indeksi)
    README.md                           (etiket şeması ve istatistik)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Repo kökünü sys.path'a ekle ki `paths` import edilebilsin.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from paths import data_home, ifc_models_dir  # noqa: E402


_TPL_PATH = Path(__file__).parent / "DATASET_README.md.tpl"


def _kinds(arg_kind: str, include_baseline: bool) -> list[str]:
    if arg_kind == "all":
        kinds = ["violated"]
        if include_baseline:
            kinds = ["baseline", "violated", "imports"]
        return kinds
    return [arg_kind]


def _iter_ifc(models_root: Path, kinds: list[str]) -> Iterable[tuple[str, Path]]:
    for k in kinds:
        d = models_root / k
        if not d.exists():
            continue
        for ifc in sorted(d.glob("*.ifc")):
            yield k, ifc


def _sidecar(ifc: Path, suffix: str) -> Path | None:
    """`<name>.ifc` → `<name>.<suffix>` (yoksa None)."""
    p = ifc.with_suffix("").with_suffix("." + suffix)
    # `.ifc` çıkartıp yeniden uzantı eklemek bazen istenmeyen sonuç
    # verir; daha güvenlisi:
    p = ifc.parent / f"{ifc.stem}.{suffix}"
    return p if p.exists() else None


def _read_labels(labels_path: Path) -> list[dict]:
    try:
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "violations" in data:
            return data["violations"]
    except Exception:
        pass
    return []


def _stats(records: list[dict]) -> dict:
    cats: Counter[str] = Counter()
    sevs: Counter[str] = Counter()
    decoys = 0
    applied = 0
    for r in records:
        if (r.get("category") or "").strip():
            cats[r["category"]] += 1
        if (r.get("severity") or "").strip():
            sevs[r["severity"]] += 1
        if r.get("is_decoy"):
            decoys += 1
        if r.get("status") == "applied":
            applied += 1
    return {
        "category_counts": dict(cats),
        "severity_counts": dict(sevs),
        "decoys": decoys,
        "applied": applied,
        "total": len(records),
    }


def _render_readme(
    out_zip: Path,
    kinds: list[str],
    n_examples: int,
    n_labels: int,
    stats: dict,
    include_graph: bool,
) -> str:
    if _TPL_PATH.exists():
        tpl = _TPL_PATH.read_text(encoding="utf-8")
    else:
        tpl = "# IFC İhlal Dataset\n\n(README şablonu bulunamadı)\n"
    cat_lines = "\n".join(
        f"  - **{c}**: {n} örnek" for c, n in sorted(stats["category_counts"].items())
    ) or "  (henüz veri yok)"
    sev_lines = "\n".join(
        f"  - **{s}**: {n}" for s, n in sorted(stats["severity_counts"].items())
    ) or "  (henüz veri yok)"
    return tpl.format(
        zip_name=out_zip.name,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        kinds=", ".join(kinds),
        n_examples=n_examples,
        n_labels=n_labels,
        n_decoys=stats["decoys"],
        n_applied=stats["applied"],
        categories=cat_lines,
        severities=sev_lines,
        include_graph="evet" if include_graph else "hayır",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="IFC ihlal datasetini zip olarak paketle.")
    p.add_argument("--out", required=True, help="Çıkış .zip yolu")
    p.add_argument(
        "--data-home", default=None,
        help="Veri klasörü (varsayılan: IFC_DATA_HOME veya kardeş ifc_desktop_doc_dataset/)",
    )
    p.add_argument(
        "--kind", choices=["violated", "baseline", "imports", "all"], default="violated",
        help="Hangi türü pakete dahil et (varsayılan: violated)",
    )
    p.add_argument(
        "--include-baseline", action="store_true",
        help="--kind=all ile birlikte baseline/imports'u da ekle (--kind=all gerekiyor)",
    )
    p.add_argument(
        "--include-graph", action="store_true",
        help=".graph.json dosyalarını da pakete koy (varsayılan: hayır)",
    )
    a = p.parse_args()

    if a.data_home:
        os.environ["IFC_DATA_HOME"] = str(Path(a.data_home).expanduser().resolve())

    home = data_home()
    models_root = ifc_models_dir()
    out_zip = Path(a.out).expanduser().resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    kinds = _kinds(a.kind, a.include_baseline)

    manifest: list[dict] = []
    all_labels: list[dict] = []
    n_examples = 0
    n_labels = 0

    print(f"📂 Veri kaynağı: {home}")
    print(f"🎯 Çıkış: {out_zip}")
    print(f"📦 Tür: {', '.join(kinds)} | Graph dahil: {a.include_graph}")
    print()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for kind, ifc in _iter_ifc(models_root, kinds):
            stem = ifc.stem
            arc_ifc = f"examples/{kind}/{stem}.ifc"
            z.write(ifc, arc_ifc)
            n_examples += 1

            labels = _sidecar(ifc, "labels.json")
            meta = _sidecar(ifc, "meta.json")
            graph = _sidecar(ifc, "graph.json")

            entry = {
                "kind": kind,
                "name": stem,
                "ifc": arc_ifc,
                "labels": None,
                "meta": None,
                "graph": None,
                "label_count": 0,
            }

            if labels is not None:
                arc = f"labels/{kind}/{stem}.labels.json"
                z.write(labels, arc)
                entry["labels"] = arc
                recs = _read_labels(labels)
                entry["label_count"] = len(recs)
                n_labels += len(recs)
                all_labels.extend(recs)
            if meta is not None:
                arc = f"meta/{kind}/{stem}.meta.json"
                z.write(meta, arc)
                entry["meta"] = arc
            if a.include_graph and graph is not None:
                arc = f"graphs/{kind}/{stem}.graph.json"
                z.write(graph, arc)
                entry["graph"] = arc

            manifest.append(entry)
            print(f"  ✓ {kind}/{stem}  ({entry['label_count']} label)")

        stats = _stats(all_labels)

        # manifest.json
        z.writestr(
            "manifest.json",
            json.dumps(
                {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "source_data_home": str(home),
                    "kinds": kinds,
                    "include_graph": a.include_graph,
                    "counts": {"examples": n_examples, "labels": n_labels},
                    "stats": stats,
                    "items": manifest,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

        # README.md
        z.writestr(
            "README.md",
            _render_readme(out_zip, kinds, n_examples, n_labels, stats, a.include_graph),
        )

    size_mb = out_zip.stat().st_size / 1024 / 1024
    print()
    print(f"✅ {n_examples} IFC + {n_labels} label → {out_zip}  ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
