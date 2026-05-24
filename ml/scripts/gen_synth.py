"""CLI: Sentetik 2-odalı baseline IFC üret.

Kullanım:
    python -m ml.scripts.gen_synth --n 50
    python -m ml.scripts.gen_synth --n 100 --seed-start 1000 --out custom/dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML.parent))

from paths import ifc_models_dir
from ml.data.synth_baseline import SynthParams, generate_batch


def main() -> None:
    p = argparse.ArgumentParser(description="Sentetik 2-odalı baseline IFC üret")
    p.add_argument("--n", type=int, required=True, help="Üretilecek IFC sayısı")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Çıkış klasörü (default: IFC_DATA_HOME/ifc_models/baseline)")
    p.add_argument("--no-db", action="store_true",
                   help="DB'ye baseline olarak yazma (sadece dosya üret)")
    p.add_argument("--no-windows", action="store_true",
                   help="Pencere ekleme")
    p.add_argument("--tag", default=None,
                   help="Dataset etiketi (default: synth_<timestamp>). "
                        "Eğitimde dataset seçerken kullanılır.")
    a = p.parse_args()

    # Default tag: basic2+1_baseline_vNN (boş bulduğun en küçük indis)
    if a.tag:
        tag = a.tag
    else:
        try:
            from violation_pool import storage as _st
            existing = {t["tag"] for t in _st.list_dataset_tags()}
        except Exception:
            existing = set()
        nxt = 1
        while f"basic2+1_baseline_v{nxt:02d}" in existing:
            nxt += 1
        tag = f"basic2+1_baseline_v{nxt:02d}"
    safe_tag = "".join(c if c.isalnum() or c in "-_+" else "_" for c in tag.strip())
    out = Path(a.out) if a.out else (ifc_models_dir() / "baseline" / safe_tag)
    params = SynthParams(add_windows=not a.no_windows)

    def _cb(i, n, info):
        print(f"  [{i}/{n}] {Path(info['ifc_path']).name}")

    print(f"📦 Dataset etiketi: {safe_tag}")
    print(f"📂 Hedef: {out}")
    print(f"🔢 Üretilecek: {a.n}  (seed başlangıç: {a.seed_start})")
    print()
    results = generate_batch(
        a.n, out,
        seed_start=a.seed_start,
        params=params,
        register_in_db=not a.no_db,
        dataset_tag=safe_tag,
        progress_cb=_cb,
    )
    print()
    print(f"✅ {len(results)} baseline üretildi: {out}")
    if not a.no_db:
        print(f"   DB'ye baseline olarak kaydedildi (kind=baseline, tag={safe_tag}).")


if __name__ == "__main__":
    main()
