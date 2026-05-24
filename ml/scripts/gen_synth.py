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
    a = p.parse_args()

    out = Path(a.out) if a.out else (ifc_models_dir() / "baseline")
    params = SynthParams(add_windows=not a.no_windows)

    def _cb(i, n, info):
        print(f"  [{i}/{n}] {Path(info['ifc_path']).name}")

    print(f"📂 Hedef: {out}")
    print(f"🔢 Üretilecek: {a.n}  (seed başlangıç: {a.seed_start})")
    print()
    results = generate_batch(
        a.n, out,
        seed_start=a.seed_start,
        params=params,
        register_in_db=not a.no_db,
        progress_cb=_cb,
    )
    print()
    print(f"✅ {len(results)} baseline üretildi: {out}")
    if not a.no_db:
        print("   DB'ye baseline olarak kaydedildi (kind=baseline).")


if __name__ == "__main__":
    main()
