"""Print dataset stats without loading torch / PyG.

Useful as a smoke test that the dataset root is wired up correctly,
and as a quick sanity check on label/decoy distributions before
launching a training run.

Usage:
    python scripts/explore.py --dataset-root ~/Desktop/codex1
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ML.parent))

from paths import data_home
from ml.data.graph_loader import load_sample
from ml.data.sqlite_reader import DatasetReader, iter_violated_with_paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=None, help="Varsayılan: IFC_DATA_HOME")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    root = a.dataset_root or str(data_home())
    with DatasetReader(root) as reader:
        entries = list(iter_violated_with_paths(reader))
    if a.limit:
        entries = entries[: a.limit]
    print(f"violated IFCs with graph+labels: {len(entries)}")

    total_nodes = 0
    total_pos = 0
    total_decoy = 0
    cat_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    per_baseline: Counter[str | None] = Counter()

    for e in entries:
        s = load_sample(e.graph_path, e.labels_path, ifc_id=e.id)
        total_nodes += len(s.y)
        total_pos += sum(s.y.values())
        total_decoy += len(s.decoy_guids)
        for guid, cat in s.category_per_pos.items():
            cat_counter[cat] += 1
        for _, nd in s.graph.nodes(data=True):
            type_counter[nd.get("ifc_type", "?")] += 1
        per_baseline[s.baseline_id] += 1

    print(f"nodes total: {total_nodes}")
    print(f"positives:   {total_pos}  ({total_pos / max(1, total_nodes):.3%})")
    print(f"decoys:      {total_decoy}")
    print(f"baselines covered: {len(per_baseline)}")
    print("top node types:")
    for t, n in type_counter.most_common(10):
        print(f"  {t:32s} {n}")
    print("violation categories:")
    for c, n in cat_counter.most_common():
        print(f"  {c:32s} {n}")


if __name__ == "__main__":
    main()
