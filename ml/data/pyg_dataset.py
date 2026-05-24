"""`torch_geometric.data.InMemoryDataset` over a codex1 dataset root.

Each violated IFC becomes one `Data` graph:
    x          [N, FEATURE_DIM] float32
    edge_index [2, E]           long
    edge_type  [E]              long  — index into EDGE_TYPES
    y          [N]              long  — 0/1
    decoy_mask [N]              bool  — True where a decoy label was attached
    category   list[str | None] (stored as a Python list on the Data object)
    ifc_id, baseline_id        identifiers for splitting / reporting

By default baselines are not included — they contribute only negatives,
and including them dilutes the positive rate. Set
`include_baselines=True` to mix them in (e.g. for graph-level tasks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from .features import FEATURE_DIM, build_node_features
from .graph_loader import EDGE_TYPES, Sample, load_sample
from .sqlite_reader import DatasetReader, iter_violated_with_paths


_EDGE_INDEX = {t: i for i, t in enumerate(EDGE_TYPES)}


def sample_to_data(sample: Sample, *, mask_numeric: bool = False,
                   mask_psets: bool = False, mask_type: bool = False) -> Data:
    """Lift a `Sample` into a PyG `Data` object.

    Mask parametreleri feature leak teşhisi içindir.
    """
    g = sample.graph
    X, node_ids = build_node_features(
        g, mask_numeric=mask_numeric,
        mask_psets=mask_psets, mask_type=mask_type,
    )
    idx_of = {nid: i for i, nid in enumerate(node_ids)}

    src: list[int] = []
    dst: list[int] = []
    et: list[int] = []
    for u, v, ed in g.edges(data=True):
        if u not in idx_of or v not in idx_of:
            continue
        src.append(idx_of[u])
        dst.append(idx_of[v])
        et.append(_EDGE_INDEX.get(ed.get("rel", ""), 0))

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_type = torch.tensor(et, dtype=torch.long) if et else torch.empty((0,), dtype=torch.long)

    y_vec = np.array([sample.y[nid] for nid in node_ids], dtype=np.int64)
    decoy_mask = np.array([nid in sample.decoy_guids for nid in node_ids], dtype=bool)
    categories = [sample.category_per_pos.get(nid) for nid in node_ids]

    data = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.from_numpy(y_vec),
        decoy_mask=torch.from_numpy(decoy_mask),
    )
    # Non-tensor metadata; PyG stores arbitrary attrs on the Data object.
    data.node_ids = node_ids
    data.categories = categories
    data.ifc_id = sample.ifc_id
    data.baseline_id = sample.baseline_id
    data.kind = sample.kind
    return data


class IFCViolationDataset(InMemoryDataset):
    """In-memory PyG dataset built from a codex1 dataset root.

    Args:
        root: directory under which we keep a `cache/` subdir for the
            processed tensor file. Pass any writable path.
        dataset_root: path to the codex1 checkout (the one that owns
            `violation_pool.sqlite` + `ifc_models/`).
    """

    def __init__(
        self,
        root: str | Path,
        dataset_root: str | Path,
        include_baselines: bool = False,
        use_rule_oracle: bool = False,
        mask_numeric_features: bool = False,
        mask_pset_features: bool = False,
        mask_type_features: bool = False,
        transform: Callable | None = None,
    ):
        self._dataset_root = Path(dataset_root).expanduser().resolve()
        self._include_baselines = include_baselines
        self._use_rule_oracle = use_rule_oracle
        self._mask_numeric = mask_numeric_features
        self._mask_psets = mask_pset_features
        self._mask_type = mask_type_features
        super().__init__(str(Path(root)), transform=transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        suffix = ""
        if self._include_baselines:
            suffix += "_with_base"
        if self._use_rule_oracle:
            suffix += "_oracle"
        if self._mask_numeric:
            suffix += "_nonum"
        if self._mask_psets:
            suffix += "_nopset"
        if self._mask_type:
            suffix += "_notype"
        return [f"ifc_violation{suffix}.pt"]

    def download(self) -> None:  # noqa: D401
        return None

    def process(self) -> None:
        from ml.data.rule_oracle import augment_sample  # local import — torch yokken de modül yüklensin

        data_list: list[Data] = []
        oracle_total = 0
        with DatasetReader(self._dataset_root) as reader:
            for entry in iter_violated_with_paths(reader):
                sample = load_sample(
                    entry.graph_path, entry.labels_path, ifc_id=entry.id
                )
                if self._use_rule_oracle:
                    n_added, _ = augment_sample(sample)
                    oracle_total += n_added
                data_list.append(sample_to_data(sample, mask_numeric=self._mask_numeric, mask_psets=self._mask_psets, mask_type=self._mask_type))
            if self._include_baselines:
                for entry in reader.list_baselines():
                    if not (entry.graph_path and entry.graph_path.exists()):
                        continue
                    sample = load_sample(entry.graph_path, None, ifc_id=entry.id)
                    if self._use_rule_oracle:
                        n_added, _ = augment_sample(sample)
                        oracle_total += n_added
                    data_list.append(sample_to_data(sample, mask_numeric=self._mask_numeric, mask_psets=self._mask_psets, mask_type=self._mask_type))
        if self._use_rule_oracle:
            print(f"[oracle] {oracle_total} kural-tabanlı pozitif etiket eklendi "
                  f"({len(data_list)} IFC üzerinden)")
        if not data_list:
            raise RuntimeError(
                f"No violated IFCs with graph+labels found under {self._dataset_root}"
            )
        self.save(data_list, self.processed_paths[0])

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def num_edge_types(self) -> int:
        return len(EDGE_TYPES)
