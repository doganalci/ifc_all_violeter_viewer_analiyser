"""Data layer.

`graph_loader`, `features`, `splits` and `sqlite_reader` are pure
Python / NumPy / NetworkX and import freely. `pyg_dataset` is exposed
lazily so the rest of the package can be used in environments that
don't have torch installed (e.g. quick CLI explore).
"""
from .graph_loader import (
    EDGE_TYPES,
    NODE_TYPES,
    Sample,
    labels_to_targets,
    load_graph,
    load_labels,
    load_sample,
)
from .features import FEATURE_DIM, build_node_features
from .sqlite_reader import DatasetReader, IFCEntry, iter_violated_with_paths
from .splits import SplitIndices, split_by_baseline


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "IFCViolationDataset":
        from .pyg_dataset import IFCViolationDataset
        return IFCViolationDataset
    if name == "sample_to_data":
        from .pyg_dataset import sample_to_data
        return sample_to_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EDGE_TYPES",
    "NODE_TYPES",
    "Sample",
    "labels_to_targets",
    "load_graph",
    "load_labels",
    "load_sample",
    "FEATURE_DIM",
    "build_node_features",
    "DatasetReader",
    "IFCEntry",
    "iter_violated_with_paths",
    "SplitIndices",
    "split_by_baseline",
    "IFCViolationDataset",
    "sample_to_data",
]
