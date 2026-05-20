"""Node feature engineering.

Layout of the feature vector for a single node (length = FEATURE_DIM):

    [ ifc_type one-hot          ] len(NODE_TYPES)
    [ OverallWidth              ] 1
    [ OverallHeight             ] 1
    [ NominalHeight             ] 1
    [ Elevation                 ] 1
    [ has_pset_DoorCommon       ] 1
    [ has_pset_WindowCommon     ] 1
    [ has_pset_WallCommon       ] 1
    [ IsExternal (0/1, NaN→0)   ] 1
    [ FireRating present (0/1)  ] 1
    [ has_layout (x,y embedded) ] 1

Missing numeric values are imputed with 0.0. We deliberately keep the
feature set small and interpretable for the MVP — Pset bags can be
folded in later via a learned embedding.
"""
from __future__ import annotations

import numpy as np

from .graph_loader import NODE_TYPES


_NUMERIC_ATTRS = ("OverallWidth", "OverallHeight", "NominalHeight", "Elevation")

_PSET_FLAGS = ("Pset_DoorCommon", "Pset_WindowCommon", "Pset_WallCommon")

FEATURE_DIM = (
    len(NODE_TYPES)
    + len(_NUMERIC_ATTRS)
    + len(_PSET_FLAGS)
    + 3  # is_external, has_fire_rating, has_layout
)


_TYPE_INDEX = {t: i for i, t in enumerate(NODE_TYPES)}


def _f(v) -> float:
    """Coerce IFC values to a float; return 0.0 for None / unparsable."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pset_bool(psets: dict, key: str, prop: str) -> int:
    """Return 1 iff psets[key][prop] is truthy."""
    block = psets.get(key) if isinstance(psets, dict) else None
    if not isinstance(block, dict):
        return 0
    v = block.get(prop)
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    s = str(v).strip().lower()
    return 1 if s in {"true", "yes", "1", "t"} else 0


def _pset_present(psets: dict, key: str) -> int:
    if not isinstance(psets, dict):
        return 0
    block = psets.get(key)
    return 1 if isinstance(block, dict) and block else 0


def _has_fire_rating(psets: dict) -> int:
    if not isinstance(psets, dict):
        return 0
    for block in psets.values():
        if isinstance(block, dict) and block.get("FireRating"):
            return 1
    return 0


def _is_external(psets: dict) -> int:
    if not isinstance(psets, dict):
        return 0
    for block in psets.values():
        if isinstance(block, dict) and "IsExternal" in block:
            return _pset_bool({"_": block}, "_", "IsExternal")
    return 0


def build_node_features(g) -> tuple[np.ndarray, list[str]]:
    """Return (X, node_ids). X has shape [N, FEATURE_DIM], rows aligned with node_ids."""
    node_ids = list(g.nodes)
    X = np.zeros((len(node_ids), FEATURE_DIM), dtype=np.float32)
    for i, nid in enumerate(node_ids):
        nd = g.nodes[nid]
        ifc_type = nd.get("ifc_type") or ""
        if ifc_type in _TYPE_INDEX:
            X[i, _TYPE_INDEX[ifc_type]] = 1.0
        off = len(NODE_TYPES)
        attrs = nd.get("attributes") or {}
        for k, name in enumerate(_NUMERIC_ATTRS):
            X[i, off + k] = _f(attrs.get(name))
        off += len(_NUMERIC_ATTRS)
        psets = nd.get("psets") or {}
        for k, name in enumerate(_PSET_FLAGS):
            X[i, off + k] = _pset_present(psets, name)
        off += len(_PSET_FLAGS)
        X[i, off + 0] = _is_external(psets)
        X[i, off + 1] = _has_fire_rating(psets)
        X[i, off + 2] = 1.0 if ("x" in nd and "y" in nd) else 0.0
    return X, node_ids
