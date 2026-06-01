"""Load a single (graph.json, labels.json) pair into a NetworkX graph
with per-node ground-truth attached.

The label semantics follow `codex1/violation_pool/ifc_inject.py`:
    y=1  iff  status == 'applied' AND is_decoy is False
    decoy nodes (status='decoy', is_decoy=True) → y=0, but tracked
        separately so we can compute `decoy_fpr` (the rate at which
        the model is fooled into flagging them).

Baseline graphs have no labels file — every node is negative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


# Keep this order frozen: it is the canonical one-hot index for
# `features.build_node_features`. New IFC types must be appended,
# never inserted, so saved checkpoints stay compatible.
NODE_TYPES: tuple[str, ...] = (
    "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace",
    "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcCovering",
    "IfcDoor", "IfcWindow", "IfcStair", "IfcStairFlight", "IfcRailing",
    "IfcRamp", "IfcColumn", "IfcBeam", "IfcOpeningElement",
)

EDGE_TYPES: tuple[str, ...] = (
    "aggregates", "contains", "bounds", "voids", "fills",
    "connects", "co_bounds_space",
)


@dataclass
class Sample:
    """One IFC model lifted into graph form, ready for feature/PyG conversion."""

    ifc_id: str
    graph: nx.MultiDiGraph
    y: dict[str, int]                # guid -> 0/1
    decoy_guids: set[str] = field(default_factory=set)
    category_per_pos: dict[str, str] = field(default_factory=dict)
    baseline_id: str | None = None
    kind: str = "violated"           # 'violated' | 'baseline'


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="links", multigraph=True, directed=True)


def load_labels(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def labels_to_targets(
    g: nx.MultiDiGraph, labels_doc: dict
) -> tuple[dict[str, int], set[str], dict[str, str]]:
    """Project the labels document onto graph node ids.

    Returns:
        y: guid -> 0/1, defined for every node in the graph.
        decoy_guids: nodes carrying a decoy label (kept negative in y).
        category_per_pos: positive guid -> rule category (for per-category recall).
    """
    y: dict[str, int] = {n: 0 for n in g.nodes}
    decoys: set[str] = set()
    cats: dict[str, str] = {}
    for lab in labels_doc.get("labels", []):
        guid = lab.get("ifc_global_id")
        if not guid or guid not in y:
            continue
        is_decoy = bool(lab.get("is_decoy"))
        status = (lab.get("status") or "").lower()
        if is_decoy or status == "decoy":
            decoys.add(guid)
            continue
        if status == "applied":
            y[guid] = 1
            if lab.get("category"):
                cats[guid] = lab["category"]
    return y, decoys, cats


def load_sample(
    graph_path: str | Path,
    labels_path: str | Path | None,
    ifc_id: str | None = None,
) -> Sample:
    """Load one IFC into a `Sample`. `labels_path=None` ⇒ all negatives."""
    g = load_graph(graph_path)
    if labels_path is None:
        y = {n: 0 for n in g.nodes}
        return Sample(
            ifc_id=ifc_id or Path(graph_path).stem.replace(".graph", ""),
            graph=g,
            y=y,
            kind="baseline",
        )
    doc = load_labels(labels_path)
    y, decoys, cats = labels_to_targets(g, doc)
    return Sample(
        ifc_id=ifc_id or doc.get("violated_id") or Path(graph_path).stem.replace(".graph", ""),
        graph=g,
        y=y,
        decoy_guids=decoys,
        category_per_pos=cats,
        baseline_id=doc.get("baseline_id"),
        kind="violated",
    )
