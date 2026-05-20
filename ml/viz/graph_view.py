"""Interactive graph rendering.

Two flavours:

* `interactive_agraph(...)` — vis-network via `streamlit-agraph`.
  Supports drag-to-rearrange (physics off after layout settles) and
  returns the clicked node id so we can drive cross-highlighting in
  the IFC view. Use this on the main inspect / GAT pages.

* `static_plotly(...)` — pure plotly figure built from the spring
  layout embedded in `graph.json`. No drag, but renders very large
  graphs cheaply and ships path overlays as extra traces. Use this
  for the path-analysis page where the layout must be stable across
  rerenders.

Both renderers share a node-colour scheme so the user can move
between pages without re-learning the legend.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from .ifc3d import (
    COLOR_BASE, COLOR_DECOY, COLOR_PATH, COLOR_PATH_ALT,
    COLOR_SELECTED, COLOR_VIOLATION,
)


_EDGE_COLORS = {
    "aggregates":      "#7f7f7f",
    "contains":        "#1f77b4",
    "bounds":          "#2ca02c",
    "voids":           "#ff7f0e",
    "fills":           "#9467bd",
    "connects":        "#17becf",
    "co_bounds_space": "#bcbd22",
}


def _node_color(
    nid: str,
    *,
    violation_guids: set[str],
    decoy_guids: set[str],
    selected_guid: str | None,
    path_guids: set[str],
    path_alt_guids: set[str],
) -> str:
    if nid == selected_guid:
        return COLOR_SELECTED
    if nid in path_guids:
        return COLOR_PATH
    if nid in path_alt_guids:
        return COLOR_PATH_ALT
    if nid in violation_guids:
        return COLOR_VIOLATION
    if nid in decoy_guids:
        return COLOR_DECOY
    return COLOR_BASE


def interactive_agraph(
    g: nx.MultiDiGraph,
    *,
    violation_guids: Iterable[str] | None = None,
    decoy_guids: Iterable[str] | None = None,
    selected_guid: str | None = None,
    path_guids: Iterable[str] | None = None,
    path_alt_guids: Iterable[str] | None = None,
    height: int = 620,
    physics: bool = True,
    key: str = "agraph",
):
    """Render an editable, draggable graph and return the clicked node id."""
    from streamlit_agraph import Node, Edge, Config, agraph

    vio = set(violation_guids or [])
    dc = set(decoy_guids or [])
    path = set(path_guids or [])
    path_alt = set(path_alt_guids or [])

    nodes: list[Node] = []
    for nid, d in g.nodes(data=True):
        color = _node_color(
            nid,
            violation_guids=vio, decoy_guids=dc,
            selected_guid=selected_guid,
            path_guids=path, path_alt_guids=path_alt,
        )
        label = (d.get("ifc_type") or "?").removeprefix("Ifc")
        name = (d.get("attributes") or {}).get("Name")
        if name:
            label = f"{label}: {name}"
        # Anchor to the embedded spring layout so first render isn't chaotic.
        x = d.get("x")
        y = d.get("y")
        nodes.append(Node(
            id=nid, label=label, color=color, size=18,
            x=float(x) * 600 if x is not None else None,
            y=float(y) * 600 if y is not None else None,
            title=_node_tooltip(nid, d),
        ))

    edges: list[Edge] = []
    # vis-network gets unhappy with parallel edges; collapse multi-edges
    # to one per (u,v,rel) bundle.
    seen: set[tuple[str, str, str]] = set()
    for u, v, ed in g.edges(data=True):
        rel = ed.get("rel", "?")
        key3 = (u, v, rel)
        if key3 in seen:
            continue
        seen.add(key3)
        edges.append(Edge(
            source=u, target=v,
            color=_EDGE_COLORS.get(rel, "#888"),
            label="" if rel == "co_bounds_space" else rel,
        ))

    cfg = Config(
        width="100%", height=height,
        directed=True,
        physics=physics,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#1aa3ff",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"renderLabel": False},
        # The vis-network defaults work; we keep this small on purpose.
    )
    return agraph(nodes=nodes, edges=edges, config=cfg)


def _node_tooltip(nid: str, d: dict) -> str:
    parts = [f"{d.get('ifc_type', '?')} · {nid[:8]}"]
    attrs = d.get("attributes") or {}
    if attrs.get("Name"):
        parts.append(f"name: {attrs['Name']}")
    for k in ("OverallWidth", "OverallHeight", "Elevation"):
        if k in attrs:
            parts.append(f"{k}: {attrs[k]}")
    psets = d.get("psets") or {}
    if psets:
        parts.append("psets: " + ", ".join(list(psets.keys())[:4]))
    return " | ".join(parts)


def static_plotly(
    g: nx.MultiDiGraph,
    *,
    violation_guids: Iterable[str] | None = None,
    decoy_guids: Iterable[str] | None = None,
    selected_guid: str | None = None,
    path_guids: Iterable[str] | None = None,
    path_alt_guids: Iterable[str] | None = None,
    path_edges: list[tuple[str, str]] | None = None,
    path_alt_edges: list[tuple[str, str]] | None = None,
    height: int = 620,
):
    """2D plotly graph view with explicit path-edge overlays."""
    import plotly.graph_objects as go

    vio = set(violation_guids or [])
    dc = set(decoy_guids or [])
    path = set(path_guids or [])
    path_alt = set(path_alt_guids or [])

    pos = {}
    for n, d in g.nodes(data=True):
        if "x" in d and "y" in d:
            pos[n] = (d["x"], d["y"])
    if len(pos) < g.number_of_nodes():
        simple = nx.Graph()
        simple.add_nodes_from(g.nodes())
        simple.add_edges_from((u, v) for u, v, _ in g.edges(data=True))
        pos = nx.spring_layout(simple, seed=7,
                               k=1.4 / max(1, simple.number_of_nodes() ** 0.5))

    # Base edges grouped by relation type.
    by_rel: dict[str, list[tuple[str, str]]] = {}
    for u, v, d in g.edges(data=True):
        by_rel.setdefault(d.get("rel", "?"), []).append((u, v))

    edge_traces = []
    for rel, pairs in by_rel.items():
        xs, ys = [], []
        for u, v in pairs:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=1, color=_EDGE_COLORS.get(rel, "#cccccc")),
            opacity=0.55, hoverinfo="none", name=rel,
        ))

    # Path overlays as their own (thicker, brighter) traces.
    for pairs, color, name in (
        (path_edges or [], COLOR_PATH, "yol"),
        (path_alt_edges or [], COLOR_PATH_ALT, "alt yol"),
    ):
        if not pairs:
            continue
        xs, ys = [], []
        for u, v in pairs:
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=4, color=color),
            opacity=0.95, hoverinfo="none", name=name,
        ))

    node_x, node_y, hover, colors, sizes = [], [], [], [], []
    for n, d in g.nodes(data=True):
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        color = _node_color(n, violation_guids=vio, decoy_guids=dc,
                            selected_guid=selected_guid,
                            path_guids=path, path_alt_guids=path_alt)
        colors.append(color)
        sizes.append(16 if color == COLOR_BASE else 22)
        hover.append(_node_tooltip(n, d) + "<br>" + n)

    fig = go.Figure(data=edge_traces + [go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=sizes, color=colors,
                    line=dict(width=1.2, color="#222")),
        hovertext=hover, hoverinfo="text", name="nodes",
    )])
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        dragmode="pan",
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig
