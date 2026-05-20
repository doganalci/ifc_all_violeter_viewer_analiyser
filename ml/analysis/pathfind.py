"""Room-to-room routing on IFC graphs.

We build a *navigation* projection of the multi-relational graph: the
nodes are IfcSpaces and IfcDoors, and an edge between two spaces is
present when a door fills an opening that voids a wall bounding both
spaces. Each navigation edge carries:

    * `door` — GUID of the IfcDoor mediating the transition
    * `door_width` — `OverallWidth` (metres) for accessibility scoring
    * `is_accessible_width` — door_width >= 0.9 m (TS ISO 21542 baseline)
    * `length` — Euclidean distance between the embedded layout xy of
      the two spaces (so weighted shortest path roughly tracks real
      walking distance)

Three path variants the user can ask for:

    * `shortest_path`   — minimum hop count (BFS).
    * `accessible_path` — minimises Σ accessibility_cost(door) along
      the route. Doors below 0.9 m incur a heavy penalty; missing
      width info adds a small penalty.
    * `widest_path`     — maximum-of-min-door-width (modified Dijkstra).
      Returns the route whose narrowest door is as wide as possible.

All three return a `PathResult` with the GUID chain on the *original*
multigraph, so callers can highlight every IfcSpace **and** every
IfcDoor traversed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx


# A door narrower than this is treated as non-accessible by TS ISO 21542 § 5
ACCESSIBLE_WIDTH_M = 0.90


@dataclass
class PathResult:
    """Outcome of a single routing query.

    `nodes` is the alternating space→door→space chain (every entry is
    a GUID in the original graph); `edges` are the (u, v) pairs the
    UI should highlight; `doors` is just the door subset for stats.
    `metric` reflects what the variant tried to optimise (length /
    cost / bottleneck-width).
    """

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    doors: list[str] = field(default_factory=list)
    metric: float | None = None
    metric_name: str = ""
    found: bool = False
    detail: str = ""

    @property
    def door_widths(self) -> list[float]:
        return [self._w[g] for g in self.doors if g in self._w]

    _w: dict[str, float] = field(default_factory=dict, repr=False)


def list_rooms(g: nx.MultiDiGraph) -> list[dict]:
    """Return IfcSpaces with display-friendly metadata, sorted by storey/name."""
    rooms = []
    for nid, d in g.nodes(data=True):
        if d.get("ifc_type") != "IfcSpace":
            continue
        attrs = d.get("attributes") or {}
        rooms.append({
            "guid": nid,
            "name": attrs.get("LongName") or attrs.get("Name") or nid[:8],
            "elevation": attrs.get("Elevation"),
        })
    rooms.sort(key=lambda r: (r["elevation"] or 0.0, r["name"]))
    return rooms


def _door_width(d: dict) -> float | None:
    attrs = d.get("attributes") or {}
    w = attrs.get("OverallWidth")
    if w is None:
        return None
    try:
        return float(w)
    except (TypeError, ValueError):
        return None


def _euclid(a: dict, b: dict) -> float:
    ax, ay = a.get("x"), a.get("y")
    bx, by = b.get("x"), b.get("y")
    if None in (ax, ay, bx, by):
        return 1.0
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def room_to_room_graph(g: nx.MultiDiGraph) -> nx.Graph:
    """Project the IFC multigraph onto an undirected space-adjacency graph.

    Two spaces are connected iff there is a door (IfcDoor) filling an
    opening (IfcOpeningElement) on a wall that bounds both spaces.
    """
    # Step 1: for every opening, find the wall(s) it voids.
    opening_to_walls: dict[str, list[str]] = {}
    for u, v, ed in g.edges(data=True):
        if ed.get("rel") == "voids":  # wall -> opening
            opening_to_walls.setdefault(v, []).append(u)

    # Step 2: for every door, find the openings it fills → walls.
    door_to_walls: dict[str, list[str]] = {}
    for u, v, ed in g.edges(data=True):
        if ed.get("rel") == "fills":  # door -> opening
            if g.nodes[u].get("ifc_type") not in {"IfcDoor"}:
                continue
            for w in opening_to_walls.get(v, []):
                door_to_walls.setdefault(u, []).append(w)

    # Step 3: for every wall, list bounded spaces.
    wall_to_spaces: dict[str, list[str]] = {}
    for u, v, ed in g.edges(data=True):
        if ed.get("rel") == "bounds":  # wall -> space
            if g.nodes[v].get("ifc_type") == "IfcSpace":
                wall_to_spaces.setdefault(u, []).append(v)

    # Step 4: for every door, the spaces it connects are the *union* of
    # spaces bounded by all of its host walls. Filter to pairs.
    nav = nx.Graph()
    for nid, d in g.nodes(data=True):
        if d.get("ifc_type") == "IfcSpace":
            nav.add_node(nid, **d)

    for door, walls in door_to_walls.items():
        spaces: set[str] = set()
        for w in walls:
            spaces.update(wall_to_spaces.get(w, []))
        spaces = [s for s in spaces if s in nav.nodes]
        if len(spaces) < 2:
            continue
        dw = _door_width(g.nodes[door])
        is_acc = dw is not None and dw >= ACCESSIBLE_WIDTH_M
        # Connect each pair through this door. If many doors join the
        # same two spaces we keep the widest one (best for accessibility).
        for i in range(len(spaces)):
            for j in range(i + 1, len(spaces)):
                a, b = spaces[i], spaces[j]
                length = _euclid(g.nodes[a], g.nodes[b])
                existing = nav.get_edge_data(a, b)
                if existing is not None:
                    if (existing.get("door_width") or 0) >= (dw or 0):
                        continue
                nav.add_edge(a, b, door=door,
                             door_width=dw, is_accessible_width=is_acc,
                             length=length)
    return nav


def _expand(nav_path: list[str], nav: nx.Graph) -> tuple[list[str], list[tuple[str, str]], list[str], dict[str, float]]:
    """Lift a space→space path onto the alternating space→door→space chain."""
    if not nav_path:
        return [], [], [], {}
    nodes: list[str] = [nav_path[0]]
    edges: list[tuple[str, str]] = []
    doors: list[str] = []
    widths: dict[str, float] = {}
    for a, b in zip(nav_path, nav_path[1:]):
        ed = nav.get_edge_data(a, b) or {}
        door = ed.get("door")
        if door:
            nodes.append(door)
            edges.append((a, door))
            edges.append((door, b))
            doors.append(door)
            w = ed.get("door_width")
            if w is not None:
                widths[door] = float(w)
        else:
            edges.append((a, b))
        nodes.append(b)
    return nodes, edges, doors, widths


def shortest_path(g: nx.MultiDiGraph, src: str, dst: str) -> PathResult:
    nav = room_to_room_graph(g)
    if src not in nav or dst not in nav:
        return PathResult(detail="kaynak/hedef oda navigasyon grafında bulunamadı",
                          metric_name="hops")
    try:
        sp = nx.shortest_path(nav, src, dst)
    except nx.NetworkXNoPath:
        return PathResult(detail="iki oda arasında yol yok", metric_name="hops")
    nodes, edges, doors, widths = _expand(sp, nav)
    return PathResult(
        nodes=nodes, edges=edges, doors=doors,
        metric=float(len(sp) - 1), metric_name="hops",
        found=True, _w=widths,
        detail=f"{len(sp) - 1} kapı, {len(sp)} oda",
    )


def accessible_path(g: nx.MultiDiGraph, src: str, dst: str) -> PathResult:
    """Minimise total accessibility cost.

    Each edge cost = length × penalty(door). Penalty is 1 for accessible
    doors, 5 for doors below the 0.9 m threshold, 2 if the width is
    unknown. Falling back to length-only when widths are unavailable
    means the route still goes somewhere sensible.
    """
    nav = room_to_room_graph(g)
    if src not in nav or dst not in nav:
        return PathResult(detail="kaynak/hedef oda navigasyon grafında bulunamadı",
                          metric_name="cost")
    for u, v, ed in nav.edges(data=True):
        w = ed.get("door_width")
        if w is None:
            penalty = 2.0
        elif w >= ACCESSIBLE_WIDTH_M:
            penalty = 1.0
        else:
            penalty = 5.0
        ed["_cost"] = max(0.05, ed.get("length") or 1.0) * penalty
    try:
        sp = nx.shortest_path(nav, src, dst, weight="_cost")
        total = nx.shortest_path_length(nav, src, dst, weight="_cost")
    except nx.NetworkXNoPath:
        return PathResult(detail="iki oda arasında yol yok", metric_name="cost")
    nodes, edges, doors, widths = _expand(sp, nav)
    return PathResult(
        nodes=nodes, edges=edges, doors=doors,
        metric=float(total), metric_name="cost",
        found=True, _w=widths,
        detail=f"{len(doors)} kapı · toplam maliyet {total:.2f}",
    )


def widest_path(g: nx.MultiDiGraph, src: str, dst: str) -> PathResult:
    """Maximum-of-minimum-door-width path (bottleneck path)."""
    nav = room_to_room_graph(g)
    if src not in nav or dst not in nav:
        return PathResult(detail="kaynak/hedef oda navigasyon grafında bulunamadı",
                          metric_name="min_width")
    # Standard transform: maximise min(w) ≡ shortest path with weight = -w
    # on a graph where each edge weight is the door width (unknown → small).
    work = nx.Graph()
    work.add_nodes_from(nav.nodes(data=True))
    for u, v, ed in nav.edges(data=True):
        w = ed.get("door_width") or 0.01
        # Modified Dijkstra below works on min, so we keep widths directly.
        work.add_edge(u, v, w=w, **ed)
    sp = _bottleneck_path(work, src, dst, weight_attr="w")
    if not sp:
        return PathResult(detail="iki oda arasında yol yok", metric_name="min_width")
    nodes, edges, doors, widths = _expand(sp, nav)
    bottleneck = min(widths.values()) if widths else 0.0
    return PathResult(
        nodes=nodes, edges=edges, doors=doors,
        metric=float(bottleneck), metric_name="min_width",
        found=True, _w=widths,
        detail=f"darboğaz kapı: {bottleneck:.2f} m · {len(doors)} kapı",
    )


def _bottleneck_path(g: nx.Graph, src: str, dst: str, weight_attr: str) -> list[str]:
    """Return a path src→dst maximising the minimum edge weight."""
    import heapq

    # Best-known minimum on any path to a given node so far (start at -inf).
    best: dict[str, float] = {n: -math.inf for n in g.nodes}
    prev: dict[str, str | None] = {n: None for n in g.nodes}
    best[src] = math.inf
    heap: list[tuple[float, str]] = [(-math.inf, src)]
    while heap:
        neg_w, u = heapq.heappop(heap)
        if -neg_w < best[u] - 1e-12:
            continue
        if u == dst:
            break
        for v in g.neighbors(u):
            w = float(g.edges[u, v][weight_attr])
            cand = min(best[u], w)
            if cand > best[v] + 1e-12:
                best[v] = cand
                prev[v] = u
                heapq.heappush(heap, (-cand, v))
    if best[dst] == -math.inf:
        return []
    out: list[str] = []
    cur: str | None = dst
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    out.reverse()
    return out


def exits_for(g: nx.MultiDiGraph) -> list[str]:
    """Heuristic: external doors (IsExternal=True) are candidate exits.

    Useful for the 'shortest path to nearest exit' query. We don't
    insert virtual exit nodes; we just return the door GUIDs.
    """
    out: list[str] = []
    for nid, d in g.nodes(data=True):
        if d.get("ifc_type") != "IfcDoor":
            continue
        psets = d.get("psets") or {}
        for block in psets.values():
            if isinstance(block, dict) and str(block.get("IsExternal")).lower() in {"true", "1"}:
                out.append(nid)
                break
    return out


def nearest_exit(g: nx.MultiDiGraph, src_space: str,
                 variant: str = "accessible") -> tuple[PathResult, str | None]:
    """Find the path from a room to the closest external door.

    Returns the best `PathResult` and the door guid used (or None).
    """
    exits = exits_for(g)
    if not exits:
        return PathResult(detail="dış kapı bulunamadı", metric_name=variant), None

    fn = {"shortest": shortest_path, "accessible": accessible_path,
          "widest": widest_path}.get(variant, accessible_path)
    nav = room_to_room_graph(g)
    # Each exit door is on one or more walls bounding some space; route
    # to whichever of those spaces gives the best path, then append the
    # door to the chain.
    door_to_walls: dict[str, list[str]] = {}
    for u, v, ed in g.edges(data=True):
        if ed.get("rel") == "fills" and g.nodes[u].get("ifc_type") == "IfcDoor":
            door_to_walls.setdefault(u, []).append(v)
    wall_to_spaces: dict[str, list[str]] = {}
    for u, v, ed in g.edges(data=True):
        if ed.get("rel") == "bounds" and g.nodes[v].get("ifc_type") == "IfcSpace":
            wall_to_spaces.setdefault(u, []).append(v)

    best: PathResult | None = None
    best_door: str | None = None
    for door in exits:
        # opening guids the door fills, then walls voided by those openings:
        opening_guids = [v for u, v, ed in g.edges(data=True)
                         if ed.get("rel") == "fills" and u == door]
        wall_guids = []
        for op in opening_guids:
            for u, v, ed in g.edges(data=True):
                if ed.get("rel") == "voids" and v == op:
                    wall_guids.append(u)
        cand_spaces: set[str] = set()
        for w in wall_guids:
            cand_spaces.update(wall_to_spaces.get(w, []))
        for sp in cand_spaces:
            res = fn(g, src_space, sp)
            if not res.found:
                continue
            # Tack the exit door onto the chain so it shows up in highlights.
            res.nodes = res.nodes + [door]
            res.edges = res.edges + [(sp, door)]
            res.doors = res.doors + [door]
            if best is None or _better(res, best, variant):
                best = res
                best_door = door
    if best is None:
        return PathResult(detail="dış kapıya erişilebilir yol bulunamadı",
                          metric_name=variant), None
    return best, best_door


def _better(a: PathResult, b: PathResult, variant: str) -> bool:
    if a.metric is None or b.metric is None:
        return False
    if variant == "widest":
        return a.metric > b.metric           # bigger min-width is better
    return a.metric < b.metric               # smaller hops/cost is better
