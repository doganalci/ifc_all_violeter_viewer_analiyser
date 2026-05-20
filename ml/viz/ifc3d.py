"""IFC → plotly 3D mesh viewer with per-GUID colour overlays.

Adapted from codex1's `ifc_viewer.py`. Two extensions compared to the
upstream version:

* `path_guids` — colours a chain of nodes in a distinct hue so we can
  draw a routing result on top of the building.
* Mesh extraction is split from figure assembly so the heavy
  tessellation step can be cached by Streamlit independently of the
  highlight overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Palette
COLOR_BASE = "#9fb3c8"
COLOR_VIOLATION = "#e6194b"
COLOR_DECOY = "#ffbb33"
COLOR_SELECTED = "#1aa3ff"
COLOR_PATH = "#33cc66"
COLOR_PATH_ALT = "#bb66ff"   # second path overlay (for comparisons)


@dataclass
class IFCMesh:
    """One tessellated IFC product, in a form ready to feed plotly.Mesh3d."""

    guid: str
    ifc_type: str
    name: str | None
    xs: list[float]
    ys: list[float]
    zs: list[float]
    i: list[int]
    j: list[int]
    k: list[int]


def extract_meshes(ifc_path: str | Path, max_elements: int = 5000) -> list[IFCMesh]:
    """Tessellate every IfcProduct with a representation. Heavy: cache me."""
    import ifcopenshell
    import ifcopenshell.geom as geom

    f = ifcopenshell.open(str(ifc_path))
    s = geom.settings()
    try:
        s.set(s.USE_WORLD_COORDS, True)
    except Exception:
        pass

    meshes: list[IFCMesh] = []
    for p in f.by_type("IfcProduct"):
        if len(meshes) >= max_elements:
            break
        if not getattr(p, "Representation", None):
            continue
        try:
            shape = geom.create_shape(s, p)
        except Exception:
            continue
        verts = list(shape.geometry.verts)
        faces = list(shape.geometry.faces)
        if not verts or not faces:
            continue
        meshes.append(IFCMesh(
            guid=p.GlobalId,
            ifc_type=p.is_a(),
            name=getattr(p, "Name", None) or None,
            xs=verts[0::3], ys=verts[1::3], zs=verts[2::3],
            i=faces[0::3], j=faces[1::3], k=faces[2::3],
        ))
    return meshes


def build_figure(
    meshes: list[IFCMesh],
    *,
    violation_guids: Iterable[str] | None = None,
    decoy_guids: Iterable[str] | None = None,
    selected_guid: str | None = None,
    path_guids: Iterable[str] | None = None,
    path_alt_guids: Iterable[str] | None = None,
    height: int = 620,
):
    """Assemble a plotly Figure from cached meshes.

    Highlight priority (later overrides earlier): violation < decoy <
    path_alt < path < selected. The selected node always wins visually
    so the user can confirm what they clicked.
    """
    import plotly.graph_objects as go

    vio = set(violation_guids or [])
    dc = set(decoy_guids or [])
    path = set(path_guids or [])
    path_alt = set(path_alt_guids or [])

    traces = []
    for m in meshes:
        if m.guid == selected_guid:
            color, opacity = COLOR_SELECTED, 1.0
            tag = "[SEÇİLİ]"
        elif m.guid in path:
            color, opacity = COLOR_PATH, 1.0
            tag = "[YOL]"
        elif m.guid in path_alt:
            color, opacity = COLOR_PATH_ALT, 1.0
            tag = "[ALT YOL]"
        elif m.guid in vio:
            color, opacity = COLOR_VIOLATION, 1.0
            tag = "[İHLAL]"
        elif m.guid in dc:
            color, opacity = COLOR_DECOY, 1.0
            tag = "[DECOY]"
        else:
            color, opacity, tag = COLOR_BASE, 0.22, ""
        traces.append(go.Mesh3d(
            x=m.xs, y=m.ys, z=m.zs, i=m.i, j=m.j, k=m.k,
            color=color, opacity=opacity, flatshading=True,
            name=m.ifc_type,
            hovertext=f"{tag} {m.ifc_type} · {m.guid} · {m.name or ''}".strip(),
            hoverinfo="text", showscale=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=height,
    )
    return fig
