"""IFC → plotly Mesh3d. Codex1'in ifc_viewer.py'sinin read-only kopyası,
GUID bazlı renklendirme (violation=kırmızı, decoy=sarı, normal=soluk)."""
from __future__ import annotations

from pathlib import Path


def ifc_to_figure(
    ifc_path: str | Path,
    highlight_guids: set[str] | None = None,
    decoy_guids: set[str] | None = None,
    max_elements: int = 5000,
):
    import plotly.graph_objects as go
    try:
        import ifcopenshell
        import ifcopenshell.geom as geom
    except Exception as e:
        raise ValueError(
            "ifcopenshell.geom (OpenCascade) kullanılamıyor. "
            "Conda'da: `conda install -c conda-forge ifcopenshell`. "
            f"Detay: {e}"
        )

    try:
        f = ifcopenshell.open(str(ifc_path))
    except Exception as e:
        raise ValueError(f"IFC açılamadı: {e}")
    s = geom.settings()
    try:
        s.set(s.USE_WORLD_COORDS, True)
    except Exception:
        pass

    hl = set(highlight_guids or [])
    dc = set(decoy_guids or [])
    meshes = []
    drawn = 0
    skipped = 0
    for p in f.by_type("IfcProduct"):
        if drawn >= max_elements:
            break
        if not getattr(p, "Representation", None):
            continue
        try:
            shape = geom.create_shape(s, p)
        except Exception:
            skipped += 1
            continue
        verts = list(shape.geometry.verts)
        faces = list(shape.geometry.faces)
        if not verts or not faces:
            continue
        xs, ys, zs = verts[0::3], verts[1::3], verts[2::3]
        ii, jj, kk = faces[0::3], faces[1::3], faces[2::3]
        is_hl = p.GlobalId in hl
        is_decoy = p.GlobalId in dc and not is_hl
        if is_hl:
            color, opacity, tag = "#e6194b", 1.0, "[İHLAL]"
        elif is_decoy:
            color, opacity, tag = "#ffbb33", 1.0, "[DECOY]"
        else:
            color, opacity, tag = "#9fb3c8", 0.25, ""
        meshes.append(
            go.Mesh3d(
                x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                color=color, opacity=opacity, flatshading=True,
                name=p.is_a(),
                hovertext=(f"{tag} {p.is_a()} · {p.GlobalId} · "
                           f"{getattr(p, 'Name', '') or ''}"),
                hoverinfo="text",
                showscale=False,
            )
        )
        drawn += 1

    if not meshes:
        raise ValueError(
            "Çizilebilecek geometri bulunamadı (IfcExtrudedAreaSolid / "
            "IfcShapeRepresentation içeren eleman yok)."
        )

    fig = go.Figure(data=meshes)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=620,
    )
    return fig, {"drawn": drawn, "skipped": skipped}
