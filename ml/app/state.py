"""Shared session state + cached dataset accessors for the Streamlit app.

All pages call into these helpers so the dataset root and currently
selected model are consistent across tabs, and heavy operations
(opening the sqlite, tessellating IFC geometry, loading large graphs)
happen once per resource.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

import sys

# Allow `from data import ...` when this module runs from `ml/`.
_ML_ROOT = Path(__file__).resolve().parents[1]
_PROG_ROOT = _ML_ROOT.parent
sys.path.insert(0, str(_ML_ROOT))
sys.path.insert(0, str(_PROG_ROOT))

from data.graph_loader import load_sample  # ml/data/graph_loader.py
from data.sqlite_reader import DatasetReader  # ml/data/sqlite_reader.py
from paths import data_home  # konsolide veri klasörü


_DATASET_ROOT_KEY = "ifc_ga.dataset_root"
_MODEL_ID_KEY = "ifc_ga.model_id"
_SELECTED_NODE_KEY = "ifc_ga.selected_node"
_BROWSE_PATH_KEY = "ifc_ga.browse_path"


def default_dataset_root() -> str:
    """Konsolide veri klasörü; eski `IFC_DATASET_ROOT` override hâlâ geçerli."""
    legacy = os.environ.get("IFC_DATASET_ROOT", "").strip()
    return legacy or str(data_home())


def get_dataset_root() -> str:
    return st.session_state.get(_DATASET_ROOT_KEY, default_dataset_root())


def set_dataset_root(p: str) -> None:
    st.session_state[_DATASET_ROOT_KEY] = p
    # New root → existing list cache is stale.
    list_entries.clear()  # type: ignore[attr-defined]


def get_selected_model_id() -> str | None:
    return st.session_state.get(_MODEL_ID_KEY)


def set_selected_model_id(mid: str | None) -> None:
    st.session_state[_MODEL_ID_KEY] = mid


def get_selected_node() -> str | None:
    return st.session_state.get(_SELECTED_NODE_KEY)


def set_selected_node(guid: str | None) -> None:
    st.session_state[_SELECTED_NODE_KEY] = guid


# ----- Cached dataset readers ------------------------------------------------

def _mtime(p: str | os.PathLike | None) -> float:
    if not p:
        return 0.0
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def _open_reader(root: str) -> DatasetReader:
    return DatasetReader(root)


def reader_for(root: str | None = None) -> DatasetReader:
    return _open_reader(root or get_dataset_root())


@st.cache_data(show_spinner=False)
def list_entries(root: str, kind: str | None = None,
                 require_graph: bool = False) -> list[dict]:
    """Return IFC entries as plain dicts.

    By default we list every entry whose `status` is 'ok' or 'partial' —
    even ones without an on-disk graph yet — so the picker exposes all
    of them. Pages that need a graph filter with `require_graph=True`.
    """
    rd = _open_reader(root)
    entries = rd.list_models(kind=kind, status="ok") if kind else rd.list_models(status="ok")
    # Also pick up entries with status='partial' (a violation run that
    # partly succeeded still has a valid IFC / graph).
    if kind:
        entries += rd.list_models(kind=kind, status="partial")
    else:
        entries += rd.list_models(status="partial")
    seen: set[str] = set()
    out: list[dict] = []
    for e in entries:
        if e.id in seen:
            continue
        seen.add(e.id)
        graph_ok = bool(e.graph_path and e.graph_path.exists())
        if require_graph and not graph_ok:
            continue
        out.append({
            "id": e.id,
            "kind": e.kind,
            "name": e.name,
            "parent_id": e.parent_id,
            "pool_run_id": e.pool_run_id,
            "ifc_path": str(e.ifc_path) if e.ifc_path else None,
            "graph_path": str(e.graph_path) if e.graph_path else None,
            "labels_path": str(e.labels_path) if e.labels_path else None,
            "meta_path": str(e.meta_path) if e.meta_path else None,
            "status": e.status,
            "graph_ok": graph_ok,
        })
    return out


def entry_by_id(root: str, model_id: str) -> dict | None:
    for e in list_entries(root, kind=None):
        if e["id"] == model_id:
            return e
    return None


def violated_children(root: str, baseline_id: str) -> list[dict]:
    """Return all violated entries whose parent_id is the given baseline."""
    return [
        e for e in list_entries(root, kind=None)
        if e["kind"] == "violated" and e.get("parent_id") == baseline_id
    ]


@st.cache_data(show_spinner=False)
def load_sample_cached(graph_path: str, labels_path: str | None,
                       graph_mtime: float, labels_mtime: float,
                       ifc_id: str) -> Any:
    return load_sample(graph_path, labels_path, ifc_id=ifc_id)


def load_sample_for(entry: dict):
    if not entry.get("graph_ok"):
        return None
    return load_sample_cached(
        entry["graph_path"], entry.get("labels_path"),
        _mtime(entry["graph_path"]), _mtime(entry.get("labels_path")),
        entry["id"],
    )


# ----- Folder browser --------------------------------------------------------

def _safe_subdirs(p: Path) -> list[Path]:
    try:
        return sorted(
            [c for c in p.iterdir() if c.is_dir() and not c.name.startswith(".")],
            key=lambda c: c.name.lower(),
        )
    except (PermissionError, OSError):
        return []


def folder_browser(start: str | Path | None = None,
                   *, key: str = "browser") -> str | None:
    """Tıklanabilir klasör gezgini. `Bu klasörü kullan` butonu basıldığında
    seçilen path string'ini döner, aksi halde None.

    Sidebar veya ana panele konabilir; her instance ayrı `key` ister.
    """
    cur = Path(st.session_state.get(_BROWSE_PATH_KEY, start or Path.home())).expanduser()
    if not cur.exists():
        cur = Path.home()

    has_db = (cur / "violation_pool.sqlite").exists()
    st.code(str(cur), language=None)

    quick = st.columns(3)
    if quick[0].button("🏠 Home", key=f"{key}_home", use_container_width=True):
        st.session_state[_BROWSE_PATH_KEY] = str(Path.home())
        st.rerun()
    if quick[1].button("🖥️ Desktop", key=f"{key}_desk", use_container_width=True):
        st.session_state[_BROWSE_PATH_KEY] = str(Path.home() / "Desktop")
        st.rerun()
    if quick[2].button("⬆️ ..", key=f"{key}_up",
                       use_container_width=True,
                       disabled=cur.parent == cur):
        st.session_state[_BROWSE_PATH_KEY] = str(cur.parent)
        st.rerun()

    if has_db:
        if st.button("✅ Bu klasörü dataset olarak kullan",
                     key=f"{key}_use", type="primary",
                     use_container_width=True):
            return str(cur)
    else:
        st.caption("⚠️ Bu klasörde `violation_pool.sqlite` yok.")

    subs = _safe_subdirs(cur)
    if not subs:
        st.caption("— alt klasör yok —")
    for d in subs[:50]:
        marker = "📊" if (d / "violation_pool.sqlite").exists() else "📁"
        if st.button(f"{marker} {d.name}", key=f"{key}_{d.name}",
                     use_container_width=True):
            st.session_state[_BROWSE_PATH_KEY] = str(d)
            st.rerun()
    if len(subs) > 50:
        st.caption(f"...ve {len(subs) - 50} klasör daha (gizli)")
    return None


# ----- Sidebar UI ------------------------------------------------------------

_KIND_LABEL = {
    "baseline": "📐 Baseline (üretilen)",
    "violated": "⚠️ Violated (ihlal enjekte)",
    "imported": "📥 Imported (dışarıdan)",
}


def sidebar_config() -> dict | None:
    """Standard sidebar shown on every page.

    Returns the entry dict for the chosen model, or None if no valid
    choice yet.
    """
    st.sidebar.title("IFC Graph Analysis")
    st.sidebar.caption("codex1 dataset üzerinde GAT + analiz aracı")

    current = get_dataset_root()
    root = st.sidebar.text_input(
        "Dataset klasörü",
        value=current,
        help="`violation_pool.sqlite` ve `ifc_models/` içeren codex1 klasörü.",
    )
    if root != current:
        set_dataset_root(root)

    # Klasör browser
    with st.sidebar.expander("📁 Klasör seç (gez)", expanded=False):
        picked = folder_browser(start=root, key="sidebar_browser")
        if picked:
            set_dataset_root(picked)
            st.rerun()

    if not Path(root).expanduser().exists():
        st.sidebar.error(f"Klasör bulunamadı: {root}")
        return None
    if not (Path(root).expanduser() / "violation_pool.sqlite").exists():
        st.sidebar.error("violation_pool.sqlite bulunamadı.")
        return None

    try:
        entries = list_entries(root, kind=None)
    except Exception as e:
        st.sidebar.error(f"DB okunamadı: {e}")
        return None
    if not entries:
        st.sidebar.warning("Henüz yüklü IFC modeli yok.")
        return None

    # Group by kind so violated + imported always show as options even
    # when most entries are baselines.
    kinds_present = sorted({e["kind"] for e in entries})
    kind_options = [k for k in ("baseline", "violated", "imported") if k in kinds_present]
    default_kind = "violated" if "violated" in kind_options else kind_options[0]
    kind = st.sidebar.selectbox(
        "Tür",
        options=kind_options,
        index=kind_options.index(default_kind),
        format_func=lambda k: _KIND_LABEL.get(k, k),
    )
    filtered = [e for e in entries if e["kind"] == kind]
    if not filtered:
        st.sidebar.warning(f"`{kind}` türünde model yok.")
        return None

    def _row_label(e: dict) -> str:
        ok = "" if e["graph_ok"] else "  ⚠️ graph yok"
        return f"{e['name']}  ·  {e['id'][:8]}{ok}"

    labels_list = [_row_label(e) for e in filtered]
    current_id = get_selected_model_id()
    idx = next((i for i, e in enumerate(filtered) if e["id"] == current_id), 0)
    sel = st.sidebar.selectbox(
        "IFC modeli",
        options=list(range(len(filtered))),
        index=min(idx, len(filtered) - 1),
        format_func=lambda i: labels_list[i],
    )
    entry = filtered[sel]
    set_selected_model_id(entry["id"])

    st.sidebar.divider()
    st.sidebar.caption(f"📄 {Path(entry['ifc_path']).name if entry['ifc_path'] else '—'}")
    if entry["parent_id"]:
        st.sidebar.caption(f"baseline → `{entry['parent_id'][:8]}`")
    if not entry["graph_ok"]:
        st.sidebar.warning(
            "Bu model için graph.json henüz üretilmemiş; "
            "graph/analiz fonksiyonları devre dışı."
        )
    return entry


def labels_summary(entry: dict) -> dict | None:
    """Return the labels.json doc for a violated IFC, or None."""
    lp = entry.get("labels_path")
    if not lp or not Path(lp).exists():
        return None
    try:
        return json.loads(Path(lp).read_text(encoding="utf-8"))
    except Exception:
        return None
