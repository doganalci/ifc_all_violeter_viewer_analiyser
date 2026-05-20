"""Read-only access to codex1's `violation_pool.sqlite`.

Adapted from `ifc_and_graph_viewer/viewer/db.py`. We never write to the
DB and open the file with `mode=ro` so concurrent writes from the
generator are safe.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class IFCEntry:
    """One row of `ifc_models`, with absolute file paths resolved."""

    id: str
    kind: str  # 'baseline' | 'violated' | 'imported'
    name: str
    parent_id: str | None
    pool_run_id: str | None
    status: str
    ifc_path: Path | None
    graph_path: Path | None
    labels_path: Path | None
    meta_path: Path | None


class DatasetReader:
    """Read-only view over a codex1 dataset root.

    The root must contain `violation_pool.sqlite` and (typically) an
    `ifc_models/` subtree. Paths stored in the DB may be absolute or
    relative; we resolve relative paths against the dataset root.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        db = self.root / "violation_pool.sqlite"
        if not db.exists():
            raise FileNotFoundError(f"violation_pool.sqlite not found under {self.root}")
        # mode=ro guarantees we cannot mutate the file even by accident.
        uri = f"file:{db}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DatasetReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _resolve(self, p: str | None) -> Path | None:
        if not p:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = self.root / path
        return path

    def _row_to_entry(self, row: sqlite3.Row) -> IFCEntry:
        return IFCEntry(
            id=row["id"],
            kind=row["kind"],
            name=row["name"],
            parent_id=row["parent_id"],
            pool_run_id=row["pool_run_id"],
            status=row["status"],
            ifc_path=self._resolve(row["file_path"]),
            graph_path=self._resolve(row["graph_path"]),
            labels_path=self._resolve(row["labels_path"]),
            meta_path=self._resolve(row["meta_path"]),
        )

    def list_models(self, kind: str | None = None, status: str = "ok") -> list[IFCEntry]:
        sql = "SELECT * FROM ifc_models WHERE status=?"
        args: list = [status]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += " ORDER BY created_at"
        return [self._row_to_entry(r) for r in self._conn.execute(sql, args)]

    def list_violated(self) -> list[IFCEntry]:
        # status='ok' (tüm ihlaller uygulandı) + 'partial' (bazıları
        # atlandı ama IFC+graph+labels yine de geçerli) — ikisi de
        # eğitim için kullanılabilir.
        out: list[IFCEntry] = []
        for status in ("ok", "partial"):
            out.extend(self.list_models(kind="violated", status=status))
        return out

    def list_baselines(self) -> list[IFCEntry]:
        return self.list_models(kind="baseline")

    def get_model(self, ifc_id: str) -> IFCEntry | None:
        row = self._conn.execute(
            "SELECT * FROM ifc_models WHERE id=?", (ifc_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def labels_for(self, ifc_id: str) -> list[dict]:
        """Return ifc_violation_labels rows for an IFC, as plain dicts."""
        rows = self._conn.execute(
            "SELECT * FROM ifc_violation_labels WHERE ifc_model_id=? "
            "ORDER BY applied_at",
            (ifc_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_pool_runs(self) -> list[dict]:
        """All violation pool runs (one row per LLM-generated label set).

        We attach the count of violated IFCs that reference each run so
        the UI can show '12 IFC' next to the pool name.
        """
        rows = self._conn.execute(
            "SELECT id, name, method, status, llm_model, created_at "
            "FROM runs WHERE status='saved' ORDER BY created_at DESC"
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            n = self._conn.execute(
                "SELECT COUNT(*) FROM ifc_models WHERE pool_run_id=?",
                (r["id"],),
            ).fetchone()[0]
            d = dict(r)
            d["n_violated"] = int(n)
            out.append(d)
        return out

    def baseline_to_violated(self) -> dict[str, list[str]]:
        """Map baseline_id -> [violated_id, ...]. Useful for splits."""
        out: dict[str, list[str]] = {}
        for r in self._conn.execute(
            "SELECT id, parent_id FROM ifc_models "
            "WHERE kind='violated' AND status='ok' AND parent_id IS NOT NULL"
        ):
            out.setdefault(r["parent_id"], []).append(r["id"])
        return out


def iter_violated_with_paths(reader: DatasetReader) -> Iterable[IFCEntry]:
    """Yield only entries whose graph + labels files actually exist on disk."""
    for e in reader.list_violated():
        if e.graph_path and e.graph_path.exists() and e.labels_path and e.labels_path.exists():
            yield e
