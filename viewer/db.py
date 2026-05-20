"""violation_pool.sqlite üzerinde read-only sorgular.

Codex1'in storage.py'siyle aynı şemayı bekler ama yazma yapmaz; bağlantıyı
`mode=ro` URI ile açar. Streamlit önbelleğine uygundur (basit dict/list döner).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _ro(db_path: Path):
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows(conn, q: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def list_ifc_models(db_path: Path, kind: str | None = None) -> list[dict]:
    q = "SELECT * FROM ifc_models"
    args: tuple = ()
    if kind:
        q += " WHERE kind=?"
        args = (kind,)
    q += " ORDER BY datetime(created_at) DESC"
    with _ro(db_path) as c:
        return _rows(c, q, args)


def get_ifc_model(db_path: Path, ifc_id: str) -> dict | None:
    with _ro(db_path) as c:
        r = c.execute("SELECT * FROM ifc_models WHERE id=?", (ifc_id,)).fetchone()
    return dict(r) if r else None


def list_children(db_path: Path, parent_id: str) -> list[dict]:
    """parent_id'ye bağlı violated IFC'ler (baseline → violated türevleri)."""
    with _ro(db_path) as c:
        return _rows(
            c,
            "SELECT * FROM ifc_models WHERE parent_id=? ORDER BY datetime(created_at) DESC",
            (parent_id,),
        )


def list_ifc_labels(db_path: Path, ifc_id: str) -> list[dict]:
    with _ro(db_path) as c:
        return _rows(
            c,
            "SELECT * FROM ifc_violation_labels WHERE ifc_model_id=? ORDER BY applied_at",
            (ifc_id,),
        )


def count_labels_per_ifc(db_path: Path) -> dict[str, dict[str, int]]:
    """ifc_id → {'total', 'applied', 'decoy', 'skipped'}."""
    with _ro(db_path) as c:
        rows = c.execute(
            """SELECT ifc_model_id,
                      COUNT(*) total,
                      SUM(CASE WHEN status='applied' AND is_decoy=0 THEN 1 ELSE 0 END) applied,
                      SUM(CASE WHEN is_decoy=1 THEN 1 ELSE 0 END) decoy,
                      SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) skipped
               FROM ifc_violation_labels GROUP BY ifc_model_id"""
        ).fetchall()
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        out[r["ifc_model_id"]] = {
            "total": r["total"] or 0,
            "applied": r["applied"] or 0,
            "decoy": r["decoy"] or 0,
            "skipped": r["skipped"] or 0,
        }
    return out


def list_runs(db_path: Path, only_saved: bool = False) -> list[dict]:
    q = "SELECT * FROM runs"
    if only_saved:
        q += " WHERE status='saved'"
    q += " ORDER BY datetime(updated_at) DESC"
    with _ro(db_path) as c:
        rows = _rows(c, q)
    for r in rows:
        try:
            r["rag_documents"] = json.loads(r.get("rag_documents") or "[]")
        except Exception:
            r["rag_documents"] = []
    return rows


def get_run(db_path: Path, run_id: str) -> dict | None:
    with _ro(db_path) as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    try:
        d["rag_documents"] = json.loads(d.get("rag_documents") or "[]")
    except Exception:
        d["rag_documents"] = []
    return d


def get_violations(db_path: Path, run_id: str) -> list[dict]:
    with _ro(db_path) as c:
        viols = _rows(
            c,
            "SELECT * FROM violations WHERE run_id=? ORDER BY created_at",
            (run_id,),
        )
        for v in viols:
            v["evidence"] = _rows(
                c,
                "SELECT document,page,clause,snippet FROM evidence WHERE violation_id=?",
                (v["id"],),
            )
    return viols


def get_violation(db_path: Path, violation_id: str) -> dict | None:
    with _ro(db_path) as c:
        r = c.execute("SELECT * FROM violations WHERE id=?", (violation_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["evidence"] = _rows(
            c,
            "SELECT document,page,clause,snippet FROM evidence WHERE violation_id=?",
            (violation_id,),
        )
    return d


def list_usage(db_path: Path, **filters: Any) -> list[dict]:
    where, args = [], []
    for k in ("pool_run_id", "ifc_model_id", "collection", "operation"):
        v = filters.get(k)
        if v:
            where.append(f"{k}=?")
            args.append(v)
    q = "SELECT * FROM llm_usage"
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY datetime(created_at) DESC"
    with _ro(db_path) as c:
        return _rows(c, q, tuple(args))


def usage_totals(db_path: Path, **filters: Any) -> dict[str, int]:
    rows = list_usage(db_path, **filters)
    return {
        "calls": len(rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),
    }


def db_summary(db_path: Path) -> dict[str, int]:
    with _ro(db_path) as c:
        def n(q: str) -> int:
            return int(c.execute(q).fetchone()[0])
        return {
            "runs": n("SELECT COUNT(*) FROM runs"),
            "violations": n("SELECT COUNT(*) FROM violations"),
            "baseline": n("SELECT COUNT(*) FROM ifc_models WHERE kind='baseline'"),
            "violated": n("SELECT COUNT(*) FROM ifc_models WHERE kind='violated'"),
            "imported": n("SELECT COUNT(*) FROM ifc_models WHERE kind='imported'"),
            "labels": n("SELECT COUNT(*) FROM ifc_violation_labels"),
            "decoys": n("SELECT COUNT(*) FROM ifc_violation_labels WHERE is_decoy=1"),
            "llm_calls": n("SELECT COUNT(*) FROM llm_usage"),
            "tokens": n("SELECT COALESCE(SUM(total_tokens),0) FROM llm_usage"),
        }
