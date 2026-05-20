"""Erişilebilirlik Dataset Viewer — codex1 violation_pool dataset'i için
read-only Streamlit inceleyici."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from viewer import db
from viewer.config import (
    DatasetPaths,
    default_root,
    find_run_export,
    first_valid_codex1,
    resolve_artifact,
)

st.set_page_config(
    page_title="Erişilebilirlik Dataset Viewer",
    page_icon="🔎",
    layout="wide",
)


# ---------------- Cached lookups ----------------
@st.cache_data(show_spinner=False)
def _summary(db_path_str: str) -> dict:
    return db.db_summary(Path(db_path_str))


@st.cache_data(show_spinner=False)
def _ifc_models(db_path_str: str) -> list[dict]:
    return db.list_ifc_models(Path(db_path_str))


@st.cache_data(show_spinner=False)
def _label_counts(db_path_str: str) -> dict[str, dict[str, int]]:
    return db.count_labels_per_ifc(Path(db_path_str))


@st.cache_data(show_spinner=False)
def _runs(db_path_str: str) -> list[dict]:
    return db.list_runs(Path(db_path_str))


@st.cache_data(show_spinner=False)
def _children(db_path_str: str, parent_id: str) -> list[dict]:
    return db.list_children(Path(db_path_str), parent_id)


@st.cache_data(show_spinner=False)
def _violations(db_path_str: str, run_id: str) -> list[dict]:
    return db.get_violations(Path(db_path_str), run_id)


def _ifc_dataframe(paths: DatasetPaths) -> pd.DataFrame:
    models = _ifc_models(str(paths.db))
    counts = _label_counts(str(paths.db))
    runs = {r["id"]: r for r in _runs(str(paths.db))}
    rows = []
    for m in models:
        c = counts.get(m["id"], {})
        pool = runs.get(m.get("pool_run_id"))
        rows.append({
            "id": m["id"],
            "kind": m["kind"],
            "name": m["name"],
            "status": m["status"],
            "labels": c.get("total", 0),
            "applied": c.get("applied", 0),
            "decoy": c.get("decoy", 0),
            "skipped": c.get("skipped", 0),
            "parent_id": m.get("parent_id") or "",
            "pool_run": pool["name"] if pool else "",
            "pool_run_id": m.get("pool_run_id") or "",
            "llm_model": m["llm_model"],
            "error": m.get("error") or "",
            "created_at": m["created_at"],
        })
    return pd.DataFrame(rows)


def _label_for(row) -> str:
    return f"[{row['kind']}] {row['name']}  ·  {row['id'][:8]}  ·  {row['labels']} et."


# ---------------- Sidebar ----------------
def _resolve_root() -> str:
    """Sidebar'daki kaynak seçimine göre kök yolu döndür."""
    mode = st.session_state.get("ds_mode", "codex1")
    if mode == "codex1":
        c = first_valid_codex1()
        if c:
            return str(c)
        return default_root()
    return st.session_state.get("ds_custom_path", default_root())


def _sidebar(df_full: pd.DataFrame | None) -> DatasetPaths | None:
    st.sidebar.markdown("### 📂 Veri Kaynağı")
    prev_mode = st.session_state.get("ds_mode", "codex1")
    mode = st.sidebar.radio(
        "Kaynak",
        ["codex1", "custom"],
        index=0 if prev_mode == "codex1" else 1,
        format_func=lambda x: {
            "codex1": "📦 codex1 (komşu repo)",
            "custom": "📂 Özel yol",
        }[x],
        key="ds_mode",
        horizontal=True,
    )

    if mode == "codex1":
        c = first_valid_codex1()
        if c:
            st.sidebar.caption(f"`{c}`")
            root = str(c)
        else:
            st.sidebar.warning("Komşu codex1 bulunamadı. 'Özel yol'a geç.")
            root = default_root()
    else:
        prev = st.session_state.get("ds_custom_path",
                                    st.session_state.get("dataset_root",
                                                         default_root()))
        root = st.sidebar.text_input(
            "Kök yol", value=prev,
            help="İçinde violation_pool.sqlite ve ifc_models/ olan klasör",
            key="ds_custom_path",
        )

    prev_root = st.session_state.get("dataset_root")
    if root != prev_root or mode != prev_mode:
        st.session_state["dataset_root"] = root
        st.cache_data.clear()
        st.session_state.pop("selected_ifc_id", None)

    paths = DatasetPaths.from_root(root)
    if not paths.is_valid:
        st.sidebar.error(f"❌ Bulunamadı: {paths.db}")
        return None

    st.sidebar.success(f"✅ {paths.root.name}")
    s = _summary(str(paths.db))
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Run", s["runs"]); c2.metric("Kural", s["violations"])
    c1.metric("Baseline", s["baseline"]); c2.metric("Violated", s["violated"])
    c1.metric("Imported", s["imported"]); c2.metric("Etiket", s["labels"])
    c1.metric("Decoy", s["decoys"]); c2.metric("Token", f"{s['tokens']:,}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dosyalar")
    if df_full is None or df_full.empty:
        st.sidebar.info("Henüz IFC yok.")
        return paths

    kinds = ["hepsi"] + sorted(df_full["kind"].unique().tolist())
    kind = st.sidebar.selectbox("Tür", kinds, key="sb_kind")
    q = st.sidebar.text_input("Ara", "", key="sb_q",
                              placeholder="name / id parçası")

    fdf = df_full
    if kind != "hepsi":
        fdf = fdf[fdf["kind"] == kind]
    if q:
        ql = q.lower()
        fdf = fdf[fdf["name"].str.lower().str.contains(ql, na=False)
                  | fdf["id"].str.lower().str.contains(ql, na=False)]

    st.sidebar.caption(f"{len(fdf)} dosya")
    st.session_state["sidebar_ids"] = fdf["id"].tolist()

    if fdf.empty:
        return paths

    options = fdf["id"].tolist()
    current_id = st.session_state.get("selected_ifc_id")
    default_idx = options.index(current_id) if current_id in options else 0
    labels = {r["id"]: _label_for(r) for _, r in fdf.iterrows()}
    sel = st.sidebar.radio(
        "Aç",
        options,
        index=default_idx,
        format_func=lambda x: labels.get(x, x[:8]),
        label_visibility="collapsed",
        key="sb_radio",
    )
    if sel != current_id:
        st.session_state["selected_ifc_id"] = sel

    return paths


# ---------------- Pool / Violations ----------------
_METHOD_LABELS = {
    "naive": "1) Naive",
    "optimized": "2) Optimized",
    "rag": "3) RAG + optimized",
    "finetune": "4) Fine-tune + optimized",
}


def page_pool(paths: DatasetPaths) -> None:
    runs = _runs(str(paths.db))
    if not runs:
        st.info("Henüz hiçbir ihlal havuzu (run) yok.")
        return

    st.subheader("📜 İhlal Havuzu")
    run_options = {
        f"{r['name']}  ·  {_METHOD_LABELS.get(r['method'], r['method'])}  "
        f"·  {r['status']}  ·  {r['llm_model']}": r["id"]
        for r in runs
    }
    sel_label = st.selectbox("Run", list(run_options.keys()), key="pool_run")
    run_id = run_options[sel_label]
    run = db.get_run(paths.db, run_id)
    if not run:
        st.error("Run okunamadı.")
        return

    viols = _violations(str(paths.db), run_id)

    # Meta header
    mc = st.columns(5)
    mc[0].metric("Toplam ihlal", len(viols))
    batches = sorted({v.get("batch_no") for v in viols if v.get("batch_no")})
    mc[1].metric("Batch", len(batches))
    mc[2].metric("Status", run["status"])
    usage = db.usage_totals(paths.db, pool_run_id=run_id)
    mc[3].metric("LLM çağrı", usage["calls"])
    mc[4].metric("Token", f"{usage['total_tokens']:,}")

    with st.expander("Run meta"):
        meta = {
            "id": run["id"],
            "name": run["name"],
            "method": _METHOD_LABELS.get(run["method"], run["method"]),
            "llm_model": run["llm_model"],
            "embedding_model": run.get("embedding_model"),
            "rag_collection": run.get("rag_collection"),
            "rag_documents": run.get("rag_documents"),
            "finetune_model_id": run.get("finetune_model_id"),
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
        }
        st.json({k: v for k, v in meta.items() if v not in (None, "", [])})
        if run.get("prompt"):
            st.markdown("**Prompt:**")
            st.code(run["prompt"])

    # Excel download
    xlsx = find_run_export(paths, run_id)
    if xlsx and xlsx.exists():
        try:
            data = xlsx.read_bytes()
            st.download_button(
                f"⬇️ Excel indir ({xlsx.name}, {len(data) / 1024:.1f} KB)",
                data=data, file_name=xlsx.name, key=f"dl_xlsx_{run_id}",
                type="primary",
            )
        except Exception as e:
            st.warning(f"Excel okunamadı: {e}")

    if not viols:
        st.info("Bu run'da ihlal yok.")
        return

    # Build df
    rows = []
    for v in viols:
        rows.append({
            "title": v.get("title"),
            "category": v.get("category"),
            "severity": v.get("severity"),
            "threshold": v.get("threshold"),
            "batch": v.get("batch_no"),
            "evidence": len(v.get("evidence", [])),
            "description": v.get("description"),
            "id": v["id"],
        })
    vdf = pd.DataFrame(rows)

    # Filters
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    with f1:
        cats = ["(hepsi)"] + sorted(
            {c for c in vdf["category"].tolist() if c}
        )
        cat = st.selectbox("Kategori", cats, key="pool_cat")
    with f2:
        sevs = ["(hepsi)"] + sorted(
            {s for s in vdf["severity"].tolist() if s}
        )
        sev = st.selectbox("Şiddet", sevs, key="pool_sev")
    with f3:
        bsel = ["(hepsi)"] + [str(b) for b in batches]
        b = st.selectbox("Batch", bsel, key="pool_batch")
    with f4:
        q = st.text_input("Ara (title / description)", "", key="pool_q")

    f = vdf.copy()
    if cat != "(hepsi)":
        f = f[f["category"] == cat]
    if sev != "(hepsi)":
        f = f[f["severity"] == sev]
    if b != "(hepsi)":
        f = f[f["batch"] == int(b)]
    if q:
        ql = q.lower()
        f = f[f["title"].fillna("").str.lower().str.contains(ql)
              | f["description"].fillna("").str.lower().str.contains(ql)]

    st.caption(f"{len(f)} / {len(vdf)} kayıt")
    st.dataframe(
        f.drop(columns=["id"]),
        use_container_width=True, hide_index=True, height=420,
        column_config={
            "title": st.column_config.TextColumn("title", width="medium"),
            "description": st.column_config.TextColumn("description", width="large"),
        },
    )

    # Charts
    if not f.empty:
        with st.expander("📊 Dağılımlar"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Kategori**")
                cc = f["category"].fillna("(boş)").value_counts()
                st.bar_chart(cc)
            with c2:
                st.markdown("**Şiddet**")
                sc = f["severity"].fillna("(boş)").value_counts()
                st.bar_chart(sc)
            with c3:
                st.markdown("**Batch**")
                bc = f["batch"].fillna(0).value_counts().sort_index()
                st.bar_chart(bc)

    # Drilldown
    st.markdown("### 🔗 Kanıt zinciri")
    if f.empty:
        st.caption("Filtre kayıtları boşalttı.")
        return
    pick_options = {
        f"{r['title'] or '(başlıksız)'}  ·  {r['category'] or '-'}  "
        f"·  batch {r['batch']}  ·  {r['id'][:8]}": r["id"]
        for _, r in f.iterrows()
    }
    sel_v_label = st.selectbox(
        "İhlal seç", list(pick_options.keys()), key=f"pool_pick_{run_id}",
    )
    vid = pick_options[sel_v_label]
    v = next((x for x in viols if x["id"] == vid), None)
    if v:
        st.json({
            "id": v["id"],
            "title": v.get("title"),
            "category": v.get("category"),
            "severity": v.get("severity"),
            "threshold": v.get("threshold"),
            "batch_no": v.get("batch_no"),
            "description": v.get("description"),
        })
        ev = v.get("evidence", [])
        if ev:
            st.markdown(f"**Kanıtlar ({len(ev)}):**")
            st.dataframe(pd.DataFrame(ev),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Kanıt yok.")


# ---------------- Browser ----------------
def page_browser(paths: DatasetPaths, df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Henüz hiç IFC üretilmemiş.")
        return

    st.subheader("🗂️ IFC Modelleri")
    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    with f1:
        kinds = ["(hepsi)"] + sorted(df["kind"].unique().tolist())
        kind_sel = st.selectbox("Tür", kinds, index=0, key="br_kind")
    with f2:
        statuses = ["(hepsi)"] + sorted(df["status"].unique().tolist())
        status_sel = st.selectbox("Status", statuses, index=0, key="br_status")
    with f3:
        runs_list = ["(hepsi)"] + sorted({p for p in df["pool_run"].tolist() if p})
        pool_sel = st.selectbox("Havuz", runs_list, index=0, key="br_pool")
    with f4:
        q = st.text_input("Ara (name / id)", "", key="br_q")

    fdf = df.copy()
    if kind_sel != "(hepsi)":
        fdf = fdf[fdf["kind"] == kind_sel]
    if status_sel != "(hepsi)":
        fdf = fdf[fdf["status"] == status_sel]
    if pool_sel != "(hepsi)":
        fdf = fdf[fdf["pool_run"] == pool_sel]
    if q:
        ql = q.lower()
        fdf = fdf[fdf["name"].str.lower().str.contains(ql, na=False)
                 | fdf["id"].str.lower().str.contains(ql, na=False)]

    max_lab = int(max(fdf["labels"].max() if not fdf.empty else 0, 1))
    min_labels = st.slider("En az etiket sayısı", 0, max_lab, 0, key="br_minlab")
    if min_labels > 0:
        fdf = fdf[fdf["labels"] >= min_labels]

    st.caption(f"{len(fdf)} kayıt")
    st.dataframe(fdf, use_container_width=True, hide_index=True, height=420)

    if fdf.empty:
        return
    st.markdown("### 🔍 Detay aç")
    options = {_label_for(r): r["id"] for _, r in fdf.iterrows()}
    sel_label = st.selectbox("Bir IFC seç", list(options.keys()), key="br_pick")
    if st.button("Detayı aç →", type="primary"):
        st.session_state["selected_ifc_id"] = options[sel_label]
        st.rerun()


# ---------------- Render helpers ----------------
def _highlight_sets(labels: list[dict]) -> tuple[set, set]:
    hl = {l["ifc_global_id"] for l in labels
          if l.get("ifc_global_id") and not l.get("is_decoy")}
    dc = {l["ifc_global_id"] for l in labels
          if l.get("ifc_global_id") and l.get("is_decoy")}
    return hl, dc


def _render_3d(paths: DatasetPaths, model: dict, hl: set, dc: set,
               key_suffix: str = "") -> None:
    ifc_path = resolve_artifact(paths, model.get("file_path"))
    if not ifc_path:
        st.error(f"IFC dosyası bulunamadı: {model.get('file_path')}")
        return
    with st.spinner("Geometri tessellate ediliyor..."):
        try:
            from viewer.ifc3d import ifc_to_figure
            fig, info = ifc_to_figure(ifc_path, hl, dc)
            st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom": True},
                            key=f"plot3d_{model['id']}_{key_suffix}")
            st.caption(f"Drawn: {info['drawn']}  ·  Skipped: {info['skipped']}  "
                       f"·  İhlal: {len(hl)}  ·  Decoy: {len(dc)}")
        except Exception as e:
            st.error(f"3D çizim başarısız: {e}")


def _render_graph(paths: DatasetPaths, model: dict, hl: set, dc: set,
                  key_suffix: str = "") -> None:
    graph_path = resolve_artifact(paths, model.get("graph_path"))
    if not graph_path:
        st.warning("Graph JSON yok.")
        return
    try:
        from viewer.graph_view import load_graph, graph_to_figure
        g = load_graph(graph_path)
        fig = graph_to_figure(g, hl, dc)
        st.plotly_chart(fig, use_container_width=True,
                        config={"scrollZoom": True},
                        key=f"plotg_{model['id']}_{key_suffix}")
        st.caption(f"Nodes: {g.number_of_nodes()}  ·  "
                   f"Edges: {g.number_of_edges()}")
    except Exception as e:
        st.error(f"Graph çizim başarısız: {e}")


def _kind_emoji(kind: str) -> str:
    return {"baseline": "🏛️", "violated": "🚫", "imported": "📥"}.get(kind, "📄")


# ---------------- Detail ----------------
def page_detail(paths: DatasetPaths) -> None:
    ifc_id = st.session_state.get("selected_ifc_id")
    if not ifc_id:
        st.info("Sol panelden bir IFC seç veya **Browser**'dan birini aç.")
        return

    model = db.get_ifc_model(paths.db, ifc_id)
    if not model:
        st.error(f"IFC bulunamadı: {ifc_id}")
        return

    sidebar_ids: list[str] = st.session_state.get("sidebar_ids", [])
    if ifc_id in sidebar_ids:
        idx = sidebar_ids.index(ifc_id)
        total = len(sidebar_ids)
    else:
        idx, total = -1, 0

    # Nav row
    nav_l, nav_mid, nav_r = st.columns([1, 4, 1])
    with nav_l:
        if total and idx > 0:
            if st.button("◀ Önceki", use_container_width=True, key="nav_prev"):
                st.session_state["selected_ifc_id"] = sidebar_ids[idx - 1]
                st.rerun()
        else:
            st.button("◀ Önceki", use_container_width=True,
                      key="nav_prev", disabled=True)
    with nav_mid:
        pos = f"{idx + 1} / {total}" if total else "—"
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;'>"
            f"{_kind_emoji(model['kind'])} <b>{model['kind'].upper()}</b> · "
            f"<code>{model['name']}</code> · "
            f"<span style='opacity:.65'>{pos}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if total and idx >= 0 and idx < total - 1:
            if st.button("Sonraki ▶", use_container_width=True, key="nav_next"):
                st.session_state["selected_ifc_id"] = sidebar_ids[idx + 1]
                st.rerun()
        else:
            st.button("Sonraki ▶", use_container_width=True,
                      key="nav_next", disabled=True)

    st.caption(f"id: `{model['id']}`  ·  status: `{model['status']}`  ·  "
               f"oluşturma: {model['created_at']}")

    # Baseline parent & violated children
    parent_model = None
    if model.get("parent_id"):
        parent_model = db.get_ifc_model(paths.db, model["parent_id"])
    children = _children(str(paths.db), model["id"]) if model["kind"] == "baseline" else []

    # Compare option
    compare = False
    if parent_model:
        compare = st.toggle(
            "📐 Baseline ile yan yana karşılaştır",
            value=st.session_state.get("compare_baseline", False),
            key="compare_baseline",
        )

    labels = db.list_ifc_labels(paths.db, ifc_id)
    hl, dc = _highlight_sets(labels)

    info_cols = st.columns(5)
    info_cols[0].metric("Etiket", len(labels))
    info_cols[1].metric("Applied",
                        sum(1 for l in labels
                            if l["status"] == "applied" and not l["is_decoy"]))
    info_cols[2].metric("Decoy", sum(1 for l in labels if l["is_decoy"]))
    info_cols[3].metric("Skipped",
                        sum(1 for l in labels if l["status"] == "skipped"))
    usage = db.usage_totals(paths.db, ifc_model_id=ifc_id)
    info_cols[4].metric("Token", f"{usage['total_tokens']:,}")

    # Derived violated list for baselines
    if model["kind"] == "baseline" and children:
        with st.expander(f"🧬 Bu baseline'dan türetilmiş {len(children)} violated"):
            counts_map = _label_counts(str(paths.db))
            child_rows = []
            for ch in children:
                cnt = counts_map.get(ch["id"], {})
                child_rows.append({
                    "id": ch["id"],
                    "name": ch["name"],
                    "labels": cnt.get("total", 0),
                    "applied": cnt.get("applied", 0),
                    "decoy": cnt.get("decoy", 0),
                    "status": ch["status"],
                    "created": ch["created_at"],
                })
            cdf = pd.DataFrame(child_rows)
            st.dataframe(cdf, use_container_width=True, hide_index=True, height=200)
            picker = st.selectbox(
                "Açılacak violated",
                [ch["id"] for ch in children],
                format_func=lambda x: next(
                    (f"{c['name']} · {c['id'][:8]}"
                     for c in children if c["id"] == x), x),
                key="children_pick",
            )
            if st.button("→ Aç", key="children_open"):
                st.session_state["selected_ifc_id"] = picker
                st.rerun()

    # Tabs
    t_3d, t_graph, t_labels, t_meta, t_files = st.tabs(
        ["🧱 3D", "🕸️ Graph", "🏷️ Etiketler", "📄 Meta", "📥 Dosyalar"]
    )

    with t_3d:
        if compare and parent_model:
            col_b, col_v = st.columns(2)
            with col_b:
                st.markdown(f"**🏛️ BASELINE** · `{parent_model['name']}`")
                _render_3d(paths, parent_model, set(), set(), "base")
            with col_v:
                st.markdown(f"**🚫 VIOLATED** · `{model['name']}`")
                _render_3d(paths, model, hl, dc, "viol")
        else:
            _render_3d(paths, model, hl, dc)

    with t_graph:
        if compare and parent_model:
            col_b, col_v = st.columns(2)
            with col_b:
                st.markdown(f"**🏛️ BASELINE** · `{parent_model['name']}`")
                _render_graph(paths, parent_model, set(), set(), "base")
            with col_v:
                st.markdown(f"**🚫 VIOLATED** · `{model['name']}`")
                _render_graph(paths, model, hl, dc, "viol")
        else:
            _render_graph(paths, model, hl, dc)

    with t_labels:
        if not labels:
            st.info("Bu IFC'de etiket yok.")
        else:
            ldf = pd.DataFrame([
                {
                    "title": l["title"], "category": l["category"],
                    "severity": l["severity"], "status": l["status"],
                    "decoy": "✓" if l["is_decoy"] else "",
                    "action": l["action"], "ifc_type": l["ifc_type"],
                    "attribute": l["attribute"],
                    "before": l["value_before"], "after": l["value_after"],
                    "guid": l["ifc_global_id"], "reason": l["reason"],
                }
                for l in labels
            ])
            st.dataframe(ldf, use_container_width=True,
                         hide_index=True, height=380)

            with st.expander("🔗 Kanıt zinciri — etiket detayı"):
                lbl_idx = st.number_input(
                    "Satır (0-bazlı)", 0, max(len(labels) - 1, 0), 0,
                    key=f"lbl_idx_{ifc_id}",
                )
                lab = labels[int(lbl_idx)]
                st.json({
                    "title": lab["title"], "category": lab["category"],
                    "severity": lab["severity"], "threshold": lab["threshold"],
                    "status": lab["status"], "is_decoy": bool(lab["is_decoy"]),
                    "action": lab["action"], "ifc_type": lab["ifc_type"],
                    "ifc_global_id": lab["ifc_global_id"],
                    "attribute": lab["attribute"],
                    "value_before": lab["value_before"],
                    "value_after": lab["value_after"],
                    "reason": lab["reason"],
                })
                try:
                    ev = json.loads(lab.get("evidence_json") or "[]")
                    if ev:
                        st.markdown("**Kanıtlar:**")
                        st.dataframe(pd.DataFrame(ev),
                                     use_container_width=True, hide_index=True)
                except Exception:
                    pass
                vid = lab.get("violation_id")
                if vid:
                    pool = db.get_violation(paths.db, vid)
                    if pool:
                        st.markdown("**Havuz kaydı:**")
                        st.json({k: v for k, v in pool.items()
                                 if k not in ("evidence",)})

    with t_meta:
        meta_path = resolve_artifact(paths, model.get("meta_path"))
        if meta_path:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st.json(meta)
            except Exception as e:
                st.error(f"meta.json okunamadı: {e}")
        else:
            st.info("meta.json yok.")
        try:
            params = json.loads(model.get("params_json") or "{}")
            if params:
                st.markdown("**params_json (DB)**")
                st.json(params)
        except Exception:
            pass
        if model.get("prompt"):
            with st.expander("Prompt"):
                st.code(model["prompt"])

    with t_files:
        for key, label in [("file_path", "IFC"),
                           ("meta_path", "meta.json"),
                           ("labels_path", "labels.json"),
                           ("graph_path", "graph.json")]:
            p = resolve_artifact(paths, model.get(key))
            if p:
                try:
                    data = p.read_bytes()
                    st.download_button(
                        f"⬇️ {label} ({p.name}, {len(data) / 1024:.1f} KB)",
                        data=data, file_name=p.name,
                        key=f"dl_{key}_{ifc_id}",
                    )
                except Exception as e:
                    st.warning(f"{label}: {e}")
            else:
                st.caption(f"— {label}: yok")


# ---------------- Main ----------------
def main() -> None:
    pre_root = st.session_state.get("dataset_root", default_root())
    pre_paths = DatasetPaths.from_root(pre_root)
    df_full = _ifc_dataframe(pre_paths) if pre_paths.is_valid else None

    paths = _sidebar(df_full)

    st.title("🔎 Erişilebilirlik Dataset Viewer")
    st.caption("TS 9111 / TS ISO 21542 ihlal havuzu ve ihlalli IFC dataset'i "
               "— read-only inceleme.")

    if not paths:
        st.error("Geçerli bir dataset köküne ihtiyaç var. Sol panelden codex1 "
                 "kök yolunu (içinde `violation_pool.sqlite` olan klasör) ver.")
        return

    df = df_full if df_full is not None else _ifc_dataframe(paths)
    tab_detail, tab_browser, tab_pool = st.tabs(
        ["🔍 IFC Detay", "🗂️ Browser", "📜 İhlal Havuzu"]
    )
    with tab_detail:
        page_detail(paths)
    with tab_browser:
        page_browser(paths, df)
    with tab_pool:
        page_pool(paths)


if __name__ == "__main__":
    main()
