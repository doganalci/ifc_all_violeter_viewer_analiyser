"""LLM-tabanlı kapı ihlali enjeksiyonu için batch runner.

run_basic_batch'ın paralel bir muadili; LLM-bazlı inject_doors_llm'i çağırır,
DB / graph / tam etiketleme / defter güncelleme dahil aynı plumbing'i sağlar.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from violation_pool import storage, ifc_graph
from violation_pool.config import settings

from llm.door_inject import LLMDoorParams, inject_doors_llm


def _labels_to_doc(ifc_id: str, baseline_id: str, labels: list[dict],
                   model: str) -> dict:
    """Eğitim pipeline'ının okuyacağı labels.json formatı."""
    out_labels = []
    for l in labels:
        out_labels.append({
            "ifc_global_id": l["ifc_global_id"],
            "category": l["category"],
            "severity": l["severity"],
            "status": "applied" if l["is_violation"] else "compliant",
            "is_decoy": False,
            "attribute": l["attribute"],
            "before": l.get("before"),
            "after": l.get("after"),
            "rule": l.get("rule"),
            "evidence": l.get("evidence"),
            "llm_rationale": l.get("llm_rationale", ""),
            "llm_intended": l.get("llm_intended", ""),
        })
    return {
        "violated_id": ifc_id,
        "baseline_id": baseline_id,
        "source": "llm_door_inject",
        "model": model,
        "labels": out_labels,
    }


def run_llm_door_batch(dataset_tag: str, *,
                       variants: int = 5, seed_start: int = 0,
                       params: LLMDoorParams | None = None,
                       model: str = "gpt-4o",
                       full_label: bool = True,
                       method_label: str = "llmgen",
                       register_in_db: bool = True,
                       progress_cb=None) -> dict:
    """dataset_tag'li tüm baseline'lara LLM-tabanlı kapı ihlali enjekte et."""
    p = params or LLMDoorParams()
    baseline_ids = storage.ifc_ids_for_tags([dataset_tag], kind="baseline")
    baselines = [storage.get_ifc_model(i) for i in baseline_ids]
    baselines = [b for b in baselines if b and b.get("status") == "ok"]
    if not baselines:
        raise RuntimeError(f"'{dataset_tag}' etiketli baseline yok.")

    out_dir = settings.ifc_dir / "violated"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"ok": 0, "err": 0, "violations": 0,
                     "hard_negatives": 0, "items": []}
    total = len(baselines) * variants
    done = 0

    _mlabel = "".join(c if c.isalnum() or c in "-_" else "_"
                      for c in str(method_label).strip()) or "llm"

    for b in baselines:
        base_stem = Path(b["file_path"]).stem
        name_base = f"{_mlabel}_{base_stem}"
        for vi in range(variants):
            seed = seed_start + done
            out_id = str(uuid.uuid4())
            n = 1
            while (out_dir / f"{name_base}_violated{n}.ifc").exists():
                n += 1
            stem = f"{name_base}_violated{n}"
            if (out_dir / f"{stem}.ifc").exists():
                stem = f"{name_base}_violated{n}_{out_id[:6]}"
            out_ifc = out_dir / f"{stem}.ifc"
            try:
                r = inject_doors_llm(b["file_path"], out_ifc,
                                     seed=seed, params=p, model=model)
                doc = _labels_to_doc(out_id, b["id"], r["labels"], model)
                lab_path = out_dir / f"{stem}.labels.json"
                lab_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                meta_path = out_dir / f"{stem}.meta.json"
                meta_path.write_text(json.dumps({
                    "ifc_id": out_id, "kind": "violated",
                    "baseline_id": b["id"], "source": "llm_door_inject",
                    "model": model, "seed": seed, "summary": r["summary"],
                }, indent=2, ensure_ascii=False), encoding="utf-8")
                # graph
                graph_path = None
                try:
                    gp = out_dir / f"{stem}.graph.json"
                    ifc_graph.build_and_save(str(out_ifc), str(gp))
                    graph_path = str(gp)
                except Exception as _ge:
                    results.setdefault("graph_errors", []).append(
                        f"{stem}: {_ge}")

                # Tam etiketleme: graph'taki HER node'a kesin etiket.
                if full_label and graph_path:
                    try:
                        from ml.data.graph_loader import load_graph as _lg
                        g = _lg(graph_path)
                        labeled = {l["ifc_global_id"] for l in doc["labels"]}
                        n_clean = 0
                        for nid in g.nodes():
                            if nid in labeled:
                                continue
                            doc["labels"].append({
                                "ifc_global_id": nid, "category": "",
                                "severity": "uygun", "status": "clean",
                                "is_decoy": False, "attribute": None,
                                "before": None, "after": None,
                                "evidence": "Baseline temiz — kesin ihlal değil",
                            })
                            n_clean += 1
                        lab_path.write_text(json.dumps(doc, indent=2,
                                            ensure_ascii=False), encoding="utf-8")
                        results["clean_labeled"] = (
                            results.get("clean_labeled", 0) + n_clean)
                    except Exception as _le:
                        results.setdefault("label_errors", []).append(
                            f"{stem}: {_le}")

                # DB
                if register_in_db:
                    mid = storage.create_ifc_model(
                        id=out_id, kind="violated",
                        name=f"{base_stem}.violated", parent_id=b["id"],
                        llm_model=model, prompt=None, pool_run_id=None,
                        params={"summary": r["summary"], "seed": seed,
                                "model": model},
                        file_path=str(out_ifc), meta_path=str(meta_path),
                        labels_path=str(lab_path), graph_path=graph_path,
                        status="ok", error=None,
                        dataset_tag=b.get("dataset_tag") or dataset_tag,
                    )
                    storage.add_ifc_labels(mid, [{**lab}
                                                 for lab in doc["labels"]])
                results["ok"] += 1
                if graph_path:
                    results["with_graph"] = results.get("with_graph", 0) + 1
                results["violations"] += r["summary"]["n_violations"]
                results["hard_negatives"] += r["summary"]["n_compliant_changes"]
                results["items"].append({"stem": stem, **r["summary"]})
            except Exception as e:
                results["err"] += 1
                results["items"].append({"stem": stem, "error": str(e)})
            done += 1
            if progress_cb:
                progress_cb(done, total, stem)

    # Defterler
    if register_in_db:
        try:
            from ml.tracking import rebuild_dataset_registry, log_operation
            from paths import data_home
            rebuild_dataset_registry(str(data_home()))
            log_operation(
                f"llm_kapi_ihlal ({method_label}, {model})",
                paket=dataset_tag, adet=results.get("ok", 0),
                ozet=f"ihlal={results.get('violations',0)} "
                     f"hard_neg={results.get('hard_negatives',0)} "
                     f"clean={results.get('clean_labeled',0)}",
                parametreler=f"variants={variants} seed_start={seed_start} "
                             f"target_vio={p.target_n_violations} "
                             f"target_comp={p.target_n_compliant}")
        except Exception:
            pass

    return results
