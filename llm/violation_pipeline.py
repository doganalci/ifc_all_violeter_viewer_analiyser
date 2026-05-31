"""RAG-tabanlı ihlal üretim pipeline'ı.

Akış:
  1. Kullanıcı baseline paketini seçer.
  2. ChromaDB collection'dan kategori sorgusuyla k=8 kural chunk çekilir.
  3. generate_rag(OPTIMIZED_PROMPT, chunks) → JSON sözel ihlal havuzu.
  4. Kategori filtresi uygulanır.
  5. Her baseline için havuzdan N ihlal seçilir.
  6. inject_violations() → IFC'ye LLM ile uygulanır, etiket kuralla ölçülür.
  7. DB hiyerarşik kayıt (violated, parent_id=baseline.id) + Excel log.

Mevcut altyapı %80 hazır olduğu için zincir kuran ince bir katman:
  * violation_pool/rag.py:retrieve
  * violation_pool/llm.py:generate_rag (OPTIMIZED_PROMPT + chunks)
  * violation_pool/ifc_inject.py:inject_violations (sözel → IFC eylem)
"""
from __future__ import annotations

import random
import time
from pathlib import Path


def run_rag_violation_pipeline(
    dataset_tag: str, *,
    collection_name: str = "default",
    categories: list[str] | None = None,
    n_violations_per_ifc: int = 3,
    violated_per_baseline: int = 1,
    decoy_ratio: float = 0.20,
    compliant_addition_ratio: float = 0.0,
    rag_k: int = 8,
    pool_oversample: float = 2.5,
    model: str = "gpt-4o",
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    progress_cb=None,
    register_in_db: bool = True,
) -> dict:
    """RAG'dan sözel ihlal çek → her baseline'a inject_violations.

    Args:
        dataset_tag: hedef baseline paketi (paketin tüm baseline'larına ihlal
            enjekte edilir).
        collection_name: ChromaDB collection adı (RAG corpus).
        categories: izin verilen kategori listesi (None = 16 kategorinin
            tümü). OPTIMIZED_PROMPT'taki TS 9111 / TS ISO 21542 kategorileri.
        n_violations_per_ifc: her baseline için kaç gerçek ihlal enjekte
            edilecek.
        decoy_ratio: ihlal sayısının yüzdesi kadar SAHTE (decoy) etiket
            (IFC modifiye edilmez).
        rag_k: RAG retrieve top-k chunk sayısı.
        pool_oversample: ihlal havuzu oversample faktörü (kategori filtreden
            sonra yetecek kadar bırakmak için).
        model: LLM model adı (hem generate_rag hem inject_violations için).
        progress_cb: (done, total, label) → None.
        register_in_db: DB'ye violated kaydı yapsın mı.

    Returns:
        {
            "ok": int,                    # başarılı violated IFC sayısı
            "err": int,                   # hata sayısı
            "violations_applied": int,    # toplam uygulanan ihlal
            "decoys": int,                # toplam decoy
            "pool_size": int,             # havuz büyüklüğü
            "rag_chunks": int,            # RAG'dan çekilen chunk sayısı
            "duration_s": float,          # toplam süre
            "items": [...]                # her IFC için detay
        }
    """
    from violation_pool import ifc_inject, rag, storage
    from violation_pool._chat import chat_endpoint
    from violation_pool.llm import generate_rag
    from violation_pool.prompts import OPTIMIZED_PROMPT

    # Local LLM override: tüm chat çağrıları (RAG havuz, inject, compliant)
    # bu endpoint'e yönlendirilir. Embedding override edilmez.
    with chat_endpoint(llm_base_url, llm_api_key):
        return _run_inner(
            dataset_tag=dataset_tag,
            collection_name=collection_name,
            categories=categories,
            n_violations_per_ifc=n_violations_per_ifc,
            violated_per_baseline=violated_per_baseline,
            decoy_ratio=decoy_ratio,
            compliant_addition_ratio=compliant_addition_ratio,
            rag_k=rag_k,
            pool_oversample=pool_oversample,
            model=model,
            progress_cb=progress_cb,
            register_in_db=register_in_db,
            llm_base_url=llm_base_url,
        )


def _run_inner(
    dataset_tag: str, *,
    collection_name: str,
    categories,
    n_violations_per_ifc: int,
    violated_per_baseline: int,
    decoy_ratio: float,
    compliant_addition_ratio: float,
    rag_k: int,
    pool_oversample: float,
    model: str,
    progress_cb,
    register_in_db: bool,
    llm_base_url: str | None,
) -> dict:
    from violation_pool import ifc_inject, rag, storage
    from violation_pool.llm import generate_rag

    t0 = time.time()
    results: dict = {
        "ok": 0, "err": 0,
        "violations_applied": 0, "decoys": 0, "compliant": 0,
        "pool_size": 0, "rag_chunks": 0,
        "items": [],
    }

    # 1) Hedef baseline'ları topla
    baseline_ids = storage.ifc_ids_for_tags([dataset_tag], kind="baseline")
    if not baseline_ids:
        raise RuntimeError(
            f"'{dataset_tag}' paketinde baseline yok. Önce sayfa 10/14 ile üret."
        )

    # 2) RAG sorgusu — kategori isimlerini ve genel arama anahtarlarını birleştir
    if categories:
        cat_query = " · ".join(categories)
    else:
        cat_query = ("Kapı Koridor Rampa Merdiven Korkuluk Asansör "
                     "Tuvalet Manevra Eşik")
    rag_query = (
        f"TS 9111 TS ISO 21542 erişilebilirlik kullanılabilirlik ihlal "
        f"kuralları — {cat_query}"
    )

    try:
        chunks = rag.retrieve(collection_name, rag_query, k=int(rag_k))
    except Exception as e:
        raise RuntimeError(
            f"RAG retrieve hatası ({collection_name}): {e}\n"
            "ChromaDB collection yok ya da boş olabilir. "
            "Legacy sayfa 99 ile doküman ingest edin."
        )
    results["rag_chunks"] = len(chunks)
    if not chunks:
        raise RuntimeError(
            f"'{collection_name}' collection boş — sorgu sonucu 0 chunk. "
            "Legacy sayfa 99 (Codex LLM Havuzu) ile doküman ingest edin."
        )

    # 3) LLM ile sözel ihlal havuzu üret (tek seferlik)
    # Hedef sayı: oversample × her baseline × n_per_ifc
    target_n = max(
        20,
        int(n_violations_per_ifc * len(baseline_ids) * float(pool_oversample)),
    )
    try:
        user_prompt = (
            "Aşağıdaki bağlam dokümanlarına dayanarak somut, ölçülebilir "
            "erişilebilirlik ihlal kuralları üret."
        )
        pool = generate_rag(
            user_prompt, chunks, model=model, n=target_n,
            usage_meta={"phase": "rag_violation_pool", "paket": dataset_tag},
        )
    except Exception as e:
        raise RuntimeError(f"generate_rag hatası: {e}")

    # 4) Kategori filtresi
    if categories:
        cat_set = set(categories)
        pool = [v for v in pool if v.get("category") in cat_set]
    results["pool_size"] = len(pool)
    if not pool:
        raise RuntimeError(
            f"İhlal havuzu boş (kategori filtre sonrası). "
            f"LLM döndürdü: {target_n} sözel ihlal, kategori uyan: 0."
        )

    # Havuza ID ata (inject_violations için lazım)
    for i, v in enumerate(pool):
        if "id" not in v or not v["id"]:
            v["id"] = f"rag_{i}_{v.get('category', 'misc')[:8]}"

    # 5) Her baseline için violated_per_baseline kez ihlal seç + enjekte.
    # Aynı baseline'a birden fazla violated IFC üretmek dataset
    # çeşitliliğini artırır; her seferinde havuzdan farklı örnek seçilir
    # ve dosya adı _violated1, _violated2 olarak otomatik artar.
    violated_per_baseline = max(1, int(violated_per_baseline))
    total = len(baseline_ids) * violated_per_baseline
    step = 0
    for base_id in baseline_ids:
        base = storage.get_ifc_model(base_id)
        if not base or base.get("status") != "ok":
            step += violated_per_baseline
            continue
        for variant_i in range(violated_per_baseline):
            step += 1
            try:
                selected_n = min(n_violations_per_ifc, len(pool))
                selected = random.sample(pool, selected_n)

                inj = ifc_inject.inject_violations(
                    baseline_id=base_id,
                    violations=selected,
                    pool_run_id=None,
                    model=model,
                    decoy_ratio=float(decoy_ratio),
                    compliant_addition_ratio=float(compliant_addition_ratio),
                    fill_from_pool=True,
                    generation_params={
                        "source": "rag_pipeline",
                        "dataset_tag": dataset_tag,
                        "collection": collection_name,
                        "categories": list(categories) if categories else "all",
                        "n_violations_per_ifc": int(n_violations_per_ifc),
                        "violated_per_baseline": violated_per_baseline,
                        "variant_index": variant_i + 1,
                        "decoy_ratio": float(decoy_ratio),
                        "compliant_addition_ratio": float(compliant_addition_ratio),
                        "rag_k": int(rag_k),
                        "pool_oversample": float(pool_oversample),
                        "model": model,
                        "llm_base_url": llm_base_url,
                        "rag_query": rag_query,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )

                summary = inj.get("summary", {})
                results["ok"] += 1
                results["violations_applied"] += summary.get(
                    "n_violations_applied",
                    summary.get("applied", summary.get("n_applied", len(selected))),
                )
                results["decoys"] += summary.get("decoys",
                                                  summary.get("n_decoys", 0))
                results["compliant"] += summary.get("compliant_additions", 0)
                results["items"].append({
                    "baseline": base.get("name"),
                    "violated": Path(inj.get("ifc_path", "")).stem,
                    "ifc_path": inj.get("ifc_path"),
                    "labels_path": inj.get("labels_path"),
                    "summary": summary,
                })
            except Exception as e:
                results["err"] += 1
                results["items"].append({
                    "baseline": base.get("name") or base_id[:8],
                    "error": str(e)[:200],
                })
            if progress_cb:
                progress_cb(step, total,
                            f"{step}/{total} · {results['ok']} OK · "
                            f"{results['err']} HATA")

    results["duration_s"] = round(time.time() - t0, 2)

    # 6) Excel log + defter güncelle
    if register_in_db:
        try:
            from llm.excel_log import log_llm_generation
            log_llm_generation(
                paket=dataset_tag,
                ifc_name=f"<batch:{results['ok']} IFC>",
                kind="rag_violation_batch",
                model=model,
                user_prompt=rag_query,
                prompt_tokens=0, completion_tokens=0,
                duration_s=results["duration_s"],
                cost_usd=0.0,
                design_summary=(
                    f"havuz={results['pool_size']} "
                    f"ihlal={results['violations_applied']} "
                    f"decoy={results['decoys']}"
                ),
                rationale=f"RAG+inject batch, {results['ok']}/{total} OK",
                parametreler={
                    "collection": collection_name,
                    "categories": categories or "all",
                    "n_violations_per_ifc": n_violations_per_ifc,
                    "decoy_ratio": decoy_ratio,
                    "rag_k": rag_k,
                },
                status="ok" if results["err"] == 0 else "partial",
                error=("" if results["err"] == 0
                       else f"{results['err']} IFC için hata"),
            )
        except Exception:
            pass

        try:
            from ml.tracking import log_operation, rebuild_dataset_registry
            from paths import data_home
            rebuild_dataset_registry(str(data_home()))
            log_operation(
                f"rag_ihlal_uretim ({model})",
                paket=dataset_tag, adet=results["ok"],
                ozet=(f"ihlal={results['violations_applied']} "
                      f"decoy={results['decoys']} "
                      f"havuz={results['pool_size']}"),
                parametreler=(
                    f"collection={collection_name} "
                    f"categories={len(categories) if categories else 16} "
                    f"n_per_ifc={n_violations_per_ifc} "
                    f"decoy_ratio={decoy_ratio}"
                ),
            )
        except Exception:
            pass

    return results
