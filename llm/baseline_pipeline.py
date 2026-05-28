"""LLM-bazlı baseline üretim pipeline'ı (hiyerarşik: ana + N varyant).

Akış:
  1. Kullanıcı template prompt verir.
  2. plan_ana_baseline → ana baseline DesignPlan + IFC üretimi.
  3. N kez plan_variant_baseline → her biri kendi LLM çağrısıyla varyant
     DesignPlan + IFC.
  4. DB kayıtları:
       Ana baseline:  kind="baseline", parent_id=None, prompt=user_prompt,
                      llm_model, params içinde sistem prompt + LLM yanıtı.
       Varyantlar:    kind="baseline", parent_id=ana.id, prompt=user_prompt,
                      llm_model, params içinde varyant özelliği.

Görüntüleyici (sayfa 15) bu hiyerarşiyi kullanır:
  * Ana baseline = paket içinde parent_id NULL olan tek baseline.
  * Varyantlar = ana'nın doğrudan çocukları.
  * İhlalliler = her varyantın doğrudan çocukları (kind="violated").
"""
from __future__ import annotations

import uuid
from pathlib import Path

from violation_pool import storage
from violation_pool.config import settings

from llm.design_planner import (
    DesignPlan, plan_ana_baseline, plan_variant_baseline,
)
from ml.data.synth_baseline_v2 import SynthParamsV2, generate_v2


def run_baseline_pipeline(user_template: str,
                          dataset_tag: str, *,
                          variants: int = 5, seed_start: int = 0,
                          model: str = "gpt-4o",
                          progress_cb=None) -> dict:
    """Ana baseline + N varyant LLM ile üretir, dosya + DB + defterler.

    Args:
        user_template: text_area'dan gelen bina tarifi (her IFC için aynı
            template kullanılır, varyasyon LLM tarafından eklenir).
        dataset_tag: paket adı (DB'de dataset_tag, dosya prefix'i).
        variants: ana baseline'a ek üretilecek varyant sayısı.
        seed_start: tohum başlangıcı. Ana = seed_start, varyant_i = seed_start+1+i.
        model: OpenAI model adı (gpt-4o vb.).
        progress_cb: (done, total, label) → None.

    Returns:
        {
          "ana_id": str | None,
          "ana_plan": DesignPlan | None,
          "ana_path": str | None,
          "variant_ids": list[str],
          "variant_plans": list[DesignPlan],
          "variant_paths": list[str],
          "errors": list[str],
        }
    """
    out_dir = settings.ifc_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    params = SynthParamsV2()

    total = 1 + variants
    done = 0
    errors: list[str] = []

    # 1) Ana baseline ----------------------------------------------------
    ana_plan: DesignPlan | None = None
    ana_id: str | None = None
    ana_path: str | None = None
    try:
        ana_plan = plan_ana_baseline(user_template, model=model,
                                     seed=seed_start)
    except Exception as e:
        errors.append(f"ana baseline · LLM çağrısı hata: {e}")
        if progress_cb:
            progress_cb(done + 1, total, "ana: HATA")
        # Ana baseline LLM hatası fatal — varyant üretmenin anlamı yok.
        return {"ana_id": None, "ana_plan": None, "ana_path": None,
                "variant_ids": [], "variant_plans": [], "variant_paths": [],
                "errors": errors}

    ana_id = str(uuid.uuid4())
    ana_stem = f"{dataset_tag}_ana_{seed_start:05d}"
    ana_path = out_dir / f"{ana_stem}.ifc"
    try:
        info = generate_v2(
            seed=seed_start, out_path=ana_path,
            params=params, override=ana_plan.to_override(),
            ifc_id=ana_id,
        )
        _register_baseline(
            info, dataset_tag=dataset_tag, parent_id=None,
            prompt=ana_plan.user_prompt, llm_model=model,
            extra_meta={"kind_label": "ana_baseline",
                        "design_plan": ana_plan.to_dict()},
        )
        ana_path = str(ana_path)
    except Exception as e:
        errors.append(f"ana baseline · IFC üretim hata: {e}")
        ana_id = None
        ana_path = None
    done += 1
    if progress_cb:
        progress_cb(done, total,
                    f"ana: {ana_stem}" if ana_id else "ana: HATA")

    if ana_id is None:
        # Ana çizilemediyse varyant üretmek anlamsız.
        return {"ana_id": None, "ana_plan": ana_plan, "ana_path": None,
                "variant_ids": [], "variant_plans": [], "variant_paths": [],
                "errors": errors}

    # 2) Varyantlar ------------------------------------------------------
    variant_ids: list[str] = []
    variant_plans: list[DesignPlan] = []
    variant_paths: list[str] = []
    for i in range(variants):
        var_seed = seed_start + 1 + i
        var_stem = f"{dataset_tag}_{var_seed:05d}"
        # Çakışan dosya adı için sayaç
        n = 1
        while (out_dir / f"{var_stem}.ifc").exists():
            var_stem = f"{dataset_tag}_{var_seed:05d}_{n}"
            n += 1
        var_path = out_dir / f"{var_stem}.ifc"
        try:
            var_plan = plan_variant_baseline(
                user_template, ana_plan, model=model, seed=var_seed,
            )
        except Exception as e:
            errors.append(f"varyant {var_seed} · LLM hata: {e}")
            done += 1
            if progress_cb:
                progress_cb(done, total, f"varyant {var_seed}: LLM HATA")
            continue

        var_id = str(uuid.uuid4())
        try:
            info_v = generate_v2(
                seed=var_seed, out_path=var_path,
                params=params, override=var_plan.to_override(),
                ifc_id=var_id,
            )
            _register_baseline(
                info_v, dataset_tag=dataset_tag, parent_id=ana_id,
                prompt=var_plan.user_prompt, llm_model=model,
                extra_meta={"kind_label": "variant_baseline",
                            "design_plan": var_plan.to_dict()},
            )
            variant_ids.append(var_id)
            variant_plans.append(var_plan)
            variant_paths.append(str(var_path))
        except Exception as e:
            errors.append(f"varyant {var_seed} · IFC hata: {e}")
        done += 1
        if progress_cb:
            progress_cb(done, total, f"varyant: {var_stem}")

    # 3) Defterler -------------------------------------------------------
    try:
        from ml.tracking import log_operation, rebuild_dataset_registry
        from paths import data_home
        rebuild_dataset_registry(str(data_home()))
        log_operation(
            f"llm_baseline_uretim ({model})",
            paket=dataset_tag,
            adet=1 + len(variant_ids),
            ozet=f"ana + {len(variant_ids)} varyant ({len(errors)} hata)",
            parametreler=(f"variants={variants} seed_start={seed_start} "
                          f"model={model}"),
        )
    except Exception:
        pass

    return {
        "ana_id": ana_id,
        "ana_plan": ana_plan,
        "ana_path": ana_path,
        "variant_ids": variant_ids,
        "variant_plans": variant_plans,
        "variant_paths": variant_paths,
        "errors": errors,
    }


def _register_baseline(info: dict, *, dataset_tag: str,
                       parent_id: str | None, prompt: str,
                       llm_model: str, extra_meta: dict) -> None:
    """codex1 storage'a baseline kaydı (hiyerarşik parent_id ile)."""
    try:
        params = dict(info["spec"].get("_meta") or {})
        params.update(extra_meta)
        storage.create_ifc_model(
            id=info["ifc_id"], kind="baseline",
            name=Path(info["ifc_path"]).stem,
            parent_id=parent_id, pool_run_id=None,
            params=params,
            file_path=info["ifc_path"],
            meta_path=info.get("meta_path"),
            labels_path=info.get("labels_path"),
            graph_path=info.get("graph_path"),
            llm_model=llm_model,
            prompt=prompt,
            status="ok", error=None,
            dataset_tag=dataset_tag,
        )
    except Exception as e:
        print(f"[llm_baseline] DB kaydı atlandı: {e}")
