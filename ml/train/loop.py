"""Training loop for the node-level GAT classifier."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from ml.data import IFCViolationDataset, split_by_baseline
from ml.data.splits import SplitIndices
from ml.model import GATNodeClassifier, HeteroGATNodeClassifier

from .config import TrainConfig
from .metrics import EvalResult, evaluate_predictions


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: TrainConfig, in_dim: int, num_edge_types: int) -> nn.Module:
    if cfg.model == "hetero_gat":
        return HeteroGATNodeClassifier(
            in_dim=in_dim, hidden_dim=cfg.hidden_dim,
            num_edge_types=num_edge_types, heads=cfg.heads, dropout=cfg.dropout,
        )
    return GATNodeClassifier(
        in_dim=in_dim, hidden_dim=cfg.hidden_dim,
        num_edge_types=num_edge_types, edge_emb_dim=cfg.edge_emb_dim,
        heads=cfg.heads, dropout=cfg.dropout,
    )


def _pos_weight(loader: DataLoader, device: str) -> torch.Tensor:
    pos = 0
    neg = 0
    for batch in loader:
        y = batch.y
        pos += int(y.sum().item())
        neg += int((y == 0).sum().item())
    w = (neg / pos) if pos > 0 else 1.0
    return torch.tensor([w], device=device)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val: EvalResult
    seconds: float


def _eval(model: nn.Module, loader: DataLoader, device: str, threshold: float) -> EvalResult:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    s_all: list[np.ndarray] = []          # raw sigmoid skorları — AUC için
    d_all: list[np.ndarray] = []
    c_all: list[str | None] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_type)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(np.int64)
            y_all.append(batch.y.cpu().numpy())
            p_all.append(preds)
            s_all.append(probs)
            d_all.append(batch.decoy_mask.cpu().numpy())
            for cats in batch.categories:
                c_all.extend(cats)
    return evaluate_predictions(
        np.concatenate(y_all),
        np.concatenate(p_all),
        np.concatenate(d_all),
        categories=c_all,
        y_score=np.concatenate(s_all),
    )


def run_training(
    cfg: TrainConfig,
    *,
    on_epoch_end: Callable[[int, float, EvalResult], None] | None = None,
    on_setup: Callable[[dict], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    filter_ifc_ids: list[str] | None = None,
    custom_splits: SplitIndices | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train the model with optional UI callbacks.

    Args:
        on_epoch_end: invoked after every epoch with the train loss and
            evaluation result on the validation split (so a Streamlit
            page can update a progress bar and metrics chart).
        on_setup: invoked once after dataset+splits resolved, with a
            small dict (`{"n_total", "feature_dim", "splits", "device",
            "run_dir", "pos_weight"}`) so the UI can print summary.
        on_log: every print-style message also forwarded here.
        filter_ifc_ids: if given, restrict the dataset to these IFC ids.
        custom_splits: if given, use these indices verbatim instead of
            running the stratified split. Indices index into the
            filtered dataset.
        should_stop: returning True aborts the next epoch (used to plug
            in a "Eğitimi durdur" button).

    Returns the same summary dict as before.
    """
    _set_seed(cfg.seed)
    run_root = cfg.ensure_dirs()
    name = cfg.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    device = cfg.resolve_device()

    def _log(msg: str) -> None:
        print(msg)
        if on_log:
            on_log(msg)

    _log(f"[train] device={device}  run_dir={run_dir}")

    ds_full = IFCViolationDataset(
        root=cfg.cache_root,
        dataset_root=cfg.dataset_root,
        include_baselines=cfg.include_baselines,
        use_rule_oracle=cfg.use_rule_oracle,
        mask_numeric_features=cfg.mask_numeric_features,
    )
    if filter_ifc_ids is not None:
        allow = set(filter_ifc_ids)
        keep = [i for i in range(len(ds_full)) if ds_full[i].ifc_id in allow]
        if not keep:
            raise RuntimeError("Filtre hiçbir IFC eşleştirmedi.")
        ds = ds_full[keep]
    else:
        ds = ds_full

    _log(f"[train] loaded {len(ds)} graphs  feature_dim={ds_full.feature_dim}")

    baseline_ids = [ds[i].baseline_id for i in range(len(ds))]
    if custom_splits is not None:
        splits = custom_splits
    else:
        splits = split_by_baseline(
            baseline_ids,
            val_frac=cfg.val_frac,
            test_frac=cfg.test_frac,
            seed=cfg.split_seed,
        )
    _log(
        f"[train] split sizes: train={len(splits.train)} "
        f"val={len(splits.val)} test={len(splits.test)}"
    )

    train_loader = DataLoader(ds[splits.train], batch_size=4, shuffle=True)
    val_loader = DataLoader(ds[splits.val], batch_size=4, shuffle=False) if splits.val else None
    test_loader = DataLoader(ds[splits.test], batch_size=4, shuffle=False) if splits.test else None

    model = _build_model(cfg, ds_full.feature_dim, ds_full.num_edge_types).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pw = (
        torch.tensor([cfg.pos_weight], device=device)
        if cfg.pos_weight is not None
        else _pos_weight(train_loader, device)
    )
    _log(f"[train] pos_weight={pw.item():.2f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    if on_setup:
        on_setup({
            "n_total": len(ds),
            "feature_dim": ds_full.feature_dim,
            "num_edge_types": ds_full.num_edge_types,
            "splits": {"train": len(splits.train), "val": len(splits.val),
                       "test": len(splits.test)},
            "device": device,
            "run_dir": str(run_dir),
            "pos_weight": float(pw.item()),
        })

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []
    stopped_early = False

    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            _log(f"[train] stop signal received before epoch {epoch}; halting.")
            stopped_early = True
            break
        t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_type)
            loss = loss_fn(logits, batch.y.float())
            loss.backward()
            optim.step()
            running += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
        train_loss = running / max(1, n)

        if val_loader is not None:
            val_res = _eval(model, val_loader, device, cfg.threshold)
        else:
            val_res = _eval(model, train_loader, device, cfg.threshold)
        dt = time.time() - t0

        _log(
            f"epoch {epoch:03d}  loss={train_loss:.4f}  "
            f"val_f1={val_res.f1:.3f}  P={val_res.precision:.3f}  "
            f"R={val_res.recall:.3f}  bal_acc={val_res.balanced_accuracy:.3f}  "
            f"MCC={val_res.mcc:+.2f}  AUC={val_res.auc_roc:.3f}  "
            f"decoy_fpr={val_res.decoy_fpr:.3f}  "
            f"[{dt:.1f}s]"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val": val_res.to_dict(),
             "seconds": dt}
        )
        if on_epoch_end:
            on_epoch_end(epoch, train_loss, val_res)

        if val_res.f1 > best_f1:
            best_f1 = val_res.f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                _log(f"[train] early stopping at epoch {epoch} (no val F1 improvement)")
                stopped_early = True
                break

    # Restore best weights for test.
    ckpt = run_dir / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    test_res = (
        _eval(model, test_loader, device, cfg.threshold).to_dict()
        if test_loader is not None
        else None
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test": test_res,
        "history": history,
        "stopped_early": stopped_early,
        "run_dir": str(run_dir),
        "ifc_ids": {
            "train": [ds[i].ifc_id for i in splits.train],
            "val":   [ds[i].ifc_id for i in splits.val],
            "test":  [ds[i].ifc_id for i in splits.test],
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"[train] done. best val F1={best_f1:.3f} @ epoch {best_epoch}")
    if test_res is not None:
        _log(f"[train] test: f1={test_res['f1']:.3f}  "
             f"P={test_res['precision']:.3f}  R={test_res['recall']:.3f}  "
             f"bal_acc={test_res['balanced_accuracy']:.3f}  "
             f"MCC={test_res['mcc']:+.2f}  AUC={test_res['auc_roc']:.3f}  "
             f"decoy_fpr={test_res['decoy_fpr']:.3f}")
        _log("[train] confusion matrix:")
        cm = test_res["confusion"]
        _log(f"  TN={cm['tn']:>6d}  FP={cm['fp']:>6d}")
        _log(f"  FN={cm['fn']:>6d}  TP={cm['tp']:>6d}")

    # macOS + torch + streamlit etkileşimindeki teardown segfault'unu
    # azaltmak için açıkça temizlik yap. Her zaman çözmez ama bazen yardımı dokunur.
    try:
        import gc
        del model, optim, loss_fn
        del train_loader
        if val_loader is not None:
            del val_loader
        if test_loader is not None:
            del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return summary
