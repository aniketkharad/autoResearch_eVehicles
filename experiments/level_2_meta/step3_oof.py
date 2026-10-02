"""step3_oof.py - Generate Level-1 5-Fold OOF Predictions with Checkpointing.

Generates out-of-fold predictions for:
- LightGBM
- XGBoost
- CatBoost
using the best hyperparameters discovered in Step 2, and validates with frozen TabPFN predictions.

Saves:
- level_2_meta/cache/lgb_oof.npy
- level_2_meta/cache/xgb_oof.npy
- level_2_meta/cache/cat_oof.npy
- level_2_meta/cache/tabpfn_oof.npy
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from common import (
    CACHE_DIR,
    RESULTS_DIR,
    SEED,
    evaluate_auc,
    load_json,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    get_or_create_cv_splits,
    load_and_preprocess_dev_data,
    scan_for_tabpfn_candidates,
)

logger = setup_logger("step3_oof")
CHECKPOINTS_DIR = CACHE_DIR / "checkpoints"


def train_and_predict_fold(
    model_name: str,
    params: Dict[str, Any],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: List[str],
    fold: int,
    early_stopping_rounds: int = 40,
    max_estimators: int = 4000,
    n_threads: int = 8,
) -> Tuple[np.ndarray, int]:
    """Trains a single model fold and returns predicted validation probabilities and best iteration."""
    if model_name == "lightgbm":
        clf = lgb.LGBMClassifier(
            n_estimators=max_estimators,
            random_state=SEED + fold,
            n_jobs=n_threads,
            verbose=-1,
            **params,
        )
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        preds = clf.predict_proba(X_va)[:, 1]
        best_iter = clf.best_iteration_

    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=max_estimators,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds,
            random_state=SEED + fold,
            n_jobs=n_threads,
            **params,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = clf.predict_proba(X_va)[:, 1]
        best_iter = clf.best_iteration

    elif model_name == "catboost":
        # CatBoost requires string/category handling
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)

        clf = cb.CatBoostClassifier(
            iterations=max_estimators,
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=cat_cols,
            early_stopping_rounds=early_stopping_rounds,
            random_seed=SEED + fold,
            thread_count=n_threads,
            verbose=False,
            **params,
        )
        clf.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), verbose=False)
        preds = clf.predict_proba(X_va_cb)[:, 1]
        best_iter = clf.get_best_iteration()
    else:
        raise ValueError(f"Unknown model {model_name}")

    return preds.astype(np.float32), int(best_iter)


def generate_tree_model_oof(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    cat_cols: List[str],
    n_splits: int = 5,
    force: bool = False,
    smoke: bool = False,
) -> np.ndarray:
    """Generates out-of-fold predictions with fold-level checkpointing and verification."""
    suffix = "_smoke" if smoke else ""
    target_cache = CACHE_DIR / f"{model_name}{suffix}_oof.npy"

    if target_cache.exists() and not force:
        oof_preds = np.load(target_cache)
        if len(oof_preds) == len(y):
            logger.info(f"Reusing existing OOF predictions for {model_name} from {target_cache}")
            print_model_oof_summary(model_name, y, oof_preds, splits)
            return oof_preds

    logger.info(f"\n{'='*70}\nGENERATING 5-FOLD OOF PREDICTIONS: {model_name.upper()}\n{'='*70}")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = len(y)
    oof_preds = np.full(n_samples, np.nan, dtype=np.float32)
    filled_mask = np.zeros(n_samples, dtype=bool)
    from concurrent.futures import ThreadPoolExecutor

    threads_per_worker = max(1, 8 // n_splits)

    def _process_fold(fold: int) -> Tuple[int, np.ndarray, int, float]:
        ckpt_file = CHECKPOINTS_DIR / f"{model_name}{suffix}_fold_{fold}.npz"
        val_idx = np.where(splits == fold)[0]
        train_idx = np.where(splits != fold)[0]

        if ckpt_file.exists() and not force:
            logger.info(f"Loading fold {fold + 1}/{n_splits} checkpoint from {ckpt_file}")
            ckpt_data = np.load(ckpt_file)
            fold_preds = ckpt_data["preds"]
            best_iter = int(ckpt_data["best_iter"])
        else:
            logger.info(f"Training fold {fold + 1}/{n_splits} (train: {len(train_idx):,}, val: {len(val_idx):,})...")
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            fold_preds, best_iter = train_and_predict_fold(
                model_name=model_name,
                params=params,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                cat_cols=cat_cols,
                fold=fold,
                n_threads=threads_per_worker,
            )
            # Save checkpoint
            np.savez_compressed(ckpt_file, preds=fold_preds, best_iter=best_iter)
            del X_tr, y_tr, X_va, y_va
            gc.collect()

        fold_auc = evaluate_auc(y[val_idx], fold_preds)
        logger.info(f"Fold {fold + 1}/{n_splits} AUC: {fold_auc:.5f} (Best Iteration: {best_iter})")
        return fold, fold_preds, best_iter, fold_auc

    with ThreadPoolExecutor(max_workers=n_splits) as executor:
        results = list(executor.map(_process_fold, range(n_splits)))

    # Sort by fold index and assemble
    results.sort(key=lambda r: r[0])
    for fold, fold_preds, best_iter, fold_auc in results:
        val_idx = np.where(splits == fold)[0]
        oof_preds[val_idx] = fold_preds
        filled_mask[val_idx] = True
        fold_aucs.append(fold_auc)
        best_iters.append(best_iter)

    # Assertions
    assert len(oof_preds) == n_samples, f"Row count mismatch: expected {n_samples}, got {len(oof_preds)}"
    assert not np.isnan(oof_preds).any(), f"NaNs detected in {model_name} OOF predictions!"
    assert not np.isinf(oof_preds).any(), f"Infs detected in {model_name} OOF predictions!"
    assert filled_mask.all(), f"Not all samples were predicted in {model_name} OOF generation!"

    # Save finalized OOF array
    np.save(target_cache, oof_preds)
    logger.info(f"Saved {model_name.upper()} OOF predictions to {target_cache}")

    print_model_oof_summary(model_name, y, oof_preds, splits, fold_aucs, best_iters)
    return oof_preds


def print_model_oof_summary(
    model_name: str,
    y_true: np.ndarray,
    oof_preds: np.ndarray,
    splits: np.ndarray,
    fold_aucs: Optional[List[float]] = None,
    best_iters: Optional[List[int]] = None,
) -> None:
    """Prints comprehensive verification metrics for an OOF prediction array."""
    if fold_aucs is None:
        n_splits = int(np.max(splits) + 1)
        fold_aucs = [evaluate_auc(y_true[splits == f], oof_preds[splits == f]) for f in range(n_splits)]

    overall_auc = evaluate_auc(y_true, oof_preds)
    p_min, p_max = float(np.min(oof_preds)), float(np.max(oof_preds))
    nan_count = int(np.isnan(oof_preds).sum())

    msg = (
        f"\n{'='*70}\n"
        f"OOF VERIFICATION SUMMARY: {model_name.upper()}\n"
        f"Number of rows:     {len(oof_preds):,}\n"
        f"Fold AUCs:          {[round(a, 5) for a in fold_aucs]}\n"
        f"Mean Fold AUC:      {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})\n"
        f"Overall OOF AUC:    {overall_auc:.5f}\n"
        f"Prediction Range:   [{p_min:.5f}, {p_max:.5f}]\n"
        f"NaN Count:          {nan_count}\n"
    )
    if best_iters:
        msg += f"Best Iterations:    {best_iters}\n"
    msg += f"{'='*70}\n"
    logger.info(msg)


def run_all_oof_generation(
    force: bool = False,
    smoke: bool = False,
    tabpfn_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Generates and validates OOF predictions for all 4 models."""
    suffix = "_smoke" if smoke else ""

    X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data()
    splits = get_or_create_cv_splits(y)

    models = ["lightgbm", "xgboost", "catboost"]
    oof_dict: Dict[str, np.ndarray] = {}

    for model_name in models:
        params_file = RESULTS_DIR / f"{model_name}{suffix}_best_params.json"
        if not params_file.exists():
            # Fall back to non-smoke params if smoke params do not exist
            params_file = RESULTS_DIR / f"{model_name}_best_params.json"
        if not params_file.exists():
            raise FileNotFoundError(f"Best hyperparameters not found at {params_file}. Run step2_tune.py first!")

        info = load_json(params_file)
        params = info["best_params"]
        logger.info(f"Loaded {model_name} best parameters from {params_file}")

        oof = generate_tree_model_oof(
            model_name=model_name,
            params=params,
            X=X,
            y=y,
            splits=splits,
            cat_cols=cat_cols,
            force=force,
            smoke=smoke,
        )
        oof_dict[model_name] = oof

    # Obtain Frozen TabPFN OOF
    tabpfn_cache = CACHE_DIR / "tabpfn_oof.npy"
    if tabpfn_path:
        tabpfn_file = Path(tabpfn_path)
    elif tabpfn_cache.exists():
        tabpfn_file = tabpfn_cache
    else:
        tabpfn_file = None

    tabpfn_preds = scan_for_tabpfn_candidates(
        target_length=len(y),
        y_true=y,
        explicit_path=tabpfn_file,
    )
    if tabpfn_preds is None:
        raise FileNotFoundError(
            "Frozen TabPFN OOF predictions could not be located. "
            "Run generate_tabpfn_dev.py first or supply --tabpfn-oof <path>."
        )

    oof_dict["tabpfn"] = tabpfn_preds
    print_model_oof_summary("tabpfn", y, tabpfn_preds, splits)

    return oof_dict


def main():
    parser = argparse.ArgumentParser(description="Step 3: Generate Level-1 OOF predictions")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tabpfn-oof", type=str, default=None)
    args = parser.parse_args()

    run_all_oof_generation(force=args.force, smoke=args.smoke, tabpfn_path=args.tabpfn_oof)


if __name__ == "__main__":
    main()
