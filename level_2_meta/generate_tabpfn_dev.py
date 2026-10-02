"""generate_tabpfn_dev.py - Generates and freezes 5-fold OOF predictions using TabPFN-3.5 Fast.

Executed once on EV_Adoption_and_Range_Anxiety_Dataset.csv.
Saves frozen OOF predictions to level_2_meta/cache/tabpfn_oof.npy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from common import (
    CACHE_DIR,
    SEED,
    detect_hardware_and_env,
    evaluate_auc,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    get_or_create_cv_splits,
    load_and_preprocess_dev_data,
)

logger = setup_logger("generate_tabpfn_dev")


def run_tabpfn_5fold_oof(
    csv_path: Path = DEFAULT_DEV_CSV,
    n_estimators: int = 2,
    device: str = "mps",
    force: bool = False,
    smoke: bool = False,
) -> np.ndarray:
    """Trains TabPFN-3.5 Fast across 5 stratified folds and saves frozen OOF predictions."""
    target_cache = CACHE_DIR / ("tabpfn_oof_smoke.npy" if smoke else "tabpfn_oof.npy")
    if target_cache.exists() and not force:
        preds = np.load(target_cache)
        logger.info(f"Reusing existing frozen TabPFN OOF predictions from {target_cache} (shape: {preds.shape})")
        return preds

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FROZEN TABPFN-3.5 FAST 5-FOLD OOF PREDICTIONS")
    logger.info("=" * 70)

    X_raw, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data(csv_path)
    splits = get_or_create_cv_splits(y, n_splits=5, force=False)

    if smoke:
        logger.info("Smoke mode enabled: processing only fold 0...")
        folds_to_run = [0]
    else:
        folds_to_run = list(range(5))

    # Preprocess for TabPFN: Ordinal encode categoricals, impute numerics with median
    X = X_raw.copy()
    for col in num_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    cat_indices = []
    for col in cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[col] = oe.fit_transform(X[[col]].astype(str)).astype(np.int32)
        cat_indices.append(X.columns.get_loc(col))

    X_mat = X.values.astype(np.float32)
    n_samples = len(y)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    fold_aucs = []

    # Check hardware
    env_info = detect_hardware_and_env()
    selected_device = device if (device == "mps" and env_info["mps_available"]) else "cpu"
    logger.info(f"Target compute device for TabPFN: {selected_device.upper()}")

    t_start_total = time.time()

    for fold in folds_to_run:
        train_idx = np.where(splits != fold)[0]
        val_idx = np.where(splits == fold)[0]

        X_tr, y_tr = X_mat[train_idx], y[train_idx]
        X_va, y_va = X_mat[val_idx], y[val_idx]

        logger.info(f"\n--- Fold {fold + 1}/5: Train {len(X_tr):,} samples | Val {len(X_va):,} samples ---")
        t0 = time.time()

        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5_FAST,
            n_estimators=n_estimators,
            device=selected_device,
            categorical_features_indices=cat_indices,
            random_state=SEED + fold,
        )

        clf.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        # Predict in chunks of 1000
        val_probs_list = []
        chunk_size = 1000
        for i in range(0, len(X_va), chunk_size):
            chunk = X_va[i : i + chunk_size]
            probs = clf.predict_proba(chunk)[:, 1]
            val_probs_list.append(probs)

        val_probs = np.concatenate(val_probs_list).astype(np.float32)
        oof_preds[val_idx] = val_probs

        fold_auc = evaluate_auc(y_va, val_probs)
        fold_aucs.append(fold_auc)
        t_fold = time.time() - t0
        logger.info(f"Fold {fold + 1} completed in {t_fold:.1f}s (fit: {t_fit:.1f}s) | ROC AUC: {fold_auc:.5f}")

    if not smoke:
        overall_auc = evaluate_auc(y, oof_preds)
        logger.info(f"\n{'='*70}")
        logger.info(f"TabPFN 5-Fold OOF Summary:")
        logger.info(f"  Fold AUCs:   {[round(a, 5) for a in fold_aucs]}")
        logger.info(f"  Mean AUC:    {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f})")
        logger.info(f"  Overall OOF: {overall_auc:.5f}")
        logger.info(f"  Total Time:  {time.time() - t_start_total:.1f}s")
        logger.info(f"{'='*70}\n")
    else:
        logger.info(f"Smoke run finished on Fold 0. Fold 0 AUC: {fold_aucs[0]:.5f}")

    target_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_cache, oof_preds)
    logger.info(f"Successfully cached frozen TabPFN OOF predictions at: {target_cache}")
    return oof_preds


def main():
    parser = argparse.ArgumentParser(description="Generate and freeze TabPFN-3.5 predictions")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DEV_CSV))
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_tabpfn_5fold_oof(
        csv_path=Path(args.data),
        n_estimators=args.n_estimators,
        device=args.device,
        force=args.force,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
