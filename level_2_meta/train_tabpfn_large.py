"""train_tabpfn_large.py - Dedicated TabPFN-3.5 5-Fold Training on Apple Silicon MPS.

Isolated process execution prevents OpenMP / Metal runtime collisions with LightGBM and CatBoost.
Generates:
- level_2_meta/cache_large/tabpfn_oof.npy (668,665)
- level_2_meta/cache_large/tabpfn_test.npy (286,571)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

# Add level_2_meta to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step1_large_data import (
    CACHE_LARGE_DIR,
    RESULTS_LARGE_DIR,
    prepare_large_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_tabpfn_large")

SEED = 101


def run_tabpfn_5fold(
    bag_sample_size: int = 5000,
    n_splits: int = 5,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    oof_cache = CACHE_LARGE_DIR / "tabpfn_oof.npy"
    test_cache = CACHE_LARGE_DIR / "tabpfn_test.npy"

    if oof_cache.exists() and test_cache.exists() and not force:
        logger.info(f"Reusing cached TabPFN-3.5 predictions from {oof_cache}")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        auc = roc_auc_score(y, oof_preds)
        return oof_preds, test_preds, auc, []

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING TABPFN-3.5 FULL (5-Fold CV on Apple Silicon MPS - Process Isolated)")
    logger.info("=" * 70)

    X_train, y, splits, X_test, test_ids, cat_cols, num_cols = prepare_large_data()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Target compute device for TabPFN: {device.upper()}")

    # Vectorized categorical encoding
    X_pfn = X_train.copy()
    X_te_pfn = X_test.copy()
    cat_indices = []
    for col in cat_cols:
        X_pfn[col] = X_train[col].cat.codes.astype(np.int32)
        X_te_pfn[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories).codes.astype(np.int32)
        cat_indices.append(int(X_pfn.columns.get_loc(col)))

    # Pre-materialize float32 matrices
    X_mat = X_pfn.values.astype(np.float32)
    X_te_mat = X_te_pfn.values.astype(np.float32)

    del X_pfn, X_te_pfn, X_train, X_test
    gc.collect()

    n_samples = len(y)
    n_test = len(X_te_mat)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    t_start = time.time()
    chunk_size = 50000

    for fold in range(n_splits):
        t_fold = time.time()
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr_full = X_mat[tr_idx]
        y_tr_full = y[tr_idx]
        X_va = X_mat[va_idx]
        y_va = y[va_idx]

        # Stratified sampling of context window for TabPFN (5,000 samples)
        idx_0 = np.where(y_tr_full == 0)[0]
        idx_1 = np.where(y_tr_full == 1)[0]
        ratio_1 = len(idx_1) / len(y_tr_full)
        n_1 = int(bag_sample_size * ratio_1)
        n_0 = bag_sample_size - n_1

        rng = np.random.default_rng(SEED + fold)
        samp_0 = rng.choice(idx_0, size=n_0, replace=False)
        samp_1 = rng.choice(idx_1, size=n_1, replace=False)
        sampled_sub_idx = np.concatenate([samp_0, samp_1])
        rng.shuffle(sampled_sub_idx)

        X_tr = X_tr_full[sampled_sub_idx]
        y_tr = y_tr_full[sampled_sub_idx]

        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5,
            n_estimators=1,
            device=device,
            categorical_features_indices=cat_indices,
            random_state=SEED + fold,
        )

        t_fit0 = time.time()
        clf.fit(X_tr, y_tr)
        t_fit = time.time() - t_fit0

        # Chunked validation inference (133k rows)
        val_probs_list = []
        for i in range(0, len(X_va), chunk_size):
            chunk = X_va[i : i + chunk_size]
            probs = clf.predict_proba(chunk)[:, 1]
            val_probs_list.append(probs)

        val_probs = np.concatenate(val_probs_list).astype(np.float32)
        oof_preds[va_idx] = val_probs

        # Chunked test inference (286k rows)
        test_probs_list = []
        for i in range(0, len(X_te_mat), chunk_size):
            chunk = X_te_mat[i : i + chunk_size]
            probs = clf.predict_proba(chunk)[:, 1]
            test_probs_list.append(probs)

        test_probs = np.concatenate(test_probs_list).astype(np.float64)
        test_preds_accum += test_probs

        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        elapsed = time.time() - t_fold
        logger.info(
            f"  Fold {fold + 1}/{n_splits} - ROC AUC: {fold_auc:.6f} "
            f"(Fit: {t_fit:.1f}s, Fold Total: {elapsed:.1f}s)"
        )

        del clf, X_tr, y_tr, X_tr_full, y_tr_full, X_va, y_va, val_probs_list, test_probs_list
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    test_preds = (test_preds_accum / n_splits).astype(np.float32)

    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds)

    total_time = time.time() - t_start
    logger.info(
        f"TabPFN-3.5 5-Fold Complete | Overall OOF ROC AUC: {overall_auc:.6f} | "
        f"Mean Fold: {np.mean(fold_aucs):.6f} | Total Time: {total_time:.1f}s"
    )

    # Update summary file
    summary_file = RESULTS_LARGE_DIR / "models_summary.json"
    summary_data = {}
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)
        except Exception:
            pass

    summary_data["TabPFN"] = {
        "oof_auc": round(float(overall_auc), 6),
        "fold_aucs": [round(float(a), 6) for a in fold_aucs],
        "oof_min": round(float(np.min(oof_preds)), 6),
        "oof_max": round(float(np.max(oof_preds)), 6),
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    return oof_preds, test_preds, overall_auc, fold_aucs


def main():
    parser = argparse.ArgumentParser(description="Train TabPFN-3.5 on 668k dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_tabpfn_5fold(force=args.force)


if __name__ == "__main__":
    main()
