"""train_models_large.py - 5-Fold Training & Prediction Generation for 668k Dataset.

Trains 4 models individually across 5 deterministic folds:
1. LightGBM
2. XGBoost
3. CatBoost
4. TabPFN-3.5 Fast (on Apple Silicon MPS)

Generates:
- level_2_meta/cache_large/<model>_oof.npy (shape: 668,665)
- level_2_meta/cache_large/<model>_test.npy (shape: 286,571)
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
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
import torch
import xgboost as xgb

# Add level_2_meta directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step1_large_data import (
    CACHE_LARGE_DIR,
    ID_COL,
    RESULTS_LARGE_DIR,
    prepare_large_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models_large")

SEED = 101
N_THREADS = 8
FREQ_COLS = ["Annual_Income_USD", "Daily_Commute_km", "Age"]


def apply_fold_frequency_encoding(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    freq_cols: List[str] = FREQ_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applies in-fold frequency encoding to prevent leakage into validation folds."""
    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_te_out = X_te.copy()

    for col in freq_cols:
        freq_map = X_tr[col].value_counts(normalize=True).to_dict()
        col_name = f"freq_{col}"
        X_tr_out[col_name] = X_tr[col].map(freq_map).fillna(0.0).astype(np.float32)
        X_va_out[col_name] = X_va[col].map(freq_map).fillna(0.0).astype(np.float32)
        X_te_out[col_name] = X_te[col].map(freq_map).fillna(0.0).astype(np.float32)

    return X_tr_out, X_va_out, X_te_out


# ---------------------------------------------------------------------------
# 1. LightGBM
# ---------------------------------------------------------------------------
def train_lightgbm_5fold(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    n_splits: int = 5,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    oof_cache = CACHE_LARGE_DIR / "lgb_oof.npy"
    test_cache = CACHE_LARGE_DIR / "lgb_test.npy"

    if oof_cache.exists() and test_cache.exists() and not force:
        logger.info(f"Reusing cached LightGBM predictions from {oof_cache}")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        auc = roc_auc_score(y, oof_preds)
        return oof_preds, test_preds, auc, []

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL 1: LIGHTGBM (5-Fold CV on 668k samples)")
    logger.info("=" * 70)

    n_samples = len(y)
    n_test = len(X_test)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    t_start = time.time()

    for fold in range(n_splits):
        t_fold = time.time()
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        # Apply strict in-fold frequency encoding
        X_tr_enc, X_va_enc, X_te_enc = apply_fold_frequency_encoding(X_tr, X_va, X_test)

        clf = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.088,
            num_leaves=36,
            max_depth=6,
            min_child_samples=30,
            max_bin=1024,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + fold,
            n_jobs=N_THREADS,
            verbose=-1,
        )

        clf.fit(
            X_tr_enc,
            y_tr,
            eval_set=[(X_va_enc, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        val_probs = clf.predict_proba(X_va_enc)[:, 1].astype(np.float32)
        oof_preds[va_idx] = val_probs

        test_probs = clf.predict_proba(X_te_enc)[:, 1].astype(np.float64)
        test_preds_accum += test_probs

        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        elapsed = time.time() - t_fold
        logger.info(
            f"  Fold {fold + 1}/{n_splits} - ROC AUC: {fold_auc:.6f} "
            f"(Best iter: {clf.best_iteration_}, Time: {elapsed:.1f}s)"
        )

        del X_tr, y_tr, X_va, y_va, X_tr_enc, X_va_enc, X_te_enc, clf
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    test_preds = (test_preds_accum / n_splits).astype(np.float32)

    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds)

    total_time = time.time() - t_start
    logger.info(
        f"LightGBM 5-Fold Complete | Overall OOF ROC AUC: {overall_auc:.6f} | "
        f"Mean Fold: {np.mean(fold_aucs):.6f} | Total Time: {total_time:.1f}s"
    )
    return oof_preds, test_preds, overall_auc, fold_aucs


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------
def train_xgboost_5fold(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    n_splits: int = 5,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    oof_cache = CACHE_LARGE_DIR / "xgb_oof.npy"
    test_cache = CACHE_LARGE_DIR / "xgb_test.npy"

    if oof_cache.exists() and test_cache.exists() and not force:
        logger.info(f"Reusing cached XGBoost predictions from {oof_cache}")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        auc = roc_auc_score(y, oof_preds)
        return oof_preds, test_preds, auc, []

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL 2: XGBOOST (5-Fold CV on 668k samples, max_bin=4096)")
    logger.info("=" * 70)

    n_samples = len(y)
    n_test = len(X_test)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    t_start = time.time()

    for fold in range(n_splits):
        t_fold = time.time()
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        # Apply in-fold frequency encoding
        X_tr_enc, X_va_enc, X_te_enc = apply_fold_frequency_encoding(X_tr, X_va, X_test)

        clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.092,
            max_depth=5,
            max_bin=4096,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="logloss",
            early_stopping_rounds=30,
            random_state=SEED + fold,
            n_jobs=N_THREADS,
        )

        clf.fit(
            X_tr_enc,
            y_tr,
            eval_set=[(X_va_enc, y_va)],
            verbose=False,
        )

        val_probs = clf.predict_proba(X_va_enc)[:, 1].astype(np.float32)
        oof_preds[va_idx] = val_probs

        test_probs = clf.predict_proba(X_te_enc)[:, 1].astype(np.float64)
        test_preds_accum += test_probs

        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        elapsed = time.time() - t_fold
        logger.info(
            f"  Fold {fold + 1}/{n_splits} - ROC AUC: {fold_auc:.6f} "
            f"(Best iter: {clf.best_iteration}, Time: {elapsed:.1f}s)"
        )

        del X_tr, y_tr, X_va, y_va, X_tr_enc, X_va_enc, X_te_enc, clf
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    test_preds = (test_preds_accum / n_splits).astype(np.float32)

    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds)

    total_time = time.time() - t_start
    logger.info(
        f"XGBoost 5-Fold Complete | Overall OOF ROC AUC: {overall_auc:.6f} | "
        f"Mean Fold: {np.mean(fold_aucs):.6f} | Total Time: {total_time:.1f}s"
    )
    return oof_preds, test_preds, overall_auc, fold_aucs


# ---------------------------------------------------------------------------
# 3. CatBoost
# ---------------------------------------------------------------------------
def train_catboost_5fold(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    n_splits: int = 5,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
    oof_cache = CACHE_LARGE_DIR / "cat_oof.npy"
    test_cache = CACHE_LARGE_DIR / "cat_test.npy"

    if oof_cache.exists() and test_cache.exists() and not force:
        logger.info(f"Reusing cached CatBoost predictions from {oof_cache}")
        oof_preds = np.load(oof_cache)
        test_preds = np.load(test_cache)
        auc = roc_auc_score(y, oof_preds)
        return oof_preds, test_preds, auc, []

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL 3: CATBOOST (5-Fold CV on 668k samples)")
    logger.info("=" * 70)

    n_samples = len(y)
    n_test = len(X_test)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    # Prepare string representations for CatBoost
    X_cb = X.copy()
    X_te_cb = X_test.copy()
    for col in cat_cols:
        X_cb[col] = X_cb[col].astype(str)
        X_te_cb[col] = X_te_cb[col].astype(str)

    t_start = time.time()

    for fold in range(n_splits):
        t_fold = time.time()
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr, y_tr = X_cb.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_cb.iloc[va_idx], y[va_idx]

        # Apply in-fold frequency encoding
        X_tr_enc, X_va_enc, X_te_enc = apply_fold_frequency_encoding(X_tr, X_va, X_te_cb)

        clf = cb.CatBoostClassifier(
            iterations=600,
            learning_rate=0.085,
            depth=5,
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=cat_cols,
            early_stopping_rounds=30,
            random_seed=SEED + fold,
            thread_count=N_THREADS,
            verbose=False,
        )

        clf.fit(X_tr_enc, y_tr, eval_set=(X_va_enc, y_va), verbose=False)

        val_probs = clf.predict_proba(X_va_enc)[:, 1].astype(np.float32)
        oof_preds[va_idx] = val_probs

        test_probs = clf.predict_proba(X_te_enc)[:, 1].astype(np.float64)
        test_preds_accum += test_probs

        fold_auc = roc_auc_score(y_va, val_probs)
        fold_aucs.append(fold_auc)
        elapsed = time.time() - t_fold
        logger.info(
            f"  Fold {fold + 1}/{n_splits} - ROC AUC: {fold_auc:.6f} "
            f"(Best iter: {clf.get_best_iteration()}, Time: {elapsed:.1f}s)"
        )

        del X_tr, y_tr, X_va, y_va, X_tr_enc, X_va_enc, X_te_enc, clf
        gc.collect()

    overall_auc = roc_auc_score(y, oof_preds)
    test_preds = (test_preds_accum / n_splits).astype(np.float32)

    np.save(oof_cache, oof_preds)
    np.save(test_cache, test_preds)

    total_time = time.time() - t_start
    logger.info(
        f"CatBoost 5-Fold Complete | Overall OOF ROC AUC: {overall_auc:.6f} | "
        f"Mean Fold: {np.mean(fold_aucs):.6f} | Total Time: {total_time:.1f}s"
    )
    return oof_preds, test_preds, overall_auc, fold_aucs


# ---------------------------------------------------------------------------
# 4. TabPFN-3.5 Fast (Apple Silicon MPS Acceleration)
# ---------------------------------------------------------------------------
def train_tabpfn_5fold(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    n_splits: int = 5,
    bag_sample_size: int = 5000,
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
    logger.info("TRAINING MODEL 4: TABPFN-3.5 FAST (5-Fold CV on Apple Silicon MPS)")
    logger.info("=" * 70)

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Target compute device for TabPFN: {device.upper()}")

    # Efficient categorical encoding for TabPFN using cat.codes
    X_pfn = X.copy()
    X_te_pfn = X_test.copy()
    cat_indices = []
    for col in cat_cols:
        X_pfn[col] = X[col].cat.codes.astype(np.int32)
        X_te_pfn[col] = pd.Categorical(X_test[col], categories=X[col].cat.categories).codes.astype(np.int32)
        cat_indices.append(X_pfn.columns.get_loc(col))

    n_samples = len(y)
    n_test = len(X_test)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(n_test, dtype=np.float64)
    fold_aucs = []

    t_start = time.time()
    chunk_size = 10000

    for fold in range(n_splits):
        t_fold = time.time()
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr_full, y_tr_full = X_pfn.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_pfn.iloc[va_idx], y[va_idx]

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

        X_tr = X_tr_full.iloc[sampled_sub_idx].values.astype(np.float32)
        y_tr = y_tr_full[sampled_sub_idx]

        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5_FAST,
            n_estimators=1,
            device=device,
            categorical_features_indices=cat_indices,
            random_state=SEED + fold,
        )

        t_fit0 = time.time()
        clf.fit(X_tr, y_tr)
        t_fit = time.time() - t_fit0

        # Chunked inference on validation fold (133k rows)
        val_probs_list = []
        X_va_mat = X_va.values.astype(np.float32)
        for i in range(0, len(X_va_mat), chunk_size):
            chunk = X_va_mat[i : i + chunk_size]
            probs = clf.predict_proba(chunk)[:, 1]
            val_probs_list.append(probs)

        val_probs = np.concatenate(val_probs_list).astype(np.float32)
        oof_preds[va_idx] = val_probs

        # Chunked inference on test set (286k rows)
        test_probs_list = []
        X_te_mat = X_te_pfn.values.astype(np.float32)
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
    return oof_preds, test_preds, overall_auc, fold_aucs


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------
def run_all_models(
    target_model: str = "all",
    force: bool = False,
) -> Dict[str, Any]:
    """Trains each model individually, generates OOF and test predictions, and saves verification metrics."""
    X_train, y, splits, X_test, test_ids, cat_cols, num_cols = prepare_large_data()

    results_summary: Dict[str, Any] = {}
    summary_file = RESULTS_LARGE_DIR / "models_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, "r") as f:
                results_summary = json.load(f)
        except Exception:
            pass

    # Model 1: LightGBM
    if target_model in ["lgb", "lightgbm", "all"]:
        oof_lgb, te_lgb, auc_lgb, fold_aucs_lgb = train_lightgbm_5fold(
            X_train, y, splits, X_test, cat_cols, force=force
        )
        results_summary["LightGBM"] = {
            "oof_auc": round(float(auc_lgb), 6),
            "fold_aucs": [round(float(a), 6) for a in fold_aucs_lgb],
            "oof_min": round(float(np.min(oof_lgb)), 6),
            "oof_max": round(float(np.max(oof_lgb)), 6),
        }

    # Model 2: XGBoost
    if target_model in ["xgb", "xgboost", "all"]:
        oof_xgb, te_xgb, auc_xgb, fold_aucs_xgb = train_xgboost_5fold(
            X_train, y, splits, X_test, cat_cols, force=force
        )
        results_summary["XGBoost"] = {
            "oof_auc": round(float(auc_xgb), 6),
            "fold_aucs": [round(float(a), 6) for a in fold_aucs_xgb],
            "oof_min": round(float(np.min(oof_xgb)), 6),
            "oof_max": round(float(np.max(oof_xgb)), 6),
        }

    # Model 3: CatBoost
    if target_model in ["cat", "catboost", "all"]:
        oof_cat, te_cat, auc_cat, fold_aucs_cat = train_catboost_5fold(
            X_train, y, splits, X_test, cat_cols, force=force
        )
        results_summary["CatBoost"] = {
            "oof_auc": round(float(auc_cat), 6),
            "fold_aucs": [round(float(a), 6) for a in fold_aucs_cat],
            "oof_min": round(float(np.min(oof_cat)), 6),
            "oof_max": round(float(np.max(oof_cat)), 6),
        }

    # Model 4: TabPFN-3.5 Fast
    if target_model in ["tabpfn", "all"]:
        oof_pfn, te_pfn, auc_pfn, fold_aucs_pfn = train_tabpfn_5fold(
            X_train, y, splits, X_test, cat_cols, force=force
        )
        results_summary["TabPFN"] = {
            "oof_auc": round(float(auc_pfn), 6),
            "fold_aucs": [round(float(a), 6) for a in fold_aucs_pfn],
            "oof_min": round(float(np.min(oof_pfn)), 6),
            "oof_max": round(float(np.max(oof_pfn)), 6),
        }

    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nAll requested models trained and saved to {summary_file}:")
    for m, info in results_summary.items():
        logger.info(f"  {m:<12} | OOF ROC AUC: {info['oof_auc']:.6f} | Range: [{info['oof_min']}, {info['oof_max']}]")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Train 4 models on 668k dataset")
    parser.add_argument("--model", choices=["lgb", "xgb", "cat", "tabpfn", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_all_models(target_model=args.model, force=args.force)


if __name__ == "__main__":
    main()
