"""generate_submission.py - Generate and submit Level-2 Stacking predictions on test.csv.

Workflow:
1. Loads test.csv and verifies schema compatibility with development features.
2. Generates test predictions from the 5 folds of tuned LightGBM, XGBoost, and CatBoost models.
3. Loads frozen TabPFN test predictions (tabpnf_35_subs.csv).
4. Rank-transforms test predictions: R_m = rankdata(P_m) / N_test.
5. Blends using Nelder-Mead / SLSQP optimal weights from weights.json:
       P_stack = sum(w_m * R_m)
6. Verifies submission integrity and saves:
   - level_2_meta/submission_level2_stack.csv
   - level_2_meta/results/submission_level2_stack.csv
7. Submits to Kaggle Playground Series s6e9 competition and checks leaderboard status.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import xgboost as xgb

from common import (
    BASE_DIR,
    CACHE_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    load_json,
    rank_transform,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    DEFAULT_TEST_CSV,
    get_or_create_cv_splits,
    load_and_preprocess_dev_data,
)

logger = setup_logger("generate_submission")
COMPETITION_ID = "playground-series-s6e9"


def load_and_preprocess_test_data(
    test_path: Path = DEFAULT_TEST_CSV,
    feature_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Loads and formats test features to match training feature specifications."""
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    logger.info(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)
    test_ids = df_test["id"].values

    if feature_cols is None:
        feature_cols = [c for c in df_test.columns if c != "id"]

    X_test = df_test[feature_cols].copy()

    # Format numeric columns as float32
    if num_cols:
        for c in num_cols:
            if c in X_test.columns:
                X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float32)

    # Format categorical columns as pandas category
    if cat_cols:
        for c in cat_cols:
            if c in X_test.columns:
                X_test[c] = X_test[c].astype("category")

    logger.info(f"Formatted test feature matrix: {X_test.shape[0]:,} samples x {X_test.shape[1]} features.")
    return X_test, test_ids


def predict_tree_models_on_test(
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    splits: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    n_splits: int = 5,
) -> Dict[str, np.ndarray]:
    """Trains 5 folds for each tree model and generates averaged test predictions."""
    models = ["lightgbm", "xgboost", "catboost"]
    test_preds: Dict[str, np.ndarray] = {}
    n_test = len(X_test)

    for model_name in models:
        params_file = RESULTS_DIR / f"{model_name}_best_params.json"
        if not params_file.exists():
            raise FileNotFoundError(f"Missing best params: {params_file}")

        info = load_json(params_file)
        params = info["best_params"]
        best_iter = info.get("best_iteration", 100)

        logger.info(f"\nGenerating test predictions for {model_name.upper()} across {n_splits} folds...")
        t0 = time.time()

        model_test_accum = np.zeros(n_test, dtype=np.float64)

        # Handle CatBoost categoricals
        if model_name == "catboost":
            X_tr_base = X_dev.copy()
            X_te_base = X_test.copy()
            for c in cat_cols:
                X_tr_base[c] = X_tr_base[c].astype(str)
                X_te_base[c] = X_te_base[c].astype(str)
        else:
            X_tr_base = X_dev
            X_te_base = X_test

        for fold in range(n_splits):
            tr_idx = np.where(splits != fold)[0]
            va_idx = np.where(splits == fold)[0]

            X_tr, y_tr = X_tr_base.iloc[tr_idx], y_dev[tr_idx]
            X_va, y_va = X_tr_base.iloc[va_idx], y_dev[va_idx]

            if model_name == "lightgbm":
                clf = lgb.LGBMClassifier(
                    n_estimators=3000,
                    random_state=SEED + fold,
                    n_jobs=8,
                    verbose=-1,
                    **params,
                )
                clf.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[lgb.early_stopping(35, verbose=False)],
                )
                p_fold = clf.predict_proba(X_te_base)[:, 1]

            elif model_name == "xgboost":
                clf = xgb.XGBClassifier(
                    n_estimators=3000,
                    tree_method="hist",
                    enable_categorical=True,
                    eval_metric="logloss",
                    early_stopping_rounds=35,
                    random_state=SEED + fold,
                    n_jobs=8,
                    **params,
                )
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                p_fold = clf.predict_proba(X_te_base)[:, 1]

            elif model_name == "catboost":
                clf = cb.CatBoostClassifier(
                    iterations=3000,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    cat_features=cat_cols,
                    early_stopping_rounds=35,
                    random_seed=SEED + fold,
                    thread_count=8,
                    verbose=False,
                    **params,
                )
                clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
                p_fold = clf.predict_proba(X_te_base)[:, 1]

            model_test_accum += p_fold
            logger.info(f"  Fold {fold + 1}/{n_splits} complete (test pred mean: {np.mean(p_fold):.4f})")

        p_test_avg = (model_test_accum / n_splits).astype(np.float32)
        test_preds[model_name] = p_test_avg
        elapsed = time.time() - t0
        logger.info(f"{model_name.upper()} 5-fold test inference finished in {elapsed:.1f}s.")

    return test_preds


def load_tabpfn_test_predictions(test_ids: np.ndarray) -> np.ndarray:
    """Loads TabPFN test predictions from tabpnf_35_subs.csv."""
    candidate_paths = [
        PROJECT_ROOT.parent / "eVechicle_pred" / "tabpnf_35_subs.csv",
        PROJECT_ROOT.parent / "eVechicle_pred" / "data" / "tabpnf_35_subs.csv",
        PROJECT_ROOT.parent / "eVechicle_pred" / "kaggle_submissions" / "tabpnf_35_subs.csv",
        CACHE_DIR / "tabpfn_test_preds.npy",
    ]

    found_path = None
    for cp in candidate_paths:
        if cp.exists():
            found_path = cp
            break

    if not found_path:
        raise FileNotFoundError(
            f"Could not locate TabPFN test predictions in candidate locations: {candidate_paths}"
        )

    logger.info(f"Loading frozen TabPFN test predictions from: {found_path}")
    if found_path.suffix == ".csv":
        df = pd.read_csv(found_path)
        # Verify alignment
        if "id" in df.columns:
            assert (df["id"].values == test_ids).all(), "Test ID alignment mismatch in TabPFN predictions!"
        target_col = "Will_Buy_EV" if "Will_Buy_EV" in df.columns else df.columns[-1]
        pfn_test = df[target_col].values.astype(np.float32)
    else:
        pfn_test = np.load(found_path).astype(np.float32)

    assert len(pfn_test) == len(test_ids), f"Length mismatch: {len(pfn_test)} vs {len(test_ids)}"
    logger.info(
        f"TabPFN test predictions loaded: {len(pfn_test):,} rows | Range: [{np.min(pfn_test):.5f}, {np.max(pfn_test):.5f}]"
    )
    return pfn_test


def build_stacked_submission() -> Path:
    """Generates the Level-2 stacking ensemble submission file."""
    # 1. Load Dev and Test Data
    X_dev, y_dev, cat_cols, num_cols, dev_ids = load_and_preprocess_dev_data(DEFAULT_DEV_CSV)
    splits = get_or_create_cv_splits(y_dev)
    X_test, test_ids = load_and_preprocess_test_data(
        DEFAULT_TEST_CSV,
        feature_cols=list(X_dev.columns),
        cat_cols=cat_cols,
        num_cols=num_cols,
    )
    n_test = len(test_ids)

    # 2. Get Tree Models Test Predictions
    tree_test_preds = predict_tree_models_on_test(
        X_dev=X_dev,
        y_dev=y_dev,
        splits=splits,
        X_test=X_test,
        cat_cols=cat_cols,
    )

    # 3. Load Frozen TabPFN Test Predictions
    pfn_test = load_tabpfn_test_predictions(test_ids)

    # 4. Load Optimal Stacking Weights
    weights_path = RESULTS_DIR / "weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}. Run step4_stack.py first!")

    weights_info = load_json(weights_path)
    weights = weights_info["final_weights"]
    logger.info(f"\nLoaded Level-2 Stacking Weights:")
    for k, v in weights.items():
        logger.info(f"  {k:<12}: {v:.5f}")

    w_lgb = weights["LightGBM"]
    w_xgb = weights["XGBoost"]
    w_cat = weights["CatBoost"]
    w_pfn = weights["TabPFN"]

    # 5. Apply Rank Transformation
    logger.info("\nApplying rank transformation across all 4 test prediction vectors...")
    r_lgb = rank_transform(tree_test_preds["lightgbm"])
    r_xgb = rank_transform(tree_test_preds["xgboost"])
    r_cat = rank_transform(tree_test_preds["catboost"])
    r_pfn = rank_transform(pfn_test)

    # 6. Compute Weighted Ensemble
    stacked_probs = (w_lgb * r_lgb + w_xgb * r_xgb + w_cat * r_cat + w_pfn * r_pfn).astype(np.float32)

    # Verification assertions
    assert len(stacked_probs) == n_test, f"Mismatch: expected {n_test}, got {len(stacked_probs)}"
    assert not np.isnan(stacked_probs).any(), "Submission contains NaNs!"
    assert not np.isinf(stacked_probs).any(), "Submission contains Infs!"
    assert (stacked_probs >= 0.0).all() and (stacked_probs <= 1.0).all(), "Predictions out of bounds!"

    # 7. Create Submission DataFrame
    sub_df = pd.DataFrame({
        "id": test_ids,
        "Will_Buy_EV": stacked_probs,
    })

    # Save to both locations
    submission_path = BASE_DIR / "submission_level2_stack.csv"
    submission_results_path = RESULTS_DIR / "submission_level2_stack.csv"

    sub_df.to_csv(submission_path, index=False)
    sub_df.to_csv(submission_results_path, index=False)

    logger.info(f"\nSuccessfully generated submission file:")
    logger.info(f"  Primary: {submission_path} ({submission_path.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"  Backup:  {submission_results_path}")
    logger.info(f"  Shape:   {sub_df.shape}")
    logger.info(f"\nSubmission Head:\n{sub_df.head(10)}")
    logger.info(f"\nDistribution Summary:\n{sub_df['Will_Buy_EV'].describe()}")

    return submission_path


def submit_to_kaggle(submission_file: Path) -> None:
    """Submits the generated submission file to Kaggle competition via CLI."""
    message = "Level-2 4-Model Rank Ensemble (LGB: 0.224, XGB: 0.216, Cat: 0.046, TabPFN: 0.514, OOF AUC: 0.90595)"
    logger.info(f"\nSubmitting {submission_file.name} to Kaggle competition '{COMPETITION_ID}'...")

    cmd = [
        "uv",
        "run",
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION_ID,
        "-f",
        str(submission_file),
        "-m",
        message,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Kaggle CLI Output:\n{res.stdout}")
    if res.stderr:
        logger.warning(f"Kaggle CLI Stderr:\n{res.stderr}")

    # Wait 8 seconds for evaluation
    time.sleep(8)

    # Check status
    logger.info("Checking submission status on Kaggle leaderboard...")
    cmd_status = [
        "uv",
        "run",
        "kaggle",
        "competitions",
        "submissions",
        "-c",
        COMPETITION_ID,
    ]
    res_status = subprocess.run(cmd_status, capture_output=True, text=True)
    logger.info(f"Recent Submissions Table:\n{res_status.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Generate and submit Level-2 stacking submission")
    parser.add_argument("--no-submit", action="store_true", help="Only generate submission without submitting")
    args = parser.parse_args()

    sub_file = build_stacked_submission()
    if not args.no_submit:
        submit_to_kaggle(sub_file)


if __name__ == "__main__":
    main()
