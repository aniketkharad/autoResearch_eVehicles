"""train.py - THE ONLY FILE MODIFIED DURING ITERATIONS.

Autonomous research agent sandbox:
- Feature engineering, architecture selection, hyperparameter tuning.
- Memory constraints: 8 CPU cores max, 8GB RAM max, float32/int32 downcasting, gc.collect().
- Compute budget: 120 seconds max per run.
- Appends every attempt (keep or regression) to results.tsv.
- Prints metric cleanly to stdout: ROC_AUC: <score>
"""

from __future__ import annotations

import datetime
import gc
import os
from pathlib import Path
import subprocess
import time
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from xgboost import XGBClassifier

from prepare import evaluate_predictions, load_splits, load_train_data

# ---------------------------------------------------------------------------
# Experiment Metadata & Configuration
# ---------------------------------------------------------------------------
TIME_BUDGET = 120
N_JOBS = 8

MODEL_ARCHITECTURE = "ensemble_asymmetric_depth5_xgb_lgb_netgreen"
EXPERIMENT_DESCRIPTION = "Iteration 14: Asymmetric ensemble - Depth-5 ultra-high-res XGBoost (max_bin=4096, lr=0.092, eval_metric=logloss, net_green_subsidy + sub_x_inc, w=0.62) + LightGBM (leaves=36, lr=0.088, max_bin=1024, w=0.38)"

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results.tsv"
HISTORY_FILE = BASE_DIR / ".results_history.tsv"
HEADER = ["timestamp", "commit_hash", "model_architecture", "val_roc_auc", "status", "description"]


def get_current_commit() -> str:
    """Gets the current short git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "initial"


def sync_and_read_history() -> Tuple[pd.DataFrame, float]:
    """Syncs results.tsv and .results_history.tsv, returning full history and historical best score."""
    records = []

    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        try:
            df_hist = pd.read_csv(HISTORY_FILE, sep="\t")
            records.append(df_hist)
        except Exception:
            pass

    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        try:
            df_res = pd.read_csv(RESULTS_FILE, sep="\t")
            records.append(df_res)
        except Exception:
            pass

    if records:
        combined = pd.concat(records, ignore_index=True).drop_duplicates(
            subset=["timestamp", "commit_hash", "model_architecture", "val_roc_auc", "description"]
        )
    else:
        combined = pd.DataFrame(columns=HEADER)

    for col in HEADER:
        if col not in combined.columns:
            combined[col] = None

    combined.to_csv(HISTORY_FILE, sep="\t", index=False)
    combined.to_csv(RESULTS_FILE, sep="\t", index=False)

    valid_scores = combined[(combined["val_roc_auc"].notna()) & (combined["status"] == "keep")]["val_roc_auc"]
    historical_best = float(valid_scores.max()) if len(valid_scores) > 0 else 0.0

    return combined, historical_best


def log_experiment(commit_hash: str, model_arch: str, score: float, status: str, description: str) -> None:
    """Appends experiment result to results.tsv and .results_history.tsv."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"{timestamp}\t{commit_hash}\t{model_arch}\t{score:.6f}\t{status}\t{description}\n"

    for path in (RESULTS_FILE, HISTORY_FILE):
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\t".join(HEADER) + "\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(row)


# ---------------------------------------------------------------------------
# Feature Engineering & Preprocessing
# ---------------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Memory-optimized preprocessing with group aggregations and EV domain features."""
    X = df.copy()

    # EV Domain features
    total_charging = X["Charging_Stations_Near_Home"] + X["Charging_Stations_Near_Work"]
    X["charging_stations_total"] = total_charging.astype(np.int32)
    X["charging_home_work_ratio"] = (X["Charging_Stations_Near_Home"] / (X["Charging_Stations_Near_Work"] + 1.0)).astype(np.float32)
    X["commute_per_station"] = (X["Daily_Commute_km"] / (total_charging + 1.0)).astype(np.float32)
    X["income_per_car"] = (X["Annual_Income_USD"] / (X["Number_of_Cars_Owned"] + 1.0)).astype(np.float32)

    # Ordinal and numeric transformations
    anxiety_map = {"Low": 1, "Medium": 2, "High": 3}
    home_charge_map = {"No": 0, "Yes": 1}
    subsidy_map = {"No": 0, "Yes": 1}

    anxiety_num = X["Range_Anxiety_Level"].map(anxiety_map).fillna(2).astype(np.int32)
    home_charge_num = X["Home_Charging_Possible"].map(home_charge_map).fillna(0).astype(np.int32)
    subsidy_num = X["Subsidy_Available"].map(subsidy_map).fillna(0).astype(np.int32)

    X["unmitigated_range_anxiety"] = (anxiety_num * (1 - home_charge_num)).astype(np.int32)
    X["green_subsidy_synergy"] = (X["Environmental_Concern_Level"] * subsidy_num).astype(np.float32)

    # Group aggregations (income & commute deviations relative to categorical cohorts)
    for grp in ["City_Type", "Current_Car_Type"]:
        grp_income = X.groupby(grp, observed=False)["Annual_Income_USD"].transform("mean")
        X[f"income_diff_{grp}"] = (X["Annual_Income_USD"] - grp_income).astype(np.float32)
        grp_commute = X.groupby(grp, observed=False)["Daily_Commute_km"].transform("mean")
        X[f"commute_diff_{grp}"] = (X["Daily_Commute_km"] - grp_commute).astype(np.float32)

    # Compute asymmetric interaction features specifically for XGBoost
    env_num = X["Environmental_Concern_Level"].astype(np.float32)
    anx_f = anxiety_num.astype(np.float32)
    sub_f = subsidy_num.astype(np.float32)
    inc_f = X["Annual_Income_USD"].astype(np.float32)

    extra_xgb = pd.DataFrame({
        "net_green_subsidy": (sub_f * (env_num - anx_f)).astype(np.float32),
        "sub_x_inc": (sub_f * (inc_f / 10000.0)).astype(np.float32),
    })

    # Categorical columns conversion
    cat_cols = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Downcast all remaining numeric columns
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype(np.float32)
    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype(np.int32)

    return X, extra_xgb


# ---------------------------------------------------------------------------
# Training & Cross-Validation Loop
# ---------------------------------------------------------------------------
def run_training() -> float:
    start_time = time.time()
    print("=" * 60)
    print(f"Starting experiment: {MODEL_ARCHITECTURE}")
    print(f"Description: {EXPERIMENT_DESCRIPTION}")
    print(f"Time Budget: {TIME_BUDGET}s | CPU Cores: {N_JOBS}")
    print("=" * 60)

    # 1. Load data and splits
    X_raw, y = load_train_data()
    splits = load_splits()
    X, extra_xgb = preprocess_features(X_raw)
    del X_raw
    gc.collect()

    oof_preds = np.zeros(len(y), dtype=np.float32)
    freq_cols = ["Annual_Income_USD", "Daily_Commute_km", "Age"]

    # 2. Iterate through 5 deterministic folds
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_start = time.time()
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()

        # Strict in-fold frequency encoding
        for col in freq_cols:
            freq_map = X_train[col].value_counts(normalize=True)
            X_train[f"freq_{col}"] = X_train[col].map(freq_map).fillna(0).astype(np.float32)
            X_val[f"freq_{col}"] = X_val[col].map(freq_map).fillna(0).astype(np.float32)

        # Model A: LightGBM on base features (leaves=36, max_depth=6, lr=0.088, n_est=500, max_bin=1024)
        model_lgb = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.088,
            num_leaves=36,
            max_depth=6,
            min_child_samples=30,
            max_bin=1024,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=101 + fold,
            n_jobs=N_JOBS,
            verbose=-1,
        )
        model_lgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds_lgb = model_lgb.predict_proba(X_val)[:, 1].astype(np.float32)
        del model_lgb
        gc.collect()

        # Add asymmetric features for Model B (XGBoost)
        for c in extra_xgb.columns:
            X_train[c] = extra_xgb[c].iloc[train_idx].values
            X_val[c] = extra_xgb[c].iloc[val_idx].values

        # Model B: XGBoost on enriched features (depth=5, lr=0.092, n_estimators=450, max_bin=4096, eval_metric=logloss)
        model_xgb = XGBClassifier(
            n_estimators=450,
            learning_rate=0.092,
            max_depth=5,
            max_bin=4096,
            tree_method="hist",
            enable_categorical=True,
            n_jobs=N_JOBS,
            random_state=101 + fold,
            eval_metric="logloss",
            early_stopping_rounds=30,
        )
        model_xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds_xgb = model_xgb.predict_proba(X_val)[:, 1].astype(np.float32)

        # Percentile Rank-Normalized Blending (0.38 LGBM + 0.62 XGBoost)
        rank_lgb = rankdata(preds_lgb) / len(preds_lgb)
        rank_xgb = rankdata(preds_xgb) / len(preds_xgb)
        fold_probs = (0.38 * rank_lgb + 0.62 * rank_xgb).astype(np.float32)
        oof_preds[val_idx] = fold_probs

        fold_score = evaluate_predictions(y_val, fold_probs)
        fold_elapsed = time.time() - fold_start
        print(f"Fold {fold} - val_roc_auc: {fold_score:.6f} ({fold_elapsed:.2f}s)")

        del model_xgb, X_train, y_train, X_val, y_val
        gc.collect()

        elapsed_so_far = time.time() - start_time
        if elapsed_so_far > TIME_BUDGET:
            raise TimeoutError(f"Execution exceeded strict time budget: {elapsed_so_far:.2f}s > {TIME_BUDGET}s")

    # 3. Overall out-of-fold evaluation
    total_elapsed = time.time() - start_time
    val_roc_auc = evaluate_predictions(y, oof_preds)

    if total_elapsed > TIME_BUDGET:
        raise TimeoutError(f"Total run exceeded strict time budget: {total_elapsed:.2f}s > {TIME_BUDGET}s")

    # 4. Check historical best and log result
    commit_hash = get_current_commit()
    _, historical_best = sync_and_read_history()

    is_improvement = val_roc_auc > historical_best
    status = "keep" if is_improvement else "regression"

    log_experiment(
        commit_hash=commit_hash,
        model_arch=MODEL_ARCHITECTURE,
        score=val_roc_auc,
        status=status,
        description=EXPERIMENT_DESCRIPTION,
    )

    # 5. Clean output summary
    print("-" * 60)
    print(f"model:            {MODEL_ARCHITECTURE}")
    print(f"val_roc_auc:      {val_roc_auc:.6f}")
    print(f"historical_best:  {historical_best:.6f}")
    print(f"status:           {status}")
    print(f"total_seconds:    {total_elapsed:.2f}s / {TIME_BUDGET}s")
    print("-" * 60)
    print(f"ROC_AUC: {val_roc_auc:.6f}")

    return val_roc_auc


if __name__ == "__main__":
    run_training()
