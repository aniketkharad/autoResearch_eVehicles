"""prepare.py - LOCKED & IMMUTABLE.

Autonomous research agents MUST NOT modify this file.
This file defines:
1. Invariant dataset constants and fixed evaluation metric (ROC AUC).
2. Deterministic stratified 5-fold cross-validation splits (saved to splits.npy).
3. Immutable ground-truth evaluation function: evaluate_predictions(y_true, y_pred_proba).
4. Strict isolation locking test.csv out of reach during autonomous iterations.

Usage:
    uv run python prepare.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Invariant Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------
TIME_BUDGET: int = 60              # Strict execution budget per run in seconds
N_SPLITS: int = 5                  # Stratified 5-fold cross-validation
RANDOM_STATE: int = 101            # Pinned seed for deterministic splits
TARGET_COL: str = "Will_Buy_EV"    # Target column name
ID_COL: str = "id"                 # Identifier column to drop from features
SPLITS_FILE: str = "splits.npy"    # Disk storage for invariant fold assignments

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv" if (BASE_DIR / "train.csv").exists() else BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "test.csv" if (BASE_DIR / "test.csv").exists() else BASE_DIR / "data" / "test.csv"


# ---------------------------------------------------------------------------
# Immutable Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true: np.ndarray | pd.Series, y_pred_proba: np.ndarray | pd.Series) -> float:
    """Ground truth evaluation harness.

    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    between true binary labels and predicted probabilities for the positive class.

    Parameters:
        y_true: 1D array-like of ground truth binary labels (0 or 1).
        y_pred_proba: 1D array-like of predicted probabilities in [0.0, 1.0].

    Returns:
        float: ROC AUC score in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred_proba, dtype=np.float64)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(f"Inputs must be 1D arrays, got y_true shape {y_true_arr.shape} and y_pred shape {y_pred_arr.shape}")

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"Sample count mismatch: y_true has {len(y_true_arr)}, y_pred has {len(y_pred_arr)}")

    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true must contain both positive and negative classes")

    if np.isnan(y_pred_arr).any() or np.isinf(y_pred_arr).any():
        raise ValueError("y_pred_proba contains NaN or Infinite values")

    return float(roc_auc_score(y_true_arr, y_pred_arr))


# ---------------------------------------------------------------------------
# Data Loading & Isolation
# ---------------------------------------------------------------------------
def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Loads training features and binary target.

    Drops the 'id' column from features.
    Maps target 'Yes' -> 1, 'No' -> 0.

    Returns:
        X (pd.DataFrame): Training feature matrix.
        y (pd.Series): Binary target series.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {TRAIN_PATH}")

    # Map target to binary integer
    target_mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1}
    y = df[TARGET_COL].map(target_mapping).astype(np.int64)

    # Feature columns (exclude target and id)
    feature_cols = [c for c in df.columns if c not in (TARGET_COL, ID_COL)]
    X = df[feature_cols].copy()

    return X, y


def load_test_data() -> None:
    """Strictly locks test.csv out of reach.

    Raises PermissionError if called during autonomous iterations.
    """
    raise PermissionError(
        "LOCKED: test.csv is completely out of reach and must never be loaded, "
        "evaluated, or probed during autonomous research iterations."
    )


# ---------------------------------------------------------------------------
# Deterministic Splits Generation & Loading
# ---------------------------------------------------------------------------
def prepare_splits(force: bool = False) -> np.ndarray:
    """Generates and persists invariant stratified 5-fold CV assignments.

    Saves a 1D int8 array of fold IDs (0 to N_SPLITS-1) to splits.npy.
    """
    splits_path = BASE_DIR / SPLITS_FILE
    if splits_path.exists() and not force:
        return np.load(splits_path)

    print(f"Generating deterministic stratified {N_SPLITS}-fold splits (seed={RANDOM_STATE})...")
    _, y = load_train_data()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_assignments = np.zeros(len(y), dtype=np.int8)

    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_assignments[val_idx] = fold

    np.save(splits_path, fold_assignments)
    print(f"Saved invariant fold assignments to {splits_path} ({len(fold_assignments):,} samples)")
    return fold_assignments


def load_splits() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Loads the deterministic 5-fold cross-validation index pairs.

    Returns:
        List of 5 tuples: (train_indices, val_indices)
    """
    splits_path = BASE_DIR / SPLITS_FILE
    if not splits_path.exists():
        prepare_splits()

    fold_assignments = np.load(splits_path)
    splits = []
    for fold in range(N_SPLITS):
        train_idx = np.where(fold_assignments != fold)[0]
        val_idx = np.where(fold_assignments == fold)[0]
        splits.append((train_idx, val_idx))

    return splits


# ---------------------------------------------------------------------------
# CLI Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("prepare.py - Autonomous Research Harness Initialization")
    print("=" * 60)

    # 1. Verify train dataset
    X, y = load_train_data()
    print(f"Train Dataset: {X.shape[0]:,} rows x {X.shape[1]} features")
    pos_rate = y.mean() * 100
    print(f"Target Distribution: {pos_rate:.2f}% Positive (Yes), {100 - pos_rate:.2f}% Negative (No)")

    # 2. Check test file existence without loading
    if TEST_PATH.exists():
        print(f"Test Dataset locked and isolated: {TEST_PATH.name} present ({TEST_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("Note: test.csv not found in workspace.")

    # 3. Generate / verify splits
    fold_assignments = prepare_splits(force=True)
    splits = load_splits()
    print(f"Cross-Validation: {len(splits)} deterministic stratified folds generated.")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        val_pos = y.iloc[va_idx].mean() * 100
        print(f"  Fold {fold}: Train={len(tr_idx):,} rows, Val={len(va_idx):,} rows, Val Positive Rate={val_pos:.2f}%")

    # 4. Self-test evaluation harness
    mock_true = np.array([0, 0, 1, 1])
    mock_pred = np.array([0.1, 0.3, 0.8, 0.9])
    test_score = evaluate_predictions(mock_true, mock_pred)
    assert test_score == 1.0, f"Expected 1.0, got {test_score}"
    print(f"Evaluation Harness: Self-test passed (ROC AUC calculation verified).")
    print("=" * 60)
    print("Harness successfully initialized and locked.")
