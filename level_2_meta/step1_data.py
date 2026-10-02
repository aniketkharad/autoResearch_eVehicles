"""step1_data.py - Data loading, schema validation, CV splits, and TabPFN discovery.

Workflow:
1. Inspects raw dataset (shape, columns, dtypes, missingness, class balance).
2. Sanitizes features preserving original domain semantics.
3. Generates deterministic Stratified 5-Fold CV splits (seed 42).
4. Searches workspace for candidate TabPFN prediction files, validates shape/range/AUC.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from common import (
    BASE_DIR,
    CACHE_DIR,
    PROJECT_ROOT,
    SEED,
    evaluate_auc,
    setup_logger,
)

logger = setup_logger("step1_data")

TARGET_COL = "Will_Buy_EV"
ID_COL = "Buyer_ID"
DEFAULT_DEV_CSV = PROJECT_ROOT / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
DEFAULT_SYNTH_CSV = PROJECT_ROOT / "train.csv"
DEFAULT_TEST_CSV = PROJECT_ROOT / "test.csv"


def inspect_dataset(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, any]:
    """Inspects dataset shape, dtypes, missing values, class balance, and potential leakage."""
    n_rows, n_cols = df.shape
    cols = list(df.columns)

    # Missing value statistics
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].to_dict()

    # Target stats if present
    target_stats = {}
    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False)
        vc_pct = df[TARGET_COL].value_counts(dropna=False, normalize=True)
        target_stats = {
            "counts": vc.to_dict(),
            "percentages": {k: round(v * 100, 2) for k, v in vc_pct.items()},
        }

    # ID detection
    detected_id = None
    for col in cols:
        if col.lower() in ["id", "buyer_id"] or (df[col].nunique() == n_rows and df[col].dtype == object):
            detected_id = col
            break

    # Categorical vs Numeric
    feature_cols = [c for c in cols if c not in [TARGET_COL, detected_id]]
    cat_cols = [
        c for c in feature_cols
        if pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    report = (
        f"\n{'='*70}\n"
        f"DATASET INSPECTION: {dataset_name}\n"
        f"{'='*70}\n"
        f"Shape:               {n_rows:,} rows x {n_cols} columns\n"
        f"Detected ID Column:  {detected_id}\n"
        f"Target Column:       {TARGET_COL} (Present: {TARGET_COL in df.columns})\n"
        f"Feature Count:       {len(feature_cols)} ({len(cat_cols)} categorical, {len(num_cols)} numeric)\n"
        f"Categorical Features:{cat_cols}\n"
        f"Numeric Features:    {num_cols}\n"
        f"Missing Values:      {len(null_cols)} columns with NaNs -> {null_cols}\n"
    )
    if target_stats:
        report += (
            f"Class Balance:       {target_stats['counts']}\n"
            f"Class Percentages:   {target_stats['percentages']}%\n"
        )
    report += f"{'='*70}\n"
    logger.info(report)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "id_col": detected_id,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "null_cols": null_cols,
        "target_stats": target_stats,
    }


def load_and_preprocess_dev_data(
    csv_path: Path = DEFAULT_DEV_CSV,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], np.ndarray]:
    """Loads and preprocesses the Stage A development dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Development dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    inspect_dataset(df, dataset_name="Stage A: Development Data")

    # Map target cleanly
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing from {csv_path}")
    target_mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1, "0": 0, "1": 1}
    y = df[TARGET_COL].map(target_mapping).values.astype(np.int64)

    # Drop ID column from features
    id_col = "Buyer_ID" if "Buyer_ID" in df.columns else ("id" if "id" in df.columns else None)
    ids = df[id_col].values if id_col else np.arange(len(df))

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, id_col]]
    cat_cols = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]
    cat_cols = [c for c in cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = df[feature_cols].copy()

    # Downcast numerics to float32
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32)

    # Convert categoricals to pandas 'category' dtype
    for col in cat_cols:
        X[col] = X[col].astype("category")

    logger.info(f"Loaded {len(X)} development rows with {X.shape[1]} features.")
    return X, y, cat_cols, num_cols, ids


def get_or_create_cv_splits(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> np.ndarray:
    """Generates or loads deterministic Stratified K-Fold split assignments (fold index per row)."""
    if cache_path is None:
        cache_path = CACHE_DIR / f"splits_dev_k{n_splits}.npy"

    if cache_path.exists() and not force:
        splits = np.load(cache_path)
        if len(splits) == len(y):
            logger.info(f"Loaded existing CV splits from {cache_path} ({len(splits)} rows, {n_splits} folds)")
            return splits

    logger.info(f"Generating new Stratified {n_splits}-Fold splits (seed={seed})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = np.zeros(len(y), dtype=np.int8)

    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        splits[val_idx] = fold

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, splits)
    logger.info(f"Saved deterministic CV splits to {cache_path}")
    return splits


def scan_for_tabpfn_candidates(
    search_root: Path = PROJECT_ROOT,
    target_length: Optional[int] = None,
    y_true: Optional[np.ndarray] = None,
    explicit_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Scans workspace for candidate TabPFN prediction files excluding oof_cache/ and level_2_meta/."""
    logger.info("\n" + "=" * 70)
    logger.info("TABPFN CANDIDATE PREDICTION DISCOVERY")
    logger.info("=" * 70)

    if explicit_path and Path(explicit_path).exists():
        logger.info(f"Using explicitly specified TabPFN prediction path: {explicit_path}")
        return _validate_and_normalize_prediction(Path(explicit_path), target_length, y_true)

    # Search workspace
    candidate_exts = {".npy", ".npz", ".csv", ".parquet"}
    excluded_dirs = {"oof_cache", "level_2_meta", ".git", ".venv", "__pycache__"}

    found_files: List[Path] = []
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in candidate_exts:
                if any(kw in f.lower() for kw in ["tabpfn", "pfn", "oof", "pred", "subs"]):
                    found_files.append(p)

    logger.info(f"Workspace scan located {len(found_files)} candidate prediction files:")
    matching_candidate: Optional[Path] = None

    for p in sorted(found_files, key=lambda x: str(x)):
        mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        shape_str = "unknown"
        try:
            if p.suffix == ".npy":
                arr = np.load(p, allow_pickle=True)
                shape_str = str(arr.shape)
                if target_length and arr.shape[0] == target_length:
                    matching_candidate = p
            elif p.suffix == ".csv":
                df_head = pd.read_csv(p, nrows=2)
                # Count lines quickly
                with open(p, "rb") as f_in:
                    lines = sum(1 for _ in f_in) - 1
                shape_str = f"({lines}, {len(df_head.columns)})"
                if target_length and lines == target_length:
                    matching_candidate = p
        except Exception as e:
            shape_str = f"Error reading: {e}"

        logger.info(f"  - {p.relative_to(PROJECT_ROOT)} | Shape: {shape_str} | Modified: {mtime}")

    if matching_candidate:
        logger.info(f"Selected candidate matching target length {target_length}: {matching_candidate}")
        return _validate_and_normalize_prediction(matching_candidate, target_length, y_true)

    # Check local cache fallback
    default_cached = CACHE_DIR / "tabpfn_oof.npy"
    if default_cached.exists():
        logger.info(f"Found existing cached TabPFN OOF predictions at: {default_cached}")
        return _validate_and_normalize_prediction(default_cached, target_length, y_true)

    logger.warning(
        f"No TabPFN prediction file found matching target length {target_length}. "
        f"Awaiting explicit path or generation."
    )
    return None


def _validate_and_normalize_prediction(
    file_path: Path,
    target_length: Optional[int] = None,
    y_true: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Validates row length, shapes, probability ranges, and ROC AUC alignment."""
    if file_path.suffix == ".npy":
        arr = np.load(file_path, allow_pickle=True)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        # Check target col or last col
        if TARGET_COL in df.columns:
            arr = df[TARGET_COL].values
        elif "prediction" in df.columns:
            arr = df["prediction"].values
        elif df.shape[1] == 2 and ("id" in df.columns[0].lower() or "buyer" in df.columns[0].lower()):
            arr = df.iloc[:, 1].values
        else:
            arr = df.iloc[:, -1].values
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Handle (N, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        logger.info("Candidate predictions have shape (N, 2); extracting positive-class column (index 1).")
        arr = arr[:, 1]
    elif arr.ndim != 1:
        arr = arr.ravel()

    # Convert to float32
    preds = arr.astype(np.float32)

    # Check bounds
    p_min, p_max = float(np.min(preds)), float(np.max(preds))
    if p_min < 0.0 or p_max > 1.0:
        logger.warning(
            f"PROMINENT WARNING: Candidate predictions outside [0, 1] range: [{p_min:.4f}, {p_max:.4f}]. "
            f"Interpreting values as logits and applying sigmoid transformation."
        )
        preds = expit(preds.astype(np.float64)).astype(np.float32)

    if target_length and len(preds) != target_length:
        raise ValueError(f"Candidate row count {len(preds)} does not match target length {target_length}!")

    if y_true is not None and len(y_true) == len(preds):
        auc = evaluate_auc(y_true, preds)
        logger.info(f"TabPFN candidate validation ROC AUC against true labels: {auc:.5f}")
        if auc < 0.50:
            logger.warning(f"Inverted ranking detected (AUC={auc:.4f} < 0.50). Inverting predictions (1 - p).")
            preds = 1.0 - preds
            auc = evaluate_auc(y_true, preds)
            logger.info(f"Corrected validation ROC AUC: {auc:.5f}")

    return preds


def main():
    parser = argparse.ArgumentParser(description="Step 1: Inspect data and CV splits")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DEV_CSV))
    parser.add_argument("--tabpfn-oof", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data(Path(args.data))
    splits = get_or_create_cv_splits(y, force=args.force)

    tabpfn_preds = scan_for_tabpfn_candidates(
        target_length=len(y),
        y_true=y,
        explicit_path=Path(args.tabpfn_oof) if args.tabpfn_oof else None,
    )
    if tabpfn_preds is not None:
        save_path = CACHE_DIR / "tabpfn_oof.npy"
        np.save(save_path, tabpfn_preds)
        logger.info(f"Saved normalized TabPFN OOF predictions to {save_path}")


if __name__ == "__main__":
    main()
