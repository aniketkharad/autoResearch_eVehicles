"""submit_large.py - Blend test predictions, create submission, and submit to Kaggle.

Loads test predictions from LightGBM, XGBoost, CatBoost, and TabPFN-3.5.
Applies optimal rank weights from weights.json.
Writes submission_large_stack.csv and submits to Kaggle playground-series-s6e9.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("submit_large")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "level_2_meta"
CACHE_LARGE_DIR = BASE_DIR / "cache_large"
RESULTS_LARGE_DIR = BASE_DIR / "results_large"
TEST_CSV = PROJECT_ROOT / "test.csv"
COMPETITION_ID = "playground-series-s6e9"

MODELS = ["LightGBM", "XGBoost", "CatBoost", "TabPFN"]
FILE_KEYS = ["lgb", "xgb", "cat", "tabpfn"]


def rank_transform(preds: np.ndarray) -> np.ndarray:
    """Transforms predictions to percentile ranks in [0, 1]."""
    return (rankdata(preds) / len(preds)).astype(np.float32)


def generate_submission_file() -> Path:
    """Generates the blended submission file."""
    logger.info(f"Loading test IDs from {TEST_CSV}...")
    df_test = pd.read_csv(TEST_CSV, usecols=["id"])
    test_ids = df_test["id"].values
    n_test = len(test_ids)

    # Load test predictions
    test_preds: Dict[str, np.ndarray] = {}
    for name, key in zip(MODELS, FILE_KEYS):
        path = CACHE_LARGE_DIR / f"{key}_test.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing test predictions file: {path}. Run train_models_large.py first!")
        arr = np.load(path)
        assert len(arr) == n_test, f"Length mismatch for {name}: {len(arr)} vs {n_test}"
        assert not np.isnan(arr).any(), f"NaNs found in {name} test predictions!"
        test_preds[name] = arr
        logger.info(
            f"Loaded {name:<12} test predictions | Range: [{np.min(arr):.5f}, {np.max(arr):.5f}] | Mean: {np.mean(arr):.5f}"
        )

    # Load weights
    weights_path = RESULTS_LARGE_DIR / "weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights at {weights_path}. Run stack_large.py first!")

    with open(weights_path, "r") as f:
        weights_info = json.load(f)
    weights = weights_info["optimal_weights"]
    stacked_auc = weights_info["stacked_auc"]

    logger.info("\nLoaded Level-2 Stacking Weights:")
    for m, w in weights.items():
        logger.info(f"  {m:<12}: {w:.6f} ({w*100:.2f}%)")

    # Compute Rank Ensemble
    logger.info("Computing percentile rank blend...")
    blended_probs = np.zeros(n_test, dtype=np.float32)
    for m in MODELS:
        r_m = rank_transform(test_preds[m])
        w_m = weights[m]
        blended_probs += w_m * r_m

    # Verification assertions
    assert len(blended_probs) == n_test
    assert not np.isnan(blended_probs).any()
    assert not np.isinf(blended_probs).any()
    assert (blended_probs >= 0.0).all() and (blended_probs <= 1.0).all()

    sub_df = pd.DataFrame({
        "id": test_ids,
        "Will_Buy_EV": blended_probs,
    })

    out_file = BASE_DIR / "submission_large_stack.csv"
    sub_df.to_csv(out_file, index=False)
    logger.info(f"\nSubmission file created: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
    logger.info(f"Shape: {sub_df.shape}")
    logger.info(f"\nSubmission Head:\n{sub_df.head(10)}")
    logger.info(f"\nDistribution Summary:\n{sub_df['Will_Buy_EV'].describe()}")

    return out_file


def submit_to_kaggle(sub_file: Path) -> None:
    """Submits file to Kaggle competition and retrieves leaderboard score."""
    weights_path = RESULTS_LARGE_DIR / "weights.json"
    desc = "Level-2 4-Model Rank Ensemble on 668k train.csv (LGB, XGB, CAT, TabPFN-3.5)"
    if weights_path.exists():
        try:
            with open(weights_path, "r") as f:
                info = json.load(f)
            desc = (
                f"Level-2 Stack (LGB:{info['optimal_weights']['LightGBM']:.3f}, "
                f"XGB:{info['optimal_weights']['XGBoost']:.3f}, "
                f"CAT:{info['optimal_weights']['CatBoost']:.3f}, "
                f"PFN:{info['optimal_weights']['TabPFN']:.3f}, OOF:{info['stacked_auc']:.5f})"
            )
        except Exception:
            pass

    logger.info(f"\nSubmitting {sub_file.name} to Kaggle competition '{COMPETITION_ID}'...")
    logger.info(f"Message: {desc}")

    cmd = [
        "uv",
        "run",
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION_ID,
        "-f",
        str(sub_file),
        "-m",
        desc,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(f"Kaggle CLI Output:\n{res.stdout}")
    if res.stderr:
        logger.warning(f"Kaggle CLI Stderr:\n{res.stderr}")

    time.sleep(10)

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
    parser = argparse.ArgumentParser(description="Submit Level-2 large stack to Kaggle")
    parser.add_argument("--no-submit", action="store_true", help="Only build file without submitting")
    args = parser.parse_args()

    sub_path = generate_submission_file()
    if not args.no_submit:
        submit_to_kaggle(sub_path)


if __name__ == "__main__":
    main()
