"""stack_large.py - Level-2 Rank Stacking and Weight Optimization on 668k dataset.

Loads OOF predictions from LightGBM, XGBoost, CatBoost, and TabPFN-3.5.
Computes Spearman diversity matrix.
Optimizes non-negative weights via multi-start SLSQP and Nelder-Mead.
Saves weights to level_2_meta/results_large/weights.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stack_large")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "level_2_meta"
CACHE_LARGE_DIR = BASE_DIR / "cache_large"
RESULTS_LARGE_DIR = BASE_DIR / "results_large"
TRAIN_CSV = PROJECT_ROOT / "train.csv"
TARGET_COL = "Will_Buy_EV"

MODELS = ["LightGBM", "XGBoost", "CatBoost", "TabPFN"]
FILE_KEYS = ["lgb", "xgb", "cat", "tabpfn"]


def rank_transform(preds: np.ndarray) -> np.ndarray:
    """Transforms predictions to percentile ranks in [0, 1]."""
    return (rankdata(preds) / len(preds)).astype(np.float32)


def load_all_oofs() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Loads all 4 OOF prediction vectors and true labels."""
    logger.info("Loading ground truth labels from train.csv...")
    df_train = pd.read_csv(TRAIN_CSV, usecols=[TARGET_COL])
    target_mapping = {"No": 0, "Yes": 1, 0: 0, 1: 1}
    y = df_train[TARGET_COL].map(target_mapping).values.astype(np.int64)

    oofs: Dict[str, np.ndarray] = {}
    for name, key in zip(MODELS, FILE_KEYS):
        path = CACHE_LARGE_DIR / f"{key}_oof.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing OOF prediction file: {path}. Run train_models_large.py first!")
        arr = np.load(path)
        assert len(arr) == len(y), f"Length mismatch for {name}: {len(arr)} vs {len(y)}"
        assert not np.isnan(arr).any(), f"NaN values found in {name} OOF predictions!"
        oofs[name] = arr

    logger.info(f"Loaded {len(oofs)} OOF prediction vectors for {len(y):,} samples.")
    return oofs, y


def compute_diversity_analysis(oofs: Dict[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """Computes Spearman rank correlation matrix and individual AUCs."""
    model_names = list(oofs.keys())
    n = len(model_names)
    corr_mat = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_mat[i, j] = 1.0
            else:
                rho, _ = spearmanr(oofs[model_names[i]], oofs[model_names[j]])
                corr_mat[i, j] = rho

    corr_df = pd.DataFrame(corr_mat, index=model_names, columns=model_names)

    logger.info("\n" + "=" * 70)
    logger.info("INDIVIDUAL MODEL PERFORMANCE & SPEARMAN CORRELATION MATRIX")
    logger.info("=" * 70)
    for name in model_names:
        auc = roc_auc_score(y, oofs[name])
        logger.info(f"  {name:<12} | Individual OOF ROC AUC: {auc:.6f}")

    logger.info("\nSpearman Rank Correlation Matrix (Diversity Analysis):")
    logger.info(f"\n{corr_df.to_string()}")
    logger.info("=" * 70 + "\n")

    return corr_df


def optimize_stack_weights(
    oofs: Dict[str, np.ndarray],
    y: np.ndarray,
) -> Tuple[Dict[str, float], float, Dict[str, float], float]:
    """Optimizes non-negative stacking weights using SLSQP and Nelder-Mead with multi-start."""
    model_names = list(oofs.keys())
    K = len(model_names)

    # Pre-rank transform all OOFs
    ranked_oofs = np.column_stack([rank_transform(oofs[m]) for m in model_names])

    # Objective function: negative ROC AUC
    def loss_func(theta: np.ndarray) -> float:
        w = softmax(theta)
        blend = np.dot(ranked_oofs, w)
        return -roc_auc_score(y, blend)

    # Initial points: uniform + individual one-hots
    initial_guesses = [
        np.zeros(K),                                # Uniform
        np.array([2.0, 2.0, 1.0, 0.5]),             # Tree biased
        np.array([1.0, 2.0, 0.5, 2.0]),             # XGB + TabPFN biased
        np.array([2.0, 1.0, 0.5, 0.5]),             # LGB biased
    ]
    for i in range(K):
        hot = np.full(K, -2.0)
        hot[i] = 3.0
        initial_guesses.append(hot)

    best_score = -1.0
    best_weights = np.ones(K) / K
    best_method = "uniform"

    logger.info("Running Multi-Start SLSQP & Nelder-Mead Optimization on Softmax Weights...")

    for method in ["SLSQP", "Nelder-Mead"]:
        for idx, g in enumerate(initial_guesses):
            opts = {"maxiter": 300, "disp": False}
            res = minimize(loss_func, g, method=method, options=opts)
            score = -res.fun
            w = softmax(res.x)
            if score > best_score:
                best_score = score
                best_weights = w
                best_method = f"{method} (start {idx})"

    # Calculate uniform baseline score
    uniform_w = np.ones(K) / K
    uniform_blend = np.dot(ranked_oofs, uniform_w)
    uniform_score = roc_auc_score(y, uniform_blend)

    weight_dict = {model_names[i]: round(float(best_weights[i]), 6) for i in range(K)}
    uniform_dict = {model_names[i]: round(1.0 / K, 6) for i in range(K)}

    logger.info(f"\nOptimization converged using {best_method}:")
    logger.info(f"  Uniform Blend ROC AUC:  {uniform_score:.6f}")
    logger.info(f"  Optimal Stack ROC AUC:  {best_score:.6f} (+{best_score - uniform_score:+.6f} over uniform)")
    logger.info(f"  Optimal Weights:")
    for m, w in weight_dict.items():
        logger.info(f"    {m:<12}: {w:.6f} ({w*100:.2f}%)")

    return weight_dict, best_score, uniform_dict, uniform_score


def run_stacking() -> Dict[str, Any]:
    """Master Level-2 stacking execution on the 668k dataset."""
    oofs, y = load_all_oofs()
    corr_df = compute_diversity_analysis(oofs, y)
    best_w, best_auc, uni_w, uni_auc = optimize_stack_weights(oofs, y)

    # Save weights
    weights_path = RESULTS_LARGE_DIR / "weights.json"
    individual_aucs = {m: round(float(roc_auc_score(y, oofs[m])), 6) for m in oofs}

    results = {
        "dataset": "train.csv (668,665 samples)",
        "models": MODELS,
        "individual_aucs": individual_aucs,
        "uniform_weights": uni_w,
        "uniform_auc": round(float(uni_auc), 6),
        "optimal_weights": best_w,
        "stacked_auc": round(float(best_auc), 6),
        "max_individual_auc": max(individual_aucs.values()),
        "lift_over_best_single": round(float(best_auc - max(individual_aucs.values())), 6),
    }

    with open(weights_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nWeights and stacking metrics saved to {weights_path}")

    # Generate Markdown Report
    report_path = RESULTS_LARGE_DIR / "stage_b_report.md"
    with open(report_path, "w") as f:
        f.write("# Stage B: Level-2 Stacking Report (668k Dataset)\n\n")
        f.write(f"- **Dataset Scale**: 668,665 rows x 25 features | 5-Fold Stratified CV\n")
        f.write(f"- **Final Stacked OOF ROC AUC**: **`{best_auc:.6f}`**\n")
        f.write(f"- **Best Single Model**: **`{results['max_individual_auc']:.6f}`**\n")
        f.write(f"- **Incremental Lift**: **`+{results['lift_over_best_single']:.6f}`**\n\n")
        f.write("## Individual Model 5-Fold OOF AUCs\n\n")
        f.write("| Model | 5-Fold OOF ROC AUC | Optimal Blend Weight |\n")
        f.write("| :--- | :---: | :---: |\n")
        for m in MODELS:
            f.write(f"| {m} | {individual_aucs[m]:.6f} | {best_w[m]:.4f} ({best_w[m]*100:.1f}%) |\n")
        # Format markdown table manually without tabulate dependency
        f.write("| | " + " | ".join(MODELS) + " |\n")
        f.write("| :--- | " + " | ".join([":---:"] * len(MODELS)) + " |\n")
        for m1 in MODELS:
            row_str = " | ".join([f"{corr_df.loc[m1, m2]:.4f}" for m2 in MODELS])
            f.write(f"| **{m1}** | {row_str} |\n")
        f.write("\n")
        f.write("## Optimization Details\n\n")
        f.write("- **Method**: Multi-Start SLSQP & Nelder-Mead on Softmax Weight Parameterization\n")
        f.write(f"- **Uniform Blend Score**: `{uni_auc:.6f}`\n")
        f.write(f"- **Optimized Stack Score**: `{best_auc:.6f}`\n")

    logger.info(f"Report saved to {report_path}")
    return results


if __name__ == "__main__":
    run_stacking()
