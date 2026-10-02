"""common.py - Shared utilities for Level-2 Stacking Ensemble pipeline.

Provides:
- Device & environment hardware detection
- Deterministic seeding
- Evaluation metrics (ROC AUC)
- Rank transformation
- Constrained blend weight optimization (SLSQP / Nelder-Mead)
- Logging & serialization utilities
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
OPTUNA_DIR = BASE_DIR / "optuna"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures structured logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def detect_hardware_and_env() -> Dict[str, Any]:
    """Detects CPU count, GPU/accelerator availability, and library versions."""
    import multiprocessing
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    import optuna
    import sklearn
    import scipy

    cpu_count = multiprocessing.cpu_count()
    cuda_available = False
    mps_available = False
    device_selected = "cpu"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if cuda_available:
            device_selected = "cuda"
        elif mps_available:
            device_selected = "mps"
    except ImportError:
        pass

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": cpu_count,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "device_selected": device_selected,
        "versions": {
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "catboost": cb.__version__,
            "optuna": optuna.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    return info


def print_hardware_summary(logger: Optional[logging.Logger] = None) -> None:
    """Prints detected hardware environment details."""
    info = detect_hardware_and_env()
    msg = (
        f"\n{'='*70}\n"
        f"HARDWARE & ENVIRONMENT SUMMARY\n"
        f"Platform:        {info['platform']} ({info['machine']})\n"
        f"CPU Logical:     {info['cpu_count']} cores\n"
        f"CUDA Available:  {info['cuda_available']}\n"
        f"MPS Available:   {info['mps_available']} (Apple Silicon Metal)\n"
        f"Selected Device: {info['device_selected'].upper()}\n"
        f"Libraries:\n"
        f"  - LightGBM:    {info['versions']['lightgbm']}\n"
        f"  - XGBoost:     {info['versions']['xgboost']}\n"
        f"  - CatBoost:    {info['versions']['catboost']}\n"
        f"  - Optuna:      {info['versions']['optuna']}\n"
        f"  - Scikit-Learn:{info['versions']['sklearn']}\n"
        f"  - SciPy:       {info['versions']['scipy']}\n"
        f"{'='*70}\n"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)


def evaluate_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Computes ROC AUC between ground truth binary labels and predicted probabilities."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred_proba, dtype=np.float64)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(f"1D arrays required. Got y_true shape {y_true_arr.shape}, y_pred shape {y_pred_arr.shape}")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"Length mismatch: y_true {len(y_true_arr)} vs y_pred {len(y_pred_arr)}")
    if np.isnan(y_pred_arr).any() or np.isinf(y_pred_arr).any():
        raise ValueError("Prediction array contains NaN or Inf values")
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("Target contains only one unique class")

    return float(roc_auc_score(y_true_arr, y_pred_arr))


def rank_transform(p: np.ndarray) -> np.ndarray:
    """Rank-transforms a 1D prediction array: R_i = rankdata(P_i) / N."""
    arr = np.asarray(p, dtype=np.float64)
    n = len(arr)
    return (rankdata(arr, method="average") / n).astype(np.float32)


def compute_spearman_matrix(P: np.ndarray, model_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes pairwise Spearman rank correlation matrix and p-values."""
    k = len(model_names)
    corr_mat = np.zeros((k, k), dtype=np.float64)
    pval_mat = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            res = spearmanr(P[:, i], P[:, j])
            corr_mat[i, j] = res.statistic
            pval_mat[i, j] = res.pvalue

    df_corr = pd.DataFrame(corr_mat, index=model_names, columns=model_names)
    df_pval = pd.DataFrame(pval_mat, index=model_names, columns=model_names)
    return df_corr, df_pval


def optimize_blend_weights(
    R: np.ndarray,
    y_true: np.ndarray,
    model_names: List[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Optimizes non-negative blend weights w >= 0, sum(w) = 1 to maximize ROC AUC.

    Runs multiple distinct initializations using SLSQP and Nelder-Mead with softmax,
    and returns the best global solution found.
    """
    n_samples, n_models = R.shape
    y_true_arr = np.asarray(y_true, dtype=np.int64)

    # Candidate initializations
    starts = [
        ("Equal Weights", np.ones(n_models) / n_models),
    ]
    for i, name in enumerate(model_names):
        w_init = np.full(n_models, 0.1 / max(1, n_models - 1))
        w_init[i] = 0.9
        w_init = w_init / np.sum(w_init)
        starts.append((f"Biased towards {name}", w_init))

    best_auc = -1.0
    best_weights = None
    best_init_name = None
    optimization_history = []

    # SLSQP Objective: minimize negative ROC AUC
    def loss_slsqp(weights: np.ndarray) -> float:
        w = np.clip(weights, 0.0, None)
        s = np.sum(w)
        if s > 0:
            w = w / s
        pred = R @ w
        return -evaluate_auc(y_true_arr, pred)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    header_msg = f"\n{'='*70}\nLEVEL-2 STACKING: MULTI-START WEIGHT OPTIMIZATION\n{'='*70}"
    if logger:
        logger.info(header_msg)
    else:
        print(header_msg)

    for init_name, w_start in starts:
        # Run SLSQP
        res = minimize(
            loss_slsqp,
            w_start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-7, "disp": False},
        )

        w_opt = np.clip(res.x, 0.0, None)
        if np.sum(w_opt) > 0:
            w_opt = w_opt / np.sum(w_opt)
        final_auc = evaluate_auc(y_true_arr, R @ w_opt)

        optimization_history.append({
            "init_name": init_name,
            "solver": "SLSQP",
            "start_weights": [round(float(x), 4) for x in w_start],
            "final_weights": [round(float(x), 4) for x in w_opt],
            "final_auc": float(final_auc),
            "converged": bool(res.success),
            "iterations": int(res.nit),
            "message": str(res.message),
        })

        run_msg = (
            f"Start: {init_name:<25} | Converged: {res.success!s:<5} | "
            f"Iters: {res.nit:3d} | AUC: {final_auc:.6f}\n"
            f"  Weights: {dict(zip(model_names, [round(float(w), 4) for w in w_opt]))}"
        )
        if logger:
            logger.info(run_msg)
        else:
            print(run_msg)

        if final_auc > best_auc:
            best_auc = final_auc
            best_weights = w_opt
            best_init_name = init_name

    # Cross-check using Nelder-Mead with softmax parameterization
    def loss_nelder_mead(theta: np.ndarray) -> float:
        e = np.exp(theta - np.max(theta))
        w = e / np.sum(e)
        return -evaluate_auc(y_true_arr, R @ w)

    # Initialize from current best weights
    theta_init = np.log(np.maximum(best_weights, 1e-4))
    res_nm = minimize(
        loss_nelder_mead,
        theta_init,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-6},
    )
    e = np.exp(res_nm.x - np.max(res_nm.x))
    w_nm = e / np.sum(e)
    auc_nm = evaluate_auc(y_true_arr, R @ w_nm)

    nm_msg = (
        f"Nelder-Mead Refinement:        | Converged: {res_nm.success!s:<5} | "
        f"Iters: {res_nm.nit:3d} | AUC: {auc_nm:.6f}\n"
        f"  Weights: {dict(zip(model_names, [round(float(w), 4) for w in w_nm]))}"
    )
    if logger:
        logger.info(nm_msg)
    else:
        print(nm_msg)

    if auc_nm > best_auc:
        best_auc = auc_nm
        best_weights = w_nm
        best_init_name = "Nelder-Mead Refinement"

    result = {
        "best_init": best_init_name,
        "best_auc": float(best_auc),
        "weights": {model: float(w) for model, w in zip(model_names, best_weights)},
        "weights_array": [float(w) for w in best_weights],
        "history": optimization_history,
    }
    return result


def save_json(data: Any, path: Path) -> None:
    """Safely saves object as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    """Loads object from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
