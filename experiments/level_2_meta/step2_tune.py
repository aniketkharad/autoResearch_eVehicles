"""step2_tune.py - Adaptive 3-Phase Optuna Tuning for LightGBM, XGBoost, and CatBoost.

Tuning Strategy:
- Sequential execution (LightGBM -> XGBoost -> CatBoost)
- Optuna n_jobs = 1, fixed seed = 42
- SQLite persistence under level_2_meta/optuna/{model}.db
- Adaptive 3-phase search:
    Phase 1: Broad structural exploration (~8 trials, including hand-picked baseline)
    Phase 2: Multivariate TPE exploitation (~20 trials)
    Phase 3: Local refinement (~12 trials)
- Early stopping (30-40 rounds) with generous iteration limits
- Tie-breaking: favors lower runtime/complexity if AUC within tie_threshold (default: 0.0001)
- Saves best params JSON and trial history CSV under level_2_meta/results/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler

from common import (
    OPTUNA_DIR,
    RESULTS_DIR,
    SEED,
    detect_hardware_and_env,
    evaluate_auc,
    print_hardware_summary,
    save_json,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    get_or_create_cv_splits,
    load_and_preprocess_dev_data,
)

logger = setup_logger("step2_tune")

# Ensure reproducibility and mute verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_lgb_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines search space for LightGBM."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 64, step=4),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 15, 60, step=5),
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0, step=0.05),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0, step=0.05),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
        "max_bin": trial.suggest_categorical("max_bin", [255, 512, 1024]),
    }


def build_xgb_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines search space for XGBoost."""
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512, 1024, 2048, 4096]),
    }


def build_cat_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Defines search space for CatBoost."""
    return {
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
    }


def get_handpicked_baseline(model_name: str) -> Dict[str, Any]:
    """Returns strong hand-picked baseline parameters derived from empirical breakthroughs."""
    if model_name == "lightgbm":
        return {
            "num_leaves": 36,
            "max_depth": 6,
            "min_child_samples": 30,
            "learning_rate": 0.088,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.01,
            "lambda_l2": 0.01,
            "min_gain_to_split": 0.0,
            "max_bin": 1024,
        }
    elif model_name == "xgboost":
        return {
            "max_depth": 5,
            "min_child_weight": 3,
            "learning_rate": 0.092,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "max_bin": 4096,
        }
    elif model_name == "catboost":
        return {
            "depth": 5,
            "l2_leaf_reg": 5.0,
            "learning_rate": 0.085,
            "random_strength": 1.0,
            "bagging_temperature": 0.5,
            "border_count": 254,
        }
    raise ValueError(f"Unknown model {model_name}")


def evaluate_trial_cv(
    model_name: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    cat_cols: List[str],
    early_stopping_rounds: int = 35,
    max_estimators: int = 3000,
    eval_folds: Optional[List[int]] = None,
    n_workers: int = 5,
) -> Tuple[float, int, float]:
    """Evaluates hyperparameters concurrently across cross-validation folds using up to 8 CPU cores."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    folds = eval_folds if eval_folds is not None else list(range(int(np.max(splits) + 1)))

    # Format data for CatBoost if needed
    if model_name == "catboost":
        X_eval = X.copy()
        for c in cat_cols:
            X_eval[c] = X_eval[c].astype(str)
    else:
        X_eval = X

    threads_per_worker = max(1, 8 // max(1, len(folds)))

    def _eval_fold(fold: int) -> Tuple[float, int]:
        tr_idx = np.where(splits != fold)[0]
        va_idx = np.where(splits == fold)[0]

        X_tr, y_tr = X_eval.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_eval.iloc[va_idx], y[va_idx]

        if model_name == "lightgbm":
            clf = lgb.LGBMClassifier(
                n_estimators=max_estimators,
                random_state=SEED + fold,
                n_jobs=threads_per_worker,
                verbose=-1,
                **params,
            )
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            preds = clf.predict_proba(X_va)[:, 1]
            best_iter = clf.best_iteration_

        elif model_name == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=max_estimators,
                tree_method="hist",
                enable_categorical=True,
                eval_metric="logloss",
                early_stopping_rounds=early_stopping_rounds,
                random_state=SEED + fold,
                n_jobs=threads_per_worker,
                **params,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = clf.predict_proba(X_va)[:, 1]
            best_iter = clf.best_iteration

        elif model_name == "catboost":
            clf = cb.CatBoostClassifier(
                iterations=max_estimators,
                loss_function="Logloss",
                eval_metric="AUC",
                cat_features=cat_cols,
                early_stopping_rounds=early_stopping_rounds,
                random_seed=SEED + fold,
                thread_count=threads_per_worker,
                verbose=False,
                **params,
            )
            clf.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            preds = clf.predict_proba(X_va)[:, 1]
            best_iter = clf.get_best_iteration()
        else:
            raise ValueError(model_name)

        auc = evaluate_auc(y_va, preds)
        del clf, X_tr, y_tr, X_va, y_va, preds
        return auc, best_iter

    workers = min(len(folds), n_workers)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_eval_fold, folds))
    else:
        results = [_eval_fold(f) for f in folds]

    fold_aucs = [r[0] for r in results]
    best_iters = [r[1] for r in results]

    mean_auc = float(np.mean(fold_aucs))
    avg_best_iter = int(np.mean(best_iters))
    elapsed = float(time.time() - t0)
    return mean_auc, avg_best_iter, elapsed


def run_optuna_study(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: np.ndarray,
    cat_cols: List[str],
    total_trials: int = 40,
    tie_threshold: float = 0.0001,
    force: bool = False,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Runs sequential 3-phase Optuna tuning study with SQLite persistence."""
    logger.info(f"\n{'='*70}\nSTARTING OPTUNA STUDY: {model_name.upper()}\n{'='*70}")

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    storage_name = f"sqlite:///{OPTUNA_DIR}/{model_name}{'_smoke' if smoke else ''}.db"
    study_name = f"{model_name}_study{'_smoke' if smoke else ''}"

    if force:
        try:
            optuna.delete_study(study_name=study_name, storage=storage_name)
            logger.info(f"Deleted existing study {study_name} (--force requested)")
        except Exception:
            pass

    # Use multivariate TPE sampler
    sampler = TPESampler(multivariate=True, seed=SEED)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"Resuming study '{study_name}': {completed_trials}/{total_trials} trials already completed.")

    if completed_trials >= total_trials:
        logger.info(f"Target trial count ({total_trials}) already reached. Skipping tuning.")
        best_t = study.best_trial
        best_info = {
            "model_name": model_name,
            "best_auc": float(best_t.value),
            "best_params": best_t.params,
            "best_iteration": best_t.user_attrs.get("best_iteration", 50),
            "training_time": best_t.user_attrs.get("elapsed_time", 0.0),
        }
        return best_info

    # Enqueue baseline trial if first time
    if completed_trials == 0:
        baseline_params = get_handpicked_baseline(model_name)
        study.enqueue_trial(baseline_params)
        logger.info(f"Enqueued handpicked baseline trial for {model_name}: {baseline_params}")

    # Partition phases
    p1_limit = max(1, int(total_trials * 0.20))  # ~8 trials
    p2_limit = max(p1_limit + 1, int(total_trials * 0.70))  # ~20 trials

    # Trials to run
    trials_to_run = total_trials - completed_trials
    eval_folds = [0] if smoke else None

    # Track trial records
    trial_records = []
    completed_trials_list = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_so_far_auc = study.best_value if len(completed_trials_list) > 0 else 0.0

    for step in range(trials_to_run):
        current_trial_idx = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) + 1

        if current_trial_idx <= p1_limit:
            current_phase = 1
        elif current_trial_idx <= p2_limit:
            current_phase = 2
        else:
            current_phase = 3

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_so_far_auc

            done_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_p = study.best_trial.params if len(done_trials) > 0 else None
            if model_name == "lightgbm":
                params = build_lgb_search_space(trial)
            elif model_name == "xgboost":
                params = build_xgb_search_space(trial)
            elif model_name == "catboost":
                params = build_cat_search_space(trial)
            else:
                raise ValueError(model_name)

            val_auc, best_iter, elapsed = evaluate_trial_cv(
                model_name=model_name,
                params=params,
                X=X,
                y=y,
                splits=splits,
                cat_cols=cat_cols,
                eval_folds=eval_folds,
            )

            trial.set_user_attr("best_iteration", best_iter)
            trial.set_user_attr("elapsed_time", elapsed)
            trial.set_user_attr("phase", current_phase)

            if val_auc > best_so_far_auc:
                best_so_far_auc = val_auc

            trial_records.append({
                "trial_number": trial.number,
                "phase": current_phase,
                "validation_auc": val_auc,
                "best_iteration": best_iter,
                "elapsed_time": elapsed,
                "best_so_far_auc": best_so_far_auc,
                "parameters": params,
            })

            logger.info(
                f"[{model_name.upper()} Phase {current_phase}] Trial {trial.number:2d}/{total_trials} | "
                f"AUC: {val_auc:.5f} (Best: {best_so_far_auc:.5f}) | "
                f"Iter: {best_iter:3d} | Time: {elapsed:.2f}s"
            )
            return val_auc

        study.optimize(objective, n_trials=1, n_jobs=1)

    # Model Selection with Tie-Breaking
    best_trial = study.best_trial
    best_auc = float(best_trial.value)
    selected_trial = best_trial

    # Check candidates within tie_threshold
    tied_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and (best_auc - t.value) <= tie_threshold
    ]
    if len(tied_trials) > 1:
        # Prefer lowest training time
        tied_trials.sort(key=lambda t: t.user_attrs.get("elapsed_time", 999.0))
        selected_trial = tied_trials[0]
        if selected_trial.number != best_trial.number:
            logger.info(
                f"Tie-breaking applied: Trial {selected_trial.number} selected over Trial {best_trial.number} "
                f"(AUC delta: {best_auc - selected_trial.value:.6f} <= {tie_threshold}, Time: {selected_trial.user_attrs.get('elapsed_time', 0):.2f}s vs {best_trial.user_attrs.get('elapsed_time', 0):.2f}s)"
            )

    baseline_trial = study.trials[0]
    baseline_auc = float(baseline_trial.value)
    selected_auc = float(selected_trial.value)
    improvement = selected_auc - baseline_auc

    logger.info(
        f"\n{'='*70}\n"
        f"{model_name.upper()} TUNING COMPLETED\n"
        f"Baseline AUC (Trial 0): {baseline_auc:.5f}\n"
        f"Best AUC:               {selected_auc:.5f} (Improvement: {improvement:+.5f})\n"
        f"Best Iteration:         {selected_trial.user_attrs.get('best_iteration')}\n"
        f"Elapsed Time:           {selected_trial.user_attrs.get('elapsed_time', 0):.2f}s\n"
        f"Selected Parameters:    {selected_trial.params}\n"
        f"{'='*70}\n"
    )

    # Save best parameters JSON
    best_info = {
        "model_name": model_name,
        "selected_trial": selected_trial.number,
        "baseline_auc": baseline_auc,
        "best_auc": selected_auc,
        "improvement": improvement,
        "best_iteration": selected_trial.user_attrs.get("best_iteration"),
        "training_time": selected_trial.user_attrs.get("elapsed_time"),
        "best_params": selected_trial.params,
    }
    save_path = RESULTS_DIR / f"{model_name}{'_smoke' if smoke else ''}_best_params.json"
    save_json(best_info, save_path)
    logger.info(f"Saved best parameters to {save_path}")

    # Save trial history CSV
    df_trials = study.trials_dataframe()
    history_path = RESULTS_DIR / f"{model_name}{'_smoke' if smoke else ''}_trials.csv"
    df_trials.to_csv(history_path, index=False)
    logger.info(f"Saved complete trial history to {history_path}")

    return best_info


def main():
    parser = argparse.ArgumentParser(description="Step 2: Optuna Tuning")
    parser.add_argument("--trials", type=int, default=40, help="Trials per model")
    parser.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (3 trials, fold 0)")
    parser.add_argument("--force", action="store_true", help="Force retune from scratch")
    args = parser.parse_args()

    print_hardware_summary(logger)

    X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data()
    splits = get_or_create_cv_splits(y)

    total_trials = 3 if args.smoke else args.trials

    for model_name in args.models:
        run_optuna_study(
            model_name=model_name,
            X=X,
            y=y,
            splits=splits,
            cat_cols=cat_cols,
            total_trials=total_trials,
            force=args.force,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
