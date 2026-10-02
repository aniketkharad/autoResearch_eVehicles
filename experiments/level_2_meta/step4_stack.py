"""step4_stack.py - Level-2 Rank Stacking, Spearman Diversity Analysis & Constrained Weight Optimization.

Workflow:
1. Assembles prediction matrix P = [lgb, xgb, cat, pfn] with shape (N, 4).
2. Computes and reports pairwise Spearman rank correlation matrix.
3. Applies rank transformation R_i = rankdata(P_i) / N (average ties).
4. Solves constrained non-negative weight optimization:
       max ROC_AUC(R @ w) subject to w >= 0, sum(w) = 1
   using multiple initializations (SLSQP & Nelder-Mead).
5. Compares:
   - Individual model AUCs
   - Equal-weight rank blend AUC
   - Trees-only blend AUC (LGB + XGB + Cat)
   - Optimized 4-model blend AUC
6. Generates Stage A Final Experiment Report under level_2_meta/reports/stage_a_report.md.
7. Enforces Review Gate before Stage B.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import (
    CACHE_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    compute_spearman_matrix,
    detect_hardware_and_env,
    evaluate_auc,
    load_json,
    optimize_blend_weights,
    rank_transform,
    save_json,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    load_and_preprocess_dev_data,
)

logger = setup_logger("step4_stack")


def assemble_prediction_matrix(
    smoke: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads OOF prediction arrays and returns matrix P (N x 4), ground truth y, and model names."""
    suffix = "_smoke" if smoke else ""
    model_names = ["LightGBM", "XGBoost", "CatBoost", "TabPFN"]

    # Target ground truth
    X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data()
    n_samples = len(y)

    # Load tree models
    preds_dict: Dict[str, np.ndarray] = {}
    for key, file_prefix in [
        ("LightGBM", f"lightgbm{suffix}_oof.npy"),
        ("XGBoost", f"xgboost{suffix}_oof.npy"),
        ("CatBoost", f"catboost{suffix}_oof.npy"),
    ]:
        p_path = CACHE_DIR / file_prefix
        if not p_path.exists():
            # Try alternate naming
            p_path = CACHE_DIR / f"{key.lower()[:3]}{suffix}_oof.npy"
        if not p_path.exists() and smoke:
            # Fall back to non-smoke if smoke file absent
            p_path = CACHE_DIR / f"{key.lower()}_oof.npy"
        if not p_path.exists():
            raise FileNotFoundError(f"Missing OOF prediction file: {p_path}. Run step3_oof.py first!")
        arr = np.load(p_path)
        assert len(arr) == n_samples, f"Shape mismatch in {file_prefix}: expected {n_samples}, got {len(arr)}"
        preds_dict[key] = arr.astype(np.float32)

    # Load TabPFN
    tabpfn_path = CACHE_DIR / "tabpfn_oof.npy"
    if not tabpfn_path.exists() and smoke:
        tabpfn_path = CACHE_DIR / "tabpfn_oof_smoke.npy"
    if not tabpfn_path.exists():
        raise FileNotFoundError(
            f"Missing TabPFN prediction file at {tabpfn_path}. Run generate_tabpfn_dev.py first!"
        )
    tabpfn_arr = np.load(tabpfn_path)
    assert len(tabpfn_arr) == n_samples, f"Shape mismatch in TabPFN: expected {n_samples}, got {len(tabpfn_arr)}"
    preds_dict["TabPFN"] = tabpfn_arr.astype(np.float32)

    # Assemble P: Column order: 1. lgb, 2. xgb, 3. cat, 4. pfn
    P = np.column_stack([
        preds_dict["LightGBM"],
        preds_dict["XGBoost"],
        preds_dict["CatBoost"],
        preds_dict["TabPFN"],
    ]).astype(np.float32)

    assert P.shape == (n_samples, 4), f"Unexpected shape for P: {P.shape}"
    assert not np.isnan(P).any(), "NaNs detected in assembled prediction matrix P!"
    assert not np.isinf(P).any(), "Infs detected in assembled prediction matrix P!"

    return P, y, model_names


def run_stacking_pipeline(
    smoke: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Runs complete Level-2 stacking ensemble, correlation analysis, and weight optimization."""
    suffix = "_smoke" if smoke else ""
    P, y, model_names = assemble_prediction_matrix(smoke=smoke)
    n_samples, n_models = P.shape

    logger.info("\n" + "=" * 70)
    logger.info("LEVEL-2 STACKING: COMPONENT OOF ROC AUC SCORES")
    logger.info("=" * 70)

    individual_aucs = {}
    for i, name in enumerate(model_names):
        auc = evaluate_auc(y, P[:, i])
        individual_aucs[name] = auc
        logger.info(f"  {name:<12} OOF ROC AUC: {auc:.5f}")

    # 1. Spearman Rank Correlation
    logger.info("\n" + "=" * 70)
    logger.info("PAIRWISE SPEARMAN RANK CORRELATION MATRIX")
    logger.info("=" * 70)
    df_corr, _ = compute_spearman_matrix(P, model_names)
    logger.info(f"\n{df_corr.to_string()}\n")

    logger.info("Spearman Diversity Highlights:")
    logger.info(f"  TabPFN vs LightGBM: {df_corr.loc['TabPFN', 'LightGBM']:.4f}")
    logger.info(f"  TabPFN vs XGBoost:  {df_corr.loc['TabPFN', 'XGBoost']:.4f}")
    logger.info(f"  TabPFN vs CatBoost: {df_corr.loc['TabPFN', 'CatBoost']:.4f}")
    logger.info(f"  Tree-to-Tree (Avg): {(df_corr.loc['LightGBM', 'XGBoost'] + df_corr.loc['LightGBM', 'CatBoost'] + df_corr.loc['XGBoost', 'CatBoost']) / 3:.4f}")

    # 2. Rank Transformation
    logger.info("\n" + "=" * 70)
    logger.info("APPLYING RANK TRANSFORMATION: R_i = rankdata(P_i) / N")
    logger.info("=" * 70)
    R = np.zeros_like(P, dtype=np.float32)
    for j in range(n_models):
        R[:, j] = rank_transform(P[:, j])

    # 3. Baselines: Equal-Weight Blend & Trees-Only Blend
    w_equal = np.ones(n_models) / n_models
    pred_equal = R @ w_equal
    auc_equal = evaluate_auc(y, pred_equal)

    # Trees-only blend (equal and optimized)
    R_trees = R[:, :3]
    w_trees_equal = np.ones(3) / 3.0
    auc_trees_equal = evaluate_auc(y, R_trees @ w_trees_equal)

    opt_trees = optimize_blend_weights(
        R_trees,
        y,
        model_names=["LightGBM", "XGBoost", "CatBoost"],
        logger=logger,
    )
    auc_trees_opt = opt_trees["best_auc"]

    # 4. Multi-Start 4-Model Weight Optimization
    opt_result = optimize_blend_weights(
        R,
        y,
        model_names=model_names,
        logger=logger,
    )
    best_weights = opt_result["weights"]
    best_weights_arr = np.array(opt_result["weights_array"], dtype=np.float32)
    auc_opt = opt_result["best_auc"]

    # Final Stacked Predictions
    stacked_oof_preds = (R @ best_weights_arr).astype(np.float32)
    stacked_cache_path = CACHE_DIR / f"stacked{suffix}_oof.npy"
    np.save(stacked_cache_path, stacked_oof_preds)
    logger.info(f"\nSaved stacked OOF predictions to {stacked_cache_path}")

    # Save weights JSON
    weights_summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_names": model_names,
        "individual_aucs": {k: round(v, 5) for k, v in individual_aucs.items()},
        "equal_weight_blend_auc": round(auc_equal, 5),
        "trees_only_equal_auc": round(auc_trees_equal, 5),
        "trees_only_optimized_auc": round(auc_trees_opt, 5),
        "trees_only_weights": opt_trees["weights"],
        "optimized_four_model_auc": round(auc_opt, 5),
        "final_weights": {k: round(v, 5) for k, v in best_weights.items()},
        "best_start_initialization": opt_result["best_init"],
        "spearman_correlation_matrix": df_corr.to_dict(),
        "optimization_history": opt_result["history"],
    }
    weights_path = RESULTS_DIR / f"weights{suffix}.json"
    save_json(weights_summary, weights_path)
    logger.info(f"Saved blend weights and metrics to {weights_path}")

    # Generate Review Gate Report
    report_content = generate_stage_a_report(
        individual_aucs=individual_aucs,
        df_corr=df_corr,
        auc_equal=auc_equal,
        auc_trees_equal=auc_trees_equal,
        auc_trees_opt=auc_trees_opt,
        opt_trees_weights=opt_trees["weights"],
        auc_opt=auc_opt,
        best_weights=best_weights,
        smoke=smoke,
    )
    report_path = REPORTS_DIR / f"stage_a_report{suffix}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"\nGenerated Stage A Final Report at {report_path}")

    # Print summary to console
    print(report_content)

    logger.info(
        f"\n{'='*70}\n"
        f"REVIEW GATE ENFORCED: STAGE A DEVELOPMENT COMPLETED\n"
        f"Do NOT proceed to Stage B (train.csv / test.csv) without explicit user review.\n"
        f"{'='*70}\n"
    )
    return weights_summary


def generate_stage_a_report(
    individual_aucs: Dict[str, float],
    df_corr: pd.DataFrame,
    auc_equal: float,
    auc_trees_equal: float,
    auc_trees_opt: float,
    opt_trees_weights: Dict[str, float],
    auc_opt: float,
    best_weights: Dict[str, float],
    smoke: bool = False,
) -> str:
    """Formats markdown experiment report for Stage A review gate."""
    hw = detect_hardware_and_env()
    top_indiv_name = max(individual_aucs, key=individual_aucs.get)
    top_indiv_auc = individual_aucs[top_indiv_name]
    gain_over_best = auc_opt - top_indiv_auc
    gain_over_trees = auc_opt - auc_trees_opt

    report = f"""# Level-2 Stacking Ensemble: Stage A Final Report {'(SMOKE TEST)' if smoke else ''}

**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Development Dataset**: `EV_Adoption_and_Range_Anxiety_Dataset.csv`  
**Sample Count**: 10,000 rows | **Target**: `Will_Buy_EV` (17.5% positive, 82.5% negative)  
**Hardware Detected**: {hw['platform']} ({hw['machine']}) | {hw['cpu_count']} CPU cores | Device: {hw['device_selected'].upper()}  

---

## 1. Individual Model OOF ROC AUC (5-Fold CV)

| Model Architecture | OOF ROC AUC | Delta vs Best Single |
| :--- | :---: | :---: |
| **LightGBM** (Tuned) | `{individual_aucs['LightGBM']:.5f}` | `{individual_aucs['LightGBM'] - top_indiv_auc:+.5f}` |
| **XGBoost** (Tuned) | `{individual_aucs['XGBoost']:.5f}` | `{individual_aucs['XGBoost'] - top_indiv_auc:+.5f}` |
| **CatBoost** (Tuned) | `{individual_aucs['CatBoost']:.5f}` | `{individual_aucs['CatBoost'] - top_indiv_auc:+.5f}` |
| **TabPFN-3.5 Fast** (Frozen) | `{individual_aucs['TabPFN']:.5f}` | `{individual_aucs['TabPFN'] - top_indiv_auc:+.5f}` |

---

## 2. Pairwise Spearman Rank Correlation Matrix

```
{df_corr.round(4).to_string()}
```

### Diversity Highlights:
- **TabPFN vs LightGBM**: `{df_corr.loc['TabPFN', 'LightGBM']:.4f}`
- **TabPFN vs XGBoost**:  `{df_corr.loc['TabPFN', 'XGBoost']:.4f}`
- **TabPFN vs CatBoost**: `{df_corr.loc['TabPFN', 'CatBoost']:.4f}`
- **Tree-to-Tree Average**: `{(df_corr.loc['LightGBM', 'XGBoost'] + df_corr.loc['LightGBM', 'CatBoost'] + df_corr.loc['XGBoost', 'CatBoost']) / 3:.4f}`

*Note*: Lower correlation between TabPFN and the tree models indicates genuinely complementary ranking information.

---

## 3. Level-2 Stacking Ensemble Comparison

| Ensemble Configuration | Validation ROC AUC | Delta vs Best Single (`{top_indiv_name}`) |
| :--- | :---: | :---: |
| **Best Single Model** ({top_indiv_name}) | `{top_indiv_auc:.5f}` | `+0.00000` |
| **Trees-Only Equal Blend** (LGB + XGB + Cat) | `{auc_trees_equal:.5f}` | `{auc_trees_equal - top_indiv_auc:+.5f}` |
| **Trees-Only Optimized Blend** | `{auc_trees_opt:.5f}` | `{auc_trees_opt - top_indiv_auc:+.5f}` |
| **4-Model Equal Rank Blend** (LGB + XGB + Cat + PFN) | `{auc_equal:.5f}` | `{auc_equal - top_indiv_auc:+.5f}` |
| **Optimized 4-Model Rank Blend** (SLSQP / Nelder-Mead) | **`{auc_opt:.5f}`** | **`{gain_over_best:+.5f}`** |

---

## 4. Final Optimized Blend Weights

```json
{{
  "LightGBM": {best_weights['LightGBM']:.4f},
  "XGBoost":  {best_weights['XGBoost']:.4f},
  "CatBoost": {best_weights['CatBoost']:.4f},
  "TabPFN":   {best_weights['TabPFN']:.4f}
}}
```

**Net Stacking Gain**:
- Improvement over Best Single Model: **`{gain_over_best:+.5f}`**
- Improvement over Trees-Only Blend:  **`{gain_over_trees:+.5f}`**

> [!NOTE]
> Component predictions are strictly Out-of-Fold. The reported stacked AUC represents the metric achieved by optimizing non-negative rank blend weights on the 5-fold OOF prediction matrix.

---

## 5. Review Gate Status: STOP & AWAIT APPROVAL
Stage A on `EV_Adoption_and_Range_Anxiety_Dataset.csv` is complete.
Do not proceed to Stage B (`train.csv`) until this result is reviewed.
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Step 4: Level-2 Stacking Ensemble")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_stacking_pipeline(smoke=args.smoke, force=args.force)


if __name__ == "__main__":
    main()
