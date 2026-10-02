# AutoResearch Knowledge Base (Phase 2): EV Tabular Classification

## Research Objectives (Phase 2)
- **Starting Benchmark**: Out-of-Fold (OOF) 5-Fold ROC AUC **`0.943616`** | Fold 0 Benchmark: **`0.943811`** (Run 11, commit `a6539cc`).
- **Current Best**: Out-of-Fold (OOF) 5-Fold ROC AUC **`0.943657`** | Fold 0 Benchmark: **`0.943945`** (Run 14, commit `2b91fce`).
- **Target**: Break past **`0.9440`** and challenge **`~0.9500 - 0.9600`**.
- **Dataset Context**: 668,665 samples, 13 features (7 numeric, 6 categorical), binary target `Will_Buy_EV` (17.46% positive class).
- **Evaluation Harness**: Deterministic Stratified 5-Fold CV (`splits.npy`, seed=101). Fixed ground truth `prepare.evaluate_predictions`.
- **System Constraints**:
  - Max 8 CPU cores (`n_jobs=8`).
  - Max 8GB RAM: downcast `float64 -> float32`, `int64 -> int32`, explicit `gc.collect()`.
  - Max 120s wall-clock compute budget per experiment.
  - **Mandatory Tier 1 Safeguard**: Screen all new prototypes on **Fold 0 only**. Only run full 5-fold CV if Fold 0 ROC AUC strictly beats **`0.943945`** (new benchmark). If $\le 0.943945$, revert immediately with `git checkout -- train.py`.

---

## Benchmark Progression (Phase 2)

| Iteration | Architecture / Technique | Fold 0 ROC AUC | 5-Fold OOF ROC AUC | Runtime | Status | Git Commit |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 11 | Phase 2 Baseline: Ultra-High-Res XGB (max_bin=4096, w=0.65) + LGB (max_bin=1024, w=0.35) | 0.943811 | 0.943616 | 113.57s | **BENCHMARK** | `a6539cc` |
| Screen | Track A: 2-Way Joint Frequency Encodings (City x Car, Income x Age, Commute x Stations) | 0.943595 | - | 27.19s (F0) | **REVERT (Tier 1)** | - |
| Screen | Track B: Out-of-Fold Target Encoding (City x Car with 5-fold inner CV, m=50) | 0.943689 | - | 24.30s (F0) | **REVERT (Tier 1)** | - |
| Screen | Track C: CatBoost Symmetric Trees (3-way blend 0.45 XGB + 0.30 LGB + 0.25 Cat) | 0.943680 | - | 116.79s (F0) | **REVERT (Tier 1 & Timeout)** | - |
| Screen | Track D: Modulo and Digit Precision Features (Income%1000, Commute decimals) | 0.943724 | - | 26.50s (F0) | **REVERT (Tier 1)** | - |
| 12 | Tree Regularization (XGB max_depth=5 max_bin=4096 + LGB leaves=36) | 0.943904 | 0.943650 | 138.09s | **REVERT (Timeout > 120s)** | `b4f283e` |
| 12b | Regularization: LGB colsample=0.95 + XGB gamma=0.05 | 0.943819 | 0.943601 | 134.43s | **REVERT (Timeout > 120s)** | `b4f283e` |
| 13 | Asymmetric Depth-5 XGB (bin=4096, lr=0.092) + LGB (leaves=36, lr=0.088) | 0.943946 | 0.943658 | 126.19s | **REVERT (Timeout > 120s)** | `f387ca4` |
| 13b | Truncated Stopping: LGB (n=450, stop=26) + XGB (n=410, stop=26) | 0.943876 | 0.943571 | 109.61s | **REVERT (Regression)** | `f387ca4` |
| **14** | **Asymmetric Depth-5 XGB (bin=4096, net_green, sub_x_inc, logloss) + LGB (leaves=36, w=0.38/0.62)** | **0.943945** | **0.943657** | **100.77s** | **KEEP (NEW BEST)** | `2b91fce` |

---

## Detailed Experiment Logs (Phase 2 Breakthroughs)

### Key Architectural Breakthroughs in Iteration 14:

1. **Computational Speedup via Logloss Evaluation in XGBoost**:
   - **Discovery**: In XGBoost, evaluating `eval_metric="auc"` on 133,733 validation samples at every iteration requires an $O(N \log N)$ sort 450 times per fold, consuming ~4.2s per fold (~21s over 5 folds).
   - **Solution**: Replacing `eval_metric="auc"` with `eval_metric="logloss"` preserves the EXACT same tree splitting trajectory and early stopping convergence point (identical AUC to the 6th decimal place: `0.943756` vs `0.943756`), while shaving **20.9 seconds** off total 5-fold runtime.
   - **Impact**: Allowed the ensemble to execute with full convergence capacity in **100.77s**, well within the strict 120s budget ceiling (19.23s safety buffer).

2. **Asymmetric Feature Allocation**:
   - **Discovery**: Feeding synthetic interaction features (`net_green_subsidy` and `sub_x_inc`) to both LightGBM and XGBoost degraded ensemble diversity and reduced blend score (`0.943849`).
   - **Mechanism**: Reserving these interaction features exclusively for XGBoost allowed XGBoost (at depth 5 with `max_bin=4096`) to exploit the strong diagonal decision boundaries, while LightGBM (with 36 leaves at depth 6) explored orthogonal domain splits independently.
   - **Empirical Validation**:
     - LightGBM alone on base features: `0.943598`
     - XGBoost alone on enriched features: `0.943918` (higher than the entire Run 11 blend!)
     - Percentile Rank Blend (0.38 LGB + 0.62 XGB): **`0.943945`** on Fold 0, and **`0.943657`** OOF across 5 folds.

3. **Domain Feature Synergy**:
   - `net_green_subsidy = Subsidy_Available * (Environmental_Concern_Level - Range_Anxiety_Level)`: exhibits a massive **+0.6365** linear correlation with `Will_Buy_EV`.
   - `sub_x_inc = Subsidy_Available * (Annual_Income_USD / 10000.0)`: exhibits **+0.4234** linear correlation, directly isolating the capital expenditure affordability of subsidized EV buyers.

---

## Current Best Configuration (`2b91fce`)
- **LightGBM**:
  - `n_estimators=500`, `learning_rate=0.088`, `num_leaves=36`, `max_depth=6`, `min_child_samples=30`, `max_bin=1024`, `subsample=0.8`, `colsample_bytree=0.8`
  - Early stopping rounds: 30
  - Blend weight: **0.38**
- **XGBoost**:
  - `n_estimators=450`, `learning_rate=0.092`, `max_depth=5`, `max_bin=4096`, `tree_method="hist"`, `enable_categorical=True`, `eval_metric="logloss"`
  - Early stopping rounds: 30
  - Blend weight: **0.62**
- **Feature Set**:
  - Both: In-fold frequency encoding on `Annual_Income_USD`, `Daily_Commute_km`, `Age`; group diffs on `City_Type` and `Current_Car_Type` for Income and Commute; base domain features.
  - XGBoost Exclusive: `net_green_subsidy`, `sub_x_inc`.
