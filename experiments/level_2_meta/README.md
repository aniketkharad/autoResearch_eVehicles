# Level-2 Meta-Stacking Pipeline

An advanced ensemble and stacking architecture combining four heterogeneous model families for tabular classification on the 668,665-row dataset.

## Model Components

1. **LightGBM**: Histogram-based gradient boosting with depth and leaf regularization.
2. **XGBoost**: Exact and high-bin tree boosting with custom domain interaction features (`net_green_subsidy`, `sub_x_inc`).
3. **CatBoost**: Oblivious trees offering orthogonal inductive bias and strong categorical regularization.
4. **TabPFN-3.5**: Tabular Prior-data Fitted Network producing Bayesian posterior distributions.

## Pipeline Architecture

* `step1_large_data.py`: Loads `data/train.csv` and `data/test.csv`, engineering interaction and frequency features without target leakage.
* `train_models_large.py`: Trains all four model families across deterministic 5-fold cross-validation, caching out-of-fold (`cache_large/*_oof.npy`) and test predictions (`cache_large/*_test.npy`).
* `stack_large.py`: Computes prediction correlation matrices (Spearman rank) and executes multi-start SLSQP and Nelder-Mead weight optimization to find the optimal ensemble blend.
* `submit_large.py`: Blends test predictions using the optimal weights and generates submission artifacts.
* `run_large_pipeline.py`: Master pipeline orchestrator.

## Quick Run

```bash
# Run full end-to-end stacking pipeline
uv run python experiments/level_2_meta/run_large_pipeline.py
```
