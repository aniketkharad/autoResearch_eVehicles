# Stage B: Level-2 Stacking Report (668k Dataset)

- **Dataset Scale**: 668,665 rows x 25 features | 5-Fold Stratified CV
- **Final Stacked OOF ROC AUC**: **`0.943691`**
- **Best Single Model**: **`0.943651`**
- **Incremental Lift**: **`+0.000040`**

## Individual Model 5-Fold OOF AUCs

| Model | 5-Fold OOF ROC AUC | Optimal Blend Weight |
| :--- | :---: | :---: |
| LightGBM | 0.943353 | 0.2624 (26.2%) |
| XGBoost | 0.943651 | 0.7376 (73.8%) |
| CatBoost | 0.942507 | 0.0000 (0.0%) |
| TabPFN | 0.937922 | 0.0000 (0.0%) |
| | LightGBM | XGBoost | CatBoost | TabPFN |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | 1.0000 | 0.9948 | 0.9840 | 0.9719 |
| **XGBoost** | 0.9948 | 1.0000 | 0.9861 | 0.9706 |
| **CatBoost** | 0.9840 | 0.9861 | 1.0000 | 0.9716 |
| **TabPFN** | 0.9719 | 0.9706 | 0.9716 | 1.0000 |

## Optimization Details

- **Method**: Multi-Start SLSQP & Nelder-Mead on Softmax Weight Parameterization
- **Uniform Blend Score**: `0.942885`
- **Optimized Stack Score**: `0.943691`
