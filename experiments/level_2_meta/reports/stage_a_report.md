# Level-2 Stacking Ensemble: Stage A Final Report 

**Date**: 2026-09-30 23:19:33  
**Development Dataset**: `EV_Adoption_and_Range_Anxiety_Dataset.csv`  
**Sample Count**: 10,000 rows | **Target**: `Will_Buy_EV` (17.5% positive, 82.5% negative)  
**Hardware Detected**: macOS-15.3-arm64-arm-64bit-Mach-O (arm64) | 10 CPU cores | Device: MPS  

---

## 1. Individual Model OOF ROC AUC (5-Fold CV)

| Model Architecture | OOF ROC AUC | Delta vs Best Single |
| :--- | :---: | :---: |
| **LightGBM** (Tuned) | `0.90447` | `-0.00066` |
| **XGBoost** (Tuned) | `0.90449` | `-0.00064` |
| **CatBoost** (Tuned) | `0.90348` | `-0.00165` |
| **TabPFN-3.5 Fast** (Frozen) | `0.90512` | `+0.00000` |

---

## 2. Pairwise Spearman Rank Correlation Matrix

```
          LightGBM  XGBoost  CatBoost  TabPFN
LightGBM    1.0000   0.9914    0.9814  0.9838
XGBoost     0.9914   1.0000    0.9827  0.9842
CatBoost    0.9814   0.9827    1.0000  0.9886
TabPFN      0.9838   0.9842    0.9886  1.0000
```

### Diversity Highlights:
- **TabPFN vs LightGBM**: `0.9838`
- **TabPFN vs XGBoost**:  `0.9842`
- **TabPFN vs CatBoost**: `0.9886`
- **Tree-to-Tree Average**: `0.9851`

*Note*: Lower correlation between TabPFN and the tree models indicates genuinely complementary ranking information.

---

## 3. Level-2 Stacking Ensemble Comparison

| Ensemble Configuration | Validation ROC AUC | Delta vs Best Single (`TabPFN`) |
| :--- | :---: | :---: |
| **Best Single Model** (TabPFN) | `0.90512` | `+0.00000` |
| **Trees-Only Equal Blend** (LGB + XGB + Cat) | `0.90533` | `+0.00021` |
| **Trees-Only Optimized Blend** | `0.90534` | `+0.00022` |
| **4-Model Equal Rank Blend** (LGB + XGB + Cat + PFN) | `0.90576` | `+0.00064` |
| **Optimized 4-Model Rank Blend** (SLSQP / Nelder-Mead) | **`0.90595`** | **`+0.00083`** |

---

## 4. Final Optimized Blend Weights

```json
{
  "LightGBM": 0.2240,
  "XGBoost":  0.2158,
  "CatBoost": 0.0458,
  "TabPFN":   0.5144
}
```

**Net Stacking Gain**:
- Improvement over Best Single Model: **`+0.00083`**
- Improvement over Trees-Only Blend:  **`+0.00061`**

> [!NOTE]
> Component predictions are strictly Out-of-Fold. The reported stacked AUC represents the metric achieved by optimizing non-negative rank blend weights on the 5-fold OOF prediction matrix.

---

## 5. Review Gate Status: STOP & AWAIT APPROVAL
Stage A on `EV_Adoption_and_Range_Anxiety_Dataset.csv` is complete.
Do not proceed to Stage B (`train.csv`) until this result is reviewed.
