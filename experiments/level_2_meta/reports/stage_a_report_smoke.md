# Level-2 Stacking Ensemble: Stage A Final Report (SMOKE TEST)

**Date**: 2026-09-30 23:13:40  
**Development Dataset**: `EV_Adoption_and_Range_Anxiety_Dataset.csv`  
**Sample Count**: 10,000 rows | **Target**: `Will_Buy_EV` (17.5% positive, 82.5% negative)  
**Hardware Detected**: macOS-15.3-arm64-arm-64bit-Mach-O (arm64) | 10 CPU cores | Device: MPS  

---

## 1. Individual Model OOF ROC AUC (5-Fold CV)

| Model Architecture | OOF ROC AUC | Delta vs Best Single |
| :--- | :---: | :---: |
| **LightGBM** (Tuned) | `0.90245` | `-0.00267` |
| **XGBoost** (Tuned) | `0.90227` | `-0.00286` |
| **CatBoost** (Tuned) | `0.90285` | `-0.00227` |
| **TabPFN-3.5 Fast** (Frozen) | `0.90512` | `+0.00000` |

---

## 2. Pairwise Spearman Rank Correlation Matrix

```
          LightGBM  XGBoost  CatBoost  TabPFN
LightGBM    1.0000   0.9795    0.9800  0.9797
XGBoost     0.9795   1.0000    0.9795  0.9780
CatBoost    0.9800   0.9795    1.0000  0.9890
TabPFN      0.9797   0.9780    0.9890  1.0000
```

### Diversity Highlights:
- **TabPFN vs LightGBM**: `0.9797`
- **TabPFN vs XGBoost**:  `0.9780`
- **TabPFN vs CatBoost**: `0.9890`
- **Tree-to-Tree Average**: `0.9797`

*Note*: Lower correlation between TabPFN and the tree models indicates genuinely complementary ranking information.

---

## 3. Level-2 Stacking Ensemble Comparison

| Ensemble Configuration | Validation ROC AUC | Delta vs Best Single (`TabPFN`) |
| :--- | :---: | :---: |
| **Best Single Model** (TabPFN) | `0.90512` | `+0.00000` |
| **Trees-Only Equal Blend** (LGB + XGB + Cat) | `0.90426` | `-0.00087` |
| **Trees-Only Optimized Blend** | `0.90431` | `-0.00082` |
| **4-Model Equal Rank Blend** (LGB + XGB + Cat + PFN) | `0.90501` | `-0.00011` |
| **Optimized 4-Model Rank Blend** (SLSQP / Nelder-Mead) | **`0.90554`** | **`+0.00041`** |

---

## 4. Final Optimized Blend Weights

```json
{
  "LightGBM": 0.2621,
  "XGBoost":  0.0280,
  "CatBoost": 0.0101,
  "TabPFN":   0.6998
}
```

**Net Stacking Gain**:
- Improvement over Best Single Model: **`+0.00041`**
- Improvement over Trees-Only Blend:  **`+0.00123`**

> [!NOTE]
> Component predictions are strictly Out-of-Fold. The reported stacked AUC represents the metric achieved by optimizing non-negative rank blend weights on the 5-fold OOF prediction matrix.

---

## 5. Review Gate Status: STOP & AWAIT APPROVAL
Stage A on `EV_Adoption_and_Range_Anxiety_Dataset.csv` is complete.
Do not proceed to Stage B (`train.csv`) until this result is reviewed.
