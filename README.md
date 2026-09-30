# autoResearch_eVehicles

Autonomous iterative machine learning experiment loop following the methodology of Andrej Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch), tailored for tabular classification optimizing **ROC AUC** on electric vehicle purchase intent.

The core idea: instead of manual trial-and-error modeling, the AI agent operates autonomously overnight in a continuous empirical loop—modifying code, training across deterministic cross-validation folds, evaluating validation ROC AUC, and ratcheting the repository forward via Git commits whenever the historical best score is beaten.

---

## Architecture

The codebase is strictly minimal and relies exclusively on Python, Git, and flat files (no heavyweight MLOps platforms like WandB or MLflow):

- **`prepare.py`** *(LOCKED - IMMUTABLE)*: Loads data, generates deterministic stratified 5-fold splits (`splits.npy`), and defines the immutable `evaluate_predictions` ROC AUC evaluation harness. Locks `test.csv` out of reach.
- **`train.py`** *(THE ONLY FILE MODIFIED BY AGENT)*: Contains feature engineering, model architecture, training loop, and logging. Must finish within a strict 60-second budget per run.
- **`program.md`** *(RESEARCH DIRECTIVE)*: Instructions and playbook directing the autonomous agent's hypotheses and research loop.
- **`dashboard.py`** *(READ-ONLY)*: Rich terminal dashboard tracking all-time best ROC AUC, total runs, win rate, and architecture breakdown.
- **`results.tsv`** *(EXPERIMENT LEDGER)*: Flat TSV recording every attempt with timestamp, commit hash, architecture, validation score, status, and description.

---

## Quick Start

### 1. Setup Environment
```bash
uv sync
```

### 2. Verify Data & Deterministic Splits
```bash
uv run python prepare.py
```

### 3. Run a Training Iteration
```bash
uv run python train.py
```

### 4. View Research Dashboard
```bash
uv run python dashboard.py
```

---

## The Ratchet Loop

1. Read `results.tsv` and `train.py`.
2. Formulate a specific hypothesis (feature engineering, hyperparameter tuning, model architecture).
3. Edit `train.py`.
4. Execute `uv run python train.py`.
5. If `val_roc_auc > historical_best`:
   ```bash
   git commit -am "feat: <model/technique> - val_roc_auc: <score>"
   git push origin main
   ```
6. If `val_roc_auc <= historical_best` or errors:
   ```bash
   git reset --hard HEAD
   git clean -fd
   ```
7. Repeat.