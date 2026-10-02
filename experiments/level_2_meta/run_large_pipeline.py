"""run_large_pipeline.py - Master Pipeline Orchestrator for 668k Dataset.

Usage:
    uv run python level_2_meta/run_large_pipeline.py
    uv run python level_2_meta/run_large_pipeline.py --step models
    uv run python level_2_meta/run_large_pipeline.py --step stack
    uv run python level_2_meta/run_large_pipeline.py --step submit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add level_2_meta directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from step1_large_data import prepare_large_data
from train_models_large import run_all_models
from stack_large import run_stacking
from submit_large import generate_submission_file, submit_to_kaggle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_large_pipeline")


def run_full_pipeline(
    step: str = "all",
    target_model: str = "all",
    force: bool = False,
    no_submit: bool = False,
) -> None:
    t_start = time.time()
    logger.info("\n" + "#" * 75)
    logger.info("MASTER LEVEL-2 STACKING PIPELINE - LARGE DATASET (668,665 ROWS)")
    logger.info("#" * 75)

    # Step 1: Data Preparation
    if step in ["1", "data", "all"]:
        logger.info("\n>>> STEP 1: PREPARING LARGE DATASET & DOMAIN FEATURES <<<")
        prepare_large_data()

    # Step 2: Model Training
    if step in ["2", "models", "all"]:
        logger.info("\n>>> STEP 2: 5-FOLD CV TRAINING (LGBM, XGB, CAT, TABPFN) <<<")
        run_all_models(target_model=target_model, force=force)

    # Step 3: Stacking & Optimization
    if step in ["3", "stack", "all"]:
        logger.info("\n>>> STEP 3: LEVEL-2 RANK STACKING & WEIGHT OPTIMIZATION <<<")
        run_stacking()

    # Step 4: Submission Generation & Kaggle Submit
    if step in ["4", "submit", "all"]:
        logger.info("\n>>> STEP 4: GENERATING TEST SUBMISSION & SUBMITTING TO KAGGLE <<<")
        sub_file = generate_submission_file()
        if not no_submit:
            submit_to_kaggle(sub_file)

    elapsed = time.time() - t_start
    logger.info(f"\nPipeline finished completely in {elapsed / 60:.2f} minutes.")


def main():
    parser = argparse.ArgumentParser(description="Run Level-2 Stacking on 668k dataset")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "data", "models", "stack", "submit", "all"], default="all")
    parser.add_argument("--model", choices=["lgb", "xgb", "cat", "tabpfn", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()

    run_full_pipeline(
        step=args.step,
        target_model=args.model,
        force=args.force,
        no_submit=args.no_submit,
    )


if __name__ == "__main__":
    main()
