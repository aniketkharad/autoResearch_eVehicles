"""run_pipeline.py - Master Orchestrator for Level-2 Stacking Pipeline.

Usage:
    uv run python level_2_meta/run_pipeline.py --mode smoke
    uv run python level_2_meta/run_pipeline.py --mode full --trials 40
    uv run python level_2_meta/run_pipeline.py --step 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add level_2_meta directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    OPTUNA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    print_hardware_summary,
    setup_logger,
)
from step1_data import (
    DEFAULT_DEV_CSV,
    get_or_create_cv_splits,
    load_and_preprocess_dev_data,
    scan_for_tabpfn_candidates,
)
from step2_tune import run_optuna_study
from step3_oof import run_all_oof_generation
from step4_stack import run_stacking_pipeline

logger = setup_logger("run_pipeline")


def run_pipeline(
    mode: str = "full",
    trials: int = 40,
    step: str = "all",
    force: bool = False,
    tabpfn_path: str | None = None,
) -> None:
    """Executes the Level-2 stacking pipeline steps."""
    is_smoke = mode == "smoke"
    actual_trials = 3 if is_smoke else trials

    logger.info("\n" + "#" * 70)
    logger.info(f"LEVEL-2 STACKING PIPELINE - MODE: {mode.upper()} ({actual_trials} trials/model)")
    logger.info("#" * 70)

    print_hardware_summary(logger)
    t_start = time.time()

    # Step 1: Data Preparation & Inspection
    if step in ["1", "all"]:
        logger.info("\n>>> EXECUTING STEP 1: DATA INSPECTION & CV SPLITS <<<")
        X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data(DEFAULT_DEV_CSV)
        splits = get_or_create_cv_splits(y, force=force)
        tabpfn_preds = scan_for_tabpfn_candidates(
            target_length=len(y),
            y_true=y,
            explicit_path=Path(tabpfn_path) if tabpfn_path else None,
        )
        if tabpfn_preds is not None:
            logger.info("Step 1 Complete: Data and frozen TabPFN predictions validated.")

    # Step 2: Model Tuning with Optuna
    if step in ["2", "all"]:
        logger.info("\n>>> EXECUTING STEP 2: ADAPTIVE OPTUNA TUNING (LGB, XGB, CAT) <<<")
        X, y, cat_cols, num_cols, ids = load_and_preprocess_dev_data(DEFAULT_DEV_CSV)
        splits = get_or_create_cv_splits(y)

        for model_name in ["lightgbm", "xgboost", "catboost"]:
            run_optuna_study(
                model_name=model_name,
                X=X,
                y=y,
                splits=splits,
                cat_cols=cat_cols,
                total_trials=actual_trials,
                force=force,
                smoke=is_smoke,
            )

    # Step 3: Level-1 OOF Prediction Generation
    if step in ["3", "all"]:
        logger.info("\n>>> EXECUTING STEP 3: LEVEL-1 OOF GENERATION & CHECKPOINTING <<<")
        run_all_oof_generation(
            force=force,
            smoke=is_smoke,
            tabpfn_path=tabpfn_path,
        )

    # Step 4: Level-2 Stacking & Review Gate
    if step in ["4", "all"]:
        logger.info("\n>>> EXECUTING STEP 4: LEVEL-2 STACKING & WEIGHT OPTIMIZATION <<<")
        run_stacking_pipeline(
            smoke=is_smoke,
            force=force,
        )

    elapsed_total = time.time() - t_start
    logger.info(f"\nPipeline finished in {elapsed_total:.1f}s.")
    logger.info(
        f"\n{'#'*70}\n"
        f"REVIEW GATE ENFORCED:\n"
        f"Stage A on development dataset completed.\n"
        f"Review report saved to level_2_meta/reports/stage_a_report{'_smoke' if is_smoke else ''}.md\n"
        f"DO NOT proceed to Stage B (train.csv / test.csv) without explicit user review.\n"
        f"{'#'*70}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Level-2 Stacking Ensemble Runner")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full", help="Execution mode")
    parser.add_argument("--trials", type=int, default=40, help="Tuning budget per model")
    parser.add_argument("--step", choices=["1", "2", "3", "4", "all"], default="all", help="Pipeline step to run")
    parser.add_argument("--force", action="store_true", help="Force recomputation of artifacts")
    parser.add_argument("--tabpfn-oof", type=str, default=None, help="Explicit path to frozen TabPFN OOF")
    args = parser.parse_args()

    run_pipeline(
        mode=args.mode,
        trials=args.trials,
        step=args.step,
        force=args.force,
        tabpfn_path=args.tabpfn_oof,
    )


if __name__ == "__main__":
    main()
