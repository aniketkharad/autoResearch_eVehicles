"""run_update_tabpfn_full.py - Update TabPFN to full V3.5, re-stack, and submit to Kaggle.

1. Runs train_tabpfn_large.py with --force to generate tabpfn_oof.npy and tabpfn_test.npy using full TabPFN-3.5.
2. Re-runs stack_large.py to optimize weights across LGB, XGB, CAT, and full TabPFN-3.5.
3. Runs submit_large.py to blend predictions, generate submission_large_stack.csv, and submit to Kaggle.
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_update_tabpfn_full")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


def run_cmd(cmd: list[str], desc: str) -> None:
    logger.info("=" * 70)
    logger.info(f"STARTING: {desc}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        logger.error(f"FAILED: {desc} exited with code {res.returncode}")
        sys.exit(res.returncode)
    logger.info(f"COMPLETED: {desc} in {time.time()-t0:.1f}s\n")


def main():
    logger.info("======================================================================")
    logger.info("LEVEL-2 ENSEMBLE PIPELINE UPDATE: FULL TABPFN-3.5 INTEGRATION")
    logger.info("======================================================================")

    # Step 1: Train full TabPFN-3.5
    run_cmd(
        ["uv", "run", "python", "experiments/level_2_meta/train_tabpfn_large.py", "--force"],
        "Step 1: Train Full TabPFN-3.5 (5-Fold CV on Apple Silicon MPS)",
    )

    # Step 2: Level-2 Stacking & Multi-Start Weight Optimization
    run_cmd(
        ["uv", "run", "python", "experiments/level_2_meta/stack_large.py"],
        "Step 2: Level-2 Stacking Diversity Analysis & Weight Optimization (Nelder-Mead / SLSQP)",
    )

    # Step 3: Kaggle Submission
    run_cmd(
        ["uv", "run", "python", "experiments/level_2_meta/submit_large.py"],
        "Step 3: Test Prediction Blending & Kaggle Competition Submission",
    )

    logger.info("======================================================================")
    logger.info("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
