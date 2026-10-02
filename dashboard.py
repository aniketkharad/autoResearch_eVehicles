"""dashboard.py - Autonomous ML Research Dashboard.

Single-file, read-only terminal dashboard using rich:
- Displays all-time best validation ROC AUC.
- Shows total runs, keep count, regressions, and win rate.
- Breakdown of performance across model architectures.
- Recent experiment log with color-coded status badges.

Usage:
    uv run python dashboard.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "results.tsv"
HISTORY_FILE = BASE_DIR / ".results_history.tsv"


def load_all_results() -> pd.DataFrame:
    """Loads and reconciles experiment history from results.tsv and .results_history.tsv."""
    dfs = []
    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        try:
            dfs.append(pd.read_csv(RESULTS_FILE, sep="\t"))
        except Exception:
            pass

    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        try:
            dfs.append(pd.read_csv(HISTORY_FILE, sep="\t"))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(
        subset=["timestamp", "commit_hash", "model_architecture", "val_roc_auc", "description"]
    )
    df["val_roc_auc"] = pd.to_numeric(df["val_roc_auc"], errors="coerce")
    return df


def render_dashboard() -> None:
    console = Console()
    df = load_all_results()

    if df.empty or df["val_roc_auc"].dropna().empty:
        console.print(
            Panel(
                "[yellow]No experiment results found yet.[/yellow]\nRun [bold green]uv run python train.py[/bold green] to record the baseline.",
                title="AutoResearch - EV Tabular Classification",
                border_style="yellow",
            )
        )
        return

    # Metrics computation
    total_runs = len(df)
    keeps = int((df["status"] == "keep").sum())
    regressions = int((df["status"] == "regression").sum())
    win_rate = (keeps / total_runs) * 100 if total_runs > 0 else 0.0

    best_idx = df["val_roc_auc"].idxmax()
    best_row = df.loc[best_idx]
    best_auc = best_row["val_roc_auc"]
    best_model = best_row["model_architecture"]
    best_commit = best_row.get("commit_hash", "unknown")

    # 1. Summary Header Panel
    summary_text = (
        f"[bold cyan]All-Time Best ROC AUC:[/bold cyan] [bold green]{best_auc:.6f}[/bold green] "
        f"([italic]{best_model}[/italic] @ [yellow]{best_commit}[/yellow])\n"
        f"[bold cyan]Total Runs:[/bold cyan] {total_runs}  |  "
        f"[bold green]Improvements (Keeps):[/bold green] {keeps}  |  "
        f"[bold red]Regressions/Discarded:[/bold red] {regressions}  |  "
        f"[bold magenta]Win Rate:[/bold magenta] {win_rate:.1f}%"
    )
    console.print(
        Panel(
            summary_text,
            title="[bold blue]AutoResearch Experiment Dashboard[/bold blue]",
            subtitle="Metric: [bold]Validation ROC AUC (5-fold Stratified CV)[/bold]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )

    # 2. Architecture Breakdown Table
    arch_table = Table(
        title="Performance Breakdown by Architecture",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        expand=True,
    )
    arch_table.add_column("Architecture", style="white")
    arch_table.add_column("Runs", justify="right")
    arch_table.add_column("Best ROC AUC", justify="right", style="green")
    arch_table.add_column("Avg ROC AUC", justify="right")
    arch_table.add_column("Keeps", justify="right", style="cyan")

    arch_group = df.groupby("model_architecture")
    for arch, group in arch_group:
        arch_runs = len(group)
        arch_best = group["val_roc_auc"].max()
        arch_avg = group["val_roc_auc"].mean()
        arch_keeps = (group["status"] == "keep").sum()
        arch_table.add_row(
            str(arch),
            str(arch_runs),
            f"{arch_best:.6f}",
            f"{arch_avg:.6f}",
            str(arch_keeps),
        )
    console.print(arch_table)

    # 3. Recent Experiment Ledger Table
    history_table = Table(
        title="Experiment Ledger (Recent Runs)",
        box=box.SIMPLE,
        header_style="bold yellow",
        expand=True,
    )
    history_table.add_column("#", justify="right", style="dim", width=4)
    history_table.add_column("Timestamp", style="dim")
    history_table.add_column("Commit", style="yellow", width=9)
    history_table.add_column("Architecture", style="cyan")
    history_table.add_column("Val ROC AUC", justify="right")
    history_table.add_column("Status", justify="center")
    history_table.add_column("Hypothesis / Description", style="italic")

    recent_df = df.tail(15).reset_index(drop=True)
    for idx, row in recent_df.iterrows():
        run_num = len(df) - len(recent_df) + idx + 1
        score_str = f"{row['val_roc_auc']:.6f}" if pd.notna(row['val_roc_auc']) else "N/A"
        status = str(row.get("status", "unknown")).lower()
        if status == "keep":
            status_cell = Text("KEEP", style="bold green")
        elif status == "regression":
            status_cell = Text("REVERT", style="bold red")
        else:
            status_cell = Text(status.upper(), style="dim")

        history_table.add_row(
            str(run_num),
            str(row.get("timestamp", "")),
            str(row.get("commit_hash", "")),
            str(row.get("model_architecture", "")),
            score_str,
            status_cell,
            str(row.get("description", "")),
        )

    console.print(history_table)


if __name__ == "__main__":
    render_dashboard()
