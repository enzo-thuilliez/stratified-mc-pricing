"""
End-to-end benchmark pipeline orchestrator.

    python generate_all.py           # QUICK_PROFILE (sanity check, ~10-15 min)
    python generate_all.py --full    # FULL_PROFILE  (full run, plan for 4-12 h)

Steps (abort on first failure — never feeds a partial CSV to the next stage)
-----
1. run_surface.py   → figures/benchmark_surface_results.csv
2. make_surface_figures.py → figures 9–14
3. summary_tables.py       → LaTeX tables + Markdown interpretation summary
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

OUT_DIR = Path("figures")
CSV_PATH = OUT_DIR / "benchmark_surface_results.csv"

EXPECTED_FIGURES = [
    "figure_9_heatmap_varratio.png",
    "figure_10_heatmap_rmse.png",
    "figure_11_runtime.png",
    "figure_12_rmse_vs_runtime.png",
    "figure_13_model_comparison.png",
    "figure_14_payoff_comparison.png",
]

EXPECTED_TABLES = [
    "tables_by_model.tex",
    "tables_by_payoff.tex",
    "tables_by_maturity.tex",
    "tables_by_moneyness.tex",
]

EXPECTED_SUMMARY = "benchmark_interpretation_summary.md"


def run_step(label: str, cmd: list) -> float:
    """
    Run one pipeline step as a subprocess, streaming its stdout/stderr live.
    Aborts the whole pipeline (sys.exit) on non-zero exit code so a partial
    CSV can never silently propagate to the next stage.
    Returns wall-clock seconds elapsed.
    """
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  STEP : {label}")
    print(f"  CMD  : {' '.join(cmd)}")
    print(sep, flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(
            f"\n[FATAL] '{label}' exited with code {result.returncode} "
            f"after {elapsed:.1f}s.\n"
            "Pipeline aborted — downstream steps were NOT run.",
            flush=True,
        )
        sys.exit(result.returncode)

    print(
        f"\n[OK] '{label}' completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)",
        flush=True,
    )
    return elapsed


def print_summary(step_times: dict, profile_name: str) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print("  PIPELINE SUMMARY")
    print(sep)

    # -- CSV
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"  [OK]      CSV     : {CSV_PATH}  ({len(df)} rows, {df.shape[1]} cols)")
    else:
        print(f"  [MISSING] CSV     : {CSV_PATH}")

    # -- Figures
    print()
    for fname in EXPECTED_FIGURES:
        p = OUT_DIR / fname
        tag = "[OK]     " if p.exists() else "[MISSING]"
        print(f"  {tag} Figure  : {p}")

    # -- LaTeX tables
    print()
    for fname in EXPECTED_TABLES:
        p = OUT_DIR / fname
        tag = "[OK]     " if p.exists() else "[MISSING]"
        print(f"  {tag} Table   : {p}")

    # -- Markdown summary
    print()
    p = OUT_DIR / EXPECTED_SUMMARY
    tag = "[OK]     " if p.exists() else "[MISSING]"
    print(f"  {tag} Summary : {p}")

    # -- Timing breakdown
    print()
    total = sum(step_times.values())
    for step, t in step_times.items():
        print(f"  {step:<45s} {t:7.1f}s  ({t / 60:.1f} min)")
    print(f"  {'TOTAL':<45s} {total:7.1f}s  ({total / 60:.1f} min)")
    print(f"\n  Profile: {profile_name}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full segmented benchmark pipeline: "
            "benchmark → figures 9-14 → LaTeX tables + Markdown summary."
        )
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run FULL_PROFILE (N=20 000, M=252, 50 replications, samplers=prng+qmc, "
            "4 payoffs). Default: QUICK_PROFILE with n_replications overridden to 4."
        ),
    )
    args = parser.parse_args()

    # --full passes --full straight through to run_surface.py, which uses
    # FULL_PROFILE unchanged (50 reps, prng+qmc).  --quick passes --quick,
    # which applies the n_replications=4 override defined in run_surface.py.
    # The override is local to run_surface.py's --quick branch; it cannot
    # bleed into the --full path.
    if args.full:
        surface_flag = "--full"
        profile_name = "FULL_PROFILE (N=20 000, M=252, 50 reps, prng+qmc)"
    else:
        surface_flag = "--quick"
        profile_name = "QUICK_PROFILE (N=2 000, M=50, 4 reps, prng only)"

    print(f"\ngenerate_all.py — profile: {profile_name}", flush=True)

    step_times: dict = {}

    # ------------------------------------------------------------------ #
    # Step 1: run the segmented surface benchmark                         #
    # ------------------------------------------------------------------ #
    step_times["1. run_surface.py"] = run_step(
        f"Surface benchmark  [{profile_name}]",
        [sys.executable, "run_surface.py", surface_flag],
    )

    # ------------------------------------------------------------------ #
    # Step 2: generate figures 9–14 from the CSV                         #
    # ------------------------------------------------------------------ #
    step_times["2. make_surface_figures.py"] = run_step(
        "Surface figures (9–14)",
        [sys.executable, "make_surface_figures.py"],
    )

    # ------------------------------------------------------------------ #
    # Step 3: generate LaTeX tables + Markdown interpretation summary     #
    # ------------------------------------------------------------------ #
    step_times["3. summary_tables.py"] = run_step(
        "Summary tables + Markdown interpretation",
        [sys.executable, "summary_tables.py"],
    )

    print_summary(step_times, profile_name)


if __name__ == "__main__":
    main()
