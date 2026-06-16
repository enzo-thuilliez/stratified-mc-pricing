"""
Driver script for the maturity x moneyness surface diagnostic figures
(Figures 9-14). Loads the pre-computed benchmark CSV and only renders
plots -- it does not run any simulation or pricing itself.
"""

import os
import sys
import traceback

import pandas as pd

from config import OUT_DIR
from visualization import (
    plot_figure9_heatmap_varratio,
    plot_figure10_heatmap_rmse,
    plot_figure11_runtime,
    plot_figure12_rmse_vs_runtime,
    plot_figure13_model_comparison,
    plot_figure14_payoff_comparison,
)

CSV_PATH = os.path.join(OUT_DIR, "benchmark_surface_results.csv")


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {CSV_PATH}: {df.shape[0]} rows, {df.shape[1]} columns.")

    figures = [
        ("Figure 9 (var-ratio heatmap)",  plot_figure9_heatmap_varratio),
        ("Figure 10 (RMSE heatmap)",      plot_figure10_heatmap_rmse),
        ("Figure 11 (runtime barplot)",   plot_figure11_runtime),
        ("Figure 12 (RMSE vs runtime)",   plot_figure12_rmse_vs_runtime),
        ("Figure 13 (model comparison)",  plot_figure13_model_comparison),
        ("Figure 14 (payoff comparison)", plot_figure14_payoff_comparison),
    ]

    failures = []
    for name, fn in figures:
        try:
            fn(df)
        except Exception:
            failures.append(name)
            print(f"FAILED: {name}")
            traceback.print_exc()

    print()
    if failures:
        print(f"{len(failures)} figure(s) failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All surface figures generated successfully.")


if __name__ == "__main__":
    main()
