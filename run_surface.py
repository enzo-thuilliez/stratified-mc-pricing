"""
Entry point for the segmented maturity x moneyness x payoff surface benchmark.

    python run_surface.py --quick   # QUICK_PROFILE (sanity check, few minutes)
    python run_surface.py --full    # FULL_PROFILE (full run)

Writes one CSV row per exact configuration to
figures/benchmark_surface_results.csv (columns: see benchmark.SURFACE_COLUMNS).
"""

import argparse
import os
import time

from benchmark import run_surface_benchmark, SURFACE_COLUMNS
from surface_grid import QUICK_PROFILE, FULL_PROFILE

OUT_PATH = os.path.join("figures", "benchmark_surface_results.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmented pricing-surface benchmark.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run QUICK_PROFILE.")
    group.add_argument("--full", action="store_true", help="Run FULL_PROFILE.")
    args = parser.parse_args()

    if args.quick:
        # Override applies only to the quick sanity-check run (without
        # touching surface_grid.py): QUICK_PROFILE's 8 replications still
        # pushed --quick to ~20 min wall-clock, so this trims it to 4.
        # FULL_PROFILE is untouched and keeps its 50 replications.
        profile = {**QUICK_PROFILE, "n_replications": 4}
        profile_name = "QUICK_PROFILE (n_replications overridden to 4)"
    else:
        profile = FULL_PROFILE
        profile_name = "FULL_PROFILE"

    print(f"Running segmented surface benchmark with {profile_name} ...")
    t0 = time.perf_counter()
    df = run_surface_benchmark(profile, verbose=True)
    elapsed = time.perf_counter() - t0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, columns=SURFACE_COLUMNS, index=False)

    print(f"\nRows written : {len(df)}")
    print(f"Total time   : {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"CSV path     : {OUT_PATH}")


if __name__ == "__main__":
    main()
