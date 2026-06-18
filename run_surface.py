"""
Entry point for the segmented maturity x moneyness x payoff surface benchmark.

    python run_surface.py --quick   # QUICK_PROFILE (sanity check, few minutes)
    python run_surface.py --full    # FULL_PROFILE (full run)

Writes one CSV row per exact configuration to
figures/benchmark_surface_results.csv (columns: see benchmark.SURFACE_COLUMNS).

Supports incremental writing and resume: if the CSV already exists and matches
the current profile, completed configurations are skipped on restart.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from benchmark import run_surface_benchmark, SURFACE_COLUMNS
from surface_grid import QUICK_PROFILE, FULL_PROFILE

OUT_PATH = os.path.join("figures", "benchmark_surface_results.csv")
META_PATH = OUT_PATH + ".meta"

CONFIG_KEY_COLS = ("model", "sampler", "payoff", "K", "T")


def _profile_fingerprint(profile: dict) -> str:
    stable = json.dumps(profile, sort_keys=True, default=str)
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def _load_done_configs(csv_path: str, expected_fingerprint: str):
    """Load already-completed config keys from an existing CSV.

    Returns (done_keys, is_compatible).
    - done_keys: set of (model, sampler, payoff, K, T) tuples
    - is_compatible: True if the meta fingerprint matches the current profile
    """
    if not os.path.isfile(csv_path):
        return set(), True

    if os.path.isfile(META_PATH):
        with open(META_PATH, "r") as f:
            saved_fp = f.read().strip()
        if saved_fp != expected_fingerprint:
            return set(), False

    done = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (row["model"], row["sampler"], row["payoff"],
                       float(row["K"]), float(row["T"]))
                done.add(key)
            except (KeyError, ValueError):
                continue
    return done, True


def _open_csv_writer(csv_path: str, write_header: bool):
    """Open the CSV in append mode and return (file_handle, csv_writer)."""
    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=SURFACE_COLUMNS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        f.flush()
    return f, writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmented pricing-surface benchmark.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_true", help="Run QUICK_PROFILE.")
    group.add_argument("--full", action="store_true", help="Run FULL_PROFILE.")
    args = parser.parse_args()

    if args.quick:
        profile = {**QUICK_PROFILE, "n_replications": 4}
        profile_name = "QUICK_PROFILE (n_replications overridden to 4)"
    else:
        profile = FULL_PROFILE
        profile_name = "FULL_PROFILE"

    fingerprint = _profile_fingerprint(profile)

    done_configs, compatible = _load_done_configs(OUT_PATH, fingerprint)
    if not compatible:
        print(f"WARNING: existing CSV was produced by a different profile.")
        print(f"  Existing meta fingerprint does not match current profile ({fingerprint}).")
        print(f"  The existing CSV will be overwritten.")
        if os.path.isfile(OUT_PATH):
            os.remove(OUT_PATH)
        if os.path.isfile(META_PATH):
            os.remove(META_PATH)
        done_configs = set()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    need_header = not os.path.isfile(OUT_PATH) or os.path.getsize(OUT_PATH) == 0
    csv_file, csv_writer = _open_csv_writer(OUT_PATH, need_header)

    with open(META_PATH, "w") as mf:
        mf.write(fingerprint)

    rows_written = 0

    def on_config_done(rows):
        nonlocal rows_written
        for row in rows:
            csv_writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())
        rows_written += len(rows)

    if done_configs:
        print(f"Resuming: {len(done_configs)} config group(s) already done, will be skipped.")

    print(f"Running segmented surface benchmark with {profile_name} ...")
    t0 = time.perf_counter()

    try:
        df = run_surface_benchmark(
            profile, verbose=True,
            on_config_done=on_config_done,
            skip_configs=done_configs,
        )
    except KeyboardInterrupt:
        csv_file.flush()
        os.fsync(csv_file.fileno())
        csv_file.close()
        print(f"\n\nInterrupted! {rows_written} new rows saved to {OUT_PATH}.")
        print("Re-run the same command to resume from where you left off.")
        sys.exit(1)
    finally:
        if not csv_file.closed:
            csv_file.close()

    elapsed = time.perf_counter() - t0

    print(f"\nNew rows written : {rows_written}")
    print(f"Total rows in CSV: {len(done_configs) * 6 + rows_written} (approx)")
    print(f"Total time       : {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"CSV path         : {OUT_PATH}")


if __name__ == "__main__":
    main()
