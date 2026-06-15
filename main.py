"""
==============================================================================
Stratified Monte Carlo Pricing and Variance Reduction
via K-Means Clustering and Machine Learning Control Variates  — ENTRY POINT
==============================================================================
Authors    : Benjamin Sulpice & Enzo Thuilliez
Supervisor : Professor Arnaud DUFAYS
Institution: EDHEC Business School / Université Côte d'Azur
Date       : June 2026 — VERSION 2 (Publication-ready)

Module structure
----------------
  config.py           — global constants and matplotlib style
  analytics.py        — Black-Scholes and Heston semi-analytic pricers
  simulation.py       — GBM / Heston Euler / Heston QE path generators
  payoffs.py          — payoff functions (European, Asian, Barrier)
  features.py         — path feature extraction and normalisation
  clustering.py       — K-Means, GMM, Neyman allocation
  estimators.py       — plain MC, antithetic, stratified estimators
  control_variates.py — RF and NN control variates
  benchmark.py        — full factorial benchmark engine
  pipeline.py         — single-config demonstration pipeline
  figures.py          — all 8 publication figures
  main.py             — entry point (parameters + orchestration)
==============================================================================
"""

import os

import config  # must be first — sets matplotlib backend before pyplot is imported

from demo import run_pipeline_demo
from benchmark import run_global_benchmark, print_summary_table
from visualization import (
    plot_figure1_trajectories,
    plot_figure2_euler_vs_qe,
    plot_figure3_clusters,
    plot_figure4_neyman_allocation,
    plot_figure5_rmse_european,
    plot_figure6_rmse_path_dependent,
    plot_figure7_nn_loss,
    plot_figure8_efficiency,
)

# =============================================================================
# Market parameters
# =============================================================================
S0, K_STRIKE = 100.0, 100.0
r, sigma, T  = 0.05, 0.20, 1.0
M            = 252          # daily steps
B_BARRIER    = 85.0         # barrier level (15 pct below spot)

# Heston parameters  (Feller: 2*kappa*theta >= xi^2  =>  0.16 >= 0.09 ✓)
V0      = 0.04   # initial variance (sigma_0 ~ 20 pct)
kappa   = 2.00   # mean-reversion speed
theta_h = 0.04   # long-run variance
xi      = 0.30   # vol-of-vol
rho_h   = -0.70  # spot-vol correlation

# Benchmark parameters
N_VALUES       = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
N_REPLICATIONS = 50     # increase to 100-200 for final publication (slow)
K_CLUSTERS     = 8

# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  STRATIFIED MC V2 — FULL PIPELINE")
    print("  Benjamin Sulpice & Enzo Thuilliez")
    print("  Supervisor: Prof. Arnaud DUFAYS")
    print("=" * 70)

    # =========================================================================
    # PART A: SINGLE-CONFIG PIPELINE DEMO (for Figures 3, 4, 7)
    # =========================================================================
    print("\n\n>>> PART A: Pipeline demonstration (GBM, European call)\n")
    demo = run_pipeline_demo(
        S0=S0, K=K_STRIKE, r=r, sigma=sigma, T=T, M=M,
        N_pilot=20_000, N_pricing=20_000, K_clusters=K_CLUSTERS,
        V0=V0, kappa=kappa, theta_h=theta_h, xi=xi, rho_h=rho_h,
    )

    # =========================================================================
    # PART B: FIGURES 1, 2 (qualitative / model comparison)
    # =========================================================================
    print("\n\n>>> PART B: Generating Figures 1–2 (trajectory and Euler vs QE)\n")

    print(">>> Generating Figure 1 (GBM vs Heston-QE trajectories)...")
    plot_figure1_trajectories(
        S0, K_STRIKE, r, sigma, T, M,
        V0, kappa, theta_h, xi, rho_h,
        N_plot=150, seed=7,
    )

    print("\n>>> Generating Figure 2 (Euler vs QE discretisation accuracy)...")
    plot_figure2_euler_vs_qe(
        S0, V0, r, kappa, theta_h, xi, rho_h, T,
        M_values=[10, 20, 50, 100, 252, 500],
        N=10_000, seed=0,
    )

    # =========================================================================
    # PART C: FIGURES 3, 4, 7 (from demo outputs)
    # =========================================================================
    print("\n\n>>> PART C: Generating Figures 3, 4, 7 (clustering, Neyman, NN)\n")

    print(">>> Generating Figure 3 (cluster scatter: K-Means vs GMM)...")
    plot_figure3_clusters(
        demo["X_pilot_raw"], demo["payoffs_pilot"],
        K_CLUSTERS, S0, K_STRIKE,
    )

    print("\n>>> Generating Figure 4 (Neyman allocation)...")
    plot_figure4_neyman_allocation(
        demo["payoffs_pilot"], demo["labels_pilot"],
        K_CLUSTERS, demo["N_pricing"],
    )

    print("\n>>> Generating Figure 7 (NN loss trajectory)...")
    plot_figure7_nn_loss(demo["loss_train"], demo["loss_val"])

    # =========================================================================
    # PART D: GLOBAL BENCHMARK (for Figures 5, 6, 8 and summary table)
    # =========================================================================
    print("\n\n>>> PART D: Running global benchmark (Figures 5, 6, 8)\n")
    print(f"  Grid: payoffs=[european_call, asian_call, barrier_doc]")
    print(f"        models =[gbm, heston_euler, heston_qe]")
    print(f"        samplers=[prng, qmc]")
    print(f"        N_values={N_VALUES}")
    print(f"        n_replications={N_REPLICATIONS}  (increase for publication)")

    df_results = run_global_benchmark(
        S0=S0, K=K_STRIKE, r=r, sigma=sigma, T=T, M=M, B=B_BARRIER,
        V0=V0, kappa=kappa, theta_h=theta_h, xi=xi, rho_h=rho_h,
        N_values=N_VALUES,
        n_replications=N_REPLICATIONS,
        K_clusters=K_CLUSTERS,
        payoff_types=["european_call", "asian_call", "barrier_doc"],
        models=["gbm", "heston_euler", "heston_qe"],
        samplers=["prng", "qmc"],
        device="cpu",
        verbose=True,
    )

    csv_path = f"{config.OUT_DIR}/benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Benchmark results saved to: {csv_path}")

    print_summary_table(df_results)

    # =========================================================================
    # PART E: FIGURES 5, 6, 8 (from benchmark results)
    # =========================================================================
    print("\n\n>>> PART E: Generating Figures 5, 6, 8 (RMSE and efficiency)\n")

    print(">>> Generating Figure 5 (RMSE: European call, GBM)...")
    plot_figure5_rmse_european(df_results)

    print("\n>>> Generating Figure 6 (RMSE: path-dependent, Heston-QE)...")
    plot_figure6_rmse_path_dependent(df_results)

    print("\n>>> Generating Figure 8 (Efficiency frontier)...")
    plot_figure8_efficiency(df_results)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ALL OUTPUTS GENERATED")
    print("=" * 70)
    outputs = [
        "figure_1_trajectories.png",
        "figure_2_heston_euler_vs_qe.png",
        "figure_3_clusters.png",
        "figure_4_neyman_allocation.png",
        "figure_5_rmse_european.png",
        "figure_6_rmse_path_dependent.png",
        "figure_7_nn_loss.png",
        "figure_8_efficiency.png",
        "benchmark_results.csv",
    ]
    for f in outputs:
        path = f"{config.OUT_DIR}/{f}"
        status = "[OK]" if os.path.isfile(path) else "[MISSING]"
        print(f"  {status}  {f}")
