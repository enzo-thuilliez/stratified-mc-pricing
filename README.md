# Stratified Monte Carlo Option Pricing

Monte Carlo option pricing with variance reduction under Black-Scholes (GBM) and Heston stochastic volatility (Quadratic Exponential scheme). Methods: antithetic variates, stratified sampling via K-Means and GMM clustering with Neyman allocation, per-cluster Random Forest and neural-network control variates, quasi-Monte Carlo (Sobol) sampling.

**Authors:** Benjamin Sulpice and Enzo Thuilliez
**Supervisor:** Professor Arnaud Dufays
**Institution:** EDHEC Business School / Universite Cote d'Azur (June 2026)

## Installation

```
pip install -r requirements.txt
```

Dependencies: NumPy, SciPy, scikit-learn, PyTorch, pandas, matplotlib.

## Project structure

### Model simulation

| File | Role |
|------|------|
| `simulation.py` | GBM (log-Euler) and Heston (Euler-Maruyama full truncation, QE scheme) path generators, each with PRNG and Sobol QMC variants |
| `config.py` | Global RNG seed, matplotlib style settings, output directory |

### Payoffs

| File | Role |
|------|------|
| `payoffs.py` | Discounted payoff functions: European call/put, arithmetic Asian call, cash-or-nothing digital call, down-and-out barrier call |

### Feature extraction and clustering

| File | Role |
|------|------|
| `features.py` | Extracts a 6-feature vector per path (terminal value, path mean, realised variance, max, min/first-passage time, momentum); standardisation |
| `clustering.py` | K-Means++ and GMM clustering on normalised path features; Neyman optimal allocation across clusters |

### Control variates

| File | Role |
|------|------|
| `control_variates.py` | Per-cluster Random Forest control variate (bias-corrected); feedforward neural network control variate with train/val split |

### Estimators

| File | Role |
|------|------|
| `estimators.py` | Plain MC, antithetic variates (GBM only), stratified estimator with Neyman allocation and standard error |

### Analytical reference prices

| File | Role |
|------|------|
| `pricing.py` | Black-Scholes closed-form call/put; Heston semi-analytic call via characteristic function integration (Gil-Pelaez inversion) |

### Benchmark engine

| File | Role |
|------|------|
| `benchmark.py` | Full-factorial benchmark driver: loops over models, samplers, payoffs, sample sizes and replications; computes RMSE, variance ratios and runtimes |
| `surface_grid.py` | Experimental grid for the maturity x moneyness x payoff surface analysis; execution profiles (`QUICK_PROFILE`, `FULL_PROFILE`); deterministic per-configuration seeding |

### Orchestration

| File | Role |
|------|------|
| `main.py` | Entry point for illustrative figures (1-8) and the global benchmark |
| `generate_all.py` | End-to-end orchestrator for the segmented surface benchmark pipeline (figures 9-14, LaTeX tables, interpretation summary) |
| `run_surface.py` | Runs the maturity x moneyness x payoff benchmark grid; writes the CSV |
| `make_surface_figures.py` | Reads the benchmark CSV and renders figures 9-14 |
| `summary_tables.py` | Generates LaTeX tables (by model, payoff, maturity, moneyness) and a Markdown interpretation summary from the CSV |

### Visualization and demo

| File | Role |
|------|------|
| `visualization.py` | All plotting functions for figures 1-14 (grayscale, print-ready) |
| `demo.py` | Single-configuration pipeline demonstration (GBM European call); produces intermediate objects used by figures 3, 4 and 7 |

## Workflows

This repository produces two distinct sets of outputs. Figures 1-8 come from the illustrative workflow; figures 9-14 come from the segmented surface benchmark.

### Workflow A -- Illustrative figures (1-8)

`main.py` runs a single-configuration demo and a global benchmark across models, samplers and sample sizes, then generates figures 1-8.

```
python main.py
```

| Output | Description |
|--------|-------------|
| `figure_1_trajectories.png` | GBM vs Heston-QE sample trajectories |
| `figure_2_heston_euler_vs_qe.png` | Euler-Maruyama vs QE discretisation accuracy across step counts |
| `figure_3_clusters.png` | K-Means vs GMM cluster assignments in feature space |
| `figure_4_neyman_allocation.png` | Neyman optimal allocation across clusters |
| `figure_5_rmse_european.png` | RMSE convergence for European call under GBM |
| `figure_6_rmse_path_dependent.png` | RMSE convergence for path-dependent payoffs under Heston-QE |
| `figure_7_nn_loss.png` | Neural network control variate training/validation loss |
| `figure_8_efficiency.png` | Efficiency frontier (RMSE vs runtime) |
| `benchmark_results.csv` | Raw benchmark results for figures 5, 6 and 8 |

### Workflow B -- Segmented surface benchmark (figures 9-14)

`generate_all.py` orchestrates the segmented surface benchmark. It sweeps over 3 maturities (0.25y, 1y, 2y) x 3 moneyness levels (ITM, ATM, OTM) x multiple payoffs, models and samplers. The three pipeline steps run sequentially; the pipeline aborts on the first failure.

```
python generate_all.py          # quick profile (~10-15 min)
python generate_all.py --full   # full profile (plan for 4-12 hours)
```

Quick profile: N=2,000 paths, M=50 steps, 4 replications, PRNG only, 3 payoffs (European, Asian, digital).
Full profile: N=20,000 paths, M=252 steps, 50 replications, PRNG + QMC, 4 payoffs (adds barrier).

Pipeline steps:

1. `run_surface.py` -- runs the benchmark grid, writes `figures/benchmark_surface_results.csv`
2. `make_surface_figures.py` -- renders figures 9-14 from the CSV
3. `summary_tables.py` -- generates LaTeX tables and a Markdown interpretation summary

| Output | Description |
|--------|-------------|
| `benchmark_surface_results.csv` | Full benchmark results (one row per configuration) |
| `figure_9_heatmap_varratio.png` | Variance ratio heatmap across maturity and moneyness |
| `figure_10_heatmap_rmse.png` | RMSE heatmap across maturity and moneyness |
| `figure_11_runtime.png` | Runtime comparison across methods |
| `figure_12_rmse_vs_runtime.png` | RMSE vs runtime scatter |
| `figure_13_model_comparison.png` | GBM vs Heston-QE performance comparison |
| `figure_14_payoff_comparison.png` | Performance breakdown by payoff type |
| `tables_by_model.tex` | LaTeX table: results aggregated by model |
| `tables_by_payoff.tex` | LaTeX table: results aggregated by payoff |
| `tables_by_maturity.tex` | LaTeX table: results aggregated by maturity |
| `tables_by_moneyness.tex` | LaTeX table: results aggregated by moneyness |
| `benchmark_interpretation_summary.md` | Auto-generated interpretation of benchmark results |

All outputs are written to the `figures/` directory.

## Methods

| Method | Description |
|--------|-------------|
| Plain MC | Baseline Monte Carlo estimator |
| Antithetic variates | Variance reduction via negated Brownian increments (GBM only) |
| Stratified (K-Means++) | Cluster paths by features, then Neyman-allocate simulation budget |
| Stratified (GMM) | Same stratification logic using Gaussian Mixture clustering |
| RF control variate | Per-cluster Random Forest trained to predict payoffs from path features |
| NN control variate | Feedforward network trained on a variance-minimising loss |

## Reproducibility

All simulations use deterministic seeding. The global seed is set in `config.py` (`RNG_SEED = 42`). For the surface benchmark, per-configuration seeds are derived deterministically via `surface_grid.make_seed()`. Each row in the benchmark CSV records the seed used.
