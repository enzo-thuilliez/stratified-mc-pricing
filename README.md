# Stratified Monte Carlo Pricing with ML Variance Reduction

**Authors:** Benjamin Sulpice & Enzo Thuilliez  
**Supervisor:** Professor Arnaud DUFAYS  
**Institution:** EDHEC Business School / Université Côte d'Azur  
**Date:** June 2026

---

## Overview

This project implements and benchmarks variance reduction techniques for Monte Carlo option pricing under two stochastic models — Black-Scholes (GBM) and Heston stochastic volatility (Quadratic Exponential scheme). The methods combine classical estimators (antithetic variates, stratification) with machine learning control variates (Random Forest, neural network).

A segmented empirical study evaluates performance across a grid of maturities, moneyness levels, and payoff types, using both pseudo-random (PRNG) and quasi-random (QMC/Sobol) samplers.

## Methods

| Method | Description |
|---|---|
| Plain MC | Baseline Monte Carlo estimator |
| Antithetic Variates | Variance reduction via negated Brownian increments |
| Stratified (k-means++) | Cluster-based stratification with Neyman allocation |
| Stratified (GMM) | Gaussian Mixture Model stratification |
| RF Control Variate | Local Random Forest per cluster as control variate |
| NN Control Variate | Global feedforward network trained on a custom variance loss |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

`generate_all.py` is the single entry point for the full pipeline. It runs the three stages in sequence and aborts on the first failure so a partial CSV is never fed to downstream steps.

**Quick profile** (sanity check, ~10–15 min):
```bash
python generate_all.py
```

**Full profile** (50 replications, PRNG + QMC, plan for several hours):
```bash
python generate_all.py --full
```

Expected outputs (all written to `figures/`):

| Output | Description |
|---|---|
| `benchmark_surface_results.csv` | Raw results from the benchmark grid, one row per configuration |
| `figure_9_heatmap_varratio.png` | Variance-ratio heatmap across the surface |
| `figure_10_heatmap_rmse.png` | RMSE heatmap across the surface |
| `figure_11_runtime.png` | Runtime barplot by method |
| `figure_12_rmse_vs_runtime.png` | RMSE vs runtime scatter |
| `figure_13_model_comparison.png` | GBM vs Heston-QE comparison |
| `figure_14_payoff_comparison.png` | Payoff-type comparison |
| `tables_by_model.tex` | LaTeX table aggregated by model |
| `tables_by_payoff.tex` | LaTeX table aggregated by payoff type |
| `tables_by_maturity.tex` | LaTeX table aggregated by maturity |
| `tables_by_moneyness.tex` | LaTeX table aggregated by moneyness |
| `benchmark_interpretation_summary.md` | Interpretive summary of the results |

## Project Structure

```
├── generate_all.py        # Pipeline orchestrator (recommended entry point)
├── run_surface.py         # Segmented surface benchmark; writes benchmark_surface_results.csv
├── make_surface_figures.py# Generates figures 9–14 from the CSV
├── summary_tables.py      # Generates 4 LaTeX tables + Markdown interpretation summary
├── surface_grid.py        # Configuration grid: maturity/moneyness/payoff axes, QUICK/FULL profiles
│
├── config.py              # Global constants and plot style
├── simulation.py          # GBM / Heston Euler / Heston QE path simulators
├── payoffs.py             # Payoff functions (European call, Asian call, Barrier, Digital call)
├── features.py            # Path feature extraction and normalisation
├── clustering.py          # K-Means, GMM, Neyman allocation
├── estimators.py          # Plain MC, antithetic, stratified estimators
├── control_variates.py    # Random Forest and Neural Network control variates
├── benchmark.py           # Benchmark engine (used by run_surface.py)
├── visualization.py       # All figure-rendering functions (figures 1–14)
│
├── main.py                # Figures 1–8 and benchmark_results.csv (standalone run)
├── demo.py                # Single-configuration pipeline demonstration
└── requirements.txt
```

## Reproducibility

Random seeds are fixed for every configuration and recorded in the `seed` column of `benchmark_surface_results.csv`. Re-running `generate_all.py` with the same profile produces identical results.

## References

- Heston, S.L. (1993). *A closed-form solution for options with stochastic volatility.*
- Andersen, L. (2008). *Simple and efficient simulation of the Heston stochastic volatility model.*
- Lord, R. et al. (2010). *A comparison of biased simulation schemes for stochastic volatility models.*
- Belomestny, D. et al. (2017). *Variance reduction for Markov chains with application to MCMC.*
