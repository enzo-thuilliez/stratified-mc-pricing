# Stratified Monte Carlo Pricing with ML Variance Reduction

**Authors:** Benjamin Sulpice & Enzo Thuilliez  
**Supervisor:** Professor Arnaud DUFAYS  
**Institution:** EDHEC Business School / Université Côte d'Azur  
**Date:** June 2026

---

## Overview

This project implements and benchmarks several **variance reduction techniques** for Monte Carlo option pricing, combining classical stratification methods with machine learning control variates.

Two stochastic models are covered — **Geometric Brownian Motion (GBM)** and the **Heston stochastic volatility model** — across three option types (European, Asian, Barrier), two samplers (PRNG and QMC/Sobol), and six pricing methods.

## Methods

| Method | Description |
|---|---|
| Plain MC | Baseline Monte Carlo estimator |
| Antithetic Variates | Variance reduction via negated Brownian increments |
| Stratified (K-Means++) | Cluster-based stratification with Neyman allocation |
| Stratified (GMM) | Gaussian Mixture Model stratification |
| RF Control Variate | Local Random Forest per cluster as control variate |
| NN Control Variate | Global feedforward network trained on custom variance loss |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Run the full pipeline** (figures + benchmark, ~2h on CPU):
```bash
python main.py
```

**Run the demo only** (single config, ~15 min):
```python
from demo import run_pipeline_demo
demo = run_pipeline_demo()
```

**Regenerate figures from existing benchmark CSV:**
```python
import pandas as pd
from visualization import plot_figure5_rmse_european

df = pd.read_csv("benchmark_results.csv")
plot_figure5_rmse_european(df)
```

## Project Structure

```
├── main.py               # Entry point
├── config.py             # Global constants and plot style
├── pricing.py            # Black-Scholes and Heston semi-analytic pricers
├── simulation.py         # GBM / Heston Euler / Heston QE path simulators
├── payoffs.py            # Payoff functions (European, Asian, Barrier)
├── features.py           # Path feature extraction and normalisation
├── clustering.py         # K-Means, GMM, Neyman allocation
├── estimators.py         # Plain MC, antithetic, stratified estimators
├── control_variates.py   # Random Forest and Neural Network control variates
├── benchmark.py          # Full factorial benchmark engine
├── demo.py               # Single-configuration pipeline demonstration
├── visualization.py      # Publication-ready figures (8 total)
└── requirements.txt
```

## Results

The benchmark produces 8 publication-ready figures and a `benchmark_results.csv`:

| Figure | Content |
|---|---|
| figure_1 | GBM vs Heston-QE simulated paths |
| figure_2 | Euler-Maruyama vs Quadratic Exponential accuracy |
| figure_3 | K-Means vs GMM cluster structure |
| figure_4 | Neyman vs uniform simulation budget allocation |
| figure_5 | RMSE vs N — European call, GBM (PRNG & QMC) |
| figure_6 | RMSE vs N — Asian & Barrier, Heston-QE |
| figure_7 | Neural network variance-loss training curve |
| figure_8 | Efficiency frontier: time vs variance reduction |

## References

- Heston, S.L. (1993). *A closed-form solution for options with stochastic volatility.*
- Andersen, L. (2008). *Simple and efficient simulation of the Heston stochastic volatility model.*
- Lord, R. et al. (2010). *A comparison of biased simulation schemes for stochastic volatility models.*
- Belomestny, D. et al. (2017). *Variance reduction for Markov chains with application to MCMC.*
