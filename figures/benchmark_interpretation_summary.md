# Benchmark Interpretation Summary

Automatically generated from `figures/benchmark_surface_results.csv` by `summary_tables.py`. All figures below are computed directly from that file.

## Variance reduction at ATM, by maturity

- At ATM, maturity `short`: **Control variate (RF)** achieves the highest mean variance reduction ratio (5207.17x) across the 6 methods compared.
- At ATM, maturity `medium`: **Control variate (RF)** achieves the highest mean variance reduction ratio (3663.73x) across the 6 methods compared.
- At ATM, maturity `long`: **Control variate (RF)** achieves the highest mean variance reduction ratio (3092.15x) across the 6 methods compared.

## Stability at long maturities (T = 2.0, `maturity_label == 'long'`)

- **Plain MC** is the most stable method at long maturity (std of variance reduction ratio = 0.00, mean = 1.00x).
- **Control variate (RF)** is the least stable at long maturity (std = 2758.61, mean = 3350.23x).

## Efficiency under Heston QE vs GBM

- **Control variate (RF)**: mean efficiency metric GBM = 0.2880, Heston QE = 0.1836 (loses 36.3% relative efficiency under Heston QE).
- **Stratified (k-means)**: mean efficiency metric GBM = 21.1366, Heston QE = 29.7016 (gains 40.5% relative efficiency under Heston QE).
- **Stratified (GMM)**: mean efficiency metric GBM = 91.3594, Heston QE = 159.3295 (gains 74.4% relative efficiency under Heston QE).
- **Plain MC**: mean efficiency metric GBM = 64.7356, Heston QE = 134.1995 (gains 107.3% relative efficiency under Heston QE).
- **Control variate (NN)**: mean efficiency metric GBM = 0.0673, Heston QE = 0.1444 (gains 114.7% relative efficiency under Heston QE).
- **Antithetic**: mean efficiency metric GBM = 26.8249, Heston QE = 80.8974 (gains 201.6% relative efficiency under Heston QE).

## Effect of path-dependency (Asian) and discontinuity (digital)

- For **Antithetic**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (17.21x vs 7.06x).
- For **Control variate (NN)**, path-dependency (Asian vs European call) yields lower mean variance reduction ratio (947.93x vs 1441.04x).
- For **Plain MC**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (1.00x vs 1.00x).
- For **Control variate (RF)**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (5592.60x vs 3783.73x).
- For **Stratified (GMM)**, path-dependency (Asian vs European call) yields lower mean variance reduction ratio (2.59x vs 3.47x).
- For **Stratified (k-means)**, path-dependency (Asian vs European call) yields lower mean variance reduction ratio (3.68x vs 4.91x).
- For **Antithetic**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.32x vs 7.06x).
- For **Control variate (NN)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (31.36x vs 1441.04x).
- For **Plain MC**, the payoff discontinuity (digital vs European call) yields higher mean variance reduction ratio (1.00x vs 1.00x).
- For **Control variate (RF)**, the payoff discontinuity (digital vs European call) yields higher mean variance reduction ratio (3991.29x vs 3783.73x).
- For **Stratified (GMM)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.24x vs 3.47x).
- For **Stratified (k-means)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.50x vs 4.91x).

## RMSE / runtime trade-off

- Methods on the RMSE/runtime Pareto frontier (no other method is both as accurate and as fast or faster): **Control variate (NN)**, **Plain MC**.
  - Control variate (NN): mean RMSE = 0.0215, mean runtime = 0.858 s.
  - Antithetic: mean RMSE = 0.0345, mean runtime = 1.243 s.
  - Stratified (GMM): mean RMSE = 0.0385, mean runtime = 3.635 s.
  - Stratified (k-means): mean RMSE = 0.0385, mean runtime = 1.191 s.
  - Plain MC: mean RMSE = 0.0385, mean runtime = 0.823 s.
  - Control variate (RF): mean RMSE = 0.0552, mean runtime = 1.414 s.

## Limits visible in the data

- **Antithetic** shows the highest relative dispersion in variance reduction ratio across configurations (coefficient of variation = 3.27), indicating unstable performance depending on the setting.
- **Stratified (GMM)** is the most computationally costly method (mean runtime = 3.635 s, vs 1.106 s averaged over all other methods, 3.3x slower), while **Plain MC** is the cheapest (mean runtime = 0.823 s). The neural-network control variate (`nn_cv`) itself has a mean runtime of 0.858 s, which is below the median across methods.
- In every configuration of the quick profile, at least one variance-reduction method outperforms plain Monte Carlo.
