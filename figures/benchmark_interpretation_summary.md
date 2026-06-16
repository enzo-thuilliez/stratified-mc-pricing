# Benchmark Interpretation Summary

Automatically generated from `figures/benchmark_surface_results.csv` by `summary_tables.py`. All figures below are computed directly from that file.

## Variance reduction at ATM, by maturity

- At ATM, maturity `short`: **Control variate (RF)** achieves the highest mean variance reduction ratio (448.23x) across the 6 methods compared.
- At ATM, maturity `medium`: **Control variate (RF)** achieves the highest mean variance reduction ratio (720.92x) across the 6 methods compared.
- At ATM, maturity `long`: **Control variate (NN)** achieves the highest mean variance reduction ratio (709.57x) across the 6 methods compared.

## Stability at long maturities (T = 2.0, `maturity_label == 'long'`)

- **Plain MC** is the most stable method at long maturity (std of variance reduction ratio = 0.00, mean = 1.00x).
- **Control variate (NN)** is the least stable at long maturity (std = 592.72, mean = 416.62x).

## Efficiency under Heston QE vs GBM

- **Control variate (RF)**: mean efficiency metric GBM = 0.1816, Heston QE = 0.0760 (loses 58.2% relative efficiency under Heston QE).
- **Stratified (k-means)**: mean efficiency metric GBM = 1.9711, Heston QE = 1.7532 (loses 11.1% relative efficiency under Heston QE).
- **Stratified (GMM)**: mean efficiency metric GBM = 19.5221, Heston QE = 23.2936 (gains 19.3% relative efficiency under Heston QE).
- **Control variate (NN)**: mean efficiency metric GBM = 0.0024, Heston QE = 0.0056 (gains 131.0% relative efficiency under Heston QE).
- **Plain MC**: mean efficiency metric GBM = 0.4914, Heston QE = 3.1076 (gains 532.4% relative efficiency under Heston QE).
- **Antithetic**: mean efficiency metric GBM = 0.3093, Heston QE = 2.2007 (gains 611.5% relative efficiency under Heston QE).

## Effect of path-dependency (Asian) and discontinuity (digital)

- For **Antithetic**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (17.75x vs 7.57x).
- For **Control variate (NN)**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (539.46x vs 471.10x).
- For **Plain MC**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (1.00x vs 1.00x).
- For **Control variate (RF)**, path-dependency (Asian vs European call) yields higher mean variance reduction ratio (419.92x vs 298.00x).
- For **Stratified (GMM)**, path-dependency (Asian vs European call) yields lower mean variance reduction ratio (2.08x vs 2.89x).
- For **Stratified (k-means)**, path-dependency (Asian vs European call) yields lower mean variance reduction ratio (3.75x vs 5.01x).
- For **Antithetic**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.31x vs 7.57x).
- For **Control variate (NN)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (16.38x vs 471.10x).
- For **Plain MC**, the payoff discontinuity (digital vs European call) yields higher mean variance reduction ratio (1.00x vs 1.00x).
- For **Control variate (RF)**, the payoff discontinuity (digital vs European call) yields higher mean variance reduction ratio (514.40x vs 298.00x).
- For **Stratified (GMM)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.10x vs 2.89x).
- For **Stratified (k-means)**, the payoff discontinuity (digital vs European call) yields lower mean variance reduction ratio (2.51x vs 5.01x).
- Barrier (down-and-out call) payoff is not available in the quick profile.

## RMSE / runtime trade-off

- Methods on the RMSE/runtime Pareto frontier (no other method is both as accurate and as fast or faster): **Control variate (NN)**, **Plain MC**.
  - Control variate (NN): mean RMSE = 0.0528, mean runtime = 0.021 s.
  - Antithetic: mean RMSE = 0.0905, mean runtime = 0.043 s.
  - Stratified (k-means): mean RMSE = 0.1550, mean runtime = 0.093 s.
  - Stratified (GMM): mean RMSE = 0.1550, mean runtime = 0.607 s.
  - Plain MC: mean RMSE = 0.1550, mean runtime = 0.019 s.
  - Control variate (RF): mean RMSE = 0.2227, mean runtime = 0.283 s.

## Limits visible in the data

- **Antithetic** shows the highest relative dispersion in variance reduction ratio across configurations (coefficient of variation = 3.23), indicating unstable performance depending on the setting.
- **Stratified (GMM)** is the most computationally costly method (mean runtime = 0.607 s, vs 0.092 s averaged over all other methods, 6.6x slower), while **Plain MC** is the cheapest (mean runtime = 0.019 s). The neural-network control variate (`nn_cv`) itself has a mean runtime of 0.021 s, which is below the median across methods.
- In every configuration of the quick profile, at least one variance-reduction method outperforms plain Monte Carlo.
