import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import RNG_SEED
from pricing import bs_call, heston_semi_analytic_call
from simulation import (
    simulate_gbm_prng, simulate_gbm_qmc,
    simulate_heston_euler_prng, simulate_heston_euler_qmc,
    simulate_heston_qe_prng, simulate_heston_qe_qmc,
    _heston_euler_from_randoms, _heston_qe_from_randoms,
)
from payoffs import (
    payoff_european_call, payoff_european_put,
    payoff_asian_call, payoff_barrier_down_out_call, payoff_digital_call,
)
from features import extract_features, extract_features_heston, normalise_features
from clustering import cluster_paths_kmeans, cluster_paths_gmm
from estimators import mc_price, antithetic_price, stratified_price, stratified_se
from control_variates import train_nn_control_variate, nn_price_estimate
from surface_grid import (
    MATURITIES, MONEYNESS_STRIKES, MARKET_PARAMS, HESTON_PARAMS,
    moneyness_ratio, moneyness_label, make_seed,
)


def _simulate_paths(
    model: str, sampler: str,
    S0, r, sigma, T, M, N, seed,
    V0=None, kappa=None, theta_h=None, xi=None, rho_h=None
):
    """
    Dispatch simulation to the appropriate (model, sampler) pair.
    Returns (S_paths, V_paths) where V_paths is None for GBM.
    """
    if model == "gbm":
        if sampler == "qmc":
            S, dt = simulate_gbm_qmc(S0, r, sigma, T, M, N, seed=seed)
        else:
            S, dt = simulate_gbm_prng(S0, r, sigma, T, M, N, seed=seed)
        return S, None
    elif model == "heston_euler":
        if sampler == "qmc":
            S, V, dt = simulate_heston_euler_qmc(
                S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        else:
            S, V, dt = simulate_heston_euler_prng(
                S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        return S, V
    elif model == "heston_qe":
        if sampler == "qmc":
            S, V, dt = simulate_heston_qe_qmc(
                S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        else:
            S, V, dt = simulate_heston_qe_prng(
                S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        return S, V
    else:
        raise ValueError(f"Unknown model: {model}")


def _compute_payoff(payoff_type: str, S_paths, K, B, r, T) -> np.ndarray:
    """Dispatch payoff computation."""
    if payoff_type == "european_call":
        return payoff_european_call(S_paths, K, r, T)
    elif payoff_type == "european_put":
        return payoff_european_put(S_paths, K, r, T)
    elif payoff_type == "asian_call":
        return payoff_asian_call(S_paths, K, r, T)
    elif payoff_type == "barrier_doc":
        return payoff_barrier_down_out_call(S_paths, K, B, r, T)
    elif payoff_type == "digital_call":
        return payoff_digital_call(S_paths, K, r, T)
    else:
        raise ValueError(f"Unknown payoff type: {payoff_type}")


def compute_reference_price(
    payoff_type: str, model: str,
    S0, K, r, sigma, T,
    V0=None, kappa=None, theta_h=None, xi=None, rho_h=None, B=None,
    N_ref: int = 500_000, M: int = 252, seed: int = 0,
    N_control: int = 5_000, control_rel_tol: float = 0.05,
) -> float:
    """
    Compute a high-precision reference price.

    European Call + GBM    : Black-Scholes closed form
    European Call + Heston : semi-analytical (Heston 1993) if it passes a
                              sanity check, Monte Carlo with 500k paths otherwise
    All other cases        : Monte Carlo with 500k paths (QE if Heston)

    The semi-analytical Heston formula is known to suffer from branch-cut
    discontinuities in the complex log ("Heston trap"): it can return a
    finite, positive number that is nevertheless far from the true price.
    A positivity check alone does not catch this, so the analytic price is
    only accepted if (a) it is at least the discounted intrinsic value
    max(S0 - K*exp(-rT), 0) -- a model-free no-arbitrage bound for a call --
    and (b) it agrees with a quick control Monte Carlo estimate within a
    relative/statistical tolerance.
    """
    if payoff_type == "european_call":
        if model == "gbm":
            return bs_call(S0, K, r, sigma, T)
        elif model in ("heston_euler", "heston_qe"):
            intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
            p = None
            try:
                p = heston_semi_analytic_call(S0, K, r, V0, kappa, theta_h, xi, rho_h, T)
            except Exception:
                p = None
            if p is not None and np.isfinite(p) and p >= intrinsic - 1e-8:
                S_ctrl, _, _ = simulate_heston_qe_prng(
                    S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N_control,
                    seed=seed + 7_919,
                )
                phi_ctrl = payoff_european_call(S_ctrl, K, r, T)
                mc_ctrl = phi_ctrl.mean()
                se_ctrl = phi_ctrl.std(ddof=1) / np.sqrt(len(phi_ctrl))
                tol = max(control_rel_tol * abs(mc_ctrl), 6.0 * se_ctrl)
                if abs(p - mc_ctrl) <= tol:
                    return p

    if model == "gbm":
        S_ref, _ = simulate_gbm_prng(S0, r, sigma, T, M, N_ref, seed=seed)
        V_ref = None
    else:
        S_ref, V_ref, _ = simulate_heston_qe_prng(
            S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N_ref, seed=seed)

    phi_ref = _compute_payoff(payoff_type, S_ref, K, B, r, T)
    return phi_ref.mean()


def run_global_benchmark(
    S0: float = 100.0,
    K: float  = 100.0,
    r: float  = 0.05,
    sigma: float = 0.20,
    T: float  = 1.0,
    M: int    = 252,
    B: float  = 85.0,
    V0: float    = 0.04,
    kappa: float = 2.0,
    theta_h: float = 0.04,
    xi: float    = 0.30,
    rho_h: float = -0.70,
    N_values: list = None,
    n_replications: int = 100,
    K_clusters: int = 8,
    payoff_types: list = None,
    models: list = None,
    samplers: list = None,
    device: str = "cpu",
    verbose: bool = True,
    nn_epochs: int = 250,
) -> pd.DataFrame:
    """
    Full factorial benchmark of all pricing methods across the experiment grid.

    Grid axes
    ---------
    Payoff types : european_call, asian_call, barrier_doc
    Models       : gbm, heston_euler, heston_qe
    Samplers     : prng, qmc
    Methods      : plain_mc, antithetic (PRNG only), stratified_kmeans,
                   stratified_gmm, rf_cv, nn_cv
    N values     : list of simulation budgets

    Metrics per configuration: price, se, var_ratio, rmse, time_s

    Returns
    -------
    pd.DataFrame
    """
    if N_values is None:
        N_values = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
    if payoff_types is None:
        payoff_types = ["european_call", "asian_call", "barrier_doc"]
    if models is None:
        models = ["gbm", "heston_euler", "heston_qe"]
    if samplers is None:
        samplers = ["prng", "qmc"]

    heston_kwargs = dict(V0=V0, kappa=kappa, theta_h=theta_h, xi=xi, rho_h=rho_h)
    records = []
    rng_master = np.random.default_rng(RNG_SEED)

    for payoff_type in payoff_types:
        for model in models:
            for sampler in samplers:

                # Antithetic variates are restricted to PRNG only.
                # Combining antithetic (-Z) with a Sobol sequence breaks the
                # low-discrepancy structure: the combined point set {u, 1-u}
                # is not a valid scrambled Sobol net and loses the QMC
                # convergence guarantee (Owen & Tribble 2005). The two
                # techniques are therefore deliberately kept separate.
                methods = ["plain_mc", "stratified_kmeans", "stratified_gmm",
                           "rf_cv", "nn_cv"]
                if sampler == "prng":
                    methods = ["plain_mc", "antithetic",
                               "stratified_kmeans", "stratified_gmm",
                               "rf_cv", "nn_cv"]

                config_tag = f"{payoff_type} | {model} | {sampler}"
                if verbose:
                    print(f"\nConfig: {config_tag}")

                ref_price = compute_reference_price(
                    payoff_type, model, S0, K, r, sigma, T, B=B, M=M,
                    **heston_kwargs
                )
                if verbose:
                    print(f"  Reference price: {ref_price:.6f}")

                # Pre-train NN once per configuration on a fixed 50k pilot
                N_pilot_nn = 50_000
                S_nn, V_nn = _simulate_paths(
                    model, sampler, S0, r, sigma, T, M, N_pilot_nn, seed=1,
                    **heston_kwargs
                )
                phi_nn = _compute_payoff(payoff_type, S_nn, K, B, r, T)
                if V_nn is not None:
                    X_nn_raw = extract_features_heston(
                        S_nn, V_nn,
                        barrier=(B if "barrier" in payoff_type else None))
                else:
                    X_nn_raw = extract_features(
                        S_nn,
                        barrier=(B if "barrier" in payoff_type else None))

                n_tr = int(0.8 * N_pilot_nn)
                X_nn_tr_raw, X_nn_val_raw = X_nn_raw[:n_tr], X_nn_raw[n_tr:]
                phi_tr, phi_val = phi_nn[:n_tr], phi_nn[n_tr:]

                X_nn_tr_sc, X_nn_val_sc, scaler_nn = normalise_features(
                    X_nn_tr_raw, X_nn_val_raw)

                if verbose:
                    print("  Pre-training NN control variate...")

                nn_net, nn_c, nn_mu, _, _ = train_nn_control_variate(
                    X_nn_tr_sc, phi_tr,
                    X_val=X_nn_val_sc, phi_val=phi_val,
                    n_epochs=nn_epochs, batch_size=256, lr=1e-3,
                    verbose=False, record_loss=True, device=device
                )
                if verbose:
                    print(f"  NN pre-trained: c*={nn_c:.4f}")

                for N in N_values:
                    stats = {m: {"sq_err": [], "ses": [], "vars": [], "times": []}
                             for m in methods}

                    for rep in range(n_replications):
                        seed_rep = int(rng_master.integers(0, 2**31))
                        barrier_kwarg = B if "barrier" in payoff_type else None

                        t0 = time.perf_counter()
                        S_rep, V_rep = _simulate_paths(
                            model, sampler, S0, r, sigma, T, M, N, seed=seed_rep,
                            **heston_kwargs
                        )
                        phi_rep = _compute_payoff(payoff_type, S_rep, K, B, r, T)
                        sim_time = time.perf_counter() - t0

                        if V_rep is not None:
                            X_raw = extract_features_heston(S_rep, V_rep,
                                                             barrier=barrier_kwarg)
                        else:
                            X_raw = extract_features(S_rep, barrier=barrier_kwarg)
                        X_sc, _ = normalise_features(X_raw)

                        plain_var = phi_rep.var(ddof=1) if len(phi_rep) >= 2 else 1.0

                        if "plain_mc" in methods:
                            t0 = time.perf_counter()
                            est, se_ = mc_price(phi_rep)
                            elapsed = sim_time + (time.perf_counter() - t0)
                            stats["plain_mc"]["sq_err"].append((est - ref_price)**2)
                            stats["plain_mc"]["ses"].append(se_)
                            stats["plain_mc"]["vars"].append(plain_var)
                            stats["plain_mc"]["times"].append(elapsed)

                        if "antithetic" in methods:
                            t0 = time.perf_counter()
                            if "barrier" in payoff_type:
                                pf_fn = lambda p: payoff_barrier_down_out_call(p, K, B, r, T)
                            elif payoff_type == "asian_call":
                                pf_fn = lambda p: payoff_asian_call(p, K, r, T)
                            else:
                                pf_fn = lambda p: payoff_european_call(p, K, r, T)

                            if model == "gbm":
                                est_av, se_av = antithetic_price(
                                    S0, r, sigma, T, M, N, pf_fn, seed=seed_rep + 1
                                )
                            else:
                                N2 = max(N // 2, 1)
                                rng2 = np.random.default_rng(seed_rep + 1)
                                U_V_pos = rng2.uniform(0, 1, (N2, M))
                                Z_S_pos = rng2.standard_normal((N2, M))
                                U_V_neg = 1.0 - U_V_pos
                                Z_S_neg = -Z_S_pos
                                if model == "heston_qe":
                                    S_p, V_p, _ = _heston_qe_from_randoms(
                                        U_V_pos, Z_S_pos,
                                        S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
                                    S_n, V_n, _ = _heston_qe_from_randoms(
                                        U_V_neg, Z_S_neg,
                                        S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
                                else:
                                    rng3 = np.random.default_rng(seed_rep + 2)
                                    Z2_pos = rng3.standard_normal((N2, M))
                                    S_p, V_p, _ = _heston_euler_from_randoms(
                                        Z_S_pos, Z2_pos,
                                        S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
                                    S_n, V_n, _ = _heston_euler_from_randoms(
                                        Z_S_neg, -Z2_pos,
                                        S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
                                av_p = 0.5 * (pf_fn(S_p) + pf_fn(S_n))
                                est_av = av_p.mean()
                                se_av = av_p.std(ddof=1) / np.sqrt(N2)

                            av_var = se_av ** 2 * (max(N // 2, 1))
                            elapsed_av = time.perf_counter() - t0 + sim_time
                            stats["antithetic"]["sq_err"].append((est_av - ref_price)**2)
                            stats["antithetic"]["ses"].append(se_av)
                            stats["antithetic"]["vars"].append(av_var)
                            stats["antithetic"]["times"].append(elapsed_av)

                        if "stratified_kmeans" in methods:
                            t0 = time.perf_counter()
                            if N >= K_clusters * 3:
                                labels_km, _, _ = cluster_paths_kmeans(X_sc, K_clusters)
                                p_km = np.array([(labels_km == k).mean()
                                                 for k in range(K_clusters)])
                                est_km = stratified_price(phi_rep, labels_km, K_clusters, p_km)
                                se_km  = stratified_se(phi_rep, labels_km, K_clusters, p_km)
                                var_km = (se_km * np.sqrt(N)) ** 2
                            else:
                                est_km, se_km = mc_price(phi_rep)
                                var_km = plain_var
                            elapsed = time.perf_counter() - t0 + sim_time
                            stats["stratified_kmeans"]["sq_err"].append((est_km - ref_price)**2)
                            stats["stratified_kmeans"]["ses"].append(se_km)
                            stats["stratified_kmeans"]["vars"].append(var_km)
                            stats["stratified_kmeans"]["times"].append(elapsed)

                        if "stratified_gmm" in methods:
                            t0 = time.perf_counter()
                            if N >= K_clusters * 3:
                                labels_gmm, _ = cluster_paths_gmm(X_sc, K_clusters)
                                p_gmm = np.array([(labels_gmm == k).mean()
                                                  for k in range(K_clusters)])
                                est_gmm = stratified_price(phi_rep, labels_gmm, K_clusters, p_gmm)
                                se_gmm  = stratified_se(phi_rep, labels_gmm, K_clusters, p_gmm)
                                var_gmm = (se_gmm * np.sqrt(N)) ** 2
                            else:
                                est_gmm, se_gmm = mc_price(phi_rep)
                                var_gmm = plain_var
                            elapsed = time.perf_counter() - t0 + sim_time
                            stats["stratified_gmm"]["sq_err"].append((est_gmm - ref_price)**2)
                            stats["stratified_gmm"]["ses"].append(se_gmm)
                            stats["stratified_gmm"]["vars"].append(var_gmm)
                            stats["stratified_gmm"]["times"].append(elapsed)

                        if "rf_cv" in methods:
                            t0 = time.perf_counter()
                            half = max(N // 2, 5)
                            X_tr_rf  = X_sc[:half];   phi_tr_rf  = phi_rep[:half]
                            X_pr_rf  = X_sc[half:];   phi_pr_rf  = phi_rep[half:]
                            if half >= 10 and len(X_pr_rf) >= 2:
                                rf = RandomForestRegressor(
                                    n_estimators=100, min_samples_leaf=5,
                                    n_jobs=-1, random_state=seed_rep % 1000
                                )
                                rf.fit(X_tr_rf, phi_tr_rf)
                                Y_tr = rf.predict(X_tr_rf)
                                Y_pr = rf.predict(X_pr_rf)
                                cov_cv = np.cov(phi_tr_rf, Y_tr)[0, 1]
                                var_Y  = Y_tr.var(ddof=1)
                                c_rf   = cov_cv / var_Y if var_Y > 1e-10 else 0.0
                                corr_rf = phi_pr_rf - c_rf * (Y_pr - Y_tr.mean())
                                est_rf  = corr_rf.mean()
                                se_rf   = corr_rf.std(ddof=1) / np.sqrt(len(corr_rf))
                                var_rf  = corr_rf.var(ddof=1)
                            else:
                                est_rf, se_rf = mc_price(phi_rep)
                                var_rf = plain_var
                            elapsed = time.perf_counter() - t0 + sim_time
                            stats["rf_cv"]["sq_err"].append((est_rf - ref_price)**2)
                            stats["rf_cv"]["ses"].append(se_rf)
                            stats["rf_cv"]["vars"].append(var_rf)
                            stats["rf_cv"]["times"].append(elapsed)

                        if "nn_cv" in methods:
                            t0 = time.perf_counter()
                            X_sc_nn = scaler_nn.transform(X_raw)
                            est_nn, se_nn, _, vr_nn = nn_price_estimate(
                                nn_net, X_sc_nn, phi_rep, nn_c, nn_mu, device=device
                            )
                            var_nn = vr_nn * plain_var
                            elapsed = time.perf_counter() - t0 + sim_time
                            stats["nn_cv"]["sq_err"].append((est_nn - ref_price)**2)
                            stats["nn_cv"]["ses"].append(se_nn)
                            stats["nn_cv"]["vars"].append(var_nn)
                            stats["nn_cv"]["times"].append(elapsed)

                    plain_var_mean = float(np.mean(stats["plain_mc"]["vars"])) \
                        if "plain_mc" in methods else 1.0

                    for method in methods:
                        s = stats[method]
                        if not s["sq_err"]:
                            continue
                        method_var = float(np.mean(s["vars"]))
                        var_ratio = (plain_var_mean / method_var
                                     if method_var > 1e-14 else 1.0)
                        records.append({
                            "payoff":     payoff_type,
                            "model":      model,
                            "sampler":    sampler,
                            "method":     method,
                            "N":          N,
                            "price":      float(ref_price),
                            "price_mean": float(ref_price),
                            "se":         float(np.mean(s["ses"])),
                            "var_ratio":  float(var_ratio),
                            "rmse":       float(np.sqrt(np.mean(s["sq_err"]))),
                            "time_s":     float(np.mean(s["times"])),
                        })

                    if verbose:
                        for method in methods:
                            s = stats[method]
                            if s["sq_err"]:
                                rmse_val = np.sqrt(np.mean(s["sq_err"]))
                                print(f"    N={N:6d} | {method:<20s} | "
                                      f"RMSE={rmse_val:.5f} | "
                                      f"time={np.mean(s['times']):.3f}s")

    return pd.DataFrame(records)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary table grouped by (payoff, model, sampler, method) at max N."""
    N_max = df["N"].max()
    sub = df[df["N"] == N_max].copy()

    print(f"\nBenchmark summary (N = {N_max}, {len(sub)} configurations)")
    header = (f"{'Payoff':<16} {'Model':<14} {'Sampler':<8} {'Method':<22} "
              f"{'SE':>8} {'Var Ratio':>10} {'RMSE':>9} {'Time(s)':>9}")
    print(header)
    print("-" * 90)
    for _, row in sub.sort_values(["payoff", "model", "sampler", "rmse"]).iterrows():
        print(f"{row['payoff']:<16} {row['model']:<14} {row['sampler']:<8} "
              f"{row['method']:<22} "
              f"{row['se']:8.5f} {row['var_ratio']:10.2f} "
              f"{row['rmse']:9.5f} {row['time_s']:9.3f}")


# Segmented surface benchmark — one row per exact
# (payoff, model, sampler, maturity, moneyness, method) configuration.
#
# Unlike run_global_benchmark (which aggregates over a single fixed
# S0/K/T and reports one row per N value), this driver walks the full
# maturity x moneyness surface defined in surface_grid.py so that
# variance-reduction gains can be localised on the price surface.

SURFACE_COLUMNS = [
    "model", "scheme", "sampler", "payoff", "method",
    "S0", "K", "moneyness", "moneyness_label", "T", "maturity_label",
    "r", "sigma",
    "heston_kappa", "heston_theta", "heston_xi", "heston_rho", "heston_v0",
    "n_paths", "n_steps", "n_replications",
    "price_mean", "price_std", "standard_error", "variance",
    "reference_price", "bias", "rmse", "variance_reduction_ratio",
    "runtime_seconds", "efficiency_metric", "seed",
]


def _record_rep(stats: dict, method: str, est: float, se: float, var: float, elapsed: float) -> None:
    """Append one replication's outcome for `method` into the stats accumulator."""
    stats[method]["ests"].append(est)
    stats[method]["ses"].append(se)
    stats[method]["vars"].append(var)
    stats[method]["times"].append(elapsed)


def _pretrain_nn_for_config(
    payoff_type: str, model: str, sampler: str,
    S0: float, K: float, r: float, sigma: float, T: float, M: int, B,
    heston_kwargs: dict, nn_epochs: int, device: str,
    pilot_seed: int, N_pilot: int,
):
    """Pre-train one NN control variate on a pilot batch, once per configuration."""
    barrier_kwarg = B if "barrier" in payoff_type else None

    S_nn, V_nn = _simulate_paths(
        model, sampler, S0, r, sigma, T, M, N_pilot, seed=pilot_seed, **heston_kwargs
    )
    phi_nn = _compute_payoff(payoff_type, S_nn, K, B, r, T)
    if V_nn is not None:
        X_nn_raw = extract_features_heston(S_nn, V_nn, barrier=barrier_kwarg)
    else:
        X_nn_raw = extract_features(S_nn, barrier=barrier_kwarg)

    n_tr = int(0.8 * N_pilot)
    X_tr_raw, X_val_raw = X_nn_raw[:n_tr], X_nn_raw[n_tr:]
    phi_tr, phi_val = phi_nn[:n_tr], phi_nn[n_tr:]
    X_tr_sc, X_val_sc, scaler_nn = normalise_features(X_tr_raw, X_val_raw)

    nn_net, nn_c, nn_mu, _, _ = train_nn_control_variate(
        X_tr_sc, phi_tr, X_val=X_val_sc, phi_val=phi_val,
        n_epochs=nn_epochs, batch_size=256, lr=1e-3,
        verbose=False, record_loss=True, device=device,
    )
    return nn_net, nn_c, nn_mu, scaler_nn


def _method_plain_mc(phi: np.ndarray):
    est, se = mc_price(phi)
    var = phi.var(ddof=1) if len(phi) >= 2 else 1.0
    return est, se, var


def _method_antithetic(
    model: str, S0: float, K: float, r: float, sigma: float, T: float, M: int, N: int,
    B, payoff_type: str, heston_kwargs: dict, seed: int,
):
    """
    Antithetic-variates estimator, with the variance expressed on the SAME
    per-budget convention as every other method (see module note above
    run_surface_benchmark): var := Var(price_estimator) * N.

    The estimator itself only has N//2 i.i.d. pair-draws even though N
    paths are simulated, so se**2 must be scaled by N (not N//2) to be
    comparable to plain_mc / stratified / control-variate "variance"
    columns, all of which are reported per the full simulation budget N.
    """
    if "barrier" in payoff_type:
        pf_fn = lambda p: payoff_barrier_down_out_call(p, K, B, r, T)
    elif payoff_type == "asian_call":
        pf_fn = lambda p: payoff_asian_call(p, K, r, T)
    elif payoff_type == "digital_call":
        pf_fn = lambda p: payoff_digital_call(p, K, r, T)
    else:
        pf_fn = lambda p: payoff_european_call(p, K, r, T)

    V0 = heston_kwargs.get("V0")
    kappa = heston_kwargs.get("kappa")
    theta_h = heston_kwargs.get("theta_h")
    xi = heston_kwargs.get("xi")
    rho_h = heston_kwargs.get("rho_h")

    if model == "gbm":
        est_av, se_av = antithetic_price(S0, r, sigma, T, M, N, pf_fn, seed=seed)
    else:
        N2 = max(N // 2, 1)
        rng2 = np.random.default_rng(seed)
        U_V_pos = rng2.uniform(0, 1, (N2, M))
        Z_S_pos = rng2.standard_normal((N2, M))
        U_V_neg = 1.0 - U_V_pos
        Z_S_neg = -Z_S_pos
        if model == "heston_qe":
            S_p, _, _ = _heston_qe_from_randoms(
                U_V_pos, Z_S_pos, S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
            S_n, _, _ = _heston_qe_from_randoms(
                U_V_neg, Z_S_neg, S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
        else:
            rng3 = np.random.default_rng(seed + 1)
            Z2_pos = rng3.standard_normal((N2, M))
            S_p, _, _ = _heston_euler_from_randoms(
                Z_S_pos, Z2_pos, S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
            S_n, _, _ = _heston_euler_from_randoms(
                Z_S_neg, -Z2_pos, S0, V0, r, kappa, theta_h, xi, rho_h, T, M)
        av_payoffs = 0.5 * (pf_fn(S_p) + pf_fn(S_n))
        est_av, se_av = av_payoffs.mean(), av_payoffs.std(ddof=1) / np.sqrt(N2)

    av_var = se_av ** 2 * N
    return est_av, se_av, av_var


def _method_stratified(phi: np.ndarray, X_sc: np.ndarray, n_clusters: int, cluster_fn):
    if len(phi) >= n_clusters * 3:
        labels = cluster_fn(X_sc, n_clusters)
        p_k = np.array([(labels == k).mean() for k in range(n_clusters)])
        est = stratified_price(phi, labels, n_clusters, p_k)
        se = stratified_se(phi, labels, n_clusters, p_k)
        var = se ** 2 * len(phi)
    else:
        est, se = mc_price(phi)
        var = phi.var(ddof=1) if len(phi) >= 2 else 1.0
    return est, se, var


def _method_rf_cv(phi: np.ndarray, X_sc: np.ndarray, seed_rep: int):
    half = max(len(phi) // 2, 5)
    X_tr, phi_tr = X_sc[:half], phi[:half]
    X_pr, phi_pr = X_sc[half:], phi[half:]
    if half >= 10 and len(X_pr) >= 2:
        rf = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1,
            random_state=seed_rep % 1000,
        )
        rf.fit(X_tr, phi_tr)
        Y_tr = rf.predict(X_tr)
        Y_pr = rf.predict(X_pr)
        cov_cv = np.cov(phi_tr, Y_tr)[0, 1]
        var_Y = Y_tr.var(ddof=1)
        c_rf = cov_cv / var_Y if var_Y > 1e-10 else 0.0
        corrected = phi_pr - c_rf * (Y_pr - Y_tr.mean())
        est, se = corrected.mean(), corrected.std(ddof=1) / np.sqrt(len(corrected))
        var = corrected.var(ddof=1)
    else:
        est, se = mc_price(phi)
        var = phi.var(ddof=1) if len(phi) >= 2 else 1.0
    return est, se, var


def _method_nn_cv(
    phi: np.ndarray, X_raw: np.ndarray, scaler_nn, nn_net, nn_c: float, nn_mu: float,
    device: str, plain_var: float,
):
    X_sc_nn = scaler_nn.transform(X_raw)
    est, se, _, vr = nn_price_estimate(nn_net, X_sc_nn, phi, nn_c, nn_mu, device=device)
    var = vr * plain_var
    return est, se, var


def _run_single_surface_config(
    payoff_type: str, model: str, sampler: str,
    S0: float, K: float, r: float, sigma: float, T: float, M: int, N: int, B,
    heston_kwargs: dict, n_replications: int, nn_epochs: int,
    n_clusters: int, device: str, config_seed: int, n_ref: int,
    verbose: bool = False,
) -> list:
    """Run every method on the same (payoff, model, sampler, T, K) configuration."""
    barrier_kwarg = B if "barrier" in payoff_type else None

    ref_price = compute_reference_price(
        payoff_type, model, S0, K, r, sigma, T, B=B, M=M, N_ref=n_ref,
        seed=config_seed, **heston_kwargs
    )

    N_pilot_nn = int(np.clip(5 * N, 5_000, 50_000))
    nn_net, nn_c, nn_mu, scaler_nn = _pretrain_nn_for_config(
        payoff_type, model, sampler, S0, K, r, sigma, T, M, B,
        heston_kwargs, nn_epochs, device,
        pilot_seed=config_seed + 1, N_pilot=N_pilot_nn,
    )

    methods = ["plain_mc", "stratified_kmeans", "stratified_gmm", "rf_cv", "nn_cv"]
    if sampler == "prng":
        methods = ["plain_mc", "antithetic", "stratified_kmeans", "stratified_gmm",
                   "rf_cv", "nn_cv"]

    stats = {m: {"ests": [], "ses": [], "vars": [], "times": []} for m in methods}
    rng_rep = np.random.default_rng(config_seed)

    for _ in range(n_replications):
        seed_rep = int(rng_rep.integers(0, 2**31))

        t0 = time.perf_counter()
        S_rep, V_rep = _simulate_paths(
            model, sampler, S0, r, sigma, T, M, N, seed=seed_rep, **heston_kwargs
        )
        phi_rep = _compute_payoff(payoff_type, S_rep, K, B, r, T)
        sim_time = time.perf_counter() - t0

        if V_rep is not None:
            X_raw = extract_features_heston(S_rep, V_rep, barrier=barrier_kwarg)
        else:
            X_raw = extract_features(S_rep, barrier=barrier_kwarg)
        X_sc, _ = normalise_features(X_raw)

        plain_var = phi_rep.var(ddof=1) if len(phi_rep) >= 2 else 1.0

        if "plain_mc" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_plain_mc(phi_rep)
            elapsed = sim_time + (time.perf_counter() - t0)
            _record_rep(stats, "plain_mc", est, se, var, elapsed)

        if "antithetic" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_antithetic(
                model, S0, K, r, sigma, T, M, N, B, payoff_type,
                heston_kwargs, seed_rep + 1,
            )
            elapsed = time.perf_counter() - t0 + sim_time
            _record_rep(stats, "antithetic", est, se, var, elapsed)

        if "stratified_kmeans" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_stratified(
                phi_rep, X_sc, n_clusters, lambda X, k: cluster_paths_kmeans(X, k)[0])
            elapsed = time.perf_counter() - t0 + sim_time
            _record_rep(stats, "stratified_kmeans", est, se, var, elapsed)

        if "stratified_gmm" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_stratified(
                phi_rep, X_sc, n_clusters, lambda X, k: cluster_paths_gmm(X, k)[0])
            elapsed = time.perf_counter() - t0 + sim_time
            _record_rep(stats, "stratified_gmm", est, se, var, elapsed)

        if "rf_cv" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_rf_cv(phi_rep, X_sc, seed_rep)
            elapsed = time.perf_counter() - t0 + sim_time
            _record_rep(stats, "rf_cv", est, se, var, elapsed)

        if "nn_cv" in methods:
            t0 = time.perf_counter()
            est, se, var = _method_nn_cv(
                phi_rep, X_raw, scaler_nn, nn_net, nn_c, nn_mu, device, plain_var)
            elapsed = time.perf_counter() - t0 + sim_time
            _record_rep(stats, "nn_cv", est, se, var, elapsed)

    plain_var_mean = float(np.mean(stats["plain_mc"]["vars"]))

    rows = []
    for method in methods:
        s = stats[method]
        if not s["ests"]:
            continue
        ests = np.array(s["ests"])
        method_var_mean = float(np.mean(s["vars"]))
        price_mean = float(ests.mean())
        runtime_seconds = float(np.mean(s["times"]))
        rows.append({
            "model": model,
            "payoff": payoff_type,
            "method": method,
            "price_mean": price_mean,
            "price_std": float(ests.std(ddof=1)) if len(ests) >= 2 else 0.0,
            "standard_error": float(np.mean(s["ses"])),
            "variance": method_var_mean,
            "reference_price": float(ref_price),
            "bias": price_mean - float(ref_price),
            "rmse": float(np.sqrt(np.mean((ests - ref_price) ** 2))),
            "variance_reduction_ratio": (
                plain_var_mean / method_var_mean if method_var_mean > 1e-14 else 1.0
            ),
            "runtime_seconds": runtime_seconds,
            # efficiency_metric = variance * runtime_seconds: a method that
            # is both low-variance AND fast scores low (better) on this
            # metric. It is the per-method analogue of the usual
            # "work-normalised variance" (Var x cost) efficiency criterion
            # used to compare variance-reduction techniques (Glasserman,
            # Monte Carlo Methods in Financial Engineering, Ch. 4): lower
            # is better, since it costs less compute to reach the same
            # statistical precision.
            "efficiency_metric": method_var_mean * runtime_seconds,
        })

    if verbose:
        tag = f"{payoff_type} | {model} | {sampler} | T={T} | K={K}"
        print(f"  [{tag}] ref={ref_price:.5f}  "
              + "  ".join(f"{r['method']}:RMSE={r['rmse']:.5f}" for r in rows))

    return rows


def run_surface_benchmark(profile: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Segmented surface benchmark: one CSV row per exact
    (payoff, model, sampler, maturity, moneyness, method) configuration,
    averaged over `n_replications` independent replications.

    This complements run_global_benchmark, which fixes a single (S0, K, T)
    and only varies the simulation budget N. Here N, M, n_replications and
    nn_epochs are fixed per `profile` (QUICK_PROFILE / FULL_PROFILE from
    surface_grid.py) and the loop instead walks the full maturity x
    moneyness x payoff x model x sampler x method grid, so that
    variance-reduction gains can be localised on the price surface.

    Variance-ratio convention
    --------------------------
    Every method's "variance" column is the per-budget-N equivalent
    variance of its price estimator, i.e. var := Var(price_estimator) * N.
    Under this convention Var(price_estimator) = var / N for every method,
    so variance_reduction_ratio = mean(plain_mc var) / mean(method var) is
    directly comparable across methods. Antithetic variates previously
    used a different (N//2-based) convention; _method_antithetic fixes
    this so all six methods are on the same footing.

    Returns
    -------
    pd.DataFrame with exactly the columns in SURFACE_COLUMNS.
    """
    N = profile["N_paths"]
    M = profile["n_steps"]
    n_replications = profile["n_replications"]
    nn_epochs = profile["nn_epochs"]
    models = profile["models"]
    payoff_types = profile["payoffs"]
    samplers = profile["samplers"]
    maturity_keys = profile["maturities"]
    moneyness_keys = profile["moneyness"]

    S0 = MARKET_PARAMS["S0"]
    r = MARKET_PARAMS["r"]
    sigma = MARKET_PARAMS["sigma"]
    V0, kappa, theta_h, xi, rho_h = (
        HESTON_PARAMS["V0"], HESTON_PARAMS["kappa"], HESTON_PARAMS["theta"],
        HESTON_PARAMS["xi"], HESTON_PARAMS["rho"],
    )
    heston_kwargs = dict(V0=V0, kappa=kappa, theta_h=theta_h, xi=xi, rho_h=rho_h)

    # Down-and-out barrier set below S0 (mirrors run_global_benchmark's
    # default B=85 for S0=100); only used when payoff_type contains "barrier".
    B_BARRIER_FRACTION = 0.85
    N_CLUSTERS = 8
    DEVICE = "cpu"

    records = []
    n_configs = (len(payoff_types) * len(models) * len(samplers)
                 * len(maturity_keys) * len(moneyness_keys))
    config_i = 0

    for p_idx, payoff_type in enumerate(payoff_types):
        for mo_idx, model in enumerate(models):
            for sa_idx, sampler in enumerate(samplers):
                for ma_idx, maturity_key in enumerate(maturity_keys):
                    for mn_idx, moneyness_key in enumerate(moneyness_keys):
                        config_i += 1
                        K = MONEYNESS_STRIKES[moneyness_key]
                        T = MATURITIES[maturity_key]
                        B = B_BARRIER_FRACTION * S0 if "barrier" in payoff_type else None

                        config_seed = make_seed(RNG_SEED, p_idx, mo_idx, sa_idx, ma_idx, mn_idx)

                        # Reference-price MC fallback budget scales with N so
                        # QUICK_PROFILE stays fast while FULL_PROFILE keeps
                        # the original 500k high-precision reference.
                        n_ref = int(np.clip(25 * N, 50_000, 500_000))

                        if verbose:
                            print(f"\n[{config_i}/{n_configs}] {payoff_type} | {model} | "
                                  f"{sampler} | maturity={maturity_key} (T={T}) | "
                                  f"moneyness={moneyness_key} (K={K})")

                        rows = _run_single_surface_config(
                            payoff_type=payoff_type, model=model, sampler=sampler,
                            S0=S0, K=K, r=r, sigma=sigma, T=T, M=M, N=N, B=B,
                            heston_kwargs=heston_kwargs,
                            n_replications=n_replications, nn_epochs=nn_epochs,
                            n_clusters=N_CLUSTERS, device=DEVICE,
                            config_seed=config_seed, n_ref=n_ref, verbose=verbose,
                        )

                        for row in rows:
                            row.update({
                                "scheme": "QE" if model == "heston_qe" else "closed_form_or_euler_na",
                                "sampler": sampler,
                                "S0": S0, "K": K,
                                "moneyness": moneyness_ratio(S0, K),
                                "moneyness_label": moneyness_label(S0, K),
                                "T": T, "maturity_label": maturity_key,
                                "r": r, "sigma": sigma,
                                "heston_kappa": kappa if model != "gbm" else np.nan,
                                "heston_theta": theta_h if model != "gbm" else np.nan,
                                "heston_xi": xi if model != "gbm" else np.nan,
                                "heston_rho": rho_h if model != "gbm" else np.nan,
                                "heston_v0": V0 if model != "gbm" else np.nan,
                                "n_paths": N, "n_steps": M,
                                "n_replications": n_replications,
                                "seed": config_seed,
                            })
                        records.extend(rows)

    df = pd.DataFrame(records)
    ordered_cols = [c for c in SURFACE_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in SURFACE_COLUMNS]
    return df[ordered_cols + extra_cols]
