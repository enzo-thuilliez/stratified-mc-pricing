import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import RNG_SEED
from analytics import bs_call, heston_semi_analytic_call
from simulation import (
    simulate_gbm_prng, simulate_gbm_qmc,
    simulate_heston_euler_prng, simulate_heston_euler_qmc,
    simulate_heston_qe_prng, simulate_heston_qe_qmc,
    _heston_euler_from_randoms, _heston_qe_from_randoms,
)
from payoffs import (
    payoff_european_call, payoff_european_put,
    payoff_asian_call, payoff_barrier_down_out_call,
)
from features import extract_features, extract_features_heston, normalise_features
from clustering import cluster_paths_kmeans, cluster_paths_gmm
from estimators import mc_price, antithetic_price, stratified_price, stratified_se
from control_variates import train_nn_control_variate, nn_price_estimate


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
    else:
        raise ValueError(f"Unknown payoff type: {payoff_type}")


def compute_reference_price(
    payoff_type: str, model: str,
    S0, K, r, sigma, T,
    V0=None, kappa=None, theta_h=None, xi=None, rho_h=None, B=None,
    N_ref: int = 500_000, M: int = 252, seed: int = 0,
) -> float:
    """
    Compute a high-precision reference price.

    European Call + GBM    : Black-Scholes closed form
    European Call + Heston : semi-analytical COS method
    All other cases        : Monte Carlo with 500k paths (QE if Heston)
    """
    if payoff_type == "european_call":
        if model == "gbm":
            return bs_call(S0, K, r, sigma, T)
        elif model in ("heston_euler", "heston_qe"):
            try:
                p = heston_semi_analytic_call(S0, K, r, V0, kappa, theta_h, xi, rho_h, T)
                if np.isfinite(p) and p > 0:
                    return p
            except Exception:
                pass

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

                methods = ["plain_mc", "stratified_kmeans", "stratified_gmm",
                           "rf_cv", "nn_cv"]
                if sampler == "prng":
                    methods = ["plain_mc", "antithetic",
                               "stratified_kmeans", "stratified_gmm",
                               "rf_cv", "nn_cv"]

                config_tag = f"{payoff_type} | {model} | {sampler}"
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"  Config: {config_tag}")
                    print(f"{'='*70}")

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
                    n_epochs=250, batch_size=256, lr=1e-3,
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

                        # ── Plain MC ──────────────────────────────────────────
                        if "plain_mc" in methods:
                            t0 = time.perf_counter()
                            est, se_ = mc_price(phi_rep)
                            elapsed = sim_time + (time.perf_counter() - t0)
                            stats["plain_mc"]["sq_err"].append((est - ref_price)**2)
                            stats["plain_mc"]["ses"].append(se_)
                            stats["plain_mc"]["vars"].append(plain_var)
                            stats["plain_mc"]["times"].append(elapsed)

                        # ── Antithetic Variates (PRNG only) ───────────────────
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

                        # ── Stratified — K-Means ──────────────────────────────
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

                        # ── Stratified — GMM ──────────────────────────────────
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

                        # ── RF Control Variate ────────────────────────────────
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

                        # ── NN Control Variate (pre-trained) ──────────────────
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

                    # ── Aggregate over replications ───────────────────────────
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

    print("\n" + "=" * 90)
    print(f"  BENCHMARK SUMMARY  (N = {N_max}, {len(sub)} configurations)")
    print("=" * 90)
    header = (f"{'Payoff':<16} {'Model':<14} {'Sampler':<8} {'Method':<22} "
              f"{'SE':>8} {'Var Ratio':>10} {'RMSE':>9} {'Time(s)':>9}")
    print(header)
    print("-" * 90)
    for _, row in sub.sort_values(["payoff", "model", "sampler", "rmse"]).iterrows():
        print(f"{row['payoff']:<16} {row['model']:<14} {row['sampler']:<8} "
              f"{row['method']:<22} "
              f"{row['se']:8.5f} {row['var_ratio']:10.2f} "
              f"{row['rmse']:9.5f} {row['time_s']:9.3f}")
    print("=" * 90)
