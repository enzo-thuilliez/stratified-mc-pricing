import numpy as np
from sklearn.ensemble import RandomForestRegressor

from analytics import bs_call
from simulation import simulate_gbm_prng
from payoffs import payoff_european_call
from features import extract_features, normalise_features
from clustering import cluster_paths_kmeans, neyman_allocation
from estimators import stratified_price, stratified_se
from control_variates import train_nn_control_variate, nn_price_estimate


def run_pipeline_demo(
    S0: float = 100.0, K: float = 100.0,
    r: float = 0.05, sigma: float = 0.20, T: float = 1.0,
    M: int = 252, N_pilot: int = 20_000, N_pricing: int = 20_000,
    K_clusters: int = 8,
    V0: float = 0.04, kappa: float = 2.0, theta_h: float = 0.04,
    xi: float = 0.30, rho_h: float = -0.70,
) -> dict:
    """
    Single-configuration pipeline demonstration (GBM, European call).

    Runs all pricing methods, prints a summary table, and returns
    artefacts needed for downstream figure generation.

    Returns
    -------
    dict with keys:
        true_price, X_pilot_raw, labels_pilot, K_clusters,
        payoffs_pilot, loss_train, loss_val, S0, K_strike, N_pricing
    """
    print("\n" + "=" * 70)
    print("  PIPELINE DEMONSTRATION  (GBM, European Call)")
    print("=" * 70)

    true_call = bs_call(S0, K, r, sigma, T)
    print(f"\n  BS call price (analytical): {true_call:.6f}")

    # ── Step 1: Path generation ───────────────────────────────────────────────
    print("\n[Step 1] Simulating pilot and pricing paths (GBM, PRNG)...")
    paths_pilot,   _ = simulate_gbm_prng(S0, r, sigma, T, M, N_pilot,   seed=10)
    paths_pricing, _ = simulate_gbm_prng(S0, r, sigma, T, M, N_pricing, seed=20)
    phi_pilot   = payoff_european_call(paths_pilot,   K, r, T)
    phi_pricing = payoff_european_call(paths_pricing, K, r, T)
    plain_se    = phi_pricing.std(ddof=1) / np.sqrt(N_pricing)
    print(f"  Plain MC price: {phi_pricing.mean():.6f}  SE={plain_se:.6f}")

    # ── Step 2: Feature extraction & normalisation ────────────────────────────
    print("\n[Step 2] Feature extraction and normalisation...")
    X_pilot_raw   = extract_features(paths_pilot)
    X_pricing_raw = extract_features(paths_pricing)
    X_pilot_s, X_pricing_s, scaler = normalise_features(X_pilot_raw, X_pricing_raw)
    print(f"  Feature matrix: {X_pilot_s.shape}  (N_pilot x d_features)")

    # ── Step 3: Clustering + Neyman allocation ────────────────────────────────
    print(f"\n[Step 3] K-Means++ clustering (K={K_clusters})...")
    labels_pilot,   _, km = cluster_paths_kmeans(X_pilot_s, K_clusters)
    labels_pricing, _, _  = cluster_paths_kmeans(X_pricing_s, K_clusters)
    alloc, p_k, sigma_k   = neyman_allocation(phi_pilot, labels_pilot, K_clusters, N_pricing)
    print(f"  {'k':>3} | {'p_k':>8} | {'sigma_k':>9} | {'n_k*':>7}")
    print(f"  {'-'*36}")
    for k in range(K_clusters):
        print(f"  {k:3d} | {p_k[k]:8.4f} | {sigma_k[k]:9.4f} | {alloc[k]:7d}")

    strat_est = stratified_price(phi_pricing, labels_pricing, K_clusters, p_k)
    strat_se  = stratified_se(phi_pricing, labels_pricing, K_clusters, p_k)
    print(f"\n  Stratified (K-Means) estimate: {strat_est:.6f}  SE={strat_se:.6f}")

    # ── Step 4: RF control variate ────────────────────────────────────────────
    print("\n[Step 4] Random Forest control variate (global, half-half split)...")
    half = N_pilot // 2
    X_tr, X_pr = X_pilot_s[:half], X_pilot_s[half:]
    phi_tr, phi_pr = phi_pilot[:half], phi_pilot[half:]
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                               n_jobs=-1, random_state=42)
    rf.fit(X_tr, phi_tr)
    Y_tr = rf.predict(X_tr)
    Y_pr = rf.predict(X_pr)
    cov_cv = np.cov(phi_tr, Y_tr)[0, 1]
    var_Y  = Y_tr.var(ddof=1)
    c_rf   = cov_cv / var_Y if var_Y > 1e-10 else 0.0
    corr   = phi_pr - c_rf * (Y_pr - Y_tr.mean())
    rf_est = corr.mean()
    rf_vr  = corr.var(ddof=1) / (phi_pr.var(ddof=1) + 1e-14)
    print(f"  RF estimate: {rf_est:.6f}  |  Var ratio: {rf_vr:.4f}  "
          f"(reduction: {1/rf_vr:.1f}x)")

    # ── Step 5: NN control variate ────────────────────────────────────────────
    print("\n[Step 5] Neural Network control variate (with train/val split)...")
    n_tr_nn = int(0.8 * N_pilot)
    X_nn_tr = X_pilot_s[:n_tr_nn]; phi_nn_tr = phi_pilot[:n_tr_nn]
    X_nn_val = X_pilot_s[n_tr_nn:]; phi_nn_val = phi_pilot[n_tr_nn:]

    nn_net, nn_c, nn_mu, loss_train, loss_val = train_nn_control_variate(
        X_nn_tr, phi_nn_tr,
        X_val=X_nn_val, phi_val=phi_nn_val,
        n_epochs=300, batch_size=256, lr=1e-3,
        verbose=True, record_loss=True
    )
    nn_est, nn_se, nn_rho, nn_vr = nn_price_estimate(
        nn_net, X_pricing_s, phi_pricing, nn_c, nn_mu
    )
    print(f"  NN estimate: {nn_est:.6f}  SE={nn_se:.6f}")
    print(f"  Correlation rho: {nn_rho:.4f}  |  Var ratio: {nn_vr:.4f}  "
          f"(reduction: {1/nn_vr:.1f}x)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    header = f"{'Method':<28} {'Estimate':>10} {'SE':>9} {'Var Reduc.':>12}"
    print(header)
    print("-" * 70)
    for name, est, se, vr in [
        ("True (Black-Scholes)",  true_call,          None,     None),
        ("Plain MC",              phi_pricing.mean(),  plain_se,  1.0),
        ("Stratified (K-Means)",  strat_est,           strat_se, None),
        ("RF Control Variate",    rf_est,              None,      1/rf_vr),
        ("NN Control Variate",    nn_est,              nn_se,     1/nn_vr),
    ]:
        se_str = f"{se:9.6f}" if se is not None else f"{'—':>9}"
        vr_str = f"{vr:10.1f}x" if vr is not None else f"{'—':>11}"
        print(f"  {name:<26} {est:10.6f} {se_str} {vr_str}")
    print("=" * 70)

    return {
        "true_price":    true_call,
        "X_pilot_raw":   X_pilot_raw,
        "labels_pilot":  labels_pilot,
        "K_clusters":    K_clusters,
        "payoffs_pilot": phi_pilot,
        "loss_train":    loss_train,
        "loss_val":      loss_val,
        "S0":            S0,
        "K_strike":      K,
        "N_pricing":     N_pricing,
    }
