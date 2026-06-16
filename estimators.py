import numpy as np


def mc_price(payoffs: np.ndarray) -> tuple[float, float]:
    """Plain MC price estimate and standard error."""
    n = len(payoffs)
    return payoffs.mean(), payoffs.std(ddof=1) / np.sqrt(n)


def antithetic_price(
    S0: float, r: float, sigma: float, T: float, M: int, N: int,
    payoff_fn, seed: int = None, **payoff_kwargs
) -> tuple[float, float]:
    """
    Antithetic variates estimator for GBM.

    Draws N/2 standard normal matrices Z; uses both Z and -Z to generate
    paired paths. The antithetic estimate is (phi(Z) + phi(-Z))/2 per pair.

    Parameters
    ----------
    payoff_fn     : callable(paths, ...) -> (N/2,) array
    payoff_kwargs : additional arguments forwarded to payoff_fn

    Returns
    -------
    estimate : float
    se       : float
    """
    N2 = max(N // 2, 1)
    rng = np.random.default_rng(seed)
    dt = T / M
    Z = rng.standard_normal((N2, M))

    def _build_paths(Z_):
        log_inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_
        log_p = np.log(S0) + np.hstack([np.zeros((N2, 1)),
                                         np.cumsum(log_inc, axis=1)])
        return np.exp(log_p)

    phi_pos = payoff_fn(_build_paths(Z),  **payoff_kwargs)
    phi_neg = payoff_fn(_build_paths(-Z), **payoff_kwargs)
    av_payoffs = 0.5 * (phi_pos + phi_neg)
    return av_payoffs.mean(), av_payoffs.std(ddof=1) / np.sqrt(N2)


def stratified_price(
    payoffs: np.ndarray, labels: np.ndarray, K: int, p_k: np.ndarray
) -> float:
    """
    Stratified (cluster-weighted) MC price estimate.

        hat{V}_0 = sum_{k} p_k * mean_{i in Ck}(phi_i)

    Returns
    -------
    float : stratified price estimate
    """
    cluster_means = np.array([
        payoffs[labels == k].mean() if (labels == k).sum() > 0 else 0.0
        for k in range(K)
    ])
    return (p_k * cluster_means).sum()


def stratified_se(
    payoffs: np.ndarray, labels: np.ndarray, K: int, p_k: np.ndarray
) -> float:
    """
    Standard error of the stratified estimator:
        SE = sqrt( sum_k p_k^2 * Var_k / n_k )

    Returns
    -------
    float : standard error (0.0 if degenerate)
    """
    se2 = 0.0
    for k in range(K):
        mask = labels == k
        n_k = mask.sum()
        if n_k >= 2:
            se2 += (p_k[k] ** 2) * payoffs[mask].var(ddof=1) / n_k
    return np.sqrt(max(se2, 0.0))
