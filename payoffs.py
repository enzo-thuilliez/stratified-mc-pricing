import numpy as np


def payoff_european_call(paths: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    """Discounted European call payoff: e^{-rT} max(S_T - K, 0)."""
    return np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)


def payoff_european_put(paths: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    """Discounted European put payoff: e^{-rT} max(K - S_T, 0)."""
    return np.exp(-r * T) * np.maximum(K - paths[:, -1], 0.0)


def payoff_asian_call(paths: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    """
    Discounted arithmetic Asian call payoff:
        e^{-rT} max( (1/M) sum_{j=1}^{M} S_{jdt} - K, 0 )
    paths[:, 0] = S_0 is excluded from the average.
    """
    avg = paths[:, 1:].mean(axis=1)
    return np.exp(-r * T) * np.maximum(avg - K, 0.0)


def payoff_digital_call(
    paths: np.ndarray, K: float, r: float, T: float, cash: float = 1.0
) -> np.ndarray:
    """
    Discounted cash-or-nothing digital call payoff:
        e^{-rT} * cash * 1{S_T > K}

    Role in the payoff taxonomy: this is the only payoff here that is
    purely discontinuous and NOT path-dependent (it depends solely on
    S_T). It isolates the effect of the terminal discontinuity on
    variance-reduction methods, separately from path-dependency, which
    the barrier payoff conflates with discontinuity.
    """
    return np.exp(-r * T) * cash * (paths[:, -1] > K).astype(float)


def payoff_barrier_down_out_call(
    paths: np.ndarray, K: float, B: float, r: float, T: float
) -> np.ndarray:
    """
    Discounted down-and-out call payoff.
    Knocked out (payoff = 0) if min_{j=1,...,M} S_{jdt} <= B.
    """
    min_path = paths[:, 1:].min(axis=1)
    alive = (min_path > B).astype(float)
    return np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0) * alive
