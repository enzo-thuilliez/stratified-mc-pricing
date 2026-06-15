import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol

from config import RNG_SEED


def simulate_gbm_prng(
    S0: float, r: float, sigma: float, T: float,
    M: int, N: int, seed: int = None
) -> tuple[np.ndarray, float]:
    """
    Simulate N GBM paths over M time steps using pseudo-random numbers.
    Exact log-Euler discretisation.

    Returns
    -------
    paths : (N, M+1) ndarray
    dt    : float
    """
    rng = np.random.default_rng(seed)
    dt = T / M
    Z = rng.standard_normal((N, M))
    log_inc = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.hstack([
        np.zeros((N, 1)),
        np.cumsum(log_inc, axis=1),
    ])
    return np.exp(log_paths), dt


def simulate_gbm_qmc(
    S0: float, r: float, sigma: float, T: float,
    M: int, N: int, seed: int = None
) -> tuple[np.ndarray, float]:
    """
    Simulate N GBM paths using a scrambled Sobol sequence (QMC).
    CDF inversion converts uniform samples to standard normals.

    Returns
    -------
    paths : (N, M+1) ndarray
    dt    : float
    """
    dt = T / M
    sampler = Sobol(d=M, scramble=True, seed=seed if seed is not None else RNG_SEED)
    n_pow2 = int(2 ** np.ceil(np.log2(max(N, 2))))
    u = sampler.random(n_pow2)[:N]
    Z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    log_inc = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.hstack([
        np.zeros((N, 1)),
        np.cumsum(log_inc, axis=1),
    ])
    return np.exp(log_paths), dt


def _heston_euler_from_randoms(
    Z1: np.ndarray, Z2: np.ndarray,
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Core Euler-Maruyama (Full Truncation) Heston simulation from pre-generated
    correlated Brownian increments.

    V_t^+ = max(V_t, 0) prevents negative variance blow-up.

    Parameters
    ----------
    Z1, Z2 : (N, M) arrays of independent standard normals

    Returns
    -------
    S_paths : (N, M+1)
    V_paths : (N, M+1)
    dt      : float
    """
    N = Z1.shape[0]
    dt = T / M
    S = np.empty((N, M + 1)); S[:, 0] = S0
    V = np.empty((N, M + 1)); V[:, 0] = V0

    ZS = Z1
    ZV = rho * Z1 + np.sqrt(1.0 - rho ** 2) * Z2

    for k in range(M):
        Vp = np.maximum(V[:, k], 0.0)
        V[:, k + 1] = (V[:, k]
                       + kappa * (theta - V[:, k]) * dt
                       + xi * np.sqrt(Vp * dt) * ZV[:, k])
        S[:, k + 1] = S[:, k] * np.exp(
            (r - 0.5 * Vp) * dt + np.sqrt(Vp * dt) * ZS[:, k]
        )
    return S, V, dt


def simulate_heston_euler_prng(
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int, N: int, seed: int = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Heston Euler-Maruyama with pseudo-random Gaussian draws."""
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((N, M))
    Z2 = rng.standard_normal((N, M))
    return _heston_euler_from_randoms(Z1, Z2, S0, V0, r, kappa, theta, xi, rho, T, M)


def simulate_heston_euler_qmc(
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int, N: int, seed: int = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Heston Euler-Maruyama with Sobol QMC draws.
    Dimensionality: 2*M (M for the asset BM, M for the variance BM).
    """
    dim = 2 * M
    n_pow2 = int(2 ** np.ceil(np.log2(max(N, 2))))
    sampler = Sobol(d=dim, scramble=True, seed=seed if seed is not None else RNG_SEED)
    u = sampler.random(n_pow2)[:N]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    Z = norm.ppf(u)
    Z1 = Z[:, :M]
    Z2 = Z[:, M:]
    return _heston_euler_from_randoms(Z1, Z2, S0, V0, r, kappa, theta, xi, rho, T, M)


def _heston_qe_from_randoms(
    U_V: np.ndarray, Z_S: np.ndarray,
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Core Quadratic Exponential (QE) scheme for the Heston variance process
    (Andersen 2008, Section 3.2.3).

    Switching criterion: psi = Var[V_{t+dt}] / (E[V_{t+dt}])^2
      psi <= 1.5 : quadratic (bell) regime
      psi >  1.5 : exponential regime

    Parameters
    ----------
    U_V : (N, M) uniform samples — for variance CDF inversion
    Z_S : (N, M) standard normals — drive log-asset

    Returns
    -------
    S_paths : (N, M+1)
    V_paths : (N, M+1)
    dt      : float
    """
    N = U_V.shape[0]
    dt = T / M
    PSI_C = 1.5

    S = np.empty((N, M + 1)); S[:, 0] = S0
    V = np.empty((N, M + 1)); V[:, 0] = V0

    e_kdt = np.exp(-kappa * dt)
    for k in range(M):
        Vk = np.maximum(V[:, k], 0.0)

        m = theta + (Vk - theta) * e_kdt
        s2 = (Vk * xi**2 * e_kdt / kappa * (1.0 - e_kdt)
              + theta * xi**2 / (2.0 * kappa) * (1.0 - e_kdt)**2)
        s2 = np.maximum(s2, 1e-14)
        psi = s2 / (m ** 2 + 1e-14)

        V_next = np.empty(N)

        mask_exp = psi > PSI_C
        if mask_exp.any():
            p_exp = (psi[mask_exp] - 1.0) / (psi[mask_exp] + 1.0)
            beta = (1.0 - p_exp) / (m[mask_exp] + 1e-14)
            u_exp = U_V[mask_exp, k]
            val_exp = np.where(
                u_exp <= p_exp,
                0.0,
                np.log((1.0 - p_exp) / np.maximum(1.0 - u_exp, 1e-14))
                / np.maximum(beta, 1e-14)
            )
            V_next[mask_exp] = val_exp

        mask_quad = ~mask_exp
        if mask_quad.any():
            b2 = np.maximum(
                2.0 / psi[mask_quad] - 1.0
                + np.sqrt(2.0 / psi[mask_quad]) * np.sqrt(2.0 / psi[mask_quad] - 1.0),
                0.0
            )
            a = m[mask_quad] / (1.0 + b2)
            z_v = norm.ppf(np.clip(U_V[mask_quad, k], 1e-10, 1 - 1e-10))
            V_next[mask_quad] = a * (np.sqrt(b2) + z_v) ** 2

        V[:, k + 1] = V_next

        Vkp1 = V[:, k + 1]
        K0 = -rho * kappa * theta / xi * dt
        K1 = (kappa * rho / xi - 0.5) * dt - rho / xi
        K2 = rho / xi
        K3 = (1.0 - rho ** 2) * dt
        log_S = (np.log(np.maximum(S[:, k], 1e-14))
                 + K0 + K1 * Vk + K2 * Vkp1
                 - 0.5 * K3 * Vk
                 + r * dt
                 + np.sqrt(K3 * Vk) * Z_S[:, k])
        S[:, k + 1] = np.exp(log_S)

    return S, V, dt


def simulate_heston_qe_prng(
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int, N: int, seed: int = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Heston QE scheme with pseudo-random numbers."""
    rng = np.random.default_rng(seed)
    U_V = rng.uniform(0.0, 1.0, (N, M))
    Z_S = rng.standard_normal((N, M))
    return _heston_qe_from_randoms(U_V, Z_S, S0, V0, r, kappa, theta, xi, rho, T, M)


def simulate_heston_qe_qmc(
    S0: float, V0: float, r: float,
    kappa: float, theta: float, xi: float, rho: float,
    T: float, M: int, N: int, seed: int = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Heston QE scheme with Sobol QMC.
    Dimensions: 2*M — first M columns for U_V, last M for Z_S.
    """
    dim = 2 * M
    n_pow2 = int(2 ** np.ceil(np.log2(max(N, 2))))
    sampler = Sobol(d=dim, scramble=True, seed=seed if seed is not None else RNG_SEED)
    u = sampler.random(n_pow2)[:N]
    u = np.clip(u, 1e-10, 1 - 1e-10)
    U_V = u[:, :M]
    Z_S = norm.ppf(u[:, M:])
    return _heston_qe_from_randoms(U_V, Z_S, S0, V0, r, kappa, theta, xi, rho, T, M)
