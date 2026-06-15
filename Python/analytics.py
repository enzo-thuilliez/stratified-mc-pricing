import numpy as np
from scipy.stats import norm


def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes closed-form price for a European call option."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes closed-form price for a European put option (put-call parity)."""
    return bs_call(S0, K, r, sigma, T) - S0 + K * np.exp(-r * T)


def heston_semi_analytic_call(
    S0: float, K: float, r: float,
    V0: float, kappa: float, theta: float, xi: float, rho: float,
    T: float,
    n_integration: int = 200,
) -> float:
    """
    Semi-analytical Heston call price via characteristic function integration
    (Heston 1993, Gil-Pelaez inversion).

        C = S0 * P1 - K * exp(-rT) * P2

    P1 and P2 are risk-neutral probabilities obtained by integrating the
    characteristic function along the real axis (Gauss-Legendre quadrature).
    """
    def char_func(phi: complex, j: int) -> complex:
        if j == 1:
            u = 0.5
            b = kappa - rho * xi
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        x = np.log(S0 / K)
        d = np.sqrt((rho * xi * phi * 1j - b) ** 2
                    - xi ** 2 * (2 * u * phi * 1j - phi ** 2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)

        denom = 1.0 - g * np.exp(d * T)
        with np.errstate(divide="ignore", invalid="ignore"):
            C_ = (r * phi * 1j * T
                  + (a / xi ** 2) * ((b - rho * xi * phi * 1j + d) * T
                                     - 2.0 * np.log(denom / (1.0 - g))))
            D_ = ((b - rho * xi * phi * 1j + d) / xi ** 2
                  * (1.0 - np.exp(d * T)) / denom)
        return np.exp(C_ + D_ * V0 + 1j * phi * x)

    def integrand(phi: float, j: int) -> float:
        return np.real(np.exp(-1j * phi * np.log(K))
                       * char_func(phi, j) / (1j * phi))

    nodes, weights = np.polynomial.legendre.leggauss(n_integration)
    a_int, b_int = 0.0, 500.0
    phi_vals = 0.5 * (b_int - a_int) * nodes + 0.5 * (b_int + a_int)
    dw = 0.5 * (b_int - a_int) * weights

    P1 = 0.5 + (1.0 / np.pi) * np.sum(dw * np.array([integrand(p, 1) for p in phi_vals]))
    P2 = 0.5 + (1.0 / np.pi) * np.sum(dw * np.array([integrand(p, 2) for p in phi_vals]))

    return S0 * P1 - K * np.exp(-r * T) * P2
