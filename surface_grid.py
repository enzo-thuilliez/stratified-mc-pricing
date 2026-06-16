"""
Experimental grid for the maturity x moneyness x payoff-type surface analysis.

This module only defines constants and pure helper functions: it does not
run any simulation, benchmark, or pricing call. It is meant to be imported
by the (future) segmented benchmark driver, which will iterate over the
maturities, moneyness levels, payoffs, samplers and models defined here.
"""

from typing import Dict, List

try:
    from config import RNG_SEED
except ImportError:
    RNG_SEED = 42


# Maturity axis

MATURITIES: Dict[str, float] = {
    "short": 0.25,
    "medium": 1.0,
    "long": 2.0,
}


# Moneyness axis (reasoned on a CALL with S0 = 100)

MONEYNESS_STRIKES: Dict[str, float] = {
    "ITM": 80.0,
    "ATM": 100.0,
    "OTM": 120.0,
}

_ITM_THRESHOLD = 1.02
_OTM_THRESHOLD = 0.98


def moneyness_ratio(S0: float, K: float) -> float:
    """Moneyness ratio S0 / K."""
    return S0 / K


def moneyness_label(S0: float, K: float) -> str:
    """
    Classify a (S0, K) pair into "ITM" / "ATM" / "OTM" for a call,
    based on the moneyness ratio S0 / K:
      - ratio > 1.02 -> "ITM"
      - ratio < 0.98 -> "OTM"
      - otherwise    -> "ATM"
    """
    ratio = moneyness_ratio(S0, K)
    if ratio > _ITM_THRESHOLD:
        return "ITM"
    if ratio < _OTM_THRESHOLD:
        return "OTM"
    return "ATM"


# Base market parameters

MARKET_PARAMS: Dict[str, float] = {
    "S0": 100.0,
    "r": 0.05,
    "sigma": 0.20,
}

# Heston parameters.
# Feller condition: 2 * kappa * theta >= xi^2
# Here: 2 * 2.0 * 0.04 = 0.16 >= 0.30^2 = 0.09 -> satisfied (no need for
# reflection/absorption hacks to keep the variance process non-negative
# in a well-behaved QE scheme).
HESTON_PARAMS: Dict[str, float] = {
    "V0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "xi": 0.30,
    "rho": -0.70,
}


# Execution profiles

QUICK_PROFILE: Dict[str, object] = {
    "N_paths": 2000,
    "n_steps": 50,
    "n_replications": 8,
    "nn_epochs": 60,
    "models": ["gbm", "heston_qe"],
    "payoffs": ["european_call", "asian_call", "digital_call"],
    "samplers": ["prng"],
    "maturities": list(MATURITIES.keys()),
    "moneyness": list(MONEYNESS_STRIKES.keys()),
}

FULL_PROFILE: Dict[str, object] = {
    "N_paths": 20000,
    "n_steps": 252,
    "n_replications": 50,
    "nn_epochs": 250,
    "models": ["gbm", "heston_qe"],
    "payoffs": ["european_call", "asian_call", "barrier_doc", "digital_call"],
    "samplers": ["prng", "qmc"],
    "maturities": list(MATURITIES.keys()),
    "moneyness": list(MONEYNESS_STRIKES.keys()),
}


# Reproducible per-configuration seeding

def make_seed(base: int, *idx: int) -> int:
    """
    Deterministically derive a reproducible seed for a given configuration,
    combining a base seed (e.g. RNG_SEED) with an arbitrary tuple of
    indices (e.g. maturity index, moneyness index, model index, ...).
    """
    seed = base
    for i in idx:
        seed = (seed * 1_000_003 + int(i)) % (2**32 - 1)
    return seed
