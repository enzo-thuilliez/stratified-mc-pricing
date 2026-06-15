import numpy as np
from sklearn.preprocessing import StandardScaler


def extract_features(paths: np.ndarray, barrier: float = None) -> np.ndarray:
    """
    Extract a (N, 6) feature matrix from simulated price paths.

    Feature index  Description
    -------------  -----------------------------------------------------------
    0              Terminal value S_T
    1              Arithmetic mean of path (S_{dt}, ..., S_T)
    2              Annualised realised variance
    3              Path maximum
    4              Path minimum  (or normalised first-passage time if barrier given)
    5              Signed log-return of second half vs first half (momentum proxy)

    Parameters
    ----------
    paths   : (N, M+1) price paths
    barrier : if given, feature 4 is replaced by normalised first-passage time

    Returns
    -------
    X : (N, 6) float64 feature matrix
    """
    N, Mp1 = paths.shape
    M = Mp1 - 1

    terminal = paths[:, -1]
    running_avg = paths[:, 1:].mean(axis=1)
    log_ret = np.diff(np.log(np.maximum(paths, 1e-14)), axis=1)
    realised_var = log_ret.var(axis=1, ddof=1) * M
    path_max = paths[:, 1:].max(axis=1)
    path_min = paths[:, 1:].min(axis=1)

    half = M // 2
    momentum = (np.log(np.maximum(paths[:, -1], 1e-14))
                - np.log(np.maximum(paths[:, half], 1e-14)))

    if barrier is not None:
        crossing = paths[:, 1:] <= barrier
        first_hit = np.argmax(crossing, axis=1).astype(float)
        never_hit = ~crossing.any(axis=1)
        first_hit[never_hit] = float(M)
        feat4 = first_hit / M
    else:
        feat4 = path_min

    return np.column_stack([terminal, running_avg, realised_var, path_max, feat4, momentum])


def extract_features_heston(
    S_paths: np.ndarray, V_paths: np.ndarray, barrier: float = None
) -> np.ndarray:
    """
    Extended 8-dimensional feature matrix for Heston paths.
    Adds mean variance and terminal variance to the standard 6 price features.

    Returns
    -------
    X : (N, 8) float64 feature matrix
    """
    X_base = extract_features(S_paths, barrier=barrier)
    V_mean = V_paths[:, 1:].mean(axis=1).reshape(-1, 1)
    V_terminal = V_paths[:, -1].reshape(-1, 1)
    return np.hstack([X_base, V_mean, V_terminal])


def normalise_features(
    X_train: np.ndarray, X_test: np.ndarray = None
) -> tuple:
    """
    Fit a StandardScaler on X_train and optionally transform X_test.

    Returns
    -------
    X_train_scaled, [X_test_scaled], scaler
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    if X_test is not None:
        return X_tr, scaler.transform(X_test), scaler
    return X_tr, scaler
