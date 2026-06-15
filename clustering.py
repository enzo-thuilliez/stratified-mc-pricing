import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def cluster_paths_kmeans(
    X_scaled: np.ndarray, K: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """
    K-Means++ clustering on the normalised feature matrix.

    Returns
    -------
    labels  : (N,) cluster assignments in {0, ..., K-1}
    centers : (K, d) cluster centroids
    model   : fitted KMeans object
    """
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    return labels, km.cluster_centers_, km


def cluster_paths_gmm(
    X_scaled: np.ndarray, K: int, random_state: int = 42
) -> tuple[np.ndarray, GaussianMixture]:
    """
    Gaussian Mixture Model clustering (full covariance).
    Hard assignments via argmax of responsibilities.

    Returns
    -------
    labels : (N,) hard cluster assignments
    model  : fitted GaussianMixture object
    """
    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        n_init=3, random_state=random_state
    )
    gmm.fit(X_scaled)
    return gmm.predict(X_scaled), gmm


def neyman_allocation(
    payoffs: np.ndarray, labels: np.ndarray, K: int, N_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Neyman optimal simulation budget allocation across clusters.

        n_k* = N * (p_k * sigma_k) / sum_{j} (p_j * sigma_j)

    Clusters with < 2 paths receive sigma_k = 0 and minimum allocation of 1.
    Rounding is corrected so sum(allocation) == N_total exactly.

    Returns
    -------
    allocation : (K,) integer path counts per cluster
    p_k        : (K,) empirical cluster proportions
    sigma_k    : (K,) within-cluster payoff standard deviations
    """
    N_pilot = len(payoffs)
    p_k = np.zeros(K)
    sigma_k = np.zeros(K)

    for k in range(K):
        mask = labels == k
        n_k = mask.sum()
        p_k[k] = n_k / N_pilot
        if n_k >= 2:
            sigma_k[k] = payoffs[mask].std(ddof=1)

    denom = (p_k * sigma_k).sum()
    if denom > 1e-14:
        weights = p_k * sigma_k / denom
    else:
        weights = p_k

    allocation = np.maximum(np.round(weights * N_total).astype(int), 1)
    diff = N_total - allocation.sum()
    if diff != 0:
        allocation[np.argmax(weights)] += diff

    return allocation, p_k, sigma_k
