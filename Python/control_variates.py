import numpy as np
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Random Forest control variate
# =============================================================================

def rf_control_variate(
    X_pilot: np.ndarray, payoffs_pilot: np.ndarray, labels_pilot: np.ndarray,
    X_pricing: np.ndarray, labels_pricing: np.ndarray,
    p_k: np.ndarray, K: int,
    n_estimators: int = 200, max_depth: int = None,
    random_state: int = 42
) -> tuple[float, float, float]:
    """
    Local Random Forest control variate estimator.

    Trains one RF per cluster to predict payoffs from path features.
    The bias-corrected payoff for observation i in cluster k is:
        phi_i^{corrected} = phi_i - c_k* (Y_k(x_i) - mu_Y_k)
    where c_k* = Cov(phi, Y_k) / Var(Y_k).

    Returns
    -------
    estimate       : float  — price estimate
    var_ratio      : float  — Var(corrected) / Var(raw)
    variance_reduc : float  — 1 / var_ratio
    """
    MIN_CLUSTER_SIZE = 10

    forests = {}
    ctrl_means = {}

    for k in range(K):
        mask = labels_pilot == k
        if mask.sum() < MIN_CLUSTER_SIZE:
            forests[k] = None
            ctrl_means[k] = payoffs_pilot[mask].mean() if mask.sum() > 0 else 0.0
            continue
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=5, n_jobs=-1, random_state=random_state
        )
        rf.fit(X_pilot[mask], payoffs_pilot[mask])
        forests[k] = rf
        ctrl_means[k] = rf.predict(X_pilot[mask]).mean()

    all_corrected = []
    all_raw = []

    for k in range(K):
        mask_pr = labels_pricing == k
        n_pr = mask_pr.sum()
        if n_pr == 0:
            continue

        phi_pr = payoffs_pilot[labels_pilot == k]
        n_tr = len(phi_pr)
        phi_pr = phi_pr[:n_pr] if n_tr >= n_pr else phi_pr

        if len(phi_pr) == 0:
            continue

        all_raw.extend(phi_pr.tolist())

        if forests[k] is None:
            all_corrected.extend(phi_pr.tolist())
            continue

        X_pr = X_pricing[mask_pr][:len(phi_pr)]
        Y_pr = forests[k].predict(X_pr)
        mu_Y_k = ctrl_means[k]

        if len(phi_pr) >= 2 and len(Y_pr) >= 2:
            cov_kk = np.cov(phi_pr, Y_pr)[0, 1]
            var_Y = np.var(Y_pr, ddof=1)
            c_k = cov_kk / var_Y if var_Y > 1e-10 else 0.0
        else:
            c_k = 0.0

        corrected_k = phi_pr - c_k * (Y_pr - mu_Y_k)
        all_corrected.extend(corrected_k.tolist())

    corrected = np.asarray(all_corrected)
    raw = np.asarray(all_raw)

    raw_var = raw.var(ddof=1) if len(raw) >= 2 and raw.var(ddof=1) > 1e-14 else 1.0
    corr_var = corrected.var(ddof=1) if len(corrected) >= 2 else raw_var
    var_ratio = corr_var / raw_var

    return corrected.mean(), var_ratio, 1.0 / max(var_ratio, 1e-10)


# =============================================================================
# Neural Network control variate
# =============================================================================

class ControlVariateNet(nn.Module):
    """
    Feedforward network  g_theta : R^d -> R.

    Architecture: Input -> Linear(128) -> BN -> ReLU -> Dropout
                        -> Linear(64)  -> BN -> ReLU -> Dropout
                        -> Linear(64)  -> BN -> ReLU -> Dropout
                        -> Linear(1)

    Kaiming (He) initialisation for all linear layers.
    """
    def __init__(self, d_in: int, dropout_p: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(64, 64),    nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_nn_control_variate(
    X_train: np.ndarray,
    phi_train: np.ndarray,
    X_val: np.ndarray = None,
    phi_val: np.ndarray = None,
    n_epochs: int = 300,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout_p: float = 0.10,
    patience: int = 30,
    device: str = "cpu",
    verbose: bool = False,
    record_loss: bool = False,
) -> tuple:
    """
    Train ControlVariateNet using the custom variance loss:

        L(theta, c) = E[ (phi - c * g_theta(x))^2 ]

    The scalar c is updated in closed form after each epoch:
        c* = Cov(phi, g_theta) / Var(g_theta)

    Early stopping halts training if validation loss does not improve
    for `patience` consecutive epochs.

    Returns
    -------
    net         : trained ControlVariateNet (eval mode)
    c_star      : optimal scalar coefficient
    mu_g        : mean of g_theta on training set
    [train_hist, val_hist] : only when record_loss=True
    """
    X_t = torch.tensor(X_train,  dtype=torch.float32, device=device)
    phi_t = torch.tensor(phi_train, dtype=torch.float32, device=device)

    has_val = (X_val is not None) and (phi_val is not None)
    if has_val:
        X_v = torch.tensor(X_val,  dtype=torch.float32, device=device)
        phi_v = torch.tensor(phi_val, dtype=torch.float32, device=device)

    d_in = X_train.shape[1]
    net = ControlVariateNet(d_in, dropout_p=dropout_p).to(device)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5, min_lr=1e-5)

    N = X_train.shape[0]
    c_star = 1.0
    train_hist, val_hist = [], []
    best_val = np.inf
    patience_c = 0

    for epoch in range(n_epochs):
        net.train()
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
            x_b = X_t[idx]
            phi_b = phi_t[idx]

            opt.zero_grad()
            g_b = net(x_b)
            loss = ((phi_b - c_star * g_b) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_hist.append(avg_train_loss)

        net.eval()
        with torch.no_grad():
            g_all = net(X_t)
            phi_mean = phi_t.mean()
            g_mean = g_all.mean()
            num = ((phi_t - phi_mean) * (g_all - g_mean)).mean().item()
            den = ((g_all - g_mean) ** 2).mean().item()
            c_star = num / den if abs(den) > 1e-10 else c_star

        if has_val:
            with torch.no_grad():
                g_v = net(X_v)
                val_loss = ((phi_v - c_star * g_v) ** 2).mean().item()
            val_hist.append(val_loss)
            sched.step(val_loss)

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                patience_c = 0
            else:
                patience_c += 1
                if patience_c >= patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch+1}")
                    break
        else:
            sched.step(avg_train_loss)

        if verbose and (epoch + 1) % 50 == 0:
            vl_str = f"  val_loss={val_hist[-1]:.6f}" if has_val else ""
            print(f"  Epoch {epoch+1:4d}/{n_epochs}  "
                  f"train_loss={avg_train_loss:.6f}  c*={c_star:.4f}{vl_str}")

    net.eval()
    with torch.no_grad():
        mu_g = net(X_t).detach().cpu().numpy().mean()

    if record_loss:
        return net, c_star, mu_g, train_hist, val_hist
    return net, c_star, mu_g


def nn_price_estimate(
    net: nn.Module,
    X_pricing: np.ndarray,
    payoffs_pricing: np.ndarray,
    c_star: float,
    mu_g: float,
    device: str = "cpu",
) -> tuple[float, float, float, float]:
    """
    Apply the trained NN control variate to fresh pricing paths.

        phi_i^{corrected} = phi_i - c* ( g_theta(x_i) - mu_g )

    Returns
    -------
    estimate  : float — price estimate
    se        : float — standard error
    rho       : float — empirical correlation between phi and g_theta
    var_ratio : float — Var(corrected) / Var(raw)
    """
    X_t = torch.tensor(X_pricing, dtype=torch.float32, device=device)
    with torch.no_grad():
        g_vals = net(X_t).detach().cpu().numpy()

    corrected = payoffs_pricing - c_star * (g_vals - mu_g)
    estimate = corrected.mean()
    se = corrected.std(ddof=1) / np.sqrt(len(corrected))

    raw_var = payoffs_pricing.var(ddof=1)
    corr_var = corrected.var(ddof=1)
    var_ratio = corr_var / (raw_var + 1e-14)

    if raw_var > 1e-14 and g_vals.var() > 1e-14:
        rho = np.corrcoef(payoffs_pricing, g_vals)[0, 1]
    else:
        rho = 0.0

    return estimate, se, rho, var_ratio
