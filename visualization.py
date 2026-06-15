import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import OUT_DIR, GRAY_SHADES, CLUSTER_MARKERS, LINE_STYLES
from simulation import (
    simulate_gbm_prng,
    simulate_heston_qe_prng,
    simulate_heston_euler_prng,
)
from payoffs import payoff_european_call
from clustering import cluster_paths_kmeans, cluster_paths_gmm


def _method_label(method: str) -> str:
    """Convert internal method key to readable label for plots."""
    labels = {
        "plain_mc":          "Plain MC",
        "antithetic":        "Antithetic variates",
        "stratified_kmeans": "Stratified (K-Means)",
        "stratified_gmm":    "Stratified (GMM)",
        "rf_cv":             "RF control variate",
        "nn_cv":             "NN control variate",
    }
    return labels.get(method, method)


def plot_figure1_trajectories(
    S0, K, r, sigma, T, M,
    V0, kappa, theta_h, xi, rho_h,
    N_plot: int = 150, seed: int = 7,
    outpath: str = None,
) -> None:
    """
    Figure 1: GBM vs Heston-QE simulated paths.
    2×2 layout: GBM paths | Heston paths | GBM vol | Heston variance paths.
    """
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_1_trajectories.png"

    paths_gbm, _          = simulate_gbm_prng(S0, r, sigma, T, M, N_plot, seed=seed)
    S_heston, V_heston, _ = simulate_heston_qe_prng(
        S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N_plot, seed=seed)

    time_grid  = np.linspace(0, T, M + 1)
    ST_gbm     = paths_gbm[:, -1]
    realised_V = V_heston[:, 1:].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    lw = 0.35

    ax = axes[0, 0]
    itm = ST_gbm > K
    for i in range(N_plot):
        ax.plot(time_grid, paths_gbm[i], color="0.10" if itm[i] else "0.72",
                lw=lw, alpha=0.55)
    ax.axhline(K, color="black", lw=1.1, ls="--", zorder=5)
    ax.set_title(r"(a) GBM paths — coloured by moneyness $S_T \gtrless K$", fontsize=9)
    ax.set_xlabel("Time $t$"); ax.set_ylabel("$S_t$")
    ax.text(0.97, 0.04, "Dark: ITM  |  Light: OTM",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")

    ax = axes[0, 1]
    v_med = np.median(realised_V)
    for i in range(N_plot):
        ax.plot(time_grid, S_heston[i],
                color="0.10" if realised_V[i] > v_med else "0.72", lw=lw, alpha=0.55)
    ax.axhline(K, color="black", lw=1.1, ls="--", zorder=5)
    ax.set_title(r"(b) Heston-QE paths — coloured by mean realised variance", fontsize=9)
    ax.set_xlabel("Time $t$"); ax.set_ylabel("$S_t$")
    ax.text(0.97, 0.04, "Dark: high vol  |  Light: low vol",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.4")

    ax = axes[1, 0]
    ax.plot(time_grid, np.full(M + 1, sigma), color="0.0", lw=1.4, ls="--",
            label=r"Constant $\sigma$")
    ax.set_ylim(0, max(sigma * 2.5, 0.05))
    ax.set_title(r"(c) GBM — constant instantaneous volatility $\sigma$", fontsize=9)
    ax.set_xlabel("Time $t$"); ax.set_ylabel(r"$\sigma_t = \sigma$")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    n_show = min(50, N_plot)
    for i in range(n_show):
        ax.plot(time_grid, V_heston[i],
                color="0.10" if realised_V[i] > v_med else "0.72", lw=lw, alpha=0.70)
    ax.axhline(theta_h, color="black", lw=1.1, ls="-.",
               label=r"Long-run variance $\bar{v}$")
    ax.axhline(V0, color="0.4", lw=0.9, ls=":", label=r"Initial variance $V_0$")
    ax.set_title(r"(d) Heston-QE — stochastic variance paths $V_t$", fontsize=9)
    ax.set_xlabel("Time $t$"); ax.set_ylabel("$V_t$")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        f"Figure 1 — GBM vs Heston-QE simulated paths  "
        f"($S_0={S0}$, $K={K}$, $r={r}$, $\\sigma={sigma}$, "
        f"$T={T}$, $N={N_plot}$)",
        fontsize=9, y=0.97
    )
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure2_euler_vs_qe(
    S0, V0, r, kappa, theta_h, xi, rho_h, T,
    M_values: list = None, N: int = 10_000, seed: int = 0,
    outpath: str = None,
) -> None:
    """
    Figure 2: Heston Euler vs QE discretisation accuracy.
    Left: bias vs M. Right: distribution of V_T.
    """
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_2_heston_euler_vs_qe.png"
    if M_values is None:
        M_values = [10, 20, 50, 100, 252, 500]

    K = S0

    S_ref, _, _ = simulate_heston_qe_prng(
        S0, V0, r, kappa, theta_h, xi, rho_h, T, 1000, 100_000, seed=seed)
    ref_price = payoff_european_call(S_ref, K, r, T).mean()
    print(f"  Heston reference price (QE, M=1000, 100k paths): {ref_price:.6f}")

    bias_euler, bias_qe = [], []
    for M in M_values:
        S_e, _, _ = simulate_heston_euler_prng(
            S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        S_q, _, _ = simulate_heston_qe_prng(
            S0, V0, r, kappa, theta_h, xi, rho_h, T, M, N, seed=seed)
        bias_euler.append(abs(payoff_european_call(S_e, K, r, T).mean() - ref_price))
        bias_qe.append(abs(payoff_european_call(S_q, K, r, T).mean() - ref_price))

    M_dist = 52
    _, V_euler_dist, _ = simulate_heston_euler_prng(
        S0, V0, r, kappa, theta_h, xi, rho_h, T, M_dist, N, seed=seed)
    _, V_qe_dist, _ = simulate_heston_qe_prng(
        S0, V0, r, kappa, theta_h, xi, rho_h, T, M_dist, N, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    Mf = np.array(M_values, dtype=float)
    ax.loglog(Mf, bias_euler, color="0.0", ls="--", lw=1.3,
              marker="s", ms=5, mfc="white", mew=0.9, label="Euler (full trunc.)")
    ax.loglog(Mf, bias_qe,   color="0.0", ls="-",  lw=1.3,
              marker="o", ms=5, mfc="white", mew=0.9, label="Quadratic Exponential (QE)")
    ax.set_xlabel("Number of time steps $M$", labelpad=5)
    ax.set_ylabel("Absolute bias vs. reference (log scale)", labelpad=5)
    ax.set_title("(a) Call price bias: Euler vs QE", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    v_euler_T = V_euler_dist[:, -1]
    v_qe_T    = V_qe_dist[:, -1]
    bins = np.linspace(0, np.percentile(np.concatenate([v_euler_T, v_qe_T]), 99), 60)
    ax.hist(v_euler_T, bins=bins, density=True, histtype="step",
            color="0.0", lw=1.2, ls="--", label=f"Euler ($M={M_dist}$)")
    ax.hist(v_qe_T, bins=bins, density=True, histtype="step",
            color="0.0", lw=1.2, ls="-",  label=f"QE ($M={M_dist}$)")
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.axvline(theta_h, color="0.0", lw=0.8, ls="-.", label=r"$\bar{v}$ (long-run)")
    ax.set_xlabel("Terminal variance $V_T$", labelpad=5)
    ax.set_ylabel("Density", labelpad=5)
    ax.set_title(f"(b) Distribution of $V_T$ — Euler vs QE ($M={M_dist}$, $N={N}$)",
                 fontsize=9)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Figure 2 — Heston discretisation accuracy: Euler-Maruyama vs "
                 "Quadratic Exponential", fontsize=9, y=0.98)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure3_clusters(
    X_pilot_raw: np.ndarray, payoffs_pilot: np.ndarray,
    K: int, S0: float, K_strike: float,
    outpath: str = None,
) -> None:
    """
    Figure 3: 2-D cluster structure in the (S_T, mean-S) feature space.
    Left: K-Means++. Right: GMM.
    """
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_3_clusters.png"

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_pilot_raw)

    labels_km,  _, _ = cluster_paths_kmeans(X_sc, K)
    labels_gmm, _    = cluster_paths_gmm(X_sc, K)

    terminal    = X_pilot_raw[:, 0]
    running_avg = X_pilot_raw[:, 1]
    n_show = min(len(terminal), 3000)
    idx = np.random.default_rng(42).choice(len(terminal), n_show, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, labels, title in zip(
        axes,
        [labels_km, labels_gmm],
        ["(a) K-Means++ clustering", "(b) GMM clustering (full covariance)"]
    ):
        legend_handles = []
        for k in range(K):
            mask = labels[idx] == k
            n_k  = mask.sum()
            if n_k == 0:
                continue
            shade  = GRAY_SHADES[k % len(GRAY_SHADES)]
            marker = CLUSTER_MARKERS[k % len(CLUSTER_MARKERS)]
            ax.scatter(terminal[idx][mask], running_avg[idx][mask],
                       c=shade, marker=marker, s=12, alpha=0.65, edgecolors="none")
            legend_handles.append(
                mpatches.Patch(facecolor=shade,
                               label=f"Cluster {k}  (n={int((labels == k).sum())})")
            )

        ax.axvline(K_strike, color="black", lw=1.0, ls="--", zorder=5)
        legend_handles.append(
            plt.Line2D([0], [0], color="black", ls="--", lw=1.0,
                       label=f"Strike $K={K_strike}$")
        )
        ax.set_xlabel("Terminal asset value $S_T$", labelpad=5)
        ax.set_ylabel(r"Path arithmetic mean $\bar{S}$", labelpad=5)
        ax.set_title(title, fontsize=9)
        ax.legend(handles=legend_handles, fontsize=7, loc="upper left",
                  ncol=2, handlelength=1.0, handletextpad=0.4, columnspacing=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        "Figure 3 — Cluster structure in the $(S_T, \\bar{S})$ feature space: "
        "K-Means vs GMM",
        fontsize=9, y=0.98
    )
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure4_neyman_allocation(
    payoffs_pilot: np.ndarray, labels_pilot: np.ndarray,
    K: int, N_total: int,
    outpath: str = None,
) -> None:
    """
    Figure 4: Neyman optimal budget vs uniform allocation (bar chart).
    Secondary axis shows within-cluster standard deviation sigma_k.
    """
    from clustering import neyman_allocation

    if outpath is None:
        outpath = f"{OUT_DIR}/figure_4_neyman_allocation.png"

    alloc, p_k, sigma_k = neyman_allocation(payoffs_pilot, labels_pilot, K, N_total)
    uniform_alloc = np.full(K, N_total / K)

    x = np.arange(K)
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width / 2, uniform_alloc, width,
            color="white", edgecolor="0.0", lw=0.9, label="Uniform allocation $N/K$")
    ax1.bar(x + width / 2, alloc, width,
            color="0.55", edgecolor="0.0", lw=0.9, label="Neyman allocation $n_k^*$")
    ax1.set_xlabel("Cluster index $k$", labelpad=5)
    ax1.set_ylabel("Number of paths allocated", labelpad=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$k={k}$\n$(p_k={p_k[k]:.2f})$" for k in range(K)], fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, sigma_k, color="0.0", ls=":", lw=1.4,
             marker="D", ms=5, mfc="white", mew=0.9,
             label=r"Within-cluster std $\hat{\sigma}_k$")
    ax2.set_ylabel(r"Payoff std $\hat{\sigma}_k$", labelpad=5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    for k in range(K):
        ratio = alloc[k] / uniform_alloc[k]
        ax1.text(k + width / 2, alloc[k] + max(N_total * 0.005, 5),
                 f"{ratio:.1f}x", ha="center", va="bottom", fontsize=7.5)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax1.set_title(
        f"Figure 4 — Neyman vs uniform simulation budget allocation ($K={K}$, "
        f"$N_{{total}}={N_total}$)",
        pad=10, fontsize=9
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure5_rmse_european(df: pd.DataFrame, outpath: str = None) -> None:
    """Figure 5: Log-log RMSE vs N for the European call under GBM (PRNG and QMC)."""
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_5_rmse_european.png"

    sub = df[(df["payoff"] == "european_call") & (df["model"] == "gbm")]
    if sub.empty:
        print("  [Fig 5] No data found. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, sampler_name in zip(axes, ["prng", "qmc"]):
        ssub = sub[sub["sampler"] == sampler_name]
        if ssub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        methods = ssub["method"].unique()
        N_vals  = sorted(ssub["N"].unique())
        Nf      = np.array(N_vals, dtype=float)

        for i, method in enumerate(methods):
            m_data = ssub[ssub["method"] == method].sort_values("N")
            if m_data.empty:
                continue
            st = LINE_STYLES[i % len(LINE_STYLES)]
            ax.loglog(m_data["N"], m_data["rmse"],
                      color="0.0", linestyle=st["ls"], linewidth=st["lw"],
                      marker=st["marker"], markersize=st["ms"],
                      markerfacecolor=st["mfc"], markeredgecolor="0.0",
                      markeredgewidth=0.8, label=_method_label(method))

        plain_mc_row = ssub[ssub["method"] == "plain_mc"].sort_values("N")
        if not plain_mc_row.empty:
            rmse0 = plain_mc_row["rmse"].iloc[0]
            N0    = plain_mc_row["N"].iloc[0]
            ax.loglog(Nf, rmse0 * np.sqrt(N0 / Nf), color="0.60", ls="--", lw=0.9,
                      label=r"$\mathcal{O}(N^{-1/2})$")
            if sampler_name == "qmc":
                ax.loglog(Nf, rmse0 * (N0 / Nf), color="0.75", ls=":", lw=0.9,
                          label=r"$\mathcal{O}(N^{-1})$")

        ax.set_xlabel("Number of simulated paths $N$", labelpad=5)
        if ax == axes[0]:
            ax.set_ylabel("RMSE", labelpad=5)
        sampler_title = ("Pseudo-random (PRNG)" if sampler_name == "prng"
                         else "Quasi-random (QMC / Sobol)")
        ax.set_title(f"({'a' if sampler_name == 'prng' else 'b'}) {sampler_title}",
                     fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", ncol=1,
                  handlelength=2.5, handletextpad=0.4)
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Figure 5 — RMSE vs $N$ (log-log): European call under GBM  "
                 "(all methods, PRNG vs QMC)", fontsize=9, y=0.98)
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure6_rmse_path_dependent(df: pd.DataFrame, outpath: str = None) -> None:
    """Figure 6: RMSE vs N for path-dependent options under Heston-QE (PRNG)."""
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_6_rmse_path_dependent.png"

    sub = df[(df["model"] == "heston_qe") & (df["sampler"] == "prng")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)

    for ax, payoff_name, panel_lbl in zip(
        axes,
        ["asian_call", "barrier_doc"],
        ["(a) Asian call — Heston-QE, PRNG",
         "(b) Barrier down-and-out call — Heston-QE, PRNG"]
    ):
        ssub = sub[sub["payoff"] == payoff_name]
        if ssub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(panel_lbl, fontsize=9)
            continue

        methods = ssub["method"].unique()
        N_vals  = sorted(ssub["N"].unique())
        Nf      = np.array(N_vals, dtype=float)

        for i, method in enumerate(methods):
            m_data = ssub[ssub["method"] == method].sort_values("N")
            if m_data.empty:
                continue
            st = LINE_STYLES[i % len(LINE_STYLES)]
            ax.loglog(m_data["N"], m_data["rmse"],
                      color="0.0", linestyle=st["ls"], linewidth=st["lw"],
                      marker=st["marker"], markersize=st["ms"],
                      markerfacecolor=st["mfc"], markeredgecolor="0.0",
                      markeredgewidth=0.8, label=_method_label(method))

        plain_mc_row = ssub[ssub["method"] == "plain_mc"].sort_values("N")
        if not plain_mc_row.empty:
            rmse0 = plain_mc_row["rmse"].iloc[0]
            N0    = plain_mc_row["N"].iloc[0]
            ax.loglog(Nf, rmse0 * np.sqrt(N0 / Nf), color="0.60", ls="--",
                      lw=0.9, label=r"$\mathcal{O}(N^{-1/2})$")

        ax.set_xlabel("$N$", labelpad=5)
        ax.set_ylabel("RMSE", labelpad=5)
        ax.set_title(panel_lbl, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", handlelength=2.5)
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(
        "Figure 6 — RMSE vs $N$ (log-log): path-dependent options under Heston-QE",
        fontsize=9, y=0.98
    )
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure7_nn_loss(
    train_hist: list, val_hist: list, outpath: str = None
) -> None:
    """Figure 7: NN variance-loss trajectory on train and validation sets."""
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_7_nn_loss.png"

    epochs = np.arange(1, len(train_hist) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, train_hist, color="0.0", lw=1.2, ls="-", alpha=0.45,
            label="Training loss (raw)")
    if val_hist:
        val_epochs = np.arange(1, len(val_hist) + 1)
        ax.plot(val_epochs, val_hist, color="0.0", lw=1.2, ls="--", alpha=0.45,
                label="Validation loss (raw)")

    w = max(10, len(train_hist) // 20)

    def _ma(series):
        if len(series) < w:
            return np.arange(len(series)) + 1, np.array(series)
        ma = np.convolve(series, np.ones(w) / w, mode="valid")
        return np.arange(w, len(series) + 1), ma

    e_ma, l_ma = _ma(train_hist)
    ax.plot(e_ma, l_ma, color="0.0", lw=1.6, ls="-",
            label=f"Training loss ({w}-epoch MA)")
    if val_hist:
        e_vma, l_vma = _ma(val_hist)
        ax.plot(e_vma, l_vma, color="0.45", lw=1.6, ls="--",
                label=f"Validation loss ({w}-epoch MA)")

    ax.set_xlabel("Training epoch", labelpad=5)
    ax.set_ylabel(
        r"$\mathcal{L}(\theta,c) = \mathbb{E}[(\phi - c\,g_\theta(x))^2]$", labelpad=5)
    ax.set_title(
        r"Figure 7 — NN variance-loss trajectory: "
        r"$\mathcal{L}(\theta, c)$ on train and validation sets",
        pad=10, fontsize=9
    )
    ax.legend(fontsize=8, loc="upper right")

    if val_hist:
        ax.annotate(
            f"Final train: {train_hist[-1]:.5f}\nFinal val: {val_hist[-1]:.5f}",
            xy=(epochs[-1], train_hist[-1]),
            xytext=(epochs[-1] * 0.65, max(train_hist) * 0.75),
            fontsize=8, color="0.2",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")


def plot_figure8_efficiency(df: pd.DataFrame, outpath: str = None) -> None:
    """Figure 8: Efficiency frontier — CPU time vs variance reduction ratio."""
    if outpath is None:
        outpath = f"{OUT_DIR}/figure_8_efficiency.png"

    N_max = df["N"].max()
    sub   = df[df["N"] == N_max].copy()
    if sub.empty:
        print("  [Fig 8] No data. Skipping.")
        return

    agg = sub.groupby("method")[["var_ratio", "time_s"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, row in agg.iterrows():
        st = LINE_STYLES[i % len(LINE_STYLES)]
        ax.scatter(row["time_s"], row["var_ratio"],
                   marker=st["marker"], s=110,
                   c="white", edgecolors="0.0", linewidths=1.0, zorder=5)
        ax.annotate(
            _method_label(row["method"]),
            xy=(row["time_s"], row["var_ratio"]),
            xytext=(row["time_s"] + row["time_s"] * 0.05,
                    row["var_ratio"] + max(agg["var_ratio"].max() * 0.01, 0.01)),
            fontsize=8, color="0.1",
        )

    ax.axhline(1.0, color="0.65", ls="--", lw=0.8, label="Variance ratio = 1 (plain MC)")
    ax.set_xlabel("Mean CPU time per replication (seconds)", labelpad=5)
    ax.set_ylabel(
        r"Variance reduction ratio $\mathrm{Var}(MC) / \mathrm{Var}(method)$", labelpad=5)
    ax.set_title(
        f"Figure 8 — Efficiency frontier: variance reduction vs computation time  "
        f"($N={N_max}$)", pad=10, fontsize=9
    )
    ax.legend(fontsize=8)
    ax.set_xscale("log")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outpath)
    plt.close()
    print(f"Saved: {outpath}")
