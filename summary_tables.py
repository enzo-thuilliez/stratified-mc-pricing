"""Generate LaTeX summary tables and a Markdown interpretation summary
from figures/benchmark_surface_results.csv.

All numbers reported here are computed directly from the CSV produced by
the benchmark run. Nothing is hard-coded.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("figures/benchmark_surface_results.csv")
OUT_DIR = Path("figures")

METHOD_LABELS = {
    "plain_mc": "Plain MC",
    "antithetic": "Antithetic",
    "stratified_kmeans": "Stratified (k-means)",
    "stratified_gmm": "Stratified (GMM)",
    "rf_cv": "Control variate (RF)",
    "nn_cv": "Control variate (NN)",
}

MODEL_LABELS = {
    "gbm": "GBM",
    "heston_qe": "Heston QE",
}

PAYOFF_LABELS = {
    "european_call": "European call",
    "asian_call": "Asian call",
    "barrier_doc": "Barrier (down-and-out call)",
    "digital_call": "Digital call",
}

PAYOFF_ORDER = ["european_call", "asian_call", "barrier_doc", "digital_call"]
MATURITY_ORDER = ["short", "medium", "long"]
MONEYNESS_ORDER = ["ITM", "ATM", "OTM"]


def method_label(m: str) -> str:
    return METHOD_LABELS.get(m, m.replace("_", " "))


def escape_tex(s: str) -> str:
    return str(s).replace("_", r"\_")


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean var_ratio / rmse / runtime per (group, method), robust to NaN."""
    agg = (
        df.groupby(group_cols + ["method"])
        .agg(
            var_ratio=("variance_reduction_ratio", "mean"),
            rmse=("rmse", "mean"),
            runtime=("runtime_seconds", "mean"),
            n=("variance_reduction_ratio", "count"),
        )
        .reset_index()
    )
    return agg


def best_method_per_group(agg: pd.DataFrame, group_cols: list[str]) -> dict:
    """Return {group_key: best_method} based on highest mean var_ratio,
    ignoring NaN rows."""
    best = {}
    for key, sub in agg.groupby(group_cols):
        sub_valid = sub.dropna(subset=["var_ratio"])
        if sub_valid.empty:
            best[key] = None
            continue
        row = sub_valid.loc[sub_valid["var_ratio"].idxmax()]
        best[key] = row["method"]
    return best


def fmt_num(x, decimals):
    if pd.isna(x):
        return "--"
    return f"{x:.{decimals}f}"


def build_table_by_group(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    group_label_map: dict,
    caption: str,
    label: str,
) -> str:
    """Build a booktabs LaTeX table with one sub-block per group value and
    one row per method. The row with the highest var_ratio in each group
    block is bolded."""

    present_groups = [g for g in group_order if g in df[group_col].unique()]
    if not present_groups:
        present_groups = sorted(df[group_col].dropna().unique().tolist())

    agg = aggregate(df, [group_col])
    best = best_method_per_group(agg, [group_col])

    methods_present = [
        m for m in METHOD_LABELS if m in df["method"].unique()
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")
    header_titles = {
        "model": "Model",
        "payoff": "Payoff",
        "maturity_label": "Maturity",
        "moneyness_label": "Moneyness",
    }
    lines.append(
        header_titles.get(group_col, group_col.replace("_", " ").title())
        + r" & Method & Mean var.\ red.\ ratio & Mean RMSE & Mean runtime (s) \\"
    )
    lines.append(r"\midrule")

    for g in present_groups:
        sub = agg[agg[group_col] == g]
        if sub.empty:
            continue
        best_method = best.get(g)
        g_label = group_label_map.get(g, str(g))
        first_row = True
        for m in methods_present:
            row = sub[sub["method"] == m]
            if row.empty:
                continue
            row = row.iloc[0]
            var_str = fmt_num(row["var_ratio"], 2)
            rmse_str = fmt_num(row["rmse"], 4)
            run_str = fmt_num(row["runtime"], 3)
            m_str = method_label(m)
            if m == best_method:
                var_str = r"\textbf{" + var_str + "}"
                m_str = r"\textbf{" + m_str + "}"
            label_cell = g_label if first_row else ""
            lines.append(f"{label_cell} & {m_str} & {var_str} & {rmse_str} & {run_str} \\\\")
            first_row = False
        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_tables(df: pd.DataFrame) -> None:
    model_map = dict(MODEL_LABELS)
    tex_model = build_table_by_group(
        df,
        "model",
        ["gbm", "heston_qe"],
        model_map,
        "Mean variance reduction ratio, RMSE and runtime by simulation "
        "model and variance-reduction method. Best-performing method per "
        "model (highest mean variance reduction ratio) shown in bold.",
        "tab:by_model",
    )
    (OUT_DIR / "tables_by_model.tex").write_text(tex_model, encoding="utf-8")

    payoff_map = dict(PAYOFF_LABELS)
    tex_payoff = build_table_by_group(
        df,
        "payoff",
        PAYOFF_ORDER,
        payoff_map,
        "Mean variance reduction ratio, RMSE and runtime by payoff type "
        "and variance-reduction method. Best-performing method per payoff "
        "(highest mean variance reduction ratio) shown in bold.",
        "tab:by_payoff",
    )
    (OUT_DIR / "tables_by_payoff.tex").write_text(tex_payoff, encoding="utf-8")

    maturity_map = {k: k.capitalize() for k in MATURITY_ORDER}
    tex_maturity = build_table_by_group(
        df,
        "maturity_label",
        MATURITY_ORDER,
        maturity_map,
        "Mean variance reduction ratio, RMSE and runtime by maturity "
        "bucket and variance-reduction method. Best-performing method per "
        "maturity (highest mean variance reduction ratio) shown in bold.",
        "tab:by_maturity",
    )
    (OUT_DIR / "tables_by_maturity.tex").write_text(tex_maturity, encoding="utf-8")

    moneyness_map = {k: k for k in MONEYNESS_ORDER}
    tex_moneyness = build_table_by_group(
        df,
        "moneyness_label",
        MONEYNESS_ORDER,
        moneyness_map,
        "Mean variance reduction ratio, RMSE and runtime by moneyness "
        "bucket and variance-reduction method. Best-performing method per "
        "moneyness bucket (highest mean variance reduction ratio) shown "
        "in bold.",
        "tab:by_moneyness",
    )
    (OUT_DIR / "tables_by_moneyness.tex").write_text(tex_moneyness, encoding="utf-8")


def safe_mean(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.mean()) if not series.empty else float("nan")


def top_methods_by_var_ratio(df: pd.DataFrame, n: int = 2) -> list[tuple[str, float]]:
    agg = df.groupby("method")["variance_reduction_ratio"].mean().dropna()
    agg = agg.sort_values(ascending=False)
    return list(agg.head(n).items())


def build_markdown_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Benchmark Interpretation Summary")
    lines.append("")
    lines.append(
        "Automatically generated from `figures/benchmark_surface_results.csv` "
        "by `summary_tables.py`. All figures below are computed directly from "
        "that file."
    )
    lines.append("")

    present_payoffs = set(df["payoff"].dropna().unique())
    present_maturities = set(df["maturity_label"].dropna().unique())
    present_moneyness = set(df["moneyness_label"].dropna().unique())
    present_models = set(df["model"].dropna().unique())

    # --- ATM variance reduction, by maturity ---
    lines.append("## Variance reduction at ATM, by maturity")
    lines.append("")
    atm = df[df["moneyness_label"] == "ATM"]
    if atm.empty:
        lines.append("- ATM configurations are not available in the quick profile.")
    else:
        for mat in MATURITY_ORDER:
            if mat not in present_maturities:
                lines.append(f"- Maturity `{mat}` is not available in the quick profile.")
                continue
            sub = atm[atm["maturity_label"] == mat]
            if sub.empty:
                lines.append(f"- No ATM data for maturity `{mat}` in the quick profile.")
                continue
            agg = sub.groupby("method")["variance_reduction_ratio"].mean().dropna()
            if agg.empty:
                lines.append(f"- ATM, maturity `{mat}`: variance reduction ratio not available (NaN).")
                continue
            best_m = agg.idxmax()
            best_v = agg.loc[best_m]
            lines.append(
                f"- At ATM, maturity `{mat}`: **{method_label(best_m)}** achieves the highest "
                f"mean variance reduction ratio ({best_v:.2f}x) across the "
                f"{sub['method'].nunique()} methods compared."
            )
    lines.append("")

    # --- Stability at long maturities ---
    lines.append("## Stability at long maturities (T = 2.0, `maturity_label == 'long'`)")
    lines.append("")
    long_df = df[df["maturity_label"] == "long"]
    if long_df.empty:
        lines.append("- Long-maturity (`long`) configurations are not available in the quick profile.")
    else:
        stats = (
            long_df.groupby("method")["variance_reduction_ratio"]
            .agg(["mean", "std"])
            .dropna()
        )
        if stats.empty:
            lines.append("- Variance reduction ratios at long maturity are all NaN in the quick profile.")
        else:
            stats = stats.sort_values("std")
            most_stable = stats.index[0]
            least_stable = stats.index[-1]
            lines.append(
                f"- **{method_label(most_stable)}** is the most stable method at long maturity "
                f"(std of variance reduction ratio = {stats.loc[most_stable, 'std']:.2f}, "
                f"mean = {stats.loc[most_stable, 'mean']:.2f}x)."
            )
            lines.append(
                f"- **{method_label(least_stable)}** is the least stable at long maturity "
                f"(std = {stats.loc[least_stable, 'std']:.2f}, "
                f"mean = {stats.loc[least_stable, 'mean']:.2f}x)."
            )
    lines.append("")

    # --- Efficiency loss under Heston QE vs GBM ---
    lines.append("## Efficiency under Heston QE vs GBM")
    lines.append("")
    if {"gbm", "heston_qe"}.issubset(present_models):
        gbm_eff = df[df["model"] == "gbm"].groupby("method")["efficiency_metric"].mean()
        heston_eff = df[df["model"] == "heston_qe"].groupby("method")["efficiency_metric"].mean()
        common_methods = sorted(set(gbm_eff.dropna().index) & set(heston_eff.dropna().index))
        if not common_methods:
            lines.append("- Efficiency metric not available (NaN) for one or both models.")
        else:
            ratios = {
                m: (heston_eff[m] - gbm_eff[m]) / gbm_eff[m] if gbm_eff[m] != 0 else float("nan")
                for m in common_methods
            }
            ratios = {m: r for m, r in ratios.items() if not pd.isna(r)}
            losers = sorted(ratios.items(), key=lambda kv: kv[1])
            for m, r in losers:
                direction = "loses" if r < 0 else "gains"
                lines.append(
                    f"- **{method_label(m)}**: mean efficiency metric GBM = {gbm_eff[m]:.4f}, "
                    f"Heston QE = {heston_eff[m]:.4f} ({direction} "
                    f"{abs(r) * 100:.1f}% relative efficiency under Heston QE)."
                )
    else:
        lines.append("- One of `gbm` / `heston_qe` is not available in the quick profile.")
    lines.append("")

    # --- Path-dependency and discontinuity effects ---
    lines.append("## Effect of path-dependency (Asian) and discontinuity (digital)")
    lines.append("")
    euro = df[df["payoff"] == "european_call"]
    if "asian_call" in present_payoffs:
        asian = df[df["payoff"] == "asian_call"]
        euro_agg = euro.groupby("method")["variance_reduction_ratio"].mean().dropna()
        asian_agg = asian.groupby("method")["variance_reduction_ratio"].mean().dropna()
        common = sorted(set(euro_agg.index) & set(asian_agg.index))
        if common:
            for m in common:
                delta = asian_agg[m] - euro_agg[m]
                trend = "lower" if delta < 0 else "higher"
                lines.append(
                    f"- For **{method_label(m)}**, path-dependency (Asian vs European call) "
                    f"yields {trend} mean variance reduction ratio "
                    f"({asian_agg[m]:.2f}x vs {euro_agg[m]:.2f}x)."
                )
        else:
            lines.append("- Variance reduction ratios for Asian vs European comparison not available (NaN).")
    else:
        lines.append("- Asian call payoff is not available in the quick profile.")

    if "digital_call" in present_payoffs:
        digital = df[df["payoff"] == "digital_call"]
        euro_agg = euro.groupby("method")["variance_reduction_ratio"].mean().dropna()
        digital_agg = digital.groupby("method")["variance_reduction_ratio"].mean().dropna()
        common = sorted(set(euro_agg.index) & set(digital_agg.index))
        if common:
            for m in common:
                delta = digital_agg[m] - euro_agg[m]
                trend = "lower" if delta < 0 else "higher"
                lines.append(
                    f"- For **{method_label(m)}**, the payoff discontinuity (digital vs European call) "
                    f"yields {trend} mean variance reduction ratio "
                    f"({digital_agg[m]:.2f}x vs {euro_agg[m]:.2f}x)."
                )
        else:
            lines.append("- Variance reduction ratios for digital vs European comparison not available (NaN).")
    else:
        lines.append("- Digital call payoff is not available in the quick profile.")

    if "barrier_doc" not in present_payoffs:
        lines.append("- Barrier (down-and-out call) payoff is not available in the quick profile.")
    lines.append("")

    # --- RMSE / runtime trade-off ---
    lines.append("## RMSE / runtime trade-off")
    lines.append("")
    trade = df.groupby("method").agg(
        rmse=("rmse", "mean"), runtime=("runtime_seconds", "mean")
    ).dropna()
    if trade.empty:
        lines.append("- RMSE/runtime data not available (NaN).")
    else:
        # Pareto frontier: lower rmse and lower runtime is better.
        pareto = []
        for m, row in trade.iterrows():
            dominated = False
            for m2, row2 in trade.iterrows():
                if m2 == m:
                    continue
                if row2["rmse"] <= row["rmse"] and row2["runtime"] <= row["runtime"] and (
                    row2["rmse"] < row["rmse"] or row2["runtime"] < row["runtime"]
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(m)
        lines.append(
            "- Methods on the RMSE/runtime Pareto frontier (no other method is both "
            "as accurate and as fast or faster): "
            + ", ".join(f"**{method_label(m)}**" for m in pareto)
            + "."
        )
        for m, row in trade.sort_values("rmse").iterrows():
            lines.append(
                f"  - {method_label(m)}: mean RMSE = {row['rmse']:.4f}, "
                f"mean runtime = {row['runtime']:.3f} s."
            )
    lines.append("")

    # --- Limits visible in the data ---
    lines.append("## Limits visible in the data")
    lines.append("")
    var_stats = df.groupby("method")["variance_reduction_ratio"].agg(["mean", "std"]).dropna()
    if not var_stats.empty:
        var_stats["cv"] = var_stats["std"] / var_stats["mean"]
        most_unstable = var_stats["cv"].idxmax()
        lines.append(
            f"- **{method_label(most_unstable)}** shows the highest relative dispersion "
            f"in variance reduction ratio across configurations "
            f"(coefficient of variation = {var_stats.loc[most_unstable, 'cv']:.2f}), "
            f"indicating unstable performance depending on the setting."
        )

    runtime_by_method = df.groupby("method")["runtime_seconds"].mean().dropna()
    if not runtime_by_method.empty:
        costliest = runtime_by_method.idxmax()
        cheapest = runtime_by_method.idxmin()
        others_mean = runtime_by_method.drop(costliest).mean()
        ratio = runtime_by_method[costliest] / others_mean if others_mean else float("nan")
        lines.append(
            f"- **{method_label(costliest)}** is the most computationally costly method "
            f"(mean runtime = {runtime_by_method[costliest]:.3f} s, vs "
            f"{others_mean:.3f} s averaged over all other methods, "
            f"{ratio:.1f}x slower), while **{method_label(cheapest)}** is the cheapest "
            f"(mean runtime = {runtime_by_method[cheapest]:.3f} s). The neural-network "
            f"control variate (`nn_cv`) itself has a mean runtime of "
            f"{runtime_by_method.get('nn_cv', float('nan')):.3f} s, which is "
            f"{'above' if runtime_by_method.get('nn_cv', 0) > runtime_by_method.median() else 'below'} "
            f"the median across methods."
        )

    # Configs where plain_mc (i.e. var_ratio == 1) is the best, i.e. no method wins.
    group_cols = ["model", "payoff", "maturity_label", "moneyness_label"]
    agg = aggregate(df, group_cols)
    no_winner = []
    for key, sub in agg.groupby(group_cols):
        sub_valid = sub.dropna(subset=["var_ratio"])
        if sub_valid.empty:
            continue
        best_row = sub_valid.loc[sub_valid["var_ratio"].idxmax()]
        if best_row["method"] == "plain_mc" or best_row["var_ratio"] <= 1.0 + 1e-9:
            no_winner.append(key)
    if no_winner:
        lines.append(
            f"- In {len(no_winner)} configuration(s), no variance-reduction method "
            "outperforms plain Monte Carlo (best mean variance reduction ratio <= 1x), e.g.: "
            + "; ".join(
                f"model={k[0]}, payoff={k[1]}, maturity={k[2]}, moneyness={k[3]}"
                for k in no_winner[:5]
            )
            + ("." if len(no_winner) <= 5 else f" (and {len(no_winner) - 5} more)." )
        )
    else:
        lines.append(
            "- In every configuration of the quick profile, at least one "
            "variance-reduction method outperforms plain Monte Carlo."
        )

    nan_count = df["variance_reduction_ratio"].isna().sum()
    if nan_count:
        lines.append(
            f"- {nan_count} row(s) have a missing (NaN) variance reduction ratio and were "
            "excluded from the averages above."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    write_tables(df)
    summary = build_markdown_summary(df)
    (OUT_DIR / "benchmark_interpretation_summary.md").write_text(summary, encoding="utf-8")
    print("Wrote:")
    for f in [
        "tables_by_model.tex",
        "tables_by_payoff.tex",
        "tables_by_maturity.tex",
        "tables_by_moneyness.tex",
        "benchmark_interpretation_summary.md",
    ]:
        print(" -", OUT_DIR / f)


if __name__ == "__main__":
    main()
