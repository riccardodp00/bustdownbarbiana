"""
Plot graphs, summaries, and drift tests.

"""

from __future__ import annotations
import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot(
    *,
    df: pd.DataFrame,
    plots_dir: str,
    journal_slug: str,
    score_col: str = "alignment_score",) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    scores = df[score_col].dropna()

    # histograms
    plt.figure(figsize=(8, 5))
    plt.hist(
        scores,
        bins=35,
        color="#2E7D32",      # medium green fill
        alpha=0.80,
        edgecolor="#1B5E20",  # darker green edge
        linewidth=0.6,
    )

    mean_score = float(scores.mean()) if len(scores) else float("nan")
    if mean_score == mean_score:  # not NaN
        plt.axvline(
            mean_score,
            color="#1B5E20",
            linewidth=2.0,
            linestyle="-",
            label=f"Mean = {mean_score:.3f}",
        )
        plt.legend(frameon=False)

    plt.title("Distribution of alignment scores")
    plt.xlabel("Cosine similarity to Aims & Scope")
    plt.ylabel("Number of papers")
    plt.grid(axis="y", color="#D0D0D0", linewidth=0.8, linestyle="--", alpha=0.6)

    fn = os.path.join(plots_dir, f"{journal_slug}_alignment_histogram.png")
    plt.tight_layout()
    plt.savefig(fn, dpi=200)
    plt.close()
    print("Saved histogram to", fn)
    
    # mean alignment by year
    if "year" in df.columns and df["year"].notna().any():
        d = df[["year", score_col]].dropna().copy()
        d = d[d["year"].astype(str).str.match(r"^\d{4}$")]
        if not d.empty:
            d["year"] = d["year"].astype(int)
            yearly = d.groupby("year")[score_col].mean().reset_index().sort_values("year")

            plt.figure(figsize=(8, 5))
            plt.plot(
                yearly["year"],
                yearly[score_col],
                color="#1B5E20",
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="#1B5E20",
                markeredgewidth=1.2,
            )
            plt.title("Mean alignment score by year")
            plt.xlabel("Year")
            plt.ylabel("Mean alignment score")
            plt.grid(color="#D0D0D0", linewidth=0.8, linestyle="--", alpha=0.6)

            fn = os.path.join(plots_dir, f"{journal_slug}_alignment_by_year.png")
            plt.tight_layout()
            plt.savefig(fn, dpi=200)
            plt.close()
            print("Saved yearly trend to", fn)

    # top/bottom 10 articles
    required = ["title", "year", score_col, "url"]
    if all(c in df.columns for c in required):
        topk = df.nlargest(10, score_col)[required]
        botk = df.nsmallest(10, score_col)[required]
        topk.to_csv(os.path.join(plots_dir, f"{journal_slug}_top10.csv"), index=False)
        botk.to_csv(os.path.join(plots_dir, f"{journal_slug}_bottom10.csv"), index=False)
        print("Saved top10/bottom10 csvs")
    else:
        missing = [c for c in required if c not in df.columns]
        print(f"Skipping top/bottom CSVs (missing columns: {missing})")

def run_drift_tests(
    df: pd.DataFrame,
    year_col: str = "year",
    score_col: str = "alignment_score",) -> Tuple[dict, pd.DataFrame]:
    
    d = df[[year_col, score_col]].copy()
    d = d.dropna()
    d = d[(d[year_col].astype(str).str.match(r"^\d{4}$"))]
    d[year_col] = d[year_col].astype(int)
    d[score_col] = d[score_col].astype(float)

    if len(d) < 5:
        raise ValueError("Not enough rows with valid year+alignment_score to run drift tests.")

    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise ImportError("Install scipy: pip install scipy")

    rho, p_spear = spearmanr(d[year_col], d[score_col])

    from scipy.stats import linregress

    lr = linregress(d[year_col], d[score_col])

    yearly = (
        d.groupby(year_col)[score_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values(year_col)
    )

    results = {
        "n": len(d),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(p_spear),
        "linreg_slope_per_year": float(lr.slope),
        "linreg_intercept": float(lr.intercept),
        "linreg_r": float(lr.rvalue),
        "linreg_pvalue": float(lr.pvalue),
        "linreg_stderr": float(lr.stderr),
    }
    return results, yearly