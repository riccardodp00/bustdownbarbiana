"""
Topic analysis.

"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import config
from sklearn.feature_extraction.text import TfidfVectorizer

def label_topics_tfidf(df: pd.DataFrame, topic_col: str, text_col: str, *, top_terms: int = 8) -> dict[int, str]:
    """
    Create human-readable topic labels using TF-IDF keywords per cluster. Returns: {topic_id: "kw1, kw2, ..."}
    """

    texts = df[text_col].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    topic_labels: dict[int, str] = {}
    for tid in sorted(df[topic_col].dropna().unique()):
        tid = int(tid)
        idx = np.where(df[topic_col].to_numpy() == tid)[0]
        if len(idx) == 0:
            topic_labels[tid] = "EMPTY"
            continue

        mean_tfidf = X[idx].mean(axis=0)
        weights = np.asarray(mean_tfidf).ravel()
        if weights.sum() == 0:
            topic_labels[tid] = "NO_KEYWORDS"
            continue

        top_ids = weights.argsort()[-top_terms:][::-1]
        topic_labels[tid] = ", ".join(terms[top_ids].tolist())

    return topic_labels

def plot_topic_shares(
    year_topic_shares: pd.DataFrame,
    plots_dir: str,
    journal_slug: str,
    topic_labels: dict[int, str],
    *,
    top_n: int = 10,) -> None:
    """
    Readable stacked area plot:
    - keeps only the top N topics by total share across years
    - merges the rest into "Other"
    - uses keyword labels in the legend
    """
    # pivot: year x topic_id
    pivot = (
        year_topic_shares.pivot(index="year", columns="topic_id", values="share")
        .fillna(0.0)
        .sort_index()
    )

    # pick top topics by total share over all years
    totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_topics = totals.head(top_n).index.tolist()

    pivot_top = pivot[top_topics].copy()
    pivot_top["Other"] = 1.0 - pivot_top.sum(axis=1)

    years = pivot_top.index.to_numpy()
    series = pivot_top.to_numpy().T

    labels = []
    for col in pivot_top.columns:
        if col == "Other":
            labels.append("Other")
        else:
            col_int = int(col)
            labels.append(f"{col_int}: {topic_labels.get(col_int, 'UNKNOWN')}")

    plt.figure(figsize=(12, 7))
    plt.stackplot(years, series, labels=labels, alpha=0.85)
    plt.title(f"Topic prevalence over time (Top {top_n} topics + Other)")
    plt.xlabel("Year")
    plt.ylabel("Share of papers")

    # legend outside plot
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    fn = os.path.join(plots_dir, f"{journal_slug}_topic_shares_top{top_n}_stacked.png")
    plt.savefig(fn, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved topic share plot to", fn)