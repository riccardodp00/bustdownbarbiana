"""
Orchestrates all other submodules in a neat, singular pipeline.
Here is a breakdown of what it does:
- loads Aims & Scope (thematic reference)
- loads/fetches corpus (from cache)
- embeds aims and scope (sentences by sentence)
- embeds abstracts
- computes mean alignment score (coherence vs aims)
- learns topics via clustering embeddings (conceptual structure)
- produces yearly topic prevalence + topic alignment trends (thematic drift)
- saves outputs + plots

"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from .topics import *
from .analysis import analyze_and_plot, run_drift_tests
from .embeddings import build_embedding_model, compute_alignment_scores_all, embed_texts
from .pubmed_client import collect_journal_abstracts_pubmed
from .utils import check_directories, caching_file, slugify, split_into_sentences

def pipeline():
    check_directories(config.DIRECTORY_DATA, config.DIRECTORY_GRAPHS)

    # used by sentence-transformers if needed
    os.environ["HF_TOKEN"] = config.HF_TOKEN

    aims_text = config.AIMS_AND_SCOPE
    if not aims_text.strip():
        print("AIMS_AND_SCOPE is empty. Please paste aims & scope.")
        return

    aims_sentences = split_into_sentences(aims_text)
    if not aims_sentences:
        print("AIMS_AND_SCOPE could not be split into sentences (empty after cleaning).")
        return
    print(f"Aims & scope: {len(aims_sentences)} sentences")

    cache_file = caching_file(
        config.DIRECTORY_DATA,
        config.SOURCE,
        config.JOURNAL,
        config.START,
        config.END,
        aims_text,
    )

    if os.path.exists(cache_file):
        print("Loading cached corpus from", cache_file)
        df = pd.read_parquet(cache_file)
    else:
        print("Collecting articles from PubMed...")
        df = collect_journal_abstracts_pubmed(
            eutils_base=config.EUTILS_BASE,
            pm_email=config.PM_EMAIL,
            pm_api_key=config.PM_API_KEY,
            journal_name=config.JOURNAL,
            start_year=config.START,
            end_year=config.END,
            max_results=config.PUBMED_RETMAX_DEFAULT,
            efetch_batch_size=config.PUBMED_EFETCH_BATCH_SIZE,
            throttle_seconds=config.PUBMED_THROTTLE_SECONDS,
        )
        df.to_parquet(cache_file, index=False)
        print("Saved corpus to", cache_file)

    print(f"Corpus size: {len(df)} papers")
    if len(df) == 0:
        print("No papers retrieved. Try adjusting JOURNAL (often PubMed uses the journal abbreviation).")
        return

    # --- Embeddings + coherence (alignment vs aims) ---
    model = build_embedding_model(config.EMBEDDING_MODEL)

    aims_embs = embed_texts(model, aims_sentences, batch_size=config.EMBEDDING_BATCH_SIZE)
    paper_embs = embed_texts(model, df["abstract"].tolist(), batch_size=config.EMBEDDING_BATCH_SIZE)

    # mean alignment across aims sentences
    _score_max, best_idx, score_mean, _score_topk3_mean = compute_alignment_scores_all(
        paper_embs, aims_embs, topk=3
    )
    df["alignment_score"] = score_mean
    df["best_aims_sentence_idx"] = best_idx
    df["best_aims_sentence"] = [aims_sentences[i] for i in best_idx]

    # topics (20 for roughly ~5k papers)
    n_papers = len(df)
    n_topics = int(getattr(config, "N_TOPICS", 20))
    n_topics = max(5, min(n_topics, 50))

    from sklearn.cluster import MiniBatchKMeans

    print(f"Clustering {n_papers} papers into {n_topics} topics (MiniBatchKMeans)...")
    km = MiniBatchKMeans(
        n_clusters=n_topics,
        random_state=0,
        batch_size=1024,
        n_init="auto",
        max_iter=200,
    )
    df["topic_id"] = km.fit_predict(paper_embs).astype(int)

    # label topics with TF-IDF keywords
    topic_labels = label_topics_tfidf(df, topic_col="topic_id", text_col="abstract", top_terms=8)
    df["topic_label"] = df["topic_id"].map(topic_labels).fillna("UNKNOWN")

    topic_labels_df = (
        pd.DataFrame({"topic_id": list(topic_labels.keys()), "topic_label": list(topic_labels.values())})
        .sort_values("topic_id")
        .reset_index(drop=True)
    )
    topic_labels_path = os.path.join(config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_topic_labels.csv")
    topic_labels_df.to_csv(topic_labels_path, index=False)
    print("Saved topic labels to", topic_labels_path)

    # most common subject per year
    d_year = df.dropna(subset=["year"]).copy()
    d_year["year"] = d_year["year"].astype(int)

    year_topic_counts = (
        d_year.groupby(["year", "topic_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["year", "count"], ascending=[True, False])
    )

    year_totals = d_year.groupby("year").size().reset_index(name="year_total")
    year_topic_counts = year_topic_counts.merge(year_totals, on="year", how="left")
    year_topic_counts["share"] = year_topic_counts["count"] / year_topic_counts["year_total"]
    year_topic_counts["topic_label"] = year_topic_counts["topic_id"].map(topic_labels).fillna("UNKNOWN")

    year_topic_counts_path = os.path.join(config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_year_topic_counts.csv")
    year_topic_counts.to_csv(year_topic_counts_path, index=False)
    print("Saved year-topic counts to", year_topic_counts_path)

    top_topic_per_year = (
        year_topic_counts.sort_values(["year", "count"], ascending=[True, False])
        .groupby("year")
        .head(1)
        .reset_index(drop=True)
    )
    top_topic_per_year_path = os.path.join(config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_top_topic_per_year.csv")
    top_topic_per_year.to_csv(top_topic_per_year_path, index=False)
    print("Saved top topic per year to", top_topic_per_year_path)

    # --- Yearly topic alignment (which topics are on/off mission over time) ---
    topic_alignment_by_year = (
        d_year.groupby(["year", "topic_id"])["alignment_score"]
        .mean()
        .reset_index(name="mean_alignment")
        .sort_values(["year", "topic_id"])
    )
    topic_alignment_by_year["topic_label"] = topic_alignment_by_year["topic_id"].map(topic_labels).fillna("UNKNOWN")
    topic_alignment_by_year_path = os.path.join(
        config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_topic_alignment_by_year.csv"
    )
    topic_alignment_by_year.to_csv(topic_alignment_by_year_path, index=False)
    print("Saved topic alignment by year to", topic_alignment_by_year_path)

    # --- Save main scored corpus ---
    out_csv = os.path.join(config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_scored.csv")
    df.to_csv(out_csv, index=False)
    print("Saved scored corpus to", out_csv)

    # --- Drift tests (mean alignment over time) + your existing plots ---
    results, yearly_summary = run_drift_tests(df, score_col="alignment_score")

    print("\n=== Drift Test Results (mean alignment) ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\n=== Yearly Summary (count/mean/median/std) ===")
    print(yearly_summary.to_string(index=False))

    yearly_summary.to_csv(
        os.path.join(config.DIRECTORY_DATA, f"{slugify(config.JOURNAL)}_mean_yearly_alignment_summary.csv"),
        index=False,
    )

    analyze_and_plot(
        df=df,
        plots_dir=config.DIRECTORY_GRAPHS,
        journal_slug=f"{slugify(config.JOURNAL)}_mean",
        score_col="alignment_score",
    )

    # --- Topic prevalence plot (evolution / drift of subjects) ---
    plot_topic_shares(
        year_topic_counts[["year", "topic_id", "share"]],
        config.DIRECTORY_GRAPHS,
        slugify(config.JOURNAL),
        topic_labels,
        top_n=10,
    )

    print("Done.")