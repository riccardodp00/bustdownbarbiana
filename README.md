# P7. Analyzing Thematic Alignment in Scientific Journals

Riccardo De Pascalis

## Overview

This project repository implements a reproducible computational pipeline for quantifying the semantic alignment between published journal articles and a journal’s stated Aims & Scope.

Using sentence embeddings and cosine similarity, the pipeline estimates how closely individual article abstracts align with the conceptual positioning of the journal. It additionally evaluates temporal trends and potential thematic drift across years.

The workflow is fully orchestrated through a single entry point: `main()`.

---

## Layout of the project

1. A target journal is picked and its Aims and Scope are garnered from its webpage.
2. Its corpus is retrieved from PubMed (with deterministic caching).
3. Generate sentence embeddings.
4. Compute alignment scores.
5. Conduct yearly aggregation and drift analysis.
6. Save scored outputs and visualizations.

---

## Repository Structure

```text
├── demo/
│   ├── nlp-project-7.ipynb
│   ├── data/
│   └── graphs/
├── .gitignore
├── main.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── src/code/
    ├── __init__.py
    ├── pipeline.py
    ├── config.py
    ├── pubmed_client.py
    ├── embeddings.py
    ├── analysis.py
    └── utils.py

```

---

## Configuration

Excluded from this repo is an `.env` file where the following environmental variables need to be stored:
- PUBMED_EMAIL (necessary to get an NBCI account);
- PUBMED_API_KEY (optional, makes retrieving faster);
- HF_EMAIL (necessary to get a HuggingFace account);
- HF_TOKEN (necessary, in read mode).

In addition, edit `thematic_alignment/config.py` with your preferred items:
- JOURNAL (name of the journal of your choice)
- START / END (range of the articles);
- AIMS_AND_SCOPE (aims and scope of the journal)
- EMBEDDING_MODEL (embedding model)

---

## Installation
```text
python -m venv .venv # alternatively python3
source .venv/bin/activate
pip install -e .
```

---

## Execution
```text
python main.py # alternatively python3
```

---

## Output

`data` directory:
- *_scored.csv;
- *_yearly_alignment_summary.csv;
- cached parquet corpus.

`graphs` directory:
- alignment distribution histogram
- yearly alignment trend plot
- top 10 aligned articles (`.csv`)
- bottom 10 aligned articles (`.csv`)

---

## License

MIT License.