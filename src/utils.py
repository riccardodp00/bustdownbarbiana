"""
Utilities: filesystem, text normalization, cache filenames, sentence splitting.

"""

from __future__ import annotations
import hashlib
import os
import re
from typing import List, Optional
from syntok.segmenter import process as syntok_process

def slugify(s: str) -> str:
    """Convert a string to a filesystem-friendly slug (lowercase, non-word to underscore)."""
    return re.sub(r"\W+", "_", (s or "")).strip("_").lower()

def check_directories(*dirs: str) -> None:
    """Create expected output directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def clean_up(s: Optional[str]) -> str:
    """Normalize text by stripping ends and collapsing internal whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_into_sentences(text: str) -> List[str]:
    """Split into sentence."""
    text = clean_up(text)
    if not text:
        return []

    sentences: List[str] = []
    for paragraph in syntok_process(text):
        for sentence in paragraph:
            s = "".join(tok.value for tok in sentence).strip()
            if s:
                sentences.append(s)

    return sentences

def caching_file(
    data_dir: str,
    source: str,
    journal: str,
    start_year: int,
    end_year: int,
    aims_text: str,
) -> str:
    """Create a deterministic cache path based on source/journal/year range and aims text hash."""
    aims_hash = hashlib.md5(clean_up(aims_text).encode("utf-8")).hexdigest()[:10]
    base = f"{source}|{journal}|{start_year}|{end_year}|{aims_hash}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
    safe = slugify(journal)[:60]
    return os.path.join(data_dir, f"{source}_{safe}_{start_year}_{end_year}_{h}.parquet")