"""
PubMed E-utilities client + XML parsing + corpus collection.

"""

from __future__ import annotations
import re
import time
import xml.etree.ElementTree as ET
from typing import List, Optional
import pandas as pd
import requests
from tqdm import tqdm
from .utils import clean_up

def pubmed_esearch(
    *,
    eutils_base: str,
    term: str,
    pm_email: str,
    pm_api_key: Optional[str] = None,
    retmax: int = 5000,) -> List[str]:
    """Call PubMed esearch to retrieve PubMed IDs (PMIDs) matching a query term."""
    
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "email": pm_email,
    }
    if pm_api_key:
        params["api_key"] = pm_api_key
    
    r = requests.get(f"{eutils_base}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])

def pubmed_efetch(
    *,
    eutils_base: str,
    pmids: List[str],
    pm_email: str,
    pm_api_key: Optional[str] = None,) -> str:
    """Call PubMed efetch to retrieve XML metadata for a list of PMIDs."""
    
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": pm_email,
    }
    if pm_api_key:
        params["api_key"] = pm_api_key
        
    r = requests.get(f"{eutils_base}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str) -> List[dict]:
    """Parse PubMed efetch XML text and extract a list of paper records with abstracts."""
    
    root = ET.fromstring(xml_text)
    out: List[dict] = []
    
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title = clean_up(article.findtext(".//Article/ArticleTitle") or "")
        
        abs_nodes = article.findall(".//Article/Abstract/AbstractText")
        abstract = clean_up(" ".join(["".join(n.itertext()).strip() for n in abs_nodes]))
        
        year = None
        y1 = article.findtext(".//JournalIssue/PubDate/Year")
        if y1 and y1.isdigit():
            year = int(y1)
        else:
            medline_date = clean_up(article.findtext(".//JournalIssue/PubDate/MedlineDate") or "")
            m = re.search(r"(19|20)\d{2}", medline_date)
            if m:
                year = int(m.group(0))
            else:
                y2 = article.findtext(".//Article/ArticleDate/Year")
                if y2 and y2.isdigit():
                    year = int(y2)
        
        journal = clean_up(article.findtext(".//Journal/Title") or "")
        
        doi = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = clean_up(aid.text or "")
                break
        
        if not abstract:
            continue
        
        out.append(
            {
                "paperId": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "venue": journal,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "doi": doi,
            }
        )
        
    return out

def collect_journal_abstracts_pubmed(
    *,
    eutils_base: str,
    pm_email: str,
    pm_api_key: Optional[str],
    journal_name: str,
    start_year: int,
    end_year: int,
    max_results: int = 5000,
    efetch_batch_size: int = 200,
    throttle_seconds: float = 0.12,) -> pd.DataFrame:
    """Fetch PubMed abstracts for a journal and year range, returning a DataFrame."""
    
    if not pm_email:
        raise RuntimeError("Set PM_EMAIL in your .env for PubMed E-utilities requests.")
    
    term = f'("{journal_name}"[jour] OR "{journal_name}"[ta]) AND ("{start_year}"[dp] : "{end_year}"[dp])'
    print("PubMed esearch term:", term)
    
    pmids = pubmed_esearch(
        eutils_base=eutils_base,
        term=term,
        pm_email=pm_email,
        pm_api_key=pm_api_key,
        retmax=max_results,
    )
    print(f"PubMed found {len(pmids)} records (before abstract filtering).")
    
    records: List[dict] = []
    for i in tqdm(range(0, len(pmids), efetch_batch_size), desc="PubMed efetch"):
        batch = pmids[i : i + efetch_batch_size]
        xml_text = pubmed_efetch(
            eutils_base=eutils_base,
            pmids=batch,
            pm_email=pm_email,
            pm_api_key=pm_api_key,
        )
        records.extend(parse_pubmed_xml(xml_text))
        time.sleep(throttle_seconds)
    
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["paperId"]).reset_index(drop=True)
        df = df[(df["year"].fillna(0) >= start_year) & (df["year"].fillna(0) <= end_year)].copy()
        
    return df