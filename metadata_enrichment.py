"""
metadata_enrichment.py

PMID-first enrichment pipeline:
    PMID -> PubMed DOI lookup -> OpenCitations first -> Semantic Scholar fallback

Inputs:
    outputs/topics/topic_*_docs.csv

Outputs:
    outputs/topics/document_metadata_enriched.csv

Usage:
    python metadata_enrichment.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
TOPICS_DIR = SCRIPT_DIR / "outputs" / "topics"
OUTPUT_CSV = TOPICS_DIR / "document_metadata_enriched.csv"


# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------

PUBMED_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OPENCITATIONS_URL = "https://api.opencitations.net/index/v1/citation-count"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper"

PUBMED_DELAY = 0.34
OPENCITATIONS_DELAY = 0.25
SEMANTIC_SCHOLAR_DELAY = 1.0

SEMANTIC_SCHOLAR_API_KEY = ""  # optional


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def autorun_figures() -> None:
    figures_path = SCRIPT_DIR / "figures.py"
    if not figures_path.exists():
        print(f"⚠️ figures.py not found at: {figures_path}")
        return

    print("🚀 Running figures.py ...")
    result = subprocess.run(["python", str(figures_path)], cwd=SCRIPT_DIR)
    if result.returncode == 0:
        print("✅ figures.py completed.")
    else:
        print(f"❌ figures.py exited with code {result.returncode}.")


def normalize_doi(doi: str) -> str:
    doi = str(doi or "").strip()
    doi = doi.removeprefix("https://doi.org/")
    doi = doi.removeprefix("http://doi.org/")
    doi = doi.removeprefix("http://dx.doi.org/")
    return doi


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# STEP 1 — BUILD UNIQUE PAPER TABLE
# ---------------------------------------------------------------------------

def build_paper_table(topics_dir: Path) -> pd.DataFrame:
    frames = []

    for path in sorted(topics_dir.glob("topic_*_docs.csv")):
        if "outlier" in path.name:
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            frames.append(df)
        except Exception as exc:
            print(f"⚠️ Could not read {path.name}: {exc}")

    if not frames:
        raise FileNotFoundError(
            f"No topic_*_docs.csv files found in '{topics_dir}'. Run export.py first."
        )

    combined = pd.concat(frames, ignore_index=True)

    keep = [
        "source_pdf",
        "source_pdf_stem",
        "title",
        "authors",
        "author_count",
        "publication_year",
    ]
    for col in keep:
        if col not in combined.columns:
            combined[col] = ""

    combined = combined[keep].copy()
    combined["source_pdf_stem"] = (
    combined["source_pdf_stem"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)   # 🔥 extract PMID digits
)

    combined = combined[combined["source_pdf_stem"].notna()]
    combined = (
        combined
        .sort_values("title", key=lambda s: s.astype(str).str.len(), ascending=False)
        .drop_duplicates(subset="source_pdf_stem", keep="first")
        .reset_index(drop=True)
    )

    print(f"Built paper table: {len(combined):,} unique PMIDs.")
    return combined


# ---------------------------------------------------------------------------
# STEP 2 — PMID -> DOI VIA PUBMED
# ---------------------------------------------------------------------------

def fetch_pubmed_xml(pmids: list[str]) -> ET.Element:
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "id": ",".join(pmids),
    }
    response = requests.get(PUBMED_URL, params=params, timeout=60)
    response.raise_for_status()
    time.sleep(PUBMED_DELAY)
    return ET.fromstring(response.content)


def extract_doi_from_article(article_el: ET.Element) -> str:
    for aid in article_el.findall(".//ArticleIdList/ArticleId"):
        if aid.attrib.get("IdType", "").lower() == "doi":
            return (aid.text or "").strip()

    for aid in article_el.findall(".//ArticleId"):
        if aid.attrib.get("IdType", "").lower() == "doi":
            return (aid.text or "").strip()

    return ""


def fetch_dois_from_pubmed(pmids: list[str]) -> dict[str, str]:
    doi_map: dict[str, str] = {}

    for batch in tqdm(batched(pmids, 200), desc="Fetching DOIs from PubMed", unit="batch"):
        try:
            root = fetch_pubmed_xml(batch)
        except Exception as exc:
            tqdm.write(f"⚠️ PubMed efetch failed for batch: {exc}")
            continue

        for article_el in root.findall(".//PubmedArticle"):
            pmid = (article_el.findtext(".//PMID") or "").strip()
            doi = extract_doi_from_article(article_el)
            if pmid:
                doi_map[pmid] = doi

    return doi_map


# ---------------------------------------------------------------------------
# STEP 3 — OPENCITATIONS FIRST
# ---------------------------------------------------------------------------

def query_opencitations(session: requests.Session, doi: str) -> str:
    doi = normalize_doi(doi)
    if not doi:
        return ""

    try:
        response = session.get(f"{OPENCITATIONS_URL}/{doi}", timeout=30)
        response.raise_for_status()
        data = response.json()
        time.sleep(OPENCITATIONS_DELAY)

        if isinstance(data, list) and data:
            return str(data[0].get("count", "")).strip()
    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# STEP 4 — PMID FALLBACK TO SEMANTIC SCHOLAR
# ---------------------------------------------------------------------------

def query_semantic_scholar_by_pmid(session: requests.Session, pmid: str) -> str:
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        response = session.get(
            f"{SEMANTIC_SCHOLAR_URL}/PMID:{pmid}",
            params={"fields": "citationCount"},
            headers=headers,
            timeout=30,
        )
        if response.status_code == 404:
            return ""
        response.raise_for_status()
        data = response.json()
        time.sleep(SEMANTIC_SCHOLAR_DELAY)
        return str(data.get("citationCount", "")).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Metadata Enrichment (PMID -> PubMed DOI -> OpenCitations first -> S2 fallback) ===\n")

    paper_df = build_paper_table(TOPICS_DIR)
    pmids = paper_df["source_pdf_stem"].astype(str).tolist()

    doi_map = fetch_dois_from_pubmed(pmids)

    enriched_rows = []
    doi_found = 0
    oc_hits = 0
    s2_hits = 0

    with requests.Session() as session:
        for i, row in enumerate(tqdm(paper_df.itertuples(index=False), total=len(paper_df), desc="Fetching citation counts")):
            pmid = str(getattr(row, "source_pdf_stem", "")).strip()
            doi = normalize_doi(doi_map.get(pmid, ""))

            if doi:
                doi_found += 1

            citation_count = ""
            enrichment_source = "not_found"

            # OpenCitations first
            if doi:
                citation_count = query_opencitations(session, doi)
                if citation_count not in {"", "0", "None"}:
                    enrichment_source = "opencitations"
                    oc_hits += 1

            # PMID fallback
            if citation_count in {"", "0", "None"}:
                citation_count = query_semantic_scholar_by_pmid(session, pmid)
                if citation_count not in {"", "0", "None"}:
                    enrichment_source = "semantic_scholar_pmid"
                    s2_hits += 1

            enriched_rows.append({
                "doi": doi,
                "citation_count": citation_count,
                "enrichment_source": enrichment_source,
                "match_score": 1.0 if enrichment_source != "not_found" else "",
            })

            if (i + 1) % 200 == 0:
                partial_df = pd.concat(
                    [paper_df.iloc[: i + 1].reset_index(drop=True), pd.DataFrame(enriched_rows)],
                    axis=1,
                )
                partial_df.to_csv(OUTPUT_CSV, index=False)

    final_df = pd.concat(
        [paper_df.reset_index(drop=True), pd.DataFrame(enriched_rows)],
        axis=1,
    )
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Enrichment complete")
    print(f"Output file                    : {OUTPUT_CSV}")
    print(f"Rows written                   : {len(final_df):,}")
    print(f"PMIDs with DOI found          : {doi_found:,} / {len(final_df):,}")
    print(f"OpenCitations hits            : {oc_hits:,}")
    print(f"Semantic Scholar fallback hits: {s2_hits:,}")
    print(f"Total citation coverage       : {(oc_hits + s2_hits) / max(len(final_df), 1):.2%}")

    autorun_figures()


if __name__ == "__main__":
    main()