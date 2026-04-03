"""
patch_pubmed_metadata.py

Patch author names, titles, and publication years in all pipeline outputs
by fetching the ground-truth values directly from PubMed efetch using the
PMID that is already stored in source_pdf_stem.

WHY THIS IS NEEDED:
    pdf_preprocess.py infers metadata from PDF text heuristics and ReportLab
    embedded fields — both of which fail silently, giving "Anonymous" / blank
    for almost every paper. The correct data has always been available on
    PubMed; this script fetches it and writes it back.

FILES PATCHED (in place):
    outputs/topics/document_metadata_enriched.csv   ← primary target
    outputs/topics/topic_*_docs.csv                 ← one per topic
    outputs/topics/topic_outliers_docs.csv           ← if present

COLUMNS WRITTEN:
    title              cleaned PubMed article title
    authors            "Last First, Last First, ..." (up to MAX_AUTHORS shown,
                       remainder collapsed to "et al.")
    author_count       integer count of all authors
    publication_year   4-digit year string
    journal            full journal title from PubMed

USAGE:
    # Basic (no API key — 3 req/s, ~45 min for 11k papers in batches of 200)
    python patch_pubmed_metadata.py

    # With NCBI API key — 10 req/s, ~8 min
    python patch_pubmed_metadata.py --api-key YOUR_KEY_HERE

    # Or set environment variable
    set NCBI_API_KEY=your_key   (Windows)
    export NCBI_API_KEY=your_key  (Mac/Linux)
    python patch_pubmed_metadata.py

    # Dry run — fetch and print first batch only, write nothing
    python patch_pubmed_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Generator, List
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent
TOPICS_DIR   = SCRIPT_DIR / "outputs" / "topics"

BASE_URL     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH_SIZE   = 200     # PMIDs per efetch call (NCBI max 500; 200 is safe)
MAX_AUTHORS  = 6       # show up to this many names, then "et al."

REQUEST_DELAY     = 0.34   # 3 req/s without key
REQUEST_DELAY_KEY = 0.11   # 10 req/s with key

ENRICHED_CSV = "document_metadata_enriched.csv"

# ---------------------------------------------------------------------------
# NCBI helpers  (reused from pubmed_download.py)
# ---------------------------------------------------------------------------

def _delay(api_key: str) -> None:
    time.sleep(REQUEST_DELAY_KEY if api_key else REQUEST_DELAY)


def fetch_records_xml(pmids: List[str], api_key: str) -> ET.Element:
    params: dict = {
        "db":      "pubmed",
        "retmode": "xml",
        "rettype": "abstract",
        "id":      ",".join(pmids),
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{BASE_URL}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    _delay(api_key)
    return ET.fromstring(resp.content)


def batched(items: List[str], size: int) -> Generator[List[str], None, None]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_pubmed_record(article_el: ET.Element) -> dict:
    """
    Extract metadata fields from a <PubmedArticle> element.
    Always returns a dict (never None) — missing fields default to "".
    """
    pmid_el = article_el.find(".//PMID")
    pmid    = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

    title_el = article_el.find(".//ArticleTitle")
    title    = (title_el.text or "").strip() if title_el is not None else ""

    # Collect all authors
    all_authors: List[str] = []
    for auth in article_el.findall(".//Author"):
        last  = (auth.findtext("LastName") or "").strip()
        first = (auth.findtext("ForeName") or "").strip()
        if last:
            all_authors.append(f"{last} {first}".strip())

    author_count = len(all_authors)
    if author_count == 0:
        author_str = ""
    elif author_count <= MAX_AUTHORS:
        author_str = ", ".join(all_authors)
    else:
        author_str = ", ".join(all_authors[:MAX_AUTHORS]) + " et al."

    journal = (article_el.findtext(".//Journal/Title") or "").strip()

    year = (
        article_el.findtext(".//PubDate/Year") or
        article_el.findtext(".//PubDate/MedlineDate") or ""
    ).strip()[:4]

    return {
        "pmid":         pmid,
        "title":        title,
        "authors":      author_str,
        "author_count": author_count,
        "journal":      journal,
        "publication_year": year,
    }


# ---------------------------------------------------------------------------
# Batch fetch — returns a dict keyed by PMID string
# ---------------------------------------------------------------------------

def fetch_metadata_for_pmids(
    pmids: List[str],
    api_key: str,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Fetch PubMed metadata for all PMIDs in batched efetch calls.
    Returns a dict: { pmid_string -> metadata_dict }
    """
    results: Dict[str, dict] = {}
    total_batches = -(-len(pmids) // BATCH_SIZE)

    iterator = batched(pmids, BATCH_SIZE)
    if verbose:
        iterator = tqdm(
            list(iterator),
            total=total_batches,
            desc="Fetching from PubMed",
            unit="batch",
        )

    for batch in iterator:
        try:
            root = fetch_records_xml(batch, api_key)
        except Exception as exc:
            tqdm.write(f"  ⚠️  efetch failed for batch: {exc}")
            continue

        for article_el in root.findall(".//PubmedArticle"):
            record = parse_pubmed_record(article_el)
            pmid   = record.get("pmid", "")
            if pmid:
                results[pmid] = record

    return results


# ---------------------------------------------------------------------------
# CSV patching
# ---------------------------------------------------------------------------

PATCH_COLS = ["title", "authors", "author_count", "publication_year", "journal"]


def patch_dataframe(df: pd.DataFrame, metadata: Dict[str, dict]) -> tuple[pd.DataFrame, int]:
    """
    Overwrite PATCH_COLS in df using ground-truth PubMed metadata.
    Returns (patched_df, n_rows_updated).
    """
    df = df.copy()

    # Ensure columns exist
    for col in PATCH_COLS:
        if col not in df.columns:
            df[col] = ""

    updated = 0
    for i, row in df.iterrows():
        pmid = str(row.get("source_pdf_stem", "")).strip()
        if not pmid or pmid not in metadata:
            continue

        record = metadata[pmid]
        for col in PATCH_COLS:
            if col in record and record[col] != "":
                df.at[i, col] = record[col]
        updated += 1

    return df, updated


def patch_csv(path: Path, metadata: Dict[str, dict], dry_run: bool) -> int:
    """Read, patch, and overwrite a single CSV. Returns number of rows updated."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"  ⚠️  Could not read {path.name}: {exc}")
        return 0

    if "source_pdf_stem" not in df.columns:
        print(f"  ⚠️  Skipping {path.name} — no source_pdf_stem column.")
        return 0

    patched_df, n_updated = patch_dataframe(df, metadata)

    if dry_run:
        print(f"  [DRY RUN] Would update {n_updated:,} rows in {path.name}")
    else:
        patched_df.to_csv(path, index=False)
        print(f"  ✅  {path.name}: {n_updated:,} rows patched.")

    return n_updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(topics_dir: Path, api_key: str, dry_run: bool) -> None:
    print(f"\n=== PubMed Metadata Patch ===")
    print(f"  Topics dir : {topics_dir}")
    print(f"  API key    : {'yes' if api_key else 'no (3 req/s — slower)'}")
    print(f"  Dry run    : {dry_run}\n")

    # ----------------------------------------------------------------
    # Step 1 — collect all unique PMIDs across all CSVs to patch
    # ----------------------------------------------------------------
    target_files = sorted(topics_dir.glob("topic_*_docs.csv"))
    enriched_csv = topics_dir / ENRICHED_CSV
    if enriched_csv.exists():
        target_files = [enriched_csv] + [f for f in target_files]

    if not target_files:
        print(f"❌  No CSV files found in '{topics_dir}'. Run export.py first.")
        return

    all_pmids: set[str] = set()
    for path in target_files:
        try:
            df = pd.read_csv(path, usecols=["source_pdf_stem"], low_memory=False)
            pmids = df["source_pdf_stem"].dropna().astype(str).str.strip()
            pmids = pmids[pmids.str.match(r"^\d+$")]  # digits only = valid PMID
            all_pmids.update(pmids.tolist())
        except Exception:
            pass

    pmid_list = sorted(all_pmids)
    print(f"Step 1: Found {len(pmid_list):,} unique PMIDs across {len(target_files)} files.")

    if not pmid_list:
        print("❌  No valid PMIDs found. Check that source_pdf_stem contains numeric PMID values.")
        return

    # ----------------------------------------------------------------
    # Step 2 — fetch all metadata from PubMed in batches
    # ----------------------------------------------------------------
    print(f"Step 2: Fetching metadata from PubMed "
          f"({len(pmid_list):,} PMIDs in batches of {BATCH_SIZE}) ...")
    print(f"        Estimated time: "
          f"~{len(pmid_list) // BATCH_SIZE * (REQUEST_DELAY_KEY if api_key else REQUEST_DELAY):.0f}s\n")

    metadata = fetch_metadata_for_pmids(pmid_list, api_key)

    found    = len(metadata)
    not_found = len(pmid_list) - found
    print(f"\n  PubMed returned records for {found:,} / {len(pmid_list):,} PMIDs.")
    if not_found:
        print(f"  ⚠️  {not_found:,} PMIDs not found (may have been retracted or merged).")

    if dry_run:
        # Show a sample of what would be written
        sample = list(metadata.values())[:3]
        print("\n  Sample records that would be written:")
        for r in sample:
            print(f"    PMID {r['pmid']}: title='{r['title'][:60]}...' "
                  f"authors='{r['authors'][:60]}' year={r['publication_year']}")
        print()

    # ----------------------------------------------------------------
    # Step 3 — patch each CSV file
    # ----------------------------------------------------------------
    print(f"Step 3: Patching {len(target_files)} CSV files ...")
    total_updated = 0

    for path in target_files:
        total_updated += patch_csv(path, metadata, dry_run)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}✅  Patch complete.")
    print(f"   Total rows updated across all files : {total_updated:,}")
    print(f"   Files patched                       : {len(target_files)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patch author/title/year metadata in pipeline CSVs using PubMed efetch."
    )
    parser.add_argument(
        "--topics-dir",
        type=Path,
        default=TOPICS_DIR,
        help=f"Topics output directory (default: {TOPICS_DIR})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NCBI_API_KEY", ""),
        help="NCBI API key (or set NCBI_API_KEY env var). Raises rate limit from 3 to 10 req/s.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse data but do not write any files. Shows a sample of what would change.",
    )
    args = parser.parse_args()

    main(
        topics_dir = args.topics_dir,
        api_key    = args.api_key,
        dry_run    = args.dry_run,
    )
