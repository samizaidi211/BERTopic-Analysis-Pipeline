"""
pdf_preprocess.py

Preprocess a single PDF into:
    1. document-level metadata
    2. a list of (chunk_id, chunk_text) tuples
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import fitz


# ---------------------------------------------------------------------------
# Scanned-PDF detection and OCR
# ---------------------------------------------------------------------------

def is_scanned_pdf(pdf_path: Path) -> bool:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if not pdf.pages:
            return True
        text = pdf.pages[0].extract_text() or ""
        return len(text.strip()) < 100


def ocr_pdf(pdf_path: Path) -> Path:
    try:
        subprocess.run(
            ["ocrmypdf", str(pdf_path), str(pdf_path)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print(f"Warning: ocrmypdf not found for {pdf_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: OCR failed for {pdf_path.name}: {e.stderr.decode()}")
    return pdf_path


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append((i + 1, text))
    return pages


# ---------------------------------------------------------------------------
# Cleaning + chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"(?<=[a-zA-Z])-\s+(?=[a-zA-Z])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    i = 0
    while i < len(words):
        window = words[i:i + chunk_size]
        chunks.append((f"chunk_{i}", " ".join(window)))
        i += step

    return chunks


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _safe_embedded_pdf_metadata(pdf_path: Path) -> Dict[str, str]:
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()

        return {
            "pdf_embedded_title": (meta.get("title") or "").strip(),
            "pdf_embedded_author": (meta.get("author") or "").strip(),
            "pdf_embedded_creationDate": (meta.get("creationDate") or "").strip(),
            "pdf_embedded_modDate": (meta.get("modDate") or "").strip(),
        }
    except Exception:
        return {
            "pdf_embedded_title": "",
            "pdf_embedded_author": "",
            "pdf_embedded_creationDate": "",
            "pdf_embedded_modDate": "",
        }


def _is_bad_author_value(value: str) -> bool:
    return str(value or "").strip().lower() in {
        "", "anonymous", "unknown", "none", "n/a"
    }


def _normalise_lines(text: str):
    return [re.sub(r"\s+", " ", l).strip() for l in (text or "").splitlines() if l.strip()]


def _infer_title_from_first_page(text: str, fallback: str = "") -> str:
    lines = _normalise_lines(text)

    for line in lines[:15]:
        if len(line) > 20 and not re.search(r"(abstract|doi|http|@)", line.lower()):
            return line

    return fallback


def _infer_authors_from_first_page(text: str, title: str):
    lines = _normalise_lines(text)

    for line in lines[:10]:
        if line != title and len(line) > 10:
            if re.search(r"(,| and |&|MD|PhD)", line):
                return line

    return ""


def _parse_author_count(authors: str) -> int:
    if not authors:
        return 0

    cleaned = re.sub(r"\b(MD|PhD|MSc)\b", "", authors, flags=re.I)
    parts = re.split(r"[;,]| and ", cleaned)
    return len([p for p in parts if p.strip()])


def _infer_publication_year(text: str, c: str = "", m: str = ""):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text + c + m)
    years = [int(y) for y in years if 1950 <= int(y) <= 2035]
    return str(min(years)) if years else ""


# ---------------------------------------------------------------------------
# Metadata extraction (FIXED)
# ---------------------------------------------------------------------------

def extract_document_metadata(pdf_path: Path, pages):
    embedded = _safe_embedded_pdf_metadata(pdf_path)
    first_page = pages[0][1] if pages else ""

    title = embedded["pdf_embedded_title"] or _infer_title_from_first_page(first_page, pdf_path.stem)

    embedded_author = embedded.get("pdf_embedded_author", "")
    inferred_authors = _infer_authors_from_first_page(first_page, title)

    if _is_bad_author_value(embedded_author):
        authors = inferred_authors
    else:
        authors = embedded_author

    if not authors:
        authors = "Unknown"

    author_count = _parse_author_count(authors)

    year = _infer_publication_year(
        first_page,
        embedded["pdf_embedded_creationDate"],
        embedded["pdf_embedded_modDate"],
    )

    return {
        "source_pdf": pdf_path.name,
        "source_pdf_stem": pdf_path.stem,
        "title": title,
        "authors": authors,
        "author_count": author_count,
        "publication_year": year,
        **embedded,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def preprocess_pdf(pdf_path: Path, chunk_size=900, overlap=150):
    if is_scanned_pdf(pdf_path):
        pdf_path = ocr_pdf(pdf_path)

    pages = extract_text_from_pdf(pdf_path)
    metadata = extract_document_metadata(pdf_path, pages)

    text = "\n".join(t for _, t in pages)
    text = clean_text(text)
    chunks = chunk_text(text, chunk_size, overlap)

    return metadata, chunks