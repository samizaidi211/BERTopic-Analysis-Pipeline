"""
text_extraction.py

Shared text-cleaning utilities.

Note: The primary preprocessing pipeline lives in pdf_preprocess.py, which is
called by ingest.py. This module exposes the same clean_text() function as a
shared utility so other modules (e.g. export, future query modules) can
normalise text consistently without importing from pdf_preprocess.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Clean raw text for use in the pipeline.

    Steps:
        1. Rejoin words hyphenated across line breaks
           (e.g. "treat-\\nment" → "treatment").
        2. Collapse all whitespace sequences to a single space.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = re.sub(r"(?<=[a-zA-Z])-\s+(?=[a-zA-Z])", "", text)  # de-hyphenation
    text = re.sub(r"\s+", " ", text).strip()                     # whitespace normalisation
    return text
