# export.py
"""
export.py

Export BERTopic results to CSV files, including metadata columns.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from bertopic import BERTopic

from config import get_logger, load_config


def load_records_from_manifests(
    manifests_folder: Path,
) -> Tuple[List[str], List[str], List[dict]]:
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[dict] = []

    for manifest_path in sorted(manifests_folder.glob("*.jsonl")):
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    metadata = record.get("metadata", {}) or {}
                    if text:
                        ids.append(record.get("id", ""))
                        texts.append(text)
                        metadatas.append(metadata)
                except json.JSONDecodeError:
                    continue

    return ids, texts, metadatas


def export_topic_summary(topic_model: BERTopic, output_dir: Path) -> None:
    topic_info = topic_model.get_topic_info()

    top_words = []
    for topic_id in topic_info["Topic"]:
        if topic_id == -1:
            top_words.append("")
        else:
            words = topic_model.get_topic(topic_id)
            top_words.append(", ".join(w for w, _ in words[:10]) if words else "")

    topic_info["top_words"] = top_words
    topic_info.to_csv(output_dir / "topic_summary.csv", index=False)


def export_topic_documents(
    texts: List[str],
    ids: List[str],
    topics: List[int],
    metadatas: List[dict],
    output_dir: Path,
) -> None:
    topic_map: Dict[int, List[int]] = {}

    for i, topic_id in enumerate(topics):
        topic_map.setdefault(topic_id, []).append(i)

    for topic_id, indices in topic_map.items():
        rows = []
        for i in indices:
            md = metadatas[i] if i < len(metadatas) else {}
            rows.append({
                "chunk_id": ids[i],
                "text": texts[i],
                "title": md.get("title", ""),
                "authors": md.get("authors", ""),
                "author_count": md.get("author_count", ""),
                "publication_year": md.get("publication_year", ""),
                "source_pdf": md.get("source_pdf", ""),
                "source_pdf_stem": md.get("source_pdf_stem", ""),
            })

        df = pd.DataFrame(rows)

        filename = (
            f"topic_{topic_id}_docs.csv"
            if topic_id != -1
            else "topic_outliers_docs.csv"
        )
        df.to_csv(output_dir / filename, index=False)


def export_document_metadata_table(metadatas: List[dict], output_dir: Path) -> None:
    rows = []
    seen = set()

    for md in metadatas:
        key = md.get("source_pdf", "")
        if not key or key in seen:
            continue
        seen.add(key)

        rows.append({
            "source_pdf": md.get("source_pdf", ""),
            "source_pdf_stem": md.get("source_pdf_stem", ""),
            "title": md.get("title", ""),
            "authors": md.get("authors", ""),
            "author_count": md.get("author_count", ""),
            "publication_year": md.get("publication_year", ""),
            "pdf_embedded_title": md.get("pdf_embedded_title", ""),
            "pdf_embedded_author": md.get("pdf_embedded_author", ""),
        })

    pd.DataFrame(rows).to_csv(output_dir / "document_metadata.csv", index=False)


def main() -> None:
    config = load_config()
    logger = get_logger("export", config)

    manifests_folder = Path(config["manifests_folder"])
    topics_dir = Path(config.get("topics_dir", "outputs/topics"))
    model_path = topics_dir / "bertopic_model.pkl"

    logger.info("=== Export started ===")

    if not model_path.exists():
        logger.error(f"No model found at '{model_path}'. Run modeling first.")
        return

    logger.info(f"Loading model from '{model_path}' ...")
    with open(model_path, "rb") as f:
        topic_model = pickle.load(f)

    ids, texts, metadatas = load_records_from_manifests(manifests_folder)

    if not texts:
        logger.error("No texts found. Run ingest.py first.")
        return

    logger.info(f"Loaded {len(texts):,} texts. Running transform...")
    topics, _ = topic_model.transform(texts)

    topics_dir.mkdir(parents=True, exist_ok=True)

    export_topic_summary(topic_model, topics_dir)
    export_topic_documents(texts, ids, topics, metadatas, topics_dir)
    export_document_metadata_table(metadatas, topics_dir)

    logger.info("=== Export complete ===")


if __name__ == "__main__":
    main()