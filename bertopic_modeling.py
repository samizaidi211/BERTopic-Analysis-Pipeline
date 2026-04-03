"""
bertopic_modeling.py

Run BERTopic topic modeling on all ingested PDF chunks.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List

from bertopic import BERTopic
from config import get_logger, load_config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts_from_manifests(manifests_folder: Path) -> List[str]:
    manifest_files = sorted(manifests_folder.glob("*.jsonl"))
    if not manifest_files:
        raise FileNotFoundError(
            f"No manifest files found in '{manifests_folder}'. Run ingest.py first."
        )

    texts: List[str] = []

    for manifest_path in manifest_files:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

    return texts


# ---------------------------------------------------------------------------
# BERTopic training
# ---------------------------------------------------------------------------

def train_bertopic(
    texts: List[str],
    output_path: Path,
    logger: logging.Logger,
) -> BERTopic:
    if not texts:
        raise ValueError("No texts provided for BERTopic training.")

    logger.info(f"Fitting BERTopic on {len(texts):,} documents ...")
    topic_model = BERTopic()
    topic_model.fit(texts)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    logger.info(f"BERTopic found {n_topics} topics (excluding outlier cluster).")

    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / "bertopic_model.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(topic_model, f)

    logger.info(f"Model saved to '{model_file}'.")

    return topic_model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config()
    logger = get_logger("bertopic_modeling", config)

    manifests_folder = Path(config["manifests_folder"])
    topics_dir = Path(config.get("topics_dir", "outputs/topics"))

    logger.info("=== BERTopic modeling started ===")

    try:
        texts = load_texts_from_manifests(manifests_folder)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    logger.info(f"Loaded {len(texts):,} text chunks.")

    try:
        train_bertopic(texts, topics_dir, logger)
    except ValueError as exc:
        logger.error(str(exc))
        return

    logger.info("=== BERTopic modeling complete ===")


if __name__ == "__main__":
    main()