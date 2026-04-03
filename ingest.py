# ingest.py
"""
Ingest academic PDFs, preprocess into chunks, embed with Ollama, and store in Chroma.
Now includes paper-level metadata in each stored record and manifest row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from llama_index.embeddings.ollama import OllamaEmbedding

from config import get_logger, load_config
from pdf_preprocess import preprocess_pdf


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def is_context_length_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        phrase in msg
        for phrase in ("exceeds the context length", "context length", "input length")
    )


def split_text_hard(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    parts: List[str] = []
    start, n = 0, len(t)
    while start < n:
        end = min(n, start + max_chars)
        part = t[start:end].strip()
        if part:
            parts.append(part)
        if end >= n:
            break
        start = max(0, end - overlap_chars)

    return parts


def embed_with_recursive_split(
    embedding_model: OllamaEmbedding,
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    max_depth: int,
    stats: Dict[str, int],
) -> List[Tuple[str, List[float]]]:
    t = (text or "").strip()
    if not t or len(t) < min_chars:
        return []

    if len(t) > max_chars:
        stats["recursive_splits"] += 1
        results: List[Tuple[str, List[float]]] = []
        for part in split_text_hard(t, max_chars=max_chars, overlap_chars=overlap_chars):
            results.extend(
                embed_with_recursive_split(
                    embedding_model, part, max_chars, overlap_chars, min_chars, max_depth, stats
                )
            )
        return results

    stats["embed_attempts"] += 1
    try:
        vector = embedding_model.get_text_embedding(t)
        stats["embed_success"] += 1
        return [(t, vector)]
    except Exception as exc:
        if is_context_length_error(exc):
            stats["embed_context_errors"] += 1
        else:
            stats["embed_other_errors"] += 1
            raise

    if max_depth <= 0 or len(t) <= 2 * min_chars:
        return []

    stats["recursive_splits"] += 1
    mid = len(t) // 2
    left_results = embed_with_recursive_split(
        embedding_model, t[:mid].strip(), max_chars, overlap_chars, min_chars, max_depth - 1, stats
    )
    right_results = embed_with_recursive_split(
        embedding_model, t[mid:].strip(), max_chars, overlap_chars, min_chars, max_depth - 1, stats
    )

    rescued = left_results + right_results
    if len(rescued) >= 2:
        stats["recursive_rescued_parts"] += len(rescued) - 1
    return rescued


def ingest_papers(reindex: bool = False) -> None:
    config = load_config()
    logger = get_logger("ingest", config)

    pdf_folder = Path(config["pdf_folder"])
    manifests_folder = Path(config["manifests_folder"])
    chroma_db_path = Path(config["chroma_db_path"])
    collection_name = config.get("collection_name", "papers")

    chunk_size = int(config.get("chunk_size", 900))
    chunk_overlap = int(config.get("chunk_overlap", 150))

    max_embed_chars = int(config.get("max_embed_tokens", 1200)) * 4
    overlap_chars = int(config.get("embed_overlap_chars", 300))
    min_chunk_chars = int(config.get("min_chunk_chars", 20))
    max_depth = int(config.get("max_recursive_depth", 10))

    if not pdf_folder.exists():
        logger.error(f"PDF folder does not exist: {pdf_folder}")
        return

    manifests_folder.mkdir(parents=True, exist_ok=True)
    chroma_db_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    embed_model_name = config["ollama"]["embedding_model"]
    embedding_model = OllamaEmbedding(model_name=embed_model_name)

    overall: Dict[str, int] = {
        "pdfs_seen": 0, "pdfs_skipped_manifest": 0,
        "pdfs_failed_preprocess": 0, "pdfs_added": 0,
        "base_chunks": 0, "base_chunks_skipped_empty": 0,
        "base_chunks_skipped_short": 0, "subtexts_total": 0,
        "records_added": 0, "embed_fail_records": 0,
        "embed_attempts": 0, "embed_success": 0,
        "embed_context_errors": 0, "embed_other_errors": 0,
        "recursive_splits": 0, "recursive_rescued_parts": 0,
    }

    for pdf_path in sorted(pdf_folder.glob("*.pdf")):
        overall["pdfs_seen"] += 1
        logger.info(f"Processing {pdf_path.name}")

        manifest_path = manifests_folder / f"{pdf_path.stem}.jsonl"
        if not reindex and manifest_path.exists():
            overall["pdfs_skipped_manifest"] += 1
            logger.info(f"Skipping (manifest exists): {pdf_path.name}")
            continue

        try:
            doc_metadata, base_chunks = preprocess_pdf(
                pdf_path, chunk_size=chunk_size, overlap=chunk_overlap
            )
        except Exception as exc:
            overall["pdfs_failed_preprocess"] += 1
            logger.error(f"Preprocessing failed for {pdf_path.name}: {exc}")
            continue

        overall["base_chunks"] += len(base_chunks)

        embed_stat_keys = (
            "embed_attempts", "embed_success", "embed_context_errors",
            "embed_other_errors", "recursive_splits", "recursive_rescued_parts",
        )
        before = {k: overall[k] for k in embed_stat_keys}

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []
        embeddings: List[List[float]] = []

        skipped_empty = skipped_short = subtexts_count = embed_fails = 0

        for chunk_id, chunk_text in base_chunks:
            chunk_text = (chunk_text or "").strip()

            if not chunk_text:
                skipped_empty += 1
                overall["base_chunks_skipped_empty"] += 1
                continue
            if len(chunk_text) < min_chunk_chars:
                skipped_short += 1
                overall["base_chunks_skipped_short"] += 1
                continue

            subtexts = split_text_hard(chunk_text, max_chars=max_embed_chars, overlap_chars=overlap_chars)
            subtexts_count += len(subtexts)
            overall["subtexts_total"] += len(subtexts)

            for j, subtext in enumerate(subtexts):
                subtext = subtext.strip()
                if len(subtext) < min_chunk_chars:
                    skipped_short += 1
                    overall["base_chunks_skipped_short"] += 1
                    continue

                full_id = f"{pdf_path.stem}::{chunk_id}::s{j:02d}"

                try:
                    embedded_parts = embed_with_recursive_split(
                        embedding_model=embedding_model,
                        text=subtext,
                        max_chars=max_embed_chars,
                        overlap_chars=overlap_chars,
                        min_chars=max(200, min_chunk_chars),
                        max_depth=max_depth,
                        stats=overall,
                    )
                except Exception as exc:
                    embed_fails += 1
                    overall["embed_fail_records"] += 1
                    logger.error(f"Embedding error for {full_id}: {exc}")
                    continue

                if not embedded_parts:
                    embed_fails += 1
                    overall["embed_fail_records"] += 1
                    logger.warning(f"No embedding produced for {full_id} (context too long after splitting).")
                    continue

                for k, (part_text, vector) in enumerate(embedded_parts):
                    part_id = f"{full_id}::r{k:02d}"
                    ids.append(part_id)
                    documents.append(part_text)
                    embeddings.append(vector)

                    metadata = {
                        "source_pdf": pdf_path.name,
                        "chunk_id": chunk_id,
                        "subtext_index": j,
                        "part_index": k,
                        "title": str(doc_metadata.get("title", "")),
                        "authors": str(doc_metadata.get("authors", "")),
                        "author_count": int(doc_metadata.get("author_count", 0) or 0),
                        "publication_year": str(doc_metadata.get("publication_year", "")),
                        "source_pdf_stem": str(doc_metadata.get("source_pdf_stem", "")),
                        "pdf_embedded_title": str(doc_metadata.get("pdf_embedded_title", "")),
                        "pdf_embedded_author": str(doc_metadata.get("pdf_embedded_author", "")),
                    }
                    metadatas.append(metadata)

        overall["records_added"] += len(ids)

        after = {k: overall[k] for k in embed_stat_keys}
        logger.info(
            f"Chunk stats for {pdf_path.name}: "
            f"base={len(base_chunks)}, subtexts={subtexts_count}, added={len(ids)}, "
            f"skipped_empty={skipped_empty}, skipped_short={skipped_short}, "
            f"embed_fail={embed_fails}, "
            f"context_errors={after['embed_context_errors'] - before['embed_context_errors']}, "
            f"recursive_splits={after['recursive_splits'] - before['recursive_splits']}"
        )

        if not ids:
            logger.warning(f"No records to add for {pdf_path.name}; skipping DB write.")
            continue

        if reindex:
            try:
                collection.delete(ids=ids)
            except Exception:
                pass

        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        with manifest_path.open("w", encoding="utf-8") as f:
            for i in range(len(ids)):
                json.dump({"id": ids[i], "text": documents[i], "metadata": metadatas[i]}, f, ensure_ascii=False)
                f.write("\n")

        overall["pdfs_added"] += 1
        logger.info(
            f"Added {len(ids)} records from {pdf_path.name} "
            f"(title='{doc_metadata.get('title', '')}', year='{doc_metadata.get('publication_year', '')}')."
        )

    logger.info("=== INGEST SUMMARY ===")
    logger.info(f"  PDFs seen:                {overall['pdfs_seen']}")
    logger.info(f"  PDFs skipped (manifest):  {overall['pdfs_skipped_manifest']}")
    logger.info(f"  PDFs failed preprocessing:{overall['pdfs_failed_preprocess']}")
    logger.info(f"  PDFs added to DB:         {overall['pdfs_added']}")
    logger.info(f"  Base chunks produced:     {overall['base_chunks']}")
    logger.info(f"  Chunks skipped (empty):   {overall['base_chunks_skipped_empty']}")
    logger.info(f"  Chunks skipped (short):   {overall['base_chunks_skipped_short']}")
    logger.info(f"  Subtexts after hard split:{overall['subtexts_total']}")
    logger.info(f"  Records added to Chroma:  {overall['records_added']}")
    logger.info(f"  Embed failures:           {overall['embed_fail_records']}")
    logger.info(f"  Embed attempts:           {overall['embed_attempts']}")
    logger.info(f"  Embed successes:          {overall['embed_success']}")
    logger.info(f"  Context-length errors:    {overall['embed_context_errors']}")
    logger.info(f"  Recursive splits:         {overall['recursive_splits']}")
    logger.info(f"  Rescued parts:            {overall['recursive_rescued_parts']}")
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDFs and store embeddings in Chroma.")
    parser.add_argument("--reindex", action="store_true", help="Re-ingest all PDFs, ignoring existing manifests.")
    args = parser.parse_args()

    ingest_papers(reindex=args.reindex)