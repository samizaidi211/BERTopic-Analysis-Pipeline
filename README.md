# LLM Thematic Analysis Pipeline

A Python pipeline for LLM-informed thematic analysis of medical literature. As a demonstration, this project analyses the thematic landscape of the *Journal of Craniofacial Surgery* over the last 30 years.

---

## Pipeline Overview

```
data/raw/*.pdf
      │
      ▼
Order:
    1. ingest.py --reindex
    2. bertopic_modeling.py
    3. export.py
    4. metadata_enrichment.py
    5. figures.py
```

---

## Setup

```bash
pip install -r requirements.txt
```

You also need [Ollama](https://ollama.com) running locally with the embedding model pulled:

```bash
ollama pull mxbai-embed-large
```

---

## Usage

All scripts are run from inside the `src/` directory:

```bash
cd src

# 1. Ingest PDFs (place PDFs in data/raw/ first)
python ingest.py

# Force re-ingest even if manifests already exist:
python ingest.py --reindex

# 2. Train BERTopic
python bertopic_modeling.py

# 3. Export results
python export.py
```

---

## Configuration

All tuneable parameters live in `src/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `ollama.embedding_model` | `mxbai-embed-large` | Ollama model used for chunk embeddings |
| `ollama.llm_model` | `llama3.1:8b` | Ollama LLM (reserved for future query module) |
| `chroma_db_path` | `data/chroma_db/` | Persistent Chroma vector store location |
| `collection_name` | `papers` | Chroma collection name |
| `pdf_folder` | `data/raw/` | Input PDF directory |
| `manifests_folder` | `data/manifests/` | JSONL manifest output directory |
| `topics_dir` | `outputs/topics/` | BERTopic model and export output directory |
| `logs_folder` | `data/logs/` | Log file directory |
| `chunk_size` | `900` | Words per chunk |
| `chunk_overlap` | `150` | Word overlap between consecutive chunks |
| `max_embed_tokens` | `1200` | Embedding context window cap (in tokens) |
| `embed_overlap_chars` | `300` | Char overlap when hard-splitting oversized chunks |
| `min_chunk_chars` | `20` | Minimum chunk length to attempt embedding |
| `max_recursive_depth` | `10` | Max recursion depth for context-length fallback splitting |

---

## Module Summary

| File | Role |
|---|---|
| `config.py` | `load_config()` and `get_logger()` shared by all modules |
| `config.yaml` | All runtime parameters |
| `pdf_preprocess.py` | PDF text extraction, cleaning, section tagging, chunking |
| `ingest.py` | Orchestrates preprocessing → embedding → Chroma storage |
| `bertopic_modeling.py` | Loads manifests, trains and saves BERTopic model |
| `export.py` | Exports topic summary and per-topic document CSVs |
| `text_extraction.py` | Shared `clean_text()` utility |
