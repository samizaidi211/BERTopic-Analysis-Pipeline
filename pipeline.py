"""
run_pipeline.py

Run the full thematic analysis pipeline end-to-end.

Order:
    1. ingest.py --reindex
    2. bertopic_modeling.py
    3. export.py
    4. metadata_enrichment.py
    5. figures.py

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent


def run_step(name: str, command: list[str]) -> None:
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {name}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(command, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")

    print(f"\n✅ COMPLETED: {name}\n")


def main() -> None:
    python_exe = sys.executable

    steps = [
        ("Ingest PDFs", [python_exe, "ingest.py", "--reindex"]),
        ("Train BERTopic", [python_exe, "bertopic_modeling.py"]),
        ("Export topics", [python_exe, "export.py"]),
        ("Enrich metadata", [python_exe, "metadata_enrichment.py"]),
    ]

    try:
        for name, command in steps:
            run_step(name, command)

        print("\n🎉 FULL PIPELINE COMPLETE\n")

    except Exception as exc:
        print(f"\n❌ PIPELINE STOPPED: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()