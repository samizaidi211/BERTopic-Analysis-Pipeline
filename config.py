"""
config.py

Central configuration loader and shared logger factory for the pipeline.

All modules call load_config() to retrieve settings from config.yaml and
get_logger(name) to obtain a consistently formatted logger.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file (relative to cwd or absolute).

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, config: dict | None = None) -> logging.Logger:
    """
    Return a named logger with a file handler (in logs_folder) and a stream handler.

    If config is provided and contains a 'logs_folder' key, log files are written
    there. Otherwise logs go only to stdout.

    Args:
        name:   Logger name (typically the module name, e.g. 'ingest').
        config: Optional config dict from load_config().

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Guard against duplicate handlers if get_logger is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    # Console handler — always present
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler — only if logs_folder is configured
    if config:
        logs_folder = Path(config.get("logs_folder", "data/logs"))
        logs_folder.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logs_folder / f"{name}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
