"""
Config loader — single source of truth.

Reads config.yaml and provides typed access.  Files are opened as UTF-8
explicitly: Python on Windows defaults to the cp1252 locale encoding, which
fails on any non-ASCII character in the file.
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict

import yaml

_CONFIG: Dict[str, Any] = {}


def default_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "config.yaml")


def load_config(path: str | None = None) -> Dict[str, Any]:
    global _CONFIG
    path = path or default_path()
    with io.open(path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> Dict[str, Any]:
    global _CONFIG
    if not _CONFIG:
        load_config()
    return _CONFIG


def get(section: str, key: str, default=None):
    return get_config().get(section, {}).get(key, default)
