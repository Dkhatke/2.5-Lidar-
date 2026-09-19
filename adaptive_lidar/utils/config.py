"""
Config loader — single source of truth.
Reads config.yaml and provides typed access.
"""
import os
import yaml
from typing import Any, Dict

_CONFIG: Dict[str, Any] = {}


def load_config(path: str = None) -> Dict[str, Any]:
    global _CONFIG
    if path is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "config.yaml")
    with open(path, "r") as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG


def get_config() -> Dict[str, Any]:
    global _CONFIG
    if not _CONFIG:
        load_config()
    return _CONFIG


def get(section: str, key: str, default=None):
    cfg = get_config()
    return cfg.get(section, {}).get(key, default)
