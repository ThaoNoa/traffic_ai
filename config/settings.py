"""
Config loader - Đọc config.yaml và expose ra toàn bộ project.
Pattern: Singleton, load 1 lần duy nhất.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

# ─── Path gốc của project ──────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML config file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


class Config:
    """
    Wrapper class để access config bằng dot notation.
    Ví dụ: cfg.detector.confidence_threshold thay vì cfg['detector']['confidence_threshold']
    """
    
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                # Convert list of lists thành list of tuples (cho points)
                setattr(self, key, [tuple(v) for v in value])
            else:
                setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# ─── Global config instance ────────────────────────────────
_cfg_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Lấy config instance (singleton pattern).
    Gọi get_config() ở bất kỳ đâu trong project.
    """
    global _cfg_instance
    if _cfg_instance is None:
        raw = load_config(config_path)
        _cfg_instance = Config(raw)
    return _cfg_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Force reload config (dùng khi debug)."""
    global _cfg_instance
    _cfg_instance = None
    return get_config(config_path)