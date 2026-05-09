"""
Logger module - Logging chuẩn cho toàn hệ thống.
Mọi module dùng get_logger(__name__) để tạo logger riêng.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import get_config, PROJECT_ROOT


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Tạo logger với format chuẩn, ghi ra cả console và file.
    
    Args:
        name: Tên logger (thường là __name__ của module)
        log_level: "DEBUG", "INFO", "WARNING", "ERROR"
    
    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Tránh duplicate handlers nếu gọi lại
    if logger.handlers:
        return logger
    
    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Handler 1: Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler 2: File
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"traffic_ai_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Shortcut để lấy logger. Gọi ở đầu mỗi module."""
    try:
        cfg = get_config()
        log_level = cfg.system.log_level
    except Exception:
        log_level = "INFO"
    return setup_logger(name, log_level)