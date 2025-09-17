from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "pixspector", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def set_level(logger: logging.Logger, level: int) -> None:
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
