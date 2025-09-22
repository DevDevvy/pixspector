from __future__ import annotations

import logging
from typing import Optional, Dict, Any

ROOT_LOGGER_NAME = "pixspector"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger under the 'pixspector' namespace for tools log window integration.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not root.handlers:
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.INFO)

    if name and name.startswith(ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}" if name else ROOT_LOGGER_NAME)


def log_analysis_step(logger: logging.Logger, component: str, message: str, score: Optional[float] = None, **details):
    """
    Log analysis steps with structured information for the tools log window.
    """
    try:
        extra_info = {"component": component, "details": details}
        if score is not None:
            extra_info["score"] = score
        logger.info(f"[{component.upper()}] {message}", extra=extra_info)
    except Exception:
        # Fail-safe logging
        logger.info(f"[{component.upper()}] {message} | score={score} | details={details}")


def log_scoring_decision(logger: logging.Logger, evidence_type: str, points: int, reason: str, value: Optional[float] = None):
    """
    Log scoring decisions for transparency.
    """
    value_str = f" (value={value:.3f})" if value is not None else ""
    logger.info(f"[SCORING] +{points} points for '{evidence_type}': {reason}{value_str}")


def log_step(logger: logging.Logger, component: str, message: str, level: int = logging.INFO, **details):
    """
    Helper to emit structured step logs. Details are included in the log record's 'extra'.
    """
    try:
        logger.log(level, f"[{component}] {message}", extra={"component": component, "details": details})
    except Exception:
        # Be fail-safe; never let logging break the pipeline.
        logger.log(level, f"[{component}] {message} | details={details}")


def set_level(logger: logging.Logger, level: int) -> None:
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
