from __future__ import annotations

import logging
from pathlib import Path


def configure_run_logger(path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("daic_foundation_tab")
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
