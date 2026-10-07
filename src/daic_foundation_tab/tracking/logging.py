from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread


def configure_run_logger(path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("daic_foundation_tab")
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@contextmanager
def phase(logger: logging.Logger, name: str):
    started = time.perf_counter()
    stopped = Event()
    logger.info("phase=%s status=started elapsed=0s ETA unavailable", name)

    def heartbeat() -> None:
        while not stopped.wait(30):
            logger.info(
                "phase=%s status=running elapsed=%.1fs ETA unavailable",
                name,
                time.perf_counter() - started,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield started
    except Exception:
        logger.exception("phase=%s status=failed elapsed=%.1fs", name, time.perf_counter() - started)
        raise
    else:
        logger.info("phase=%s status=complete elapsed=%.1fs", name, time.perf_counter() - started)
    finally:
        stopped.set()
        thread.join()


def log_progress(
    logger: logging.Logger, name: str, completed: int, total: int, started: float
) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (total - completed) / completed if completed else None
    unit = {"preparing features": "participants", "training": "epochs", "repeated holdout": "repeats", "bootstrapping development": "resamples"}.get(name, "items")
    logger.info(
        "phase=%s progress=%s/%s %s (%.1f%%) elapsed=%.1fs throughput=%.2f %s/s ETA=%s",
        name,
        completed,
        total,
        unit,
        100 * completed / total if total else 0,
        elapsed,
        completed / elapsed if elapsed > 0 else 0,
        unit,
        f"{eta:.1f}s" if eta is not None else "unavailable",
    )
