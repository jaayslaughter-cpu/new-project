"""
Structured JSON logging for Railway.

Railway captures stdout and displays it in the dashboard. Using JSON
format means each log line is searchable and filterable by level/module.

Usage (in __main__.py or at startup):
    from options_bot.logging_config import setup_logging
    setup_logging()
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for Railway log ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts":      datetime.now(tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log)


def setup_logging(level: str | None = None) -> None:
    """
    Configure root logger with JSON output to stdout.

    Parameters
    ----------
    level : str or None
        Log level string (DEBUG, INFO, WARNING, ERROR).
        Falls back to LOG_LEVEL env var, then INFO.
    """
    resolved_level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    # Remove any existing handlers
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "yfinance", "peewee", "apscheduler.executors"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised at level %s", resolved_level
    )
