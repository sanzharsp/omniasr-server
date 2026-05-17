"""Structured JSON logging for production.

The formatter emits one JSON document per log line with stable keys, so log
aggregators (Loki, Elastic, Datadog) can parse without regex. Falls back to
plain text when `LOG_FORMAT=plain` for local development readability.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from app.middleware import RequestIdFilter


_RESERVED_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "request_id", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        # Surface anything passed via `logger.info("...", extra={...})`.
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install the root logger configuration for the process.

    Honors:
      LOG_LEVEL  (default INFO)
      LOG_FORMAT (json | plain, default json)
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "json").strip().lower()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(RequestIdFilter())

    if fmt == "plain":
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(request_id)s] %(levelname)s %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level_name)

    # uvicorn installs its own handlers — strip them so we have a single sink.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
