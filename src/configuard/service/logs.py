"""Structured JSON logs keyed by request ID. Callers pass only safe fields:
never filenames, paths, media bytes, or API keys."""

from __future__ import annotations

import contextvars
import json
import logging
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
LOGGER_NAME = "configuard.service"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
               "level": record.levelname, "event": record.getMessage(),
               "request_id": getattr(record, "request_id", None) or request_id_var.get()}
        out.update(getattr(record, "fields", {}) or {})
        return json.dumps(out, default=str)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: str) -> None:
    log = get_logger()
    log.setLevel(level.upper())
    if not any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
    log.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    get_logger().log(level, event, extra={"fields": fields})
