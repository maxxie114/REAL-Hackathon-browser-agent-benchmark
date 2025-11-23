"""Central logging utilities for the arena package."""
from __future__ import annotations

import logging
import os
import re
from typing import Union

_CONFIGURED_SENTINEL = "_arena_logging_configured"
_DATA_URL_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)


class _RedactDataURLFilter(logging.Filter):
    """Redact inline data URLs so debug logs stay readable."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "data:image" not in message:
            return True

        redacted = _DATA_URL_PATTERN.sub("data:image;base64,[redacted]", message)
        record.msg = redacted
        record.args = None
        return True


def _coerce_level(level: Union[str, int, None]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level_name = level.upper()
        return getattr(logging, level_name, logging.INFO)
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def configure_logging(level: Union[str, int, None] = None, *, force: bool = False) -> None:
    """Configure the root logger once, defaulting to LOG_LEVEL or INFO."""

    resolved_level = _coerce_level(level)

    already_configured = getattr(logging, _CONFIGURED_SENTINEL, False)
    if already_configured and not force:
        logging.getLogger(__name__).debug("Logging already configured; skipping reconfiguration")
        return

    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )

    root_logger = logging.getLogger()
    root_logger.addFilter(_RedactDataURLFilter())

    # Quiet noisy third-party loggers that tend to dump raw payloads when DEBUG.
    for noisy_logger in ("openai", "httpx", "httpcore", "patchright"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    setattr(logging, _CONFIGURED_SENTINEL, True)


__all__ = ["configure_logging"]
