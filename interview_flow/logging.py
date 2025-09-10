from __future__ import annotations

import logging
import os
import sys


# Define THINKING level and wire convenience method
THINKING_LEVEL = logging.DEBUG + 5
logging.addLevelName(THINKING_LEVEL, "THINKING")
setattr(logging, "THINKING", THINKING_LEVEL)


def _thinking(self, msg, *args, **kwargs):
    if self.isEnabledFor(THINKING_LEVEL):
        self._log(THINKING_LEVEL, msg, args, **kwargs)


setattr(logging.Logger, "thinking", _thinking)


class _ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\x1b[36m",      # cyan
        "INFO": "\x1b[32m",       # green
        "WARNING": "\x1b[33m",    # yellow
        "ERROR": "\x1b[31m",      # red
        "CRITICAL": "\x1b[35m",   # magenta
        "THINKING": "\x1b[38;5;208m",  # orange
    }
    RESET = "\x1b[0m"

    def __init__(self, fmt: str, datefmt=None, use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # Make a shallow copy of the record with colored levelname if enabled
        if self.use_color and record.levelname in self.COLORS:
            colored = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record = logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=record.msg,
                args=record.args,
                exc_info=record.exc_info,
            )
            record.levelname = colored
        return super().format(record)


_configured = False


def _configure_root(level_name: str | None = None) -> logging.Logger:
    global _configured
    level_env = (level_name or os.getenv("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_env, logging.INFO)
    no_color = os.getenv("LOG_NO_COLOR") == "1"
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False

    base = logging.getLogger("app")
    base.setLevel(level)
    base.propagate = False
    if not base.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        handler.setFormatter(_ColorFormatter(fmt, datefmt=datefmt, use_color=(not no_color and is_tty)))
        base.addHandler(handler)
    _configured = True
    return base


def get_logger(name: str | None = None, *, level_name: str | None = None) -> logging.Logger:
    base = logging.getLogger("app")
    if not _configured or not base.handlers:
        base = _configure_root(level_name)
    if name:
        return base.getChild(name)
    return base


# Default shared logger for simple imports
logger = get_logger()
