"""Logging to a file, with progress bars on the console only.

`setup_logging(path)` sends log records to `path` and, through `tqdm.write`, to the console without breaking
progress bars. `progress(...)` is tqdm on stderr; bars are never written to the log file.
"""

from __future__ import annotations

import logging
import sys
from functools import partial
from pathlib import Path

from tqdm import tqdm

LOGGER_NAME = "landslide"
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

progress = partial(tqdm, file=sys.stderr, dynamic_ncols=True, mininterval=0.5)


class _TqdmConsoleHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """A child of the package logger (`landslide.<name>`), or the package logger itself."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(log_file: Path | str | None = None, console: bool = True,
                  level: int = logging.INFO) -> logging.Logger:
    """Configure the package logger once per process. Re-calling replaces the handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter(_FMT, _DATEFMT)
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if console:
        ch = _TqdmConsoleHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
