# vergil/utils/logger.py
"""
Structured Logging for VERGIL
==============================

Configures structured JSON logging for all modules.
Console: INFO level (keep Colab output readable).
File: DEBUG level (full detail for replaying episodes).

Log levels used:
- DEBUG: per-step state details (CDG, reward components, extraction results)
- INFO: episode-level events (reset, terminal, curriculum promotion)
- WARNING: unexpected but recoverable (invalid actions, failed edges)
- ERROR: requires investigation (serialization failure, missing profiles)
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(log_dir: str = '/tmp/vergil_logs',
                  level: int = logging.DEBUG,
                  episode_id: Optional[str] = None) -> None:
    """Configure structured JSON logging for all VERGIL modules."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger('vergil')
    root.setLevel(logging.DEBUG)
    root.handlers = []

    # Console handler (INFO only)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    root.addHandler(console)

    # File handler (DEBUG — full detail)
    log_file = log_dir_path / f"vergil_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)

    logging.getLogger('vergil').info(f"Logging initialized. File: {log_file}")


class JSONFormatter(logging.Formatter):
    """Structured JSON log format for machine-readable output."""

    def format(self, record):
        entry = {
            'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S'),
            'level': record.levelname,
            'module': record.name,
            'msg': record.getMessage(),
        }
        if record.exc_info:
            entry['exc'] = self.formatException(record.exc_info)
        return json.dumps(entry)
