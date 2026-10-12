#!/usr/bin/env python
"""CLI helper script for the audit stage.

Usage:
    python scripts/audit.py [--config configs/default.yaml] [--raw-data-dir data/raw]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to sys.path to ensure runnable directly without prior installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bonetrace.cli import main

if __name__ == "__main__":
    sys.exit(main(["audit", *sys.argv[1:]]))
