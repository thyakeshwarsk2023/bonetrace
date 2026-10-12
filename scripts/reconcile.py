#!/usr/bin/env python
"""CLI helper script for Phase 1.5 reconciliation and annotation verification.

Usage:
    python scripts/reconcile.py [--config configs/default.yaml] [--raw-data-dir data/raw]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bonetrace.config import load_config
from bonetrace.data.reconcile import run_reconciliation

if __name__ == "__main__":
    cfg = load_config()
    res = run_reconciliation(cfg)
    print("Phase 1.5 Reconciliation completed successfully.")
