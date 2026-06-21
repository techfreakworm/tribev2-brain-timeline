"""Pytest bootstrap: make the ``src/`` layout importable without an install.

Adds ``<repo>/src`` to ``sys.path`` so ``import tribescore`` resolves when
running ``pytest`` straight from a checkout (no ``pip install -e .`` needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
