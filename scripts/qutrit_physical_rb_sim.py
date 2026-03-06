#!/usr/bin/env python3
"""CLI wrapper for the physical qutrit RB simulation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qutrit_rb.qutrit_physical_rb_sim import main


if __name__ == "__main__":
    main()
