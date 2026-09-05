from __future__ import annotations

import sys
from pathlib import Path

CTRL = Path(__file__).resolve().parents[1]
if str(CTRL) not in sys.path:
    sys.path.insert(0, str(CTRL))
