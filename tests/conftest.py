"""Make the assignment directory importable regardless of pytest's rootdir."""

import sys
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))
