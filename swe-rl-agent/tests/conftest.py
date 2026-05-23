import os
import sys
from pathlib import Path

# Make src importable without install.
ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Force settings to skip .env load in tests.
os.environ.setdefault("LOG_LEVEL", "WARNING")
