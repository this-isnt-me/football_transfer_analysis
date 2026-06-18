import sys
from pathlib import Path

# Ensure the repo root is importable so `from src.data_layer import ...` works
# regardless of the directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))