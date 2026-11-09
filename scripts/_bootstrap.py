"""Make the `stackprs` package importable when running scripts directly.

Each pipeline script imports this first so you don't need `pip install -e .`.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
