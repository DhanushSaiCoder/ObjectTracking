"""Compat package to expose BoT-SORT FastReID as `fast_reid.fastreid`."""
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FASTREID_PARENT = _REPO_ROOT / "third_party" / "bot_sort" / "fast_reid"

if _FASTREID_PARENT.exists():
    # Make the BoT-SORT fast_reid directory discoverable as this package's path.
    __path__ = [str(_FASTREID_PARENT)]
    if str(_FASTREID_PARENT) not in sys.path:
        sys.path.insert(0, str(_FASTREID_PARENT))
else:
    __path__ = []
