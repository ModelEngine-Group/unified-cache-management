import ctypes
import importlib
import os
from pathlib import Path

_metrics_handle = None
_metrics_lib = Path(__file__).with_name("libmetrics.so")
if _metrics_lib.exists():
    _metrics_handle = ctypes.CDLL(
        str(_metrics_lib),
        mode=getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0),
    )
ucmmetrics = importlib.import_module(f"{__name__}.ucmmetrics")
