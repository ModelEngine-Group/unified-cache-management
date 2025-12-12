import dataclasses
import functools
from collections.abc import Mapping
from typing import Any, Dict, List

from common.db_utils import write_to_db


def _align_and_split(name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Align a mixed data package (single values and/or lists) and split it into
    """
    if not data:
        return []

    aligned: Dict[str, List[Any]] = {}
    lengths: Dict[str, int] = {}
    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            aligned[k] = list(v)
        else:
            aligned[k] = [v]
        lengths[k] = len(aligned[k])

    max_len = max(lengths.values())

    for k, lst in aligned.items():
        if len(lst) < max_len:
            lst.extend([lst[-1]] * (max_len - len(lst)))

    return [{k: aligned[k][i] for k in aligned} for i in range(max_len)]


def post_process(table_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Unified post-processing entry point. Supports two calling styles:
    """
    results = []
    if "_data" in kwargs:
        name = kwargs.get("_name", table_name)
        results = _align_and_split(name, kwargs["_data"])
        for result in results:
            write_to_db(name, result)
        return results
    return []


def _ensure_list(obj):
    """
    Ensure the object is returned as a list.
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (str, bytes, Mapping)):
        return [obj]
    if hasattr(obj, "__iter__") and not hasattr(obj, "__len__"):  # 如 generator
        return list(obj)
    return [obj]


def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert various object types to a dictionary for DB writing.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "_asdict"):  # namedtuple
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Cannot convert {type(obj)} to dict for DB writing")


def proj_process(table_name: str, **kwargs) -> List[Dict[str, Any]]:
    if "_proj" not in kwargs:
        return []
    name = kwargs.get("_name", table_name)
    raw_input = kwargs["_proj"]
    raw_results = _ensure_list(raw_input)

    processed_results = []
    for result in raw_results:
        try:
            dict_result = _to_dict(result)
            write_to_db(name, dict_result)
            processed_results.append(dict_result)
        except Exception as e:
            raise ValueError(f"Failed to process item in _proj: {e}") from e

    return processed_results


# ---------------- decorator ----------------
def export_vars(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # If the function returns a dict containing '_data' or '_proj', post-process it
        if isinstance(result, dict):
            if "_data" in result:
                return post_process(func.__name__, **result)
            if "_proj" in result:
                return proj_process(func.__name__, **result)
        # Otherwise return unchanged
        return result

    return wrapper


# ---------------- usage examples ----------------
@export_vars
def capture():
    """All single values via 'name' + 'data'"""
    return {"name": "demo", "_data": {"accuracy": 0.1, "loss": 0.3}}


# quick test
if __name__ == "__main__":
    print("capture():      ", capture())
