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


# ---------------- decorator ----------------
def export_vars(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # If the function returns a dict containing '_data' or 'data', post-process it
        if isinstance(result, dict):
            if "_data" in result or "data" in result:
                return post_process(func.__name__, **result)
        # Otherwise return unchanged
        return result

    return wrapper


# ---------------- usage examples ----------------
@export_vars
def capture():
    """All single values via 'name' + 'data'"""
    return {"name": "demo", "_data": {"accuracy": 0.1, "loss": 0.3}}


@export_vars
def capture_list():
    """All lists via '_name' + '_data'"""
    return {
        "_name": "demo",
        "_data": {
            "accuracy": [0.1, 0.2, 0.3],
            "loss": [0.1, 0.2, 0.3],
        },
    }


@export_vars
def capture_mix():
    """Mixed single + lists via '_name' + '_data'"""
    return {
        "_name": "demo",
        "_data": {
            "length": 10086,  # single value
            "accuracy": [0.1, 0.2, 0.3],  # list
            "loss": [0.1, 0.2, 0.3],  # list
        },
    }


# quick test
if __name__ == "__main__":
    print("capture():      ", capture())
    print("capture_list(): ", capture_list())
    print("capture_mix():  ", capture_mix())
