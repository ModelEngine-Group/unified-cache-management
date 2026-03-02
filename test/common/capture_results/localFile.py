import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)


# Global state
_backup_path: Optional[Path] = None
_test_build_id: Optional[str] = None


def _get_backup_config() -> Dict[str, Any]:
    default_path = "results/"
    default_config = {"backup": default_path, "enabled": True}

    try:
        # Lazy import to avoid potential circular dependencies
        from common.config_utils import config_utils as config_instance

        results_config = config_instance.get_config("results", [])
        if not isinstance(results_config, list):
            logger.warning("Config 'results' is not a list, using defaults.")
            return default_config

        for item in results_config:
            if isinstance(item, dict) and "localFile" in item:
                # Safe navigation for nested dictionary
                path = item.get("localFile", {}).get("path", default_path)
                logger.debug(f"Found localFile config with path: {path}")
                return {"backup": path, "enabled": True}

        logger.info("No 'localFile' configuration found, using defaults.")
        return default_config

    except Exception as e:
        logger.warning(f"Failed to load config, using defaults: {e}")
        return default_config


def _initialize_backup_path() -> Path:
    global _backup_path
    if _backup_path is not None:
        return _backup_path

    config = _get_backup_config()
    backup_str = config.get("backup", "results/")
    _backup_path = Path(backup_str).resolve()

    try:
        _backup_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create backup directory {_backup_path}: {e}")
        # Fallback to current working directory
        _backup_path = Path("results/").resolve()
        _backup_path.mkdir(parents=True, exist_ok=True)

    return _backup_path


def _write_to_jsonl(table_name: str, data: Dict[str, Any]) -> bool:
    try:
        backup_path = _initialize_backup_path()
    except Exception as e:
        logger.error(f"Backup path initialization failed: {e}")
        return False

    file_path = backup_path / f"{table_name}.jsonl"

    try:
        record = data.copy()
        with file_path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write("\n")

        logger.info(f"Data written to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}", exc_info=True)
        return False


def write_results(table_name: str, data: Dict[str, Any]) -> bool:
    return _write_to_jsonl(table_name, data)


def jsonl_to_csv(
    jsonl_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
    flatten: bool = False,
) -> Path:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    csv_path = Path(csv_path) if csv_path else jsonl_path.with_suffix(".csv")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if flatten:
                    record = _flatten_dict(record)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")

    if not records:
        raise ValueError(f"No valid records found in {jsonl_path}")

    # Efficiently collect unique fieldnames preserving order
    fieldnames_set = set()
    fieldnames = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames_set:
                fieldnames_set.add(key)
                fieldnames.append(key)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Converted {len(records)} records to {csv_path}")
    return csv_path


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    PRJ_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PRJ_ROOT))
    write_results("test_results", {"a": 4, "b": "hello"})
    write_results("test_results", {"a": 4, "b": '{"a": 4, "b": "hello"}'})
    jsonl_to_csv("results/test_results.jsonl")
