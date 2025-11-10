import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import peewee
from common.config_utils import config_utils as config_instance
from peewee import AutoField, Model, MySQLDatabase, TextField

logger = logging.getLogger("db_handler")
logger.setLevel(logging.DEBUG)

# Avoid adding handlers multiple times
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

# Global DB instance and lock for thread-safe singleton
_db_instance: Optional[MySQLDatabase] = None
_db_lock = threading.Lock()
_test_build_id: Optional[str] = None
_backup_path: Optional[Path] = None
_db_enabled: bool = False  # from config


def _get_db() -> Optional[MySQLDatabase]:
    """Return a singleton MySQLDatabase instance based on YAML configuration."""
    global _db_instance, _backup_path, _db_enabled

    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                db_config = config_instance.get_config("database", {})
                _db_enabled = db_config.get("enabled", False)

                backup_str = db_config.get("backup", "results/")
                _backup_path = Path(backup_str).resolve()
                _backup_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Backup directory set to: {_backup_path}")

                if not _db_enabled:
                    return None

                try:
                    _db_instance = MySQLDatabase(
                        db_config.get("name", "test_db"),
                        user=db_config.get("user", "root"),
                        password=db_config.get("password", ""),
                        host=db_config.get("host", "localhost"),
                        port=db_config.get("port", 3306),
                        charset=db_config.get("charset", "utf8mb4"),
                    )
                    logger.info(
                        f"Database instance created for: {_db_instance.database}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create database instance: {e}")
                    _db_instance = None

    return _db_instance


def _set_test_build_id(build_id: Optional[str] = None) -> None:
    """Set or generate a unique test build ID."""
    global _test_build_id
    _test_build_id = build_id or "default_build_id"
    logger.debug(f"Test build ID set to: {_test_build_id}")


def _get_test_build_id() -> str:
    """Return the current test build ID, generating one if necessary."""
    global _test_build_id
    if _test_build_id is None:
        _set_test_build_id()
    return _test_build_id


class BaseEntity(Model):
    """Base PeeWee model class using the singleton database."""

    class Meta:
        database = _get_db()


def _backup_to_file(table_name: str, data: Dict[str, Any]) -> None:
    """Write data to a JSON Lines (.jsonl) file in the backup directory."""
    if not _backup_path:
        logger.warning("Backup path is not set. Skipping backup.")
        return

    file_path = _backup_path / f"{table_name}.jsonl"
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
        logger.info(f"Data backed up to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write backup file {file_path}: {e}")


def write_to_db(table_name: str, data: Dict[str, Any]) -> bool:
    """
    Attempt to insert data into the specified database table.
    If the table doesn't exist or an error occurs, back up to a JSONL file.
    """
    db = _get_db()
    data["test_build_id"] = _get_test_build_id()

    # Skip DB entirely if disabled
    if not _db_enabled or db is None:
        _backup_to_file(table_name, data)
        return False

    try:
        if not db.table_exists(table_name):
            logger.warning(f"Table '{table_name}' does not exist. Writing to backup.")
            _backup_to_file(table_name, data)
            return False

        # Get existing columns and filter data
        columns = db.get_columns(table_name)
        col_names = {col.name for col in columns}
        filtered_data = {k: v for k, v in data.items() if k in col_names}

        # Build dynamic model for insertion
        fields = {"id": AutoField()}
        for col in columns:
            if col.name != "id":
                fields[col.name] = TextField(null=True)

        DynamicEntity = type(
            f"{table_name.capitalize()}DynamicModel",
            (BaseEntity,),
            {
                "Meta": type("Meta", (), {"database": db, "table_name": table_name}),
                **fields,
            },
        )

        with db.atomic():
            DynamicEntity.insert(filtered_data).execute()
        logger.info(f"Successfully inserted data into table '{table_name}'.")
        return True

    except peewee.PeeweeException as e:
        logger.error(
            f"Database write error for table '{table_name}': {e}", exc_info=True
        )
    except Exception as e:
        logger.critical(
            f"Unexpected error during DB write for '{table_name}': {e}", exc_info=True
        )

    # Fallback to backup on any failure
    _backup_to_file(table_name, data)
    return False


def database_connection(build_id: str) -> None:
    """Test database connection and set the build ID."""
    logger.info(f"Setting test build ID: {build_id}")
    _set_test_build_id(build_id)

    db = _get_db()
    if not _db_enabled:
        logger.info("Database connection skipped because enabled=false.")
        return

    if db is None:
        logger.error("No database instance available.")
        return

    logger.info(f"Attempting connection to database: {db.database}")
    try:
        db.connect(reuse_if_open=True)
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}", exc_info=True)
    finally:
        if not db.is_closed():
            db.close()
            logger.debug("Database connection closed.")
