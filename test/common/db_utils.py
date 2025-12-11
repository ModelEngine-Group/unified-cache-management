import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Lazy imports for database components
peewee = None
PostgresqlDatabase = None
Model = None
AutoField = None
DateTimeField = None
TextField = None

logger = logging.getLogger("db_handler")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    # Basic config only once
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Global state
_db_instance = None
_db_lock = threading.Lock()
_test_build_id: Optional[str] = None
_backup_path: Optional[Path] = None
_db_enabled: bool = False


def _ensure_peewee_imported():
    """Import peewee components only when DB is enabled."""
    global peewee, PostgresqlDatabase, Model, AutoField, DateTimeField, TextField
    if peewee is None:
        import peewee
        from peewee import AutoField as _AF
        from peewee import DateTimeField as _DTF
        from peewee import Model as _Model
        from peewee import PostgresqlDatabase as _PGDB
        from peewee import TextField as _TF

        PostgresqlDatabase = _PGDB
        Model = _Model
        AutoField = _AF
        DateTimeField = _DTF
        TextField = _TF


def _get_db():
    """Return a singleton PostgresqlDatabase instance if enabled."""
    global _db_instance, _backup_path, _db_enabled

    if _db_instance is not None:
        return _db_instance

    with _db_lock:
        if _db_instance is not None:
            return _db_instance

        db_config = _get_db_config()
        _db_enabled = db_config.get("enabled", False)

        backup_str = db_config.get("backup", "results/")
        _backup_path = Path(backup_str).resolve()
        _backup_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory set to: {_backup_path}")

        if not _db_enabled:
            return None

        # Only import peewee when enabled
        _ensure_peewee_imported()

        try:
            _db_instance = PostgresqlDatabase(
                db_config.get("name", "test_db"),
                user=db_config.get("user", "postgres"),
                password=db_config.get("password", ""),
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
            )
            logger.info(
                f"PostgreSQL database instance created for: {_db_instance.database}"
            )
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL database instance: {e}")
            _db_instance = None

    return _db_instance


def _get_db_config():
    """Wrapper to get config without early peewee dependency."""
    from common.config_utils import config_utils as config_instance

    return config_instance.get_config("database", {})


def _set_test_build_id(build_id: Optional[str] = None) -> None:
    global _test_build_id
    _test_build_id = build_id or "default_build_id"
    logger.debug(f"Test build ID set to: {_test_build_id}")


def _get_test_build_id() -> str:
    global _test_build_id
    if _test_build_id is None:
        _set_test_build_id()
    return _test_build_id


def _backup_to_file(table_name: str, data: Dict[str, Any]) -> None:
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
    # Always add build ID
    data["test_build_id"] = _get_test_build_id()

    # Early exit if DB disabled
    db_config = _get_db_config()
    if not db_config.get("enabled", False):
        _backup_to_file(table_name, data)
        return False

    # Load DB and peewee only when needed
    db = _get_db()
    if db is None:
        _backup_to_file(table_name, data)
        return False

    _ensure_peewee_imported()

    try:
        # Check if table exists
        table_exists = db.table_exists(table_name)

        # Get or create dynamic model
        columns = db.get_columns(table_name) if table_exists else []
        col_names = {col.name for col in columns} if table_exists else set()

        # Ensure required fields are present
        all_fields = {"id", "created_at", "test_build_id"}
        all_fields.update(data.keys())

        # Build field definitions
        fields = {
            "id": AutoField(),
            "created_at": DateTimeField(default=datetime.utcnow),
            "test_build_id": TextField(null=True),
        }

        # Add TextField for all other data keys (including future ones)
        for key in data.keys():
            if key not in fields:
                fields[key] = TextField(null=True)

        # Define dynamic model
        Meta = type("Meta", (), {"database": db, "table_name": table_name})

        attrs = {"Meta": Meta, **fields}
        DynamicModel = type(f"{table_name.capitalize()}DynamicModel", (Model,), attrs)

        # Create table if not exists
        if not table_exists:
            db.create_tables([DynamicModel], safe=True)
            logger.info(
                f"Table '{table_name}' created with id, created_at, test_build_id, and dynamic fields."
            )

        # Prepare data for insert (only include fields that exist in model)
        model_fields = set(fields.keys())
        filtered_data = {k: v for k, v in data.items() if k in model_fields}

        # Insert
        with db.atomic():
            DynamicModel.insert(filtered_data).execute()

        logger.info(f"Successfully inserted data into table '{table_name}'.")
        return True

    except Exception as e:
        logger.error(
            f"Error during DB write for table '{table_name}': {e}", exc_info=True
        )
        _backup_to_file(table_name, data)
        return False


def database_connection(build_id: str) -> None:
    logger.info(f"Setting test build ID: {build_id}")
    _set_test_build_id(build_id)

    db_config = _get_db_config()
    if not db_config.get("enabled", False):
        logger.info("Database connection skipped because enabled=false.")
        return

    db = _get_db()
    if db is None:
        logger.error("No database instance available.")
        return

    logger.info(f"Attempting connection to database: {db.database}")
    try:
        db.connect(reuse_if_open=True)
        logger.info("PostgreSQL connection successful.")
    except Exception as e:
        logger.error(f"PostgreSQL connection failed: {e}", exc_info=True)
    finally:
        if not db.is_closed():
            db.close()
            logger.debug("Database connection closed.")
