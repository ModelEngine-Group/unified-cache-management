import threading
from typing import Optional

from common.config_utils import config_utils as config_instance
from peewee import Model, MySQLDatabase

# Global DB instance and lock for thread-safe singleton
_db_instance: Optional[MySQLDatabase] = None
_db_lock = threading.Lock()

# Global build id, unique per pytest session
_test_build_id: Optional[str] = None

_DB_ENABLED = None


def _is_db_enabled() -> bool:
    """Check if DB operations are enabled via config."""
    global _DB_ENABLED
    if _DB_ENABLED is None:
        db_config = config_instance.get_config("database", {})
        _DB_ENABLED = db_config.get("enabled", True)
        if not isinstance(_DB_ENABLED, bool):
            _DB_ENABLED = True
    return _DB_ENABLED


def _get_db() -> Optional[MySQLDatabase]:
    """
    Return a singleton MySQLDatabase instance based on YAML configuration.
    Returns None if DB is disabled.
    """
    global _db_instance
    if not _is_db_enabled():
        return None

    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                db_config = config_instance.get_config("database", {})
                _db_instance = MySQLDatabase(
                    db_config.get("name", "test_db"),
                    user=db_config.get("user", "root"),
                    password=db_config.get("password", ""),
                    host=db_config.get("host", "localhost"),
                    port=db_config.get("port", 3306),
                    charset=db_config.get("charset", "utf8mb4"),
                )
    return _db_instance


def _set_test_build_id(build_id: Optional[str] = None):
    """
    Set or generate a unique test build id.
    This should be called once per pytest session (e.g., in conftest.py).
    """
    global _test_build_id
    if build_id:
        _test_build_id = build_id
    else:
        _test_build_id = "hello"


def _get_test_build_id() -> str:
    """
    Return the current test build id (auto-generated if missing).
    """
    global _test_build_id
    if _test_build_id is None:
        _set_test_build_id()
    return _test_build_id


class BaseModel(Model):
    """Base model for all tables."""

    class Meta:
        database = _get_db()


from typing import Any, Dict

import peewee
from peewee import AutoField, TextField


def write_to_db(table_name: str, data: Dict[str, Any]) -> bool:
    """
    Generic database insert method using dynamic model introspection.
    Automatically appends 'test_build_id' field for traceability.
    If the table does not exist, it will be created with an auto-incrementing 'id' primary key.
    If DB is disabled, this becomes a no-op and returns True.
    """
    if not _is_db_enabled():
        print("[DB] Disabled: Skipping write to %s", table_name)
        return True

    db = _get_db()
    if db is None:
        return False

    try:
        data["test_build_id"] = _get_test_build_id()

        table_exists = table_name in db.get_tables()
        fields = {"id": AutoField()}  # Auto-incrementing primary key

        if not table_exists:
            for key in data.keys():
                if key != "id":  # Avoid conflict with primary key
                    fields[key] = TextField(null=True)  # Or infer type as needed
        else:
            columns = db.get_columns(table_name)
            col_names = {col.name for col in columns}
            for col in columns:
                if col.name == "id":
                    continue
                fields[col.name] = peewee.Field()
            # Filter data to only include known columns
            data = {k: v for k, v in data.items() if k in col_names or k == "id"}

        # Create dynamic model
        DynamicModel = type(
            table_name.capitalize(),
            (BaseModel,),
            {
                "Meta": type("Meta", (), {"database": db, "table_name": table_name}),
                **fields,
            },
        )

        if not table_exists:
            db.create_tables([DynamicModel])

        with db.atomic():
            DynamicModel.insert(data).execute()
        return True

    except peewee.IntegrityError as e:
        print(f"[DB WRITE ERROR] IntegrityError: {e}")
    except peewee.OperationalError as e:
        print(f"[DB WRITE ERROR] OperationalError: {e}")
    except Exception as e:
        print(f"[DB WRITE ERROR] Unexpected error: {e}")
    return False


def database_connection(build_id: str) -> None:
    """Test DB connection and set build ID."""
    print("[DB] Test Build ID: %s", build_id)
    _set_test_build_id(build_id)

    if not _is_db_enabled():
        print("[DB] Disabled: Connection test skipped.")
        return

    db = _get_db()
    if db is None:
        print("[DB] Cannot get DB instance for connection test")
        return

    print("[DB] Connecting to database: %s", db.database)
    try:
        db.connect(reuse_if_open=True)
        print("[DB] Connection OK")
    except Exception as e:
        print("[DB] Connection failed: %s", e)
    finally:
        # Optional: close if you're in a short-lived context (e.g., pytest setup)
        # In long-running apps, avoid closing — let Peewee manage it.
        if not db.is_closed():
            db.close()
