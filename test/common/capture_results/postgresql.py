import logging
import time
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# ============================================================================
# Lazy Dependency Loader
# ============================================================================


class DbDeps:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        from peewee import (
            AutoField,
            BooleanField,
            DateTimeField,
            DoubleField,
            Field,
            Model,
            TextField,
        )
        from playhouse.migrate import PostgresqlMigrator, migrate
        from playhouse.postgres_ext import BinaryJSONField, PostgresqlDatabase

        self.AutoField = AutoField
        self.BooleanField = BooleanField
        self.DateTimeField = DateTimeField
        self.DoubleField = DoubleField
        self.Field = Field
        self.Model = Model
        self.TextField = TextField
        self.BinaryJSONField = BinaryJSONField
        self.PostgresqlDatabase = PostgresqlDatabase
        self.PostgresqlMigrator = PostgresqlMigrator
        self.migrate = migrate

        self._initialized = True


deps = DbDeps()


# ============================================================================
# Global State
# ============================================================================

_db_config: Optional[Dict[str, Any]] = None
_db_instance = None
_schema_checked_tables = set()


# ============================================================================
# Config
# ============================================================================


def _get_pg_config() -> Dict[str, Any]:
    global _db_config

    if _db_config is not None:
        return _db_config

    default = {
        "host": "127.0.0.1",
        "port": 5432,
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "retry": 3,
    }

    try:
        from common.config_utils import config_utils

        results = config_utils.get_config("results", [])
        for item in results:
            if isinstance(item, dict) and "postgresql" in item:
                _db_config = {**default, **item["postgresql"]}
                return _db_config
    except Exception as e:
        logger.warning("Using default DB config: %s", e)

    _db_config = default
    return _db_config


def _get_db():
    global _db_instance
    if _db_instance is None:
        conf = _get_pg_config()
        _db_instance = deps.PostgresqlDatabase(
            database=conf["dbname"],
            host=conf["host"],
            port=conf["port"],
            user=conf["user"],
            password=conf["password"],
        )
    return _db_instance


# ============================================================================
# Schema Utilities
# ============================================================================


def _infer_field(value: Any) -> deps.Field:
    if value is None:
        return deps.TextField(null=True)
    if isinstance(value, bool):
        return deps.BooleanField(null=True)
    if isinstance(value, (int, float)):
        return deps.DoubleField(null=True)
    if hasattr(value, "isoformat"):
        return deps.DateTimeField(null=True)
    if isinstance(value, (dict, list)):
        return deps.BinaryJSONField(null=True)
    return deps.TextField(null=True)


def _get_existing_columns(db, table_name: str):
    cursor = db.execute_sql(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table_name,),
    )
    return {row[0] for row in cursor.fetchall()}


def _ensure_schema(db, table_name: str, record: Dict[str, Any]):

    # Table does not exist -> Create
    if not db.table_exists(table_name):
        logger.info("Creating table '%s'", table_name)

        fields = {
            "id": deps.AutoField(),
        }

        for k, v in record.items():
            if k == "id":
                continue
            fields[k] = _infer_field(v)

        ModelCls = type(
            table_name.capitalize(),
            (deps.Model,),
            {
                **fields,
                "__module__": __name__,
                "Meta": type(
                    "Meta",
                    (),
                    {"database": db, "table_name": table_name},
                ),
            },
        )

        db.create_tables([ModelCls])
        return

    # Table exists -> Check for missing columns
    existing = _get_existing_columns(db, table_name)

    missing = {k: v for k, v in record.items() if k not in existing and k != "id"}

    if not missing:
        return

    logger.info("Adding columns to '%s': %s", table_name, list(missing.keys()))

    migrator = deps.PostgresqlMigrator(db)

    operations = [
        migrator.add_column(table_name, k, _infer_field(v)) for k, v in missing.items()
    ]

    with db.atomic():
        deps.migrate(*operations)


def _create_model(db, table_name: str, record: Dict[str, Any]):

    fields = {
        "id": deps.AutoField(),
    }

    for k, v in record.items():
        if k == "id":
            continue
        fields[k] = _infer_field(v)

    return type(
        table_name.capitalize(),
        (deps.Model,),
        {
            **fields,
            "__module__": __name__,
            "Meta": type(
                "Meta",
                (),
                {"database": db, "table_name": table_name},
            ),
        },
    )


# ============================================================================
# Write Entry
# ============================================================================


def write_results(table_name: str, data: Dict[str, Any]) -> bool:

    db = _get_db()
    retry = _get_pg_config().get("retry", 3)
    record = data.copy()

    for attempt in range(retry):
        try:
            db.connect(reuse_if_open=True)

            if table_name not in _schema_checked_tables:
                _ensure_schema(db, table_name, record)
                _schema_checked_tables.add(table_name)

            ModelCls = _create_model(db, table_name, record)

            with db.atomic():
                pk = ModelCls.insert(**record).execute()

            logger.info(f"Table {table_name} Insert  successful!, ID=%s", pk)
            return True

        except Exception as e:
            logger.error("Insert failed (%d/%d): %s", attempt + 1, retry, e)
            time.sleep(2**attempt)

        finally:
            if not db.is_closed():
                db.close()

    return False


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    PRJ_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PRJ_ROOT))

    mock_config = MagicMock()
    mock_config.get_config.return_value = [
        {
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "dbname": "ucm_test",
                "user": "postgres",
                "password": "123456",
                "retry": 3,
            }
        }
    ]

    with patch("common.config_utils.config_utils", mock_config):
        test_data = {
            "status": "false",
            "input": 4000,
            "tpot": 0.25,
            "e2e": 0.6,
            "metrics": {"latency": 0.05, "success": True},
            "tags": ["api", "v1"],
        }
        print("Test data: %s", test_data)
        write_results("test_db", test_data)
        print("Test data: %s", test_data)
        result = write_results("test_db", test_data)

        print("\nFinal result: %s" % ("SUCCESS" if result else "FAILED"))
