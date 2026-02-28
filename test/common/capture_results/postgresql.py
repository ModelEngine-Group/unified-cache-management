import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Type

# Peewee ORM imports
from peewee import (
    BooleanField,
    DateTimeField,
    DoubleField,
    Field,
    Model,
    OperationalError,
    TextField,
    UUIDField,
)
from playhouse.migrate import PostgresqlMigrator, migrate
from playhouse.postgres_ext import BinaryJSONField, PostgresqlDatabase

# ============================================================================
# Database Configuration
# ============================================================================
logger = logging.getLogger(__name__)


# Timezone constant: UTC+8
UTC_8 = timezone(timedelta(hours=8))

# Module-level state
_test_build_id: Optional[str] = None
_db_config: Optional[Dict[str, Any]] = None


def set_build_id(build_id: str) -> None:
    """Set explicit build ID for test tracking."""
    global _test_build_id
    _test_build_id = build_id
    logger.info("Build ID set to: %s", build_id)


def _get_test_build_id() -> str:
    """Get current test build ID, auto-generating if not set."""
    global _test_build_id

    if _test_build_id is None:
        ts = datetime.now(UTC_8)
        _test_build_id = f"build_{ts.strftime('%Y%m%d_%H%M%S')}"
        logger.debug("Auto-generated test build ID: %s", _test_build_id)

    return _test_build_id


def _get_pg_config() -> Dict[str, Any]:
    """Retrieve and cache PostgreSQL configuration with fallback defaults."""
    global _db_config

    if _db_config is not None:
        return _db_config

    default_config = {
        "host": "127.0.0.1",
        "port": 5432,
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
        "retry": 5,
    }

    try:
        from common.config_utils import config_utils as config_instance

        results_config = config_instance.get_config("results", [])

        if not isinstance(results_config, list):
            logger.warning("Config 'results' is not a list; using PostgreSQL defaults.")
            _db_config = default_config
            return _db_config

        for item in results_config:
            if isinstance(item, dict) and "postgresql" in item:
                pg_conf = item.get("postgresql", {})
                _db_config = {**default_config, **pg_conf}
                logger.debug(
                    "Loaded PostgreSQL config: host=%s, retry=%s",
                    _db_config.get("host"),
                    _db_config.get("retry"),
                )
                return _db_config

        logger.info("No 'postgresql' configuration found; using defaults.")

    except Exception as e:
        logger.warning("Failed to load PostgreSQL config, using defaults: %s", e)

    _db_config = default_config
    return _db_config


def _create_db_instance() -> PostgresqlDatabase:
    """Create a new PostgresqlDatabase instance from config."""
    config = _get_pg_config()
    logger.debug(
        "Creating DB connection: %s:%s/%s",
        config.get("host"),
        config.get("port"),
        config.get("dbname"),
    )
    return PostgresqlDatabase(
        database=config.get("dbname"),
        host=config.get("host"),
        port=config.get("port"),
        user=config.get("user"),
        password=config.get("password"),
    )


# ============================================================================
# Schema Inference & Management
# ============================================================================


def _infer_field_type(value: Any, nullable: bool = True) -> Field:
    if value is None:
        return TextField(null=nullable)
    if isinstance(value, bool):
        return BooleanField(null=nullable)
    if isinstance(value, (int, float)):
        return DoubleField(null=nullable)
    if isinstance(value, datetime):
        return DateTimeField(null=nullable)
    if isinstance(value, (dict, list)):
        return BinaryJSONField(null=nullable)
    return TextField(null=nullable)


def _debug_table_schema(
    db: PostgresqlDatabase, table_name: str
) -> Dict[str, Dict[str, Any]]:
    """Fetch current table schema from PostgreSQL for debugging."""
    try:
        cursor = db.execute_sql(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
            """,
            (table_name,),
        )
        columns = {}
        for row in cursor.fetchall():
            col_name, data_type, is_nullable, col_default = row
            columns[col_name] = {
                "type": data_type,
                "nullable": is_nullable == "YES",
                "default": col_default,
            }
        logger.debug("Table '%s' schema: %s", table_name, columns)
        return columns
    except Exception as e:
        logger.error("Failed to introspect schema for '%s': %s", table_name, e)
        return {}


def _sanitize_record_for_schema(
    data: Dict[str, Any], schema: Dict[str, Dict[str, Any]]
) -> None:
    """
    Sanitize record values to match existing database column types.

    Replaces incompatible values (e.g., string 'nan' in integer column)
    with None to prevent insertion errors.
    """
    if not schema:
        return

    numeric_types = {
        "integer",
        "bigint",
        "smallint",
        "real",
        "double precision",
        "numeric",
    }
    float_types = {"real", "double precision", "numeric"}

    for key, value in data.items():
        if key not in schema or value is None:
            continue

        col_info = schema[key]
        db_type = col_info["type"]

        # Handle string values in numeric columns
        if db_type in numeric_types and isinstance(value, str):
            logger.warning(
                "Schema mismatch for column '%s' (DB: %s). "
                "Received string value '%s'; replacing with None.",
                key,
                db_type,
                value,
            )
            data[key] = None

        # Handle NaN floats in integer columns
        elif isinstance(value, float) and db_type not in float_types and value != value:
            logger.warning(
                "Schema mismatch for column '%s' (DB: %s). "
                "Received float NaN; replacing with None.",
                key,
                db_type,
            )
            data[key] = None


def _ensure_table_schema(
    db: PostgresqlDatabase, table_name: str, data: Dict[str, Any]
) -> bool:
    logger.debug(
        "Ensuring schema for table '%s' with keys: %s",
        table_name,
        list(data.keys()),
    )

    # Define standard fields that should always exist
    standard_fields = {
        "id": UUIDField(primary_key=True, default=uuid.uuid4),
        "created_at": DateTimeField(null=True),
        "test_build_id": TextField(null=True),
    }

    # Create table if it doesn't exist
    if not db.table_exists(table_name):
        logger.info("Table '%s' does not exist; creating...", table_name)

        model_attrs = standard_fields.copy()
        for key, value in data.items():
            if key in standard_fields:
                continue
            model_attrs[key] = _infer_field_type(value, nullable=True)

        DynamicModel = _create_model_class(table_name, model_attrs, db, table_name)

        try:
            db.create_tables([DynamicModel], safe=True)
            logger.info("Created table '%s' with UUID primary key", table_name)
            _debug_table_schema(db, table_name)
            return True
        except Exception as e:
            logger.error("Failed to create table '%s': %s", table_name, e)
            return False

    # Table exists: check for missing columns and sanitize data
    existing_columns = _debug_table_schema(db, table_name)
    _sanitize_record_for_schema(data, existing_columns)

    # Identify missing non-standard columns
    missing_columns = {
        k: v
        for k, v in data.items()
        if k not in existing_columns and k not in standard_fields
    }

    if not missing_columns:
        logger.debug("All columns exist in '%s'", table_name)
        return True

    logger.info(
        "Adding %d new columns to '%s': %s",
        len(missing_columns),
        table_name,
        list(missing_columns.keys()),
    )

    migrator = PostgresqlMigrator(db)
    operations = [
        migrator.add_column(table_name, key, _infer_field_type(value, nullable=True))
        for key, value in missing_columns.items()
    ]

    try:
        with db.atomic():
            migrate(*operations)
        logger.info("Migration successful for '%s'", table_name)

        # Verify migration
        updated_schema = _debug_table_schema(db, table_name)
        still_missing = {k for k in missing_columns if k not in updated_schema}
        if still_missing:
            logger.error(
                "Migration incomplete; columns still missing: %s", still_missing
            )
            return False
        return True

    except Exception as e:
        logger.error("Migration failed for '%s': %s", table_name, e)
        _debug_table_schema(db, table_name)
        return False


def _create_model_class(
    name: str, fields: Dict[str, Field], db: PostgresqlDatabase, table_name: str
) -> Type[Model]:
    """Factory function to create dynamic Peewee Model classes."""
    return type(
        name.capitalize(),
        (Model,),
        {
            **fields,
            "__module__": __name__,
            "Meta": type(
                "Meta",
                (),
                {"database": db, "table_name": table_name, "schema": "public"},
            ),
        },
    )


def _create_dynamic_model(
    db: PostgresqlDatabase, table_name: str, record: Dict[str, Any]
) -> Type[Model]:
    """Create a Peewee Model class dynamically based on record structure."""
    fields = {
        "id": UUIDField(primary_key=True),
        "created_at": DateTimeField(null=True),
        "test_build_id": TextField(null=True),
    }

    for key, value in record.items():
        if key in fields:
            continue
        fields[key] = _infer_field_type(value, nullable=True)

    model = _create_model_class(table_name, fields, db, table_name)
    logger.debug(
        "Created dynamic model '%s' with fields: %s",
        table_name.capitalize(),
        list(fields.keys()),
    )
    return model


# ============================================================================
# Main Write Function
# ============================================================================


def write_results(
    table_name: str,
    data: Dict[str, Any],
    build_id: Optional[str] = None,
    debug: bool = False,
) -> bool:

    config = _get_pg_config()
    max_retries = config.get("retry", 5)
    logger.info("Writing to '%s' (max_retries=%s)", table_name, max_retries)

    # Prepare record with standard fields (UTC+8 timezone)
    record = data.copy()
    now_utc8 = datetime.now(UTC_8)
    # Force naive datetime
    tc_tinaive_ume = now_utc8.replace(tzinfo=None)

    record.setdefault("id", uuid.uuid4())
    record.setdefault("created_at", tc_tinaive_ume)
    record.setdefault("test_build_id", build_id or _get_test_build_id())

    logger.debug("Record to insert: %s", record)

    for attempt in range(max_retries + 1):
        db = _create_db_instance()
        try:
            db.connect(reuse_if_open=True)
            logger.debug("Connected to database")

            if not _ensure_table_schema(db, table_name, record):
                raise RuntimeError(f"Schema preparation failed for '{table_name}'")

            DynamicModel = _create_dynamic_model(db, table_name, record)

            logger.debug("Inserting record into '%s'...", table_name)
            with db.atomic():
                result = DynamicModel.insert(**record).execute()
                logger.info(
                    "Insert successful! ID=%s, rows_affected=%s",
                    record["id"],
                    result,
                )
            return True

        except (OperationalError, Exception) as e:
            logger.error(
                "Attempt %d/%d failed: %s: %s",
                attempt + 1,
                max_retries + 1,
                type(e).__name__,
                e,
            )

            if db and not db.is_closed():
                _debug_table_schema(db, table_name)

            if attempt < max_retries:
                wait_time = 2**attempt
                logger.info("Retrying in %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Giving up on '%s'.", table_name)
                return False
        finally:
            if db and not db.is_closed():
                db.close()
                logger.debug("Connection closed")

    return False


# ============================================================================
# Test Entry Point
# ============================================================================

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

        result = write_results(
            "test_db",
            test_data,
            build_id="test_build_001",
            debug=True,
        )

        print("\nFinal result: %s" % ("SUCCESS" if result else "FAILED"))
