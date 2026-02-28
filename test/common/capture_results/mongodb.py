import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# PyMongo imports
from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    OperationFailure,
    ServerSelectionTimeoutError,
)

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


def _get_mongo_config() -> Dict[str, Any]:
    """Retrieve and cache MongoDB configuration with fallback defaults."""
    global _db_config

    if _db_config is not None:
        return _db_config

    default_config = {
        "host": "127.0.0.1",
        "port": 27017,
        "dbname": "admin",
        "user": None,
        "password": None,
        "retry": 5,
    }

    try:
        from common.config_utils import config_utils as config_instance

        results_config = config_instance.get_config("results", [])

        if not isinstance(results_config, list):
            logger.warning("Config 'results' is not a list; using MongoDB defaults.")
            _db_config = default_config
            return _db_config

        for item in results_config:
            if isinstance(item, dict) and "mongodb" in item:
                mongo_conf = item.get("mongodb", {})
                _db_config = {**default_config, **mongo_conf}
                logger.debug(
                    "Loaded MongoDB config: host=%s, port=%s",
                    _db_config.get("host"),
                    _db_config.get("port"),
                )
                return _db_config

        logger.info("No 'mongodb' configuration found; using defaults.")

    except Exception as e:
        logger.warning("Failed to load MongoDB config, using defaults: %s", e)

    _db_config = default_config
    return _db_config


# ============================================================================
# Main Write Function
# ============================================================================


def write_results(
    table_name: str,  # In MongoDB, this corresponds to the Collection name
    data: Dict[str, Any],
    build_id: Optional[str] = None,
    debug: bool = False,
) -> bool:
    """
    Writes a record to MongoDB.

    Strategy: Create a new connection for each write batch and close it immediately.
    This avoids connection timeout issues in long-running, low-frequency tasks.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    config = _get_mongo_config()
    max_retries = config.get("retry", 5)
    logger.info("Writing to collection '%s' (max_retries=%s)", table_name, max_retries)

    # Prepare record with standard fields (UTC+8 timezone handling)
    record = data.copy()
    now_utc8 = datetime.now(UTC_8)
    # Force naive datetime to match Postgres script behavior
    naive_time = now_utc8.replace(tzinfo=None)

    record.setdefault("id", uuid.uuid4())
    record.setdefault("created_at", naive_time)
    record.setdefault("test_build_id", build_id or _get_test_build_id())

    logger.debug("Record to insert: %s", record)

    for attempt in range(max_retries + 1):
        client: Optional[MongoClient] = None
        try:
            # 1. Establish connection (No connection pooling used here)
            # We set serverSelectionTimeoutMS to avoid hanging on dead connections
            client = MongoClient(
                host=config.get("host"),
                port=config.get("port"),
                username=config.get("user"),
                password=config.get("password"),
                authSource=config.get("authSource"),
                uuidRepresentation="standard",
                serverSelectionTimeoutMS=5000,  # Fail fast if server unreachable
            )

            # Force connection check
            client.admin.command("ping")
            logger.debug("Connected to MongoDB server")

            db = client[config.get("dbname")]
            collection = db[table_name]

            # 2. Insert Data
            logger.debug("Inserting record into '%s'...", table_name)
            result = collection.insert_one(record)

            logger.info(
                "Insert successful! ID=%s",
                result.inserted_id,
            )
            return True

        except (
            ConnectionFailure,
            ServerSelectionTimeoutError,
            OperationFailure,
            Exception,
        ) as e:
            logger.error(
                "Attempt %d/%d failed: %s: %s",
                attempt + 1,
                max_retries + 1,
                type(e).__name__,
                e,
            )

            if attempt < max_retries:
                wait_time = 2**attempt
                logger.info("Retrying in %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Giving up on '%s'.", table_name)
                return False
        finally:
            # 3. Always close connection to release resources
            if client:
                client.close()
                logger.debug("MongoDB connection closed")

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

    # Mock configuration
    mock_config = MagicMock()
    mock_config.get_config.return_value = [
        {
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "dbname": "myapp",
                "user": "root",
                "password": "123456",
                "authSource": "admin",
                "retry": 3,
            }
        }
    ]

    with patch("common.config_utils.config_utils", mock_config):
        test_data = {
            "status": "ok",
            "ttft": 0.5,
            "metrics": {"latency": 12.5, "success": True},
            "tags": ["mongo", "test"],
        }

        result = write_results(
            "test_collection",
            test_data,
            build_id="mongo_build_001",
            debug=True,
        )

        print("\nFinal result: %s" % ("SUCCESS" if result else "FAILED"))
