import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 延迟导入
peewee = None
PostgresqlDatabase = None
Model = None
AutoField = None
DateTimeField = None
TextField = None
IntegerField = None
FloatField = None
BooleanField = None
PostgresqlMigrator = None
migrate = None
Introspector = None

logger = logging.getLogger("db_handler")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 全局状态
_test_build_id: Optional[str] = None
_backup_path: Optional[Path] = None

# 定义 UTC+8 时区
TZ_UTC8 = timezone(timedelta(hours=8))


def _ensure_peewee_imported():
    """延迟导入 peewee 组件"""
    global peewee, PostgresqlDatabase, Model, AutoField, DateTimeField, TextField
    global IntegerField, FloatField, BooleanField, PostgresqlMigrator, migrate, Introspector

    if peewee is None:
        import peewee
        from peewee import AutoField as _AF
        from peewee import BooleanField as _BF
        from peewee import DateTimeField as _DTF
        from peewee import FloatField as _FF
        from peewee import IntegerField as _IF
        from peewee import Model as _Model
        from peewee import PostgresqlDatabase as _PGDB
        from peewee import TextField as _TF
        from playhouse.migrate import PostgresqlMigrator as _PGM
        from playhouse.migrate import migrate as _MIG
        from playhouse.reflection import Introspector as _Intro

        peewee = peewee
        PostgresqlDatabase = _PGDB
        Model = _Model
        AutoField = _AF
        DateTimeField = _DTF
        TextField = _TF
        IntegerField = _IF
        FloatField = _FF
        BooleanField = _BF
        PostgresqlMigrator = _PGM
        migrate = _MIG
        Introspector = _Intro


def _get_db_config():
    from common.config_utils import config_utils as config_instance

    return config_instance.get_config("database", {})


def _create_db_instance():
    """
    工厂方法：每次调用都创建一个新的数据库实例。
    不再维护全局单例，由调用者管理生命周期。
    """
    _ensure_peewee_imported()
    db_config = _get_db_config()

    if not db_config.get("enabled", False):
        return None

    try:
        return PostgresqlDatabase(
            db_config.get("name", "test_db"),
            user=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
        )
    except Exception as e:
        logger.error(f"Failed to create DB instance: {e}")
        return None


def _set_test_build_id(build_id: Optional[str] = None) -> None:
    global _test_build_id
    _test_build_id = build_id or "default_build_id"


def _get_test_build_id() -> str:
    global _test_build_id
    if _test_build_id is None:
        _set_test_build_id()
    return _test_build_id


def _init_backup_path():
    """初始化备份路径"""
    global _backup_path
    if _backup_path is None:
        db_config = _get_db_config()
        backup_str = db_config.get("backup", "results/")
        _backup_path = Path(backup_str).resolve()
        _backup_path.mkdir(parents=True, exist_ok=True)


def _backup_to_file(table_name: str, data: Dict[str, Any]) -> None:
    _init_backup_path()
    file_path = _backup_path / f"{table_name}.jsonl"
    try:
        with file_path.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
            f.write("\n")
        logger.info(f"Data backed up to {file_path}")
    except Exception as e:
        logger.error(f"Backup failed: {e}")


def _clean_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _infer_field_type(value: Any):
    _ensure_peewee_imported()
    if value is None:
        return TextField(null=True)
    if isinstance(value, bool):
        return BooleanField(null=True)
    if isinstance(value, int):
        return IntegerField(null=True)
    if isinstance(value, float):
        return FloatField(null=True)
    return TextField(null=True)


def write_to_db(table_name: str, data: Dict[str, Any]) -> bool:
    # 1. 数据预处理
    data["test_build_id"] = _get_test_build_id()
    clean_data = {k: _clean_value(v) for k, v in data.items()}

    db_config = _get_db_config()
    if not db_config.get("enabled", False):
        _backup_to_file(table_name, clean_data)
        return False

    # 2. 创建新连接实例
    db = _create_db_instance()
    if db is None:
        _backup_to_file(table_name, clean_data)
        return False

    try:
        # 3. 使用上下文管理器自动管理连接
        # 进入 with 块时自动 connect，退出时自动 close
        with db:
            _ensure_peewee_imported()

            # 检查并同步 Schema
            if not db.table_exists(table_name):
                _create_table_with_data(db, table_name, clean_data)
            else:
                _sync_table_schema(db, table_name, clean_data)

            # 动态模型构建
            DynamicModel = _get_dynamic_model(db, table_name, clean_data)

            model_fields = set(DynamicModel._meta.fields.keys())
            insert_data = {k: v for k, v in clean_data.items() if k in model_fields}

            # 插入数据
            DynamicModel.insert(insert_data).execute()
            logger.info(f"Inserted data into '{table_name}'")
            return True

    except Exception as e:
        logger.error(f"DB write failed for '{table_name}': {e}", exc_info=True)
        _backup_to_file(table_name, clean_data)
        return False
    # 此处不需要 finally: db.close()，因为 'with db' 已经处理了关闭


def _create_table_with_data(db, table_name: str, data: Dict[str, Any]):
    """创建新表"""
    attrs = {
        "id": AutoField(),
        "created_at": DateTimeField(default=lambda: datetime.now(TZ_UTC8)),
        "test_build_id": TextField(null=True, index=True),
    }

    for key, value in data.items():
        if key in attrs:
            continue
        attrs[key] = _infer_field_type(value)

    Meta = type("Meta", (), {"database": db, "table_name": table_name})
    attrs["Meta"] = Meta

    DynamicModel = type(f"{table_name.capitalize()}Model", (Model,), attrs)
    db.create_tables([DynamicModel], safe=True)
    logger.info(f"Table '{table_name}' created.")


def _sync_table_schema(db, table_name: str, data: Dict[str, Any]):
    """自动迁移：增加新列"""
    columns = db.get_columns(table_name)
    existing_cols = {col.name for col in columns}

    migrator = PostgresqlMigrator(db)

    for key, value in data.items():
        if key not in existing_cols:
            field_type = _infer_field_type(value)
            logger.info(f"Schema migration: Adding column '{key}' to '{table_name}'")
            try:
                migrate(migrator.add_column(table_name, key, field_type))
            except Exception as e:
                logger.error(f"Migration failed for column '{key}': {e}")


def _get_dynamic_model(db, table_name: str, data: Dict[str, Any]):
    """构建动态模型用于插入"""
    columns = db.get_columns(table_name)

    fields = {
        "id": AutoField(),
        "created_at": DateTimeField(default=lambda: datetime.now(TZ_UTC8)),
    }

    for col in columns:
        if col.name in ["id", "created_at"]:
            continue
        # 优先使用传入数据的类型推断，否则默认Text
        if col.name in data:
            fields[col.name] = _infer_field_type(data[col.name])
        else:
            fields[col.name] = TextField(null=True)

    Meta = type("Meta", (), {"database": db, "table_name": table_name})
    attrs = {"Meta": Meta, **fields}
    return type(f"{table_name.capitalize()}DynamicModel", (Model,), attrs)


def read_from_db(
    table_name: str, filters: Optional[Dict[str, Any]] = None, limit: int = 1
) -> List[Dict[str, Any]]:
    db_config = _get_db_config()
    if not db_config.get("enabled", False):
        return []

    db = _create_db_instance()
    if db is None:
        return []

    _ensure_peewee_imported()

    try:
        with db:
            if not db.table_exists(table_name):
                return []

            introspector = Introspector.from_database(db)
            DynamicModel = introspector.generate_models(table_names=[table_name]).get(
                table_name
            )

            if DynamicModel is None:
                return []

            query = DynamicModel.select()

            if filters:
                for key, value in filters.items():
                    if hasattr(DynamicModel, key):
                        query = query.where(getattr(DynamicModel, key) == value)

            query = query.order_by(DynamicModel.created_at.desc()).limit(limit)
            return [row.__data__ for row in query]

    except Exception as e:
        logger.error(f"Read from DB failed: {e}")
        return []


def database_connection(build_id: str) -> None:
    """
    初始化测试构建ID，并测试数据库连通性。
    此时不再保留长连接。
    """
    logger.info(f"Setting test build ID: {build_id}")
    _set_test_build_id(build_id)

    db_config = _get_db_config()
    if not db_config.get("enabled", False):
        logger.info("Database disabled.")
        return

    # 测试连接
    db = _create_db_instance()
    if db is None:
        logger.error("Database config invalid.")
        return

    try:
        with db:
            logger.info(f"Database connection test successful: {db.database}")
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
