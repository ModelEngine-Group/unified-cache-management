import importlib
from typing import Callable

from unifiedcache.logger import init_logger
from unifiedcache.ucm_connector.base import UcmKVStoreBase

logger = init_logger(__name__)


class UcmConnectorFactory:
    _registry: dict[str, Callable[[], type[UcmKVStoreBase]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector woth a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[UcmKVStoreBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(
            cls,
            connector_name: str,
            config: dict
    ) -> UcmKVStoreBase:
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(
                f"Unsupported connector type: {connector_name}")
        assert issubclass(connector_cls, UcmKVStoreBase)
        logger.info("Creating connector with name: %s", connector_name)
        return connector_cls(config)


UcmConnectorFactory.register_connector(
    "UcmOceanStore",
    "unifiedcache.ucm_connector.ucm_connector",
    "UcmOceanStore")

UcmConnectorFactory.register_connector(
    "UcmMooncakeStore",
    "unifiedcache.ucm_connector.ucm_mooncake",
    "UcmMooncakeStore")

UcmConnectorFactory.register_connector(
    "UcmNfsStore",
    "unifiedcache.ucm_connector.ucm_nfs_store",
    "UcmNfsStore")