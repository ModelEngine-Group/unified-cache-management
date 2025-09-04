import importlib
from typing import Callable

from vllm.config import VllmConfig

from unifiedcache.integration.vllm.ucm_sparse.base import UcmSparseBase, UcmSparseRole
from unifiedcache.logger import init_logger

logger = init_logger(__name__)


class UcmSparseFactory:
    _registry: dict[str, Callable[[], type[UcmSparseBase]]] = {}

    @classmethod
    def register_sparse_method(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a sparse attention method with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Sparse attention method '{name}' is already registered.")

        def loader() -> type[UcmSparseBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_sparse_method(
        cls, config: "VllmConfig", role: UcmSparseRole
    ) -> UcmSparseBase:
        sparse_method_name = config.kv_transfer_config.kv_connector_extra_config[
            "ucm_sparse_method"
        ]
        if sparse_method_name in cls._registry:
            sparse_method_cls = cls._registry[sparse_method_name]()
        else:
            raise ValueError(f"Unsupported sparse method type: {sparse_method_name}")
        assert issubclass(sparse_method_cls, UcmSparseBase)
        logger.info("Creating sparse method with name: %s", sparse_method_name)
        return sparse_method_cls(config, role)


# Register available sparse methods
UcmSparseFactory.register_sparse_method("ESA", "unifiedcache.ucm_sparse.esa", "ESA")
UcmSparseFactory.register_sparse_method("GSA", "unifiedcache.ucm_sparse.gsa", "GSA")