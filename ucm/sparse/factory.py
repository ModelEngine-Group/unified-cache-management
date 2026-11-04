import importlib
from typing import Callable

from vllm.config import VllmConfig

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseBase, UcmSparseRole

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
        cls,
        config: "VllmConfig",
        ucm_config: dict,
        role: UcmSparseRole,
    ) -> UcmSparseBase:
        sparse_method_name, _ = next(iter(ucm_config.items()))
        if sparse_method_name in cls._registry:
            sparse_method_cls = cls._registry[sparse_method_name]()
        else:
            raise ValueError(f"Unsupported sparse method type: {sparse_method_name}")
        assert issubclass(sparse_method_cls, UcmSparseBase)
        logger.info("Creating sparse method with name: %s", sparse_method_name)
        return sparse_method_cls(config, role)


# Register available sparse methods
UcmSparseFactory.register_sparse_method("ESA", "ucm.sparse.esa.esa", "ESA")
UcmSparseFactory.register_sparse_method(
    "GSAOnDevice", "ucm.sparse.gsa_on_device.gsa_on_device", "GSAOnDevice"
)
# GSA 同时支持 PD 混合与 PD 分离模式（内部依据 PD 模式自作分支）。
UcmSparseFactory.register_sparse_method("GSA", "ucm.sandbox.sparse.gsav2.gsa", "GSA")
UcmSparseFactory.register_sparse_method(
    "KVStarMultiStep", "ucm.sparse.kvstar.multistep", "KVStarMultiStep"
)
UcmSparseFactory.register_sparse_method("Blend", "ucm.sparse.blend.blend", "Blend")
