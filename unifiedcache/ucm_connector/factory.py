#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

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
    "UcmDram",
    "unifiedcache.ucm_connector.ucm_dram",
    "UcmDram")
UcmConnectorFactory.register_connector(
    "UcmNfsStore", "unifiedcache.ucm_connector.ucm_nfs_store", "UcmNfsStore"
)
UcmConnectorFactory.register_connector(
    "UcmLocalStore", "unifiedcache.ucm_connector.ucm_local_store", "UcmLocalStore"
)