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

from ucm.logger import init_logger
from ucm.store.ucmstore_v1 import UcmKVStoreBaseV1

logger = init_logger(__name__)


class UcmConnectorFactoryV1:
    _registry: dict[str, Callable[[], type[UcmKVStoreBaseV1]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[UcmKVStoreBaseV1]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, connector_name: str, config: dict) -> UcmKVStoreBaseV1:
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        assert issubclass(connector_cls, UcmKVStoreBaseV1)
        logger.info("Creating connector with name: %s", connector_name)
        return connector_cls(config)

UcmConnectorFactoryV1.register_connector(
    "UcmPcStore", "ucm.store.pcstore.pcstore_connector", "UcmPcStore"
)