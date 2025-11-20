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

from typing import Any, Dict

import yaml

from ucm.logger import init_logger

logger = init_logger(__name__)


class Config:
    def __init__(self, kv_transfer_config: Any):
        self.kv_transfer_config = kv_transfer_config
        self.config: Dict[str, Any] = {}
        self._load_config()

    def load_ucm_config_from_yaml(self, file_path: str) -> Dict[str, Any]:
        if not file_path:
            logger.warning("No UCM config file path provided.")
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                if not isinstance(config, dict):
                    logger.warning(
                        f"Config file {file_path} does not contain a dictionary. "
                        "Returning empty config."
                    )
                    return {}
                logger.info(f"Loaded UCM config from {file_path}")
                return config
        except FileNotFoundError:
            logger.error(f"UCM config file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config file {file_path}: {e}")
            return {}

    def _load_config(self) -> None:
        has_extra_config = (
            self.kv_transfer_config is not None
            and hasattr(self.kv_transfer_config, "kv_connector_extra_config")
            and self.kv_transfer_config.kv_connector_extra_config is not None
        )
        if not has_extra_config:
            self.config = self._get_default_config()
        else:
            extra_config = self.kv_transfer_config.kv_connector_extra_config
            if "UCM_CONFIG_FILE" in extra_config:
                config_file = extra_config["UCM_CONFIG_FILE"]
                self.config = self.load_ucm_config_from_yaml(config_file)
            else:
                if extra_config == {}:
                    self.config = self._get_default_config()
                else:
                    self.config = dict(extra_config)
                    logger.info("Using kv_connector_extra_config from terminal input")

    def _get_default_config(self) -> Dict[str, Any]:
        config = {"ucm_connector_name": "UcmDramStore"}
        logger.warning(f"No UCM config provided, using default configuration {config}")
        return config

    def get_config(self) -> Dict[str, Any]:
        logger.info(f"Using UCM with config: {self.config}")
        return self.config
