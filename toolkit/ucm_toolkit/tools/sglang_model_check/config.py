"""Configuration shared with the isolated checker process."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

CONFIG_ENV = "UCM_SGLANG_MODEL_CHECK_CONFIG"


@dataclass(frozen=True)
class CheckConfig:
    model: str
    mode: str = "inspect"
    device_id: str = "0"
    platform: str = "auto"
    page_size: int = 64
    pages: int = 2
    layout: str = "page_first"
    dtype: str = "auto"
    trust_remote_code: bool = True
    skip_meta_model: bool = False
    storage_backends: str | None = None
    connector_name: str = "UcmPipelineStore"
    connector_module_path: str | None = None
    output: str | None = None
    verbose: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CheckConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_env(cls) -> "CheckConfig":
        raw = os.environ.get(CONFIG_ENV)
        if not raw:
            raise ValueError(f"{CONFIG_ENV} is not set")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{CONFIG_ENV} must contain a JSON object")
        return cls.from_mapping(data)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)
