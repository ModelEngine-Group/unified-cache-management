"""HiCache YAML configuration without SGLang or native UCM dependencies."""

import ast
import logging
import os
import re
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import yaml


source_path = (
    Path(__file__).resolve().parents[1] / "ucm/integration/sglang/ucm_connector.py"
)
tree = ast.parse(source_path.read_text(encoding="utf-8"))
tree.body = [
    node
    for node in tree.body
    if getattr(node, "name", None)
    in {"_load_extra_config_from_yaml", "UnifiedCacheStoreConfig"}
]
config_module = types.ModuleType("_test_hicache_config")
config_module.__dict__.update(
    re=re,
    dataclass=dataclass,
    Path=Path,
    Any=Any,
    Dict=Dict,
    Optional=Optional,
    yaml=yaml,
    logger=logging.getLogger("test_hicache_config"),
    get_world_group=lambda: types.SimpleNamespace(local_rank=3),
)
sys.modules[config_module.__name__] = config_module
exec(compile(tree, str(source_path), "exec"), config_module.__dict__)


class HiCacheConfigTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.connector = {
            "ucm_connector_name": "UcmPipelineStore",
            "ucm_connector_module_path": "custom.store",
            "ucm_connector_config": {
                "store_pipeline": "Posix",
                "storage_backends": "/cache-a:/cache-b",
                "io_direct": False,
                "posix_io_engine": "psync",
            },
        }

    def write_yaml(self, document, name="ucm_config.yaml"):
        path = self.root / name
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        return str(path)

    def load(self, extra=None, *, logical=False):
        storage_config = types.SimpleNamespace(extra_config=extra, is_mla_model=True)
        host_pool = types.SimpleNamespace(
            kv_buffer=None if logical else object(),
            page_size=4,
            get_size_per_token=lambda: 16,
        )
        return config_module.UnifiedCacheStoreConfig.load_from_config(
            storage_config, host_pool
        )

    def test_yaml_file_preserves_hicache_sizing(self):
        path = self.write_yaml({"ucm_connectors": [self.connector]})
        config = self.load({"kv_connector_extra_config": {"config_file": path}})
        self.assertEqual(config.name, "UcmPipelineStore")
        self.assertEqual(config.module_path, "custom.store")
        self.assertEqual(config.config["storage_backends"], ["/cache-a", "/cache-b"])
        self.assertEqual(config.config["posix_io_engine"], "psync")
        self.assertIs(config.config["io_direct"], False)
        self.assertEqual(config.config["store_pipeline"], "Posix")
        self.assertEqual(config.config["device_id"], 3)
        for key in ("tensor_size", "shard_size", "block_size"):
            self.assertEqual(config.config[key], 64)

    def test_yaml_file_takes_precedence_over_inline_config(self):
        path = self.write_yaml({"ucm_connectors": [self.connector]})
        config = self.load(
            {
                "kv_connector_extra_config": {
                    "config_file": path,
                    "ucm_connectors": [],
                }
            }
        )
        self.assertEqual(config.name, "UcmPipelineStore")

    def test_inline_connector_list(self):
        config = self.load(
            {"kv_connector_extra_config": {"ucm_connectors": [self.connector]}}
        )
        self.assertEqual(config.name, "UcmPipelineStore")
        self.assertEqual(config.config["store_pipeline"], "Posix")
        self.assertEqual(config.config["posix_io_engine"], "psync")
        self.assertIs(config.config["io_direct"], False)

    def test_omitted_pipeline_defaults_to_posix(self):
        del self.connector["ucm_connector_config"]["store_pipeline"]
        with self.assertNoLogs(config_module.logger, level="WARNING"):
            config = self.load({"kv_connector_extra_config": self.connector})
        self.assertEqual(config.config["store_pipeline"], "Posix")
        self.assertNotIn("store_pipeline", self.connector["ucm_connector_config"])

    def test_non_posix_pipeline_is_preserved_with_warning_for_json_and_yaml(self):
        for pipeline in ("Cache|Posix", "Cache", "", None):
            with self.subTest(pipeline=pipeline):
                self.connector["ucm_connector_config"]["store_pipeline"] = pipeline
                document = {"ucm_connectors": [self.connector]}
                path = self.write_yaml(document)
                for extra in (document, {"config_file": path}):
                    with self.assertLogs(config_module.logger, level="WARNING") as logs:
                        config = self.load({"kv_connector_extra_config": extra})
                    self.assertEqual(config.config["store_pipeline"], pipeline)
                    self.assertIn("startup to fail", logs.output[0])

    def test_legacy_inline_config_remains_supported(self):
        config = self.load({"kv_connector_extra_config": self.connector})
        self.assertEqual(config.name, "UcmPipelineStore")

    def test_legacy_yaml_and_logical_pool_remain_supported(self):
        path = self.write_yaml({"kv_connector_extra_config": self.connector})
        config = self.load(
            {"kv_connector_extra_config": {"config_file": path}}, logical=True
        )
        self.assertEqual(config.name, "UcmPipelineStore")
        self.assertEqual(config.config["tensor_size"], 0)
        self.assertEqual(config.config["block_size"], 0)

    def test_rejects_ambiguous_or_invalid_connector_entries(self):
        for connectors in ([], [self.connector, self.connector], {}, ["invalid"]):
            with self.subTest(connectors=connectors):
                path = self.write_yaml({"ucm_connectors": connectors})
                for config in (
                    {"config_file": path},
                    {"ucm_connectors": connectors},
                ):
                    with self.assertRaises(ValueError):
                        self.load({"kv_connector_extra_config": config})

    def test_environment_and_top_level_config_file_are_not_used(self):
        path = self.write_yaml({"ucm_connectors": [self.connector]})
        os.environ["UNIFIEDCACHE_CONFIG_FILE"] = path
        for extra in (None, {"config_file": path}):
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, "kv_connector_extra_config"):
                    self.load(extra)

    def test_missing_file_reports_the_path(self):
        path = str(self.root / "missing.yaml")
        with self.assertRaises(FileNotFoundError) as error:
            self.load({"kv_connector_extra_config": {"config_file": path}})
        self.assertEqual(error.exception.filename, path)


if __name__ == "__main__":
    unittest.main()
