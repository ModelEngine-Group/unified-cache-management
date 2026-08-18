import importlib.util
import sys
import types
import unittest
from pathlib import Path


class FakeNativePipeline:
    def __init__(self):
        self.stacks = []

    def Stack(self, name, path, config):
        self.stacks.append((name, path, config))


class YuanRongPipelineBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ucm_module = types.ModuleType("ucm")
        ucm_module.__path__ = []
        store_module = types.ModuleType("ucm.store")
        store_module.__path__ = []
        pipeline_module = types.ModuleType("ucm.store.pipeline")
        pipeline_module.__path__ = []
        fake_native = types.ModuleType("ucm.store.pipeline.ucmpipelinestore")
        fake_native.PipelineStore = FakeNativePipeline

        class StoreNotFoundError(RuntimeError):
            pass

        class StoreUnhealthyError(RuntimeError):
            pass

        fake_native.StoreNotFoundError = StoreNotFoundError
        fake_native.StoreUnhealthyError = StoreUnhealthyError
        pipeline_module.ucmpipelinestore = fake_native

        store_v1 = types.ModuleType("ucm.store.ucmstore_v1")

        class Task:
            pass

        class UcmKVStoreBaseV1:
            def __init__(self, config):
                self.config = config

        store_v1.Task = Task
        store_v1.UcmKVStoreBaseV1 = UcmKVStoreBaseV1

        class UnionableMeta(type):
            def __ror__(cls, other):
                return object

        numpy_module = types.ModuleType("numpy")
        numpy_module.ndarray = UnionableMeta("ndarray", (), {})
        numpy_module.uint8 = int
        numpy_module.uint64 = int
        torch_module = types.ModuleType("torch")
        torch_module.Tensor = type("Tensor", (), {})

        sys.modules["ucm"] = ucm_module
        sys.modules["ucm.store"] = store_module
        sys.modules["ucm.store.pipeline"] = pipeline_module
        sys.modules["ucm.store.pipeline.ucmpipelinestore"] = fake_native
        sys.modules["ucm.store.ucmstore_v1"] = store_v1
        sys.modules["numpy"] = numpy_module
        sys.modules["torch"] = torch_module

        connector_path = (
            Path(__file__).resolve().parents[1]
            / "ucm"
            / "store"
            / "pipeline"
            / "connector.py"
        )
        spec = importlib.util.spec_from_file_location(
            "yuanrong_pipeline_connector_under_test", connector_path
        )
        cls.connector = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.connector
        spec.loader.exec_module(cls.connector)

    def test_worker_posix_layout_uses_contiguous_yuanrong_object(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "tensor_size_list": [64, 96],
            "shard_size": 192,
            "block_size": 768,
            "yuanrong_load_worker_count": 8,
            "io_direct": False,
        }

        self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

        self.assertEqual([entry[0] for entry in pipeline.stacks], ["Posix", "YuanRong"])
        posix_config = pipeline.stacks[0][2]
        self.assertEqual(posix_config["tensor_size"], 160)
        self.assertEqual(posix_config["shard_size"], 160)
        self.assertEqual(posix_config["block_size"], 640)
        self.assertIs(pipeline.stacks[1][2], config)
        self.assertEqual(pipeline.stacks[1][2]["yuanrong_load_worker_count"], 8)

    def test_scheduler_uses_worker_published_compact_posix_block_size(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": -1,
            "block_size": 640,
            "io_direct": False,
        }

        self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

        self.assertNotIn("tensor_size", pipeline.stacks[0][2])
        self.assertEqual(pipeline.stacks[0][2]["block_size"], 640)

    def test_direct_io_derives_4096_alignment(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "tensor_size_list": [4096],
            "shard_size": 4096,
            "block_size": 4096,
            "io_direct": True,
        }

        self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

        self.assertEqual([entry[0] for entry in pipeline.stacks], ["Posix", "YuanRong"])
        self.assertNotIn("yuanrong_memory_alignment", pipeline.stacks[1][2])

    def test_direct_io_requires_aligned_payload_size(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "tensor_size_list": [4097],
            "shard_size": 4097,
            "block_size": 4097,
            "io_direct": True,
        }

        with self.assertRaisesRegex(ValueError, "aligned to 4096"):
            self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

    def test_aio_requires_direct_io(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "tensor_size_list": [4096],
            "shard_size": 4096,
            "block_size": 4096,
            "posix_io_engine": "aio",
            "io_direct": False,
        }

        with self.assertRaisesRegex(ValueError, "requires io_direct=true"):
            self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

    def test_aio_is_accepted_with_direct_io_alignment(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "tensor_size_list": [4096],
            "shard_size": 4096,
            "block_size": 4096,
            "posix_io_engine": "aio",
            "io_direct": True,
        }

        self.connector._yuanrong_posix_pipeline_builder(config, pipeline)

        self.assertEqual([entry[0] for entry in pipeline.stacks], ["Posix", "YuanRong"])
        self.assertEqual(pipeline.stacks[0][2]["posix_io_engine"], "aio")

    def test_unknown_io_engine_is_rejected(self):
        pipeline = FakeNativePipeline()
        config = {
            "device_id": 0,
            "posix_io_engine": "unknown",
        }

        with self.assertRaisesRegex(ValueError, "invalid posix_io_engine=unknown"):
            self.connector._yuanrong_posix_pipeline_builder(config, pipeline)


if __name__ == "__main__":
    unittest.main()
