"""EC configuration, state and batched loads; CPU tensors and mocked Store I/O."""

import base64
import copy
import ctypes
import hashlib
import io
import json
import os
import re
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch
import yaml
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from ucm.integration.vllm import ucm_ec_connector as ec
from ucm.store.cache.capabilities import cache_core_version
from ucm.utils import Config


def cpu_chunk(address, layout):
    """View raw CPU memory without retaining the original Tensor owner."""
    buffer = (ctypes.c_ubyte * layout.chunk_bytes).from_address(int(address))
    return torch.frombuffer(buffer, dtype=layout.dtype).reshape(
        layout.rows_per_chunk, layout.width
    )


class ECConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "ucm_connector_name": "CustomRegisteredStore",
            "ucm_connector_config": {"storage_backends": "/first:/second"},
            "encoder_cache_config": {"chunk_size": 3},
        }
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                model="org/model",
                dtype=torch.bfloat16,
                hf_config=None,
                multimodal_config=None,
                revision=None,
                code_revision=None,
                tokenizer_revision=None,
                get_inputs_embeds_size=lambda: 5120,
            ),
            parallel_config=SimpleNamespace(
                data_parallel_rank=0,
                tensor_parallel_size=4,
            ),
            instance_id="instance",
            ec_transfer_config=SimpleNamespace(engine_id="engine"),
        )

    def layout(self, config=None):
        return ec.resolve_encoder_cache_layout(
            self.vllm_config,
            self.config["encoder_cache_config"] if config is None else config,
        )

    def create_store(self, role=ec.ECConnectorRole.SCHEDULER):
        with (
            patch.object(ec.UcmConnectorFactoryV1, "create_connector") as factory,
            patch.object(ec, "_check_shm_capacity") as check_shm,
            patch.object(ec, "_scheduler_read_unique_id", return_value="uid-scheduler"),
            patch.object(ec, "_worker_generate_unique_id", return_value="uid-worker"),
        ):
            result = ec.create_ucm_ec_store(
                vllm_config=self.vllm_config,
                role=role,
                layout=self.layout(),
                ec_config=self.config,
                device_id=-1 if role == ec.ECConnectorRole.SCHEDULER else 0,
            )
        self.check_shm = check_shm
        self.assertIs(result, factory.return_value)
        name, config = factory.call_args.args
        self.assertEqual(name, "CustomRegisteredStore")
        return config

    def test_ec_yaml_loads_without_kv_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ec.yaml"
            path.write_text(yaml.safe_dump({"ucm_ec_connector": self.config}))
            transfer = SimpleNamespace(
                ec_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            self.assertEqual(Config.load_ec_config(transfer), self.config)

    def test_invalid_ec_yaml_fails_at_loading_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ec.yaml"
            transfer = SimpleNamespace(
                ec_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            for content in ("", "[", "[]", "ucm_ec_connector: []"):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        Config.load_ec_config(transfer)

    def test_existing_kv_config_entrypoints_are_preserved(self):
        self.assertEqual(Config(None).get_config(), {})
        inline = {"ucm_connectors": [{"ucm_connector_name": "ExistingStore"}]}
        self.assertEqual(
            Config(SimpleNamespace(kv_connector_extra_config=inline)).get_config(),
            inline,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kv.yaml"
            path.write_text(yaml.safe_dump(inline))
            transfer = SimpleNamespace(
                kv_connector_extra_config={"UCM_CONFIG_FILE": str(path)}
            )
            self.assertEqual(Config(transfer).get_config(), inline)

    def test_chunk_size_preserves_rows_across_widths_and_dtypes(self):
        for width, dtype, size in ((4096, torch.bfloat16, 2), (5120, torch.float32, 4)):
            with self.subTest(width=width, dtype=dtype):
                self.vllm_config.model_config.dtype = dtype
                self.vllm_config.model_config.get_inputs_embeds_size = lambda: width
                layout = self.layout()
                self.assertEqual(layout.rows_per_chunk, 3)
                self.assertEqual(layout.chunk_bytes, 3 * width * size)

    def test_invalid_chunk_rows_are_rejected(self):
        for rows in (0, -1, 1.5, "128", True, None):
            with self.subTest(rows=rows), self.assertRaises(ec.EncoderCacheLayoutError):
                self.layout({"chunk_size": rows})

    def test_explicit_ec_width_does_not_require_generic_input_width(self):
        model = self.vllm_config.model_config
        model.get_inputs_embeds_size = Mock(side_effect=AssertionError("not needed"))
        layout = self.layout({"chunk_size": 128, "encoder_cache_hidden_dim": 4096})
        self.assertEqual(layout.width, 4096)
        self.assertEqual(layout.chunk_bytes, 128 * 4096 * 2)
        model.get_inputs_embeds_size.assert_not_called()

    def test_deepstack_width_uses_structure_independently_of_model_name(self):
        model = self.vllm_config.model_config
        model.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                out_hidden_size=1024,
                deepstack_visual_indexes=[1, 3, 5],
            )
        )
        model.get_inputs_embeds_size = Mock(side_effect=AssertionError("not needed"))
        for name in ("org/renamed-model", "/local/checkpoint"):
            with self.subTest(name=name):
                model.model = name
                self.assertEqual(self.layout().width, 4096)

    def test_empty_deepstack_falls_back_to_inputs_embeds_width(self):
        self.vllm_config.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(
                out_hidden_size=1024,
                deepstack_visual_indexes=[],
            ),
        )
        self.assertEqual(self.layout().width, 5120)

    def test_missing_or_none_deepstack_falls_back_to_inputs_embeds_width(self):
        for vision_config in (
            SimpleNamespace(out_hidden_size=1024),
            SimpleNamespace(out_hidden_size=1024, deepstack_visual_indexes=None),
        ):
            with self.subTest(vision_config=vision_config):
                self.vllm_config.model_config.hf_config = SimpleNamespace(
                    vision_config=vision_config,
                )
                self.assertEqual(self.layout().width, 5120)

    def test_missing_output_width_uses_vllm_input_width(self):
        self.vllm_config.model_config.hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(deepstack_visual_indexes=[1, 3]),
        )
        self.assertEqual(self.layout().width, 5120)

    def test_nonpositive_explicit_width_is_rejected(self):
        for width in (0, -1):
            with (
                self.subTest(width=width),
                self.assertRaises(ec.EncoderCacheLayoutError),
            ):
                self.layout({"chunk_size": 128, "encoder_cache_hidden_dim": width})

    def test_measured_output_width_controls_layout_and_chunk_keys(self):
        original = self.hash_connector()
        self.vllm_config.ec_transfer_config.mm_encoder_output_widths = {"image": 8192}
        with self.assertRaisesRegex(ec.EncoderCacheLayoutError, "disagrees"):
            self.hash_connector()
        measured = self.hash_connector(
            encoder_config={"encoder_cache_hidden_dim": 8192}
        )
        self.assertEqual(measured.layout.width, 8192)
        self.assertEqual(measured.layout.chunk_bytes, 3 * 8192 * 2)
        self.assertNotEqual(self.chunk_keys(original), self.chunk_keys(measured))
        with self.assertRaisesRegex(ec.EncoderCacheLayoutError, "disagrees"):
            self.layout({"chunk_size": 3, "encoder_cache_hidden_dim": 5120})

    def test_disabled_epd_keeps_ordinary_configuration(self):
        implicit = self.hash_connector()
        explicit = self.hash_connector(
            encoder_config={"enable_epd_disaggregate": False}
        )
        self.assertFalse(explicit.enable_epd_disaggregate)
        self.assertEqual(self.chunk_keys(implicit), self.chunk_keys(explicit))

    def test_pruning_is_rejected_without_an_enable_switch(self):
        self.vllm_config.model_config.multimodal_config = SimpleNamespace(
            is_multimodal_pruning_enabled=lambda: True,
        )
        with (
            patch.object(ec.ECConnectorBase, "__init__", return_value=None),
            patch.object(ec, "create_ucm_ec_store") as create_store,
            self.assertRaises(ec.EncoderCacheLayoutError),
        ):
            ec.UCMECConnector(self.vllm_config, ec.ECConnectorRole.SCHEDULER)
        create_store.assert_not_called()

    def test_ec_meta_preserves_model_identity(self):
        model = self.vllm_config.model_config
        model.revision = "weights-revision"
        model.code_revision = "code-revision"
        model.tokenizer_revision = "tokenizer-revision"
        model.hf_config = SimpleNamespace(
            _commit_hash="commit",
            architectures=["CustomArchitecture"],
        )
        model.multimodal_config = SimpleNamespace(compute_hash=lambda: "mm-hash")
        self.assertEqual(
            json.loads(ec.build_ec_hash_meta(self.vllm_config, {})),
            [
                "ucm:ec:v2",
                [
                    "org/model",
                    "weights-revision",
                    "code-revision",
                    "tokenizer-revision",
                    "commit",
                    ["CustomArchitecture"],
                    "mm-hash",
                ],
                [],
            ],
        )

    def test_explicit_namespace_bypasses_model_identity(self):
        self.assertEqual(
            json.loads(
                ec.build_ec_hash_meta(SimpleNamespace(), {"cache_namespace": "shared"})
            ),
            ["ucm:ec:v2", "shared", []],
        )

    def hash_connector(self, config=None, encoder_config=None):
        config = copy.deepcopy(self.vllm_config if config is None else config)
        config.ec_transfer_config.is_ec_producer = True
        config.ec_transfer_config.is_ec_consumer = True
        ec_config = copy.deepcopy(self.config)
        ec_config["ucm_connector_name"] = "UcmPipelineStore"
        ec_config["ucm_connector_config"]["store_pipeline"] = "Cache|Posix"
        if encoder_config is not None:
            ec_config["encoder_cache_config"].update(encoder_config)
        with (
            patch.object(Config, "load_ec_config", return_value=ec_config),
            patch.object(ec, "create_ucm_ec_store", return_value=Mock()),
        ):
            return ec.UCMECConnector(config, ec.ECConnectorRole.SCHEDULER)

    def chunk_keys(self, connector, identifier="image", num_embeds=7):
        return ec.make_chunk_ids(
            identifier=identifier,
            num_embeds=num_embeds,
            layout=connector.layout,
            hasher=connector._block_hasher,
        )

    def test_producer_requires_queued_event_waits_before_opening_store(self):
        for name, pipeline, supported in (
            ("UcmPipelineStore", "Cache|Posix", True),
            ("UcmPipelineStore", "Cache|Compress|Posix", True),
            ("UcmPipelineStore", "Mooncake|Posix", True),
            ("UcmPipelineStore", "YuanRong|Posix", True),
            ("UcmPipelineStore", "Dram", False),
            ("UcmPipelineStore", "Delegator", False),
            ("UcmPipelineStore", "Posix", False),
            ("UcmPipelineStore", "Fake", False),
            ("UcmPipelineStore", "Custom", False),
            ("CustomRegisteredStore", "Cache|Posix", False),
        ):
            for producer in (True, False):
                with self.subTest(store=name, pipeline=pipeline, producer=producer):
                    self.config["ucm_connector_name"] = name
                    self.config["ucm_connector_config"]["store_pipeline"] = pipeline
                    transfer = self.vllm_config.ec_transfer_config
                    transfer.is_ec_producer = producer
                    transfer.is_ec_consumer = True
                    with (
                        patch.object(
                            Config, "load_ec_config", return_value=self.config
                        ),
                        patch.object(ec, "create_ucm_ec_store") as create_store,
                    ):
                        if producer and not supported:
                            with self.assertRaisesRegex(
                                ValueError, "prerequisite event"
                            ):
                                ec.UCMECConnector(
                                    self.vllm_config, ec.ECConnectorRole.SCHEDULER
                                )
                            create_store.assert_not_called()
                        else:
                            ec.UCMECConnector(
                                self.vllm_config, ec.ECConnectorRole.SCHEDULER
                            )
                            create_store.assert_called_once()

    def test_chunk_keys_isolate_layout_even_with_explicit_namespace(self):
        for namespace in ({}, {"cache_namespace": "shared"}):
            base = self.hash_connector(encoder_config=namespace)
            for change in ("width", "dtype", "rows"):
                with self.subTest(namespace=namespace, change=change):
                    config = copy.deepcopy(self.vllm_config)
                    overrides = dict(namespace)
                    if change == "width":
                        overrides["encoder_cache_hidden_dim"] = 4096
                    elif change == "dtype":
                        # Same element size, different interpretation.
                        config.model_config.dtype = torch.float16
                    else:
                        overrides["chunk_size"] = 4
                    other = self.hash_connector(config, overrides)
                    self.assertEqual(
                        base._block_hasher.meta_bytes, other._block_hasher.meta_bytes
                    )
                    self.assertNotEqual(self.chunk_keys(base), self.chunk_keys(other))

    def test_chunk_keys_ignore_kv_deployment_metadata(self):
        base = self.chunk_keys(self.hash_connector())
        for change in ("tp", "rank", "dp", "speculative", "sparse"):
            with self.subTest(change=change):
                config = copy.deepcopy(self.vllm_config)
                if change == "tp":
                    config.parallel_config.tensor_parallel_size = 2
                elif change == "rank":
                    config.parallel_config.rank = 1
                elif change == "dp":
                    config.parallel_config.data_parallel_rank = 1
                elif change == "speculative":
                    config.speculative_config = SimpleNamespace(
                        method="mtp", num_speculative_tokens=3
                    )
                else:
                    config.additional_config = {
                        "enable_sparse_sfa_c8": True,
                        "enable_sparse_li_c8": True,
                    }
                self.assertEqual(base, self.chunk_keys(self.hash_connector(config)))

    def test_chunk_keys_isolate_deepstack_layers_including_equal_width(self):
        for namespace in ({}, {"cache_namespace": "shared"}):
            keys = []
            widths = []
            for indexes in ([], [8, 16], [7, 15], [16, 8]):
                config = copy.deepcopy(self.vllm_config)
                config.model_config.hf_config = SimpleNamespace(
                    vision_config=SimpleNamespace(
                        out_hidden_size=1024, deepstack_visual_indexes=indexes
                    )
                )
                if not indexes:
                    # Bare out_hidden_size no longer widens the layout; pin the
                    # fallback width so all four variants stay width-equal and
                    # only the deepstack salt separates their keys.
                    config.model_config.get_inputs_embeds_size = lambda: 1024
                connector = self.hash_connector(config, namespace)
                keys.append(self.chunk_keys(connector))
                widths.append(connector.layout.width)
            self.assertEqual(widths, [1024, 3072, 3072, 3072])
            self.assertEqual(len(set(keys)), 4)

    def test_chunk_keys_preserve_identity_rows_and_index_isolation(self):
        connector = self.hash_connector()
        keys = self.chunk_keys(connector)
        self.assertEqual(keys, self.chunk_keys(self.hash_connector()))
        self.assertEqual(len(set(keys)), 3)
        self.assertNotEqual(keys, self.chunk_keys(connector, identifier="other"))
        self.assertNotEqual(keys, self.chunk_keys(connector, num_embeds=8))
        for field in ("model", "revision"):
            config = copy.deepcopy(self.vllm_config)
            setattr(config.model_config, field, "other")
            self.assertNotEqual(keys, self.chunk_keys(self.hash_connector(config)))
        self.assertNotEqual(
            self.chunk_keys(
                self.hash_connector(encoder_config={"cache_namespace": "a"})
            ),
            self.chunk_keys(
                self.hash_connector(encoder_config={"cache_namespace": "b"})
            ),
        )

    def test_scheduler_keys_match_saved_chunks_and_load_round_trip(self):
        scheduler = self.hash_connector()
        scheduler.ensure_cache_available(FakeRequest("request", ("image", 7)), 0)
        state = scheduler.identifier_to_blocks["image"]
        worker = self.hash_connector()
        worker.is_save_rank = True
        worker.device = torch.device("cpu")
        worker._save_device = Mock()
        worker._save_device.get_event_handle.return_value = 11
        stored = {}

        def dump(block_ids, shard_indices, addresses, **kwargs):
            for block_id, (address,) in zip(block_ids, addresses):
                stored[block_id] = cpu_chunk(address, worker.layout).clone()
            return Mock()

        worker.store.dump_data.side_effect = dump
        tensor = torch.ones((7, worker.layout.width), dtype=worker.layout.dtype)
        worker.save_caches({"image": tensor}, "image")
        worker.get_finished(set())
        ids, _, _ = worker.store.dump_data.call_args.args
        self.assertEqual(ids, list(state.chunk_ids))

        def load(block_ids, shard_indices, destinations, prerequisite_handle=0):
            self.assertEqual(prerequisite_handle, 11)
            self.assertEqual(block_ids, ids)
            for block_id, (address,) in zip(block_ids, destinations):
                cpu_chunk(address, worker.layout).copy_(stored[block_id])
            return Mock()

        worker.store.load_data.side_effect = load
        worker._connector_metadata = ec.UCMECConnectorMetadata(
            loads={"image": ec.ECLoadSpec(7, state.chunk_ids)}
        )
        cache = {}
        worker.start_load_caches(cache)
        torch.testing.assert_close(cache["image"], tensor)

    def test_store_overrides_do_not_mutate_yaml_and_split_backends(self):
        self.config["ucm_connector_config"].update(
            share_buffer_enable=False,
            local_rank_size=99,
        )
        original = copy.deepcopy(self.config)
        config = self.create_store(ec.ECConnectorRole.WORKER)
        self.assertEqual(self.config, original)
        self.assertEqual(config["storage_backends"], ["/first", "/second"])
        self.assertTrue(config["share_buffer_enable"])
        self.assertEqual(config["local_rank_size"], 4)
        self.assertEqual(config["tensor_size_list"], [3 * 5120 * 2])
        self.assertEqual(config["shard_size"], 32768)
        self.assertEqual(config["block_size"], 32768)
        self.assertNotIn("gpu_kv_buffer_addrs", config)
        self.assertNotIn("gpu_kv_buffer_sizes", config)

    def test_shared_buffer_capacity_defaults_and_is_checked(self):
        config = self.create_store()
        expected_default = 160 if cache_core_version() else 128
        self.assertEqual(config["cache_buffer_capacity_gb"], expected_default)
        self.check_shm.assert_called_once_with(expected_default)

        self.config["ucm_connector_config"]["cache_buffer_capacity_gb"] = 32
        config = self.create_store()
        self.assertEqual(config["cache_buffer_capacity_gb"], 32)
        self.check_shm.assert_called_once_with(32)

    def test_gdr_is_disabled_and_warns_only_when_configured(self):
        with patch.object(ec.logger, "warning") as warning:
            self.assertFalse(self.create_store()["use_gdr"])
            warning.assert_not_called()
        for value in (True, False):
            self.config["ucm_connector_config"]["use_gdr"] = value
            with (
                self.subTest(value=value),
                patch.object(ec.logger, "warning") as warning,
            ):
                self.assertFalse(self.create_store()["use_gdr"])
                warning.assert_called_once()

    def test_gc_owner_overrides_user_flag_and_passes_capacity_to_store(self):
        for role in ec.ECConnectorRole:
            for dp_rank in (0, 1):
                for capacity in (0, 10240):
                    with self.subTest(role=role, dp_rank=dp_rank, capacity=capacity):
                        self.vllm_config.parallel_config.data_parallel_rank = dp_rank
                        owner = role == ec.ECConnectorRole.SCHEDULER and dp_rank == 0
                        self.config["ucm_connector_config"].update(
                            posix_gc_enable=not owner,
                            posix_capacity_gb=capacity,
                        )
                        config = self.create_store(role)
                        self.assertEqual(config["posix_gc_enable"], owner)
                        self.assertEqual(config["posix_capacity_gb"], capacity)
                        if owner:
                            self.assertEqual(config["block_size"], 32768)
                        elif role == ec.ECConnectorRole.SCHEDULER:
                            self.assertNotIn("block_size", config)
                        else:
                            self.assertEqual(config["block_size"], 32768)

    def test_yuanrong_posix_gc_uses_persisted_object_size(self):
        self.config["ucm_connector_config"]["store_pipeline"] = "YuanRong|Posix"
        config = self.create_store(ec.ECConnectorRole.SCHEDULER)
        self.assertEqual(config["block_size"], 3 * 5120 * 2)

    def test_store_namespace_handshake_default_and_override(self):
        def build_store(role, explicit_id=None):
            ec_config = copy.deepcopy(self.config)
            if explicit_id is not None:
                ec_config["encoder_cache_config"]["store_unique_id"] = explicit_id
            with (
                patch.object(ec.UcmConnectorFactoryV1, "create_connector") as factory,
                patch.object(ec, "_check_shm_capacity"),
                patch.object(
                    ec, "_scheduler_read_unique_id", return_value="uid-scheduler"
                ) as scheduler_uid,
                patch.object(
                    ec, "_worker_generate_unique_id", return_value="uid-worker"
                ) as worker_uid,
            ):
                config = ec.create_ucm_ec_store(
                    vllm_config=self.vllm_config,
                    role=role,
                    layout=self.layout(),
                    ec_config=ec_config,
                    device_id=-1 if role == ec.ECConnectorRole.SCHEDULER else 0,
                )
                self.assertIs(config, factory.return_value)
                _, config = factory.call_args.args
            return config, scheduler_uid, worker_uid

        for role in ec.ECConnectorRole:
            with self.subTest(role=role):
                config, scheduler_uid, worker_uid = build_store(role)
                if role == ec.ECConnectorRole.SCHEDULER:
                    self.assertEqual(config["unique_id"], "uid-scheduler.ec.dp0")
                    scheduler_uid.assert_called_once_with(prefix="ucm_ec")
                    worker_uid.assert_not_called()
                else:
                    self.assertEqual(config["unique_id"], "uid-worker.ec.dp0")
                    worker_uid.assert_called_once_with(prefix="ucm_ec")
                    scheduler_uid.assert_not_called()

        for role in ec.ECConnectorRole:
            with self.subTest(role=role, unique_id="explicit"):
                config, scheduler_uid, worker_uid = build_store(role, "explicit")
                self.assertEqual(config["unique_id"], "explicit")
                worker_uid.assert_not_called()
                scheduler_uid.assert_not_called()


class FakeRequest:
    def __init__(self, request_id, *features):
        self.request_id = request_id
        self.mm_features = []
        self.append(*features)

    def append(self, *features):
        offset = sum(feature.mm_position.length for feature in self.mm_features)
        for identifier, rows in features:
            self.mm_features.append(
                SimpleNamespace(
                    identifier=identifier,
                    rows=rows,
                    mm_position=PlaceholderRange(offset=offset, length=rows),
                )
            )
            offset += rows

    def get_num_encoder_embeds(self, index):
        return self.mm_features[index].rows


class ECConnectorLoadTest(unittest.TestCase):
    def setUp(self):
        self.connector = ec.UCMECConnector.__new__(ec.UCMECConnector)
        self.connector.enable_epd_disaggregate = False
        self.connector._is_producer = False
        self.connector.device = torch.device("cpu")
        self.connector.enable_event_sync = True
        self.connector._save_device = Mock()
        self.recorded_events = []
        self.load_events = {}
        self.connector._save_device.get_event_handle.side_effect = self.record_event
        self.expected_event = 0
        self.connector.layout = ec.EncoderCacheLayout(
            width=2, dtype=torch.float32, rows_per_chunk=3, chunk_bytes=24
        )
        self.connector._connector_metadata = ec.UCMECConnectorMetadata(
            loads={
                "A": ec.ECLoadSpec(num_embeds=4, chunk_ids=(b"A0", b"A1")),
                "B": ec.ECLoadSpec(num_embeds=3, chunk_ids=(b"B0",)),
                "C": ec.ECLoadSpec(num_embeds=2, chunk_ids=(b"C0",)),
            }
        )
        self.cache = {}
        self.events = []
        self.destinations = {}
        self.submit_failure = None
        self.wait_failures = set()
        self.failure = RuntimeError("injected Store failure")
        self.connector.store = Mock()
        self.connector.store.load_data.side_effect = self.load
        self.connector.store.wait.side_effect = self.wait

    def record_event(self):
        handle = 71 + len(self.recorded_events)
        self.recorded_events.append(handle)
        self.expected_event = handle
        return handle

    def load(self, chunk_ids, shard_indices, destinations, prerequisite_handle=0):
        self.assertEqual(prerequisite_handle, self.expected_event)
        identifier = chr(chunk_ids[0][0])
        self.load_events[identifier] = prerequisite_handle
        self.events.append(("load", identifier))
        self.assertEqual(shard_indices, [0] * len(chunk_ids))
        self.assertEqual(destinations.dtype, np.dtype(np.uint64))
        self.assertEqual(destinations.shape, (len(chunk_ids), 1))
        self.assertTrue(destinations.flags.c_contiguous)
        np.testing.assert_array_equal(
            np.diff(destinations[:, 0]),
            np.full(len(chunk_ids) - 1, self.connector.layout.chunk_bytes),
        )
        if identifier == self.submit_failure:
            raise self.failure
        self.destinations[identifier] = destinations
        return identifier

    def wait(self, identifier):
        self.events.append(("wait", identifier))
        # A destination must not be exposed before its transfer completes.
        self.assertNotIn(identifier, self.cache)
        handle = self.load_events[identifier]
        if handle:
            self.assertNotIn(
                unittest.mock.call(handle),
                self.connector._save_device.destroy_event_handle.call_args_list,
            )
        if identifier in self.wait_failures:
            raise self.failure
        for index, (address,) in enumerate(self.destinations[identifier]):
            cpu_chunk(address, self.connector.layout).fill_(ord(identifier) + index)

    def test_submits_batch_before_waiting_and_publishes_unpadded_tensors(self):
        self.connector.enable_epd_disaggregate = True
        resident = torch.ones(1, 2)
        self.cache["B"] = resident
        self.connector.start_load_caches(self.cache)

        self.assertEqual(
            self.events, [("load", "A"), ("load", "C"), ("wait", "A"), ("wait", "C")]
        )
        self.assertIs(self.cache["B"], resident)
        torch.testing.assert_close(
            self.cache["A"], torch.tensor([[65.0, 65.0]] * 3 + [[66.0, 66.0]])
        )
        torch.testing.assert_close(self.cache["C"], torch.full((2, 2), 67.0))

    def test_submission_errors_drain_earlier_tasks_and_preserve_original_error(self):
        for failure_kind in ("submit", "metadata", "allocation", "submit_and_wait"):
            with self.subTest(failure_kind=failure_kind):
                self.setUp()
                expected_type = ec.UCMEncoderCacheError
                if failure_kind == "metadata":
                    self.connector._connector_metadata.loads["B"] = ec.ECLoadSpec(
                        num_embeds=4, chunk_ids=(b"B0",)
                    )
                    expected_type = ec.EncoderCacheLayoutError
                elif failure_kind == "allocation":
                    expected_type = MemoryError
                else:
                    self.submit_failure = "B"
                if failure_kind == "submit_and_wait":
                    self.wait_failures.add("A")

                allocate = torch.empty

                allocations = 0

                def allocate_or_fail(shape, **kwargs):
                    nonlocal allocations
                    allocations += 1
                    if failure_kind == "allocation" and allocations == 2:
                        raise MemoryError("injected allocation failure")
                    return allocate(shape, **kwargs)

                with patch.object(ec.torch, "empty", side_effect=allocate_or_fail):
                    with self.assertRaises(expected_type) as caught:
                        self.connector.start_load_caches(self.cache)

                waits = [
                    identifier for event, identifier in self.events if event == "wait"
                ]
                self.assertEqual(waits, ["A"])
                self.assertNotIn("B", self.cache)
                self.assertNotIn("C", self.cache)
                self.assertEqual("A" in self.cache, not self.wait_failures)
                expected_destroyed = (
                    [72, 71] if failure_kind in {"submit", "submit_and_wait"} else [71]
                )
                self.assertEqual(
                    self.connector._save_device.destroy_event_handle.call_args_list,
                    [unittest.mock.call(handle) for handle in expected_destroyed],
                )
                if expected_type is ec.UCMEncoderCacheError:
                    self.assertEqual(caught.exception.identifier, "B")
                    self.assertIs(caught.exception.__cause__, self.failure)

    def test_wait_errors_drain_all_tasks_and_only_publish_successful_items(self):
        self.wait_failures = {"A", "B"}
        with self.assertRaises(ec.UCMEncoderCacheError) as caught:
            self.connector.start_load_caches(self.cache)
        self.assertEqual(
            self.events,
            [("load", key) for key in "ABC"] + [("wait", key) for key in "ABC"],
        )
        self.assertEqual(caught.exception.identifier, "A")
        self.assertIs(caught.exception.__cause__, self.failure)
        self.assertEqual(set(self.cache), {"C"})
        torch.testing.assert_close(self.cache["C"], torch.full((2, 2), 67.0))
        self.assertEqual(
            self.connector._save_device.destroy_event_handle.call_args_list,
            [unittest.mock.call(handle) for handle in (71, 72, 73)],
        )

    def test_each_allocation_records_and_submits_its_own_event(self):
        timeline = []
        allocate = torch.empty

        def allocate_and_record(*args, **kwargs):
            timeline.append("allocate")
            return allocate(*args, **kwargs)

        def record():
            timeline.append("record")
            return self.record_event()

        def submit(*args, **kwargs):
            timeline.append("submit")
            return self.load(*args, **kwargs)

        def wait(task):
            timeline.append("wait")
            return self.wait(task)

        self.connector._save_device.get_event_handle.side_effect = record
        self.connector._save_device.destroy_event_handle.side_effect = (
            lambda handle: timeline.append("destroy")
        )
        self.connector.store.load_data.side_effect = submit
        self.connector.store.wait.side_effect = wait
        with patch.object(ec.torch, "empty", side_effect=allocate_and_record):
            self.connector.start_load_caches(self.cache)
        self.assertEqual(
            timeline,
            ["allocate", "record", "submit"] * 3 + ["wait", "destroy"] * 3,
        )
        self.connector._save_device.synchronize.assert_not_called()

    def test_event_record_error_drains_prior_submissions(self):
        def record_or_fail():
            if self.recorded_events:
                raise self.failure
            return self.record_event()

        self.connector._save_device.get_event_handle.side_effect = record_or_fail
        with self.assertRaises(RuntimeError) as caught:
            self.connector.start_load_caches(self.cache)
        self.assertIs(caught.exception, self.failure)
        self.assertEqual(self.events, [("load", "A"), ("wait", "A")])
        self.connector._save_device.destroy_event_handle.assert_called_once_with(71)

    def test_event_failure_or_disabled_mode_synchronizes_before_submission(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                self.setUp()
                self.expected_event = 0
                self.connector.enable_event_sync = not disabled
                self.connector._save_device.get_event_handle.side_effect = None
                self.connector._save_device.get_event_handle.return_value = 0
                submit = self.load

                def load_after_sync(*args, **kwargs):
                    self.assertEqual(
                        self.connector._save_device.synchronize.call_count,
                        len(self.destinations) + 1,
                    )
                    return submit(*args, **kwargs)

                self.connector.store.load_data.side_effect = load_after_sync
                self.connector.start_load_caches(self.cache)
                self.connector._save_device.destroy_event_handle.assert_not_called()
                if disabled:
                    self.connector._save_device.get_event_handle.assert_not_called()

    def test_resident_items_do_not_create_or_synchronize_event(self):
        self.cache.update({key: torch.ones(1, 2) for key in "ABC"})
        self.connector.start_load_caches(self.cache)
        self.connector.store.load_data.assert_not_called()
        self.connector._save_device.get_event_handle.assert_not_called()
        self.connector._save_device.synchronize.assert_not_called()
        self.connector._save_device.destroy_event_handle.assert_not_called()

    def test_load_debug_logs_item_and_step_summaries(self):
        self.cache["B"] = torch.ones(1, 2)
        with self.assertLogs(ec.logger, level="DEBUG") as captured:
            self.connector.start_load_caches(self.cache)

        messages = [record.getMessage() for record in captured.records]
        self.assertTrue(
            any(
                "EC load item A: shape=[4, 2], chunks=2, bytes=48" in message
                for message in messages
            )
        )
        self.assertTrue(
            any(
                "EC load item C: shape=[2, 2], chunks=1, bytes=24" in message
                for message in messages
            )
        )
        self.assertTrue(
            any(
                "EC loads: submitted=2, loaded=2, failed=0, skipped=1, "
                "chunks=3, bytes=72" in message
                for message in messages
            )
        )

    def test_load_wait_failure_is_counted_in_step_summary(self):
        self.cache["B"] = torch.ones(1, 2)
        self.wait_failures.add("A")
        with (
            self.assertLogs(ec.logger, level="DEBUG") as captured,
            self.assertRaises(ec.UCMEncoderCacheError),
        ):
            self.connector.start_load_caches(self.cache)

        messages = [record.getMessage() for record in captured.records]
        self.assertTrue(
            any(
                "EC loads: submitted=2, loaded=1, failed=1, skipped=1, "
                "chunks=1, bytes=24" in message
                for message in messages
            )
        )


class ECConnectorSaveTest(unittest.TestCase):
    def setUp(self):
        self.connector = ec.UCMECConnector.__new__(ec.UCMECConnector)
        self.connector.enable_epd_disaggregate = False
        self.connector.layout = ec.EncoderCacheLayout(
            width=2, dtype=torch.float32, rows_per_chunk=3, chunk_bytes=24
        )
        self.connector._is_producer = True
        self.connector.is_save_rank = True
        self.connector._pending_saves = []
        self.connector._save_device = Mock()
        self.connector._step_saved_ids = set()
        self.connector._local_encoder_ids = set()
        self.connector._save_device.get_event_handle.side_effect = [11, 12, 13]
        self.connector._block_hasher = ec.RequestHasher(meta="test")
        self.connector.pending_loads = {}
        self.connector.step_verified_hits = set()
        self.connector.req_to_state = {}
        self.connector.identifier_to_blocks = {}
        self.connector._connector_metadata = ec.UCMECConnectorMetadata()
        self.connector.store = Mock(spec=ec.UcmKVStoreBaseV1)
        self.calls = []
        self.transfers = []
        self.completed = []
        self.wait_failures = set()
        self.connector.store.dump_data.side_effect = self.dump
        self.connector.store.wait.side_effect = self.wait

    def dump(self, ids, shards, addresses, prerequisite_handle=0):
        self.assertEqual(addresses.dtype, np.dtype(np.uint64))
        self.assertEqual(addresses.shape, (len(ids), 1))
        self.assertTrue(addresses.flags.c_contiguous)
        self.assertEqual(shards, [0] * len(ids))
        task = len(self.transfers)
        self.calls.append(("dump", task))
        self.transfers.append((addresses.copy(), prerequisite_handle))
        return task

    def wait(self, task):
        self.calls.append(("wait", task))
        addresses, event = self.transfers[task]
        # Event and raw-pointer owners must survive through completion.
        self.assertNotIn(
            ((event,), {}),
            self.connector._save_device.destroy_event_handle.call_args_list,
        )
        if task in self.wait_failures:
            raise RuntimeError("injected wait failure")
        self.completed.append(
            torch.cat(
                [
                    cpu_chunk(address, self.connector.layout).clone()
                    for (address,) in addresses
                ]
            )
        )

    def test_epd_cached_and_new_outputs_share_step_dedup(self):
        self.connector.enable_epd_disaggregate = True
        metadata = ec.UCMECConnectorMetadata(save_identifiers=("A", "B", "A"))
        self.connector.bind_connector_metadata(metadata)
        cache = {"A": torch.ones(4, 2), "unrelated": torch.ones(3, 2)}
        self.connector.start_save_caches(cache)
        self.assertEqual(self.calls, [("dump", 0)])
        self.assertIs(self.connector._pending_saves[0].tensor, cache["A"])

        cache["B"] = torch.full((3, 2), 2.0)
        self.connector.save_caches(cache, "B")
        self.connector.save_caches(cache, "A")
        self.assertEqual(self.calls, [("dump", 0), ("dump", 1)])
        self.assertEqual(self.connector._step_saved_ids, {"A", "B"})
        self.connector.get_finished(set())
        self.assertFalse(self.connector._step_saved_ids)
        torch.testing.assert_close(self.completed[0][:4], cache["A"])
        torch.testing.assert_close(self.completed[1], cache["B"])

        # A later request must submit the same cached tensor again.
        self.connector.bind_connector_metadata(
            ec.UCMECConnectorMetadata(save_identifiers=("A",))
        )
        self.connector.start_save_caches(cache)
        self.assertEqual(self.calls[-1], ("dump", 2))
        self.connector.get_finished(set())

    def test_epd_cached_save_submission_failure_can_retry(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.bind_connector_metadata(
            ec.UCMECConnectorMetadata(save_identifiers=("A",))
        )
        cache = {"A": torch.ones(4, 2)}
        self.connector.store.dump_data.side_effect = RuntimeError("submit failed")
        with patch.object(ec.logger, "exception") as logged:
            self.connector.start_save_caches(cache)
        logged.assert_called_once()
        self.assertNotIn("A", self.connector._step_saved_ids)
        self.connector._save_device.destroy_event_handle.assert_called_once_with(11)
        self.connector.store.dump_data.side_effect = self.dump
        self.connector.save_caches(cache, "A")
        self.assertEqual(self.connector._step_saved_ids, {"A"})
        self.connector.get_finished(set())

    def test_cached_save_hook_respects_mode_role_rank_and_metadata(self):
        for epd, producer, writer in (
            (False, True, True),
            (True, False, True),
            (True, True, False),
        ):
            with self.subTest(epd=epd, producer=producer, writer=writer):
                self.setUp()
                self.connector.enable_epd_disaggregate = epd
                self.connector._is_producer = producer
                self.connector.is_save_rank = writer
                self.connector.bind_connector_metadata(
                    ec.UCMECConnectorMetadata(save_identifiers=("A",))
                )
                self.connector.start_save_caches({"A": torch.ones(3, 2)})
                self.assertFalse(self.calls)
        self.setUp()
        self.connector.enable_epd_disaggregate = True
        self.connector.bind_connector_metadata(object())
        with self.assertRaisesRegex(TypeError, "UCMECConnectorMetadata"):
            self.connector.start_save_caches({})

    def test_submits_all_items_before_waiting_and_preserves_padding(self):
        inputs = [
            torch.arange(n * 2, dtype=torch.float32).reshape(n, 2) for n in (7, 6, 2)
        ]
        for index, tensor in enumerate(inputs):
            self.connector.save_caches({str(index): tensor}, str(index))
        self.assertEqual(self.calls, [("dump", i) for i in range(3)])
        self.connector._save_device.destroy_event_handle.assert_not_called()
        # Full chunks point directly into each input; only a tail is allocated.
        for tensor, (addresses, _) in zip(inputs, self.transfers):
            for index in range(tensor.shape[0] // 3):
                self.assertEqual(
                    int(addresses[index, 0]), tensor.data_ptr() + index * 24
                )
        self.assertEqual(self.connector.get_finished(set()), (None, None))
        self.assertEqual(
            self.calls,
            [("dump", i) for i in range(3)] + [("wait", i) for i in range(3)],
        )
        for tensor, saved in zip(inputs, self.completed):
            torch.testing.assert_close(saved[: len(tensor)], tensor)
            self.assertEqual(torch.count_nonzero(saved[len(tensor) :]).item(), 0)
        self.assertFalse(self.connector._pending_saves)
        self.assertEqual(self.connector._save_device.destroy_event_handle.call_count, 3)
        self.connector.get_finished(set())
        self.assertEqual(self.connector.store.wait.call_count, 3)

    def test_owners_survive_cache_eviction_and_are_released_after_wait(self):
        tensor = torch.ones(4, 2)
        reference = weakref.ref(tensor)
        cache = {"image": tensor}
        self.connector.save_caches(cache, "image")
        tail_reference = weakref.ref(self.connector._pending_saves[0].tail)
        cache.clear()
        del tensor
        self.assertIsNotNone(reference())
        self.assertIsNotNone(tail_reference())
        self.connector.get_finished(set())
        self.assertIsNone(reference())
        self.assertIsNone(tail_reference())
        torch.testing.assert_close(self.completed[0][:4], torch.ones(4, 2))

    def test_wait_failure_does_not_skip_later_tasks_or_event_cleanup(self):
        for identifier in ("A", "B"):
            self.connector.save_caches({identifier: torch.ones(3, 2)}, identifier)
        self.wait_failures.add(0)
        self.connector.get_finished(set())
        self.assertEqual(
            self.calls, [("dump", 0), ("dump", 1), ("wait", 0), ("wait", 1)]
        )
        self.assertEqual(self.connector._save_device.destroy_event_handle.call_count, 2)
        self.assertFalse(self.connector._pending_saves)

    def test_submission_failure_releases_its_event_and_keeps_earlier_task(self):
        self.connector.save_caches({"A": torch.ones(3, 2)}, "A")
        self.connector.store.dump_data.side_effect = RuntimeError("submission failed")
        self.connector.save_caches({"B": torch.ones(4, 2)}, "B")
        self.connector._save_device.destroy_event_handle.assert_called_once_with(12)
        self.connector.get_finished(set())
        self.assertEqual(self.calls, [("dump", 0), ("wait", 0)])

    def test_clear_metadata_preserves_pending_saves_and_their_owners(self):
        tensor = torch.ones(4, 2)
        cache = {"A": tensor}
        self.connector.save_caches(cache, "A")
        tensor_reference = weakref.ref(tensor)
        tail_reference = weakref.ref(self.connector._pending_saves[0].tail)
        cache.clear()
        del tensor

        self.connector.clear_connector_metadata()

        self.assertIsNone(self.connector._connector_metadata)
        self.assertEqual(self.calls, [("dump", 0)])
        self.assertEqual(len(self.connector._pending_saves), 1)
        self.assertIsNotNone(tensor_reference())
        self.assertIsNotNone(tail_reference())
        self.connector._save_device.destroy_event_handle.assert_not_called()
        self.connector.get_finished(set())
        self.assertEqual(self.calls, [("dump", 0), ("wait", 0)])
        self.assertIsNone(tensor_reference())
        self.assertIsNone(tail_reference())

    def test_shutdown_drains_before_clearing_state_and_closing_store(self):
        for wait_fails in (False, True):
            with self.subTest(wait_fails=wait_fails):
                self.setUp()
                self.connector.save_caches({"A": torch.ones(4, 2)}, "A")
                store = self.connector.store
                self.connector.pending_loads["A"] = Mock()
                if wait_fails:
                    self.wait_failures.add(0)

                def wait(task):
                    self.assertIn("A", self.connector.pending_loads)
                    self.assertIsNotNone(self.connector._connector_metadata)
                    self.wait(task)

                store.wait.side_effect = wait
                # close is optional on the Store interface.
                store.close = Mock(side_effect=lambda: self.calls.append(("close", 0)))
                self.connector.shutdown()
                self.assertEqual(self.calls, [("dump", 0), ("wait", 0), ("close", 0)])
                self.assertIsNone(self.connector._connector_metadata)
                self.assertFalse(self.connector.pending_loads)
                self.assertFalse(self.connector._pending_saves)
                self.assertIsNone(self.connector.store)
                self.connector._save_device.destroy_event_handle.assert_called_once_with(
                    11
                )
                self.connector.shutdown()
                store.wait.assert_called_once_with(0)
                store.close.assert_called_once()

    def test_event_is_recorded_after_padding_before_submission(self):
        def record():
            self.assertFalse(self.transfers)
            # Captured tail has already been filled on the compute stream.
            torch.testing.assert_close(tails[0][:1], torch.ones(1, 2))
            self.assertEqual(torch.count_nonzero(tails[0][1:]).item(), 0)
            return 11

        tails = []
        zeros = torch.zeros

        def allocate(*args, **kwargs):
            tail = zeros(*args, **kwargs)
            tails.append(tail)
            return tail

        self.connector._save_device.get_event_handle.side_effect = record
        with patch.object(ec.torch, "zeros", side_effect=allocate):
            self.connector.save_caches({"A": torch.ones(4, 2)}, "A")
        self.assertEqual(self.transfers[0][1], 11)
        self.connector._save_device.synchronize.assert_not_called()
        self.connector.get_finished(set())

    def test_save_records_event_without_waiting_with_or_without_tail(self):
        for num_rows in (3, 4):
            with self.subTest(num_rows=num_rows):
                self.setUp()
                device = self.connector._save_device

                def record():
                    self.calls.append(("record", 11))
                    return 11

                device.get_event_handle.side_effect = record
                tensor = torch.ones(num_rows, 2)
                # Device owns platform operations; EC must not create or wait
                # on its own CUDA/NPU events or streams.
                with (
                    patch.object(ec.torch, "cuda") as cuda,
                    patch.object(ec.torch, "npu", create=True) as npu,
                ):
                    self.connector.save_caches({"A": tensor}, "A")
                self.assertEqual(self.calls, [("record", 11), ("dump", 0)])
                self.assertEqual(device.mock_calls, [call.get_event_handle()])
                self.assertEqual(cuda.mock_calls, [])
                self.assertEqual(npu.mock_calls, [])
                self.assertEqual(self.transfers[0][1], 11)
                self.connector.store.wait.assert_not_called()
                self.connector.get_finished(set())
                device.destroy_event_handle.assert_called_once_with(11)

    def test_dump_debug_logs_shape_and_padded_size(self):
        with self.assertLogs(ec.logger, level="DEBUG") as captured:
            self.connector.save_caches({"A": torch.ones(7, 2)}, "A")
        message = captured.records[0].getMessage()
        self.assertIn(
            "EC dump item A: shape=[7, 2], chunks=3, tail_rows=1, bytes=72",
            message,
        )
        self.assertIn("submit_ms=", message)

        with self.assertLogs(ec.logger, level="DEBUG") as captured:
            self.connector.save_caches({"B": torch.ones(6, 2)}, "B")
        self.assertIn(
            "EC dump item B: shape=[6, 2], chunks=2, tail_rows=0, bytes=48",
            captured.records[0].getMessage(),
        )

    def test_failed_event_skips_dump_and_preserves_prior_work(self):
        for failure in (0, RuntimeError("injected event failure")):
            with self.subTest(failure=failure):
                self.setUp()
                self.connector._save_device.get_event_handle.side_effect = [11, failure]
                self.connector.save_caches({"A": torch.ones(3, 2)}, "A")
                with patch.object(ec.logger, "exception") as log_error:
                    self.connector.save_caches({"B": torch.ones(4, 2)}, "B")
                log_error.assert_called_once()
                self.assertEqual(self.calls, [("dump", 0)])
                self.assertEqual(len(self.connector._pending_saves), 1)
                self.connector._save_device.synchronize.assert_not_called()
                self.connector._save_device.destroy_event_handle.assert_not_called()
                self.connector.get_finished(set())
                self.assertEqual(self.calls, [("dump", 0), ("wait", 0)])
                self.connector._save_device.destroy_event_handle.assert_called_once_with(
                    11
                )

    def test_producer_and_rank_gates_prevent_submission(self):
        for producer, rank in ((False, True), (True, False)):
            self.connector._is_producer = producer
            self.connector.is_save_rank = rank
            self.connector.save_caches({}, "absent")
        self.assertFalse(self.calls)
        self.connector._save_device.get_event_handle.assert_not_called()

    def test_invalid_layout_never_submits_and_prior_work_can_be_drained(self):
        self.connector.save_caches({"A": torch.ones(3, 2)}, "A")
        with self.assertRaises(ec.EncoderCacheLayoutError):
            self.connector.save_caches({"B": torch.ones(3, 4)}, "B")
        self.connector.get_finished(set())
        self.connector.clear_connector_metadata()
        self.assertEqual(self.calls, [("dump", 0), ("wait", 0)])

    def test_address_arithmetic_handles_unsigned_high_bit_and_storage_offset(self):
        addresses = self.connector._chunk_addresses(
            Mock(data_ptr=lambda: 2**63 + 128), 3
        )
        self.assertEqual(
            addresses[:, 0].tolist(), [2**63 + 128 + i * 24 for i in range(3)]
        )
        tensor = torch.arange(16, dtype=torch.float32).reshape(8, 2)[2:]
        self.connector.save_caches({"A": tensor}, "A")
        self.connector.get_finished(set())
        torch.testing.assert_close(self.completed[0], tensor)


class ECConnectorStateTest(unittest.TestCase):
    def setUp(self):
        self.connector = ec.UCMECConnector.__new__(ec.UCMECConnector)
        self.connector.enable_epd_disaggregate = False
        self.connector._is_producer = False
        self.connector.req_to_state = {}
        self.connector.identifier_to_blocks = {}
        self.connector.step_verified_hits = set()
        self.connector.pending_loads = {}
        self.connector._pending_saves = []
        self.connector._local_encoder_ids = set()
        self.connector._step_saved_ids = set()
        self.connector._is_consumer = True
        self.connector._connector_metadata = None
        self.connector.layout = ec.EncoderCacheLayout(
            width=2, dtype=torch.float32, rows_per_chunk=3, chunk_bytes=24
        )
        self.connector._block_hasher = Mock(
            side_effect=lambda value: hashlib.sha256(repr(value).encode()).digest()
        )
        self.connector.store = Mock()
        self.connector.store.lookup_on_prefix.side_effect = lambda ids: len(ids) - 1

    def ensure(self, request):
        self.assertTrue(self.connector.ensure_cache_available(request, 0))

    def select_load(self, request, index=0):
        identifier = request.mm_features[index].identifier
        self.assertTrue(self.connector.has_cache_item(identifier))
        self.connector.update_state_after_alloc(request, index)

    def metadata(self, *, freed_mm_hashes=(), **scheduled_tokens):
        return self.connector.build_connector_meta(
            SimpleNamespace(
                num_scheduled_tokens=scheduled_tokens,
                free_encoder_mm_hashes=list(freed_mm_hashes),
            )
        )

    def test_streaming_append_registers_new_items_once_and_reuses_shared_state(self):
        first = FakeRequest("first", ("X", 4), ("X", 4))
        other = FakeRequest("other", ("Y", 7))
        self.ensure(first)
        self.ensure(other)
        shared = self.connector.identifier_to_blocks["Y"]
        hash_calls = self.connector._block_hasher.call_count
        first.append(("X", 4), ("Y", 7), ("Y", 7), ("Z", 2))
        self.ensure(first)
        self.ensure(first)

        states = self.connector.identifier_to_blocks
        self.assertIs(states["Y"], shared)
        self.assertIs(
            self.connector.req_to_state["first"].identifier_states["Y"], shared
        )
        self.assertIs(
            self.connector.req_to_state["other"].identifier_states["Y"], shared
        )
        # Only Z is new to the connector, and its two rows fit in one chunk.
        self.assertEqual(self.connector._block_hasher.call_count, hash_calls + 1)
        self.assertEqual(
            {key: state.request_ref_count for key, state in states.items()},
            {"X": 1, "Y": 2, "Z": 1},
        )
        self.select_load(first, 3)
        self.select_load(first, 5)
        self.assertEqual(set(self.metadata(first=1).loads), {"Y", "Z"})

        self.connector.request_finished(first)
        self.connector.request_finished(first)
        self.assertEqual(set(states), {"Y"})
        self.assertEqual(shared.request_ref_count, 1)
        self.connector.request_finished(other)
        self.assertFalse(states)
        self.assertFalse(self.connector.req_to_state)

    def test_unchanged_features_are_not_revalidated(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        request.get_num_encoder_embeds = Mock(side_effect=AssertionError("old feature"))
        self.ensure(request)
        request.get_num_encoder_embeds.assert_not_called()

    def test_empty_request_can_later_receive_media(self):
        request = FakeRequest("request")
        self.ensure(request)
        request.append(("X", 4))
        self.ensure(request)
        self.assertTrue(self.connector.has_cache_item("X"))

    def test_append_conflict_keeps_old_registration_and_commits_no_partial_batch(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        request.append(("Y", 5), ("X", 8))
        with self.assertRaisesRegex(ec.EncoderCacheLayoutError, "within request"):
            self.connector.ensure_cache_available(request, 0)
        self.assertEqual(set(self.connector.identifier_to_blocks), {"X"})
        self.assertEqual(
            self.connector.req_to_state["request"].processed_feature_count, 1
        )

        # A corrected uncommitted batch must still be visited on the next call.
        request.mm_features[-1].rows = 4
        self.ensure(request)
        self.assertEqual(self.connector.identifier_to_blocks["Y"].request_ref_count, 1)
        self.assertEqual(self.connector.identifier_to_blocks["X"].request_ref_count, 1)

    def test_duplicate_in_new_batch_with_different_rows_is_rejected(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        request.append(("Y", 5), ("Y", 8))
        with self.assertRaisesRegex(ec.EncoderCacheLayoutError, "within request"):
            self.connector.ensure_cache_available(request, 0)
        self.assertEqual(set(self.connector.identifier_to_blocks), {"X"})

    def test_cross_request_append_conflict_does_not_change_existing_refs(self):
        request = FakeRequest("request", ("X", 4))
        other = FakeRequest("other", ("Y", 5))
        self.ensure(request)
        self.ensure(other)
        request.append(("Z", 2), ("Y", 8))
        with self.assertRaisesRegex(
            ec.EncoderCacheLayoutError, "across active requests"
        ):
            self.connector.ensure_cache_available(request, 0)
        self.assertEqual(set(self.connector.identifier_to_blocks), {"X", "Y"})
        self.connector.request_finished(request)
        self.assertEqual(self.connector.identifier_to_blocks["Y"].request_ref_count, 1)

    def test_removed_features_are_rejected_without_losing_refs(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        request.mm_features.clear()
        with self.assertRaisesRegex(ec.EncoderCacheLayoutError, "cannot be removed"):
            self.connector.ensure_cache_available(request, 0)
        self.connector.request_finished(request)
        self.assertFalse(self.connector.identifier_to_blocks)

    def test_cancelled_request_loads_are_filtered_using_final_schedule(self):
        cancelled = FakeRequest("cancelled", ("X", 4))
        surviving = FakeRequest("surviving", ("Y", 5))
        self.ensure(cancelled)
        self.ensure(surviving)
        self.select_load(cancelled)
        self.select_load(surviving)
        metadata = self.metadata(surviving=2, cancelled=0)
        self.assertEqual(set(metadata.loads), {"Y"})
        self.assertFalse(self.connector.pending_loads)
        self.assertFalse(self.connector.step_verified_hits)
        # Step rollback must not end the preempted request's registration.
        self.ensure(cancelled)
        self.assertEqual(self.connector.identifier_to_blocks["X"].request_ref_count, 1)
        self.assertFalse(self.metadata(surviving=1).loads)
        self.assertEqual(set(metadata.loads), {"Y"})

    def test_local_sharing_survivor_keeps_load_without_its_own_alloc_callback(self):
        cancelled = FakeRequest("cancelled", ("X", 4))
        surviving = FakeRequest("surviving", ("X", 4))
        self.ensure(cancelled)
        self.ensure(surviving)
        self.select_load(cancelled)
        # The cache manager can let surviving reuse X without a connector callback.
        self.assertEqual(set(self.metadata(surviving=1).loads), {"X"})

    def test_evicted_load_is_filtered_despite_future_shared_identifier(self):
        manager = EncoderCacheManager(cache_size=8)
        preempted = FakeRequest("A", ("X", 4))
        surviving = FakeRequest("B", ("Y", 4), ("X", 4))
        other = FakeRequest("C", ("Z", 4))
        for request in (preempted, surviving, other):
            self.ensure(request)

        self.assertTrue(manager.can_allocate(preempted, 0, 8, 0))
        manager.allocate(preempted, 0)
        self.select_load(preempted)
        # B only uses Y in this step; its future X has no cache reference.
        self.assertTrue(manager.can_allocate(surviving, 0, 8, 0))
        manager.allocate(surviving, 0)
        manager.free(preempted)
        self.assertTrue(manager.can_allocate(other, 0, 8, 0))
        manager.allocate(other, 0)
        freed = manager.get_freed_mm_hashes()

        self.assertEqual(set(manager.cached), {"Y", "Z"})
        self.assertEqual(freed, ["X"])
        metadata = self.metadata(freed_mm_hashes=freed, B=4, C=4)
        self.assertNotIn("X", metadata.loads)
        self.assertFalse(self.connector.pending_loads)
        self.assertFalse(self.connector.step_verified_hits)

    def test_referenced_shared_load_survives_preemption(self):
        manager = EncoderCacheManager(cache_size=4)
        preempted = FakeRequest("A", ("X", 4))
        surviving = FakeRequest("B", ("X", 4))
        for request in (preempted, surviving):
            self.ensure(request)

        self.assertTrue(manager.can_allocate(preempted, 0, 4, 0))
        manager.allocate(preempted, 0)
        self.select_load(preempted)
        self.assertTrue(manager.check_and_update_cache(surviving, 0))
        manager.free(preempted)
        freed = manager.get_freed_mm_hashes()

        self.assertEqual(manager.cached["X"], {"B"})
        self.assertEqual(freed, [])
        # B shares X without its own connector allocation callback.
        metadata = self.metadata(freed_mm_hashes=freed, B=4)
        self.assertEqual(set(metadata.loads), {"X"})

    def test_same_step_reallocation_keeps_external_load(self):
        manager = EncoderCacheManager(cache_size=4)
        preempted = FakeRequest("A", ("X", 4))
        other = FakeRequest("C", ("Z", 4))
        surviving = FakeRequest("B", ("X", 4))
        for request in (preempted, other, surviving):
            self.ensure(request)

        self.assertTrue(manager.can_allocate(preempted, 0, 4, 0))
        manager.allocate(preempted, 0)
        self.select_load(preempted)
        manager.free(preempted)
        self.assertTrue(manager.can_allocate(other, 0, 4, 0))
        manager.allocate(other, 0)
        self.assertNotIn("X", manager.cached)
        manager.free(other)
        self.assertTrue(manager.can_allocate(surviving, 0, 4, 0))
        manager.allocate(surviving, 0)
        self.select_load(surviving)
        freed = manager.get_freed_mm_hashes()

        self.assertEqual(manager.cached, {"X": {"B"}})
        self.assertEqual(freed, ["Z"])
        metadata = self.metadata(freed_mm_hashes=freed, B=4)
        self.assertEqual(set(metadata.loads), {"X"})

    def test_no_scheduled_requests_produces_no_loads(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        self.select_load(request)
        self.assertFalse(self.metadata().loads)

    def test_hit_summary_logged_once_at_request_finish(self):
        request = FakeRequest("request", ("X", 4), ("Y", 5))
        self.ensure(request)
        self.select_load(request, 0)
        self.select_load(request, 1)
        # Steps never log; hits accumulate silently until the request ends.
        with self.assertNoLogs(ec.logger, level="INFO"):
            metadata = self.metadata(request=1)
        self.assertEqual(set(metadata.loads), {"X", "Y"})

        with self.assertLogs(ec.logger, level="INFO") as captured:
            self.connector.request_finished(request)
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertIn("req=request", message)
        self.assertIn("mm_items=2", message)
        self.assertIn("hits_total=2", message)

        # Finishing again is a no-op and logs nothing.
        with self.assertNoLogs(ec.logger, level="INFO"):
            self.connector.request_finished(request)

    def test_all_miss_request_gets_zero_hit_summary_at_finish(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        # No alloc callback ever verifies a hit, so steps stay silent...
        with self.assertNoLogs(ec.logger, level="INFO"):
            self.metadata(request=1)
        # ...and the request's single summary line reports the zero.
        with self.assertLogs(ec.logger, level="INFO") as captured:
            self.connector.request_finished(request)
        message = captured.records[0].getMessage()
        self.assertIn("req=request", message)
        self.assertIn("mm_items=1", message)
        self.assertIn("hits_total=0", message)

    def test_lookup_boundary_returns_drive_the_full_hit_predicate(self):
        # {-1, len-2, len-1} are the three boundary returns of the store's
        # prefix lookup; only len-1 may count as a hit (doc §8.2 matrix).
        # A partial prefix must never be scheduled as an external load: the
        # item is dropped from the encoder compute plan on a hit, so loading
        # a truncated tensor would be unrecoverable.
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        chunks = len(self.connector.identifier_to_blocks["X"].chunk_ids)
        self.assertGreaterEqual(chunks, 2)
        for last, expect_hit in ((-1, False), (chunks - 2, False), (chunks - 1, True)):
            with self.subTest(last=last, expect_hit=expect_hit):
                self.setUp()
                request = FakeRequest("request", ("X", 4))
                self.ensure(request)
                self.connector.store.lookup_on_prefix.side_effect = (
                    lambda ids, last=last: last
                )
                self.assertEqual(self.connector.has_cache_item("X"), expect_hit)
                self.connector.update_state_after_alloc(request, 0)
                self.assertEqual(
                    set(self.metadata(request=1).loads),
                    {"X"} if expect_hit else set(),
                )
                with self.assertLogs(ec.logger, level="INFO") as captured:
                    self.connector.request_finished(request)
                self.assertIn(
                    f"hits_total={1 if expect_hit else 0}",
                    captured.records[0].getMessage(),
                )

    def test_shutdown_clears_streaming_and_step_state(self):
        request = FakeRequest("request", ("X", 4))
        self.ensure(request)
        request.append(("Y", 5))
        self.ensure(request)
        self.select_load(request, 1)
        store = self.connector.store
        self.connector._local_encoder_ids.add("old")
        self.connector._step_saved_ids.add("old")
        self.connector.shutdown()
        self.assertFalse(self.connector.req_to_state)
        self.assertFalse(self.connector.identifier_to_blocks)
        self.assertFalse(self.connector.pending_loads)
        self.assertFalse(self.connector.step_verified_hits)
        self.assertFalse(self.connector._local_encoder_ids)
        self.assertFalse(self.connector._step_saved_ids)
        store.close.assert_called_once()

    def test_epd_local_reuse_survives_request_finish_until_actual_eviction(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 1000
        first = FakeRequest("first", ("X", 4))
        self.ensure(first)
        self.select_load(first)
        self.assertNotIn("X", self.connector._local_encoder_ids)
        self.assertEqual(set(self.metadata(first=4).loads), {"X"})
        self.assertEqual(self.connector._local_encoder_ids, {"X"})
        self.connector.update_state_after_free(first, 0)
        self.connector.request_finished(first)
        self.assertEqual(self.connector._local_encoder_ids, {"X"})

        lookup = self.connector.store.lookup_on_prefix
        lookup.reset_mock()
        lookup.side_effect = lambda ids: -1
        second = FakeRequest("second", ("X", 4))
        with self.assertLogs(ec.logger, level="DEBUG") as logs:
            self.ensure(second)
        self.assertTrue(any("EC local cache reuse" in line for line in logs.output))
        lookup.assert_not_called()
        state = self.connector.req_to_state[second.request_id]
        self.assertFalse(state.store_visible_identifiers)

        self.metadata(freed_mm_hashes=("X",))
        self.assertFalse(self.connector._local_encoder_ids)
        self.assertFalse(self.connector.ensure_cache_available(second, 0))
        lookup.assert_called_once()

    def test_epd_only_final_loads_enter_local_records(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 1000
        cancelled = FakeRequest("cancelled", ("X", 4))
        survivor = FakeRequest("survivor", ("Y", 4))
        evicted = FakeRequest("evicted", ("Z", 4))
        for request in (cancelled, survivor, evicted):
            self.ensure(request)
            self.select_load(request)
        metadata = self.metadata(
            freed_mm_hashes=("Z",), cancelled=0, survivor=4, evicted=4
        )
        self.assertEqual(set(metadata.loads), {"Y"})
        self.assertEqual(self.connector._local_encoder_ids, {"Y"})
        self.assertFalse(metadata.save_identifiers)

    def test_epd_local_and_store_hits_remain_separate_while_waiting(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 1000
        self.connector._local_encoder_ids.add("local")
        request = FakeRequest("mixed", ("local", 4), ("stored", 4), ("missing", 4))
        self.ensure(request)
        state = self.connector.req_to_state[request.request_id]
        # Start a fresh visibility observation after computing chunk IDs.
        state.store_visible_identifiers.clear()
        ids = {
            key: list(item.chunk_ids) for key, item in state.identifier_states.items()
        }
        lookup = self.connector.store.lookup_on_prefix
        lookup.reset_mock()
        lookup.side_effect = lambda blocks: (
            len(blocks) - 1 if blocks == ids["stored"] else -1
        )
        with patch.object(ec.time, "monotonic", return_value=100.0):
            self.assertFalse(self.connector.ensure_cache_available(request, 0))
        self.assertEqual(
            lookup.call_args_list, [call(ids["stored"]), call(ids["missing"])]
        )
        self.assertEqual(state.store_visible_identifiers, {"stored"})
        self.assertEqual(state.store_wait_started_at, 100.0)
        with patch.object(ec.time, "monotonic", return_value=100.02):
            self.assertFalse(self.connector.ensure_cache_available(request, 0))
        self.assertEqual(lookup.call_args_list[-1], call(ids["missing"]))
        self.assertEqual(state.store_wait_started_at, 100.0)

    def test_epd_producer_save_candidates_include_cached_and_shared_media(self):
        self.connector.enable_epd_disaggregate = True
        self.connector._is_producer = True
        self.connector._is_consumer = False
        active = FakeRequest("active", ("cached", 4), ("new", 4), ("cached", 4))
        shared = FakeRequest("shared", ("cached", 4))
        cancelled = FakeRequest("cancelled", ("other", 4))
        for request in (active, shared, cancelled):
            self.ensure(request)
        # No allocation callback: all candidates can already be in E HBM.
        metadata = self.metadata(active=4, shared=4, cancelled=0)
        self.assertEqual(metadata.save_identifiers, ("cached", "new"))
        self.assertFalse(metadata.loads)
        self.assertFalse(self.connector._local_encoder_ids)
        self.assertEqual(
            self.metadata(freed_mm_hashes=("cached",), active=4).save_identifiers,
            ("new",),
        )
        self.assertFalse(self.metadata().save_identifiers)
        self.connector.enable_epd_disaggregate = False
        self.assertFalse(self.metadata(active=4).save_identifiers)

    def test_epd_visibility_defers_only_missing_requests_and_never_preloads(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 1000
        waiting = FakeRequest("waiting", ("X", 4))
        ready = FakeRequest("ready", ("Y", 3))
        self.connector._register_encoder_inputs(waiting)
        missing_ids = list(self.connector.identifier_to_blocks["X"].chunk_ids)
        self.connector.store.lookup_on_prefix.side_effect = lambda ids: (
            -1 if ids == missing_ids else len(ids) - 1
        )
        with patch.object(ec.time, "monotonic", return_value=100.0) as now:
            self.assertFalse(self.connector.ensure_cache_available(waiting, 0))
            queries = self.connector.store.lookup_on_prefix.call_count
            self.assertFalse(self.connector.ensure_cache_available(waiting, 0))
            self.assertEqual(self.connector.store.lookup_on_prefix.call_count, queries)
            self.assertTrue(self.connector.ensure_cache_available(ready, 0))
            self.assertFalse(self.connector.pending_loads)

            # Even if Store changes after the gate, EPD must select external load.
            self.connector.store.lookup_on_prefix.reset_mock()
            self.select_load(ready)
            self.connector.store.lookup_on_prefix.assert_not_called()
            self.assertEqual(set(self.metadata(ready=3).loads), {"Y"})

            now.return_value = 100.02
            self.connector.store.lookup_on_prefix.side_effect = lambda ids: len(ids) - 1
            self.assertTrue(self.connector.ensure_cache_available(waiting, 0))
            self.connector.store.lookup_on_prefix.reset_mock()
            self.assertTrue(self.connector.ensure_cache_available(waiting, 2))
            self.connector.store.lookup_on_prefix.assert_not_called()

    def test_epd_visibility_timeout_does_not_renew_and_append_rechecks(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 100
        request = FakeRequest("request", ("X", 4))
        self.connector.store.lookup_on_prefix.return_value = -1
        self.connector.store.lookup_on_prefix.side_effect = None
        with patch.object(ec.time, "monotonic", return_value=10.0) as now:
            self.assertFalse(self.connector.ensure_cache_available(request, 0))
            now.return_value = 10.02
            self.assertFalse(self.connector.ensure_cache_available(request, 0))
            now.return_value = 10.101
            with self.assertRaisesRegex(TimeoutError, "request"):
                self.connector.ensure_cache_available(request, 0)
        self.connector.request_finished(request)
        self.assertFalse(self.connector.req_to_state)
        self.assertFalse(self.connector.identifier_to_blocks)

        request = FakeRequest("append", ("X", 4))
        self.connector.store.lookup_on_prefix.side_effect = lambda ids: len(ids) - 1
        with patch.object(ec.time, "monotonic", return_value=20.0) as now:
            self.assertTrue(self.connector.ensure_cache_available(request, 0))
            request.append(("Y", 7))
            self.connector.store.lookup_on_prefix.side_effect = None
            self.connector.store.lookup_on_prefix.return_value = -1
            self.assertFalse(self.connector.ensure_cache_available(request, 0))
            state = self.connector.req_to_state["append"]
            self.assertEqual(state.store_visible_identifiers, {"X"})
            self.assertEqual(state.store_wait_started_at, 20.0)
            now.return_value = 20.02
            self.connector.store.lookup_on_prefix.side_effect = lambda ids: len(ids) - 1
            self.assertTrue(self.connector.ensure_cache_available(request, 0))
            self.assertEqual(state.store_visible_identifiers, {"X", "Y"})
            self.assertIsNone(state.store_wait_started_at)

            self.connector.store.lookup_on_prefix.reset_mock()
            request.append(("Y", 7))
            self.assertTrue(self.connector.ensure_cache_available(request, 0))
            self.connector.store.lookup_on_prefix.assert_not_called()

    def test_epd_visibility_kv_coverage_and_rollback(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 100
        lookup = self.connector.store.lookup_on_prefix
        for elapsed in (0.001, 0.101):
            with self.subTest(elapsed=elapsed):
                request = FakeRequest(f"coverage-{elapsed}", ("X", 4), ("Y", 3))
                self.connector._register_encoder_inputs(request)
                x_ids = list(self.connector.identifier_to_blocks["X"].chunk_ids)
                y_ids = list(self.connector.identifier_to_blocks["Y"].chunk_ids)
                lookup.reset_mock()
                lookup.side_effect = lambda ids: len(ids) - 1
                with patch.object(ec.time, "monotonic", return_value=10.0) as now:
                    # The [0, 4) item is fully covered; the future item is not.
                    self.assertTrue(self.connector.ensure_cache_available(request, 4))
                    lookup.assert_called_once_with(y_ids)
                    state = self.connector.req_to_state[request.request_id]
                    self.assertEqual(state.store_visible_identifiers, {"Y"})

                    # A rollback into X needs its entire item, even if Y vanished.
                    lookup.reset_mock()
                    lookup.side_effect = None
                    lookup.return_value = -1
                    self.assertFalse(self.connector.ensure_cache_available(request, 3))
                    lookup.assert_called_once_with(x_ids)
                    self.assertEqual(state.store_wait_started_at, 10.0)

                    # Coverage wins over both the throttle and an expired wait.
                    now.return_value = 10.0 + elapsed
                    self.assertTrue(self.connector.ensure_cache_available(request, 7))
                    lookup.assert_called_once_with(x_ids)
                    self.assertIsNone(state.store_wait_started_at)
                    self.assertEqual(state.next_lookup_at, 0.0)
                    self.assertEqual(state.store_visible_identifiers, {"Y"})

                    # Skipping X for KV coverage never marks it as a lookup hit.
                    self.assertFalse(self.connector.ensure_cache_available(request, 3))
                    self.assertEqual(lookup.call_args_list, [call(x_ids), call(x_ids)])
                    self.assertEqual(state.store_wait_started_at, now.return_value)
                self.connector.request_finished(request)

    def test_epd_visibility_keeps_full_hits_while_other_items_wait(self):
        self.connector.enable_epd_disaggregate = True
        self.connector.epd_timeout_ms = 1000
        request = FakeRequest("waiting", ("A", 4), ("A", 4), ("B", 7), ("C", 2))
        self.connector._register_encoder_inputs(request)
        ids = {
            key: list(value.chunk_ids)
            for key, value in self.connector.identifier_to_blocks.items()
        }
        lookup = self.connector.store.lookup_on_prefix
        lookup.side_effect = lambda blocks: (
            len(blocks) - 2 if blocks == ids["B"] else len(blocks) - 1
        )
        with patch.object(ec.time, "monotonic", return_value=100.0) as now:
            # First A is covered, but its second occurrence still requires EC.
            self.assertFalse(self.connector.ensure_cache_available(request, 4))
            state = self.connector.req_to_state[request.request_id]
            self.assertEqual(state.store_visible_identifiers, {"A"})
            self.assertEqual(lookup.call_args_list, [call(ids["A"]), call(ids["B"])])
            self.assertFalse(self.connector.ensure_cache_available(request, 4))
            self.assertEqual(lookup.call_count, 2)

            # A may disappear; the request-local positive snapshot is retained.
            lookup.side_effect = lambda blocks: (
                -1 if blocks == ids["A"] else len(blocks) - 2
            )
            now.return_value = 100.02
            self.assertFalse(self.connector.ensure_cache_available(request, 4))
            self.assertEqual(state.store_visible_identifiers, {"A"})
            self.assertEqual(state.store_wait_started_at, 100.0)
            self.assertEqual(lookup.call_args_list[-1], call(ids["B"]))

            now.return_value = 100.04
            lookup.side_effect = lambda blocks: (
                len(blocks) - 2 if blocks == ids["C"] else len(blocks) - 1
            )
            self.assertFalse(self.connector.ensure_cache_available(request, 4))
            self.assertEqual(state.store_visible_identifiers, {"A", "B"})
            self.assertEqual(state.store_wait_started_at, 100.0)

            now.return_value = 100.06
            lookup.side_effect = lambda blocks: len(blocks) - 1
            self.assertTrue(self.connector.ensure_cache_available(request, 4))
            self.assertEqual(state.store_visible_identifiers, {"A", "B", "C"})
            self.assertIsNone(state.store_wait_started_at)
            self.assertEqual(
                lookup.call_args_list,
                [
                    call(ids["A"]),
                    call(ids["B"]),
                    call(ids["B"]),
                    call(ids["B"]),
                    call(ids["C"]),
                    call(ids["C"]),
                ],
            )

            # An identical item in another request must perform its own lookup.
            lookup.reset_mock()
            lookup.side_effect = None
            lookup.return_value = -1
            other = FakeRequest("other", ("A", 4))
            self.assertFalse(self.connector.ensure_cache_available(other, 0))
            lookup.assert_called_once_with(ids["A"])
        self.connector.request_finished(request)
        self.assertNotIn(request.request_id, self.connector.req_to_state)

    def test_epd_producer_returns_native_metadata_without_consumer_state(self):
        self.connector.enable_epd_disaggregate = True
        self.connector._is_producer = True
        self.connector._is_consumer = False
        resolver = Mock()
        resolver.fields_for.return_value = {"image_grid_thw"}
        self.connector._metadata_resolver = resolver
        request = FakeRequest("encoder", ("image", 4), ("image", 4))
        for feature in request.mm_features:
            feature.modality = "image"
            feature.data = SimpleNamespace(
                get_data=lambda: {"image_grid_thw": torch.tensor([[1, 2, 2]])}
            )
        self.assertFalse(self.connector.req_to_state)
        delayed, params = self.connector.request_finished(request)
        self.assertFalse(delayed)
        self.assertEqual(
            params,
            {
                "image": {
                    "modality": "image",
                    "item_indices": [0, 1],
                    "metadata": {"image_grid_thw": [[1, 2, 2]]},
                }
            },
        )


class EncoderWidthTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            hf_config=None, get_inputs_embeds_size=Mock(return_value=5120)
        )
        self.engine = SimpleNamespace(
            model_config=self.model, ec_transfer_config=SimpleNamespace()
        )

    def width(self, options=None):
        return ec.resolve_encoder_cache_width(self.engine, options or {})

    def test_measured_width_checks_explicit_layout(self):
        self.engine.ec_transfer_config.mm_encoder_output_widths = {
            "image": 8192,
            "video": 8192,
        }
        self.assertEqual(self.width({"encoder_cache_hidden_dim": 8192}), 8192)
        self.model.get_inputs_embeds_size.assert_not_called()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.width({"encoder_cache_hidden_dim": 5120})

    def test_late_measurement_cannot_silently_change_worker_layout(self):
        self.assertEqual(self.width(), 5120)
        self.engine.ec_transfer_config.mm_encoder_output_widths = {"image": 8192}
        with self.assertRaisesRegex(ValueError, "Set encoder_cache_hidden_dim=8192"):
            self.width()
        self.engine.ec_transfer_config.mm_encoder_output_widths = {"image": 5120}
        self.assertEqual(self.width(), 5120)

    def test_variable_or_invalid_measured_width_is_rejected(self):
        for measured in (
            {"image": 8192, "audio": 5120},
            {"image": 0},
            {"image": True},
            {"image": 1.5},
            [8192],
            [],
        ):
            self.engine.ec_transfer_config.mm_encoder_output_widths = measured
            with self.subTest(measured=measured), self.assertRaises(ValueError):
                self.width()

    def test_explicit_width_needs_no_model_guess(self):
        self.assertEqual(self.width({"encoder_cache_hidden_dim": 4096}), 4096)
        self.model.get_inputs_embeds_size.assert_not_called()
        for value in (True, 0, -1, "4096", 1.5):
            with self.subTest(width=value), self.assertRaises(ValueError):
                self.width({"encoder_cache_hidden_dim": value})

    def test_legacy_deepstack_layout_without_private_vllm_import(self):
        vision = SimpleNamespace(
            out_hidden_size=1024, deepstack_visual_indexes=[1, 3, 5]
        )
        self.model.hf_config = SimpleNamespace(vision_config=vision)
        self.assertEqual(self.width(), 4096)
        self.model.get_inputs_embeds_size.assert_not_called()
        for layers in ([], None):
            vision.deepstack_visual_indexes = layers
            self.assertEqual(self.width(), 5120)
        del vision.out_hidden_size
        vision.deepstack_visual_indexes = [1, 3]
        self.assertEqual(self.width(), 5120)


@unittest.skipUnless(
    os.getenv("UCM_EPD_PROXY_URL"),
    "Set UCM_EPD_PROXY_URL for container EPD validation.",
)
class EPDEndToEndTest(unittest.TestCase):
    def test_image_request_and_reuse(self):
        from PIL import Image

        image = io.BytesIO()
        Image.new("RGB", (224, 224), color=(220, 30, 20)).save(image, format="PNG")
        image_url = (
            "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
        )
        payload = {
            "model": os.environ["UCM_EPD_MODEL"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Describe the image in one sentence."},
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for attempt in range(2):
            with self.subTest(attempt=attempt):
                request = urllib.request.Request(
                    os.environ["UCM_EPD_PROXY_URL"].rstrip("/")
                    + "/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
                self.assertTrue(result["choices"][0]["message"]["content"])
                self.assertGreater(result["usage"]["prompt_tokens"], 0)

        if log_dir := os.getenv("UCM_EPD_LOG_DIR"):
            logs = Path(log_dir)
            encoder = (logs / "encoder.log").read_text()
            decoder = (logs / "prefill-decode.log").read_text()
            proxy = (logs / "proxy.log").read_text()
            self.assertIn("as metadata references", proxy)
            saved = set(re.findall(r"EC dump item ([^: ]+):", encoder))
            loaded = set(re.findall(r"EC load item ([^: ]+):", decoder))
            self.assertTrue(
                saved & loaded, "E and PD must transfer the same identifier."
            )
            self.assertRegex(
                encoder, r"EC cached saves: candidates=\d+, submitted=[1-9]"
            )
            self.assertIn("EC local cache reuse:", decoder)


@unittest.skipUnless(
    os.getenv("UCM_EPD_DELETE_REPRO_ROOT"),
    "Set UCM_EPD_DELETE_REPRO_ROOT for the destructive, isolated EPD repro.",
)
class EPDSharedStorageDeletionReproTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ["UCM_EPD_DELETE_REPRO_ROOT"]).resolve()
        marker = self.root / ".epd-delete-repro"
        self.assertEqual(marker.read_text(), os.environ["UCM_EPD_DELETE_REPRO_TOKEN"])
        self.backend = (self.root / "shared").resolve()
        self.assertEqual(self.backend.parent, self.root)
        config = yaml.safe_load((self.root / "epd.yaml").read_text())
        configured = config["ucm_ec_connector"]["ucm_connector_config"]
        self.assertEqual(Path(configured["storage_backends"]).resolve(), self.backend)
        self.logs = self.root / "logs"

    def payload(self):
        from PIL import Image

        image = io.BytesIO()
        Image.new("RGB", (224, 224), color=(30, 160, 210)).save(image, format="PNG")
        return {
            "model": os.environ["UCM_EPD_MODEL"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image.getvalue()).decode()
                            },
                        },
                        {"type": "text", "text": "Name the main color briefly."},
                    ],
                }
            ],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def post(self, label):
        request = urllib.request.Request(
            os.environ["UCM_EPD_PROXY_URL"].rstrip("/") + "/v1/chat/completions",
            data=json.dumps(self.payload()).encode(),
            headers={"Content-Type": "application/json", "x-request-id": label},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status, body = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, body = error.code, error.read().decode()
        result = {
            "status": status,
            "elapsed_s": time.monotonic() - started,
            "body": body,
        }
        (self.root / f"{label}.json").write_text(json.dumps(result, indent=2))
        print(label, "status=", status, "elapsed_s=", result["elapsed_s"], flush=True)
        return result

    def cache_files(self):
        files = []
        for path in self.backend.rglob("*"):
            if not path.is_file() or not re.fullmatch(r"[0-9a-f]{32}", path.name):
                continue
            self.assertFalse(path.is_symlink())
            self.assertTrue(path.resolve().is_relative_to(self.backend))
            files.append(path)
        return files

    def counters(self):
        return {
            "e_dump": (self.logs / "encoder.log").read_text().count("EC dump item "),
            "pd_load": (self.logs / "prefill-decode.log")
            .read_text()
            .count("EC load item "),
        }

    def test_delete_then_repeat_with_warm_pd(self):
        self.assertFalse(self.cache_files(), "Use a fresh, dedicated backend.")
        first = self.post("first")
        self.assertEqual(first["status"], 200, first)
        self.assertTrue(first["body"]["choices"][0]["message"]["content"])
        before = self.counters()
        self.assertGreater(before["e_dump"], 0)
        self.assertGreater(before["pd_load"], 0)
        files = self.cache_files()
        self.assertTrue(files, "No published EC file to delete.")
        deleted = [{"path": str(path), "bytes": path.stat().st_size} for path in files]
        for path in files:
            path.unlink()
        self.assertFalse(self.cache_files())
        second = self.post("second-warm-pd")
        after = self.counters()
        (self.root / "warm-observation.json").write_text(
            json.dumps({"deleted": deleted, "before": before, "after": after}, indent=2)
        )
        self.assertEqual(second["status"], 200, second)
        self.assertTrue(second["body"]["choices"][0]["message"]["content"])
        self.assertGreater(
            after["e_dump"], before["e_dump"], "Warm E must submit save."
        )
        self.assertEqual(after["pd_load"], before["pd_load"], "PD must reuse local EC.")
        self.assertFalse(
            self.cache_files(), "Cache2 READY unexpectedly rewrote backend."
        )

    def test_replay_after_pd_restart(self):
        self.assertTrue((self.root / "warm-observation.json").is_file())
        self.assertTrue((self.root / "pd-restarted.json").is_file())
        self.assertFalse(self.cache_files())
        before = self.counters()["e_dump"]
        result = self.post("third-cold-pd")
        self.assertEqual(result["status"], 500, result)
        self.assertIn(
            "Timed out waiting for encoder cache visibility",
            (self.logs / "prefill-decode.log").read_text(),
        )
        self.assertGreater(
            self.counters()["e_dump"], before, "Warm E must submit save."
        )
        self.assertFalse(self.cache_files())


if __name__ == "__main__":
    unittest.main()
