"""Exercise Python preparation without SGLang, torch or native UCM libraries."""

import ast
import copy
import hashlib
import json
import logging
import re
import sys
import threading
import types
import unittest
from typing import Any, List
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


def load_source(relative_path, definitions=None, **dependencies):
    path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if definitions is not None:
        tree.body = [node for node in tree.body if getattr(node, "name", None) in definitions]
    name = "_test_" + path.stem
    module = types.ModuleType(name)
    module.__dict__.update(
        dataclass=dataclass, np=np, copy=copy, hashlib=hashlib, json=json,
        logging=logging, re=re, threading=threading, Path=Path,
        UnifiedCacheLinker=object, get_device_module=lambda: None,
        Any=Any, List=List, Task=object, UcmKVStoreBaseV1=object,
        torch=types.SimpleNamespace(Tensor=object),
    )
    module.__dict__.update(dependencies)
    sys.modules[name] = module
    code = compile(tree, str(path), "exec", flags=__import__("__future__").annotations.compiler_flag)
    exec(code, module.__dict__)
    return module


adapter_module = load_source("ucm/integration/sglang/ucm_direct_store_adapter.py")
linker_module = load_source("ucm/integration/sglang/ucm_direct_linker.py", **{
    name: getattr(adapter_module, name)
    for name in ("UcmDirectStoreAdapter", "UcmLayerTransfer", "UcmPoolMetadata")
})
pipeline_module = load_source(
    "ucm/store/pipeline/connector.py",
    definitions={"_PreparedPipelineTransfer", "UcmPipelineStoreTransTask", "UcmPipelineStore"},
)


class LayoutPreparationTest(unittest.TestCase):
    def transfer(self, **changes):
        transfer = adapter_module.UcmLayerTransfer(
            store_group="g", source_pool="kv", pools=("kv",), layers=(0, 2),
            shard_size=8, tensor_size_list=(8,), page_keys=("a", "b"),
            addresses=np.array([[[100], [108]], [[200], [208]]], dtype=np.uint64),
            fragments=((8, 0), (8, 0)),
        )
        return replace(transfer, **changes)

    def adapter(self):
        adapter = adapter_module.UcmDirectStoreAdapter(
            connector_name="test", module_path=None, store_config={}, identity={},
        )
        self.schema_calls = 0
        def ensure(*args, **kwargs):
            self.schema_calls += 1
            return self
        adapter._ensure_store_schema = ensure
        return adapter

    def test_layout_reused_but_dynamic_addresses_checked(self):
        adapter = self.adapter()
        first = adapter.prepare_load([self.transfer()])[0][0]
        second = adapter.prepare_load([self.transfer(page_keys=("c", "d"))])[0][0]
        self.assertEqual(self.schema_calls, 1)
        self.assertIs(first.shards_by_layer, second.shards_by_layer)
        self.assertNotEqual(first.block_ids, second.block_ids)
        with self.assertRaisesRegex(ValueError, "address/schema"):
            adapter.prepare_load([self.transfer(addresses=np.zeros((1, 2, 1), dtype=np.uint64))])
        with self.assertRaisesRegex(ValueError, "changed its layout"):
            adapter.prepare_load([self.transfer(layers=(0, 1))])

    def test_bundle_cache_preserves_composite_addresses(self):
        linker = object.__new__(linker_module.UcmDirectLinker)
        def entry(name, base, size):
            return types.SimpleNamespace(
                name=name, indices_from_pool="kv", layer_mapping={0: 0, 1: 1},
                packed=True, buffer_meta=[[(base, size, size), (base + 1000, size, size)]],
                _component_offsets=[[0, 0]], prepare_locations=lambda indices: indices,
            )
        linker.pools = {"kv": entry("kv", 100, 8), "indexer": entry("indexer", 200, 4)}
        linker.num_layers = 2
        linker._bundles = linker._build_pool_bundles()
        linker._layer_pointer_layouts = linker._build_layer_pointer_layouts()
        # Request preparation must use the cached schema, not rebuild it.
        linker._entry_layer_schema = lambda *args: self.fail("schema rebuilt")
        transfers = [types.SimpleNamespace(name=name, keys=("a", "b"),
                      hit_policy="all_pages", host_indices=[0, 2]) for name in linker.pools]
        result = linker._page_transfers(transfers, for_io=True)[0]
        self.assertEqual(result.tensor_size_list, (4, 8))
        self.assertEqual(result.shard_size, 12)
        np.testing.assert_array_equal(result.addresses, [
            [[200, 100], [208, 116]], [[1200, 1100], [1208, 1116]],
        ])
        lookup = linker._page_transfers(transfers, for_io=False)[0]
        self.assertIs(lookup.fragments, result.fragments)

    def test_prepared_pipeline_reuses_ids_and_preserves_shards_and_event(self):
        calls = []
        class Native:
            def Load(self, ids, indexes, addresses):
                calls.append((ids, indexes, addresses, None))
                return len(calls)
            def Dump(self, ids, indexes, addresses, event):
                calls.append((ids, indexes, addresses, event))
                return len(calls)
        store = object.__new__(pipeline_module.UcmPipelineStore)
        store.store_ = Native()
        adapter = self.adapter()
        adapter._ensure_store_schema = lambda *args, **kwargs: store
        transfer = self.transfer()
        prepared = adapter.prepare_load([transfer])
        self.assertIs(prepared[0][0], prepared[2][0])
        self.assertEqual(adapter.submit_prepared_load_layer(prepared, 1), [])
        adapter.submit_prepared_load_layer(prepared, 0)
        adapter.submit_prepared_load_layer(prepared, 2)
        self.assertIs(calls[0][0], calls[1][0])
        self.assertFalse(calls[0][0].flags.writeable)
        self.assertEqual(calls[0][0].tobytes(), b"".join(adapter._block_ids(transfer)))
        np.testing.assert_array_equal(calls[0][1], [0, 0])
        np.testing.assert_array_equal(calls[1][1], [1, 1])
        np.testing.assert_array_equal(calls[0][2], transfer.addresses[0])
        np.testing.assert_array_equal(calls[1][2], transfer.addresses[1])
        adapter.submit_offload([transfer], prerequisite_handle=42)
        self.assertEqual([call[3] for call in calls[2:]], [42, 42])

    def test_generic_fallback_and_partial_submit_drains_tasks(self):
        calls, waited = [], []
        class Store:
            def load_data(self, ids, indexes, addresses):
                calls.append((ids, indexes, addresses))
                if len(calls) == 2:
                    raise RuntimeError("submission failed")
                return 1
            def wait(self, task):
                waited.append(task)
        adapter = self.adapter()
        adapter._ensure_store_schema = lambda *args, **kwargs: Store()
        transfer = self.transfer(layers=(0, 0))
        prepared = adapter.prepare_load([transfer])
        with self.assertLogs(adapter_module.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                adapter.submit_prepared_load_layer(prepared, 0)
        self.assertEqual(waited, [1])
        self.assertEqual(calls[0][0], adapter._block_ids(transfer))
        self.assertEqual(calls[1][1], [1, 1])

    def test_load_index_contains_only_relevant_groups(self):
        adapter = self.adapter()
        first = self.transfer()
        second = self.transfer(store_group="other", layers=(1, 3))
        prepared = adapter.prepare_load([first, second])
        self.assertEqual(set(prepared), {0, 1, 2, 3})
        self.assertEqual([x.transfer.store_group for x in prepared[0]], ["g"])
        self.assertEqual([x.transfer.store_group for x in prepared[1]], ["other"])


if __name__ == "__main__":
    unittest.main()
