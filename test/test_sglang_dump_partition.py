"""Pool-aware dump partitioning without SGLang or GPU dependencies."""

import queue
import types
import unittest
from dataclasses import dataclass, replace
from unittest.mock import mock_open, patch

import numpy as np
from common.sglang_utils import linker_module


@dataclass
class Transfer:
    name: str
    keys: list[str]
    host_indices: object
    hit_policy: str = "all_pages"


class Indices:
    def __init__(self, values, copies):
        self.values = np.asarray(values)
        self.copies = copies

    def reshape(self, *shape):
        return Indices(self.values.reshape(*shape), self.copies)

    def __getitem__(self, item):
        return Indices(self.values[item], self.copies)

    def detach(self):
        return self

    def cpu(self):
        self.copies.append(self.values.copy())
        return self.values.copy()


class DumpPartitionTest(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            linker_module, "PoolName", types.SimpleNamespace(KV="kv"), create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def linker(self, rank, ranks=(0, 1, 2, 3)):
        obj = object.__new__(linker_module.UcmDirectLinker)
        obj._tp_rank, obj._tp_size = rank, 4
        obj._dump_ranks = ranks
        obj._lookup_owner = True
        obj._dp_rank = obj._attn_dp_rank = 0
        obj.num_layers = 2

        def entry(name, source, base, size, divisor):
            return types.SimpleNamespace(
                name=name,
                indices_from_pool=source,
                page_size=2,
                layer_mapping={0: 0, 1: 1},
                packed=True,
                buffer_meta=[[(base, size, size), (base + 1000, size, size)]],
                _component_offsets=[[0, 0]],
                prepare_locations=lambda indices: indices.reshape(-1, 2)[:, 0]
                // divisor,
            )

        obj.pools = {
            "kv": entry("kv", "kv", 100, 8, 1),
            "indexer": entry("indexer", "kv", 200, 4, 2),
            "swa": entry("swa", "swa", 300, 4, 2),
            "state": entry("state", "swa", 400, 8, 2),
        }
        obj.pool_group = types.SimpleNamespace(rank_replicated=True)
        obj.adapter = types.SimpleNamespace(
            _connector_name="UcmPipelineStore",
            _store_config={"store_pipeline": "Cache|Posix"},
            _lookup_store_signatures={"g": ("shared", ("/cache",))},
        )
        obj._bundles = obj._build_pool_bundles()
        obj._layer_pointer_layouts = obj._build_layer_pointer_layouts()
        return obj

    def transfers(self, pages, copies, offset=0):
        full = Indices(np.arange(pages * 2) + offset, copies)
        tail = Indices(np.arange(4) + offset, copies)
        return [
            Transfer("kv", [f"page-{i}" for i in range(pages)], full),
            Transfer("indexer", [f"page-{i}" for i in range(pages)], full),
            Transfer("swa", ["tail-a", "tail-b"], tail, "trailing_pages"),
            Transfer("state", ["tail-a", "tail-b"], tail, "trailing_pages"),
        ]

    def test_full_pages_are_disjoint_complete_and_balanced(self):
        for pages in (1, 2, 3, 4, 5, 11):
            keys_by_rank = []
            tail_owners = []
            for rank in range(4):
                copies = []
                obj = self.linker(rank)
                transfers = self.transfers(pages, copies, offset=rank * 100)
                selected = {t.name: t for t in obj._partition_offload(transfers)}
                self.assertEqual(copies, [])
                kv = selected.get("kv")
                keys_by_rank.append(kv.keys if kv else [])
                if kv is not None:
                    start = pages * rank // 4
                    end = pages * (rank + 1) // 4
                    np.testing.assert_array_equal(
                        kv.host_indices.values,
                        transfers[0].host_indices.values[start * 2 : end * 2],
                    )
                    self.assertEqual(kv.keys, selected["indexer"].keys)
                    self.assertIs(kv.host_indices, selected["indexer"].host_indices)
                if "swa" in selected:
                    tail_owners.append(rank)
                    self.assertIs(selected["swa"], transfers[2])
                    self.assertIs(selected["state"], transfers[3])
            self.assertEqual(
                [key for keys in keys_by_rank for key in keys],
                [f"page-{i}" for i in range(pages)],
            )
            self.assertLessEqual(
                max(map(len, keys_by_rank)) - min(map(len, keys_by_rank)), 1
            )
            self.assertEqual(len(tail_owners), 1)

    def test_only_selected_indices_are_snapshotted_and_addresses_match(self):
        for rank in range(4):
            copies = []
            obj = self.linker(rank)
            transfers = self.transfers(12, copies)
            selected = obj._partition_offload(transfers)
            physical = obj._page_transfers(selected, for_io=True)
            # One snapshot for KV+indexer, and one extra only at the tail owner.
            self.assertEqual(sum(len(c) == 6 for c in copies), 1)
            self.assertEqual(
                len(copies), 2 if any(t.name == "swa" for t in selected) else 1
            )
            full = next(t for t in physical if t.source_pool == "kv")
            start = 12 * rank // 4
            slots = np.arange(start * 2, (start + 3) * 2, 2)
            expected = np.column_stack((200 + (slots // 2) * 4, 100 + slots * 8))
            np.testing.assert_array_equal(full.addresses[0], expected)
            np.testing.assert_array_equal(full.addresses[1], expected + 1000)

    def test_tail_can_be_owned_without_any_full_pages(self):
        transfers = self.transfers(1, [])
        # Each domain gets its own complete KV and tail representation.
        for ranks in ((0, 1), (2, 3)):
            all_selected = {
                r: self.linker(r, ranks)._partition_offload(transfers) for r in ranks
            }
            self.assertEqual(
                sum(
                    len(t.keys)
                    for ts in all_selected.values()
                    for t in ts
                    if t.name == "kv"
                ),
                1,
            )
            self.assertEqual(
                sum(t.name == "swa" for ts in all_selected.values() for t in ts), 1
            )
        # Auxiliary-only nodes are distributed as an indivisible boundary.
        selected = [self.linker(r)._partition_offload(transfers[2:]) for r in range(4)]
        self.assertEqual(sum(bool(ts) for ts in selected), 1)
        self.assertTrue(
            all(not ts or {t.name for t in ts} == {"swa", "state"} for ts in selected)
        )

    def test_tail_only_rank_prepares_no_full_kv_indices(self):
        copies = []
        obj = self.linker(1)
        transfers = self.transfers(1, copies)
        # page-0 selects tail owner 1; the sole Full page goes to rank 3.
        selected = obj._partition_offload(transfers)
        self.assertEqual({t.name for t in selected}, {"swa", "state"})
        physical = obj._page_transfers(selected, for_io=True)
        self.assertEqual({t.source_pool for t in physical}, {"swa"})
        self.assertEqual(len(copies), 1)
        self.assertEqual(len(copies[0]), 4)

    def test_deepseek_physical_pools_follow_their_logical_source(self):
        renames = {
            "kv": "deepseek_v4_c4",
            "indexer": "deepseek_v4_c4_indexer",
            "state": "deepseek_v4_c4_state",
        }
        all_keys = []
        for rank in range(4):
            obj = self.linker(rank)
            pools = {}
            for name, entry in obj.pools.items():
                entry.name = renames.get(name, name)
                pools[entry.name] = entry
            obj.pools = pools
            transfers = [
                replace(t, name=renames.get(t.name, t.name))
                for t in self.transfers(8, [])
            ]
            selected = {t.name: t for t in obj._partition_offload(transfers)}
            full = selected["deepseek_v4_c4"]
            indexer = selected["deepseek_v4_c4_indexer"]
            self.assertEqual(len(full.keys), 2)
            self.assertEqual(full.keys, indexer.keys)
            self.assertIs(full.host_indices, indexer.host_indices)
            self.assertEqual("swa" in selected, "deepseek_v4_c4_state" in selected)
            all_keys.extend(full.keys)
        self.assertEqual(all_keys, [f"page-{i}" for i in range(8)])

    def test_single_rank_and_nonreplicated_pools_keep_full_transfers(self):
        transfers = self.transfers(5, [])
        for rank in range(4):
            self.assertIs(
                self.linker(rank, (rank,))._partition_offload(transfers), transfers
            )
        self.assertEqual(self.linker(1, (0,))._partition_offload(transfers), [])

    def test_offload_resolves_once_and_empty_rank_records_no_events(self):
        obj = self.linker(0)
        transfers = self.transfers(1, [])
        expanded = []
        obj._expand = lambda value, **kwargs: expanded.append(value) or value
        obj._freeze_gc_once = lambda: None
        obj._offload_queue = queue.Queue()
        events = []
        obj._record_transfer_events = lambda: events.append(True) or [object()]
        expected = obj._partition_offload(transfers)
        self.assertTrue(obj.offload(transfers))
        self.assertEqual(len(expanded), 1)
        queued, ready = obj._offload_queue.get_nowait()
        self.assertEqual([t.name for t in queued], [t.name for t in expected])
        self.assertEqual(len(events), int(bool(expected)))
        if not expected:
            self.assertEqual(ready, [])

    def test_peer_election_is_scoped_to_shared_cache_and_attention_group(self):
        signature = (("g", ("shared", ("/cache",))),)
        identities = [("host-a", signature)] * 2 + [("host-b", signature)] * 2
        group = object()
        for rank in range(4):
            obj = self.linker(rank, (0,))

            def gather(output, identity, *, group):
                self.assertIs(group, expected_group)
                self.assertEqual(identity, identities[rank])
                output[:] = identities

            expected_group = group
            with (
                patch.object(
                    linker_module,
                    "torch",
                    types.SimpleNamespace(
                        distributed=types.SimpleNamespace(
                            is_initialized=lambda: True, all_gather_object=gather
                        )
                    ),
                ),
                patch("builtins.open", mock_open(read_data=identities[rank][0])),
            ):
                obj._select_store_ranks(group, 1)
            ranks = (0, 1) if rank < 2 else (2, 3)
            self.assertEqual(obj._dump_ranks, ranks)
            self.assertEqual(obj._lookup_owner, rank == ranks[0])

    def test_unshared_or_cp_keeps_original_owner_policy(self):
        obj = self.linker(1, (0,))
        obj.adapter._store_config["share_buffer_enable"] = False

        def gather(output, identity, *, group):
            self.assertIsNone(identity)
            output[:] = [None] * 4

        with patch.object(
            linker_module,
            "torch",
            types.SimpleNamespace(
                distributed=types.SimpleNamespace(
                    is_initialized=lambda: True, all_gather_object=gather
                )
            ),
        ):
            obj._select_store_ranks(object(), 1)
        self.assertEqual(obj._dump_ranks, (0,))
        self.assertTrue(obj._lookup_owner)
        obj._select_store_ranks(object(), 2)
        self.assertEqual(obj._dump_ranks, (0,))
