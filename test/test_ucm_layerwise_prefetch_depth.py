"""Unit tests for UCMLayerWiseConnector layerwise prefetch depth.

The layerwise connector drives per-layer KV loads synchronously: layer 0 is
submitted by ``start_load_kv()``, and ``wait_for_layer_load(i)`` must submit
the next ``layerwise_prefetch_depth`` layers (``i+1 .. i+depth``) once layer
*i* completes. These tests pin down that contract for depth=1 (the original
behavior) and depth>1 (pipelined prefetch), including dedup across overlapping
submit windows, repeated hook visits, end-of-model bounds, depth clamping and
the load-duration stats recording point.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing the metrics test module installs the vllm/torch stub packages and
# exposes the connector class with all external dependencies faked out.
from test_ucm_connector_metrics import (  # noqa: E402
    UCMLayerWiseConnector,
    _reset_fakes,
    fake_ucmmetrics,
)


def _make_connector(layer_count, depth):
    connector = object.__new__(UCMLayerWiseConnector)
    connector._connector_metadata = object()
    connector.need_load = True
    metadata = SimpleNamespace(
        request_meta={
            "req-1": SimpleNamespace(
                load_block_ids=([b"a", b"b"], [1, 2]),
                dump_block_ids=([], []),
            )
        }
    )
    connector._get_connector_metadata = lambda: metadata
    connector.layer_name_to_id = {f"layer.{i}": i for i in range(layer_count)}
    connector.layer_ids = set(range(layer_count))
    connector.first_layer_id = 0
    connector.load_tasks = {}
    connector._rank_consistency = SimpleNamespace(wait_load=lambda task: None)
    connector.kv_cache_layout = SimpleNamespace(shard_size=64)
    connector._failure_req_ids = set()
    connector._layerwise_load_start_by_layer = {}
    connector._layerwise_layer_load_duration_sum_ms = 0.0
    connector._layerwise_load_bytes = 0
    connector.layerwise_prefetch_depth = depth
    connector._submitted_layer_ids = set()
    connector._record_layerwise_load_duration = Mock(return_value=False)
    connector._submit_request_load_tasks_for_layer = Mock()
    return connector


def _submitted_layers(connector):
    return [
        call.args[0]
        for call in connector._submit_request_load_tasks_for_layer.call_args_list
    ]


def _wait(connector, layer_idx):
    # Seed the tasks the submit chain would have created for this layer, as
    # start_load_kv() / the previous wait_for_layer_load() would have done.
    connector.load_tasks[layer_idx] = {"req-1": object()}
    connector.wait_for_layer_load(f"layer.{layer_idx}")


def test_depth_one_submits_exactly_the_next_layer():
    """depth=1 must preserve the original behavior: submit layer i+1."""
    connector = _make_connector(layer_count=6, depth=1)
    _wait(connector, 0)
    assert _submitted_layers(connector) == [1]

    _wait(connector, 1)
    assert _submitted_layers(connector) == [1, 2]


def test_depth_four_submits_four_layers_ahead():
    connector = _make_connector(layer_count=8, depth=4)
    _wait(connector, 0)
    assert _submitted_layers(connector) == [1, 2, 3, 4]


def test_overlapping_windows_do_not_double_submit():
    connector = _make_connector(layer_count=8, depth=4)
    _wait(connector, 0)  # window 1..4
    _wait(connector, 1)  # window 2..5; only layer 5 is new
    assert _submitted_layers(connector) == [1, 2, 3, 4, 5]

    _wait(connector, 2)  # window 3..6; only layer 6 is new
    assert _submitted_layers(connector) == [1, 2, 3, 4, 5, 6]


def test_window_clamps_at_end_of_model():
    connector = _make_connector(layer_count=3, depth=4)
    _wait(connector, 0)  # window 1..4 clamped to 1..2
    assert _submitted_layers(connector) == [1, 2]

    _wait(connector, 1)  # window 2..4, everything already submitted
    assert _submitted_layers(connector) == [1, 2]

    _wait(connector, 2)  # last layer: nothing ahead to submit
    assert _submitted_layers(connector) == [1, 2]


def test_repeated_hook_visit_does_not_resubmit():
    connector = _make_connector(layer_count=6, depth=4)
    _wait(connector, 0)
    # MTP / rollback paths revisit the same layer name after its tasks were
    # popped; the early return must not trigger further submissions.
    _wait(connector, 0)
    assert _submitted_layers(connector) == [1, 2, 3, 4]


def test_depth_clamped_to_minimum_one():
    connector = _make_connector(layer_count=6, depth=0)
    _wait(connector, 0)
    assert _submitted_layers(connector) == [1]


def test_load_duration_stats_recorded_on_last_layer_only():
    _reset_fakes()
    connector = _make_connector(layer_count=3, depth=4)
    connector._record_layerwise_load_duration = Mock(return_value=True)
    connector._layerwise_layer_load_duration_sum_ms = 12.5
    connector._layerwise_load_bytes = 4096

    _wait(connector, 0)  # layers still ahead -> no stats yet
    assert fake_ucmmetrics.updated == []

    _wait(connector, 2)  # last layer -> batch stats flushed once
    # 4096 accumulated earlier + layer 2's wait adds 2 blocks x 64B shard size
    assert fake_ucmmetrics.updated == [
        {
            "layerwise_batch_load_duration_sum_ms": 12.5,
            "load_bytes_total": 4352,
        }
    ]
