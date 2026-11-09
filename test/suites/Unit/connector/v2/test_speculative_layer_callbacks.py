"""A repeated layer callback must not resubmit a step's dump plan."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]


def test_repeated_layer_visits_submit_once_per_step():
    from ucm.integration.vllm.v2.connector_worker import UCMConnectorWorker

    worker = object.__new__(UCMConnectorWorker)
    worker.use_layerwise = True
    worker.layout = SimpleNamespace(
        layer_id_by_name={"layer.0.attn": 0, "layer.0.indexer": 0, "layer.1": 1}
    )
    worker._dump_submitted_layer_ids = set()
    worker._failed_load_request_ids = set()
    worker._failed_dump_jobs = set()
    worker._transfer_tasks_by_job = {}
    worker._layer_load_tasks = {}
    worker._dump_tasks = []
    worker._submit_request_dumps = Mock()
    request = SimpleNamespace(prepare_get_failed=False)
    metadata = SimpleNamespace(requests={"r": request})

    worker.begin_step(metadata)
    for name in ("layer.0.attn", "layer.0.indexer", "layer.0.attn", "layer.1"):
        worker.save_kv_layer(metadata, name, None, None)
    assert worker._submit_request_dumps.call_args_list == [
        call(request, 0),
        call(request, 1),
    ]

    worker.begin_step(metadata)
    worker.save_kv_layer(metadata, "layer.0.attn", None, None)
    assert worker._submit_request_dumps.call_args_list[-1] == call(request, 0)
    assert worker._submit_request_dumps.call_count == 3
