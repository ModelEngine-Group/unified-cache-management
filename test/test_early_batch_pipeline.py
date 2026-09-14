"""Real worker threads/control callbacks, mocked storage; no SSD/NPU claims."""
import importlib
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

package = ModuleType("batch_pipeline_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
early = importlib.import_module(package.__name__ + ".early_prefetch")


def wait(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "bounded worker test wait expired"
        time.sleep(.002)


class Control:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def send(self, message):
        self.messages.append(message)

    def close(self):
        pass


@pytest.fixture
def worker():
    store = Mock()
    connector = SimpleNamespace(store=store, tp_rank=0, tp_size=8, is_mla=True,
        unique_id="batch-unit", request_hasher=Mock(side_effect=lambda block: bytes(reversed(block))),
        _vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=0)))
    instance = early.EarlyWorker(connector, Control)
    yield instance
    instance.close()


def blocks(count):
    return [i.to_bytes(16, "little") for i in range(count)]


def result(handle, hit, work):
    return dict(handle=handle, total_hit_blocks=hit, remaining_l2_work=work)


def test_first_batch_reads_before_next_reserve_finishes(worker):
    reading = threading.Event()
    second_entered = threading.Event()
    allow_second = threading.Event()
    calls = []

    def reserve(ids):
        calls.append(ids)
        if len(calls) == 2:
            second_entered.set()
            assert reading.wait(2), "reserve still precedes all reads"
            assert allow_second.wait(2)
        return result(len(calls), len(ids), len(ids))

    worker.store.reserve_prefix.side_effect = reserve
    worker.store.prefetch_reserved.side_effect = lambda handle: (reading.set(), {"l2_read_shards": 1})[1]
    worker.receive(dict(op="reserve", key="overlap", request="r", blocks=blocks(96)))
    try:
        assert second_entered.wait(1)
        assert reading.wait(1)
        assert not worker.jobs["overlap"].ready  # no premature whole-request READY
        assert not worker.control.messages
    finally:
        allow_second.set()
    wait(lambda: worker.jobs["overlap"].ready)
    assert [len(ids) for ids in calls] == [32, 64]
    assert worker.control.messages[-1]["hit"] == 96
    worker.bind(SimpleNamespace(early_keys={"r": "overlap"}, request_meta={
        "r": SimpleNamespace(load_block_ids=([b"a" * 16], []))}))
    worker.store.set_load_reservations.assert_called_with([1, 2])
    worker.loaded({"r"})
    assert [call.args[0] for call in worker.store.release_reservation.call_args_list] == [1, 2]


def test_all_l1_and_short_prefix_never_use_data_queue(worker):
    worker.store.reserve_prefix.side_effect = [result(1, 32, 0), result(2, 7, 0)]
    worker.receive(dict(op="reserve", key="short", blocks=blocks(512)))
    wait(lambda: worker.jobs["short"].ready)
    assert worker.control.messages[-1]["hit"] == 39
    assert worker.store.reserve_prefix.call_count == 2
    worker.store.prefetch_reserved.assert_not_called()
    assert not worker.preparing and not worker.selected


def test_late_reservation_after_cancel_is_released_without_more_candidates(worker):
    entered, finish = threading.Event(), threading.Event()

    def reserve(ids):
        entered.set()
        assert finish.wait(2)
        return result(7, len(ids), len(ids))

    worker.store.reserve_prefix.side_effect = reserve
    worker.receive(dict(op="reserve", key="cancel", blocks=blocks(512)))
    try:
        assert entered.wait(1)
        worker.receive(dict(op="release", key="cancel"))
    finally:
        finish.set()
    wait(lambda: worker.store.release_reservation.called)
    worker.store.release_reservation.assert_called_once_with(7)
    worker.store.prefetch_reserved.assert_not_called()
    assert not worker.control.messages


def test_later_lookup_failure_releases_earlier_batches_and_falls_back(worker):
    worker.store.reserve_prefix.side_effect = [result(1, 32, 0), RuntimeError("L2 failed")]
    worker.receive(dict(op="reserve", key="failed", blocks=blocks(96)))
    wait(lambda: worker.control.messages)
    assert worker.control.messages[-1]["state"] == "FAILED"
    worker.store.release_reservation.assert_called_once_with(1)
    assert not worker.jobs["failed"].ready


@pytest.mark.parametrize("mla, rank", [(True, 7), (False, 0), (False, 7)])
def test_each_block_identity_is_computed_once_across_batches(worker, mla, rank):
    worker.connector.is_mla, worker.connector.tp_rank = mla, rank
    handles = iter(range(1, 10))
    worker.store.reserve_prefix.side_effect = lambda ids: result(next(handles), len(ids), 0)
    message = dict(op="reserve", key="once", blocks=blocks(96))
    worker.receive(message)
    worker.receive(message)
    wait(lambda: worker.jobs["once"].ready)
    assert worker.store.reserve_prefix.call_count == 2
    assert worker.connector.request_hasher.call_count == (96 if not mla and rank else 0)
    submitted = [block for call in worker.store.reserve_prefix.call_args_list for block in call.args[0]]
    assert submitted == ([bytes(reversed(block)) for block in message["blocks"]]
                         if not mla and rank else message["blocks"])


def test_hrrn_service_estimate_is_remaining_shards_not_total_prefix():
    long = early.WorkerBatch(0, 0, result(1, 512, 1), entered=0)
    short = early.WorkerBatch(0, 0, result(2, 2, 4), entered=0)
    assert early.EarlyWorker.ratio(long, 10) == 11
    assert early.EarlyWorker.ratio(long, 10) > early.EarlyWorker.ratio(short, 10)
