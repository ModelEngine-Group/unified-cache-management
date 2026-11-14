"""Exercise the production v2 roles against an explicitly configured Store."""

from bisect import bisect_right
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from ucm.integration.vllm.v2.store_client import (
    PreparedJob,
    StoreIOError,
    TransferBatch,
)


def read_published_record(role, io, key, spans, kv_buffers, kv_device):
    import torch

    # A fresh Store request can only find the key after CompletePut. Reading
    # the entire record at offset zero also checks the writer's Store offsets.
    source = role._store_scheduler
    request_id = uuid4().hex
    dest = torch.full(
        (max(offset + size for offset, _, size in spans),),
        0xA5,
        dtype=torch.uint8,
        device=kv_device,
    )
    if kv_device.type != "cpu":
        getattr(torch, kv_device.type).current_stream(kv_device).synchronize()
    keys = (key,)
    batch = TransferBatch(
        keys,
        np.array([[dest.data_ptr()]], dtype=np.uint64),
        np.array([[dest.numel()]], dtype=np.uint64),
        np.zeros((1, 1), dtype=np.uint64),
    )
    source.on_new_request(request_id)
    token = None
    task = None
    try:
        hit_index, _ = source.lookup_on_prefix(request_id, keys)
        if hit_index != 0:
            raise AssertionError(f"Published record {key.hex()} did not hit")
        token = source.prepare_get(request_id, keys)
        job = PreparedJob(role._allocate_job_id(), keys, token)
        task = io.submit_get(job, (batch,))
        io.wait(task)
        actual = dest.cpu().numpy()
        for offset, ptr, size in spans:
            index = bisect_right(kv_buffers, ptr, key=lambda buf: buf.data_ptr()) - 1
            if index < 0:
                raise AssertionError(f"Unknown KV pointer {ptr:#x}")
            raw = kv_buffers[index]
            start = ptr - raw.data_ptr()
            if start + size > raw.numel():
                raise AssertionError(
                    f"KV range at {ptr:#x} exceeds its backing storage"
                )
            expected = raw[start : start + size].cpu().numpy()
            if not np.array_equal(actual[offset : offset + size], expected):
                raise AssertionError(
                    f"Store bytes differ at offset {offset} of {key.hex()}"
                )
    finally:
        # An exception with unknown IO status must not unpin the record. The
        # caller's ExitStack drains the Worker with this frame on the traceback.
        if task is None or task.success is not None:
            if token is not None:
                source.complete_get(request_id, token)
            source.on_request_finished(request_id)


def execute_store_step(engine, scheduler, worker, output, model_output, views):
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.outputs import KVConnectorOutput

    metadata = output.kv_connector_metadata
    worker.bind_connector_metadata(metadata)
    with set_forward_context(
        None, engine, num_tokens=output.total_num_scheduled_tokens
    ):
        worker.start_load_kv(get_forward_context())
        for name, view in views.items():
            worker.wait_for_layer_load(name)
            worker.save_kv_layer(name, view, None)
        worker.wait_for_save()
    worker.get_transfer_results(set())
    result = worker.build_connector_worker_meta()
    worker.clear_connector_metadata()
    model_output.kv_connector_output = KVConnectorOutput(
        kv_connector_worker_meta=result
    )
    scheduler.update_from_output(output, model_output)
    return result


def finish_store_work(engine, scheduler, worker):
    from vllm.v1.outputs import make_empty_encoder_model_runner_output

    role = worker.connector_worker
    while scheduler.has_requests():
        output = scheduler.schedule()
        execute_store_step(
            engine,
            scheduler,
            worker,
            output,
            make_empty_encoder_model_runner_output(output),
            {},
        )
        # The check can wait explicitly for Store-only jobs before the
        # next empty step gathers their receipts.
        for task in role._prefetch_offload_tasks.values():
            try:
                role._store_worker.wait(task)
            except StoreIOError:
                # The terminal failure still needs CompleteOffload.
                pass


def dump_and_check(
    engine,
    scheduler,
    worker,
    output,
    model_output,
    request_id,
    views,
    records,
    kv_buffers,
    kv_device,
):
    metadata = output.kv_connector_metadata
    req = metadata.requests[request_id]
    if req.load_plans or req.prepare_get_failed or req.prefetch_job:
        raise AssertionError("Expected a cold request with only Dump plans")
    expected_jobs = set()
    keys = {}
    for plan in req.dump_plans:
        if plan.prepared_job is None:
            raise AssertionError(f"PreparePut failed for {plan.restore_policy.name}")
        expected_jobs.add(plan.prepared_job.job_id)
        for start, key in zip(plan.token_starts, plan.ucm_keys, strict=True):
            window = plan.restore_policy, int(start)
            if window in keys:
                raise AssertionError(f"Duplicate Dump window {window}")
            keys[window] = key
    if keys.keys() != records.keys() or len(set(keys.values())) != len(records):
        raise AssertionError(
            "Scheduler Dump keys do not cover all expected windows exactly once"
        )

    io = worker.connector_worker._store_worker
    submitted_tasks = []
    submit_put = io.submit_put

    def record_put(*args, **kwargs):
        task = submit_put(*args, **kwargs)
        submitted_tasks.append(task)
        return task

    with patch.object(io, "submit_put", record_put):
        result = execute_store_step(
            engine, scheduler, worker, output, model_output, views
        )
    results = (
        tuple(job for job in result.job_results if job.job_id in expected_jobs)
        if result
        else ()
    )
    if {job.job_id for job in results} != expected_jobs or not all(
        job.success for job in results
    ):
        raise AssertionError("Worker did not report all prepared Dump jobs successful")
    dump_tasks = [task for task in submitted_tasks if task.job_id in expected_jobs]
    if {task.job_id for task in dump_tasks} != expected_jobs or not all(
        task.success for task in dump_tasks
    ):
        raise AssertionError("A Dump job had no successful native SubmitPut")
    role = scheduler.connector.connector_scheduler
    for window, key in keys.items():
        read_published_record(
            role,
            io,
            key,
            records[window],
            kv_buffers,
            kv_device,
        )
