"""Exercise the production v2 roles against an explicitly configured host Store."""

import ctypes
from uuid import uuid4

import numpy as np

from ucm.integration.vllm.v2.store_client import PreparedJob, TransferBatch


def read_published_record(role, io, key, spans):
    # A fresh Store request can only find the key after CompletePut. Reading
    # the entire record at offset zero also checks the writer's Store offsets.
    source = role._store_scheduler
    request_id = uuid4().hex
    dest = np.full(sum(size for _, _, size in spans), 0xA5, dtype=np.uint8)
    keys = (key,)
    batch = TransferBatch(
        keys,
        np.array([[dest.ctypes.data]], dtype=np.uint64),
        np.array([[dest.size]], dtype=np.uint64),
        np.zeros((1, 1), dtype=np.uint64),
    )
    source.on_new_request(request_id)
    prepared = False
    task = None
    try:
        if source.lookup_on_prefix(request_id, keys) != 0:
            raise AssertionError(f"Published record {key.hex()} did not hit")
        job = PreparedJob(
            role._allocate_job_id(), keys, source.prepare_get(request_id, keys)
        )
        prepared = True
        task = io.submit_get(job, (batch,))
        io.wait(task)
        for offset, ptr, size in spans:
            if dest[offset : offset + size].tobytes() != ctypes.string_at(ptr, size):
                raise AssertionError(
                    f"Store bytes differ at offset {offset} of {key.hex()}"
                )
    finally:
        # An exception with unknown IO status must not unpin the record. The
        # caller's ExitStack drains the Worker with this frame on the traceback.
        if task is None or task.success is not None:
            if prepared:
                source.complete_get(request_id)
            source.on_request_finished(request_id)


def dump_and_check(
    engine, scheduler, worker, output, model_output, request_id, views, records
):
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.outputs import KVConnectorOutput

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
        for start, key in zip(
            plan.token_starts, plan.ucm_keys_by_pp_rank[0], strict=True
        ):
            window = plan.restore_policy, int(start)
            if window in keys:
                raise AssertionError(f"Duplicate Dump window {window}")
            keys[window] = key
    if keys.keys() != records.keys() or len(set(keys.values())) != len(records):
        raise AssertionError(
            "Scheduler Dump keys do not cover all expected windows exactly once"
        )

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
    results = result.job_results if result else ()
    if {job.job_id for job in results} != expected_jobs or not all(
        job.success for job in results
    ):
        raise AssertionError("Worker did not report all prepared Dump jobs successful")
    tasks = worker.connector_worker._transfer_tasks_by_job
    if any(
        not tasks.get(job) or not all(t.success for t in tasks[job])
        for job in expected_jobs
    ):
        raise AssertionError("A Dump job had no successful native SubmitPut")
    role = scheduler.connector.connector_scheduler
    for window, key in keys.items():
        read_published_record(
            role, worker.connector_worker._store_worker, key, records[window]
        )
