from contextlib import contextmanager
from typing import Generator

from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

# class KVConnectorModelRunnerMixin:


@staticmethod
@contextmanager
def _get_kv_connector_output(
    scheduler_output: "SchedulerOutput",
    wait_for_save: bool = True,
) -> Generator["KVConnectorOutput", None, None]:
    output = KVConnectorOutput()

    # Update KVConnector with the KVConnector metadata forward().
    kv_connector = get_kv_transfer_group()
    assert isinstance(kv_connector, KVConnectorBase)
    assert scheduler_output.kv_connector_metadata is not None
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # Background KV cache transfers happen here.
    # These transfers are designed to be async and the requests
    # involved may be disjoint from the running requests.
    # Do this here to save a collective_rpc.
    kv_connector.start_load_kv(get_forward_context())
    try:
        yield output
    finally:
        if wait_for_save:
            kv_connector.wait_for_save()

        output.finished_sending, output.finished_recving = kv_connector.get_finished(
            scheduler_output.finished_req_ids
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        # if output.invalid_block_ids:
        #    logger.warning(
        #        f"[kv-load] model-runner sees invalid_block_ids={len(output.invalid_block_ids)} "
        #        f"sample={list(output.invalid_block_ids)[:10]}"
        #    )

        output.kv_connector_stats = KVConnectorModelRunnerMixin.get_kv_connector_stats()
        kv_connector.clear_connector_metadata()
