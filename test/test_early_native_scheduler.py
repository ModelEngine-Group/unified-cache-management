"""Real vLLM Scheduler/APC/budget with Early matching; storage and H2D mocked.

Run with the locked vLLM source's test helpers on PYTHONPATH. A tiny local
config is created by the fixture; no model weights or network are needed.
"""
from unittest.mock import Mock
import pytest
from transformers import OPTConfig
from tests.v1.core.utils import create_scheduler, create_requests
from test_early_ready_reuse import connector
from ucm.integration.vllm.early_prefetch import Job


@pytest.fixture
def scheduler_factory(tmp_path):
    OPTConfig(architectures=["OPTForCausalLM"], hidden_size=64, ffn_dim=128,
              num_hidden_layers=2, num_attention_heads=2,
              max_position_embeddings=131072, vocab_size=1000).to_json_file(tmp_path / "config.json")
    return lambda: create_scheduler(model=str(tmp_path), max_num_seqs=8,
        max_num_batched_tokens=4096, block_size=128, max_model_len=131072,
        enable_prefix_caching=True, skip_tokenizer_init=True)


def schedule(connector, scheduler_factory, states, *, hit=48, same_prompt=False):
    scheduler = scheduler_factory()
    connector.block_size = connector.hash_block_size = 128
    requests = create_requests(num_requests=2, num_tokens=8192, block_size=128,
                               same_prompt=same_prompt)
    for request, state in zip(requests, states):
        scheduler.add_request(request)
        job = Job(request, request.all_token_ids, connector._early.identity(request))
        job.state, job.hit = state, hit
        job.blocks = [i.to_bytes(16, "big") for i in range(64)]
        connector._early.jobs[request.request_id] = job
        connector._early.by_key[job.key] = job
    adapter = Mock()
    adapter.get_num_new_matched_tokens.side_effect = connector.get_num_new_matched_tokens
    adapter.build_connector_meta.return_value = None
    scheduler.connector = adapter
    output = scheduler.schedule()
    return scheduler, output, adapter


def test_ready_external_hits_only_charge_uncached_tokens(connector, scheduler_factory):
    _, output, _ = schedule(connector, scheduler_factory, ("READY", "READY"))
    # 8192 prompt - 6144 protected external hit = 2048 actual compute each.
    assert output.num_scheduled_tokens == {"0": 2048, "1": 2048}
    assert output.total_num_scheduled_tokens == 4096
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_same_wave_hbm_hit_does_not_wait_for_pending_redundant_prefetch(connector, scheduler_factory):
    scheduler, output, adapter = schedule(connector, scheduler_factory,
        ("READY", "PENDING"), same_prompt=True)
    # Real allocate_slots caches the first request's blocks. Native APC leaves
    # the last block for logits; it can never return 8192 for this prompt.
    assert [call.args[1] for call in adapter.get_num_new_matched_tokens.call_args_list] == [0, 8064]
    assert output.num_scheduled_tokens == {"0": 2048, "1": 128}
    assert output.total_num_scheduled_tokens == 2176
    assert len(scheduler.skipped_waiting) == 0
    connector._early.control.send.assert_called_once()
    assert connector._early.control.send.call_args.args[0]["op"] == "cancel"
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_insufficient_cache_does_not_fake_two_requests_into_budget(connector, scheduler_factory):
    _, output, _ = schedule(connector, scheduler_factory, ("READY", "READY"), hit=32)
    assert output.num_scheduled_tokens == {"0": 4096}
    assert output.total_num_scheduled_tokens == 4096


def test_pending_without_hbm_coverage_still_skips(connector, scheduler_factory):
    scheduler, output, _ = schedule(connector, scheduler_factory, ("READY", "PENDING"))
    assert output.num_scheduled_tokens == {"0": 2048}
    assert len(scheduler.skipped_waiting) == 1
    connector._early.control.send.assert_not_called()
