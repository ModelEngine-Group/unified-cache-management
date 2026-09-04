from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from ucm.integration.vllm.hla_connector import GroupInfo, KVCacheGroupManager
from ucm.integration.vllm.request_hasher import RequestHasher, RequestHashError


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
        additional_config={},
    )


def _request(identifier):
    return SimpleNamespace(
        all_token_ids=list(range(16)),
        mm_features=[
            SimpleNamespace(
                identifier=identifier,
                mm_position=SimpleNamespace(offset=4, length=5),
            )
        ],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )


def _manager_with_groups(full_attention_block_sizes):
    hasher = RequestHasher(_config(), 0)
    manager = KVCacheGroupManager.__new__(KVCacheGroupManager)
    manager.request_hasher = hasher
    manager.groups_by_id = []
    manager.full_attn_groups = []
    manager.state_groups = []

    for group_id, block_size in enumerate(full_attention_block_sizes):
        seed = hasher((b"UCM_GROUP_SEED", hasher.seed, group_id))
        group = GroupInfo(
            group_id=group_id,
            block_size=block_size,
            layer_names=(f"fa.{group_id}",),
            seed=seed,
            block_hasher=hasher.make_request_block_hasher(block_size, seed),
        )
        manager.groups_by_id.append(group)
        manager.full_attn_groups.append(group)

    state_id = len(manager.groups_by_id)
    state_seed = hasher((b"UCM_GROUP_SEED", hasher.seed, state_id))
    state_group = GroupInfo(
        group_id=state_id,
        block_size=8,
        layer_names=("mamba.0",),
        seed=state_seed,
        is_mamba_align=True,
    )
    manager.groups_by_id.append(state_group)
    manager.state_groups.append(state_group)
    manager.lcm_block_size = 8
    return manager, state_group


def test_full_attention_groups_hash_request_at_their_own_boundaries():
    manager, _ = _manager_with_groups([4, 8])

    small = manager.compute_all_group_block_ids(_request("small-image"))
    large = manager.compute_all_group_block_ids(_request("large-image"))

    assert len(small[0]) == 4
    assert len(small[1]) == 2
    assert small[0][0] == large[0][0]
    assert small[0][1:] != large[0][1:]
    assert small[1] != large[1]


def test_mamba_state_key_inherits_primary_attention_request_semantics():
    manager, state_group = _manager_with_groups([4])
    small_blocks = manager.compute_all_group_block_ids(_request("small-image"))
    large_blocks = manager.compute_all_group_block_ids(_request("large-image"))

    small_state = manager.compute_mamba_align_state_hash(state_group, 8, small_blocks)
    large_state = manager.compute_mamba_align_state_hash(state_group, 8, large_blocks)

    assert small_state is not None
    assert large_state is not None
    assert small_state != large_state


def test_hash_failure_in_any_group_disables_whole_lookup_without_partial_keys():
    """Design §10: an HLA hash failure in any group must fail the whole
    external lookup; a partial set of group keys is never returned."""
    manager, _ = _manager_with_groups([4, 8])
    request = SimpleNamespace(all_token_ids=list(range(16)))

    attempted = {"n": 0}

    def good_hasher(req):
        attempted["n"] += 1
        return [b"\x00"] * (len(req.all_token_ids) // 4)

    def failing_hasher(_req):
        raise RequestHashError("simulated group hash failure")

    manager.groups_by_id[0].block_hasher = good_hasher
    manager.groups_by_id[1].block_hasher = failing_hasher

    with pytest.raises(RequestHashError):
        manager.compute_all_group_block_ids(request)
    assert attempted["n"] == 1
