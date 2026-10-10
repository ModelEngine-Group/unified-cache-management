from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("triton")

from ucm.integration.vllm.blend_connector import UCMBlendConnector
from ucm.integration.vllm.request_hasher import RequestHasher, RequestHashError


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
        additional_config={},
    )


def _plain_text_request():
    return SimpleNamespace(
        all_token_ids=list(range(8)),
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )


def _connector_for_hash_test():
    connector = UCMBlendConnector.__new__(UCMBlendConnector)
    connector.block_size = 4
    connector.request_hasher = RequestHasher(_config(), 0)
    connector._seed = connector.request_hasher.seed
    connector.request_block_hasher = connector.request_hasher.make_request_block_hasher(
        connector.block_size, connector._seed
    )
    return connector


def test_chunk_build_and_lookup_use_the_same_plain_text_hash_schema():
    connector = _connector_for_hash_test()
    request = _plain_text_request()

    prefix_hashes = connector.request_block_hasher(request)
    chunk_hashes = connector._generate_chunk_hashes(request.all_token_ids)

    assert chunk_hashes == prefix_hashes


@pytest.mark.parametrize(
    "field,value",
    [
        ("mm_features", [SimpleNamespace(identifier="image")]),
        ("lora_request", SimpleNamespace(lora_name="adapter")),
        ("cache_salt", "tenant"),
        ("prompt_embeds", object()),
    ],
)
def test_cacheblend_rejects_request_semantics_it_cannot_hash(field, value):
    request = _plain_text_request()
    setattr(request, field, value)

    with pytest.raises(RequestHashError):
        UCMBlendConnector._validate_supported_request(request)


def test_get_num_new_matched_tokens_returns_zero_on_unsupported_request():
    """Connector-level fail-closed (design §10): an unhashable request returns
    zero external hit tokens instead of raising through the scheduler."""
    connector = _connector_for_hash_test()
    request = SimpleNamespace(
        all_token_ids=list(range(8)),
        mm_features=[
            SimpleNamespace(
                identifier="image",
                mm_position=SimpleNamespace(offset=0, length=4),
            )
        ],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
        request_id="req-mm-1",
    )

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
