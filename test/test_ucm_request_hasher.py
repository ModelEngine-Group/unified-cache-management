import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ucm.integration.vllm.request_hasher as request_hasher_module
from ucm.integration.vllm.request_hasher import RequestHasher, RequestHashError


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        speculative_config=None,
        additional_config={},
    )


def _request(token_ids, **kwargs):
    defaults = {
        "mm_features": [],
        "lora_request": None,
        "cache_salt": None,
        "prompt_embeds": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(all_token_ids=list(token_ids), **defaults)


def _mm_feature(identifier, offset, length):
    return SimpleNamespace(
        identifier=identifier,
        mm_position=SimpleNamespace(offset=offset, length=length),
    )


def _semantic_extra_keys(request, start, end, start_mm_idx):
    extras = []
    mm_features = request.mm_features
    curr_mm_idx = start_mm_idx
    while curr_mm_idx < len(mm_features):
        feature = mm_features[curr_mm_idx]
        offset = feature.mm_position.offset
        length = feature.mm_position.length
        if end <= offset:
            break
        if start >= offset + length:
            curr_mm_idx += 1
            continue
        assert feature.identifier is not None
        extras.append((feature.identifier, offset - start))
        if end >= offset + length:
            curr_mm_idx += 1
        else:
            break

    if request.lora_request is not None:
        extras.append(request.lora_request.lora_name)
    if start == 0 and request.cache_salt:
        extras.append(request.cache_salt)
    if request.prompt_embeds is not None:
        extras.append(("prompt_embeds", request.prompt_embeds[start:end]))
    return (tuple(extras) if extras else None), curr_mm_idx


@pytest.fixture(autouse=True)
def _install_extra_key_helper(monkeypatch):
    monkeypatch.setattr(
        request_hasher_module,
        "generate_block_hash_extra_keys",
        _semantic_extra_keys,
    )


def test_multimodal_identifier_changes_covering_and_following_blocks():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(16)
    small_image = _request(
        tokens, mm_features=[_mm_feature("small-image", offset=4, length=5)]
    )
    large_image = _request(
        tokens, mm_features=[_mm_feature("large-image", offset=4, length=5)]
    )

    small_hashes = block_hasher(small_image)
    large_hashes = block_hasher(large_image)

    assert small_hashes[0] == large_hashes[0]
    assert small_hashes[1:] != large_hashes[1:]
    assert all(a != b for a, b in zip(small_hashes[1:], large_hashes[1:]))


def test_multimodal_position_is_part_of_block_semantics():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(12)

    at_boundary = _request(
        tokens, mm_features=[_mm_feature("same-image", offset=4, length=2)]
    )
    inside_block = _request(
        tokens, mm_features=[_mm_feature("same-image", offset=5, length=2)]
    )

    boundary_hashes = block_hasher(at_boundary)
    inside_hashes = block_hasher(inside_block)
    assert boundary_hashes[0] == inside_hashes[0]
    assert boundary_hashes[1:] != inside_hashes[1:]


@pytest.mark.parametrize(
    "left,right",
    [
        (
            {"lora_request": SimpleNamespace(lora_name="lora-a")},
            {"lora_request": SimpleNamespace(lora_name="lora-b")},
        ),
        ({"cache_salt": "tenant-a"}, {"cache_salt": "tenant-b"}),
        ({"prompt_embeds": b"abcdefgh"}, {"prompt_embeds": b"ABCDEFGH"}),
    ],
)
def test_other_request_semantics_change_hashes(left, right):
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(8)
    assert block_hasher(_request(tokens, **left)) != block_hasher(
        _request(tokens, **right)
    )


def test_closure_resets_parent_and_multimodal_cursor_for_each_request():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(
        range(12), mm_features=[_mm_feature("image", offset=3, length=6)]
    )

    assert block_hasher(request) == block_hasher(request)


def test_partial_tail_is_not_hashed_and_new_tuple_breaks_old_keys():
    hasher = RequestHasher(_config(), 0)
    block_hasher = hasher.make_request_block_hasher(4)
    request = _request(range(6))

    hashes = block_hasher(request)
    legacy_first_hash = hasher((hasher.seed, tuple(range(4))))

    assert len(hashes) == 1
    assert hashes[0] != legacy_first_hash


def test_helper_unavailable_allows_only_token_only_requests(monkeypatch):
    monkeypatch.setattr(request_hasher_module, "generate_block_hash_extra_keys", None)
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)

    assert len(block_hasher(_request(range(4)))) == 1
    with pytest.raises(RequestHashError):
        block_hasher(
            _request(
                range(2),
                mm_features=[_mm_feature("image", offset=0, length=2)],
            )
        )
    with pytest.raises(RequestHashError):
        block_hasher(
            _request(range(2), lora_request=SimpleNamespace(lora_name="adapter"))
        )


def test_missing_multimodal_identifier_fails_closed():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(range(4), mm_features=[_mm_feature(None, offset=0, length=4)])

    with pytest.raises(RequestHashError):
        block_hasher(request)


def test_unhashed_tail_does_not_prevalidate_multimodal_identifier():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(range(2), mm_features=[_mm_feature(None, offset=0, length=2)])

    assert block_hasher(request) == []


def test_hash_is_independent_of_pythonhashseed():
    module_path = request_hasher_module.__file__
    script = f"""
import importlib.util
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "request_hasher_seed_test", {module_path!r}
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = SimpleNamespace(
    model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
    parallel_config=SimpleNamespace(tensor_parallel_size=2),
    speculative_config=None,
    additional_config={{}},
)
hasher = module.RequestHasher(config, 0)
print(hasher((b"parent", (1, 2, 3), None)).hex())
"""

    outputs = []
    for seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        # Suppress vllm platform-plugin loading logs so stdout stays clean.
        env["VLLM_PLUGINS"] = ""
        stdout = subprocess.check_output(
            [sys.executable, "-c", script], env=env, text=True
        )
        # Importing the module may still emit log lines on stdout; keep only
        # the printed md5 hexdigest (32 hex chars).
        hex_lines = [
            line.strip()
            for line in stdout.splitlines()
            if len(line.strip()) == 32
            and all(c in "0123456789abcdef" for c in line.strip())
        ]
        outputs.append(hex_lines[-1])

    assert outputs[0] == outputs[1]


def _shared_prefix_len(hashes_a, hashes_b):
    """Number of leading block hashes two request hash chains share."""
    limit = min(len(hashes_a), len(hashes_b))
    for i in range(limit):
        if hashes_a[i] != hashes_b[i]:
            return i
    return limit


def test_different_image_shares_only_text_prefix_before_image():
    """Core anti-false-hit property: same text prefix + different image must
    hit only up to the last text block before the image, then diverge."""
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(16)
    req_a = _request(tokens, mm_features=[_mm_feature("img-a", offset=4, length=4)])
    req_b = _request(tokens, mm_features=[_mm_feature("img-b", offset=4, length=4)])

    hashes_a = block_hasher(req_a)
    hashes_b = block_hasher(req_b)

    assert _shared_prefix_len(hashes_a, hashes_b) == 1
    assert all(a != b for a, b in zip(hashes_a[1:], hashes_b[1:]))


def test_identical_request_shares_full_prefix():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(range(16), mm_features=[_mm_feature("img", offset=4, length=4)])
    hashes = block_hasher(request)
    assert _shared_prefix_len(hashes, block_hasher(request)) == len(hashes)


def test_image_position_change_moves_hit_boundary():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(16)
    early = _request(tokens, mm_features=[_mm_feature("img", offset=4, length=4)])
    late = _request(tokens, mm_features=[_mm_feature("img", offset=8, length=4)])
    assert _shared_prefix_len(block_hasher(early), block_hasher(late)) == 1


def test_multimodal_cursor_advances_monotonically_across_blocks(monkeypatch):
    """White-box: the mm cursor must advance per block so image 2 lands on
    image 2's block, not image 1's."""
    calls = []

    def recording_helper(request, start, end, start_mm_idx):
        result, next_idx = _semantic_extra_keys(request, start, end, start_mm_idx)
        calls.append((start, start_mm_idx, next_idx))
        return result, next_idx

    monkeypatch.setattr(
        request_hasher_module, "generate_block_hash_extra_keys", recording_helper
    )
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(
        range(16),
        mm_features=[
            _mm_feature("img-1", offset=0, length=4),
            _mm_feature("img-2", offset=8, length=4),
        ],
    )
    block_hasher(request)

    assert [start for start, _, _ in calls] == [0, 4, 8, 12]
    assert [enter for _, enter, _ in calls] == [0, 1, 1, 2]
    assert [leave for _, _, leave in calls] == [1, 1, 2, 2]


def test_second_image_change_diverges_only_at_second_image_block():
    """Changing only the second image keeps image-1 blocks reusable and
    diverges from image 2's block onward."""
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(16)
    both = _request(
        tokens,
        mm_features=[
            _mm_feature("img-1", offset=0, length=4),
            _mm_feature("img-2", offset=8, length=4),
        ],
    )
    second_changed = _request(
        tokens,
        mm_features=[
            _mm_feature("img-1", offset=0, length=4),
            _mm_feature("img-2-b", offset=8, length=4),
        ],
    )
    hashes_a = block_hasher(both)
    hashes_b = block_hasher(second_changed)
    assert _shared_prefix_len(hashes_a, hashes_b) == 2
    assert all(a != b for a, b in zip(hashes_a[2:], hashes_b[2:]))


def test_pure_image_requests_diverge_from_block_zero():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    tokens = range(16)
    req_a = _request(tokens, mm_features=[_mm_feature("img-a", offset=0, length=16)])
    req_b = _request(tokens, mm_features=[_mm_feature("img-b", offset=0, length=16)])
    assert _shared_prefix_len(block_hasher(req_a), block_hasher(req_b)) == 0


def test_identical_pure_image_request_shares_all_blocks():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(range(16), mm_features=[_mm_feature("img", offset=0, length=16)])
    hashes = block_hasher(request)
    assert _shared_prefix_len(hashes, block_hasher(request)) == len(hashes)


def test_pure_image_partial_tail_not_hashed():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    request = _request(range(10), mm_features=[_mm_feature("img", offset=0, length=10)])
    assert len(block_hasher(request)) == 2


def test_empty_token_list_returns_empty():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(4)
    assert block_hasher(_request([])) == []


def test_block_size_larger_than_sequence_returns_empty():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(100)
    assert block_hasher(_request(range(4))) == []


def test_block_size_one_hashes_every_token():
    block_hasher = RequestHasher(_config(), 0).make_request_block_hasher(1)
    base = _request(range(8))
    changed = _request(range(8))
    changed_tokens = list(changed.all_token_ids)
    changed_tokens[3] = 999
    changed.all_token_ids = changed_tokens

    base_hashes = block_hasher(base)
    changed_hashes = block_hasher(changed)
    assert len(base_hashes) == 8
    assert _shared_prefix_len(base_hashes, changed_hashes) == 3
    assert all(a != b for a, b in zip(base_hashes[3:], changed_hashes[3:]))


@pytest.mark.parametrize(
    "right_config,expect_equal",
    [
        (lambda: _config(), True),
        (
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
                parallel_config=SimpleNamespace(tensor_parallel_size=3),
                speculative_config=None,
                additional_config={},
            ),
            False,
        ),
        (
            lambda: SimpleNamespace(
                model_config=SimpleNamespace(model="org/other", dtype="bfloat16"),
                parallel_config=SimpleNamespace(tensor_parallel_size=2),
                speculative_config=None,
                additional_config={},
            ),
            False,
        ),
    ],
)
def test_namespace_isolation_and_reproducibility_across_instances(
    right_config, expect_equal
):
    """Same deployment config across two hasher instances yields identical
    block IDs; rank/parallelism/model differences do not."""
    left = RequestHasher(_config(), 0).make_request_block_hasher(4)
    right = RequestHasher(right_config(), 0).make_request_block_hasher(4)
    request = _request(range(16))
    result = left(request) == right(request)
    assert result is expect_equal
