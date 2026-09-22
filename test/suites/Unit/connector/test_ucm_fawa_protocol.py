"""Run in the vLLM environment; tensor tests use CPU, not NPU kernels."""

import io
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from ucm.integration.vllm import ucm_connector
from ucm.integration.vllm import hma_connector
from ucm.integration.vllm.hma_connector import KVCacheGroupLayout, UCMFAWAConnector


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("padding", [0, 256])
def test_full_alias_layout_addresses_and_restore(block_size, padding):
    page_stride = block_size * 132 + padding
    raw = torch.arange(page_stride * 3, dtype=torch.int64).to(torch.uint8)
    key = torch.as_strided(raw, (3, block_size, 1, 128), (page_stride, 128, 128, 1))
    scale = torch.as_strided(
        raw.view(torch.float32),
        (3, block_size, 1, 1),
        (page_stride // 4, 1, 1, 1),
        storage_offset=block_size * 32,
    )
    full = torch.as_strided(raw, (3, block_size, 1, 132), (page_stride, 132, 132, 1))
    layout = KVCacheGroupLayout(
        {"model.layers.0.indexer": (key, scale, full)},
        is_ascend_layout=True,
        expected_block_size=block_size,
    )
    assert layout.base_ptrs.tolist() == [full.data_ptr()]
    assert layout.segment_tensor_size_list(block_size * 4, block_size * 4) == [
        block_size * 132
    ]
    assert layout.extract_addrs(np.array([2, 0])).tolist() == [
        [full.data_ptr() + 2 * page_stride],
        [full.data_ptr()],
    ]
    saved = full[2].clone()
    saved_key, saved_scale = key[2].clone(), scale[2].view(torch.uint8).clone()
    full[2].zero_()
    full[2].copy_(saved)
    assert torch.equal(key[2], saved_key)
    assert torch.equal(scale[2].view(torch.uint8), saved_scale)
    with pytest.raises(ValueError, match="complete-page"):
        layout.segment_tensor_size_list(block_size * 2, block_size * 4)


@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_hma_base_block_size_old_and_new(block_size, monkeypatch):
    for logical_mode, spec in enumerate(
        (
            SimpleNamespace(block_size=block_size, compress_ratio=4),
            SimpleNamespace(
                block_size=block_size * 4,
                storage_block_size=block_size,
                compress_ratio=4,
            ),
        )
    ):
        from ucm.integration.vllm.fawa_layout import ascend_block_geometry

        monkeypatch.setattr(
            hma_connector,
            "ascend_block_geometry",
            lambda spec: ascend_block_geometry(
                spec, block_size_is_logical=bool(logical_mode)
            ),
        )
        cfg = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
        assert UCMFAWAConnector._get_ascend_base_block_size(cfg) == block_size


def test_c128_partial_page_offsets_remain_supported():
    raw = torch.empty(3 * (128 * 640 + 256), dtype=torch.uint8)
    stride = 128 * 640 + 256
    tensor = torch.as_strided(raw, (3, 128, 1, 640), (stride, 640, 640, 1))
    layout = KVCacheGroupLayout(
        {"model.layers.3.attention": tensor},
        is_ascend_layout=True,
        expected_block_size=128,
    )
    assert layout.segment_tensor_size_list(512, 16384) == [4 * 640]
    ptrs = layout.extract_addrs_with_offsets(
        np.array([2, 0, 1]),
        16384,
        np.array([0, 512, 15872]),
    )
    assert ptrs.tolist() == [
        [tensor.data_ptr() + 2 * stride],
        [tensor.data_ptr() + 4 * 640],
        [tensor.data_ptr() + stride + 124 * 640],
    ]


def test_fawa_scheduler_uses_worker_sizes_and_rejects_missing_publication(monkeypatch):
    sizes = {}
    monkeypatch.setattr(
        hma_connector,
        "_worker_publish_block_size",
        lambda size, dp, store_suffix: sizes.__setitem__(store_suffix, size),
    )
    monkeypatch.setattr(
        hma_connector,
        "_scheduler_read_block_size",
        lambda store_suffix: sizes.get(store_suffix),
    )
    monkeypatch.setattr(
        hma_connector.UcmConnectorFactoryV1,
        "create_connector",
        lambda name, config, module_path: config,
    )
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=0)
    )
    connector.unique_id = "engine"
    connector.file_size = {}
    connector.device_id = 0
    connector.is_mla = True
    connector.tp_size = 2
    connector._base_store_config = lambda suffix: ("test", None, {})
    connector._set_default_shm_buffer_capacity = lambda config: None
    connector._role = ucm_connector.KVConnectorRole.WORKER
    fa = connector._create_store("FA", "fa", [640, 132])
    wa = connector._create_store("WA", "wa", [8192, 1])
    assert (fa["block_size"], wa["block_size"]) == (4096, 12288)
    connector._role = ucm_connector.KVConnectorRole.SCHEDULER
    assert connector._create_store("FA", "fa", None)["block_size"] == 4096
    assert connector._create_store("WA", "wa", None)["block_size"] == 12288
    sizes.clear()
    with pytest.raises(RuntimeError, match="worker-published"):
        connector._create_store("FA", "fa", None)


def test_worker_record_sizes_are_isolated_by_store_and_parent_pid(monkeypatch):
    files = {}

    class Writer(io.StringIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            if not self.closed:
                files[self.path] = self.getvalue()
            super().close()

    def open_file(path, mode="r"):
        if mode == "w":
            return Writer(path)
        if path not in files:
            raise FileNotFoundError(path)
        return io.StringIO(files[path])

    monkeypatch.setattr(ucm_connector, "open", open_file, raising=False)
    monkeypatch.setattr(ucm_connector.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(ucm_connector.os, "getpid", lambda: 101)
    monkeypatch.setattr(ucm_connector.os, "getppid", lambda: 100)
    monkeypatch.setattr(
        ucm_connector.os,
        "replace",
        lambda src, dst: files.__setitem__(dst, files.pop(src)),
    )
    monkeypatch.setattr(
        ucm_connector, "get_world_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    for store_suffix, size in (
        ("", 4096),
        ("fa", 8192),
        ("wa", 12288),
    ):
        ucm_connector._worker_publish_block_size(size, 0, store_suffix)
    for store_suffix, size in (
        ("", 4096),
        ("fa", 8192),
        ("wa", 12288),
    ):
        assert ucm_connector._scheduler_read_block_size(store_suffix) == size
    assert ucm_connector._scheduler_read_block_size("unknown") is None
    ucm_connector._worker_publish_block_size(32768, 1, "fa")
    assert ucm_connector._scheduler_read_block_size("fa") == 8192
    assert set(files) == {
        "/dev/shm/ucm_blocksize_100",
        "/dev/shm/ucm_blocksize_100_fa",
        "/dev/shm/ucm_blocksize_100_wa",
    }
    monkeypatch.setattr(ucm_connector.os, "getppid", lambda: 200)
    assert ucm_connector._scheduler_read_block_size("fa") is None
    ucm_connector._worker_publish_block_size(16384, 0, "fa")
    assert ucm_connector._scheduler_read_block_size("fa") == 16384
    monkeypatch.setattr(ucm_connector.os, "getppid", lambda: 100)
    assert ucm_connector._scheduler_read_block_size("fa") == 8192


def test_fawa_uses_common_key_format_with_custom_hash_size():

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(model="org/model", dtype="bfloat16"),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        speculative_config=None,
        additional_config={},
    )
    connector = UCMFAWAConnector.__new__(UCMFAWAConnector)
    connector._kv_cache_config = None
    connector._role = ucm_connector.KVConnectorRole.SCHEDULER
    connector.hash_block_size = 4
    connector.request_hasher = connector._make_request_hasher(cfg)
    connector._seed = connector.request_hasher.seed
    connector._bind_request_block_hasher()
    request = SimpleNamespace(
        all_token_ids=list(range(10)),
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
    )
    base = connector.request_hasher.make_request_block_hasher(4)(request)
    keys = connector.request_block_hasher(request)
    assert len(keys) == 2
    assert keys == base
    assert all(len(key) == 16 and key[14:] == bytes(2) for key in keys)
    direct = ucm_connector.UCMDirectConnector.__new__(ucm_connector.UCMDirectConnector)
    direct._kv_cache_config = None
    assert direct._make_request_hasher(cfg).seed == connector._seed
