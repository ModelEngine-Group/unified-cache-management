"""CPU-only checks of wait ordering, batch boundaries, and runtime-hook coexistence."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import wrapt


@pytest.fixture
def runtime(monkeypatch):
    hooks = {}

    def when_imported(name):
        def register(callback):
            hooks[name] = callback
            return callback

        return register

    monkeypatch.setattr(wrapt, "when_imported", when_imported)
    path = (
        Path(__file__).parents[3] / "ucm/integration/vllm/patch/layerwise_sync_patch.py"
    )
    spec = importlib.util.spec_from_file_location("layerwise_sync_patch_test", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    patch.install()
    events = []

    class Layerwise:
        def __init__(self):
            self._connector_metadata = True
            self.need_load = True
            self.layer_name_to_id = {"attention": 0, "indexer": 0, "next": 1}
            self.fail_wait = False

        def start_load_kv(self, context, **kwargs):
            events.append(("start", context, kwargs))
            return "started"

        def wait_for_layer_load(self, name):
            events.append(("wait", name))
            if self.fail_wait:
                raise RuntimeError("load failed")
            events.append(("submit_next", self.layer_name_to_id[name] + 1))
            return "loaded"

    ucm_module = ModuleType("ucm.integration.vllm.ucm_connector")
    ucm_module.UCMLayerWiseConnector = Layerwise
    monkeypatch.setitem(sys.modules, ucm_module.__name__, ucm_module)
    hooks[ucm_module.__name__](ucm_module)
    native = Layerwise()
    native.start_load_kv("batch")
    group = SimpleNamespace(
        connector=native,
        has_connector_metadata=lambda: True,
        wait_for_layer_load=native.wait_for_layer_load,
    )
    transfer = ModuleType("vllm.distributed.kv_transfer")
    transfer.has_kv_transfer_group = lambda: True
    transfer.is_v1_kv_transfer_group = lambda: True
    transfer.get_kv_transfer_group = lambda: group
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)
    context = SimpleNamespace(attn_metadata=object())
    forward_context = ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, forward_context.__name__, forward_context)

    class MLA:
        is_sparse = True
        indexer = object()
        mla_attn = SimpleNamespace(layer_name="attention")

        def forward(self, value, *, scale=1):
            events.append(("indexer", value))
            # The upstream main-attention entry still makes its original wait.
            native.wait_for_layer_load("attention")
            events.append(("attention", value))
            events.append(("save", value))
            return value * scale

    mla_module = SimpleNamespace(MultiHeadLatentAttentionWrapper=MLA)
    hooks["vllm.model_executor.layers.mla"](mla_module)
    events.clear()
    return SimpleNamespace(
        patch=patch,
        hooks=hooks,
        events=events,
        native=native,
        group=group,
        transfer=transfer,
        context=context,
        ucm_module=ucm_module,
        mla_module=mla_module,
        mla=MLA(),
    )


def test_wait_precedes_indexer_and_preserves_forward_and_save(runtime):
    assert runtime.mla.forward(7, scale=3) == 21
    assert runtime.events == [
        ("wait", "attention"),
        ("submit_next", 1),
        ("indexer", 7),
        ("attention", 7),
        ("save", 7),
    ]


def test_wait_is_deduplicated_by_layer_id_and_reset_for_each_batch(runtime):
    native = runtime.native
    assert native.wait_for_layer_load("indexer") == "loaded"
    assert native.wait_for_layer_load("attention") is None
    assert native.wait_for_layer_load("next") == "loaded"
    assert runtime.events == [
        ("wait", "indexer"),
        ("submit_next", 1),
        ("wait", "next"),
        ("submit_next", 2),
    ]
    assert native.start_load_kv("second", extra=True) == "started"
    native.wait_for_layer_load("attention")
    assert runtime.events[-3:] == [
        ("start", "second", {"extra": True}),
        ("wait", "attention"),
        ("submit_next", 1),
    ]


@pytest.mark.parametrize("field", ["_connector_metadata", "need_load"])
def test_inactive_batch_does_not_wait_or_mark_layer(runtime, field):
    setattr(runtime.native, field, False)
    runtime.native.wait_for_layer_load("attention")
    assert runtime.events == []
    setattr(runtime.native, field, True)
    assert runtime.native.wait_for_layer_load("attention") == "loaded"


def test_failed_wait_is_not_marked_complete_and_stops_indexer(runtime):
    runtime.native.fail_wait = True
    with pytest.raises(RuntimeError, match="load failed"):
        runtime.mla.forward(4)
    assert runtime.events == [("wait", "attention")]
    runtime.native.fail_wait = False
    runtime.mla.forward(4)
    assert runtime.events[1:] == [
        ("wait", "attention"),
        ("submit_next", 1),
        ("indexer", 4),
        ("attention", 4),
        ("save", 4),
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "dense",
        "no_indexer",
        "no_group",
        "not_v1",
        "not_layerwise",
        "no_connector_metadata",
        "no_attention_metadata",
    ],
)
def test_nonmatching_paths_keep_the_original_attention_wait(runtime, reason):
    if reason == "dense":
        runtime.mla.is_sparse = False
    elif reason == "no_indexer":
        runtime.mla.indexer = None
    elif reason == "no_group":
        runtime.transfer.has_kv_transfer_group = lambda: False
    elif reason == "not_v1":
        runtime.transfer.is_v1_kv_transfer_group = lambda: False
    elif reason == "not_layerwise":
        runtime.group.connector = object()
    elif reason == "no_connector_metadata":
        runtime.group.has_connector_metadata = lambda: False
    else:
        runtime.context.attn_metadata = None
    assert runtime.mla.forward(3) == 3
    assert runtime.events == [
        ("indexer", 3),
        ("wait", "attention"),
        ("submit_next", 1),
        ("attention", 3),
        ("save", 3),
    ]


def test_installing_again_does_not_wrap_methods_twice(runtime):
    start = type(runtime.native).start_load_kv
    wait = type(runtime.native).wait_for_layer_load
    forward = type(runtime.mla).forward
    runtime.patch.install()
    runtime.hooks["ucm.integration.vllm.ucm_connector"](runtime.ucm_module)
    runtime.hooks["vllm.model_executor.layers.mla"](runtime.mla_module)
    assert type(runtime.native).start_load_kv is start
    assert type(runtime.native).wait_for_layer_load is wait
    assert type(runtime.mla).forward is forward


def test_existing_experiment_hook_markers_prevent_double_install(runtime):
    def start(self, *args, **kwargs):
        pass

    def wait(self, name):
        pass

    def forward(self, *args, **kwargs):
        pass

    forward._glm52_sync_fix = True
    old_connector = type(
        "PreviouslyPatched",
        (),
        {"_glm52_sync_fix": True, "start_load_kv": start, "wait_for_layer_load": wait},
    )
    old_mla = type("PreviouslyPatchedMLA", (), {"forward": forward})
    runtime.hooks["ucm.integration.vllm.ucm_connector"](
        SimpleNamespace(UCMLayerWiseConnector=old_connector)
    )
    runtime.hooks["vllm.model_executor.layers.mla"](
        SimpleNamespace(MultiHeadLatentAttentionWrapper=old_mla)
    )
    assert old_connector.start_load_kv is start
    assert old_connector.wait_for_layer_load is wait
    assert old_mla.forward is forward
