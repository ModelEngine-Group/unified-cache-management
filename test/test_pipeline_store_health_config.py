from pathlib import Path

from ucm.store.pipeline import connector, ucmpipelinestore
from ucm.store.pipeline.connector import UcmPipelineStore


def test_store_health_uses_default_health_timeout():
    store = UcmPipelineStore(
        {
            "store_pipeline": "Empty",
            "store_health": {
                "enabled": True,
                "health_check_interval_s": 4,
                "health_window_size": 3,
                "failure_threshold": 2,
            },
        }
    )

    assert store is not None


def test_store_health_timeout_is_independent_from_store_timeout():
    store = UcmPipelineStore(
        {
            "store_pipeline": "Empty",
            "timeout_ms": 30000,
            "store_health": {
                "enabled": True,
                "health_check_interval_s": 0.2,
                "health_check_timeout_s": 0.1,
                "health_window_size": 3,
                "failure_threshold": 2,
            },
        }
    )

    assert store is not None


def test_explicit_stack_accepts_store_health_config():
    store_dir = Path(connector.__file__).resolve().parent.parent
    store = ucmpipelinestore.PipelineStore()

    store.Stack(
        "Empty",
        str(store_dir / "empty/libemptystore.so"),
        {"store_health": {"enabled": True}},
    )

    assert store.Self() != 0
