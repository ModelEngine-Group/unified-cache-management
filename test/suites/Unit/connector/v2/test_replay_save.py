"""Save planning must exclude persistent KV already covered by a prefix hit."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.stage(0), pytest.mark.feature("connector_v2")]


@pytest.mark.parametrize(
    ("start", "scheduled", "local_hit", "external_hit", "expected_start", "expected_end"),
    [
        (29312, 3456, 0, 29440, 29440, 32768),
        (29312, 3456, 29440, 0, 29440, 32768),
        (29312, 64, 0, 29440, 29440, 29376),
        (29376, 128, 0, 29440, 29440, 29504),
        (128, 128, 128, 0, 128, 256),  # A failed load has been rolled back.
    ],
)
def test_replay_does_not_dump_the_hit(
    start, scheduled, local_hit, external_hit, expected_start, expected_end
):
    from ucm.integration.vllm.v2.connector_scheduler import UCMConnectorScheduler

    scheduler = object.__new__(UCMConnectorScheduler)
    scheduler.build_dispatch_plans = Mock(return_value=())
    scheduler._state_dump_plans = Mock(return_value=())
    state = SimpleNamespace(
        request=SimpleNamespace(num_computed_tokens=start),
        num_tokens=32768,
        hbm_hit_tokens=local_hit,
        external_hit_tokens=external_hit,
        needs_load=False,
    )

    scheduler._build_request_plan("request", state, (), scheduled)

    load, dump = scheduler.build_dispatch_plans.call_args_list
    assert load.args[2:4] == (local_hit + external_hit, local_hit + external_hit)
    assert dump.args[2:4] == (expected_start, expected_end)
    assert dump.kwargs == {"is_load": False}
