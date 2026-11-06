"""Tests for tracking the connector-v2 worker API across its moves.

The refactor that split ``UCMConnector`` into a facade plus
``UCMConnectorWorker`` moved ``spec`` / ``layout`` / ``transfer_builder`` onto
``connector_worker``, renamed ``tokens_per_chunk`` to ``chunk_size`` and
``attention_type`` to ``restore_policy``. The checker has to follow those
moves, and -- more importantly -- has to **fail** when it cannot, because a v2
run whose internals are not found would skip every v2 check and still report
PASS.
"""

from __future__ import annotations

import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ucm_toolkit.tools.model_check.common import (  # noqa: E402
    check_transfers_inside_native_blocks,
    connector_is_v2,
    group_block_tokens,
    plan_restore_policy,
    require_metadata_kind,
    spec_chunk_tokens,
    v2_worker_internals,
)
from ucm_toolkit.tools.model_check.config import LEGACY_CONNECTOR_MODULE  # noqa: E402

V2_MODULE = "ucm.integration.vllm.v2.ucm_connector"


def _vllm_config(module):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_module_path=module)
    )


def _internals(**overrides):
    """An object shaped like UCMConnectorWorker."""

    fields = {"spec": object(), "layout": object(), "transfer_builder": object()}
    fields.update(overrides)
    return SimpleNamespace(**fields)


class WorkerInternalsTest(unittest.TestCase):
    def test_finds_them_on_the_object_itself(self):
        """The shape before the facade split."""

        worker = _internals()
        self.assertIs(v2_worker_internals(worker), worker)

    def test_finds_them_behind_connector_worker(self):
        """The shape after the facade split."""

        inner = _internals()
        facade = SimpleNamespace(connector_worker=inner, connector_scheduler=object())
        self.assertIs(v2_worker_internals(facade), inner)

    def test_finds_them_behind_a_legacy_connector_attribute(self):
        inner = _internals()
        self.assertIs(v2_worker_internals(SimpleNamespace(connector=inner)), inner)

    def test_finds_them_two_levels_down(self):
        """A facade reached through the toolkit's own ``connector`` handle."""

        inner = _internals()
        outer = SimpleNamespace(connector=SimpleNamespace(connector_worker=inner))
        self.assertIs(v2_worker_internals(outer), inner)

    def test_a_partial_object_is_not_accepted(self):
        """Half the markers means the checks would silently do nothing."""

        half = SimpleNamespace(spec=object(), layout=object())
        with self.assertRaises(AssertionError):
            v2_worker_internals(half)

    def test_failure_says_what_it_looked_at_and_how_to_fix_it(self):
        with self.assertRaises(AssertionError) as caught:
            v2_worker_internals(SimpleNamespace(connector=SimpleNamespace()))
        message = str(caught.exception)
        self.assertIn("_V2_WORKER_ATTRS", message)
        self.assertIn("never let a run report PASS", message)


class ChunkTokensTest(unittest.TestCase):
    def test_reads_the_current_name(self):
        self.assertEqual(spec_chunk_tokens(SimpleNamespace(chunk_size=128)), 128)

    def test_still_reads_the_previous_name(self):
        self.assertEqual(spec_chunk_tokens(SimpleNamespace(tokens_per_chunk=256)), 256)

    def test_prefers_the_current_name(self):
        spec = SimpleNamespace(chunk_size=128, tokens_per_chunk=999)
        self.assertEqual(spec_chunk_tokens(spec), 128)

    def test_refuses_to_guess(self):
        with self.assertRaises(AssertionError):
            spec_chunk_tokens(SimpleNamespace())


class RestorePolicyTest(unittest.TestCase):
    class Policy(Enum):
        FA = "FA"
        WA = "WA"
        STATE = "State"

    def test_reads_the_current_name(self):
        plan = SimpleNamespace(restore_policy=self.Policy.FA)
        self.assertEqual(plan_restore_policy(plan), "FA")

    def test_still_reads_the_previous_name(self):
        plan = SimpleNamespace(attention_type=self.Policy.STATE)
        self.assertEqual(plan_restore_policy(plan), "STATE")

    def test_window_and_state_are_not_mistaken_for_full_attention(self):
        for member in (self.Policy.WA, self.Policy.STATE):
            plan = SimpleNamespace(restore_policy=member)
            self.assertNotEqual(plan_restore_policy(plan), "FA")

    def test_refuses_to_guess(self):
        with self.assertRaises(AssertionError):
            plan_restore_policy(SimpleNamespace())


class GroupBlockTokensTest(unittest.TestCase):
    """``tokens_per_block`` became ``block_size`` + ``block_token_span``.

    The layout-only log read the old name, which no run exercised until the
    Ascend and 5090 device-less runs both failed on the line above it.
    """

    def test_reads_the_current_span(self):
        group = SimpleNamespace(block_token_span=256, block_size=128)
        self.assertEqual(group_block_tokens(group), 256)

    def test_falls_back_to_block_size_alone(self):
        self.assertEqual(group_block_tokens(SimpleNamespace(block_size=128)), 128)

    def test_still_reads_the_previous_name(self):
        self.assertEqual(group_block_tokens(SimpleNamespace(tokens_per_block=64)), 64)

    def test_refuses_to_guess(self):
        with self.assertRaises(AssertionError):
            group_block_tokens(SimpleNamespace())


class LayoutDescriptionTest(unittest.TestCase):
    """``describe_ucm_layout`` must survive a spec that dropped a field."""

    class Kind(Enum):
        FULL_ATTENTION = "FULL_ATTENTION"

    def _spec(self, **extra):
        group = SimpleNamespace(
            group_id=0,
            kinds=frozenset({self.Kind.FULL_ATTENTION}),
            block_token_span=128,
            tail_tokens=None,
            is_persistent=True,
        )
        fields = {
            "groups": (group,),
            "chunk_size": 128,
            "device_type": "cuda",
            "blocks_per_chunk": (1,),
        }
        fields.update(extra)
        return SimpleNamespace(**fields)

    def _describe(self, spec):
        from ucm_toolkit.tools.model_check import common

        lines = []
        original, common.log = common.log, lines.append
        try:
            common.describe_ucm_layout(
                SimpleNamespace(
                    connector_worker=SimpleNamespace(
                        spec=spec, layout=object(), transfer_builder=object()
                    )
                )
            )
        finally:
            common.log = original
        return "\n".join(lines)

    def test_omits_a_field_the_spec_no_longer_carries(self):
        """Upstream calls scheduler granularity a diagnostic, not metadata."""

        text = self._describe(self._spec())
        self.assertIn("chunk_size=128", text)
        self.assertIn("device_type=cuda", text)
        self.assertNotIn("scheduler_block_size", text)
        self.assertIn("tokens_per_block=128", text)

    def test_still_reports_it_when_present(self):
        text = self._describe(self._spec(scheduler_block_size=128))
        self.assertIn("scheduler_block_size=128", text)


class MetadataKindTest(unittest.TestCase):
    """The launch config decides v2 vs legacy; the metadata must agree."""

    def test_v2_module_is_v2(self):
        self.assertTrue(connector_is_v2(_vllm_config(V2_MODULE)))

    def test_legacy_module_is_not_v2(self):
        self.assertFalse(connector_is_v2(_vllm_config(LEGACY_CONNECTOR_MODULE)))

    def test_matching_metadata_is_accepted(self):
        v2_meta = SimpleNamespace(requests={})
        legacy_meta = SimpleNamespace(request_meta={})
        self.assertTrue(require_metadata_kind(_vllm_config(V2_MODULE), v2_meta, "dump"))
        self.assertFalse(
            require_metadata_kind(
                _vllm_config(LEGACY_CONNECTOR_MODULE), legacy_meta, "dump"
            )
        )

    def test_renamed_v2_metadata_is_refused_not_downgraded(self):
        renamed = SimpleNamespace(request_plans={})
        with self.assertRaises(AssertionError) as ctx:
            require_metadata_kind(_vllm_config(V2_MODULE), renamed, "dump")
        self.assertIn(V2_MODULE, str(ctx.exception))
        self.assertIn("requests", str(ctx.exception))

    def test_v2_metadata_on_a_legacy_run_is_refused(self):
        with self.assertRaises(AssertionError):
            require_metadata_kind(
                _vllm_config(LEGACY_CONNECTOR_MODULE),
                SimpleNamespace(requests={}),
                "load",
            )


class TransferRangeCheckTest(unittest.TestCase):
    """The range check must fail, not report 0, when it has nothing to check."""

    def _worker(self, transfers=()):
        builder = SimpleNamespace(build_load_transfers=lambda metadata: transfers)
        return _internals(
            layout=SimpleNamespace(layer_id_by_name={}),
            transfer_builder=builder,
            use_layerwise=False,
        )

    def test_legacy_metadata_is_refused(self):
        with self.assertRaises(ValueError):
            check_transfers_inside_native_blocks(
                None, self._worker(), SimpleNamespace(request_meta={})
            )

    def test_no_load_ranges_is_an_error_not_zero(self):
        with self.assertRaises(AssertionError):
            check_transfers_inside_native_blocks(
                None, self._worker(), SimpleNamespace(requests={})
            )


if __name__ == "__main__":
    unittest.main()
