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
    plan_restore_policy,
    spec_chunk_tokens,
    v2_worker_internals,
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


if __name__ == "__main__":
    unittest.main()
