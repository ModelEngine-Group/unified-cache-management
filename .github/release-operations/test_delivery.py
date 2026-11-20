"""Tag gate, manual rerun, and immutable delivery receipt contracts."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller import Controller
from delivery import TRACKING_START, main
from syntax import OperationError, parse_release
from test_controller import PLAN, REPOSITORY, MemoryGitHub

SHA = "a" * 40
RUN = {
    "tag": "v0.12.0",
    "version": "0.12.0",
    "source_sha": SHA,
    "run_id": 100,
    "run_attempt": 1,
    "url": f"https://github.com/{REPOSITORY}/actions/runs/100/attempts/1",
}


class DeliveryGitHub(MemoryGitHub):
    def __init__(self):
        super().__init__()
        self.tag_sha = SHA
        self.receipts = {}
        self.source_run = {
            "id": 100,
            "run_attempt": 2,
            "path": ".github/workflows/release-tag.yml",
            "event": "push",
            "head_branch": RUN["tag"],
            "head_sha": SHA,
            "conclusion": None,
        }
        self.artifacts = []
        self.comparisons = []
        self.published = {
            "id": 9,
            "tag_name": RUN["tag"],
            "draft": False,
            "body": "Human release notes.\n\n<!-- ucm-release:begin -->\nInstall commands.\n<!-- ucm-release:end -->\n",
            "html_url": f"https://github.com/{REPOSITORY}/releases/tag/{RUN['tag']}",
        }

    def rest(self, path, method="GET", data=None):
        if path == "actions/runs/100":
            return copy.deepcopy(self.source_run)
        if path.startswith("actions/runs/") and "/artifacts?" in path:
            return {"artifacts": copy.deepcopy(self.artifacts)}
        if path.startswith("milestones/") and method == "GET":
            return copy.deepcopy(
                next(
                    m for m in self.milestones if m["number"] == int(path.split("/")[1])
                )
            )
        if path == "commits/" + RUN["tag"]:
            return {"sha": self.tag_sha}
        if path.startswith("compare/"):
            self.comparisons.append(path)
        if path.startswith("releases/tags/"):
            return copy.deepcopy(self.published)
        if path == "releases/9" and method == "PATCH":
            self.published.update(data)
            return copy.deepcopy(self.published)
        return super().rest(path, method, data)

    def artifact_json(self, artifact_id, filename):
        return copy.deepcopy(self.receipts[artifact_id])

    def receipt(self, attempt=1):
        self.artifacts.append(
            {
                "id": attempt,
                "name": f"ucm-release-acceptance-run-100-attempt-{attempt}",
                "expired": False,
            }
        )
        self.receipts[attempt] = {
            "status": "complete",
            "repository": REPOSITORY,
            "tag": RUN["tag"],
            "source_sha": SHA,
            "actions_run_id": 100,
            "run_attempt": attempt,
        }


class DeliveryContracts(unittest.TestCase):
    def setUp(self):
        self.api = DeliveryGitHub()
        self.controller = Controller(
            self.api, {"repository": REPOSITORY, "projects": {}}
        )
        self.controller.initialize_or_update(1, "release_init")
        self.api.milestones[0]["state"] = "open"
        for number in (3, 4):
            self.api.items[number]["state"] = "closed"

    def delivery(self):
        return self.controller.load(1)[1]["deliveries"][RUN["tag"]]

    def test_gate_blocks_open_work_and_writes_report(self):
        self.api.items[3]["state"] = "open"
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        state = self.controller.load(1)[1]
        self.assertEqual(state["release_check"]["open_features"], [3])
        self.assertEqual(
            state["release_check"]["candidate"], {"ref": RUN["tag"], "sha": SHA}
        )
        self.assertEqual(self.delivery()["status"], "failed")
        self.assertNotEqual(self.api.milestones[0]["state"], "closed")

    def test_gate_inspects_tag_commit_instead_of_moving_branch(self):
        pr = self.api.add_pr(10, "#3")
        pr.update(state="closed", merged_at="2026-11-20", merge_commit_sha="b" * 40)
        self.controller.delivery_gate(RUN)
        self.assertEqual(self.api.comparisons, [f"compare/{'b' * 40}...{SHA}"])
        self.assertTrue(self.delivery()["gate_passed"])
        self.assertEqual(self.delivery()["status"], "publishing")

    def test_gate_blocks_scope_edits_without_applying_them(self):
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "### Change Log",
            "- New requirement\n\n  description:\n    New work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log",
        )
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        self.assertEqual(len(self.api.items), 4)

    def test_missing_duplicate_and_uninitialized_versions_fail_gate(self):
        with self.assertRaisesRegex(OperationError, "found 0"):
            self.controller.delivery_gate(dict(RUN, tag="v0.13.0", version="0.13.0"))
        self.api.items[5] = self.api.make_issue(5, "Duplicate", PLAN, ["release"])
        with self.assertRaisesRegex(OperationError, "found 2"):
            self.controller.delivery_gate(RUN)
        del self.api.items[1]
        with self.assertRaisesRegex(OperationError, "/release_init"):
            self.controller.delivery_gate(RUN)

    def test_new_attempt_replaces_failure_and_delayed_events_cannot_overwrite_it(self):
        self.controller.delivery_gate(RUN)
        self.controller.delivery_finished(RUN, "failure")
        again = dict(
            RUN, run_attempt=2, url=RUN["url"].replace("attempts/1", "attempts/2")
        )
        self.controller.delivery_started(again)
        self.assertEqual(self.delivery()["status"], "publishing")
        self.assertEqual(self.delivery()["run_attempt"], 2)
        self.controller.delivery_finished(RUN, "cancelled")
        self.assertEqual(self.delivery()["status"], "publishing")
        # Rerun-all executes the gate again even when the lifecycle notification
        # has already marked this new attempt as publishing.
        self.controller.delivery_gate(again)
        self.assertEqual(self.delivery()["status"], "publishing")

    def test_terminal_result_survives_late_start_and_regular_sync(self):
        self.controller.delivery_gate(RUN)
        self.controller.delivery_finished(RUN, "cancelled")
        self.controller.delivery_started(RUN)
        self.controller.reconcile(1)
        self.assertEqual(self.delivery()["status"], "cancelled")
        self.assertIn("Release Delivery", self.api.items[1]["body"])
        self.assertIn("Attempt 1", self.api.items[1]["body"])

    def test_success_requires_receipt_then_preserves_notes_and_closes_milestone(self):
        self.controller.delivery_gate(RUN)
        with self.assertRaisesRegex(OperationError, "receipt is unavailable"):
            self.controller.delivery_finished(RUN, "success")
        self.assertNotEqual(self.api.milestones[0]["state"], "closed")
        again = dict(RUN, run_attempt=2)
        self.controller.delivery_gate(again)
        self.api.receipt(2)
        self.controller.delivery_finished(again, "success")
        self.controller.delivery_finished(again, "success")
        self.assertEqual(self.delivery()["status"], "delivered")
        self.assertEqual(self.api.milestones[0]["state"], "closed")
        self.assertEqual(self.api.published["body"].count(TRACKING_START), 1)
        self.assertIn("Human release notes.", self.api.published["body"])
        self.assertIn("Install commands.", self.api.published["body"])
        self.assertIn("Release tracking: #1", self.api.published["body"])
        self.assertEqual(self.api.items[1]["state"], "open")
        self.assertEqual(self.api.items[2]["state"], "open")

    def test_wrong_receipt_or_retargeted_tag_cannot_archive(self):
        self.controller.delivery_gate(RUN)
        self.api.receipt()
        self.api.receipts[1]["source_sha"] = "c" * 40
        with self.assertRaisesRegex(OperationError, "does not match"):
            self.controller.delivery_finished(RUN, "success")
        self.assertNotEqual(self.api.milestones[0]["state"], "closed")
        self.api.receipts[1]["source_sha"] = SHA
        self.api.tag_sha = "c" * 40
        with self.assertRaisesRegex(OperationError, "no longer points"):
            self.controller.delivery_finished(dict(RUN, run_attempt=2), "success")
        self.assertNotEqual(self.api.milestones[0]["state"], "closed")

    def test_partial_rerun_can_reuse_successful_delivery_receipt(self):
        self.controller.delivery_gate(RUN)
        self.api.receipt()
        self.controller.delivery_finished(dict(RUN, run_attempt=3), "success")
        self.assertEqual(self.delivery()["status"], "delivered")
        self.assertEqual(self.delivery()["run_attempt"], 3)

    def test_v_tag_rc_version_is_supported_by_release_template(self):
        parsed = parse_release(PLAN.replace("0.12.0", "0.12.0rc1"), REPOSITORY)
        self.assertEqual(parsed["version"], "0.12.0rc1")

    def observe(self, attempt=2, action="in_progress"):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps({"repository": REPOSITORY, "projects": {}}), encoding="utf-8"
            )
            event = Path(directory) / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "action": action,
                        "workflow_run": {"id": 100, "run_attempt": attempt},
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "GITHUB_REPOSITORY": REPOSITORY,
                        "GITHUB_EVENT_PATH": str(event),
                        "GITHUB_RUN_ID": "999",
                    },
                ),
                patch("sys.argv", ["delivery.py", "observe", "--config", str(config)]),
                patch("controller.GitHub", return_value=self.api),
            ):
                main()

    def test_partial_rerun_event_uses_publisher_identity_and_attempt_link(self):
        self.controller.delivery_gate(RUN)
        self.controller.delivery_finished(RUN, "failure")
        self.observe()
        self.assertEqual(self.delivery()["run_id"], 100)
        self.assertEqual(self.delivery()["run_attempt"], 2)
        self.assertEqual(self.delivery()["status"], "publishing")
        self.assertTrue(self.delivery()["url"].endswith("/100/attempts/2"))

    def test_stale_observer_and_other_tag_routes_leave_release_untouched(self):
        self.observe(attempt=1, action="completed")
        self.assertNotIn("deliveries", self.controller.load(1)[1])
        for tag in ("draft/v0.12.0", "nightly/v0.12.0"):
            self.api.source_run["head_branch"] = tag
            self.observe()
        self.assertNotIn("deliveries", self.controller.load(1)[1])

    def test_observer_rejects_other_workflow_and_records_cancellation(self):
        self.api.source_run["path"] = ".github/workflows/other.yml"
        with self.assertRaisesRegex(OperationError, "Unexpected release workflow"):
            self.observe()
        self.api.source_run["path"] = ".github/workflows/release-tag.yml"
        self.api.source_run["conclusion"] = "cancelled"
        self.observe(action="completed")
        self.assertEqual(self.delivery()["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
