"""Tag gate, manual rerun, and immutable delivery receipt contracts."""

import copy
import unittest

from controller import Controller
from delivery import PUBLISHED_START, TRACKING_START
from syntax import OperationError
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
        self.artifacts = []
        self.published = {
            "id": 9,
            "tag_name": RUN["tag"],
            "draft": False,
            "body": "Human release notes.\n\n<!-- ucm-release:begin -->\nInstall commands.\n<!-- ucm-release:end -->\n",
            "html_url": f"https://github.com/{REPOSITORY}/releases/tag/{RUN['tag']}",
        }

    def rest(self, path, method="GET", data=None):
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
        for number in (1, 2, 3, 4):
            self.api.items[number]["state"] = "closed"

    def test_gate_finds_closed_checklist_and_verifies_closed_archive(self):
        self.controller.delivery_gate(RUN)
        self.assertFalse(self.api.comments)
        self.api.items[2]["state"] = "open"
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        self.assertEqual(self.api.milestones[0]["state"], "open")

    def test_gate_counts_manually_added_unlabeled_issues(self):
        issue = self.api.make_issue(8, "Manual release item", "", [])
        issue["milestone"] = self.api.milestones[0]
        self.api.items[8] = issue
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        issue["state"] = "closed"
        self.controller.delivery_gate(RUN)

    def test_gate_blocks_pending_scope_edits_without_applying_them(self):
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "### Change Log",
            "- New requirement\n\n  description:\n    New work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log",
        )
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        self.assertEqual(len(self.api.items), 4)

    def test_missing_duplicate_and_uninitialized_versions_fail_gate(self):
        with self.assertRaisesRegex(OperationError, "Expected one Milestone"):
            self.controller.delivery_gate(dict(RUN, tag="v0.13.0", version="0.13.0"))
        duplicate = self.api.make_issue(5, "Duplicate Checklist", PLAN, ["release"])
        duplicate["milestone"] = self.api.milestones[0]
        self.api.items[5] = duplicate
        with self.assertRaisesRegex(OperationError, "Expected one Release Checklist"):
            self.controller.delivery_gate(RUN)
        del self.api.items[5]
        del self.api.parents[2]
        with self.assertRaisesRegex(OperationError, "Configure the Bug Archive"):
            self.controller.delivery_gate(RUN)

    def test_receipt_required_before_closing_milestone(self):
        with self.assertRaisesRegex(OperationError, "receipt is unavailable"):
            self.controller.delivery_finished(RUN, "success")
        self.assertEqual(self.api.milestones[0]["state"], "open")
        self.api.receipt()
        self.controller.delivery_finished(RUN, "success")
        self.assertEqual(self.api.milestones[0]["state"], "closed")
        self.assertIn("Human release notes.", self.api.published["body"])
        self.assertIn("Install commands.", self.api.published["body"])
        self.assertIn("Bug Archive: #2", self.api.published["body"])
        self.controller.delivery_finished(RUN, "success")
        self.assertEqual(self.api.published["body"].count(TRACKING_START), 1)
        self.assertEqual(self.api.items[1]["body"].count(PUBLISHED_START), 1)
        self.assertIn(self.api.published["html_url"], self.api.items[1]["body"])
        self.assertFalse(self.api.comments)

    def test_wrong_receipt_or_retargeted_tag_cannot_close_milestone(self):
        self.api.receipt()
        self.api.receipts[1]["actions_run_id"] = 999
        with self.assertRaisesRegex(OperationError, "does not match"):
            self.controller.delivery_finished(RUN, "success")
        self.api.receipts[1]["actions_run_id"] = 100
        self.api.tag_sha = "b" * 40
        with self.assertRaisesRegex(OperationError, "no longer points"):
            self.controller.delivery_finished(RUN, "success")
        self.assertEqual(self.api.milestones[0]["state"], "open")

    def test_partial_rerun_can_reuse_successful_delivery_receipt(self):
        self.api.receipt()
        self.controller.delivery_finished(dict(RUN, run_attempt=2), "success")
        self.assertEqual(self.api.milestones[0]["state"], "closed")

    def test_failed_release_leaves_milestone_and_checkout_intact(self):
        before = self.api.published["body"]
        self.controller.delivery_finished(RUN, "failure")
        self.assertEqual(self.api.milestones[0]["state"], "open")
        self.assertEqual(self.api.published["body"], before)
        self.assertFalse(self.api.comments)


if __name__ == "__main__":
    unittest.main()
