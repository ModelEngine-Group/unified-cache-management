"""Tag gate, manual rerun, and immutable delivery receipt contracts."""

import copy
import unittest

from controller import Controller
from delivery import PUBLISHED_START, TRACKING_START
from presentation import COMMAND_END, checklist_region, command_feedback
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
        self.workflow_run = {
            "id": 100,
            "run_attempt": 1,
            "path": ".github/workflows/release-tag.yml",
            "event": "push",
            "head_repository": {"full_name": REPOSITORY},
            "head_branch": RUN["tag"],
            "head_sha": SHA,
            "status": "in_progress",
            "conclusion": None,
            "updated_at": "2026-11-25T10:00:00Z",
        }
        self.jobs = []
        self.published = {
            "id": 9,
            "tag_name": RUN["tag"],
            "draft": False,
            "body": "Human release notes.\n\n<!-- ucm-release:begin -->\nInstall commands.\n<!-- ucm-release:end -->\n",
            "html_url": f"https://github.com/{REPOSITORY}/releases/tag/{RUN['tag']}",
        }

    def rest(self, path, method="GET", data=None):
        if path == "actions/runs/100":
            return copy.deepcopy(self.workflow_run)
        if path.startswith("actions/runs/100/attempts/") and "/jobs?" in path:
            return {"jobs": copy.deepcopy(self.jobs)}
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
        _, delivery = checklist_region(self.api.items[1]["body"], "delivery")
        _, check = checklist_region(self.api.items[1]["body"], "check")
        self.assertIn("**v0.12.0 · Publishing**", delivery)
        self.assertIn(f"Candidate `v0.12.0` · `{SHA}`", check)
        self.assertIn("**Ready**", check)
        self.assertNotIn("Pending `/update`", check)
        self.api.items[2]["state"] = "open"
        with self.assertRaisesRegex(OperationError, "blocked"):
            self.controller.delivery_gate(RUN)
        _, check = checklist_region(self.api.items[1]["body"], "check")
        self.assertIn("**Blocked**", check)
        self.assertIn("Open Checklist or Bug Archive · #2", check)
        self.assertIn("**v0.12.0 · Failed**", self.api.items[1]["body"])
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

    def test_scope_diff_is_visible_without_changing_the_milestone_gate(self):
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = body.replace(
            "release_date: TBD", "release_date: 2026-12-01"
        )
        self.controller.delivery_gate(RUN)
        _, check = checklist_region(self.api.items[1]["body"], "check")
        self.assertIn("**Ready**", check)
        self.assertIn("Edited release fields · release_date", check)
        self.assertIsNone(self.api.milestones[0]["due_on"])

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
        checklist = self.api.items[1]
        checklist["body"] = command_feedback(
            checklist["body"], "/release_check", "Succeeded", "", None, None
        )
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
        self.assertIn("**v0.12.0 · Delivered**", self.api.items[1]["body"])
        self.assertTrue(self.api.items[1]["body"].rstrip().endswith(COMMAND_END))
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

    def observe(self, attempt=1):
        self.controller.observe_delivery(
            {"workflow_run": {"id": 100, "run_attempt": attempt}}
        )

    def test_observer_records_failed_jobs_and_a_cancelled_retry(self):
        self.observe()
        self.assertIn("**v0.12.0 · Checking**", self.api.items[1]["body"])
        self.controller.delivery_gate(RUN)
        self.observe()
        self.assertIn("**v0.12.0 · Publishing**", self.api.items[1]["body"])
        self.api.workflow_run.update(status="completed", conclusion="failure")
        self.api.jobs = [{"name": "Wheel · CUDA", "conclusion": "failure"}]
        self.observe()
        self.assertIn("**v0.12.0 · Failed**", self.api.items[1]["body"])
        self.assertIn("Failed jobs · Wheel · CUDA", self.api.items[1]["body"])
        self.api.workflow_run.update(
            run_attempt=2, status="in_progress", conclusion=None
        )
        self.observe(attempt=2)
        self.assertIn("**v0.12.0 · Publishing**", self.api.items[1]["body"])
        self.assertIn("Attempt 2", self.api.items[1]["body"])
        self.api.workflow_run.update(status="completed", conclusion="cancelled")
        self.observe(attempt=2)
        cancelled = self.api.items[1]["body"]
        self.assertIn("**v0.12.0 · Cancelled**", cancelled)
        self.assertIn("Finished 2026-11-25T10:00:00Z", cancelled)
        self.observe()
        self.controller.write_delivery(1, RUN, "failed", error="Older result")
        self.assertEqual(self.api.items[1]["body"], cancelled)
        self.assertEqual(self.api.milestones[0]["state"], "open")
        self.assertFalse(self.api.comments)

    def test_observer_checks_source_and_ignores_nonformal_tags(self):
        self.api.workflow_run["head_repository"]["full_name"] = "other/repository"
        with self.assertRaisesRegex(OperationError, "Unexpected release workflow"):
            self.observe()
        self.api.workflow_run["head_repository"]["full_name"] = REPOSITORY
        original = self.api.items[1]["body"]
        for tag in ("draft/v0.12.0", "nightly/v0.12.0-20261125-1"):
            self.api.workflow_run["head_branch"] = tag
            self.observe()
        self.assertEqual(self.api.items[1]["body"], original)


if __name__ == "__main__":
    unittest.main()
