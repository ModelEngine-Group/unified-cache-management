"""Bounded native reads, lightweight routing, and first-association contracts."""

import copy
import re
import unittest
from unittest.mock import patch

from controller import Controller, GitHub
from syntax import OperationError
from test_controller import PLAN, REPOSITORY, MemoryGitHub


class CountingGitHub(MemoryGitHub):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.max_calls = 10000
        self.enforce = False

    def hit(self, kind, path):
        if self.enforce and len(self.calls) >= self.max_calls:
            raise OperationError("API/time budget")
        if (
            re.match(r"pulls\?|issues\?state=all&sort=updated", path)
            or "/reviews" in path
            or path.startswith("compare/")
        ):
            raise AssertionError(
                "Historical PR/discussion/compare reads are forbidden: " + path
            )
        self.calls.append((kind, path))

    def rest(self, path, method="GET", data=None):
        self.hit(method, path)
        return super().rest(path, method, data)

    def pages(self, path):
        if path == "issues?state=all&labels=release":
            raise AssertionError("Historical release enumeration is forbidden")
        self.hit("LIST", path)
        return super().pages(path)

    def call(self, path, method="GET", data=None):
        self.hit("CALL", (data or {}).get("query", path))
        return super().call(path, method, data)

    def graphql(self, query, **variables):
        self.hit("PROJECT", query)
        return super().graphql(query, **variables)


class NativeStateContracts(unittest.TestCase):
    def setUp(self):
        self.api = CountingGitHub()
        self.controller = Controller(
            self.api,
            {"repository": REPOSITORY, "projects": {}, "sync": {"max_api_calls": 300}},
        )
        self.controller.initialize_or_update(1, "release_init")
        self.api.calls.clear()

    def test_unrelated_issue_uses_no_checklists_or_comments(self):
        self.controller.handle({"issue": {"number": 9000, "labels": []}}, "issues")
        self.assertEqual(self.api.calls, [])

    def test_unrelated_pr_uses_only_batched_pr_lookup(self):
        self.api.add_pr(90, "")
        self.controller.handle({"pull_request": {"number": 90}}, "pull_request_target")
        self.assertEqual(len(self.api.calls), 1)
        self.assertIn("closingIssuesReferences", self.api.calls[0][1])

    def test_direct_issue_event_reads_only_affected_milestone(self):
        self.controller.handle({"issue": self.api.items[3]}, "issues")
        checklists = [
            path
            for kind, path in self.api.calls
            if kind == "LIST" and "labels=release" in path
        ]
        self.assertEqual(checklists, ["issues?state=all&labels=release&milestone=1"])
        self.assertFalse(any("/comments" in path for _, path in self.api.calls))

    def test_demilestoned_event_refreshes_previous_total(self):
        issue = self.api.items[4]
        issue["milestone"] = None
        self.controller.handle(
            {
                "issue": issue,
                "action": "demilestoned",
                "milestone": self.api.milestones[0],
            },
            "issues",
        )
        self.assertNotIn("| #4 |", self.api.items[1]["body"])
        _, report = self.controller.release_check(1)
        self.assertNotIn(4, report["open_issues"])
        self.assertFalse(report["pending_requirements"])

    def test_periodic_sync_with_closed_checklist_and_active_milestone(self):
        for number in (1, 2):
            self.api.items[number]["state"] = "closed"
        for number in range(1000, 1100):
            item = self.api.make_issue(number, str(number), "", ["release"])
            item.update(
                state="closed",
                milestone={"number": number, "state": "closed", "title": str(number)},
            )
            self.api.items[number] = item
        self.api.add_pr(10, "Fixes #3")
        self.controller.handle({"schedule": "*/15 * * * *"}, "schedule")
        self.assertIn("| #3 | #10 (open)", self.api.items[1]["body"])
        self.assertLess(len(self.api.calls), 18)
        self.assertFalse(
            any(
                "/comments" in path or "sub_issues" in path and "issues/10" in path
                for _, path in self.api.calls
            )
        )

    def test_periodic_sync_skips_closed_milestones(self):
        self.api.milestones[0]["state"] = "closed"
        self.controller.handle({}, "schedule")
        self.assertEqual(self.api.calls, [("LIST", "milestones?state=open")])

    def test_periodic_sync_skips_partial_initialization(self):
        del self.api.parents[2]
        self.controller.handle({}, "schedule")
        self.assertFalse(
            any("closedByPullRequestsReferences" in path for _, path in self.api.calls)
        )

    def test_existing_pr_milestone_is_preserved(self):
        pr = self.api.add_pr(10, "Fixes #3")
        pr["milestone"] = {"number": 9, "title": "other"}
        self.controller.reconcile(1)
        self.assertEqual(pr["milestone"]["number"], 9)

    def test_ambiguous_association_is_filled_when_target_becomes_unique(self):
        self.api.items[4]["milestone"] = {"number": 9, "title": "other"}
        pr = self.api.add_pr(10, "Fixes #3, fixes #4")
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])
        self.api.native_links[4] = []
        self.controller.reconcile(1)
        self.assertEqual(pr["milestone"]["number"], 1)

    def test_native_pagination_is_complete_on_both_sides(self):
        self.api.page_size = 2
        for number in range(10, 14):
            self.api.add_pr(number, "Fixes #3, fixes #4")
        extra = self.api.make_issue(8, "Extra", "", [])
        extra["milestone"] = self.api.milestones[0]
        self.api.items[8] = extra
        self.api.native_links[8] = [(10, REPOSITORY)]
        state = self.controller.reconcile(1)
        self.assertEqual(set(state["prs"]), {"10", "11", "12", "13"})
        self.assertEqual(state["prs"]["10"]["requirements"], [3, 4, 8])

    def test_ninety_merged_pr_check_fits_budget_without_comment_reads(self):
        for number in (1, 2, 3, 4):
            self.api.items[number]["state"] = "closed"
        for number in range(100, 190):
            self.api.add_pr(number, "Fixes #3").update(
                state="closed",
                merged_at="2026-11-20",
                milestone=self.api.milestones[0],
            )
        self.controller.reconcile(1)
        self.api.enforce = True
        self.api.calls.clear()
        _, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertLess(len(self.api.calls), 30)
        self.assertFalse(any("/comments" in path for _, path in self.api.calls))
        self.assertFalse(any(method == "PATCH" for method, _ in self.api.calls))

    def test_ninety_new_prs_can_sync_milestones_and_projects_in_one_run(self):
        for number in (3, 4):
            self.api.projects[f"I_{number}"] = {"P_FEATURE": f"ITEM_{number}"}
        for number in range(100, 190):
            self.api.add_pr(number, f"Fixes #{3 + number % 2}")
        self.api.enforce = True

        self.controller.handle({}, "schedule")
        self.assertLess(len(self.api.calls), 240)
        for number in range(100, 190):
            self.assertEqual(self.api.prs[number]["milestone"]["number"], 1)
            self.assertIn(
                "P_FEATURE", self.api.projects[self.api.prs[number]["node_id"]]
            )

        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertLess(len(self.api.calls), 30)
        self.assertFalse(
            any(
                method == "PATCH" or "addProjectV2ItemById" in path
                for method, path in self.api.calls
            )
        )

    def test_one_failed_release_does_not_starve_another(self):
        self.api.items[50] = self.api.make_issue(
            50, "Second release", PLAN.replace("0.12.0", "0.13.0"), ["release"]
        )
        self.controller.initialize_or_update(50, "release_init")
        self.api.items[52]["state"] = "closed"
        pages = self.api.pages

        def fail_first(path):
            if path == "issues/2/sub_issues":
                raise OperationError("archive unavailable")
            return pages(path)

        with patch.object(self.api, "pages", side_effect=fail_first):
            with self.assertRaisesRegex(
                OperationError, "Release #1: archive unavailable"
            ):
                self.controller.handle({}, "schedule")
        self.assertIn("| #52 | - | @harrisonyhq | closed", self.api.items[50]["body"])

    def test_budget_stops_before_extra_request(self):
        api = GitHub(REPOSITORY)
        api.max_calls = 1
        api.budget()
        with self.assertRaisesRegex(OperationError, "API/time budget"):
            api.budget()


if __name__ == "__main__":
    unittest.main()
