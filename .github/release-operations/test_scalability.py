"""Bounded native reads, lightweight routing, and first-association contracts."""

import copy
import re
import unittest
from unittest.mock import Mock, patch

from controller import Controller, GitHub
from directory import MARKER, pack, unpack
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


class LightweightContracts(unittest.TestCase):
    def setUp(self):
        self.api = CountingGitHub()
        self.config = {
            "repository": REPOSITORY,
            "projects": {},
            "sync": {"max_api_calls": 10000},
        }
        self.controller = Controller(self.api, self.config)
        self.controller.initialize_or_update(1, "release_init")

    def add_history(self, count=100):
        directory = self.controller.directory()
        for number in range(10000, 10000 + count):
            directory.data["releases"][str(number)] = {
                "version": str(number),
                "milestone": number,
                "archive": number + 100000,
                "active": False,
                "issues": [],
                "prs": [],
                "operation": {},
            }
        directory.write()
        self.api.calls.clear()

    def second_release(self):
        self.api.items[50] = self.api.make_issue(
            50, "Second release", PLAN.replace("0.12.0", "0.13.0"), ["release"]
        )
        self.controller.initialize_or_update(50, "release_init")
        state = self.controller.load(50)[1]
        return next(iter(state["scope"].values()))["number"]

    def test_persistent_archive_error_does_not_starve_later_release(self):
        requirement = self.second_release()
        archive = self.controller.load(1)[1]["archive"]
        pages = self.api.pages

        def fail_archive(path):
            if path == f"issues/{archive}/sub_issues":
                raise OperationError("HTTP 404: archive unavailable")
            return pages(path)

        with patch.object(self.api, "pages", side_effect=fail_archive):
            for status in ("closed", "open"):
                self.api.items[requirement]["state"] = status
                with self.assertRaisesRegex(OperationError, "Release #1:.*archive"):
                    self.controller.handle({}, "schedule")
                state = self.controller.load(50)[1]
                self.assertEqual(state["observed"][str(requirement)]["state"], status)
                self.assertIn("archive unavailable", self.api.items[1]["body"])

    def test_milestone_error_is_aggregated_after_other_versions_sync(self):
        requirement = self.second_release()
        self.api.items[requirement]["state"] = "closed"
        rest = self.api.rest

        def fail_milestone(path, method="GET", data=None):
            if path == "milestones/1" and method == "GET":
                raise OperationError("HTTP 404: milestone unavailable")
            return rest(path, method, data)

        with patch.object(self.api, "rest", side_effect=fail_milestone):
            with self.assertRaisesRegex(OperationError, "Release #1:.*milestone"):
                self.controller.handle({}, "schedule")
        self.assertEqual(
            self.controller.load(50)[1]["observed"][str(requirement)]["state"], "closed"
        )

    def test_failure_receipt_error_does_not_prevent_other_versions(self):
        requirement = self.second_release()
        self.api.items[requirement]["state"] = "closed"
        reconcile, save = self.controller.reconcile, self.controller.save

        def fail_sync(release, event=None):
            if release == 1:
                raise OperationError("Project configuration changed")
            return reconcile(release, event)

        def fail_receipt(release, state, result):
            if release == 1:
                raise OperationError("Receipt write failed")
            return save(release, state, result)

        with (
            patch.object(self.controller, "reconcile", side_effect=fail_sync),
            patch.object(self.controller, "save", side_effect=fail_receipt),
        ):
            with self.assertRaisesRegex(
                OperationError,
                "Project configuration changed; failure report unavailable: Receipt write failed",
            ):
                self.controller.handle({}, "schedule")
        self.assertEqual(
            self.controller.load(50)[1]["observed"][str(requirement)]["state"], "closed"
        )

    def test_unrelated_issue_does_not_load_historical_states(self):
        self.add_history()
        self.controller.handle({"issue": {"number": 9000, "labels": []}}, "issues")
        self.assertEqual(len(self.api.calls), 1)
        self.assertIn("release-directory", self.api.calls[0][1])

    def test_schedule_loads_only_active_release(self):
        self.add_history()
        self.controller.handle({}, "schedule")
        comments = [
            p
            for kind, p in self.api.calls
            if kind == "LIST" and p.endswith("/comments")
        ]
        self.assertEqual(comments, ["issues/1/comments"])

    def test_delivery_lookup_uses_directory_without_loading_states(self):
        self.add_history()
        with self.controller.snapshots():
            self.assertEqual(self.controller.delivery_release("v0.12.0"), 1)
        self.assertEqual(len(self.api.calls), 1)

    def test_manual_check_loads_only_requested_release(self):
        self.add_history()
        self.controller.release_check(1)
        self.assertFalse(any("issues/10000" in p for _, p in self.api.calls))

    def test_archived_release_is_refreshed_only_by_explicit_commands(self):
        self.api.milestones[0]["state"] = "closed"
        self.controller.reconcile(1)
        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertFalse(any(p.endswith("/comments") for _, p in self.api.calls))
        self.controller.release_check(1)
        self.assertTrue(any(p == "issues/1/comments" for _, p in self.api.calls))

    def test_old_pr_is_accepted_through_explicit_native_link(self):
        self.api.add_pr(10, "").update(created_at="2000-01-01T00:00:00Z")
        self.api.native_links[3] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [3])

    def test_existing_pr_milestone_is_preserved(self):
        pr = self.api.add_pr(10, "Fixes #3")
        pr["milestone"] = {"number": 9, "title": "other"}
        self.controller.reconcile(1)
        self.assertEqual(pr["milestone"]["number"], 9)

    def test_clear_unlink_relink_and_retarget_never_refill_milestone(self):
        pr = self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        self.assertEqual(pr["milestone"]["number"], 1)
        pr["milestone"] = None
        self.api.native_links.clear()
        self.controller.reconcile(1)
        self.api.items[4]["milestone"] = {"number": 9, "title": "other"}
        self.api.native_links[4] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [4])

    def test_ambiguous_first_association_is_not_filled_later(self):
        self.api.items[4]["milestone"] = {"number": 9, "title": "other"}
        pr = self.api.add_pr(10, "Fixes #3, fixes #4")
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])
        self.api.native_links[4] = []
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])

    def test_pr_event_routes_by_native_links_without_reading_discussions(self):
        self.add_history()
        self.api.add_pr(10, "Fixes #3")
        self.controller.handle(
            {"pull_request": {"number": 10}, "action": "opened"}, "pull_request_target"
        )
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [3])
        self.assertFalse(
            any(
                re.match(r"issues/10/comments|pulls/10", path)
                for _, path in self.api.calls
            )
        )

    def test_native_pagination_is_complete_on_both_sides(self):
        self.api.page_size = 2
        for n in range(10, 14):
            self.api.add_pr(n, "Fixes #3, fixes #4")
        extra = self.api.make_issue(8, "Extra", "", ["requirement"])
        extra["milestone"] = self.api.milestones[0]
        self.api.items[8] = extra
        self.api.native_links[8] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        state = self.controller.load(1)[1]
        self.assertEqual(set(state["prs"]), {"10", "11", "12", "13"})
        self.assertIn(8, self.controller._pr_issues[10])

    def test_ninety_merged_pr_check_fits_default_budget_and_repeats(self):
        for n in (3, 4):
            self.api.items[n]["state"] = "closed"
        for n in range(100, 190):
            self.api.add_pr(n, "Fixes #3").update(
                state="closed", merged_at="2026-11-20"
            )
        self.controller.reconcile(1)
        self.config["sync"]["max_api_calls"] = 200
        self.api.enforce = True
        counts = []
        for _ in range(2):
            self.api.calls.clear()
            _, report = self.controller.release_check(1)
            self.assertEqual(report["status"], "ready")
            self.assertLess(len(self.api.calls), 200)
            self.assertFalse(
                any(re.fullmatch(r"pulls/\d+", path) for _, path in self.api.calls)
            )
            counts.append(len(self.api.calls))
        self.assertLessEqual(counts[1], counts[0])

    def test_budget_stops_before_extra_request(self):
        api = GitHub(REPOSITORY)
        api.max_calls = 1
        response = Mock(returncode=0, stdout="{}", stderr="")
        with patch("controller.subprocess.run", return_value=response) as transport:
            api.rest("")
            with self.assertRaisesRegex(OperationError, "budget"):
                api.rest("")
            self.assertEqual(transport.call_count, 1)

    def test_directory_encoding_is_bounded(self):
        value = {
            "releases": {"1": {"version": "0.12.0"}},
            "seen_prs": list(range(10000)),
        }
        self.assertEqual(unpack(pack(value)), value)
        self.assertLess(len(pack(value)), 60000)
        self.assertTrue(MARKER.startswith("<!--"))

    def test_missing_project_status_reports_configuration_error(self):
        self.controller.config["projects"]["Bugfix"] = {
            "id": "P",
            "fields": {"Status": {"id": "S", "options": []}},
        }
        with self.assertRaisesRegex(OperationError, "missing Status option Done"):
            self.controller.project_item(self.api.items[3], "Bugfix", status="Done")

    def test_failed_native_query_keeps_previous_snapshot(self):
        self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        previous = copy.deepcopy(self.controller.load(1)[1]["prs"])
        self.api.native_failure = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.assertEqual(self.controller.load(1)[1]["prs"], previous)

    def test_new_bug_and_fix_pr_receive_target_in_the_same_refresh(self):
        self.config["projects"]["Bugfix"] = {
            "id": "P_BUG",
            "fields": {
                "Status": {
                    "id": "S",
                    "options": [
                        {"name": n, "id": n}
                        for n in ("Ready", "In progress", "In review", "Done")
                    ],
                }
            },
        }
        archive = self.controller.load(1)[1]["archive"]
        self.api.items[8] = self.api.make_issue(8, "Bug", "", ["bug"])
        self.api.parents[8] = archive
        pr = self.api.add_pr(10, "Fixes #8")
        self.controller.handle(
            {"pull_request": {"number": 10}, "action": "opened"}, "pull_request_target"
        )
        self.assertEqual(pr["milestone"]["number"], 1)
        self.controller.release_check(1)
        self.assertEqual(pr["milestone"]["number"], 1)
        self.assertIn(8, self.controller.load(1)[1]["release_check"]["open_bugs"])

    def test_archived_issue_event_does_not_refresh_the_version(self):
        self.api.milestones[0]["state"] = "closed"
        self.controller.reconcile(1)
        self.api.calls.clear()
        self.controller.handle(
            {"issue": self.api.items[1], "action": "edited"}, "issues"
        )
        self.assertEqual(len(self.api.calls), 1)

    def test_project_cleanup_preserves_another_managed_relationship(self):
        self.api.projects["I_3"] = {"P_LOCAL": "L"}
        self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        other = self.api.make_issue(8, "Other requirement", "", ["requirement"])
        other["milestone"] = {"number": 9, "title": "other"}
        self.api.items[8] = other
        self.api.projects["I_8"] = {"P_OTHER": "O"}
        directory = self.controller.directory()
        directory.data["releases"]["99"] = {
            "version": "other",
            "milestone": 9,
            "archive": 98,
            "active": False,
            "issues": [8],
            "prs": [],
            "operation": {},
        }
        directory.write()
        self.api.native_links.clear()
        self.api.native_links[8] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        self.assertEqual(set(self.api.projects["I_10"]), {"P_OTHER"})

    def test_defaulted_bug_blocks_the_first_completion_check(self):
        self.config["projects"]["Bugfix"] = {
            "id": "P_BUG",
            "fields": {
                "Status": {"id": "S", "options": [{"name": "Ready", "id": "R"}]}
            },
        }
        archive = self.controller.load(1)[1]["archive"]
        self.api.items[8] = self.api.make_issue(8, "Bug", "", [])
        self.api.parents[8] = archive
        _, report = self.controller.release_check(1)
        self.assertIn(8, report["open_bugs"])

    def test_missing_target_at_first_link_requires_manual_choice(self):
        self.api.items[4]["milestone"] = None
        pr = self.api.add_pr(10, "Fixes #3, fixes #4")
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])
        self.api.items[4]["milestone"] = self.api.milestones[0]
        self.controller.reconcile(1)
        self.assertIsNone(pr["milestone"])

    def test_steady_schedule_leaves_unchanged_metadata_unwritten(self):
        self.api.add_pr(10, "Fixes #3")
        self.controller.handle({}, "schedule")
        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertFalse(
            any(kind in ("POST", "PATCH", "DELETE") for kind, _ in self.api.calls)
        )

    def test_unrelated_pr_event_reads_no_release_checkpoints(self):
        self.add_history()
        self.api.add_pr(10, "Ordinary change")
        self.controller.handle(
            {"pull_request": {"number": 10}, "action": "opened"}, "pull_request_target"
        )
        self.assertEqual(len(self.api.calls), 2)
        self.assertFalse(any(path.endswith("/comments") for _, path in self.api.calls))
