"""Request-count and recovery contracts at repository scale."""

import copy
import re
import unittest
from unittest.mock import Mock, patch

from controller import Controller, GitHub
from reference_index import MANIFEST, ReferenceIndex, ReferenceStore, empty_index, pack
from syntax import OperationError
from test_controller import PLAN, REPOSITORY, MemoryGitHub


class CountingGitHub(MemoryGitHub):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.index_reads = 0
        self.index_writes = 0
        self.fail_manifest = False

    def rest(self, path, method="GET", data=None):
        self.calls.append((method, path))
        if path.startswith("labels/"):
            return {"name": path.split("/")[-1]}
        if (
            self.fail_manifest
            and method == "PATCH"
            and (data or {}).get("body", "").startswith(MANIFEST)
        ):
            self.fail_manifest = False
            raise OperationError("Interrupted before manifest publication")
        return super().rest(path, method, data)

    def pages(self, path):
        self.calls.append(("LIST", path))
        if path == "pulls?state=all":
            raise AssertionError("Unbounded historical PR scans are forbidden")
        return super().pages(path)

    def index_read(self):
        self.index_reads += 1
        return super().index_read()

    def index_write(self, value):
        self.index_writes += 1
        return super().index_write(value)

    def discussion_reads(self):
        return [
            p
            for method, p in self.calls
            if method == "LIST"
            and re.fullmatch(
                r"(?:issues/\d+/comments|pulls/\d+/(?:reviews|comments))", p
            )
            and int(p.split("/")[1]) in self.prs
        ]


class RepositoryScaleContracts(unittest.TestCase):
    def setUp(self):
        self.api = CountingGitHub()
        self.config = {
            "repository": REPOSITORY,
            "projects": {},
            "sync": {"prs_per_run": 10},
        }
        self.controller = Controller(self.api, self.config)

    def history(self, count=1519, created_at="2026-02-01T00:00:00Z"):
        for number in range(10, count + 10):
            pr = self.api.add_pr(number, "Historical change")
            pr["created_at"] = created_at
            pr["updated_at"] = created_at

    def test_1519_pr_backfill_scans_ten_then_resumes_at_eleven(self):
        self.history()
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(len(self.api.discussion_reads()), 30)
        self.assertEqual(self.api.index_data["bootstrap"]["offset"], 10)
        self.api.calls.clear()
        Controller(self.api, self.config).handle({}, "schedule")
        self.assertEqual(len(self.api.discussion_reads()), 30)
        self.assertEqual(self.api.index_data["bootstrap"]["offset"], 20)
        self.assertEqual(len(self.api.index_data["prs"]), 20)
        self.assertLess(len(self.api.calls), 90)

    def test_pr_event_reads_only_its_discussions_and_preserves_backfill_cursor(self):
        self.history()
        self.controller.initialize_or_update(1, "release_init")
        requirement = next(iter(self.controller.load(1)[1]["scope"].values()))["number"]
        self.api.prs[1500]["body"] = f"#{requirement}"
        self.api.calls.clear()
        self.controller.handle(
            {"pull_request": {"number": 1500}, "action": "edited"},
            "pull_request_target",
        )
        self.assertEqual(len(self.api.discussion_reads()), 3)
        self.assertEqual(
            {int(p.split("/")[1]) for p in self.api.discussion_reads()}, {1500}
        )
        self.assertEqual(self.api.index_data["bootstrap"]["offset"], 10)
        self.assertEqual(
            self.controller.load(1)[1]["prs"]["1500"]["requirements"], [requirement]
        )

    def test_review_event_refreshes_one_pr_instead_of_all_history(self):
        self.history()
        self.controller.initialize_or_update(1, "release_init")
        requirement = next(iter(self.controller.load(1)[1]["scope"].values()))["number"]
        self.api.prs[1500]["body"] = f"#{requirement}"
        self.api.calls.clear()
        self.controller.handle(
            {
                "workflow_run": {
                    "conclusion": "success",
                    "path": ".github/workflows/release-review-observer.yml",
                    "pull_requests": [{"number": 1500}],
                }
            },
            "workflow_run",
        )
        self.assertEqual(len(self.api.discussion_reads()), 3)

    def test_unrelated_issue_and_uninitialized_repository_skip_discovery(self):
        self.history()
        self.controller.handle({}, "schedule")
        self.assertEqual(self.api.index_reads, 0)
        self.assertEqual(self.api.discussion_reads(), [])
        self.controller.initialize_or_update(1, "release_init")
        self.api.calls.clear()
        before = self.api.index_reads
        self.controller.handle(
            {"issue": {"number": 9000, "labels": [], "body": "Unrelated"}}, "issues"
        )
        self.assertEqual(self.api.index_reads, before)
        self.assertEqual(self.api.discussion_reads(), [])

    def test_incomplete_history_blocks_release_readiness(self):
        self.history()
        self.controller.initialize_or_update(1, "release_init")
        for entry in self.controller.load(1)[1]["scope"].values():
            self.api.items[entry["number"]]["state"] = "closed"
        _, report = self.controller.release_check(1)
        self.assertFalse(report["reference_index"]["complete"])
        self.assertEqual(report["status"], "blocked")

    def test_existing_release_prs_cannot_bypass_the_discovery_batch_limit(self):
        self.history(500)
        self.controller.initialize_or_update(1, "release_init")
        _, state = self.controller.load(1)
        requirement = next(iter(state["scope"].values()))["number"]
        state["prs"] = {
            str(n): {
                "requirements": [requirement],
                "bugs": [],
                "merged": False,
                "state": "open",
                "mode": "auto",
                "last": None,
            }
            for n in range(10, 510)
        }
        self.controller.save(1, state, "Seed an existing version's PR ownership")
        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertEqual(len(self.api.discussion_reads()), 30)
        self.assertGreater(len(self.api.index_data["pending"]), 400)
        self.assertFalse(self.controller.load(1)[1]["reference_index"]["complete"])
        self.api.calls.clear()
        _, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "blocked")
        self.assertLess(
            sum(
                method == "GET" and bool(re.fullmatch(r"pulls/\d+", path))
                for method, path in self.api.calls
            ),
            50,
        )

    def test_steady_poll_does_not_rewrite_release_objects(self):
        self.controller.initialize_or_update(1, "release_init")
        self.controller.handle({}, "schedule")
        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertEqual(
            [
                p
                for method, p in self.api.calls
                if method in ("POST", "PATCH", "DELETE")
            ],
            [],
        )

    def test_completed_releases_are_rotated_instead_of_all_reconciled(self):
        catalog = {
            n: {
                "initialized": True,
                "milestone": {"number": n, "state": "closed"},
                "scope": {},
                "bugs": {},
                "prs": {},
            }
            for n in range(1, 101)
        }
        seen = []
        for _ in range(2):
            controller = Controller(self.api, self.config)
            controller.releases = lambda: list(catalog)
            controller.load = lambda n: (None, copy.deepcopy(catalog[n]))
            controller.reconcile = lambda n, event: seen.append(n)
            controller.reconcile_all(replay_commands=False)
        self.assertEqual(seen, [1, 2])

    def test_api_budget_stops_before_excess_request(self):
        api = GitHub(REPOSITORY)
        api.max_calls = 1
        response = Mock(returncode=0, stdout="{}", stderr="")
        with patch("controller.subprocess.run", return_value=response) as transport:
            api.rest("")
            with self.assertRaisesRegex(OperationError, "budget"):
                api.rest("")
            self.assertEqual(transport.call_count, 1)

    def test_mutations_are_paced_without_delaying_reads(self):
        response = Mock(returncode=0, stdout="{}", stderr="")
        with (
            patch("controller.time.monotonic", return_value=100),
            patch("controller.time.sleep") as sleep,
            patch("controller.subprocess.run", return_value=response),
        ):
            api = GitHub(REPOSITORY)
            api.rest("issues/1", "PATCH", {"state": "closed"})
            api.rest("issues/1")
            sleep.assert_not_called()
            api.rest("issues/2", "PATCH", {"state": "closed"})
            sleep.assert_called_once_with(1)

    def test_background_replays_initialization_when_its_event_was_missed(self):
        command = self.api.comment(1, "/release_init")
        self.controller.handle({}, "schedule")
        state = self.controller.load(1)[1]
        self.assertTrue(state["initialized"])
        self.assertEqual(state["command_cursor"], command["id"])


class ReleaseWindowContracts(unittest.TestCase):
    setUp = RepositoryScaleContracts.setUp
    history = RepositoryScaleContracts.history

    def test_1519_older_prs_need_no_discussion_reads_and_do_not_block_release(self):
        self.history(created_at="2020-01-01T00:00:00Z")
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(self.api.discussion_reads(), [])
        state = self.controller.load(1)[1]
        self.assertTrue(state["reference_index"]["complete"])
        for entry in state["scope"].values():
            self.api.items[entry["number"]]["state"] = "closed"
        _, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["reference_index"]["since"], "2026-01-01T00:00:00Z")
        self.assertEqual(self.api.discussion_reads(), [])

    def test_bulk_updates_to_old_prs_do_not_reopen_the_discovery_window(self):
        self.history(created_at="2020-01-01T00:00:00Z")
        self.controller.initialize_or_update(1, "release_init")
        for pr in self.api.prs.values():
            pr["updated_at"] = "9999-01-01T00:00:00Z"
        for entry in self.controller.load(1)[1]["scope"].values():
            self.api.items[entry["number"]]["state"] = "closed"
        self.api.calls.clear()
        _, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertFalse(any("sort=updated" in path for _, path in self.api.calls))

    def test_new_pr_is_discovered_after_an_empty_window_without_a_webhook(self):
        self.history(created_at="2020-01-01T00:00:00Z")
        self.controller.initialize_or_update(1, "release_init")
        requirement = next(iter(self.controller.load(1)[1]["scope"].values()))["number"]
        pr = self.api.add_pr(5000, f"#{requirement}")
        pr.update(created_at="2026-02-01T00:00:00Z", updated_at="2026-02-01T00:00:00Z")
        self.api.calls.clear()
        self.controller.handle({}, "schedule")
        self.assertEqual(len(self.api.discussion_reads()), 3)
        self.assertEqual(
            self.controller.load(1)[1]["prs"]["5000"]["requirements"], [requirement]
        )

    def test_removing_requirement_keeps_the_release_window_without_rereading(self):
        self.prepare_requirement_pair()
        self.api.calls.clear()
        body = self.api.items[1]["body"]
        begin = body.index("- Connector v2")
        end = body.index("- Store interface")
        self.api.items[1]["body"] = body[:begin] + body[end:]
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertEqual(self.api.index_data["since"], self.api.items[1]["created_at"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [4])

    def test_old_pr_edits_comments_reviews_and_native_links_skip_discussions(self):
        self.history(1, created_at="2020-01-01T00:00:00Z")
        self.controller.initialize_or_update(1, "release_init")
        requirement = next(iter(self.controller.load(1)[1]["scope"].values()))["number"]
        self.api.prs[10]["body"] = f"#{requirement}"
        self.api.prs[10]["updated_at"] = "9999-01-01T00:00:00Z"
        self.api.calls.clear()
        self.controller.handle(
            {"pull_request": {"number": 10}, "action": "edited"}, "pull_request_target"
        )
        comment = self.api.comment(10, f"#{requirement}")
        self.controller.handle(
            {
                "issue": {"number": 10, "pull_request": {}},
                "comment": comment,
                "sender": comment["user"],
                "action": "created",
            },
            "issue_comment",
        )
        self.controller.handle(
            {
                "workflow_run": {
                    "conclusion": "success",
                    "path": ".github/workflows/release-review-observer.yml",
                    "pull_requests": [{"number": 10}],
                }
            },
            "workflow_run",
        )
        self.api.native_links[requirement] = [(10, REPOSITORY)]
        self.controller.handle({}, "schedule")
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertNotIn("10", self.controller.load(1)[1]["prs"])
        self.assertIsNone(self.api.prs[10]["milestone"])

    def existing_requirements(self):
        self.api.items[1]["body"] = PLAN.replace("[]", "#3", 1).replace("[]", "#4", 1)
        for number, created in (
            (3, "2026-03-01T00:00:00Z"),
            (4, "2026-05-01T00:00:00Z"),
        ):
            item = self.api.make_issue(
                number, f"Requirement {number}", "", ["requirement"]
            )
            item["created_at"] = created
            self.api.items[number] = item

    def prepare_requirement_pair(self):
        self.existing_requirements()
        pr = self.api.add_pr(10, "#3 #4")
        pr.update(created_at="2026-04-01T00:00:00Z", updated_at="2026-04-01T00:00:00Z")
        same = self.api.add_pr(11, "#4")
        same.update(
            created_at="2026-05-01T00:00:00Z", updated_at="2026-05-01T00:00:00Z"
        )
        self.api.projects["I_3"] = {"P_EARLY": "E"}
        self.api.projects["I_4"] = {"P_LATE": "L"}
        self.controller.initialize_or_update(1, "release_init")
        return self.controller.load(1)[1]

    def test_later_requirement_is_linked_to_an_earlier_pr_in_the_release_window(self):
        state = self.prepare_requirement_pair()
        self.assertEqual(state["prs"]["10"]["requirements"], [3, 4])
        self.assertEqual(state["prs"]["11"]["requirements"], [4])
        self.assertEqual(set(self.api.projects["I_10"]), {"P_EARLY", "P_LATE"})
        self.assertEqual(set(self.api.projects["I_11"]), {"P_LATE"})
        self.assertEqual(self.api.index_data["since"], self.api.items[1]["created_at"])

    def test_comment_links_a_later_requirement_to_an_existing_pr(self):
        self.existing_requirements()
        self.api.projects["I_4"] = {"P_LATE": "L"}
        self.api.add_pr(10, "Implementation").update(
            created_at="2026-04-01T00:00:00Z", updated_at="2026-04-01T00:00:00Z"
        )
        self.controller.initialize_or_update(1, "release_init")
        comment = self.api.comment(10, "#4")
        self.controller.handle(
            {
                "issue": {"number": 10, "pull_request": {}},
                "comment": comment,
                "sender": comment["user"],
                "action": "created",
            },
            "issue_comment",
        )
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [4])
        self.assertIn("P_LATE", self.api.projects["I_10"])

    def test_development_links_a_later_requirement_to_an_existing_pr(self):
        self.existing_requirements()
        self.api.add_pr(10, "Implementation").update(
            created_at="2026-04-01T00:00:00Z", updated_at="2026-04-01T00:00:00Z"
        )
        self.controller.initialize_or_update(1, "release_init")
        self.api.native_links[4] = [(10, REPOSITORY)]
        self.controller.handle({}, "schedule")
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [4])

    def test_reference_to_a_future_release_does_not_create_a_target_conflict(self):
        self.existing_requirements()
        self.api.add_pr(10, "#3 #20").update(
            created_at="2026-04-01T00:00:00Z", updated_at="2026-04-01T00:00:00Z"
        )
        for number in (20, 21):
            item = self.api.make_issue(
                number, "Future requirement", "", ["requirement"]
            )
            item["created_at"] = "2026-08-01T00:00:00Z"
            self.api.items[number] = item
        self.controller.initialize_or_update(1, "release_init")
        future = self.api.make_issue(
            15,
            "Future release",
            PLAN.replace("0.12.0", "0.13.0")
            .replace("[]", "#20", 1)
            .replace("[]", "#21", 1),
            ["release"],
        )
        future["created_at"] = "2026-07-01T00:00:00Z"
        self.api.items[15] = future
        self.controller.initialize_or_update(15, "release_init")
        self.controller.reconcile(1)
        record = self.controller.load(1)[1]["prs"]["10"]
        self.assertEqual(record["requirements"], [3])
        self.assertFalse(record["conflict"])

    def test_adding_existing_requirement_reuses_the_fixed_release_window(self):
        self.api.items[1]["created_at"] = "2024-01-01T00:00:00Z"
        self.history(1, created_at="2025-02-01T00:00:00Z")
        self.api.prs[10]["body"] = "#5"
        earlier = self.api.make_issue(5, "Earlier requirement", "", ["requirement"])
        earlier["created_at"] = "2025-01-01T00:00:00Z"
        self.api.items[5] = earlier
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(len(self.api.discussion_reads()), 3)
        self.api.calls.clear()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "### Change Log",
            "- Earlier requirement #5\n\n  description:\n    Earlier work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log",
        )
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.index_data["since"], self.api.items[1]["created_at"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["requirements"], [5])
        self.assertEqual(self.api.discussion_reads(), [])

    def test_release_creation_is_the_floor_for_early_existing_requirements(self):
        self.existing_requirements()
        for number in (3, 4):
            self.api.items[number]["created_at"] = "2020-01-01T00:00:00Z"
        self.history(created_at="2025-01-01T00:00:00Z")
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertEqual(self.api.index_data["since"], self.api.items[1]["created_at"])
        for number in (3, 4):
            self.api.items[number]["state"] = "closed"
        _, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        # An explicitly assigned old PR still participates in delivery checks.
        self.api.prs[1500]["milestone"] = self.api.milestones[0]
        _, report = self.controller.release_check(1)
        self.assertIn(1500, report["unmerged_prs"])
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertNotIn("1500", self.controller.load(1)[1]["prs"])

    def test_pairing_uses_only_release_creation_time(self):
        self.existing_requirements()
        self.api.items[1]["created_at"] = "2026-04-01T00:00:00Z"
        self.api.add_pr(9, "#3").update(
            created_at="2026-03-15T00:00:00Z", updated_at="2026-03-15T00:00:00Z"
        )
        self.api.add_pr(10, "#3 #4").update(
            created_at="2026-04-01T00:00:00Z", updated_at="2026-04-01T00:00:00Z"
        )
        self.api.add_pr(11, "#4").update(
            created_at="2026-05-01T00:00:00Z", updated_at="2026-05-01T00:00:00Z"
        )
        self.controller.initialize_or_update(1, "release_init")
        state = self.controller.load(1)[1]
        self.assertEqual(state["prs"]["10"]["requirements"], [3, 4])
        self.assertEqual(state["prs"]["11"]["requirements"], [4])
        self.assertNotIn("9", state["prs"])
        self.assertEqual(len(self.api.discussion_reads()), 6)
        self.api.native_links[3] = [(9, REPOSITORY)]
        self.api.calls.clear()
        self.controller.handle(
            {"pull_request": {"number": 9}, "action": "edited"}, "pull_request_target"
        )
        self.controller.handle({}, "schedule")
        self.assertNotIn("9", self.controller.load(1)[1]["prs"])
        self.assertEqual(self.api.discussion_reads(), [])

    def test_windows_are_independent_even_when_an_older_release_uses_the_batch(self):
        self.history(created_at="2020-02-01T00:00:00Z")
        index = ReferenceIndex(self.api, lambda pr: {})
        index.select(1, "2020-01-01T00:00:00Z")
        index.advance_once()
        self.assertFalse(index.status()["complete"])
        index.select(2, "2026-01-01T00:00:00Z")
        index.advance_once()
        self.assertTrue(index.status()["complete"])
        self.assertEqual(index.status()["indexed_prs"], 0)
        self.assertEqual(index.scanned, 10)
        resumed = ReferenceIndex(self.api, lambda pr: {})
        resumed.select(1, "2020-01-01T00:00:00Z")
        resumed.advance_once()
        self.assertEqual(resumed.status()["indexed_prs"], 20)

    def test_legacy_global_backfill_is_replaced_by_the_current_issue_window(self):
        self.history(created_at="2020-01-01T00:00:00Z")
        legacy = empty_index()
        legacy["bootstrap"]["offset"] = 20
        legacy["pending"] = list(range(10, 1519))
        self.api.index_data = legacy
        self.controller.initialize_or_update(1, "release_init")
        self.assertTrue(self.controller.load(1)[1]["reference_index"]["complete"])
        self.assertEqual(self.api.discussion_reads(), [])
        self.assertEqual(self.api.index_data["pending"], [])

    def test_metadata_only_native_candidate_fetches_body_before_indexing(self):
        self.controller.initialize_or_update(1, "release_init")
        pr = self.api.add_pr(10, "#4")
        pr.update(created_at="2026-02-01T00:00:00Z", updated_at="2026-02-01T00:00:00Z")
        index = ReferenceIndex(self.api, self.controller.discussion_references)
        index.select(1, "2026-01-01T00:00:00Z")
        index.data["bootstrap"]["empty"] = False
        index.note(pr)
        index.flush()
        self.api.native_links[3] = [(10, REPOSITORY)]
        self.controller.handle({}, "schedule")
        self.assertEqual(
            set(self.controller.load(1)[1]["prs"]["10"]["requirements"]), {3, 4}
        )

    def test_target_bug_before_new_requirements_keeps_its_fix_in_release_checks(self):
        self.config["projects"]["Bugfix"] = {
            "id": "P_BUG",
            "fields": {
                "Status": {
                    "id": "STATUS",
                    "options": [
                        {"name": name, "id": name}
                        for name in ("Ready", "In progress", "In review", "Done")
                    ],
                }
            },
        }
        self.controller.initialize_or_update(1, "release_init")
        bug = self.api.make_issue(8, "Earlier source Bug", "", ["bug"])
        bug.update(created_at="2026-02-01T00:00:00Z", state="closed")
        self.api.items[8] = bug
        self.api.parents[8] = 2
        self.api.add_pr(10, "#8").update(
            created_at="2026-03-01T00:00:00Z", updated_at="2026-03-01T00:00:00Z"
        )
        self.api.add_pr(11, "Unrelated").update(
            created_at="2026-05-01T00:00:00Z", updated_at="2026-05-01T00:00:00Z"
        )
        self.api.items[15] = self.api.make_issue(
            15,
            "Next release",
            PLAN.replace("0.12.0", "0.13.0")
            .replace("[]", "#20", 1)
            .replace("[]", "#21", 1),
            ["release"],
        )
        for number in (20, 21):
            self.api.items[number] = self.api.make_issue(
                number, "Next feature", "", ["requirement"]
            )
            self.api.items[number].update(
                created_at="2026-04-01T00:00:00Z", state="closed"
            )
        self.controller.initialize_or_update(15, "release_init")
        bug["milestone"] = self.controller.load(15)[1]["milestone"]
        self.api.index_data["prs"]["11"]["checked_at"] = "2026-01-01T00:00:00Z"
        self.config["sync"]["prs_per_run"] = 1
        _, report = self.controller.release_check(15)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(
            report["reference_index"]["since"], self.api.items[15]["created_at"]
        )
        _, report = self.controller.release_check(15)
        self.assertIn(10, report["unmerged_prs"])


class IndexRecoveryContracts(unittest.TestCase):
    def test_partial_discussion_failure_saves_completed_cursor(self):
        api = CountingGitHub()
        for n in range(10, 20):
            api.add_pr(n, "")
            api.prs[n]["updated_at"] = "2020-01-01T00:00:00Z"

        def read(pr):
            if pr["number"] == 16:
                raise OperationError("Interrupted discussion read")
            return {3: [pr["html_url"]]}

        index = ReferenceIndex(api, read)
        with self.assertRaisesRegex(OperationError, "Interrupted discussion"):
            index.advance()
        self.assertEqual(set(api.index_data["prs"]), {"19", "18", "17"})
        self.assertEqual(api.index_data["bootstrap"]["offset"], 3)
        index = ReferenceIndex(api, lambda pr: {})
        index.advance()
        self.assertTrue(index.status()["complete"])
        self.assertEqual(len(api.index_data["prs"]), 10)

    def test_interrupted_shard_write_leaves_committed_manifest_readable(self):
        api = CountingGitHub()
        store = ReferenceStore(api)
        data = store.load()
        data["prs"]["10"] = {
            "refs": {"3": ["original"]},
            "checked_at": "2026-01-01T00:00:00Z",
        }
        store.write(data)
        changed = copy.deepcopy(data)
        changed["prs"]["10"]["refs"] = {"4": ["replacement"]}
        changed["bootstrap"]["offset"] = 1
        api.fail_manifest = True
        with self.assertRaisesRegex(OperationError, "Interrupted before manifest"):
            store.write(changed)
        recovered = ReferenceStore(api)
        self.assertEqual(recovered.load(), data)
        recovered.write(changed)
        self.assertEqual(ReferenceStore(api).load(), changed)

    def test_untrusted_index_comments_are_ignored_and_oversize_pack_fails_early(self):
        api = CountingGitHub()
        store = ReferenceStore(api)
        data = store.load()
        store.write(data)
        api.comment(store.issue["number"], MANIFEST + "\n" + pack({"prs": {"999": {}}}))
        self.assertEqual(ReferenceStore(api).load(), data)
        with self.assertRaisesRegex(OperationError, "decoded size"):
            pack({"value": "x" * 5_000_000})


if __name__ == "__main__":
    unittest.main()
