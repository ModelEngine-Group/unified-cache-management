import unittest
from pathlib import Path

import test_controller as fixtures
from folding import fold_requirements
from syntax import OperationError, parse_release, patch_pull_requests, patch_references
from taskcards import task_requirements, tasks


class TaskCardContracts(unittest.TestCase):
    def requirement_guide(self):
        template = (
            Path(__file__).parent.parent / "ISSUE_TEMPLATE/900-release-checklist.yml"
        ).read_text()
        field = template.split("    id: requirements\n", 1)[1]
        value = field.split("      value: |\n", 1)[1].split("    validations:", 1)[0]
        return "\n".join(line[8:] if line else "" for line in value.splitlines())

    def test_commented_template_is_not_a_requirement_and_survives_writeback(self):
        guide = self.requirement_guide()
        body = task_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
            "### Requirements\n", "### Requirements\n\n" + guide + "\n"
        )
        expected = parse_release(body, fixtures.REPOSITORY)
        self.assertEqual(len(expected["entries"]), 2)
        entries = expected["entries"]
        updated = patch_references(
            body, {e["key"]: n for e, n in zip(entries, [14, 13])}, fixtures.REPOSITORY
        )
        updated = patch_pull_requests(updated, {}, fixtures.REPOSITORY)
        self.assertIn(guide, updated)
        self.assertEqual(
            [
                e["number"]
                for e in parse_release(updated, fixtures.REPOSITORY)["entries"]
            ],
            [14, 13],
        )

    def test_comment_only_template_needs_a_real_requirement(self):
        body = (
            fixtures.PLAN.split("### Requirements", 1)[0]
            + "### Requirements\n\n"
            + self.requirement_guide()
            + "\n### Change Log\n\nreason: Initial scope.\n"
        )
        self.assertFalse(parse_release(body, fixtures.REPOSITORY)["entries"])
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = body
        with self.assertRaises(OperationError):
            x.initialize()
        self.assertEqual(len(x.api.items), 1)

    def test_conversion_preserves_plan_and_is_idempotent(self):
        body = task_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        self.assertEqual(
            parse_release(body, fixtures.REPOSITORY),
            parse_release(fixtures.PLAN, fixtures.REPOSITORY),
        )
        self.assertEqual(task_requirements(body, fixtures.REPOSITORY), body)
        self.assertIn("- [ ] Connector v2\n  initial integration []", body)
        self.assertIn("  <summary>Details</summary>", body)

    def test_old_status_badges_are_removed_during_conversion(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
            "initial integration []</summary>",
            "initial integration [] <!-- ucm-issue-state:open --> 🟢 open</summary>",
        )
        converted = task_requirements(body, fixtures.REPOSITORY)
        self.assertNotIn("ucm-issue-state", converted)
        self.assertNotIn("🟢", converted)
        self.assertEqual(
            parse_release(converted, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )

    def test_writeback_keeps_checkbox_description_and_nested_lists(self):
        body = task_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
            "- [ ]", "- [x]", 1
        )
        entries = parse_release(body, fixtures.REPOSITORY)["entries"]
        body = patch_references(
            body, {e["key"]: n for e, n in zip(entries, [14, 13])}, fixtures.REPOSITORY
        )
        self.assertIn("- [x] Connector v2\n  initial integration #14", body)
        expected = parse_release(body, fixtures.REPOSITORY)
        updated = patch_pull_requests(
            body,
            {14: [{"number": 29, "state": "closed", "merged_at": "today"}]},
            fixtures.REPOSITORY,
        )
        self.assertIn("    - #29 | merged", updated)
        self.assertIn("  <summary>Details</summary>", updated)
        self.assertEqual(parse_release(updated, fixtures.REPOSITORY), expected)
        self.assertEqual(
            patch_pull_requests(
                updated,
                {14: [{"number": 29, "state": "closed", "merged_at": "today"}]},
                fixtures.REPOSITORY,
            ),
            updated,
        )

    def test_short_issue_reference_is_reused(self):
        body = task_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
            "- [ ] Connector v2\n  initial integration []", "- [ ] #14"
        )
        entry = parse_release(body, fixtures.REPOSITORY)["entries"][0]
        self.assertEqual(entry["number"], 14)
        self.assertEqual(
            parse_release(
                patch_references(body, {}, fixtures.REPOSITORY), fixtures.REPOSITORY
            )["entries"][0],
            entry,
        )

    def test_initialization_and_repeated_update_create_once(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = task_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        before = x.initialize()
        self.assertIn(
            "- [ ] Connector v2\n  initial integration #3", x.api.items[1]["body"]
        )
        self.assertNotIn("ucm-issue-state", x.api.items[1]["body"])
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(x.controller.reconcile(1)["scope"], before["scope"])
        self.assertEqual(len(x.api.items), 4)

    def test_conversion_of_existing_release_keeps_scope_revision(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        before = x.initialize()
        x.api.items[1]["body"] = task_requirements(
            x.api.items[1]["body"], fixtures.REPOSITORY
        )
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(x.controller.reconcile(1)["scope"], before["scope"])

    def test_checkbox_edit_and_issue_reopen_do_not_change_scope(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = task_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        before = x.initialize()
        for checked, state in [("x", "closed"), (" ", "open")]:
            x.api.items[1]["body"] = (
                x.api.items[1]["body"].replace("- [ ] Connector", "- [x] Connector")
                if checked == "x"
                else x.api.items[1]["body"].replace(
                    "- [x] Connector", "- [ ] Connector"
                )
            )
            x.api.items[3]["state"] = state
            x.controller.reconcile(1)
            self.assertEqual(x.controller.reconcile(1)["scope"], before["scope"])

    def test_malformed_details_are_rejected(self):
        body = task_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        for bad in [
            body.replace("  </details>", "", 1),
            body.replace(
                "  <summary>Details</summary>", "  <summary>Other</summary>", 1
            ),
            body.replace("  assignee:", "assignee:", 1),
        ]:
            with self.subTest(body=bad), self.assertRaises(OperationError):
                parse_release(bad, fixtures.REPOSITORY)
