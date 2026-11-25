import unittest
from html import escape
from unittest.mock import patch

import folding
import syntax
import test_controller as fixtures
from folding import fold_requirements
from presentation import OVERVIEW_START, release_body
from syntax import OperationError, parse_release, patch_pull_requests, patch_references


class FoldingContracts(unittest.TestCase):
    def test_layout_entry_count_mismatches_fail_before_writeback(self):
        flat = fixtures.PLAN
        folded = fold_requirements(flat, fixtures.REPOSITORY)
        cases = (
            (folding.patch_references, folded, None),
            (folding.patch_pull_requests, folded, "folding.patch_references"),
            (syntax.patch_references, flat, None),
            (syntax.patch_pull_requests, flat, "syntax.patch_references"),
            (folding.fold_requirements, flat, "syntax.patch_references"),
        )
        for function, body, bypass in cases:
            parsed = parse_release(body, fixtures.REPOSITORY)
            for entries in (
                parsed["entries"][:-1],
                parsed["entries"] + [parsed["entries"][0]],
            ):
                with self.subTest(function=function.__name__, entries=len(entries)):
                    arguments = (
                        (body, fixtures.REPOSITORY)
                        if function is folding.fold_requirements
                        else (body, {}, fixtures.REPOSITORY)
                    )
                    with patch(
                        "syntax.parse_release",
                        return_value=dict(parsed, entries=entries),
                    ):
                        if bypass:
                            with patch(bypass, return_value=body):
                                with self.assertRaisesRegex(
                                    OperationError, "count does not match"
                                ):
                                    function(*arguments)
                        else:
                            with self.assertRaisesRegex(
                                OperationError, "count does not match"
                            ):
                                function(*arguments)

    def test_last_requirements_section_preserves_managed_regions(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        prefix, rest = body.split("### Requirements", 1)
        requirements, change = rest.split("### Change Log", 1)
        body = prefix + "### Change Log" + change + "\n### Requirements" + requirements
        body = release_body(body, {"scope": {}, "bugs": {}})
        updated = patch_pull_requests(body, {}, fixtures.REPOSITORY)
        self.assertEqual(
            parse_release(updated, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )
        self.assertIn(OVERVIEW_START, updated)

    def test_folded_scope_addition_and_removal(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.initialize()
        x.api.items[1]["body"] = x.api.items[1]["body"].replace(
            "### Change Log",
            "<details>\n<summary>Additional feature</summary>\n\nassignee:\n\ndescription:\n  Additional work.\n\npull_requests:\n\n</details>\n\n### Change Log",
        )
        x.controller.initialize_or_update(1, "update")
        self.assertIn(
            "Additional feature #5</summary>",
            x.api.items[1]["body"],
        )
        begin = x.api.items[1]["body"].index("<details>\n<summary>Additional feature")
        end = x.api.items[1]["body"].index("</details>", begin) + len("</details>")
        x.api.items[1]["body"] = (
            x.api.items[1]["body"][:begin] + x.api.items[1]["body"][end:]
        )
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(x.api.items[5]["milestone"]["number"], 1)

    def test_conversion_preserves_scope_and_is_idempotent(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        self.assertEqual(
            parse_release(body, fixtures.REPOSITORY),
            parse_release(fixtures.PLAN, fixtures.REPOSITORY),
        )
        self.assertEqual(fold_requirements(body, fixtures.REPOSITORY), body)
        self.assertIn("<summary>Connector v2\ninitial integration []</summary>", body)

    def test_reference_formats_and_html_titles_remain_stable(self):
        for title in (
            "Integration & API <v2>",
            f"Integration [Issue](https://github.com/{fixtures.REPOSITORY}/issues/14)",
            f"Integration https://github.com/{fixtures.REPOSITORY}/issues/14",
            f"[GLM \\[API\\]](https://github.com/{fixtures.REPOSITORY}/issues/14)",
            "Integration #14",
        ):
            with self.subTest(title=title):
                body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY).replace(
                    "<summary>Connector v2\ninitial integration []</summary>",
                    "<summary>" + escape(title, quote=False) + "</summary>",
                )
                entry = parse_release(body, fixtures.REPOSITORY)["entries"][0]
                patched = patch_references(
                    body, {entry["key"]: 14}, fixtures.REPOSITORY
                )
                self.assertIn("#14</summary>", patched)
                self.assertEqual(
                    patch_references(patched, {}, fixtures.REPOSITORY), patched
                )
                self.assertEqual(
                    parse_release(patched, fixtures.REPOSITORY)["entries"][0]["title"],
                    entry["title"],
                )

    def test_pr_refresh_keeps_description_and_card_boundaries(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        entries = parse_release(body, fixtures.REPOSITORY)["entries"]
        body = patch_references(
            body, {e["key"]: n for e, n in zip(entries, [14, 13])}, fixtures.REPOSITORY
        )
        rows = {14: [{"number": 29, "state": "closed", "merged_at": "today"}]}
        updated = patch_pull_requests(body, rows, fixtures.REPOSITORY)
        self.assertIn("  - #29 | merged", updated)
        self.assertEqual(updated.count("</details>"), 2)
        self.assertEqual(
            parse_release(updated, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )
        self.assertEqual(
            patch_pull_requests(updated, rows, fixtures.REPOSITORY), updated
        )
        removed = patch_pull_requests(updated, {}, fixtures.REPOSITORY)
        self.assertNotIn("#29", removed)
        self.assertEqual(
            parse_release(removed, fixtures.REPOSITORY),
            parse_release(body, fixtures.REPOSITORY),
        )

    def test_invalid_cards_fail_before_creation(self):
        body = fold_requirements(fixtures.PLAN, fixtures.REPOSITORY)
        variants = [
            body.replace("</details>", "", 1),
            body.replace("assignee: @harrisonyhq", "owner: @harrisonyhq", 1),
            body.replace("description:", "assignee:\n\ndescription:", 1),
            body.replace("  Implement the connector.", "Implement the connector.", 1),
            body.replace("</summary>", "", 1),
        ]
        for value in variants:
            with self.subTest(body=value), self.assertRaises(OperationError):
                parse_release(value, fixtures.REPOSITORY)

    def test_plain_title_creation_and_repeated_update(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.initialize()
        self.assertIn(
            "<summary>Connector v2\ninitial integration #3</summary>",
            x.api.items[1]["body"],
        )
        x.controller.initialize_or_update(1, "update")
        self.assertEqual(len(x.api.items), 4)

    def test_migration_preserves_initialized_scope_and_objects(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        before = x.initialize()
        x.api.items[1]["body"] = fold_requirements(
            x.api.items[1]["body"], fixtures.REPOSITORY
        )
        x.controller.initialize_or_update(1, "update")
        after = x.controller.reconcile(1)
        self.assertEqual(after["scope"], before["scope"])
        self.assertEqual(len(x.api.items), 4)

    def test_lost_creation_response_is_recovered(self):
        x = fixtures.RepositoryOperations()
        x.setUp()
        x.api.items[1]["body"] = fold_requirements(
            fixtures.PLAN.replace(" []", ""), fixtures.REPOSITORY
        )
        x.api.lose_response_for = "Store interface"
        with self.assertRaises(OperationError):
            x.initialize()
        x.initialize()
        self.assertEqual(len(x.api.items), 4)
        self.assertIn("#4</summary>", x.api.items[1]["body"])
