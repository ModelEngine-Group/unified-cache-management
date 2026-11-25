import copy
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from controller import Controller, GitHub, bug_progress
from presentation import NAV_START, OVERVIEW_END, OVERVIEW_START, release_body
from syntax import (
    OperationError,
    bind_entries,
    field,
    issue_reference,
    parse_release,
    patch_pull_requests,
    patch_references,
)
from taskcards import task_requirements

REPOSITORY = "harrisonyhq/unified-cache-management"
PLAN = """### Release Information

version: 0.12.0
branch: main
release_manager: @harrisonyhq
test_lead:
release_date: TBD

### Requirements

- Connector v2
  initial integration []

  description:
    Implement the connector.

    Acceptance criteria:
    - Pass the integration test.
    - Document the interface.

  assignee: @harrisonyhq

  pull_requests:

- Store interface []

  description:
    Implement the first interface.

  assignee:

  pull_requests:

### Change Log

reason: Initial scope.
"""


class ParserContracts(unittest.TestCase):
    def test_historical_release_keeps_text_and_gets_the_moved_overview(self):
        original = "## Version information\n\nLegacy release notes.\n"
        updated = release_body(original, {"scope": {}, "bugs": {}})
        self.assertTrue(updated.startswith(original))
        self.assertIn("### Release Overview", updated)
        self.assertEqual(release_body(updated, {"scope": {}, "bugs": {}}), updated)

    def test_managed_regions_are_idempotent_and_do_not_change_scope(self):
        state = {
            "scope": {},
            "bugs": {},
            "milestone": {"html_url": "https://example.com/milestone"},
            "version": "0.12.0",
            "archive": 8,
        }
        updated = release_body(PLAN, state)
        self.assertEqual(
            parse_release(updated, REPOSITORY), parse_release(PLAN, REPOSITORY)
        )
        self.assertEqual(release_body(updated, state), updated)
        self.assertLess(updated.index(NAV_START), updated.index("version:"))
        self.assertTrue(updated.rstrip().endswith(OVERVIEW_END))
        self.assertEqual(updated.count(OVERVIEW_START), 1)

    def test_overview_unifies_requirement_and_bug_prs(self):
        state = {
            "scope": {"issue-14": {"number": 14, "developer": "owner"}},
            "bugs": {"8": {}},
            "prs": {"20": {"requirements": [14], "bugs": [8], "merged": True}},
            "observed": {"8": {"state": "closed", "assignees": ["fixer"]}},
        }
        body = release_body(PLAN, state)
        self.assertIn("| Issue | Pull requests | Assignee | State |", body)
        self.assertIn("| #14 | #20 (merged) | @owner |", body)
        self.assertIn("| #8 | #20 (merged) | @fixer | closed", body)
        self.assertNotIn("| Item |", body)
        self.assertNotIn("缺陷与修复", body)

    def test_malformed_managed_markers_preserve_source(self):
        body = PLAN + OVERVIEW_START + "Human text"
        with self.assertRaises(OperationError):
            release_body(body, {"scope": {}, "bugs": {}})

    def test_native_pr_display_keeps_issue_identity(self):
        body = PLAN.replace(
            "initial integration []",
            f"initial integration [Feature](https://github.com/{REPOSITORY}/issues/14)",
        )
        updated = patch_pull_requests(
            body,
            {14: [{"number": 20, "state": "closed", "merged_at": None}]},
            REPOSITORY,
        )
        self.assertIn("initial integration #14", updated)
        self.assertIn("    - #20 | closed", updated)
        self.assertEqual(
            parse_release(updated, REPOSITORY), parse_release(body, REPOSITORY)
        )

    def test_multiline_title_description_and_empty_fields(self):
        parsed = parse_release(PLAN, REPOSITORY)
        self.assertEqual(
            parsed["entries"][0]["title"], "Connector v2 initial integration"
        )
        self.assertIn(
            "\n\nAcceptance criteria:\n- Pass", parsed["entries"][0]["description"]
        )
        self.assertIsNone(parsed["entries"][1]["developer"])
        self.assertEqual(field("assignee:\nnext: @someone", "assignee"), "")
        self.assertEqual(parsed["metadata"]["test_lead"], "")

    def test_actual_release_checklist_is_english_and_parses(self):
        root = Path(__file__).parent
        template = (
            root.parent / "ISSUE_TEMPLATE/900-release-checklist.yml"
        ).read_text()
        self.assertTrue(template.isascii())
        block = re.search(
            r"(?ms)^        - \[ \] Requirement title\n.*?(?=^        -->)", template
        )[0]
        requirement = "\n".join(line[8:] for line in block.splitlines())
        example = (
            PLAN.split("### Requirements", 1)[0]
            + "### Requirements\n\n"
            + requirement
            + "\n### Change Log\n\nreason: Initial scope.\n"
        )
        self.assertIn("  pull_requests:\n    -\n", requirement)
        entries = parse_release(example, REPOSITORY, read_pr_requests=True)["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["requested_prs"], [])
        self.assertNotIn("label: Pull Requests", template)
        self.assertFalse((root.parent / "ISSUE_TEMPLATE/010-release-plan.yml").exists())

    def test_only_requirements_section_is_parsed(self):
        value = PLAN.replace(
            "### Change Log", "### Functional Tests\n\n- [ ] A test\n\n### Change Log"
        )
        self.assertEqual(len(parse_release(value, REPOSITORY)["entries"]), 2)

    def test_reference_writeback_is_idempotent_and_preserves_description(self):
        entries = parse_release(PLAN, REPOSITORY)["entries"]
        mappings = {e["key"]: n for e, n in zip(entries, [14, 13])}
        patched = patch_references(PLAN, mappings, REPOSITORY)
        self.assertIn("initial integration #14", patched)
        self.assertEqual(patch_references(patched, mappings, REPOSITORY), patched)
        actual = parse_release(patched, REPOSITORY)["entries"]
        self.assertEqual([e["key"] for e in actual], ["issue-14", "issue-13"])
        self.assertEqual(actual[0]["description"], entries[0]["description"])
        self.assertEqual(actual, bind_entries(entries, mappings))

    def test_full_issue_url_normalizes_only_the_title_reference(self):
        url = f"https://github.com/{REPOSITORY}/issues/14"
        body = PLAN.replace("initial integration []", f"initial integration [{url}]")
        body = body.replace("Implement the connector.", f"See [{url}].")
        before = parse_release(body, REPOSITORY)["entries"]
        patched = patch_references(body, {}, REPOSITORY)
        self.assertIn("initial integration #14", patched)
        self.assertIn(f"See [{url}].", patched)
        self.assertEqual(parse_release(patched, REPOSITORY)["entries"], before)
        self.assertEqual(patch_references(patched, {}, REPOSITORY), patched)

    def test_writeback_preserves_brackets_in_multiline_title(self):
        body = PLAN.replace("- Connector v2", "- Connector [v2]")
        key = parse_release(body, REPOSITORY)["entries"][0]["key"]
        patched = patch_references(body, {key: 14}, REPOSITORY)
        self.assertIn("- Connector [v2]\n  initial integration #14", patched)
        self.assertEqual(parse_release(patched, REPOSITORY)["entries"][0]["number"], 14)

    def test_pr_refresh_normalizes_legacy_full_url_without_scope_changes(self):
        body = PLAN.replace(
            "initial integration []",
            f"initial integration [https://github.com/{REPOSITORY}/issues/14]",
        )
        patched = patch_pull_requests(body, {}, REPOSITORY)
        self.assertIn("initial integration #14", patched)
        self.assertEqual(
            parse_release(body, REPOSITORY)["entries"],
            parse_release(patched, REPOSITORY)["entries"],
        )

    def test_existing_references_and_issue15_regression(self):
        body = PLAN.replace(
            "Connector v2\n  initial integration []",
            "Support GLM5.5 [https://github.com/" + REPOSITORY + "/issues/14]",
        )
        body = body.replace("Store interface []", "SGLang EPD [#13]")
        self.assertEqual(
            [e["number"] for e in parse_release(body, REPOSITORY)["entries"]], [14, 13]
        )

    def test_common_pasted_formats_normalize_to_markdown_links(self):
        url = f"https://github.com/{REPOSITORY}/issues/14"
        expected = None
        for ref in (
            f"[Issue]({url})",
            f"[#14]({url})",
            f"[{url}]",
            "[#14]",
            url,
            "#14",
            f"{REPOSITORY}#14",
            f"[Issue]({url}#issuecomment-123)",
            "[Issue](#14 )",
            f"[Issue]( {url} )",
        ):
            with self.subTest(reference=ref):
                body = PLAN.replace(
                    "initial integration []", "initial integration " + ref
                )
                entry = parse_release(body, REPOSITORY)["entries"][0]
                expected = expected or entry
                self.assertEqual(entry, expected)
                patched = patch_references(body, {}, REPOSITORY)
                self.assertIn("initial integration #14", patched)
                self.assertEqual(patch_references(patched, {}, REPOSITORY), patched)
                self.assertEqual(
                    parse_release(patched, REPOSITORY)["entries"][0], entry
                )

    def test_whole_title_can_be_a_markdown_link(self):
        url = f"https://github.com/{REPOSITORY}/issues/14"
        body = PLAN.replace(
            "- Connector v2\n  initial integration []",
            f"- [Connector integration]({url})",
        )
        entry = parse_release(body, REPOSITORY)["entries"][0]
        self.assertEqual(entry["title"], "Connector integration")
        self.assertEqual(entry["number"], 14)
        self.assertEqual(
            parse_release(patch_references(body, {}, REPOSITORY), REPOSITORY),
            parse_release(body, REPOSITORY),
        )

    def test_issue15_whole_title_and_short_destination(self):
        body = PLAN.replace(
            "- Connector v2\n  initial integration []", "- [GLM5.5 support](#14 )"
        )
        patched = patch_references(body, {}, REPOSITORY)
        self.assertIn("- GLM5.5 support #14", patched)
        self.assertEqual(
            parse_release(body, REPOSITORY), parse_release(patched, REPOSITORY)
        )

    def test_multiline_markdown_link_label_round_trips(self):
        url = f"https://github.com/{REPOSITORY}/issues/14"
        body = PLAN.replace(
            "- Connector v2\n  initial integration []",
            f"- [Connector v2\n  initial integration]({url})",
        )
        before = parse_release(body, REPOSITORY)["entries"]
        patched = patch_references(body, {}, REPOSITORY)
        self.assertIn("- Connector v2 initial integration #14", patched)
        self.assertEqual(before, parse_release(patched, REPOSITORY)["entries"])
        self.assertEqual(patch_references(patched, {}, REPOSITORY), patched)

    def test_new_requirement_uses_empty_markdown_destination(self):
        body = PLAN.replace("initial integration []", "initial integration [Issue]()")
        entry = parse_release(body, REPOSITORY)["entries"][0]
        self.assertIsNone(entry["number"])
        self.assertEqual(entry["title"], "Connector v2 initial integration")
        patched = patch_references(body, {entry["key"]: 14}, REPOSITORY)
        self.assertIn(
            "initial integration #14",
            patched,
        )
        self.assertEqual(
            parse_release(patched, REPOSITORY)["entries"][0],
            dict(entry, key="issue-14", number=14),
        )

    def test_markdown_link_destination_controls_mentions(self):
        self.assertEqual(
            issue_reference(
                f"[#99](https://github.com/{REPOSITORY}/issues/14)", REPOSITORY
            ),
            14,
        )
        with self.assertRaises(OperationError):
            issue_reference("#0", REPOSITORY)

    def test_invalid_markdown_destinations_are_rejected(self):
        for ref in (
            "[Issue](https://github.com/other/repo/issues/14)",
            f"[Issue](https://github.com/{REPOSITORY}/pull/14)",
            "[Issue](javascript:alert)",
        ):
            with self.subTest(reference=ref), self.assertRaises(OperationError):
                parse_release(
                    PLAN.replace(
                        "initial integration []", "initial integration " + ref
                    ),
                    REPOSITORY,
                )

    def test_invalid_input_reports_line_before_creation(self):
        for bad in [
            PLAN.replace("  description:", " description:", 1),
            PLAN.replace("    Implement the connector.", "  Implement the connector."),
            PLAN.replace("  assignee: @harrisonyhq", "  assignee: @a @b"),
            PLAN.replace(
                "  assignee: @harrisonyhq",
                "  assignee: @harrisonyhq\n  assignee: @other",
            ),
            PLAN.replace("Store interface []", "Store interface [#abc]"),
            PLAN.replace(
                "Store interface []",
                "Store interface [https://github.com/a/b/issues/1]",
            ),
            PLAN.replace(
                "Store interface []",
                "Store interface [https://github.com/" + REPOSITORY + "/pull/1]",
            ),
            PLAN.replace("Store interface []", "Connector v2 initial integration []"),
        ]:
            with self.subTest(body=bad), self.assertRaises(OperationError):
                parse_release(bad, REPOSITORY)

    def test_pr_display_does_not_change_scope(self):
        body = patch_references(
            PLAN, {parse_release(PLAN, REPOSITORY)["entries"][0]["key"]: 14}, REPOSITORY
        )
        before = parse_release(body, REPOSITORY)["entries"]
        updated = patch_pull_requests(
            body, {14: [{"number": 33, "state": "open", "merged_at": None}]}, REPOSITORY
        )
        self.assertIn("| open", updated)
        self.assertEqual(before, parse_release(updated, REPOSITORY)["entries"])
        self.assertEqual(
            patch_pull_requests(
                updated,
                {14: [{"number": 33, "state": "open", "merged_at": None}]},
                REPOSITORY,
            ),
            updated,
        )

    def test_repository_root_endpoint(self):
        api = GitHub(REPOSITORY)
        api.call = lambda path, method, data: path
        self.assertEqual(api.rest(""), "repos/" + REPOSITORY)

    def test_release_date_validation(self):
        self.assertEqual(
            parse_release(
                PLAN.replace("release_date: TBD", "release_date: 2026-12-01"),
                REPOSITORY,
            )["metadata"]["release_date"],
            "2026-12-01",
        )
        with self.assertRaises(OperationError):
            parse_release(
                PLAN.replace("release_date: TBD", "release_date: tomorrow"), REPOSITORY
            )


class MemoryGitHub:
    def __init__(self):
        self.items = {1: self.make_issue(1, "[Release] UCM 0.12.0", PLAN, ["release"])}
        self.comments, self.milestones, self.parents = [], [], {}
        self.prs, self.reviews, self.inline, self.events = {}, {}, {}, {}
        self.projects = {}
        self.project_statuses = {}
        self.lose_response_for = None
        self.lose_milestone_response = False
        self.lose_project_response = False
        self.reactions = {}
        self.pins = {}
        self.pin_failure = False
        self.reaction_failure = False
        self.minimized_comments = set()
        self.minimize_failure = False
        self.native_links = {}
        self.native_failure = False
        self.internal_items = {}
        self.page_size = 100

    @staticmethod
    def make_issue(number, title, body, labels):
        return {
            "number": number,
            "id": number + 100,
            "title": title,
            "body": body,
            "labels": [{"name": x} for x in labels],
            "milestone": None,
            "assignees": [],
            "state": "open",
            "created_at": "2026-01-01T00:00:00Z",
            "node_id": f"I_{number}",
            "html_url": f"https://github.com/{REPOSITORY}/issues/{number}",
        }

    def call(self, path, method="GET", data=None):
        if path == "graphql":
            if "addCloseIssueReferences" in data["query"]:
                payload = data["variables"]["input"]
                issue = next(
                    i for i in self.items.values() if i["node_id"] == payload["issueId"]
                )
                links = self.native_links.setdefault(issue["number"], [])
                for node_id in payload["pullRequestIds"]:
                    pr = next(p for p in self.prs.values() if p["node_id"] == node_id)
                    links.append((pr["number"], REPOSITORY))
                return {
                    "data": {
                        "addCloseIssueReferences": {"issue": {"id": payload["issueId"]}}
                    }
                }
            if "minimizeComment" in data["query"]:
                if self.minimize_failure:
                    raise OperationError("Minimize API unavailable")
                self.minimized_comments.add(data["variables"]["id"])
                return {
                    "data": {
                        "minimizeComment": {"minimizedComment": {"isMinimized": True}}
                    }
                }
            query, variables = data["query"], data["variables"]
            if "closedByPullRequestsReferences" in query:
                if self.native_failure:
                    raise OperationError("Native associations unavailable")
                self.native_include_closed = "includeClosedPrs:true" in query
                numbers = [int(n) for n in re.findall(r"i(\d+):issue", query)]

                def page(number):
                    nodes = self.native_links.get(number, [])
                    offset = int(variables.get("after") or 0)
                    selected = nodes[offset : offset + self.page_size]
                    return {
                        "closedByPullRequestsReferences": {
                            "nodes": [
                                {"number": p, "repository": {"nameWithOwner": r}}
                                for p, r in selected
                            ],
                            "pageInfo": {
                                "hasNextPage": offset + len(selected) < len(nodes),
                                "endCursor": str(offset + len(selected)),
                            },
                        }
                    }

                result = (
                    {f"i{n}": page(n) for n in numbers}
                    if numbers
                    else {"issue": page(variables["number"])}
                )
                return {"data": {"repository": result}}
            if "closingIssuesReferences" in query:
                numbers = [int(n) for n in re.findall(r"p(\d+):pullRequest", query)]

                def detail(number):
                    pr = self.prs[number]
                    nodes = [
                        i
                        for i, links in self.native_links.items()
                        if (number, REPOSITORY) in links
                    ]
                    offset = int(variables.get("after") or 0)
                    selected = nodes[offset : offset + self.page_size]
                    issues = [
                        {
                            "number": n,
                            "id": self.items[n]["node_id"],
                            "title": self.items[n]["title"],
                            "state": self.items[n]["state"].upper(),
                            "labels": {"nodes": self.items[n]["labels"]},
                            "milestone": self.items[n]["milestone"],
                            "repository": {"nameWithOwner": REPOSITORY},
                        }
                        for n in selected
                    ]
                    return {
                        "number": number,
                        "id": pr["node_id"],
                        "url": pr["html_url"],
                        "state": pr["state"].upper(),
                        "mergedAt": pr.get("merged_at"),
                        "isDraft": pr.get("draft", False),
                        "milestone": pr.get("milestone"),
                        "closingIssuesReferences": {
                            "nodes": issues,
                            "pageInfo": {
                                "hasNextPage": offset + len(selected) < len(nodes),
                                "endCursor": str(offset + len(selected)),
                            },
                        },
                    }

                result = (
                    {f"p{n}": detail(n) for n in numbers}
                    if numbers
                    else {"pullRequest": detail(variables["number"])}
                )
                return {"data": {"repository": result}}
            if self.pin_failure:
                raise OperationError("Pin API unavailable")
            variables = data["variables"]
            if data["query"].startswith("query"):
                n = variables["number"]
                return {
                    "data": {
                        "repository": {
                            "issue": {"id": f"I_{n}", "isPinned": n in self.pins},
                            "pinnedIssues": {
                                "nodes": [
                                    {
                                        "issue": {"number": number},
                                        "pinnedBy": {
                                            "__typename": actor,
                                            "login": actor,
                                        },
                                    }
                                    for number, actor in self.pins.items()
                                ]
                            },
                        }
                    }
                }
            n = int(variables["id"].split("_")[1])
            self.pins[n] = "Bot"
            return {"data": {}}
        return {"login": path.split("/")[-1]}

    def pages(self, path):
        if path == "milestones?state=open":
            return copy.deepcopy(
                [item for item in self.milestones if item["state"] == "open"]
            )
        if path.startswith("milestones"):
            return copy.deepcopy(self.milestones)
        if path.endswith("/comments") and path.startswith("issues/"):
            number = int(path.split("/")[1])
            return copy.deepcopy([c for c in self.comments if c["issue"] == number])
        if path.endswith("/events"):
            return copy.deepcopy(self.events.get(int(path.split("/")[1]), []))
        if path.startswith("pulls/"):
            _, number, kind = path.split("/")
            return copy.deepcopy(
                (self.reviews if kind == "reviews" else self.inline).get(
                    int(number), []
                )
            )
        if path.endswith("/sub_issues"):
            number = int(path.split("/")[1])
            return copy.deepcopy(
                [
                    self.items[n]
                    for n, parent in self.parents.items()
                    if parent == number
                ]
            )
        if path == "pulls?state=all":
            return copy.deepcopy(list(self.prs.values()))
        if path.startswith("issues?"):
            values = list((self.items | self.prs | self.internal_items).values())
            if "labels=" in path:
                label = path.split("labels=")[1].split("&")[0]
                values = [i for i in values if {"name": label} in i["labels"]]
            if "milestone=" in path:
                target = int(path.split("milestone=")[1].split("&")[0])
                values = [
                    i for i in values if (i["milestone"] or {}).get("number") == target
                ]
            return copy.deepcopy(values)
        raise AssertionError(path)

    def rest(self, path, method="GET", data=None):
        if path.startswith("labels/"):
            return {"name": path.split("/")[-1]}
        if path.startswith("pulls?state=all&sort=created"):
            page = int(path.split("page=")[-1])
            rows = sorted(
                self.prs.values(),
                key=lambda row: (row["created_at"], row["number"]),
                reverse="direction=desc" in path,
            )
            return copy.deepcopy(rows[(page - 1) * 100 : page * 100])
        if path.startswith("issues?state=all&sort=updated"):
            from urllib.parse import parse_qs, urlsplit

            query = parse_qs(urlsplit(path).query)
            since = query["since"][0]
            rows = [
                dict(pr, pull_request={})
                for pr in self.prs.values()
                if pr.get("updated_at", "9999") >= since
            ]
            rows.sort(key=lambda row: row.get("updated_at", "9999"), reverse=True)
            page = int(query["page"][0])
            return copy.deepcopy(rows[(page - 1) * 100 : page * 100])
        if path.startswith("issues/comments/") and path.endswith("/reactions"):
            if self.reaction_failure:
                raise OperationError("Reaction endpoint unavailable")
            return self.reactions.setdefault(
                int(path.split("/")[2]), {"content": data["content"]}
            )
        if path == "":
            return {"default_branch": "main"}
        if path.startswith("commits/"):
            return {"sha": "a" * 40}
        if path.startswith("collaborators/"):
            return {"permission": "write"}
        parts = path.split("/")
        if parts[0] == "pulls" and len(parts) == 2:
            return copy.deepcopy(self.prs[int(parts[1])])
        if path.startswith("compare/"):
            return {"status": getattr(self, "comparison_status", "ahead")}
        if parts[0] == "milestones" and len(parts) == 2:
            item = next(m for m in self.milestones if m["number"] == int(parts[1]))
            if data:
                item.update(data)
            return copy.deepcopy(item)
        if path == "milestones" and method == "POST":
            item = dict(
                data,
                number=len(self.milestones) + 1,
                state="open",
                html_url="https://example.com/milestone",
            )
            self.milestones.append(item)
            return copy.deepcopy(item)
        if path == "issues" and method == "POST":
            internal = "release-directory" in data["labels"]
            number = 900000 if internal else max(self.items) + 1
            item = self.make_issue(number, data["title"], data["body"], data["labels"])
            item["milestone"] = next(
                (m for m in self.milestones if m["number"] == data.get("milestone")),
                None,
            )
            (self.internal_items if internal else self.items)[number] = item
            if data["title"] == self.lose_response_for:
                self.lose_response_for = None
                raise OperationError(
                    "transport interrupted after server accepted issue"
                )
            return copy.deepcopy(item)
        if parts[:2] == ["issues", "comments"] and method == "PATCH":
            comment = next(c for c in self.comments if c["id"] == int(parts[2]))
            comment.update(data)
            return copy.deepcopy(comment)
        if parts[0] == "issues" and len(parts) == 2:
            if int(parts[1]) not in self.items | self.internal_items:
                raise OperationError("HTTP 404")
            item = (self.items | self.internal_items)[int(parts[1])]
            if method == "PATCH":
                for key, value in data.items():
                    if key == "milestone":
                        item[key] = next(
                            (m for m in self.milestones if m["number"] == value),
                            (
                                {"number": value, "title": str(value)}
                                if value is not None
                                else None
                            ),
                        )
                        if int(parts[1]) in self.prs:
                            self.prs[int(parts[1])][key] = item[key]
                            if self.lose_milestone_response:
                                self.lose_milestone_response = False
                                raise OperationError(
                                    "Milestone response lost after acceptance"
                                )
                    elif key == "assignees":
                        item[key] = [{"login": x} for x in value]
                    elif key == "labels":
                        item[key] = [{"name": x} for x in value]
                    else:
                        item[key] = value
            return copy.deepcopy(item)
        if parts[0] == "issues" and parts[-1] == "comments" and method == "POST":
            comment = dict(
                data,
                issue=int(parts[1]),
                id=max([c["id"] for c in self.comments] + [0]) + 1,
                user={"login": "github-actions[bot]", "type": "Bot"},
            )
            comment["node_id"] = "IC_" + str(comment["id"])
            self.comments.append(comment)
            return copy.deepcopy(comment)
        if parts[-1] == "parent":
            parent = self.parents.get(int(parts[1]))
            if parent is None:
                raise OperationError("HTTP 404")
            return copy.deepcopy(self.items[parent])
        if parts[-1] == "sub_issues" and method == "POST":
            child = next(
                i for i in self.items.values() if i["id"] == data["sub_issue_id"]
            )
            self.parents[child["number"]] = int(parts[1])
            return copy.deepcopy(child)
        if parts[-1] == "labels" and method == "POST":
            self.items[int(parts[1])]["labels"] += [
                {"name": label} for label in data["labels"]
            ]
            return []
        raise AssertionError((path, method, data))

    def graphql(self, query, **variables):
        if query.startswith("query"):

            def project_node(node_id):
                return {
                    "id": node_id,
                    "projectItems": {
                        "nodes": [
                            {
                                "id": item,
                                "project": {"id": p},
                                "fieldValueByName": self.project_statuses.get(item),
                            }
                            for p, item in self.projects.get(node_id, {}).items()
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    },
                }

            if "ids" in variables:
                return {
                    "nodes": [project_node(node_id) for node_id in variables["ids"]]
                }
            return {"node": project_node(variables["id"])}
        if "addProjectV2ItemById" in query:
            values = self.projects.setdefault(variables["content"], {})
            item = values.setdefault(
                variables["project"], variables["content"] + "_" + variables["project"]
            )
            if self.lose_project_response:
                self.lose_project_response = False
                raise OperationError("Project response lost after acceptance")
            return {"addProjectV2ItemById": {"item": {"id": item}}}
        if "deleteProjectV2Item" in query:
            for values in self.projects.values():
                if values.get(variables["project"]) == variables["item"]:
                    del values[variables["project"]]
            return {"deleteProjectV2Item": {"deletedItemId": variables["item"]}}
        if "updateProjectV2ItemFieldValue" in query:
            data = variables["input"]
            self.project_statuses[data["itemId"]] = {
                "name": data["value"]["singleSelectOptionId"]
            }
            return {}
        raise AssertionError(query)

    def add_pr(self, number, body):
        item = self.make_issue(number, "Implementation", body, [])
        item["pull_request"] = {}
        self.items[number] = item
        pr = dict(
            item,
            html_url=f"https://github.com/{REPOSITORY}/pull/{number}",
            merged_at=None,
            draft=False,
        )
        self.prs[number] = pr
        for value in re.findall(
            r"(?i)(?:fix(?:es|ed)?|close(?:s|d)?|resolve(?:s|d)?)\s+#(\d+)", body
        ):
            self.native_links.setdefault(int(value), []).append((number, REPOSITORY))
        return pr

    def comment(self, number, body):
        item = {
            "issue": number,
            "id": max([c["id"] for c in self.comments] + [0]) + 1,
            "body": body,
            "user": {"login": "developer", "type": "User"},
            "html_url": "https://example.com/comment",
        }
        self.comments.append(item)
        return item


class RepositoryOperations(unittest.TestCase):
    def test_initialize_links_pr_numbers_and_urls_to_new_requirement(self):
        merged = self.api.add_pr(10, "")
        merged.update(state="closed", merged_at="2026-01-02")
        self.api.add_pr(11, "")
        self.api.items[1]["body"] = task_requirements(PLAN, REPOSITORY).replace(
            "  pull_requests:\n",
            f"  pull_requests:\n    - #10\n    - https://github.com/{REPOSITORY}/pull/11\n"
            f"    - [Same PR](https://github.com/{REPOSITORY}/pull/10)\n",
            1,
        )
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertFalse(self.api.native_links)
        self.assertFalse(self.api.comments)

        state = self.initialize()
        requirement = state["prs"]["10"]["requirements"][0]
        self.assertEqual(
            self.api.native_links[requirement], [(10, REPOSITORY), (11, REPOSITORY)]
        )
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        body = self.api.items[1]["body"]
        self.assertIn("    - #10 | merged\n    - #11 | open", body)
        self.assertIn(f"| #{requirement} | #10 (merged), #11 (open) |", body)
        self.assertNotIn("Same PR", body)

    def test_update_imports_pending_prs_and_clears_rendered_rows_after_unlink(self):
        state = self.initialize()
        requirement = 3
        self.api.projects[self.api.items[requirement]["node_id"]] = {"P_TEST": "item"}
        pr = self.api.add_pr(10, "")
        reference = f"    - [PR](https://github.com/{REPOSITORY}/pull/10)"
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "  pull_requests:\n", "  pull_requests:\n" + reference + "\n", 1
        )
        self.controller.handle({"issue": self.api.items[1]}, "issues")
        self.assertIn(reference, self.api.items[1]["body"])
        self.assertFalse(self.api.native_links)
        self.controller.initialize_or_update(1, "update", True)
        self.assertFalse(self.api.native_links)

        self.controller.initialize_or_update(1, "update")
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.native_links[requirement], [(10, REPOSITORY)])
        self.assertIn(f"| #{requirement} | #10 (open) |", self.api.items[1]["body"])
        self.assertIn("P_TEST", self.api.projects[pr["node_id"]])

        del self.api.native_links[requirement]
        self.controller.initialize_or_update(1, "update")
        self.assertFalse(self.api.native_links)
        self.assertNotIn("#10 | open", self.api.items[1]["body"])
        self.assertIn("P_TEST", self.api.projects[pr["node_id"]])

    def test_pr_link_retry_recovers_after_lost_mutation_response(self):
        self.api.add_pr(10, "")
        self.api.items[1]["body"] = PLAN.replace(
            "  pull_requests:\n", "  pull_requests:\n    - #10\n", 1
        )
        call = self.api.call

        def lose_response(path, method="GET", data=None):
            result = call(path, method, data)
            if "addCloseIssueReferences" in (data or {}).get("query", ""):
                raise OperationError("link response lost")
            return result

        with self.controller.snapshots():
            with patch.object(self.api, "call", side_effect=lose_response):
                with self.assertRaisesRegex(OperationError, "link response lost"):
                    self.initialize()
            self.assertIn("    - #10\n", self.api.items[1]["body"])
            state = self.initialize()
        requirement = state["prs"]["10"]["requirements"][0]
        self.assertEqual(self.api.native_links[requirement], [(10, REPOSITORY)])
        self.assertIn("#10 | open", self.api.items[1]["body"])

    def test_init_rejects_cross_repository_pr_before_creating_issues(self):
        self.api.items[1]["body"] = PLAN.replace(
            "  pull_requests:\n",
            "  pull_requests:\n    - https://github.com/elsewhere/project/pull/10\n",
            1,
        )
        with self.assertRaisesRegex(
            OperationError, "same-repository PR number or link"
        ):
            self.initialize()
        self.assertEqual(set(self.api.items), {1})
        self.assertFalse(self.api.native_links)

    def test_plain_titles_create_issues_and_write_back_multiline_references(self):
        self.api.items[1]["body"] = PLAN.replace(" []", "")
        state = self.initialize()
        self.assertEqual(
            set(state["scope"]), {"issue-1", "issue-2", "issue-3", "issue-4"}
        )
        self.assertIn(
            "- Connector v2\n  initial integration #3", self.api.items[1]["body"]
        )
        self.assertIn("- Store interface #4", self.api.items[1]["body"])
        self.assertIn("Acceptance criteria:", self.api.items[3]["body"])
        self.assertNotIn("Release: #1", self.api.items[3]["body"])
        self.assertNotIn("Release: #1", self.api.items[4]["body"])
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(len(self.api.items), 4)

    def test_update_creates_plain_title_requirement_once(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "### Change Log",
            "- New feature\n\n  description:\n    New work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log",
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIn("- New feature #5", self.api.items[1]["body"])
        self.assertEqual(self.api.items[5]["title"], "New feature")
        self.assertIn("Release: #1", self.api.items[5]["body"])
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(len(self.api.items), 5)

    def test_description_trailing_whitespace_does_not_block_check(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "Implement the first interface.", "Implement the first interface. "
        )
        _, report = self.controller.release_check(1)
        self.assertFalse(report["pending_requirements"])
        self.controller.initialize_or_update(1, "update")
        self.assertIn("Implement the first interface. \n", self.api.items[1]["body"])

    def test_release_check_lists_open_features_bugs_and_missing_prs(self):
        self.initialize()
        bug = self.api.make_issue(8, "Open target Bug", "", ["bug"])
        bug["milestone"] = self.api.milestones[0]
        self.api.items[8] = bug
        _, report = self.controller.release_check(1)
        self.assertEqual(report["open_issues"], [1, 2, 3, 4, 8])
        self.assertEqual(report["issues_without_pr"], [3, 4, 8])
        self.assertEqual(report["status"], "blocked")
        self.assertIn("| #8 |", self.api.items[1]["body"])
        self.assertNotIn('"status": "stale"', self.api.items[1]["body"])

    def test_release_check_distinguishes_open_and_closed_unmerged_prs(self):
        self.initialize()
        self.api.add_pr(10, "Fixes #3")
        self.api.add_pr(11, "Fixes #4")["state"] = "closed"
        _, report = self.controller.release_check(1)
        self.assertEqual(report["unmerged_prs"], [10])
        self.assertEqual(report["closed_unmerged_prs"], [11])

    def test_release_check_describes_pending_update_without_creating_scope(self):
        self.initialize()
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "- New feature []\n\n  description:\n    Planned work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log\n\nreason: Scope adjustment.\n"
        )
        count = len(self.api.items)
        _, report = self.controller.release_check(1)
        self.assertEqual(report["pending_requirements"], ["New feature"])
        self.assertEqual(len(self.api.items), count)

    def test_check_uses_merge_state_without_commit_comparison(self):
        self.initialize()
        for n in (1, 2, 3, 4):
            self.api.items[n]["state"] = "closed"
        self.api.add_pr(10, "Fixes #3").update(state="closed", merged_at="2026-11-19")
        state, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertNotIn("prs_outside_candidate", report)
        self.assertNotIn("verification", state)

    def test_release_check_command_does_not_create_machine_comments(self):
        self.initialize()
        comment = self.api.comment(1, "/release_check")
        self.controller.handle(
            {"issue": self.api.items[1], "comment": comment, "sender": comment["user"]},
            "issue_comment",
        )
        self.assertEqual(self.api.comments, [comment])
        self.assertNotIn("### Release Check", self.api.items[1]["body"])

    def test_verify_command_is_retired_and_does_not_close_issue(self):
        self.initialize()
        comment = self.api.comment(1, "/verify #3 passed https://example.com/report")
        self.controller.handle(
            {"issue": self.api.items[1], "comment": comment, "sender": comment["user"]},
            "issue_comment",
        )
        self.assertEqual(self.api.items[3]["state"], "open")
        self.assertNotIn(comment["id"], self.api.reactions)
        self.assertFalse(hasattr(self.controller, "verify"))

    def test_release_check_includes_pr_assigned_directly_to_milestone(self):
        self.initialize()
        self.api.add_pr(10, "Infrastructure change")["milestone"] = self.api.milestones[
            0
        ]
        _, report = self.controller.release_check(1)
        self.assertEqual(report["unmerged_prs"], [10])

    def test_cross_release_bug_fix_is_in_target_release_check(self):
        self.initialize()
        self.api.items[50] = self.api.make_issue(
            50,
            "Other release",
            PLAN.replace("version: 0.12.0", "version: 0.13.0"),
            ["release"],
        )
        self.controller.initialize_or_update(50, "release_init")
        other = self.controller.reconcile(50)
        bug = self.api.make_issue(8, "Retargeted Bug", "", ["bug"])
        bug["milestone"] = self.api.milestones[0]
        self.api.items[8] = bug
        self.api.parents[8] = other["archive"]
        self.api.add_pr(10, "Fixes #8")
        _, report = self.controller.release_check(1)
        self.assertIn(8, report["open_issues"])
        self.assertIn(10, report["unmerged_prs"])

    def test_native_development_link_updates_metadata_and_table(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_NATIVE": "REQ"}
        self.api.add_pr(10, "No textual references")
        self.api.native_links[3] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        state = self.controller.reconcile(1)
        self.assertEqual(state["prs"]["10"]["requirements"], [3])
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertIn("P_NATIVE", self.api.projects["I_10"])
        self.assertIn("| #3 | #10 (open)", self.api.items[1]["body"])
        self.api.native_links[3] = []
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertIn("P_NATIVE", self.api.projects["I_10"])
        self.assertNotIn("#10", self.api.items[1]["body"])

    def test_ordinary_mentions_do_not_supply_associations(self):
        self.initialize()
        self.api.add_pr(10, "Related #3")
        self.api.comment(10, "Implements #4")
        self.api.reviews[10] = [{"body": "Fixes #3", "user": {"type": "User"}}]
        self.controller.reconcile(1)
        self.assertNotIn("10", self.controller.reconcile(1)["prs"])
        self.assertIsNone(self.api.prs[10]["milestone"])

    def test_native_links_keep_merged_pr_and_closed_status(self):
        self.initialize()
        pr = self.api.add_pr(10, "")
        self.api.native_links[3] = [(10, REPOSITORY)]
        pr.update(state="closed", merged_at="2026-11-19T00:00:00Z")
        self.controller.reconcile(1)
        self.assertTrue(self.api.native_include_closed)
        self.assertIn("| #3 | #10 (merged)", self.api.items[1]["body"])

    def test_native_link_pagination_and_repository_filter(self):
        self.api.native_links[3] = [
            (10, REPOSITORY),
            (11, "other/repo"),
            (12, REPOSITORY),
        ]
        with self.controller.snapshots():
            refs = self.controller.native_references({3})
        self.assertEqual(set(refs), {10, 12})

    def test_native_query_failure_preserves_existing_associations(self):
        self.initialize()
        self.api.add_pr(10, "")
        self.api.native_links[3] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        self.api.native_failure = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_refresh_creates_no_directory_or_machine_comments(self):
        self.initialize()
        original = self.api.items[1]["body"]
        self.controller.reconcile(1)
        self.assertFalse(self.api.comments)
        self.assertFalse(self.api.internal_items)
        self.assertIn("**Bug Archive:** #2", original)
        self.assertIn("| #3 |", original)
        self.assertEqual(self.api.items[1]["body"], original)

    def test_human_checklist_edits_survive_display_refresh(self):
        state = self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "reason: Initial scope.",
            "reason: Human decision.\n\n### Custom Notes\n\n- [x] Human checklist",
        )
        self.controller.reconcile(1)
        self.assertIn("- [x] Human checklist", self.api.items[1]["body"])
        self.assertIn("reason: Human decision.", self.api.items[1]["body"])

    def test_only_open_release_is_pinned(self):
        state = self.initialize()
        self.assertEqual(self.api.pins, {1: "Bot"})
        self.assertNotIn(state["archive"], self.api.pins)
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(self.api.pins, {1: "Bot"})
        state = self.controller.reconcile(1)
        self.api.pins.clear()
        self.api.items[1]["state"] = "closed"
        self.controller.release_pin(1)
        self.assertFalse(self.api.pins)

    def test_dry_run_does_not_pin(self):
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertFalse(self.api.pins)

    def test_manual_pin_is_preserved_during_sync(self):
        self.api.pins[1] = "User"
        state = self.initialize()
        self.controller.release_pin(1)
        self.assertEqual(self.api.pins, {1: "User"})

    def test_manual_unpin_is_preserved(self):
        self.initialize()
        self.api.pins.clear()
        self.controller.initialize_or_update(1, "release_init")
        self.assertFalse(self.api.pins)

    def test_full_pin_slots_do_not_block_initialization(self):
        self.api.pins = {40: "User", 41: "Bot", 42: "User"}
        state = self.initialize()
        self.assertEqual(state["archive"], 2)
        self.assertNotIn(1, self.api.pins)

    def test_pin_api_failure_is_reported_without_blocking_initialization(self):
        self.api.pin_failure = True
        with patch("builtins.print") as log:
            self.assertEqual(self.initialize()["archive"], 2)
        self.assertTrue(
            any("Pin API unavailable" in str(call) for call in log.call_args_list)
        )

    def test_pr_comment_mentions_are_ignored(self):
        comment = self.api.comment(10, "Related #13")
        self.controller.handle(
            {"issue": {"number": 10, "pull_request": {}}, "comment": comment},
            "issue_comment",
        )
        self.assertFalse(self.api.reactions)
        self.assertFalse(self.api.internal_items)

    def test_pr_comment_edit_is_ignored(self):
        comment = self.api.comment(10, "Reference removed")
        self.controller.handle(
            {
                "action": "edited",
                "issue": {"number": 10, "pull_request": {}},
                "comment": comment,
                "changes": {"body": {"from": "Related #13"}},
            },
            "issue_comment",
        )
        self.assertFalse(self.api.reactions)

    def test_unrelated_bot_and_deleted_comments_have_no_reaction(self):
        self.controller.reconcile_all = lambda event: None
        for body, kind, action in (
            ("Hello", "User", "created"),
            ("Related #13", "Bot", "created"),
            ("Related #13", "User", "deleted"),
        ):
            comment = self.api.comment(10, body)
            comment["user"]["type"] = kind
            self.controller.handle(
                {
                    "action": action,
                    "issue": {"number": 10, "pull_request": {}},
                    "comment": comment,
                },
                "issue_comment",
            )
        self.assertFalse(self.api.reactions)

    def test_unauthorized_command_has_no_reaction(self):
        def reject(actor):
            raise OperationError("Permission denied")

        self.controller.authorize = reject
        command = self.api.comment(1, "/release_init")
        with self.assertRaises(OperationError):
            self.controller.handle(
                {
                    "issue": self.api.items[1],
                    "comment": command,
                    "sender": command["user"],
                },
                "issue_comment",
            )
        self.assertFalse(self.api.reactions)

    def setUp(self):
        self.api = MemoryGitHub()
        self.controller = Controller(
            self.api,
            {
                "repository": REPOSITORY,
                "projects": {
                    "Bugfix": {
                        "id": "P_BUG",
                        "fields": {
                            "Status": {
                                "id": "STATUS",
                                "options": [
                                    {"name": n, "id": n}
                                    for n in [
                                        "Ready",
                                        "In progress",
                                        "In review",
                                        "Done",
                                    ]
                                ],
                            }
                        },
                    }
                },
            },
        )

    def initialize(self):
        return self.controller.initialize_or_update(1, "release_init")

    def test_preview_performs_no_writes_and_apply_creates_flat_requirements(self):
        self.api.items[1]["body"] = PLAN.replace("[]", "[Issue]()")
        before = self.api.items[1]["body"]
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertEqual(set(self.api.items), {1})
        self.assertEqual(self.api.items[1]["body"], before)
        self.assertFalse(self.api.milestones)
        state = self.initialize()
        self.assertFalse(self.api.comments)
        self.assertEqual(self.api.parents, {state["archive"]: 1})
        self.assertIn("Acceptance criteria:", self.api.items[3]["body"])
        self.assertEqual(len(self.api.items), 4)

    def test_recovery_after_lost_create_response(self):
        self.api.items[1]["body"] = PLAN.replace(" []", "")
        self.api.lose_response_for = "Store interface"
        with self.assertRaises(OperationError):
            self.initialize()
        self.initialize()
        self.assertEqual(len(self.api.items), 4)
        self.assertEqual(len(self.api.milestones), 1)

    def test_remove_retains_issue_and_manual_milestone(self):
        self.initialize()
        self.api.items[4]["milestone"] = {"number": 22, "title": "0.13.0"}
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "### Change Log\n\nreason: Deferred.\n"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertNotIn("issue-4", self.controller.reconcile(1)["scope"])
        self.assertEqual(self.api.items[4]["milestone"]["number"], 22)

    def test_checklist_removal_leaves_milestone_as_source_of_scope(self):
        self.initialize()
        self.api.parents[4] = 3
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")] + "### Change Log\n"
        )
        state = self.controller.initialize_or_update(1, "update")
        self.assertIn("issue-4", state["scope"])
        self.assertEqual(self.api.items[4]["milestone"]["number"], 1)
        self.assertEqual(self.api.parents[4], 3)
        self.api.items[4]["milestone"] = None
        self.assertNotIn("issue-4", self.controller.reconcile(1)["scope"])

    def test_invalid_plan_creates_no_repository_objects(self):
        self.api.items[1]["body"] = PLAN.replace("  description:", " description:", 1)
        with self.assertRaises(OperationError):
            self.initialize()
        self.assertEqual(len(self.api.items), 1)
        self.assertFalse(self.api.milestones)

    def test_metadata_update_sets_manager_and_milestone_date(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "release_date: TBD", "release_date: 2026-12-01"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.milestones[0]["due_on"], "2026-12-01T23:59:59Z")
        self.assertEqual(self.api.items[1]["assignees"], [{"login": "harrisonyhq"}])

    def test_existing_issue_description_is_preserved(self):
        self.initialize()
        self.api.items[3]["body"] = "Existing discussion."
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.api.items[3]["body"], "Existing discussion.")

    def test_native_links_union_projects_and_unlink_preserves_milestone(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_CONNECTOR": "M_REQ"}
        self.api.projects["I_4"] = {"P_STORE": "M_REQ2"}
        self.api.add_pr(10, "")
        self.api.native_links.update({3: [(10, REPOSITORY)], 4: [(10, REPOSITORY)]})
        self.controller.reconcile(1)
        self.assertEqual(
            self.controller.reconcile(1)["prs"]["10"]["requirements"], [3, 4]
        )
        self.assertEqual(set(self.api.projects["I_10"]), {"P_CONNECTOR", "P_STORE"})
        self.api.native_links.clear()
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertTrue(self.api.projects["I_10"])
        self.assertNotIn("#10", self.api.items[1]["body"])

    def test_pr_comment_cannot_trigger_metadata_updates(self):
        self.initialize()
        self.api.add_pr(10, "")
        comment = self.api.comment(10, "Fixes #3")
        self.controller.handle(
            {"issue": {"number": 10, "pull_request": {}}, "comment": comment},
            "issue_comment",
        )
        self.assertIsNone(self.api.prs[10]["milestone"])

    def test_bot_receipts_are_excluded_from_mentions(self):
        self.initialize()
        self.api.add_pr(10, "")
        comment = self.api.comment(10, "Related #3")
        comment["user"]["type"] = "Bot"
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])

    def test_pr_state_change_refreshes_display_without_scope_revision(self):
        self.initialize()
        pr = self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        pr["merged_at"] = "2026-11-18T00:00:00Z"
        pr["state"] = "closed"
        self.controller.reconcile(1)
        self.assertIn("| merged", self.api.items[1]["body"])

    def test_conflicting_versions_leave_milestone_unassigned(self):
        self.initialize()
        self.api.items[4]["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.add_pr(10, "Fixes #3, fixes #4")
        state = self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertTrue(state["prs"]["10"]["conflict"])

    def test_empty_pr_milestone_is_refilled_when_target_is_unique(self):
        self.initialize()
        self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        self.api.prs[10]["milestone"] = None
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_lost_milestone_response_recovers_automatic_ownership(self):
        self.initialize()
        self.api.add_pr(10, "Fixes #3")
        self.api.lose_milestone_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_lost_project_response_reuses_existing_item_on_retry(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "Fixes #3")
        self.api.lose_project_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        projects = dict(self.api.projects["I_10"])
        self.controller.reconcile(1)
        self.assertEqual(self.api.projects["I_10"], projects)

    def test_native_bug_parent_assigns_version_without_source_field(self):
        state = self.initialize()
        self.api.items[8] = self.api.make_issue(8, "Bug", "Observed failure.", [])
        self.api.parents[8] = state["archive"]
        self.controller.reconcile(1)
        self.assertEqual(
            self.api.items[8]["milestone"]["number"], state["milestone"]["number"]
        )
        self.assertIn({"name": "bug"}, self.api.items[8]["labels"])
        self.assertIn("P_BUG", self.api.projects["I_8"])

    def test_maintainer_closed_features_need_no_verification_records(self):
        self.initialize()
        for n in (1, 2, 3, 4):
            self.api.items[n]["state"] = "closed"
        state, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertNotIn("verification", state)
        self.assertNotIn(
            "Pending", self.api.items[1]["body"].split("### Release Check")[0]
        )

    def test_unassigned_bug_does_not_block_other_versions(self):
        self.initialize()
        self.api.items[99] = self.api.make_issue(99, "Unassigned", "", ["bug"])
        _, report = self.controller.release_check(1)
        self.assertNotIn(99, report["open_issues"])

    def test_bug_status_matches_merge_and_issue_state(self):
        self.assertEqual(
            bug_progress(False, {"9"}, True), ("status:verification", "In review")
        )
        self.assertEqual(bug_progress(True, {"9"}, True), ("status:verified", "Done"))


if __name__ == "__main__":
    unittest.main()
