import copy
import re
import unittest
from pathlib import Path

from controller import Controller, GitHub, bug_progress, manual_mode
from presentation import NAV_START, OVERVIEW_END, OVERVIEW_START, release_body
from syntax import (
    OperationError,
    bind_entries,
    field,
    issue_reference,
    parse_release,
    patch_pull_requests,
    patch_references,
    scope_diff,
)

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
        self.assertEqual(len(parse_release(example, REPOSITORY)["entries"]), 1)
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
        previous = {e["key"]: e for e in bind_entries(entries, mappings)}
        self.assertFalse(any(scope_diff(previous, actual).values()))

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
        self.assertFalse(
            any(
                scope_diff(
                    {"issue-14": dict(entry, key="issue-14", number=14)},
                    [parse_release(patched, REPOSITORY)["entries"][0]],
                ).values()
            )
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
            values = self.projects.get(variables["id"], {})
            return {
                "node": {
                    "projectItems": {
                        "nodes": [
                            {"id": item, "project": {"id": p}}
                            for p, item in values.items()
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
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
    def test_plain_titles_create_issues_and_write_back_multiline_references(self):
        self.api.items[1]["body"] = PLAN.replace(" []", "")
        state = self.initialize()
        self.assertEqual(set(state["scope"]), {"issue-3", "issue-4"})
        self.assertIn(
            "- Connector v2\n  initial integration #3", self.api.items[1]["body"]
        )
        self.assertIn("- Store interface #4", self.api.items[1]["body"])
        self.assertIn("Acceptance criteria:", self.api.items[3]["body"])
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(len(self.api.items), 4)
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_update_creates_plain_title_requirement_once(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "### Change Log",
            "- New feature\n\n  description:\n    New work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log",
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIn("- New feature #5", self.api.items[1]["body"])
        self.assertEqual(self.api.items[5]["title"], "New feature")
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(len(self.api.items), 5)
        self.assertEqual(self.controller.load(1)[1]["revision"], 2)

    def test_description_trailing_whitespace_keeps_check_and_revision_stable(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "Implement the first interface.", "Implement the first interface. "
        )
        _, report = self.controller.release_check(1)
        self.assertFalse(any(report["unsynced"].values()))
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)
        self.assertIn("Implement the first interface. \n", self.api.items[1]["body"])

    def test_description_content_change_is_pending_update(self):
        self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "Implement the first interface.", "Implement the revised interface. "
        )
        _, report = self.controller.release_check(1)
        self.assertEqual(report["unsynced"]["changed"], ["#4"])

    def test_release_check_lists_open_features_bugs_and_missing_prs(self):
        self.initialize()
        bug = self.api.make_issue(8, "Open target Bug", "", ["bug"])
        bug["milestone"] = self.api.milestones[0]
        self.api.items[8] = bug
        _, report = self.controller.release_check(1)
        self.assertEqual(report["open_features"], [3, 4])
        self.assertEqual(report["open_bugs"], [8])
        self.assertEqual(report["features_without_pr"], [3, 4])
        self.assertEqual(report["status"], "blocked")
        self.assertIn("Feature Issues without PRs", self.api.items[1]["body"])
        self.assertNotIn('"status": "stale"', self.api.items[1]["body"])

    def test_release_check_distinguishes_open_and_closed_unmerged_prs(self):
        self.initialize()
        self.api.add_pr(10, "Fixes #3")
        self.api.add_pr(11, "Fixes #4")["state"] = "closed"
        _, report = self.controller.release_check(1)
        self.assertEqual(report["unmerged_prs"], [10])
        self.assertEqual(report["closed_unmerged_prs"], [11])
        self.assertFalse(report["features_without_pr"])

    def test_release_check_describes_pending_update_without_creating_scope(self):
        self.initialize()
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "- New feature []\n\n  description:\n    Planned work.\n\n  assignee:\n\n  pull_requests:\n\n### Change Log\n\nreason: Scope adjustment.\n"
        )
        count = len(self.api.items)
        _, report = self.controller.release_check(1)
        self.assertEqual(report["unsynced"]["removed"], ["#4"])
        self.assertEqual(
            report["unsynced"]["added"], ["New feature (Issue not created)"]
        )
        self.assertEqual(len(self.api.items), count)
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_check_uses_merge_state_without_commit_comparison(self):
        self.initialize()
        for n in (3, 4):
            self.api.items[n]["state"] = "closed"
        self.api.add_pr(10, "Fixes #3").update(state="closed", merged_at="2026-11-19")
        state, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertNotIn("prs_outside_candidate", report)
        self.assertNotIn("verification", state)

    def test_release_check_command_does_not_replay_itself(self):
        self.initialize()
        comment = self.api.comment(1, "/release_check")
        self.controller.handle(
            {"issue": {"number": 1}, "comment": comment, "sender": comment["user"]},
            "issue_comment",
        )
        state = self.controller.load(1)[1]
        self.assertIn(str(comment["id"]), state["processed_comments"])
        self.assertIn("✅ [Release check]", self.api.items[1]["body"])

    def test_verify_command_is_retired_and_does_not_close_issue(self):
        self.initialize()
        comment = self.api.comment(1, "/verify #3 passed https://example.com/report")
        self.controller.handle(
            {"issue": {"number": 1}, "comment": comment, "sender": comment["user"]},
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
        other = self.controller.load(50)[1]
        bug = self.api.make_issue(8, "Retargeted Bug", "", ["bug"])
        bug["milestone"] = self.api.milestones[0]
        self.api.items[8] = bug
        self.api.parents[8] = other["archive"]
        self.api.add_pr(10, "Fixes #8")
        _, report = self.controller.release_check(1)
        self.assertIn(8, report["open_bugs"])
        self.assertIn(10, report["unmerged_prs"])

    def test_operation_status_runs_then_succeeds_and_old_comment_is_hidden(self):
        observed = []
        original = self.controller.sync_release_body

        def capture(release, state):
            observed.append(
                (
                    state.get("operation", {}).get("status"),
                    state.get("operation", {}).get("kind"),
                )
            )
            return original(release, state)

        self.controller.sync_release_body = capture
        self.initialize()
        body = self.api.items[1]["body"]
        self.assertTrue(any(status == "running" for status, _ in observed))
        self.assertIn("✅ [Release initialization]", body)
        self.assertNotIn("🔄", body)
        self.assertEqual({kind for _, kind in observed}, {"Release initialization"})
        comment, state = self.controller.load(1)
        self.assertNotIn("版本操作结果", comment["body"])
        self.assertNotIn("同步完成", comment["body"])
        self.assertIn(comment["node_id"], self.api.minimized_comments)
        self.assertTrue(state["checkpoint_minimized"])

    def test_failure_status_has_short_error_and_retry_becomes_green(self):
        self.api.items[1]["body"] = PLAN.replace("version: 0.12.0", "version: invalid")
        with self.assertRaises(OperationError):
            self.controller.initialize_or_update(1, "release_init")
        self.assertIn("❌ [Release initialization]", self.api.items[1]["body"])
        self.assertEqual(self.controller.load(1)[1]["operation"]["status"], "failure")
        self.api.items[1]["body"] = PLAN
        self.initialize()
        self.assertIn("✅ [Release initialization]", self.api.items[1]["body"])
        self.assertNotIn("❌", self.api.items[1]["body"])

    def test_latest_sync_replaces_status_without_appending_lines(self):
        self.initialize()
        self.controller.run_url = "https://example.com/runs/new"
        self.controller.reconcile(1)
        body = self.api.items[1]["body"]
        self.assertIn(
            "✅ [Release synchronization](https://example.com/runs/new)", body
        )
        self.assertNotIn("✅ [Release initialization]", body)
        self.assertEqual(body.count("✅ ["), 1)

    def test_preview_plan_moves_to_body_and_retains_single_checkpoint(self):
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertIn("✅ [Release initialization preview]", self.api.items[1]["body"])
        self.assertIn("<summary>Preview plan</summary>", self.api.items[1]["body"])
        self.initialize()
        self.assertNotIn("<summary>Preview plan</summary>", self.api.items[1]["body"])
        self.assertEqual(len(self.api.comments), 1)

    def test_finalizer_marks_interrupted_current_run_and_preserves_other_runs(self):
        self.initialize()
        _, state = self.controller.load(1)
        state["operation"]["status"] = "running"
        self.controller.save(1, state, "Interrupted")
        self.controller.finalize_operations("failure")
        self.assertIn("❌", self.api.items[1]["body"])
        state = self.controller.load(1)[1]
        state["operation"].update(url="https://example.com/other", status="running")
        self.controller.save(1, state, "Other run")
        self.controller.finalize_operations("failure")
        self.assertEqual(self.controller.load(1)[1]["operation"]["status"], "running")

    def test_minimize_failure_keeps_checkpoint_and_operation_success(self):
        self.api.minimize_failure = True
        state = self.initialize()
        self.assertEqual(state["operation"]["status"], "success")
        self.assertTrue(state["initialized"])

    def test_native_development_link_updates_metadata_and_table(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_NATIVE": "REQ"}
        self.api.add_pr(10, "No textual references")
        self.api.native_links[3] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        state = self.controller.load(1)[1]
        self.assertEqual(state["prs"]["10"]["requirements"], [3])
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertIn("P_NATIVE", self.api.projects["I_10"])
        self.assertIn("| #3 | #10 (open)", self.api.items[1]["body"])
        self.api.native_links[3] = []
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertNotIn("P_NATIVE", self.api.projects["I_10"])
        self.assertNotIn("#10", self.api.items[1]["body"])

    def test_ordinary_mentions_do_not_supply_associations(self):
        self.initialize()
        self.api.add_pr(10, "Related #3")
        self.api.comment(10, "Implements #4")
        self.api.reviews[10] = [{"body": "Fixes #3", "user": {"type": "User"}}]
        self.controller.reconcile(1)
        self.assertNotIn("10", self.controller.load(1)[1]["prs"])
        self.assertIsNone(self.api.prs[10]["milestone"])

    def test_native_links_keep_merged_pr_and_closed_status(self):
        self.initialize()
        pr = self.api.add_pr(10, "")
        self.api.native_links[3] = [(10, REPOSITORY)]
        pr.update(state="closed", merged_at="2026-11-19T00:00:00Z")
        self.controller.reconcile(1)
        self.assertTrue(self.api.native_include_closed)
        self.assertIn("| #3 | #10 (merged)", self.api.items[1]["body"])

    def test_native_bug_link_uses_fix_target_and_bugfix_project(self):
        state = self.initialize()
        bug = self.api.make_issue(8, "Native Bug", "", ["bug"])
        bug["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.items[8] = bug
        self.api.parents[8] = state["archive"]
        self.api.add_pr(10, "")
        self.api.native_links[8] = [(10, REPOSITORY)]
        self.controller.reconcile(1)
        record = self.controller.load(1)[1]["prs"]["10"]
        self.assertEqual(record["bugs"], [8])
        self.assertEqual(record["last"], 2)
        self.assertIn("P_BUG", self.api.projects["I_10"])

    def test_native_link_pagination_and_repository_filter(self):
        self.api.native_links[3] = [
            (10, REPOSITORY),
            (11, "other/repo"),
            (12, REPOSITORY),
        ]
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

    def test_comment_is_compact_and_body_contains_managed_overview(self):
        self.initialize()
        comment, state = self.controller.load(1)
        self.assertNotIn("| Issue |", comment["body"])
        self.assertNotIn("主 Bug Issue：", comment["body"])
        self.assertNotIn("Milestone：[目标版本]", comment["body"])
        self.assertNotIn("缺陷与修复 PR", comment["body"])
        body = self.api.items[1]["body"]
        self.assertIn("**Bug Archive:** #2", body)
        self.assertIn("| #3 |", body)
        self.assertTrue(body.rstrip().endswith(OVERVIEW_END))
        self.controller.save(1, state, "Refreshed")
        self.assertEqual(self.api.items[1]["body"], body)

    def test_human_checklist_edits_survive_display_refresh(self):
        state = self.initialize()
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "reason: Initial scope.",
            "reason: Human decision.\n\n### Custom Notes\n\n- [x] Human checklist",
        )
        self.controller.save(1, state, "Refreshed")
        self.assertIn("- [x] Human checklist", self.api.items[1]["body"])
        self.assertIn("reason: Human decision.", self.api.items[1]["body"])

    def test_only_open_release_is_pinned(self):
        state = self.initialize()
        self.assertEqual(self.api.pins, {1: "Bot"})
        self.assertNotIn(state["archive"], self.api.pins)
        self.controller.initialize_or_update(1, "release_init")
        self.assertEqual(self.api.pins, {1: "Bot"})
        state = self.controller.load(1)[1]
        self.api.pins.clear()
        self.api.items[1]["state"] = "closed"
        self.controller.release_pin(1, state)
        self.assertFalse(self.api.pins)

    def test_dry_run_does_not_pin(self):
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertFalse(self.api.pins)

    def test_manual_pin_is_preserved_during_sync(self):
        self.api.pins[1] = "User"
        state = self.initialize()
        self.controller.release_pin(1, state)
        self.assertEqual(self.api.pins, {1: "User"})

    def test_manual_unpin_is_preserved(self):
        self.initialize()
        self.api.pins.clear()
        self.controller.initialize_or_update(1, "release_init")
        self.assertFalse(self.api.pins)

    def test_full_pin_slots_do_not_block_initialization(self):
        self.api.pins = {40: "User", 41: "Bot", 42: "User"}
        state = self.initialize()
        self.assertTrue(state["initialized"])
        self.assertIn("three", state["pin_warning"])
        self.assertNotIn(1, self.api.pins)
        self.api.pins.pop(42)
        self.controller.initialize_or_update(1, "release_init")
        self.assertIn(1, self.api.pins)

    def test_pin_api_failure_is_reported_without_blocking_initialization(self):
        self.api.pin_failure = True
        state = self.initialize()
        self.assertTrue(state["initialized"])
        self.assertIn("Pin API unavailable", self.api.items[1]["body"])

    def test_command_acknowledges_before_start_and_reuses_reaction(self):
        command = self.api.comment(1, "/release_init --dry-run")
        original = self.controller.initialize_or_update

        def execute(*args, **kwargs):
            self.assertEqual(self.api.reactions[command["id"]]["content"], "eyes")
            return original(*args, **kwargs)

        self.controller.initialize_or_update = execute
        event = {"issue": {"number": 1}, "comment": command, "sender": command["user"]}
        self.controller.handle(event, "issue_comment")
        self.controller.handle(event, "issue_comment")
        self.assertEqual(len(self.api.reactions), 1)

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

    def test_reaction_failure_does_not_block_command(self):
        self.api.reaction_failure = True
        command = self.api.comment(1, "/release_init --dry-run")
        self.controller.handle(
            {"issue": {"number": 1}, "comment": command, "sender": command["user"]},
            "issue_comment",
        )
        self.assertIn(
            str(command["id"]), self.controller.load(1)[1]["processed_comments"]
        )

    def test_unauthorized_command_has_no_reaction(self):
        def reject(actor):
            raise OperationError("Permission denied")

        self.controller.authorize = reject
        command = self.api.comment(1, "/release_init")
        with self.assertRaises(OperationError):
            self.controller.handle(
                {"issue": {"number": 1}, "comment": command, "sender": command["user"]},
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
        self.controller.initialize_or_update(1, "release_init")
        return self.controller.load(1)[1]

    def test_preview_then_apply_replaces_result_and_creates_flat_requirements(self):
        self.api.items[1]["body"] = PLAN.replace("[]", "[Issue]()")
        self.controller.initialize_or_update(1, "release_init", True)
        self.assertEqual(len(self.api.items), 1)
        self.assertFalse(self.api.milestones)
        cid = self.api.comments[0]["id"]
        state = self.initialize()
        self.assertEqual(len(self.api.comments), 1)
        self.assertEqual(self.api.comments[0]["id"], cid)
        self.assertNotIn("预览", self.api.comments[0]["body"])
        self.assertEqual(self.api.parents, {state["archive"]: 1})
        self.assertEqual(set(state["scope"]), {"issue-3", "issue-4"})
        self.assertIn("Acceptance criteria:", self.api.items[3]["body"])
        self.assertEqual(len(self.api.items), 4)
        self.controller.initialize_or_update(1, "update")
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

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
        self.assertEqual(set(self.controller.load(1)[1]["scope"]), {"issue-3"})
        self.assertEqual(self.api.items[4]["milestone"]["number"], 22)

    def test_remove_clears_owned_milestone_and_retains_native_parent(self):
        self.initialize()
        self.api.parents[4] = 3
        body = self.api.items[1]["body"]
        self.api.items[1]["body"] = (
            body[: body.index("- Store interface")]
            + "### Change Log\n\nreason: Deferred.\n"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIsNone(self.api.items[4]["milestone"])
        self.assertEqual(self.api.parents[4], 3)
        self.assertEqual(self.api.items[4]["state"], "open")

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
        self.assertEqual(self.controller.load(1)[1]["revision"], 2)

    def test_existing_issue_description_is_preserved_and_assignee_can_clear(self):
        self.initialize()
        self.api.items[3]["body"] += "\nManual implementation notes."
        self.api.items[1]["body"] = self.api.items[1]["body"].replace(
            "assignee: @harrisonyhq", "assignee:"
        )
        self.controller.initialize_or_update(1, "update")
        self.assertIn("Manual implementation notes.", self.api.items[3]["body"])
        self.assertEqual(self.api.items[3]["assignees"], [])

    def test_pending_commands_are_replayed_in_order(self):
        preview = self.api.comment(1, "/release_init --dry-run")
        apply = self.api.comment(1, "/release_init")
        self.controller.handle(
            {"issue": {"number": 1}, "comment": apply, "sender": apply["user"]},
            "issue_comment",
        )
        state = self.controller.load(1)[1]
        self.assertTrue(state["initialized"])
        self.assertEqual(
            state["processed_comments"], [str(preview["id"]), str(apply["id"])]
        )

    def test_native_links_union_projects_and_unlink_preserves_milestone(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_CONNECTOR": "M_REQ"}
        self.api.projects["I_4"] = {"P_STORE": "M_REQ2"}
        self.api.add_pr(10, "")
        self.api.native_links.update({3: [(10, REPOSITORY)], 4: [(10, REPOSITORY)]})
        self.controller.reconcile(1)
        self.assertEqual(
            self.controller.load(1)[1]["prs"]["10"]["requirements"], [3, 4]
        )
        self.assertEqual(set(self.api.projects["I_10"]), {"P_CONNECTOR", "P_STORE"})
        self.api.native_links.clear()
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)
        self.assertFalse(self.api.projects["I_10"])
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
        self.assertEqual(self.controller.load(1)[1]["revision"], 1)

    def test_conflicting_versions_and_manual_metadata_are_preserved(self):
        self.initialize()
        self.api.items[4]["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "Fixes #3, fixes #4")
        self.api.projects["I_10"] = {"P_MANUAL": "MANUAL"}
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertIn("conflicting", self.api.items[1]["body"])
        self.assertEqual(set(self.api.projects["I_10"]), {"P_REQ", "P_MANUAL"})
        self.api.native_links.clear()
        self.controller.reconcile(1)
        self.assertEqual(self.api.projects["I_10"], {"P_MANUAL": "MANUAL"})

    def test_manual_milestone_clear_and_project_remove_persist(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "Fixes #3")
        self.controller.reconcile(1)
        self.api.prs[10]["milestone"] = None
        self.api.projects["I_10"].clear()
        self.controller.reconcile(1)
        self.assertIsNone(self.api.prs[10]["milestone"])
        self.assertFalse(self.api.projects["I_10"])
        self.assertIn(10, self.controller.directory().data["seen_prs"])

    def test_lost_milestone_response_recovers_automatic_ownership(self):
        self.initialize()
        self.api.add_pr(10, "Fixes #3")
        self.api.lose_milestone_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.controller.reconcile(1)
        self.assertEqual(self.api.prs[10]["milestone"]["number"], 1)

    def test_lost_project_response_recovers_item_identity(self):
        self.initialize()
        self.api.projects["I_3"] = {"P_REQ": "REQ"}
        self.api.add_pr(10, "Fixes #3")
        self.api.lose_project_response = True
        with self.assertRaises(OperationError):
            self.controller.reconcile(1)
        self.controller.reconcile(1)
        record = self.controller.load(1)[1]["prs"]["10"]["projects"]["P_REQ"]
        self.assertEqual(record["item"], self.api.projects["I_10"]["P_REQ"])
        self.api.projects["I_10"].clear()
        self.controller.reconcile(1)
        self.assertFalse(self.api.projects["I_10"])

    def test_bugfix_mention_uses_target_version_and_bugfix_project(self):
        state = self.initialize()
        bug = self.api.make_issue(8, "Bug", "", ["bug"])
        bug["milestone"] = {"number": 2, "title": "0.13.0"}
        self.api.items[8] = bug
        self.api.parents[8] = state["archive"]
        self.api.add_pr(10, "Fixes #8")
        self.controller.reconcile(1)
        self.assertIn("P_BUG", self.api.projects["I_10"])
        self.assertEqual(self.controller.load(1)[1]["prs"]["10"]["last"], 2)
        self.assertEqual(self.api.parents[8], state["archive"])

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
        for n in (3, 4):
            self.api.items[n]["state"] = "closed"
        state, report = self.controller.release_check(1)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["features_without_pr"], [3, 4])
        self.assertNotIn("verification", state)
        self.assertNotIn(
            "Pending", self.api.items[1]["body"].split("### Release Check")[0]
        )

    def test_unassigned_bug_does_not_block_other_versions(self):
        self.initialize()
        self.api.items[99] = self.api.make_issue(99, "Unassigned", "", ["bug"])
        _, report = self.controller.release_check(1)
        self.assertNotIn(99, report["open_bugs"])

    def test_manual_milestone_ownership_and_bug_status(self):
        self.assertEqual(
            manual_mode({"mode": "auto", "last": 1}, None)["mode"], "manual"
        )
        self.assertEqual(
            manual_mode({"mode": "auto", "last": 1}, 1, True)["mode"], "manual"
        )
        self.assertEqual(
            bug_progress(False, {"9"}, True), ("status:verification", "In review")
        )
        self.assertEqual(bug_progress(True, {"9"}, True), ("status:verified", "Done"))


if __name__ == "__main__":
    unittest.main()
