"""Release operations driven by GitHub Issue and Milestone state."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from associations import PullRequestAssociations
from checks import ReleaseChecks
from delivery import ReleaseDelivery
from ownership import milestone_number
from pinning import ReleasePinning
from presentation import release_body, report_result
from syntax import (
    OperationError,
    bind_entries,
    parse_release,
    patch_pull_requests,
    patch_references,
)

ORIGIN = "<!-- ucm-origin:{release}:{key} -->"
ARCHIVE_LABEL = "release-bug-archive"


def has_label(issue, label):
    return any(item["name"] == label for item in issue.get("labels", []))


def bug_progress(closed: bool, merged: set[str], has_pr: bool) -> tuple[str, str]:
    if closed:
        return "status:verified", "Done"
    if merged:
        return "status:verification", "In review"
    if has_pr:
        return "status:fixing", "In progress"
    return "status:triage", "Ready"


class GitHub:
    def __init__(self, repository: str):
        self.repository = repository
        self.requests = 0
        self.max_calls = 200
        self.started = time.monotonic()
        self.last_write = None

    def budget(self):
        if self.requests >= self.max_calls or time.monotonic() - self.started > 600:
            raise OperationError(
                "Version operation reached its API/time budget; resume the next run"
            )
        self.requests += 1

    def call(self, path: str, method="GET", data=None, project=False):
        self.budget()
        args = ["gh", "api", path, "--method", method]
        if data is not None:
            args += ["--input", "-"]
        env = dict(os.environ)
        if project:
            token = env.get("UCM_PROJECT_TOKEN")
            if not token:
                raise OperationError("配置 UCM_PROJECT_TOKEN 后启用 Project 同步")
            env["GH_TOKEN"] = token
        mutating = method != "GET" and (
            path != "graphql"
            or (data or {}).get("query", "").lstrip().startswith("mutation")
        )
        if mutating and self.last_write is not None:
            time.sleep(max(0, 1 - (time.monotonic() - self.last_write)))
        result = subprocess.run(
            args,
            input=None if data is None else json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
            text=True,
            capture_output=True,
            env=env,
            timeout=90,
        )
        if mutating:
            self.last_write = time.monotonic()
        if result.returncode:
            raise OperationError(result.stderr.strip() or result.stdout.strip())
        parsed = json.loads(result.stdout) if result.stdout.strip() else None
        if isinstance(parsed, dict) and parsed.get("errors"):
            raise OperationError(json.dumps(parsed["errors"], ensure_ascii=False))
        return parsed

    def rest(self, path: str, method="GET", data=None):
        endpoint = "repos/" + self.repository
        return self.call(endpoint + ("/" + path if path else ""), method, data)

    def pages(self, path: str) -> list:
        items = []
        for page in range(1, 101):
            part = self.rest(
                path + ("&" if "?" in path else "?") + f"per_page=100&page={page}"
            )
            if not isinstance(part, list):
                raise OperationError("预期分页数组：" + path)
            items.extend(part)
            if len(part) < 100:
                return items
        raise OperationError("分页超出处理上限")

    def graphql(self, query: str, **variables):
        return self.call(
            "graphql", "POST", {"query": query, "variables": variables}, project=True
        )["data"]

    def artifact_json(self, artifact_id, filename):
        self.budget()
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{self.repository}/actions/artifacts/{artifact_id}/zip",
            ],
            capture_output=True,
            timeout=90,
        )
        if result.returncode:
            raise OperationError(result.stderr.decode("utf-8", errors="replace"))
        if len(result.stdout) > 2_000_000:
            raise OperationError("Delivery receipt archive is too large")
        with ZipFile(BytesIO(result.stdout)) as archive:
            files = [i for i in archive.infolist() if i.filename == filename]
            if len(files) != 1 or files[0].file_size > 2_000_000:
                raise OperationError("Delivery receipt archive has invalid contents")
            return json.loads(archive.read(files[0]))


class Controller(
    PullRequestAssociations, ReleasePinning, ReleaseChecks, ReleaseDelivery
):
    def __init__(self, api: GitHub, config: dict):
        self.api, self.config = api, config
        self.repository = config["repository"]
        self.api.max_calls = int(config.get("sync", {}).get("max_api_calls", 200))
        self._running = False

    @contextmanager
    def snapshots(self):
        if self._running:
            yield
            return
        self._running = True
        self._pull_requests, self._pr_issues = {}, {}
        self._native_issue_snapshot = {}
        self._project_memberships = None
        self._creation_snapshot = {}
        self._checklists = {}
        self._milestone_items = {}
        self._archives = {}
        try:
            yield
        finally:
            self._running = False

    def issue(self, number):
        issue = self.api.rest(f"issues/{number}")
        if "pull_request" in issue:
            raise OperationError(f"#{number} 是 PR，请引用 Issue")
        return issue

    def checklists(self, milestone):
        if milestone not in self._checklists:
            self._checklists[milestone] = [
                item
                for item in self.api.pages(
                    f"issues?state=all&labels=release&milestone={milestone}"
                )
                if "pull_request" not in item
            ]
        return self._checklists[milestone]

    def milestone_items(self, milestone):
        if milestone not in self._milestone_items:
            self._milestone_items[milestone] = self.api.pages(
                f"issues?state=all&milestone={milestone}"
            )
        return self._milestone_items[milestone]

    def archive(self, release):
        if release not in self._archives:
            archives = [
                issue
                for issue in self.api.pages(f"issues/{release}/sub_issues")
                if has_label(issue, ARCHIVE_LABEL)
            ]
            if len(archives) > 1:
                raise OperationError(f"Multiple Bug Archives for Release #{release}")
            self._archives[release] = next(iter(archives), None)
        return self._archives[release]

    def create_or_recover(self, release, milestone, key, title, body, label):
        marker = ORIGIN.format(release=release, key=key)
        cache_key = (milestone, label)
        if cache_key not in self._creation_snapshot:
            self._creation_snapshot[cache_key] = self.api.pages(
                f"issues?state=all&labels={label}&milestone={milestone}"
            )
        candidates = self._creation_snapshot[cache_key]
        matches = [
            item
            for item in candidates
            if "pull_request" not in item and marker in (item.get("body") or "")
        ]
        if len(matches) > 1:
            raise OperationError(f"Multiple Issues match the origin {key}")
        if matches:
            return matches[0]
        item = self.api.rest(
            "issues",
            "POST",
            {
                "title": title,
                "body": marker + "\n\n" + body,
                "labels": [label],
                "milestone": milestone,
            },
        )
        candidates.append(item)
        return item

    def patch_issue(self, issue, fields):
        current = dict(
            issue,
            milestone=milestone_number(issue),
            assignees=[person["login"] for person in issue.get("assignees", [])],
        )
        changed = {
            key: value for key, value in fields.items() if value != current.get(key)
        }
        if not changed:
            return issue
        return self.api.rest(f"issues/{issue['number']}", "PATCH", changed)

    def parent(self, issue_number: int):
        # GET /parent returns 404 for an Issue with no parent.
        try:
            return self.api.rest(f"issues/{issue_number}/parent")
        except OperationError as error:
            if "404" in str(error):
                return None
            raise

    def attach(self, child: dict, parent: int, old_parent: int | None = None):
        current = self.parent(child["number"])
        if current and current["number"] == parent:
            return
        if current and current["number"] != old_parent:
            raise OperationError(
                f"#{child['number']} 已属于父需求 #{current['number']}"
            )
        self.api.rest(
            f"issues/{parent}/sub_issues",
            "POST",
            {"sub_issue_id": child["id"], "replace_parent": bool(current)},
        )
        actual = self.parent(child["number"])
        if not actual or actual["number"] != parent:
            raise OperationError("父关系回读失败")

    def project_item(
        self,
        issue: dict,
        project_name: str,
        status=None,
    ):
        project = self.config["projects"][project_name]
        current = self.memberships(issue)
        if project["id"] not in current:
            added = self.api.graphql(
                """mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
                project=project["id"],
                content=issue["node_id"],
            )["addProjectV2ItemById"]["item"]
            current[project["id"]] = added["id"]
        item_id = current[project["id"]]
        if status and self._project_status.get(item_id) != status:
            spec = project.get("fields", {}).get("Status", {})
            options = {o["name"]: o["id"] for o in spec.get("options", [])}
            if status not in options:
                raise OperationError(
                    f"Project {project_name}: missing Status option {status}"
                )
            self.api.graphql(
                """mutation($input:UpdateProjectV2ItemFieldValueInput!){updateProjectV2ItemFieldValue(input:$input){projectV2Item{id}}}""",
                input={
                    "projectId": project["id"],
                    "itemId": item_id,
                    "fieldId": spec["id"],
                    "value": {"singleSelectOptionId": options[status]},
                },
            )
            self._project_status[item_id] = status

    def inspect_release(self, source):
        release = source["number"]
        target = milestone_number(source)
        if not has_label(source, "release") or target is None:
            raise OperationError(
                f"Configure the Release Checklist #{release} Milestone first"
            )
        archive = self.archive(release)
        if not archive or milestone_number(archive) != target:
            raise OperationError(
                f"Configure the Bug Archive of Release #{release} first"
            )
        milestone = source["milestone"]
        if not milestone.get("html_url"):
            milestone = self.api.rest(f"milestones/{target}")
        return {
            "release": release,
            "version": milestone["title"],
            "milestone": milestone,
            "archive": archive["number"],
            "scope": {},
            "bugs": {},
            "prs": {},
            "observed": {},
        }

    def initialize_or_update(self, release, command, dry_run=False):
        with self.snapshots():
            return self.apply_plan(release, command, dry_run)

    def apply_plan(self, release, command, dry_run):
        source = self.issue(release)
        if not has_label(source, "release"):
            raise OperationError("版本入口需要 release 标签")
        parsed = parse_release(source["body"], self.repository, read_pr_requests=True)
        entries = parsed["entries"]
        target = milestone_number(source)
        archive = self.archive(release)
        initialized = (
            target is not None and archive and milestone_number(archive) == target
        )
        if command == "update" and not initialized:
            raise OperationError("先执行 /release_init")
        if not initialized and not entries:
            raise OperationError("初始化至少填写一个需求")
        version = parsed["version"]
        if target is not None and source["milestone"]["title"] != version:
            raise OperationError("版本号已绑定，请创建另一版本入口")
        if target is None:
            matches = [
                item
                for item in self.api.pages("milestones?state=all")
                if item["title"] == version
            ]
            if len(matches) > 1:
                raise OperationError(f"Multiple Milestones for version {version}")
            target = matches[0]["number"] if matches else None
        if target is not None and any(
            item["number"] != release for item in self.checklists(target)
        ):
            raise OperationError(f"Milestone {version} already has a Release Checklist")

        metadata = parsed["metadata"]
        logins = {
            login.lstrip("@")
            for login in (
                metadata.get("release_manager"),
                metadata.get("test_lead"),
                *(entry["developer"] for entry in entries),
            )
            if login
        }
        for login in sorted(logins):
            self.api.call(f"users/{login}")
        self.api.rest(f"commits/{metadata['branch']}")
        requested = {pr for entry in entries for pr in entry["requested_prs"]}
        if requested:
            self.pull_requests(requested)

        existing = {}
        for entry in entries:
            number = entry["number"]
            if number is None:
                continue
            if number == release or (archive and number == archive["number"]):
                raise OperationError("管理容器请使用独立条目")
            item = self.issue(number)
            owner = milestone_number(item)
            if owner is not None and (
                owner != target or item["milestone"]["title"] != version
            ):
                raise OperationError(f"#{number} 已关联其他 Milestone，请维护者处理")
            existing[number] = item

        if dry_run:
            report_result(
                f"Preview /{command}",
                [
                    {
                        "issue": entry["number"],
                        "title": entry["title"],
                        "requested_prs": entry["requested_prs"],
                    }
                    for entry in entries
                ],
            )
            return
        if self.issue(release)["body"] != source["body"]:
            raise OperationError("Release body changed; retry the command")

        milestone_fields = {
            "description": (
                f"Release tracking: #{release}\n"
                f"Release manager: {metadata.get('release_manager', '')}\n"
                f"Test lead: {metadata.get('test_lead', '')}"
            ),
            "due_on": (
                metadata["release_date"] + "T23:59:59Z"
                if metadata.get("release_date") not in (None, "", "TBD")
                else None
            ),
        }
        if target is None:
            milestone = self.api.rest(
                "milestones", "POST", dict(milestone_fields, title=version)
            )
            target = milestone["number"]
        else:
            milestone = self.api.rest(f"milestones/{target}")
            if milestone.get("state") == "closed":
                raise OperationError(f"Milestone {version} is closed")
            if any(milestone.get(key) != val for key, val in milestone_fields.items()):
                milestone = self.api.rest(
                    f"milestones/{target}", "PATCH", milestone_fields
                )
        fields = {"milestone": target}
        if metadata.get("release_manager"):
            fields["assignees"] = [metadata["release_manager"].lstrip("@")]
        source = self.patch_issue(source, fields)
        self._checklists[target] = [source]

        if not archive:
            archive = self.create_or_recover(
                release,
                target,
                "__bugs__",
                f"[Bug Archive] UCM {version}",
                f"Source version: {version}\nRelease: #{release}\n\n"
                f"Create Bug sub-issues here. The default fix version is {version}.",
                ARCHIVE_LABEL,
            )
            self.attach(archive, release)
        archive = self.patch_issue(archive, {"milestone": target})
        self._archives[release] = archive

        objects = {}
        for entry in entries:
            number = entry["number"]
            if number is None:
                description = entry["description"]
                if command == "update":
                    description = f"Release: #{release}\n\n" + description
                item = self.create_or_recover(
                    release,
                    target,
                    entry["key"],
                    entry["title"],
                    description,
                    "requirement",
                )
            else:
                item = existing[number]
            fields = {"milestone": target}
            if entry["developer"]:
                fields["assignees"] = [entry["developer"]]
            self.patch_issue(item, fields)
            objects[entry["key"]] = item["number"]

        latest = self.issue(release)
        expected = dict(parsed, entries=bind_entries(entries, objects))
        latest_plan = parse_release(
            latest["body"], self.repository, read_pr_requests=True
        )
        latest_plan["entries"] = bind_entries(latest_plan["entries"], objects)
        if latest_plan != expected:
            raise OperationError(
                "Release body changed while applying the plan; retry the command"
            )
        patched = patch_references(latest["body"], objects, self.repository)
        latest = self.patch_issue(latest, {"body": patched})
        self.link_requested_pull_requests(expected["entries"])
        if not initialized:
            self.release_pin(release)
        state = self.reconcile(release, source=latest)
        report_result(
            "Release plan applied", {"checklist": release, "milestone": version}
        )
        return state

    def refresh_release_body(self, source, state, rows):
        original = source["body"]
        try:
            body = patch_pull_requests(original, rows, self.repository)
        except OperationError as error:
            print(f"Cannot refresh PR references for #{source['number']}: {error}")
            body = original
        patched = release_body(body, state)
        if len(patched) > 60000:
            raise OperationError("Release body reached the managed display limit")
        if patched != original:
            if self.issue(source["number"])["body"] != original:
                raise OperationError(
                    "Release body changed during rendering; retry synchronization"
                )
            self.patch_issue(source, {"body": patched})

    def reconcile(self, release, event=None, *, source=None):
        with self.snapshots():
            return self.sync_release(source or self.issue(release), event or {})

    def sync_release(self, source, event):
        state = self.inspect_release(source)
        target = state["milestone"]["number"]
        items = {
            issue["number"]: issue
            for issue in self.milestone_items(target)
            if "pull_request" not in issue
        }
        archive = self.archive(source["number"])
        # Fetch these from their native links even if the Milestone list lags a write.
        items[source["number"]] = source
        items[archive["number"]] = archive
        archive_children = self.api.pages(f"issues/{archive['number']}/sub_issues")
        archived = {
            child["number"] for child in archive_children if "pull_request" not in child
        }
        removed_by_user = (
            (event.get("issue") or {}).get("number")
            if event.get("action") == "demilestoned"
            and (event.get("sender") or {}).get("type") == "User"
            else None
        )
        for child in archive_children:
            if "pull_request" in child or milestone_number(child) is not None:
                continue
            if child["number"] == removed_by_user or any(
                change["event"] == "demilestoned"
                and change.get("actor", {}).get("type") == "User"
                for change in self.api.pages(f"issues/{child['number']}/events")
            ):
                continue
            child = self.patch_issue(child, {"milestone": target})
            items[child["number"]] = child
        for number, item in sorted(items.items()):
            if number in archived or has_label(item, "bug"):
                state["bugs"][str(number)] = item
            else:
                state["scope"][f"issue-{number}"] = {"number": number}
            state["observed"][str(number)] = {
                "state": item["state"],
                "assignees": [person["login"] for person in item.get("assignees", [])],
            }

        containers = {source["number"], archive["number"]}
        rows = self.sync_pull_requests(state, set(items) - containers)
        if "Bugfix" in self.config["projects"]:
            self.prefetch_memberships(state["bugs"].values())
        for number, bug in state["bugs"].items():
            related = rows.get(int(number), [])
            label, project_status = bug_progress(
                bug["state"] == "closed",
                {pr["number"] for pr in related if pr["merged_at"]},
                bool(related),
            )
            labels = [entry["name"] for entry in bug["labels"]]
            if label not in labels or "bug" not in labels:
                desired = [entry for entry in labels if not entry.startswith("status:")]
                if "bug" not in desired:
                    desired.append("bug")
                self.api.rest(
                    f"issues/{number}",
                    "PATCH",
                    {"labels": desired + [label]},
                )
            if "Bugfix" in self.config["projects"]:
                self.project_item(bug, "Bugfix", status=project_status)
        self.refresh_release_body(source, state, rows)
        return state

    def authorize(self, actor):
        permission = self.api.rest(f"collaborators/{actor}/permission")["permission"]
        if permission not in ("admin", "maintain", "write"):
            raise OperationError("此操作需要仓库 write、maintain 或 admin 权限")

    def handle(self, event, event_name):
        with self.snapshots():
            if event_name == "issue_comment":
                self.handle_command(event)
            elif event_name in ("schedule", "workflow_dispatch"):
                self.reconcile_all()
            else:
                self.reconcile_all(event)

    def handle_command(self, event):
        source, comment = event["issue"], event["comment"]
        if (
            event.get("action", "created") != "created"
            or "pull_request" in source
            or comment["user"]["type"] == "Bot"
            or not has_label(source, "release")
        ):
            return
        command = comment["body"].strip()
        is_plan = re.fullmatch(r"/(release_init|update)( --dry-run)?", command)
        if not is_plan and command != "/release_check":
            return
        self.authorize(event["sender"]["login"])
        if is_plan:
            self.apply_plan(source["number"], is_plan[1], bool(is_plan[2]))
        else:
            self.release_check(source["number"])

    def reconcile_all(self, event=None):
        with self.snapshots():
            errors = []
            for source in self.affected_releases(event or {}):
                try:
                    target = milestone_number(source)
                    if len(self.checklists(target)) != 1:
                        raise OperationError(
                            f"Multiple Release Checklists for Milestone #{target}"
                        )
                    archive = self.archive(source["number"])
                    if not archive or milestone_number(archive) != milestone_number(
                        source
                    ):
                        continue
                    self.reconcile(source["number"], event, source=source)
                except OperationError as error:
                    errors.append(f"Release #{source['number']}: {error}")
            if errors:
                raise OperationError("\n".join(errors))

    def archive_milestone(self, issue):
        if has_label(issue, "bug"):
            parent = self.parent(issue["number"]) or {}
            if has_label(parent, ARCHIVE_LABEL):
                return milestone_number(parent)
        return None

    def affected_releases(self, event):
        if not event:
            targets = {
                milestone["number"]
                for milestone in self.api.pages("milestones?state=open")
            }
        elif issue := event.get("issue"):
            targets = {
                milestone_number(issue),
                (event.get("milestone") or {}).get("number"),
            }
            if not milestone_number(issue):
                targets.add(self.archive_milestone(issue))
        elif pr := event.get("pull_request"):
            number = pr["number"]
            current = self.pull_requests({number})[number]
            targets = {milestone_number(current), milestone_number(pr)}
            for linked in self._pr_issues[number].values():
                targets.add(milestone_number(linked) or self.archive_milestone(linked))
        else:
            return []
        return [
            source
            for target in sorted(targets - {None})
            for source in self.checklists(target)
            if (source.get("milestone") or {}).get("state") == "open"
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--event", type=Path, default=os.environ.get("GITHUB_EVENT_PATH")
    )
    parser.add_argument(
        "--event-name", default=os.environ.get("GITHUB_EVENT_NAME", "schedule")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    repository = os.environ.get("GITHUB_REPOSITORY", config["repository"])
    if repository != config["repository"]:
        raise OperationError("部署仓库与 config.json 的 repository 不一致")
    event = json.loads(args.event.read_text(encoding="utf-8")) if args.event else {}
    Controller(GitHub(repository), config).handle(event, args.event_name)
    print(
        json.dumps(
            {"status": "completed", "event": args.event_name, "repository": repository}
        )
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
