"""Evaluate the current Milestone for release readiness."""

import hashlib
import json

from ownership import milestone_number
from presentation import (
    checklist_region,
    release_check_body,
    report_result,
    set_checklist_region,
)
from syntax import OperationError, parse_release


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def plan_baseline(plan):
    return {
        "requirements": {
            str(entry["number"]): fingerprint(
                [entry["title"], entry["description"].rstrip(), entry["developer"]]
            )
            for entry in plan["entries"]
            if entry["number"] is not None
        },
        "metadata": {
            key: fingerprint(value) for key, value in sorted(plan["metadata"].items())
        },
    }


def save_plan_baseline(body, plan):
    baseline = json.dumps(plan_baseline(plan), separators=(",", ":"), sort_keys=True)
    updated = set_checklist_region(body, "plan", f"<!-- {baseline} -->")
    if updated != body:
        updated = set_checklist_region(
            updated,
            "check",
            "### Release Check\n\nRun `/release_check` to check the current plan.",
        )
    return updated


def pending_plan_changes(body, plan):
    try:
        _, region = checklist_region(body, "plan")
        if not region:
            return {}
        snapshot = region.splitlines()[1].removeprefix("<!-- ").removesuffix(" -->")
        previous = json.loads(snapshot)
        requirements, metadata = previous["requirements"], previous["metadata"]
        if not isinstance(requirements, dict) or not isinstance(metadata, dict):
            return {}
        if not all(number.isdecimal() for number in requirements):
            return {}
    except (ValueError, IndexError, KeyError, TypeError):
        return {}
    current = plan_baseline(plan)
    return {
        "added": [
            entry["title"] + (f" #{entry['number']}" if entry["number"] else "")
            for entry in plan["entries"]
            if str(entry["number"]) not in requirements
        ],
        "removed": [
            f"#{n}"
            for n in sorted(requirements.keys() - current["requirements"].keys())
        ],
        "changed": [
            f"#{n}"
            for n, value in current["requirements"].items()
            if n in requirements and value != requirements[n]
        ],
        "metadata": [
            name
            for name in sorted(current["metadata"].keys() | metadata.keys())
            if current["metadata"].get(name) != metadata.get(name)
        ],
    }


class ReleaseChecks:
    def write_release_check(self, release, report, run_url):
        issue = self.issue(release)
        self.patch_issue(
            issue, {"body": release_check_body(issue["body"], report, run_url)}
        )

    def release_check(self, release, candidate_sha=None, tag=None):
        with self.snapshots():
            state = self.reconcile(release)
            source = self.issue(release)
            plan = parse_release(source["body"], self.repository, read_pr_requests=True)
            if plan["version"] != state["version"]:
                raise OperationError(
                    "Release version differs from the Checklist Milestone"
                )
            target = state["milestone"]["number"]
            items = self.milestone_items(target)
            linked = dict(state["prs"])
            for item in items:
                if "pull_request" in item:
                    linked.setdefault(str(item["number"]), {})
            prs = self.pull_requests({int(number) for number in linked})
            tracked_issues = {int(number) for number in state["observed"]}
            pending = [
                entry["title"]
                for entry in plan["entries"]
                if entry["number"] is None
                or (entry["number"] in tracked_issues and entry["requested_prs"])
            ]
            issues_with_pr = {
                issue
                for record in state["prs"].values()
                for issue in record["requirements"] + record["bugs"]
            }
            report = {
                "pending_requirements": pending,
                "issues_without_pr": sorted(
                    tracked_issues - issues_with_pr - {release, state["archive"]}
                ),
                "open_issues": sorted(
                    int(number)
                    for number, item in state["observed"].items()
                    if item["state"] != "closed"
                ),
                "unmerged_prs": [],
                "closed_unmerged_prs": [],
                "other_version_prs": [],
                "version_conflicts": [],
            }
            open_issues = report["open_issues"]
            report["open_bugs"] = [
                number for number in open_issues if str(number) in state["bugs"]
            ]
            report["open_checklists"] = [
                number
                for number in open_issues
                if number in (release, state["archive"])
            ]
            report["open_requirements"] = sorted(
                set(open_issues)
                - set(report["open_bugs"])
                - set(report["open_checklists"])
            )
            unsynced = pending_plan_changes(source["body"], plan)
            if any(unsynced.values()):
                report["unsynced"] = unsynced
            if candidate_sha:
                report["candidate"] = {"ref": tag, "sha": candidate_sha}
            for number, record in linked.items():
                pr = prs[int(number)]
                if record.get("conflict"):
                    report["version_conflicts"].append(pr["number"])
                if milestone_number(pr) not in (None, target):
                    report["other_version_prs"].append(pr["number"])
                elif not pr["merged_at"]:
                    key = (
                        "closed_unmerged_prs"
                        if pr["state"] == "closed"
                        else "unmerged_prs"
                    )
                    report[key].append(pr["number"])
            report["status"] = (
                "blocked"
                if any(
                    report[key]
                    for key in (
                        "pending_requirements",
                        "open_issues",
                        "unmerged_prs",
                        "version_conflicts",
                    )
                )
                else "ready"
            )
            report_result("Release check", report)
            return state, report
