"""Evaluate the current Milestone for release readiness."""

from ownership import milestone_number
from presentation import report_result
from syntax import OperationError, parse_release


class ReleaseChecks:
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
