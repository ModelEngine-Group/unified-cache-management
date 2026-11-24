"""Release inspection based on maintainer-owned Issue state and linked PRs."""

from operations import tracked
from ownership import milestone_number
from syntax import OperationError, bind_entries, parse_release, scope_diff


class ReleaseChecks:
    @tracked("Release check")
    def release_check(self, release, candidate_sha=None, tag=None):
        _, initial = self.load(release)
        if not initial.get("initialized"):
            raise OperationError("Initialize the release before running /release_check")
        # Target Milestone queries include Bugs from other source archives.
        self.reconcile_all(replay_commands=False, target_release=release)
        _, state = self.load(release)
        parsed = parse_release(self.issue(release)["body"], self.repository)
        current = bind_entries(parsed["entries"], state["objects"])
        delta = scope_diff(state["scope"], current)
        by_key = {e["key"]: e for e in current}
        unsynced = {
            "added": [
                by_key[k]["title"]
                + (
                    f" #{by_key[k]['number']}"
                    if by_key[k].get("number")
                    else " (Issue not created)"
                )
                for k in delta["added"]
            ],
            "removed": [f"#{state['scope'][k]['number']}" for k in delta["removed"]],
            "changed": [f"#{by_key[k]['number']}" for k in delta["changed"]],
            "metadata": [
                key
                for key in parsed["metadata"]
                if parsed["metadata"].get(key) != state.get("metadata", {}).get(key)
            ],
        }
        if (
            parsed["version"] != state["version"]
            and "version" not in unsynced["metadata"]
        ):
            unsynced["metadata"].append("version")
        target = state["milestone"]["number"]
        features = {
            e["number"]: self.issue(e["number"]) for e in state["scope"].values()
        }
        bugs = {i["number"]: i for i in self._target_bugs.get(release, [])}
        linked, conflicts = {}, set()
        for pr, record in state["prs"].items():
            relevant = (
                set(record.get("requirements", [])) & features.keys()
                | set(record.get("bugs", [])) & bugs.keys()
            )
            if relevant:
                linked[pr] = relevant
                if record.get("conflict"):
                    conflicts.add(int(pr))
        for item in self._milestone_items[release]:
            if "pull_request" in item:
                linked.setdefault(str(item["number"]), set())
        self.pull_requests({int(n) for n in linked})
        linked_features = set().union(*linked.values()) if linked else set()
        report = {
            "url": self.run_url,
            "revision": state["revision"],
            "unsynced": unsynced,
            "features_without_pr": sorted(features.keys() - linked_features),
            "open_features": sorted(
                n for n, i in features.items() if i["state"] != "closed"
            ),
            "open_bugs": sorted(n for n, i in bugs.items() if i["state"] != "closed"),
            "unmerged_prs": [],
            "closed_unmerged_prs": [],
            "other_version_prs": [],
            "version_conflicts": [],
        }
        if candidate_sha:
            report["candidate"] = {"ref": tag, "sha": candidate_sha}
        for key, issue_numbers in linked.items():
            pr = self.pull_request(int(key))
            number = int(key)
            targets = {milestone_number((features | bugs)[n]) for n in issue_numbers}
            if number in conflicts or (
                len(targets) > 1 and milestone_number(pr) is None
            ):
                report["version_conflicts"].append(number)
            if milestone_number(pr) not in (None, target):
                report["other_version_prs"].append(number)
                continue
            if not pr.get("merged_at"):
                report[
                    "unmerged_prs" if pr["state"] == "open" else "closed_unmerged_prs"
                ].append(number)
        report["status"] = (
            "blocked"
            if any(unsynced.values())
            or any(
                report[k]
                for k in (
                    "open_features",
                    "open_bugs",
                    "unmerged_prs",
                    "version_conflicts",
                )
            )
            else "ready"
        )
        state["release_check"] = report
        self.save(release, state, "Release check completed")
        return state, report
