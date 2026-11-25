"""Release Tag checks and completion, driven by Actions receipts."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from presentation import (
    append_checklist_section,
    checklist_region,
    replace_region,
    report_result,
    set_checklist_region,
    warning_text,
)
from syntax import OperationError

TRACKING_START = "<!-- ucm-release-tracking:start -->"
TRACKING_END = "<!-- ucm-release-tracking:end -->"
PUBLISHED_START = "<!-- ucm-published-release:start -->"
PUBLISHED_END = "<!-- ucm-published-release:end -->"
FORMAL_TAG = re.compile(
    r"v(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}(?:rc(?:0|[1-9][0-9]*))?"
)
DELIVERY_ATTEMPT = re.compile(
    r"<!-- ucm-delivery-attempt:([0-9]+):([0-9]+):(checking|publishing|delivered|failed|cancelled):([01]) -->"
)
TERMINAL = {"delivered", "failed", "cancelled"}


def delivery_run(repository, tag, sha, run_id, attempt, version=None):
    if not FORMAL_TAG.fullmatch(tag or ""):
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or run_id <= 0 or attempt <= 0:
        raise OperationError("Invalid release run identity")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return {
        "tag": tag,
        "version": version or tag[1:],
        "source_sha": sha,
        "run_id": run_id,
        "run_attempt": attempt,
        "url": f"{server}/{repository}/actions/runs/{run_id}/attempts/{attempt}",
    }


def delivery_body(body, run, status, *, error="", release_url="", completed_at=None):
    _, existing = checklist_region(body, "delivery")
    previous = DELIVERY_ATTEMPT.search(existing)
    key = (run["run_id"], run["run_attempt"])
    gate_passed = False
    if previous:
        old_key = (int(previous[1]), int(previous[2]))
        old_status = previous[3]
        gate_passed = old_key[0] == key[0] and previous[4] == "1"
        if old_key > key:
            return body
        if old_key == key and (
            status in ("started", old_status)
            or old_status == "delivered"
            or (old_status in TERMINAL and status not in TERMINAL)
        ):
            return body
    if status == "started":
        status = "publishing" if gate_passed else "checking"
    if status == "checking":
        gate_passed = False
    elif status in ("publishing", "delivered"):
        gate_passed = True

    icon = {
        "checking": "🔄",
        "publishing": "🔄",
        "delivered": "✅",
        "failed": "❌",
        "cancelled": "⚪",
    }[status]
    run_id, attempt = key
    marker = (
        f"<!-- ucm-delivery-attempt:{run_id}:{attempt}:{status}:{int(gate_passed)} -->"
    )
    details = (
        f"{icon} **{run['tag']} · {status.title()}**"
        f" · [Actions]({run['url']}) · Attempt {attempt}"
    )
    if release_url:
        details += f" · [GitHub Release]({release_url})"
    if status in TERMINAL:
        finished = completed_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        details += "\n\nFinished " + warning_text(finished)
    if error:
        details += "\n\n" + warning_text(error)
    return set_checklist_region(
        body, "delivery", f"### Release Delivery\n\n{marker}\n\n{details}"
    )


class ReleaseDelivery:
    def delivery_release(self, tag, version=None):
        version = version or tag[1:]
        milestones = [
            milestone
            for milestone in self.api.pages("milestones?state=all")
            if milestone["title"] == version
        ]
        if len(milestones) != 1:
            raise OperationError(
                f"Expected one Milestone for {tag}; found {len(milestones)}"
            )
        matches = self.checklists(milestones[0]["number"])
        if len(matches) != 1:
            raise OperationError(
                f"Expected one Release Checklist for {tag}; found {[item['number'] for item in matches]}"
            )
        return matches[0]

    def write_delivery(self, release, run, status, **details):
        issue = self.issue(release)
        self.patch_issue(
            issue, {"body": delivery_body(issue["body"], run, status, **details)}
        )

    def report_delivery(self, release, run, status, **details):
        try:
            self.write_delivery(release, run, status, **details)
        except OperationError as error:
            print(f"Could not write delivery status for #{release}: {error}")

    def delivery_gate(self, run):
        with self.snapshots():
            release = self.delivery_release(run["tag"], run["version"])["number"]
            self.report_delivery(release, run, "checking")
            try:
                _, report = self.release_check(
                    release, candidate_sha=run["source_sha"], tag=run["tag"]
                )
                try:
                    self.write_release_check(release, report, run["url"])
                except OperationError as error:
                    print(f"Could not write release check for #{release}: {error}")
                if report["status"] != "ready":
                    raise OperationError(
                        "Version check blocked publication; see the Actions summary"
                    )
            except OperationError as error:
                self.report_delivery(release, run, "failed", error=str(error))
                raise
            self.report_delivery(release, run, "publishing")

    def observe_delivery(self, event):
        source = event["workflow_run"]
        current = self.api.rest(f"actions/runs/{int(source['id'])}")
        if current["run_attempt"] != source["run_attempt"]:
            return
        if (
            current["path"] != ".github/workflows/release-tag.yml"
            or current["event"] != "push"
            or (current.get("head_repository") or {}).get("full_name")
            != self.repository
        ):
            raise OperationError("Unexpected release workflow source")
        run = delivery_run(
            self.repository,
            current["head_branch"],
            current["head_sha"],
            current["id"],
            current["run_attempt"],
        )
        if not run:
            return
        result = current.get("conclusion")
        completed = current["status"] == "completed"
        if completed and result == "success":
            return
        with self.snapshots():
            try:
                release = self.delivery_release(run["tag"])["number"]
            except OperationError as error:
                print(f"Could not locate the Checklist for {run['tag']}: {error}")
                return
            if not completed:
                self.write_delivery(release, run, "started")
                return
            status = "cancelled" if result == "cancelled" else "failed"
            reason = f"Actions concluded with {result or 'failure'}."
            if status == "failed":
                try:
                    jobs = self.api.rest(
                        f"actions/runs/{run['run_id']}/attempts/{run['run_attempt']}/jobs?per_page=100"
                    )["jobs"]
                    failed = [
                        job["name"] for job in jobs if job["conclusion"] == "failure"
                    ]
                    if failed:
                        reason += " Failed jobs · " + ", ".join(failed)
                except OperationError as error:
                    print(f"Could not look up failed jobs: {error}")
            self.write_delivery(
                release,
                run,
                status,
                error=reason,
                completed_at=current.get("updated_at"),
            )

    def delivery_receipt(self, run):
        prefix = f"ucm-release-acceptance-run-{run['run_id']}"
        artifacts = []
        for page in range(1, 101):
            batch = self.api.rest(
                f"actions/runs/{run['run_id']}/artifacts?per_page=100&page={page}"
            )["artifacts"]
            artifacts.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise OperationError("Too many delivery receipt artifact pages")
        eligible = []
        for artifact in artifacts:
            match = re.fullmatch(
                re.escape(prefix) + r"-attempt-(\d+)", artifact["name"]
            )
            if (
                match
                and int(match[1]) <= run["run_attempt"]
                and not artifact.get("expired")
            ):
                eligible.append((int(match[1]), artifact))
        if not eligible:
            raise OperationError("Complete delivery acceptance receipt is unavailable")
        attempt, artifact = max(eligible, key=lambda entry: entry[0])
        receipt = self.api.artifact_json(artifact["id"], "acceptance.json")
        expected = {
            "status": "complete",
            "repository": self.repository,
            "source_sha": run["source_sha"],
            "tag": run["tag"],
            "actions_run_id": run["run_id"],
            "run_attempt": attempt,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise OperationError(
                "Delivery acceptance receipt does not match this Tag and run"
            )
        return receipt

    def delivery_finished(self, run, result):
        if result != "success":
            return
        with self.snapshots():
            source = self.delivery_release(run["tag"], run.get("version"))
            state = self.inspect_release(source)
            self.delivery_receipt(run)
            encoded = quote(run["tag"], safe="")
            if self.api.rest("commits/" + encoded)["sha"] != run["source_sha"]:
                raise OperationError(
                    "Release Tag no longer points to the checked commit"
                )
            published = self.api.rest("releases/tags/" + encoded)
            if published["tag_name"] != run["tag"] or published["draft"]:
                raise OperationError(
                    "Published GitHub Release does not match the release Tag"
                )

            navigation = (
                f"Release tracking: #{source['number']} · "
                f"[Milestone]({state['milestone']['html_url']}) · "
                f"Bug Archive: #{state['archive']}"
            )
            original = published.get("body") or ""
            body, section = replace_region(
                original, TRACKING_START, TRACKING_END, navigation
            )
            body = body.rstrip() + "\n\n" + section + "\n"
            if body != original:
                self.api.rest(f"releases/{published['id']}", "PATCH", {"body": body})
            checklist = self.issue(source["number"])
            checklist_body, link = replace_region(
                checklist["body"],
                PUBLISHED_START,
                PUBLISHED_END,
                f"**Published Release:** [{run['tag']}]({published['html_url']})",
            )
            self.patch_issue(
                checklist, {"body": append_checklist_section(checklist_body, link)}
            )

            milestone = state["milestone"]
            if milestone["state"] != "closed":
                self.api.rest(
                    f"milestones/{milestone['number']}", "PATCH", {"state": "closed"}
                )
            self.write_delivery(
                source["number"], run, "delivered", release_url=published["html_url"]
            )
            report_result("Published release", published["html_url"])


def main():
    from controller import Controller, GitHub

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gate", "finish", "observe"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    repository = os.environ["GITHUB_REPOSITORY"]
    if repository != config["repository"]:
        raise OperationError("Running repository must match config.json.repository")
    controller = Controller(GitHub(repository), config)
    if args.mode == "observe":
        event = json.loads(
            Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
        )
        controller.observe_delivery(event)
        return
    run = delivery_run(
        repository,
        os.environ["RELEASE_TAG"],
        os.environ["SOURCE_SHA"],
        int(os.environ["GITHUB_RUN_ID"]),
        int(os.environ["GITHUB_RUN_ATTEMPT"]),
        os.environ.get("RELEASE_VERSION"),
    )
    if not run:
        raise OperationError("Expected a canonical Stable or RC Tag")
    if args.mode == "gate":
        controller.delivery_gate(run)
    else:
        controller.delivery_finished(run, os.environ.get("RELEASE_RESULT"))


if __name__ == "__main__":
    main()
