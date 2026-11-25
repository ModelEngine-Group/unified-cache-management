"""Release Tag checks and completion, driven by Actions receipts."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from presentation import replace_region, report_result
from syntax import OperationError

TRACKING_START = "<!-- ucm-release-tracking:start -->"
TRACKING_END = "<!-- ucm-release-tracking:end -->"
PUBLISHED_START = "<!-- ucm-published-release:start -->"
PUBLISHED_END = "<!-- ucm-published-release:end -->"


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

    def delivery_gate(self, run):
        with self.snapshots():
            release = self.delivery_release(run["tag"], run["version"])
            _, report = self.release_check(
                release["number"], candidate_sha=run["source_sha"], tag=run["tag"]
            )
            if report["status"] != "ready":
                raise OperationError(
                    "Version check blocked publication; see the Actions summary"
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
                checklist, {"body": checklist_body.rstrip() + "\n\n" + link + "\n"}
            )

            milestone = state["milestone"]
            if milestone["state"] != "closed":
                self.api.rest(
                    f"milestones/{milestone['number']}", "PATCH", {"state": "closed"}
                )
            report_result("Published release", published["html_url"])


def main():
    from controller import Controller, GitHub

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gate", "finish"))
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    repository = os.environ["GITHUB_REPOSITORY"]
    if repository != config["repository"]:
        raise OperationError("Running repository must match config.json.repository")
    tag, sha = os.environ["RELEASE_TAG"], os.environ["SOURCE_SHA"]
    run_id, attempt = int(os.environ["GITHUB_RUN_ID"]), int(
        os.environ["GITHUB_RUN_ATTEMPT"]
    )
    if not tag.startswith("v"):
        print("Version-operation hooks apply only to v* Tags")
        return
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or run_id <= 0 or attempt <= 0:
        raise OperationError("Invalid release run identity")
    run = {
        "tag": tag,
        "version": os.environ.get("RELEASE_VERSION") or tag[1:],
        "source_sha": sha,
        "run_id": run_id,
        "run_attempt": attempt,
    }
    controller = Controller(GitHub(repository), config)
    if args.mode == "gate":
        controller.delivery_gate(run)
    else:
        controller.delivery_finished(run, os.environ.get("RELEASE_RESULT"))


if __name__ == "__main__":
    main()
