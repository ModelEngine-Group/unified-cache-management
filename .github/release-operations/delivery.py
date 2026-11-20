"""Connect immutable v* Tag publication to release operations.

Short gate/report jobs share the controller's repository concurrency group.
The publisher owns delivery transitions; ordinary reconciliation only renders them.
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from presentation import replace_region
from syntax import OperationError, field

TRACKING_START = "<!-- ucm-release-tracking:start -->"
TRACKING_END = "<!-- ucm-release-tracking:end -->"
TERMINAL = {"delivered", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def attempt_key(record):
    return int(record["run_id"]), int(record["run_attempt"])


class ReleaseDelivery:
    def delivery_release(self, tag, version=None):
        matches = []
        for number in self.releases():
            _, state = self.load(number)
            versions = {
                state.get("version"),
                field(self.issue(number)["body"], "version"),
            }
            if tag in state.get("deliveries", {}) or (version or tag[1:]) in versions:
                matches.append(number)
        if len(matches) != 1:
            raise OperationError(
                f"Expected one Release Issue for {tag}; found {len(matches)}: {matches}"
            )
        return matches[0]

    def delivery_write(self, release, run, status, **details):
        _, state = self.load(release)
        records = state.setdefault("deliveries", {})
        previous = records.get(run["tag"])
        if previous:
            if attempt_key(run) < attempt_key(previous):
                return False
            if attempt_key(run) == attempt_key(previous):
                if previous["status"] in TERMINAL and status not in TERMINAL:
                    return False
                if (
                    previous["status"] == "publishing"
                    and status == "checking"
                    and not details.get("gate_active")
                ):
                    return False
                if previous["status"] == "delivered" and status != "delivered":
                    return False
        record = (
            dict(previous)
            if previous and attempt_key(run) == attempt_key(previous)
            else {}
        )
        if (
            previous
            and previous.get("source_sha") == run["source_sha"]
            and previous.get("release_url")
        ):
            record.setdefault("release_url", previous["release_url"])
        record.update(run, status=status, updated_at=now())
        record.setdefault("started_at", now())
        record.update(details)
        if status not in TERMINAL:
            record.pop("error", None)
            record.pop("completed_at", None)
        records[run["tag"]] = record
        self.save(release, state, "Release delivery: " + status)
        return True

    def delivery_gate(self, run):
        release = self.delivery_release(run["tag"], run["version"])
        if not self.delivery_write(
            release, run, "checking", gate_active=True, gate_passed=False
        ):
            raise OperationError(
                "This release attempt has already been superseded or completed"
            )
        try:
            _, state = self.load(release)
            if not state.get("initialized"):
                raise OperationError("Run /release_init before pushing the release Tag")
            if state.get("version") != run["version"]:
                raise OperationError(
                    "Tag version does not match the initialized release"
                )
            self.issue(state["archive"])
            self.api.rest(f"milestones/{state['milestone']['number']}")
            _, report = self.release_check(
                release, candidate_sha=run["source_sha"], tag=run["tag"]
            )
            if report["status"] != "ready":
                raise OperationError(
                    "Version check blocked publication; see Release Check"
                )
            self.delivery_write(
                release, run, "publishing", gate_passed=True, gate_active=False
            )
        except Exception as error:
            self.delivery_write(
                release, run, "failed", error=" ".join(str(error).split())[:240]
            )
            raise
        return release

    def delivery_started(self, run):
        release = self.delivery_release(run["tag"])
        _, state = self.load(release)
        previous = state.get("deliveries", {}).get(run["tag"], {})
        passed = previous.get("run_id") == run["run_id"] and previous.get(
            "gate_passed", False
        )
        self.delivery_write(
            release, run, "publishing" if passed else "checking", gate_passed=passed
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
        release = self.delivery_release(run["tag"])
        _, state = self.load(release)
        previous = state.get("deliveries", {}).get(run["tag"])
        if previous and attempt_key(run) < attempt_key(previous):
            return
        if (
            previous
            and attempt_key(run) == attempt_key(previous)
            and previous["status"] == "delivered"
        ):
            return
        if result != "success":
            error = (
                previous.get("error")
                if previous and attempt_key(previous) == attempt_key(run)
                else None
            )
            self.delivery_write(
                release,
                run,
                "cancelled" if result == "cancelled" else "failed",
                error=error or "Release workflow " + result,
            )
            return
        try:
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
                f"Release tracking: #{release} · "
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
            self.api.rest(
                f"milestones/{state['milestone']['number']}",
                "PATCH",
                {"state": "closed"},
            )
            state["milestone"]["state"] = "closed"
            self.save(release, state, "Release Milestone archived")
            self.delivery_write(
                release,
                run,
                "delivered",
                release_url=published["html_url"],
                completed_at=now(),
            )
        except Exception as error:
            self.delivery_write(
                release, run, "failed", error=" ".join(str(error).split())[:240]
            )
            raise


def main():
    # Import here so the controller can mix in ReleaseDelivery without a cycle.
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
    api = GitHub(repository)
    controller = Controller(api, config)
    if args.mode == "observe":
        event = json.loads(
            Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")
        )
        source = event["workflow_run"]
        current = api.rest(f"actions/runs/{int(source['id'])}")
        if current["run_attempt"] != source["run_attempt"]:
            print("Ignoring an older workflow attempt")
            return
        if (
            current["path"] != ".github/workflows/release-tag.yml"
            or current["event"] != "push"
        ):
            raise OperationError("Unexpected release workflow source")
        tag, sha = current["head_branch"], current["head_sha"]
        run_id, attempt = current["id"], current["run_attempt"]
        result = current.get("conclusion")
    else:
        tag, sha = os.environ["RELEASE_TAG"], os.environ["SOURCE_SHA"]
        run_id, attempt = int(os.environ["GITHUB_RUN_ID"]), int(
            os.environ["GITHUB_RUN_ATTEMPT"]
        )
        result = os.environ.get("RELEASE_RESULT")
    if not tag or not tag.startswith("v"):
        print("Version-operation hooks apply only to v* Tags")
        return
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or run_id <= 0 or attempt <= 0:
        raise OperationError("Invalid release run identity")
    run = {
        "tag": tag,
        "source_sha": sha,
        "run_id": run_id,
        "run_attempt": attempt,
        "url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{repository}/actions/runs/{run_id}/attempts/{attempt}",
    }
    controller.run_url = run["url"]
    if args.mode == "gate":
        run["version"] = os.environ["RELEASE_VERSION"]
        with controller.snapshots():
            controller.delivery_gate(run)
    elif (
        args.mode == "finish"
        or event["action"] == "completed"
        or current.get("status") == "completed"
    ):
        controller.delivery_finished(run, result or "failure")
    else:
        try:
            controller.delivery_started(run)
        except OperationError as error:
            # The first start event may precede initialization or canonical Tag
            # classification. The mandatory gate still reports the exact error.
            print("Release start notification: " + str(error))


if __name__ == "__main__":
    main()
