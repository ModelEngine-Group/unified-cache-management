"""Small repository-owned release directory and checkpoint encoding."""

import base64
import copy
import json
import zlib

from syntax import OperationError

MARKER = "<!-- ucm-release-directory:v1 -->"
LABEL = "release-directory"


def pack(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if len(raw) > 5_000_000:
        raise OperationError("Checkpoint exceeds its decoded size limit")
    return base64.b64encode(zlib.compress(raw)).decode()


def unpack(value):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value, validate=True), 5_000_001)
    if len(raw) > 5_000_000 or not decoder.eof:
        raise OperationError("Checkpoint exceeds its decoded size limit")
    return json.loads(raw)


class ReleaseDirectory:
    def __init__(self, api):
        self.api = api
        rows = api.pages(f"issues?state=all&labels={LABEL}")
        matches = [
            i
            for i in rows
            if "pull_request" not in i and (i.get("body") or "").startswith(MARKER)
        ]
        if len(matches) > 1:
            raise OperationError("Duplicate release directories")
        self.issue = matches[0] if matches else None
        self.data = (
            unpack(self.issue["body"].split("\n", 1)[1])
            if self.issue
            else {"releases": {}, "seen_prs": []}
        )

    def write(self):
        body = MARKER + "\n" + pack(self.data)
        if len(body) > 60000:
            raise OperationError("Release directory reached the comment capacity")
        if self.issue:
            if body != self.issue["body"]:
                self.issue = self.api.rest(
                    f"issues/{self.issue['number']}", "PATCH", {"body": body}
                )
            return
        try:
            self.api.rest("labels/" + LABEL)
        except OperationError as error:
            if "404" not in str(error):
                raise
            self.api.rest(
                "labels",
                "POST",
                {
                    "name": LABEL,
                    "color": "ededed",
                    "description": "Internal release routing directory",
                },
            )
        self.issue = self.api.rest(
            "issues",
            "POST",
            {"title": "Release directory", "body": body, "labels": [LABEL]},
        )
        self.issue = self.api.rest(
            f"issues/{self.issue['number']}", "PATCH", {"state": "closed"}
        )

    def update(self, release, state, source):
        milestone = state.get("milestone") or {}
        row = {
            "version": state.get("version"),
            "milestone": milestone.get("number"),
            "archive": state.get("archive"),
            "active": bool(
                state.get("initialized")
                and source["state"] != "closed"
                and milestone.get("state") != "closed"
            ),
            "issues": sorted(
                {e["number"] for e in state["scope"].values()}
                | {int(n) for n in state["bugs"]}
            ),
            "prs": sorted(
                int(n)
                for n, p in state["prs"].items()
                if p.get("requirements") or p.get("bugs")
            ),
            "operation": state.get("operation", {}),
        }
        seen = sorted(set(self.data["seen_prs"]) | {int(n) for n in state["prs"]})
        if (
            self.data["releases"].get(str(release)) != row
            or seen != self.data["seen_prs"]
        ):
            self.data["releases"][str(release)] = row
            self.data["seen_prs"] = seen
            self.write()

    def claim_prs(self, numbers):
        seen = set(self.data["seen_prs"])
        first = set(numbers) - seen
        if first:
            self.data["seen_prs"] = sorted(seen | first)
            self.write()
        return first

    def rows(self):
        return {int(n): copy.deepcopy(row) for n, row in self.data["releases"].items()}
