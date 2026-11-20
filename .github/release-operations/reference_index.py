"""Bounded, resumable PR reference discovery with a repository-owned checkpoint."""

import base64
import copy
import hashlib
import json
import re
import zlib
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from syntax import OperationError

MANIFEST = "<!-- ucm-reference-index:manifest:v1 -->"
SHARD = "<!-- ucm-reference-index:shard:{}:v1 -->"
LABEL = "release-index"


def timestamp(offset=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def pack(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    if len(raw) > 5_000_000:
        raise OperationError("Checkpoint exceeds its decoded size limit")
    return base64.b64encode(zlib.compress(raw)).decode()


def unpack(value):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(value, validate=True), 5_000_001)
    if len(raw) > 5_000_000 or not decoder.eof:
        raise OperationError("Reference checkpoint exceeds its decoded size limit")
    return json.loads(raw)


def empty_index():
    return {
        "bootstrap": {"page": 1, "offset": 0, "complete": False},
        "poll": {"since": timestamp(-120)},
        "prs": {},
        "pending": [],
    }


class ReferenceStore:
    """Publish shard changes before atomically switching the manifest pointers.

    Each shard retains its committed payload while a new generation is written.
    An interrupted flush leaves the old manifest readable and its cursor unchanged.
    Only comments authored by github-actions[bot] supply checkpoint data.
    """

    def __init__(self, api):
        self.api = api
        self.issue = None
        self.manifest_comment = None
        self.shard_comments = {}
        self.pointers = {}

    def load(self):
        issues = self.api.pages(f"issues?state=all&labels={LABEL}")
        candidates = [
            i
            for i in issues
            if "pull_request" not in i and MANIFEST in (i.get("body") or "")
        ]
        if len(candidates) > 1:
            raise OperationError("Duplicate release reference-index Issues")
        if not candidates:
            return empty_index()
        self.issue = candidates[0]
        comments = self.api.pages(f"issues/{self.issue['number']}/comments")
        for comment in comments:
            if comment["user"]["login"] != "github-actions[bot]":
                continue
            if comment["body"].startswith(MANIFEST):
                if self.manifest_comment:
                    raise OperationError("Duplicate reference-index manifests")
                self.manifest_comment = comment
            match = re.match(
                r"<!-- ucm-reference-index:shard:(\d+):v1 -->", comment["body"]
            )
            if match:
                if match[1] in self.shard_comments:
                    raise OperationError("Duplicate reference-index shards")
                self.shard_comments[match[1]] = comment
        if not self.manifest_comment:
            return empty_index()
        metadata = unpack(self.manifest_comment["body"].split("\n", 1)[1])
        self.pointers = metadata.pop("shards", {})
        prs = {}
        for bucket, digest in self.pointers.items():
            comment = self.shard_comments.get(bucket)
            if not comment:
                raise OperationError("Reference-index shard is missing")
            versions = json.loads(comment["body"].split("\n", 1)[1])
            encoded = versions.get(digest)
            if not encoded or hashlib.sha256(encoded.encode()).hexdigest() != digest:
                raise OperationError("Reference-index shard checksum mismatch")
            prs.update(unpack(encoded))
        return dict(metadata, prs=prs)

    def write(self, value):
        if self.issue is None:
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
                        "description": "Internal release reference-index checkpoint",
                    },
                )
            self.issue = self.api.rest(
                "issues",
                "POST",
                {
                    "title": "Release reference index",
                    "body": MANIFEST
                    + "\n\nAutomatically maintained PR reference index and discovery progress.",
                    "labels": [LABEL],
                },
            )
        if self.issue.get("state") != "closed":
            self.issue = self.api.rest(
                f"issues/{self.issue['number']}", "PATCH", {"state": "closed"}
            )
        buckets = {}
        for number, entry in value["prs"].items():
            buckets.setdefault(str(int(number) // 100), {})[number] = entry
        pointers = {}
        for bucket, entries in sorted(buckets.items()):
            encoded = pack(entries)
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            pointers[bucket] = digest
            if self.pointers.get(bucket) == digest:
                continue
            old = self.shard_comments.get(bucket)
            versions = {digest: encoded}
            committed = self.pointers.get(bucket)
            if old and committed:
                versions[committed] = json.loads(old["body"].split("\n", 1)[1])[
                    committed
                ]
            body = (
                SHARD.format(bucket)
                + "\n"
                + json.dumps(versions, separators=(",", ":"))
            )
            self.shard_comments[bucket] = self.comment(old, body)
        metadata = {key: entry for key, entry in value.items() if key != "prs"}
        body = MANIFEST + "\n" + pack(dict(metadata, shards=pointers))
        self.manifest_comment = self.comment(self.manifest_comment, body)
        self.pointers = pointers

    def comment(self, previous, body):
        if len(body) > 60000:
            raise OperationError("Reference-index shard exceeds the comment capacity")
        if previous:
            if previous["body"] == body:
                return previous
            return self.api.rest(
                f"issues/comments/{previous['id']}", "PATCH", {"body": body}
            )
        return self.api.rest(
            f"issues/{self.issue['number']}/comments", "POST", {"body": body}
        )


class ReferenceIndex:
    def __init__(self, api, read_references, limit=10):
        self.api = api
        self.read_references = read_references
        self.limit = max(1, min(int(limit), 50))
        self.data = api.index_read()
        self.original = copy.deepcopy(self.data)
        self.scanned = 0
        self.active_scope = self.data.get("active_scope")
        self.advanced = set()

    def select(self, scope, since):
        """Keep discovery coverage separate for each release's Issue window."""
        self._save_window()
        key = str(scope)
        windows = self.data.setdefault("windows", {})
        window = windows.get(key)
        if window is None or window.get("since") != since:
            window = windows[key] = {
                "since": since,
                "bootstrap": {
                    "page": 1,
                    "offset": 0,
                    "complete": since is None,
                    "empty": True,
                },
                "poll": {"since": timestamp(-120)},
                "pending": [],
            }
        self.active_scope = key
        self.data["active_scope"] = key
        for field in ("since", "bootstrap", "poll", "pending"):
            self.data[field] = window[field]

    def _save_window(self):
        if self.active_scope is not None:
            self.data.setdefault("windows", {}).setdefault(
                self.active_scope, {}
            ).update(
                {
                    key: self.data[key]
                    for key in ("since", "bootstrap", "poll", "pending")
                }
            )

    def eligible(self, pr):
        since = self.data.get("since")
        if self.active_scope is not None and since is None:
            return False
        if since is None:
            return True
        created = pr.get("created_at")
        if not created:
            raise OperationError("PR creation time is unavailable")
        return created >= since

    def advance_once(self):
        key = (self.active_scope, self.data.get("since"))
        if key not in self.advanced:
            self.advanced.add(key)
            self.advance()

    def queue(self, number):
        pending = self.data.setdefault("pending", [])
        if number not in pending:
            pending.append(number)

    def can_refresh(self):
        return self.scanned < self.limit

    def indexed(self, number):
        return bool(self.data["prs"].get(str(number), {}).get("checked_at"))

    def note(self, pr):
        entry = self.data["prs"].setdefault(str(pr["number"]), {"refs": {}})
        entry["pr"] = {
            key: pr[key]
            for key in (
                "number",
                "node_id",
                "html_url",
                "milestone",
                "state",
                "merged_at",
                "draft",
                "merge_commit_sha",
                "updated_at",
                "created_at",
            )
            if key in pr
        }
        entry["full"] = "merged_at" in pr

    def refresh(self, pr):
        if not self.eligible(pr):
            return False
        if not self.can_refresh():
            raise OperationError(
                "PR reference discovery reached its per-run batch limit"
            )
        started = timestamp(-1)
        refs = self.read_references(pr)
        self.scanned += 1
        self.data["prs"][str(pr["number"])] = {
            "refs": {str(number): urls for number, urls in refs.items()},
            "checked_at": started,
            "pr": {
                key: pr[key]
                for key in (
                    "number",
                    "node_id",
                    "html_url",
                    "milestone",
                    "state",
                    "merged_at",
                    "draft",
                    "merge_commit_sha",
                    "updated_at",
                    "created_at",
                )
                if key in pr
            },
            "full": "merged_at" in pr,
        }
        pending = self.data.setdefault("pending", [])
        if pr["number"] in pending:
            pending.remove(pr["number"])
        for window in self.data.get("windows", {}).values():
            if pr["number"] in window["pending"]:
                window["pending"].remove(pr["number"])
        return True

    def references(self, number):
        return {
            int(n): urls
            for n, urls in self.data["prs"].get(str(number), {}).get("refs", {}).items()
        }

    def candidates(self, issues):
        return {
            int(number)
            for number, entry in self.data["prs"].items()
            if issues & self.references(number).keys()
            and (
                not self.data.get("since")
                or not entry.get("pr", {}).get("created_at")
                or self.eligible(entry["pr"])
            )
        }

    def advance(self):
        try:
            self._advance()
        finally:
            self.flush()

    def _advance(self):
        """At most limit discussion scans, plus bounded list pages, per run."""
        if self.active_scope is not None and self.data.get("since") is None:
            return
        bootstrap = self.data["bootstrap"]
        fresh_sweep = (
            not bootstrap["complete"]
            and bootstrap["page"] == 1
            and bootstrap["offset"] == 0
        )
        sweep_started = timestamp(-120)
        if (
            self.active_scope is not None
            and bootstrap["complete"]
            and bootstrap.get("empty")
        ):
            newest = self.api.rest(
                "pulls?state=all&sort=created&direction=desc&per_page=100&page=1"
            )
            if not newest or not self.eligible(newest[0]):
                self.data["poll"] = {"since": sweep_started}
                return
            bootstrap.update(page=1, offset=0, complete=False)
            fresh_sweep = True
        for number in list(self.data.get("pending", [])):
            if not self.can_refresh():
                break
            pr = self.api.rest(f"pulls/{number}")
            if not self.eligible(pr):
                self.note(pr)
                self.data["pending"].remove(number)
                continue
            self.refresh(pr)
        budget = self.limit - self.scanned
        poll = self.data["poll"]
        if "started" not in poll:
            poll.update(started=timestamp(-120), page=1, offset=0)
        for _ in range(3):
            rows = self.api.rest(
                "issues?state=all&sort=updated&direction=desc&since="
                + quote(poll["since"], safe="")
                + f"&per_page=100&page={poll['page']}"
            )
            while poll["offset"] < len(rows):
                row = rows[poll["offset"]]
                entry = self.data["prs"].get(str(row["number"]), {})
                if (
                    "pull_request" in row
                    and self.eligible(row)
                    and entry.get("checked_at", "") < row.get("updated_at", "9999")
                ):
                    if budget <= 0:
                        break
                    self.refresh(row)
                    budget -= 1
                poll["offset"] += 1
            if poll["offset"] < len(rows):
                break
            if len(rows) < 100:
                self.data["poll"] = {"since": poll["started"]}
                break
            poll.update(page=poll["page"] + 1, offset=0)
        bootstrap = self.data["bootstrap"]
        bootstrap_pages = 0
        while not bootstrap["complete"] and bootstrap_pages < 3:
            bootstrap_pages += 1
            rows = self.api.rest(
                "pulls?state=all&sort=created&direction=desc"
                + f"&per_page=100&page={bootstrap['page']}"
            )
            while bootstrap["offset"] < len(rows):
                pr = rows[bootstrap["offset"]]
                if not self.eligible(pr):
                    bootstrap["complete"] = True
                    break
                bootstrap["empty"] = False
                cached = self.data["prs"].get(str(pr["number"]))
                if not self.indexed(pr["number"]) or cached.get(
                    "checked_at", ""
                ) < pr.get("updated_at", ""):
                    if budget <= 0:
                        break
                    self.refresh(pr)
                    budget -= 1
                else:
                    cached.setdefault("pr", {})["created_at"] = pr["created_at"]
                bootstrap["offset"] += 1
            if bootstrap["complete"] or bootstrap["offset"] < len(rows):
                break
            if bootstrap["offset"] == len(rows):
                if len(rows) < 100:
                    bootstrap["complete"] = True
                else:
                    bootstrap.update(page=bootstrap["page"] + 1, offset=0)
        if fresh_sweep and bootstrap["complete"]:
            # This pass verified the whole eligible creation window. Older PR
            # updates outside that window must not keep readiness pending.
            self.data["poll"] = {"since": sweep_started}

    def flush(self):
        self._save_window()
        if self.data != self.original:
            self.api.index_write(self.data)
            self.original = copy.deepcopy(self.data)

    def status(self):
        return {
            "complete": self.data["bootstrap"]["complete"]
            and "started" not in self.data["poll"]
            and not self.data.get("pending"),
            "indexed_prs": sum(
                self.indexed(number)
                and (
                    self.active_scope is None
                    or bool(
                        entry.get("pr", {}).get("created_at")
                        and self.eligible(entry["pr"])
                    )
                )
                for number, entry in self.data["prs"].items()
            ),
            "pending_prs": len(self.data.get("pending", [])),
            "since": self.data.get("since"),
        }
