"""Structured Markdown syntax for the Release Checklist."""

import hashlib
import re
from datetime import date

VERSION_RE = re.compile(r"\d+\.\d+\.\d+(?:rc\d+)?(?:[-+][A-Za-z0-9.-]+)?\Z")
REF_RE = re.compile(
    r"(?<![\w/])(?:https://github\.com/([\w.-]+/[\w.-]+)/issues/|"
    r"(?:([\w.-]+/[\w.-]+))?#)(\d+)\b"
)
MARKDOWN_LINK_RE = re.compile(
    r"\[(?P<label>(?:\\.|[^\]\\])*)\]\([ \t]*(?P<url>[^\s()]*?)[ \t]*\)$"
)
ISSUE_URL_RE = re.compile(
    r"https://github\.com/[\w.-]+/[\w.-]+/issues/[1-9]\d*(?:[?#][^\s]*)?\Z"
)
PR_URL_RE = re.compile(
    r"https://github\.com/([\w.-]+/[\w.-]+)/pull/([1-9]\d*)/?(?:[?#]\S*)?\Z",
    re.I,
)
PR_REFERENCE_RE = re.compile(r"(?:(?:([\w.-]+/[\w.-]+))?#)?([1-9]\d*)\Z")
SYNCED_PR_ROW_RE = re.compile(r"- #[1-9]\d* \| (?:open|draft|closed|merged)\Z")


class OperationError(ValueError):
    pass


def field(body, name):
    match = re.search(
        r"^[ \t]*" + re.escape(name) + r"[ \t]*[:：][ \t]*([^\r\n]*?)[ \t]*$",
        body,
        re.M,
    )
    return match[1] if match else ""


def issue_reference(value, repository):
    link = MARKDOWN_LINK_RE.fullmatch(value.strip())
    if link:
        value = link["url"]
    matches = list(REF_RE.finditer(value))
    if len(matches) != 1:
        raise OperationError("Expected one Issue reference: " + value)
    match = matches[0]
    target = match[1] or match[2] or repository
    if target.lower() != repository.lower():
        raise OperationError("Cross-repository Issue reference: " + target)
    number = int(match[3])
    if number <= 0:
        raise OperationError("Issue numbers must be positive")
    return number


def pending_pull_request_lines(lines):
    return [
        line
        for line in lines
        if line.strip() not in ("", "-")
        and not SYNCED_PR_ROW_RE.fullmatch(line.strip())
    ]


def pull_request_reference(line, repository):
    value = line.strip().removeprefix("- ").strip()
    link = MARKDOWN_LINK_RE.fullmatch(value)
    if link:
        value = link["url"]
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    match = PR_URL_RE.fullmatch(value) or PR_REFERENCE_RE.fullmatch(value)
    if match and (match[1] or repository).lower() == repository.lower():
        return int(match[2])
    return None


def render_pull_request_lines(lines, prs, repository, indent):
    confirmed = {pr["number"]: pr for pr in prs}
    result = [
        line
        for line in pending_pull_request_lines(lines)
        if pull_request_reference(line, repository) not in confirmed
    ]
    for number, pr in sorted(confirmed.items()):
        status = (
            "merged"
            if pr.get("merged_at")
            else (
                "closed"
                if pr["state"] == "closed"
                else "draft" if pr.get("draft") else "open"
            )
        )
        result.append(f"{indent}- #{number} | {status}")
    return result + [""]


def split_requirement_reference(value, repository):
    """Split a trailing reference; the URL is authoritative for Markdown links."""
    match = MARKDOWN_LINK_RE.search(value)
    if match:
        prefix = value[: match.start()].rstrip()
        label, url = match["label"], match["url"]
        if url and not (ISSUE_URL_RE.fullmatch(url) or re.fullmatch(r"#\d+", url)):
            raise OperationError(
                "Use a GitHub Issue URL or #number as the link destination"
            )
        number = issue_reference(url, repository) if url else None
        return prefix, number, label if url or not prefix else None, match.start()
    match = re.search(r"\[([^\[\]]*)\]$", value)
    if match:
        reference = match[1].strip()
        if reference and not (
            re.fullmatch(r"#\d+", reference) or ISSUE_URL_RE.fullmatch(reference)
        ):
            raise OperationError(
                "Use #number or an Issue URL to reference an existing Issue"
            )
        return (
            value[: match.start()].rstrip(),
            issue_reference(reference, repository) if reference else None,
            None,
            match.start(),
        )
    match = re.search(r"(?<!\S)(https://\S+|(?:[\w.-]+/[\w.-]+)?#\d+)$", value)
    if match:
        reference = match[1]
        if reference.startswith("https://") and not ISSUE_URL_RE.fullmatch(reference):
            raise OperationError("Use a complete GitHub Issue URL")
        return (
            value[: match.start()].rstrip(),
            issue_reference(reference, repository),
            None,
            match.start(),
        )
    return value, None, None, len(value)


def mask_bot_regions(body):
    body = re.sub(
        r"<!-- ucm-release-(navigation|overview):start -->.*?<!-- ucm-release-\1:end -->",
        lambda m: "\n" * m[0].count("\n"),
        body,
        flags=re.S,
    )
    return re.sub(
        r"<!-- ucm-requirement-template\b.*?-->",
        lambda m: "\n" * m[0].count("\n"),
        body,
        flags=re.S,
    )


def sections(body):
    lines = body.splitlines()
    heads = [
        (i, line[4:])
        for i, line in enumerate(mask_bot_regions(body).splitlines())
        if line.startswith("### ")
    ]
    found = {}
    for pos, (start, name) in enumerate(heads):
        if name in found:
            raise OperationError(f"Line {start + 1}: duplicate section {name}")
        end = heads[pos + 1][0] if pos + 1 < len(heads) else len(lines)
        found[name] = (start + 1, end)
    return lines, found


def person(value, location):
    if not value or value == "TBD":
        return None
    if not re.fullmatch(r"@[A-Za-z0-9][A-Za-z0-9-]*", value):
        raise OperationError(f"{location}: use one @username or leave blank")
    return value[1:]


def blocks(body):
    lines, found = sections(body)
    if "Requirements" not in found:
        raise OperationError(
            "Missing '### Requirements'; use the Release Checklist template"
        )
    start, end = found["Requirements"]
    visible = mask_bot_regions(body).splitlines()
    starts = [i for i in range(start, end) if visible[i].startswith("- ")]
    if any(visible[i].strip() for i in range(start, starts[0] if starts else end)):
        raise OperationError(f"Line {start + 1}: requirements must start with '- '")
    result = []
    for pos, first in enumerate(starts):
        last = starts[pos + 1] if pos + 1 < len(starts) else end
        result.append((first, last))
    return lines, result


def parse_release(body, repository, *, read_pr_requests=False):
    from folding import unfold_requirements

    body = unfold_requirements(body)
    body = mask_bot_regions(body)
    lines, found = sections(body)
    if "Release Information" not in found:
        raise OperationError(
            "Missing '### Release Information'; use the Release Checklist template"
        )
    start, end = found["Release Information"]
    info = {}
    allowed = {
        "version",
        "branch",
        "release_manager",
        "test_lead",
        "release_date",
    }
    for i in range(start, end):
        if not lines[i].strip():
            continue
        match = re.fullmatch(r"([a-z_]+):[ ]*(.*)", lines[i])
        if match and match[1] == "goal":
            continue  # Read existing checklists while dropping the retired field.
        if not match or match[1] not in allowed or match[1] in info:
            raise OperationError(f"Line {i + 1}: invalid or duplicate release field")
        info[match[1]] = match[2].strip()
    if not VERSION_RE.fullmatch(info.get("version", "")):
        raise OperationError("version: use a version such as 0.12.0")
    for role in ("release_manager", "test_lead"):
        person(info.get(role, ""), role)
    if not info.get("branch"):
        raise OperationError("branch: specify the release branch")
    if info.get("release_date") not in (None, "", "TBD"):
        try:
            date.fromisoformat(info["release_date"])
        except ValueError:
            raise OperationError("release_date: use YYYY-MM-DD or TBD") from None
    entries, keys = [], set()
    _, ranges = blocks(body)
    for first, last in ranges:

        def fail(message, line=first):
            raise OperationError(
                f"Requirement at line {first + 1}, line {line + 1}: {message}"
            )

        fields = {}
        for i in range(first + 1, last):
            line = lines[i]
            if "\t" in line[: len(line) - len(line.lstrip())]:
                fail("use spaces for indentation", i)
            if not line.strip():
                continue
            if not line.startswith("  "):
                fail("expected indentation of at least two spaces", i)
            if line.startswith("  ") and not line.startswith("   "):
                match = re.fullmatch(
                    r"  (description|assignee|pull_requests):(.*)", line
                )
                if match:
                    if match[1] in fields:
                        fail("duplicate field " + match[1], i)
                    fields[match[1]] = (i, match[2].strip())
                elif fields:
                    fail("unknown field or invalid indentation", i)
        if set(fields) != {"description", "assignee", "pull_requests"}:
            fail("expected description:, assignee:, and pull_requests:")
        desc, owner, prs = (
            fields[name][0] for name in ("description", "assignee", "pull_requests")
        )
        if not first < desc < owner < prs:
            fail("expected description, assignee, pull_requests in that order")
        if fields["description"][1] or fields["pull_requests"][1]:
            fail(
                "put description and pull_requests content on indented following lines"
            )
        title_lines = [lines[first][2:]]
        for i in range(first + 1, desc):
            if lines[i].strip():
                if not lines[i].startswith("  ") or lines[i].startswith("   "):
                    fail("title continuation must use two spaces", i)
                title_lines.append(lines[i].strip())
        title = " ".join(title_lines).strip()
        try:
            prefix, number, label, _ = split_requirement_reference(title, repository)
        except OperationError as error:
            fail(str(error))
        title = (
            prefix
            or re.sub(r"\\([\\\[\]()])", r"\1", label or "")
            or (f"Issue #{number}" if number else "")
        )
        if not title:
            fail("provide a requirement title")
        if len(title) > 256:
            fail("title exceeds 256 characters")
        for a, b in ((desc + 1, owner), (prs + 1, last)):
            for i in range(a, b):
                if lines[i].strip() and not lines[i].startswith("    "):
                    fail("field content must use at least four spaces", i)
        if any(lines[i].strip() for i in range(owner + 1, prs)):
            fail("assignee must be a single line", owner)
        description = "\n".join(
            line[4:] if line else "" for line in lines[desc + 1 : owner]
        ).strip("\n")
        key = (
            f"issue-{number}"
            if number
            else "new-" + hashlib.sha256(title.encode()).hexdigest()[:20]
        )
        if key in keys:
            fail("duplicate Issue reference or new requirement title")
        keys.add(key)
        entry = {
            "key": key,
            "number": number,
            "title": title,
            "title_text": title,
            "description": description,
            "developer": person(fields["assignee"][1], f"Line {owner + 1}"),
        }
        if read_pr_requests:
            requested = set()
            for line in pending_pull_request_lines(lines[prs + 1 : last]):
                pr = pull_request_reference(line, repository)
                if pr is None:
                    fail(
                        f"{title}: use a same-repository PR number or link: {line.strip()}"
                    )
                requested.add(pr)
            entry["requested_prs"] = sorted(requested)
        entries.append(entry)
    change_start, change_end = found.get("Change Log", (0, 0))
    return {
        "version": info["version"],
        "metadata": info,
        "entries": entries,
        "change_note": field("\n".join(lines[change_start:change_end]), "reason"),
    }


def bind_entries(entries, objects):
    return [
        dict(
            e,
            number=e["number"] or objects.get(e["key"]),
            key=(
                f"issue-{e['number'] or objects[e['key']]}"
                if e["number"] or e["key"] in objects
                else e["key"]
            ),
        )
        for e in entries
    ]


def patch_references(body, mappings, repository):
    from folding import patch_references as patch_folded_references

    folded = patch_folded_references(body, mappings, repository)
    if folded is not None:
        return folded
    parsed = parse_release(body, repository)
    lines, ranges = blocks(body)
    if len(parsed["entries"]) != len(ranges):
        raise OperationError("Requirement block count does not match parsed entries")
    for entry, (start, end) in zip(parsed["entries"], ranges):
        number = entry["number"] or mappings.get(entry["key"])
        if not number:
            continue
        title_end = next(i for i in range(start, end) if lines[i] == "  description:")
        parts = [
            (i, lines[i][2:].strip())
            for i in range(start, title_end)
            if lines[i].strip()
        ]
        title = " ".join(text for _, text in parts)
        prefix, _, label, offset = split_requirement_reference(title, repository)
        link = (
            f"#{number}"
            if prefix
            else re.sub(r"\\([\\\[\]()])", r"\1", label or "") + f" #{number}"
        )
        if offset == len(title):
            i, text = parts[-1]
            lines[i] = lines[i][:2] + text + " " + link
            continue
        cursor = 0
        for position, (i, text) in enumerate(parts):
            if offset < cursor + len(text):
                lines[i] = lines[i][:2] + text[: offset - cursor] + link
                for remaining, _ in parts[position + 1 :]:
                    lines[remaining] = ""
                break
            cursor += len(text) + 1
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def patch_pull_requests(body, rows, repository):
    from folding import patch_pull_requests as patch_folded_pull_requests

    folded = patch_folded_pull_requests(body, rows, repository)
    if folded is not None:
        return folded
    body = patch_references(body, {}, repository)
    parsed = parse_release(body, repository)
    lines, ranges = blocks(body)
    if len(parsed["entries"]) != len(ranges):
        raise OperationError("Requirement block count does not match parsed entries")
    for entry, (start, end) in reversed(list(zip(parsed["entries"], ranges))):
        first = next(i for i in range(start, end) if lines[i] == "  pull_requests:")
        lines[first + 1 : end] = render_pull_request_lines(
            lines[first + 1 : end], rows.get(entry["number"], []), repository, "    "
        )
    return "\n".join(lines).rstrip() + "\n"
