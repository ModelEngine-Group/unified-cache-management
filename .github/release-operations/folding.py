"""Collapsible requirement cards with a flat representation for scope parsing."""

import re
from html import escape, unescape

from syntax import OperationError, mask_bot_regions, render_pull_request_lines, sections

ISSUE_STATE_RE = re.compile(
    r"\s*<!-- ucm-issue-state:(?:open|closed) -->\s*(?:🟢 open|🟣 closed)\s*$"
)


def cards(body):
    lines, found = sections(body)
    if "Requirements" not in found:
        return None
    start, end = found["Requirements"]
    visible = mask_bot_regions(body).splitlines()
    first = next((i for i in range(start, end) if visible[i].strip()), end)
    if first == end or not lines[first].startswith("<details"):
        return None
    result, i = [], first

    def fail(index, message):
        raise OperationError(f"Requirement at line {index + 1}: {message}")

    while i < end:
        if not visible[i].strip():
            i += 1
            continue
        begin = i
        if lines[i] not in ("<details>", "<details open>"):
            fail(i, "expected <details> for each requirement")
        i += 1
        while i < end and not lines[i].strip():
            i += 1
        summary = i
        if i == end or not lines[i].startswith("<summary>"):
            fail(begin, "expected <summary>Requirement title</summary>")
        title, text = [], lines[i][len("<summary>") :]
        while not text.endswith("</summary>"):
            text = ISSUE_STATE_RE.sub("", text)
            if "<" in text or i + 1 == end:
                fail(i, "unclosed or invalid summary")
            title.append(unescape(text.strip()))
            i += 1
            text = lines[i].strip()
        title.append(
            unescape(ISSUE_STATE_RE.sub("", text[: -len("</summary>")]).strip())
        )
        title = [part for part in title if part]
        if not title:
            fail(summary, "provide a requirement title")
        summary_end = i + 1
        i += 1
        fields = {}
        while i < end and lines[i] != "</details>":
            line = lines[i]
            if line.strip():
                match = re.fullmatch(r"(assignee|description|pull_requests):(.*)", line)
                if match:
                    if match[1] in fields:
                        fail(i, "duplicate field " + match[1])
                    fields[match[1]] = (i, match[2].strip())
                elif not line.startswith("  ") or not fields:
                    fail(i, "expected a field or content indented by two spaces")
                if "\t" in line[: len(line) - len(line.lstrip())]:
                    fail(i, "use spaces for indentation")
            i += 1
        if i == end:
            fail(begin, "missing </details>")
        if list(fields) != ["assignee", "description", "pull_requests"]:
            fail(
                begin, "expected assignee:, description:, pull_requests: in that order"
            )
        owner, desc, prs = [fields[name][0] for name in fields]
        if fields["description"][1] or fields["pull_requests"][1]:
            fail(begin, "put description and pull_requests content on following lines")
        if any(line.strip() for line in lines[owner + 1 : desc]):
            fail(owner, "assignee must be a single line")
        result.append(
            dict(
                start=begin,
                end=i + 1,
                close=i,
                summary=summary,
                summary_end=summary_end,
                title=title,
                fields=fields,
            )
        )
        i += 1
    return lines, result


def unfold_requirements(body):
    from taskcards import as_folded

    body = as_folded(body)
    found = cards(body)
    if found is None:
        return body
    lines, records = found
    for card in reversed(records):
        fields = card["fields"]
        desc, prs = fields["description"][0], fields["pull_requests"][0]
        description = lines[desc + 1 : prs]
        pulls = lines[prs + 1 : card["close"]]
        block = ["- " + " ".join(card["title"]), "", "  description:"]
        block += ["  " + line if line else "" for line in description]
        block += ["  assignee: " + fields["assignee"][1], "", "  pull_requests:"]
        block += ["  " + line if line else "" for line in pulls]
        lines[card["start"] : card["end"]] = block + [""]
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def patch_references(body, mappings, repository):
    from taskcards import patch

    task = patch(
        body, repository, lambda value: patch_references(value, mappings, repository)
    )
    if task is not None:
        return task
    found = cards(body)
    if found is None:
        return None
    from syntax import parse_release, split_requirement_reference

    lines, records = found
    entries = parse_release(body, repository)["entries"]
    if len(records) != len(entries):
        raise OperationError("Requirement card count does not match parsed entries")
    for card, entry in reversed(list(zip(records, entries))):
        number = entry["number"] or mappings.get(entry["key"])
        if not number:
            continue
        parts = card["title"]
        title = " ".join(parts)
        prefix, _, label, offset = split_requirement_reference(title, repository)
        link = (
            f"#{number}"
            if prefix
            else re.sub(r"\\([\\\[\]()])", r"\1", label or "") + f" #{number}"
        )
        if offset == len(title):
            parts[-1] += " " + link
        else:
            cursor = 0
            for position, text in enumerate(parts):
                if offset < cursor + len(text):
                    parts = parts[:position] + [text[: offset - cursor] + link]
                    break
                cursor += len(text) + 1
        rendered = [escape(part, quote=False) for part in parts]
        rendered[0] = "<summary>" + rendered[0]
        rendered[-1] += "</summary>"
        lines[card["summary"] : card["summary_end"]] = rendered
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def patch_pull_requests(body, rows, repository):
    from taskcards import patch

    task = patch(
        body, repository, lambda value: patch_pull_requests(value, rows, repository)
    )
    if task is not None:
        return task
    if cards(body) is None:
        return None
    from syntax import parse_release

    body = patch_references(body, {}, repository)
    lines, records = cards(body)
    entries = parse_release(body, repository)["entries"]
    if len(records) != len(entries):
        raise OperationError("Requirement card count does not match parsed entries")
    for card, entry in reversed(list(zip(records, entries))):
        first, end = card["fields"]["pull_requests"][0] + 1, card["close"]
        lines[first:end] = render_pull_request_lines(
            lines[first:end], rows.get(entry["number"], []), repository, "  "
        )
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def fold_requirements(body, repository):
    """Convert a flat checklist without changing its semantic release scope."""
    if cards(body) is not None:
        return body
    from syntax import blocks, parse_release
    from syntax import patch_references as patch_flat_references

    body = patch_flat_references(body, {}, repository)
    entries = parse_release(body, repository)["entries"]
    lines, ranges = blocks(body)
    if len(entries) != len(ranges):
        raise OperationError("Requirement block count does not match parsed entries")
    for entry, (start, end) in reversed(list(zip(entries, ranges))):
        desc = next(i for i in range(start, end) if lines[i] == "  description:")
        title = [lines[start][2:]] + [
            line.strip() for line in lines[start + 1 : desc] if line.strip()
        ]
        title = [escape(part, quote=False) for part in title]
        title[0] = "<summary>" + title[0]
        title[-1] += "</summary>"
        prs = next(i for i in range(start, end) if lines[i] == "  pull_requests:")
        pulls = [line[2:] if line else "" for line in lines[prs + 1 : end]]
        block = (
            ["<details>"]
            + title
            + [
                "",
                (
                    "assignee: "
                    + ("@" + entry["developer"] if entry["developer"] else "")
                ).rstrip(),
                "",
                "description:",
            ]
        )
        block += [
            "  " + line if line else "" for line in entry["description"].split("\n")
        ]
        block += ["", "pull_requests:"] + pulls + ["</details>", ""]
        lines[start:end] = block
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")
