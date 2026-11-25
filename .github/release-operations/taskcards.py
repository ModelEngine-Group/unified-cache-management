"""Native Issue task-list references with their details nested underneath."""

import re
from html import escape, unescape

from syntax import OperationError, mask_bot_regions, sections

HEADER = re.compile(r"- \[([ xX])\] (.+)")


def tasks(body):
    lines, found = sections(body)
    if "Requirements" not in found:
        return None
    start, end = found["Requirements"]
    visible = mask_bot_regions(body).splitlines()
    i = next((n for n in range(start, end) if visible[n].strip()), end)
    if i == end or HEADER.fullmatch(lines[i]) is None:
        return None
    records = []
    while i < end:
        if not visible[i].strip():
            i += 1
            continue
        first = i
        match = HEADER.fullmatch(lines[i])
        if not match:
            raise OperationError(f"Line {i+1}: expected a requirement task-list item")
        title = [unescape(match[2])]
        i += 1
        while i < end and lines[i] not in ("  <details>", "  <details open>"):
            if lines[i].strip():
                if not lines[i].startswith("  ") or lines[i].lstrip().startswith("<"):
                    raise OperationError(f"Line {i+1}: expected indented <details>")
                title.append(unescape(lines[i].strip()))
            i += 1
        details = i
        i += 1
        while i < end and lines[i] != "  </details>":
            if lines[i].strip() and not lines[i].startswith("  "):
                raise OperationError(
                    f"Line {i+1}: indent requirement details by two spaces"
                )
            i += 1
        if i == end or details >= end:
            raise OperationError(f"Line {first+1}: missing indented </details>")
        inner = [line[2:] if line else "" for line in lines[details + 1 : i]]
        summary = next((n for n, line in enumerate(inner) if line.strip()), None)
        if summary is None or inner[summary] != "<summary>Details</summary>":
            raise OperationError(
                f"Line {details+2}: expected <summary>Details</summary>"
            )
        title = [escape(part, quote=False) for part in title]
        title[0] = "<summary>" + title[0]
        title[-1] += "</summary>"
        inner[summary : summary + 1] = title
        records.append(
            dict(
                start=first,
                end=i + 1,
                checked=match[1],
                folded=[lines[details][2:]] + inner + ["</details>"],
            )
        )
        i += 1
    return lines, records


def as_folded(body):
    found = tasks(body)
    if found is None:
        return body
    lines, records = found
    for record in reversed(records):
        lines[record["start"] : record["end"]] = record["folded"]
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def task_requirements(body, repository, checks=None):
    """Convert existing cards while preserving their semantic scope and checkbox state."""
    from folding import cards, fold_requirements

    found = tasks(body)
    if found is not None:
        return body
    body = fold_requirements(body, repository)
    found = cards(body)
    if found is None:
        return body
    lines, records = found
    for index, card in reversed(list(enumerate(records))):
        title = [escape(part, quote=False) for part in card["title"]]
        checked = checks[index] if checks is not None else " "
        block = [f"- [{checked}] " + title[0]] + ["  " + part for part in title[1:]]
        block += ["", "  " + lines[card["start"]], "  <summary>Details</summary>"]
        block += [
            "  " + line if line else ""
            for line in lines[card["summary_end"] : card["end"]]
        ]
        lines[card["start"] : card["end"]] = block
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


def patch(body, repository, update):
    found = tasks(body)
    if found is None:
        return None
    updated = update(as_folded(body))
    return task_requirements(updated, repository, [r["checked"] for r in found[1]])
