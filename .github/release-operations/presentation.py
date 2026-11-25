"""Bot-owned Release body regions, separate from human-authored scope."""

import json
import os

from syntax import OperationError

NAV_START = "<!-- ucm-release-navigation:start -->"
NAV_END = "<!-- ucm-release-navigation:end -->"
OVERVIEW_START = "<!-- ucm-release-overview:start -->"
OVERVIEW_END = "<!-- ucm-release-overview:end -->"


def replace_region(body, start, end, content):
    if body.count(start) != body.count(end) or body.count(start) > 1:
        raise OperationError("Invalid Bot region markers: " + start)
    if start in body:
        a, b = body.index(start), body.index(end)
        if b < a:
            raise OperationError("Invalid Bot region order: " + start)
        body = body[:a] + body[b + len(end) :]
    return body, start + "\n" + content + "\n" + end


def release_body(body, state):
    if "### Release Information" in body:
        navigation = []
        if state.get("milestone"):
            navigation.append(
                f"**Milestone:** [{state.get('version', '')}]({state['milestone']['html_url']})"
            )
        if state.get("archive"):
            navigation.append(f"**Bug Archive:** #{state['archive']}")
        body, nav = replace_region(
            body,
            NAV_START,
            NAV_END,
            " · ".join(navigation) or "Version links will appear after initialization.",
        )
        heading = "### Release Information"
        index = body.index(heading) + len(heading)
        body = body[:index] + "\n\n" + nav + "\n\n" + body[index:].lstrip("\r\n")

    requests = {}
    for number, record in state.get("prs", {}).items():
        if record.get("merged"):
            status = "merged"
        elif record.get("state") == "closed":
            status = "closed"
        elif record.get("draft"):
            status = "draft"
        else:
            status = "open"
        for issue in record.get("requirements", []) + record.get("bugs", []):
            requests.setdefault(issue, []).append(f"#{number} ({status})")

    rows = []
    entries = {str(e["number"]): e for e in state["scope"].values()}
    entries.update({n: entries.get(n, {}) for n in state.get("bugs", {})})
    for number, entry in entries.items():
        observed = state.get("observed", {}).get(number, {})
        owners = observed.get(
            "assignees", ([entry["developer"]] if entry.get("developer") else [])
        )
        prs = requests.get(int(number), [])
        rows.append(
            f"| #{number} | {', '.join(prs) or '-'} | {', '.join('@' + owner for owner in owners) or '-'} | {observed.get('state', 'open')} |"
        )
    overview = (
        "### Release Overview\n\nAutomatically maintained by the release bot.\n\n"
        "| Issue | Pull requests | Assignee | State |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
    )
    body, managed = replace_region(body, OVERVIEW_START, OVERVIEW_END, overview)
    return body.rstrip() + "\n\n" + managed + "\n"


def report_result(title, result):
    message = f"### {title}\n\n" + (
        result
        if isinstance(result, str)
        else "```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```"
    )
    print(message)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a", encoding="utf-8") as output:
            output.write(message + "\n\n")
