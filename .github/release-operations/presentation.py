"""Bot-owned Release body regions, separate from human-authored scope."""

import json
import os
from html import escape

from syntax import OperationError

NAV_START = "<!-- ucm-release-navigation:start -->"
NAV_END = "<!-- ucm-release-navigation:end -->"
OVERVIEW_START = "<!-- ucm-release-overview:start -->"
OVERVIEW_END = "<!-- ucm-release-overview:end -->"
COMMAND_START = "<!-- ucm-release-command:start -->"
COMMAND_END = "<!-- ucm-release-command:end -->"


def replace_region(body, start, end, content=None):
    if body.count(start) != body.count(end) or body.count(start) > 1:
        raise OperationError("Invalid Bot region markers: " + start)
    section = ""
    if start in body:
        a, b = body.index(start), body.index(end)
        if b < a:
            raise OperationError("Invalid Bot region order: " + start)
        b += len(end)
        section = body[a:b]
        body = body[:a] + body[b:]
    if content is not None:
        section = start + "\n" + content + "\n" + end
    return body, section


def append_checklist_section(body, section):
    body, feedback = replace_region(body, COMMAND_START, COMMAND_END)
    sections = (body.rstrip(), section, feedback)
    return "\n\n".join(part for part in sections if part) + "\n"


def checklist_region(body, name, content=None):
    start = f"<!-- ucm-release-{name}:start -->"
    end = f"<!-- ucm-release-{name}:end -->"
    return replace_region(body, start, end, content)


def set_checklist_region(body, name, content):
    remainder, old = checklist_region(body, name)
    if not content and not old:
        return body
    section = checklist_region("", name, content)[1] if content else ""
    if old == section:
        return body
    if old and section:
        return body.replace(old, section, 1)
    return append_checklist_section(remainder, section)


def warning_text(message):
    return escape(" ".join(str(message).split())[:500])


PIN_WARNING_PREFIX = "- **Pinning** · "


def update_attention(body, *, warnings=None, pin_warning=None):
    _, old = checklist_region(body, "attention")
    lines = [line for line in old.splitlines() if line.startswith("- ")]
    pin = next((line for line in lines if line.startswith(PIN_WARNING_PREFIX)), "")
    if pin_warning is not None:
        pin = PIN_WARNING_PREFIX + warning_text(pin_warning) if pin_warning else ""
    if warnings is None:
        lines = [line for line in lines if not line.startswith(PIN_WARNING_PREFIX)]
    else:
        lines = ["- " + warning_text(warning) for warning in warnings]
    if pin:
        lines.append(pin)
    content = (
        "<details>\n<summary>Attention required</summary>\n\n"
        + "\n".join(lines)
        + "\n\n</details>"
        if lines
        else ""
    )
    return set_checklist_region(body, "attention", content)


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
    return set_checklist_region(body, "overview", overview)


def command_feedback(body, command, status, details, comment_url, run_url):
    icon = {
        "Running": "🔄",
        "Succeeded": "✅",
        "Blocked": "⚠️",
        "Failed": "❌",
        "Cancelled": "⚪",
    }[status]
    destinations = (("Command comment", comment_url), ("Actions Summary", run_url))
    links = [f"[{label}]({url})" for label, url in destinations if url]
    heading = f"### Latest Release Command\n\n{icon} **{status}** · `{command}`"
    if links:
        heading += " · " + " · ".join(links)
    body, _ = replace_region(body, COMMAND_START, COMMAND_END)
    body = body.rstrip()
    content = heading
    if details:
        if len(body) + len(details) + len(heading) < 60_000:
            preview = status == "Succeeded" and command.endswith(" --dry-run")
            label = "Preview plan" if preview else "Details"
            content += (
                f"\n\n<details>\n<summary>{label}</summary>"
                f"\n\n{details}\n\n</details>"
            )
        else:
            content += "\n\nSee the Actions Summary for the full result."
    _, section = replace_region("", COMMAND_START, COMMAND_END, content)
    return body + "\n\n" + section + "\n"


def preview_details(entries):
    lines = []
    for entry in entries:
        title = escape(entry["title"])
        if entry["issue"]:
            line = f"- Reuse Issue #{entry['issue']} · {title}"
        else:
            line = f"- Create Issue · {title}"
        if entry["requested_prs"]:
            line += " · Link PRs " + ", ".join(f"#{n}" for n in entry["requested_prs"])
        lines.append(line)
    return "\n".join(lines)


def check_details(report, *, split_open_issues=False):
    issue_categories = (
        (
            ("open_checklists", "Open Checklist or Bug Archive"),
            ("open_requirements", "Open requirements and other Issues"),
            ("open_bugs", "Open Bugs"),
        )
        if split_open_issues
        else (("open_issues", "Open Issues"),)
    )
    categories = issue_categories + (
        ("unmerged_prs", "Open PRs awaiting merge"),
        ("version_conflicts", "PR Milestone conflicts"),
        ("issues_without_pr", "Issues without PRs"),
        ("closed_unmerged_prs", "Closed PRs without merge"),
        ("other_version_prs", "PRs assigned to another version"),
    )
    lines = []
    if report["pending_requirements"]:
        lines.append(
            "- Pending requirements or PR links · "
            + ", ".join(escape(title) for title in report["pending_requirements"])
        )
    for key, label in categories:
        if report[key]:
            lines.append(f"- {label} · " + ", ".join(f"#{n}" for n in report[key]))
    for key, label in (
        ("added", "Added requirements"),
        ("removed", "Removed requirements"),
        ("changed", "Edited requirements"),
        ("metadata", "Edited release fields"),
    ):
        values = report.get("unsynced", {}).get(key, [])
        if values:
            lines.append(
                f"- Pending `/update` · {label} · "
                + ", ".join(escape(value) for value in values)
            )
    return "\n".join(lines)


def release_check_body(body, report, run_url):
    status = "⚠️ **Blocked**" if report["status"] == "blocked" else "✅ **Ready**"
    if run_url:
        status += f" · [Actions Summary]({run_url})"
    content = f"### Release Check\n\n{status}"
    if candidate := report.get("candidate"):
        content += f"\n\nCandidate `{candidate['ref']}` · `{candidate['sha']}`"
    if details := check_details(report, split_open_issues=True):
        remainder, _ = checklist_region(body, "check")
        if len(remainder) + len(content) + len(details) < 60_000 or not run_url:
            content += "\n\n" + details
        else:
            content += "\n\nSee the Actions Summary for the full report."
    return set_checklist_region(body, "check", content)


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
