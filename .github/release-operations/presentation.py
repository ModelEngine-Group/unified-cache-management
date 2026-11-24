"""Bot-owned Release body regions, separate from human-authored scope."""

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

    rows = []
    entries = {str(e["number"]): e for e in state["scope"].values()}
    entries.update({n: entries.get(n, {}) for n in state.get("bugs", {})})
    for number, entry in entries.items():
        observed = state.get("observed", {}).get(number, {})
        owners = observed.get(
            "assignees", ([entry["developer"]] if entry.get("developer") else [])
        )
        prs = []
        for pr, record in state.get("prs", {}).items():
            if int(number) in record.get("requirements", []) + record.get("bugs", []):
                status = (
                    "merged"
                    if record.get("merged")
                    else (
                        "closed"
                        if record.get("state") == "closed"
                        else "draft" if record.get("draft") else "open"
                    )
                )
                prs.append(f"#{pr} ({status})")
        rows.append(
            f"| #{number} | {', '.join(prs) or '-'} | {', '.join('@' + owner for owner in owners) or '-'} | {observed.get('state', 'open')} |"
        )
    overview = (
        "### Release Overview\n\nAutomatically maintained by the release bot.\n\n"
    )
    operation = state.get("operation")
    if operation:
        icon = {"running": "🔄", "success": "✅", "failure": "❌"}[operation["status"]]
        overview += f"{icon} [{operation['kind']}]({operation['url']})"
        if operation.get("error"):
            overview += " — " + operation["error"]
        overview += "\n\n"
        if "preview" in operation["kind"] and operation["status"] != "running":
            overview += (
                "<details>\n<summary>Preview plan</summary>\n\n"
                + state.get("last_result", "")
                + "\n\n</details>\n\n"
            )
    warnings = state.get("sync_warnings", []) + (
        [state["pin_warning"]] if state.get("pin_warning") else []
    )
    if warnings:
        overview += (
            "<details>\n<summary>Attention required</summary>\n\n"
            + "\n".join("- " + w for w in warnings)
            + "\n\n</details>\n\n"
        )
    overview += (
        "| Issue | Pull requests | Assignee | State |\n| --- | --- | --- | --- |\n"
        + "\n".join(rows)
    )
    deliveries = state.get("deliveries", {})
    if deliveries:
        overview += "\n\n### Release Delivery\n"
        for tag, delivery in sorted(deliveries.items()):
            status = delivery["status"]
            icon = (
                "✅"
                if status == "delivered"
                else "❌" if status in ("failed", "cancelled") else "🔄"
            )
            overview += (
                f"\n- {icon} **{tag}: {status.title()}**"
                f" · [Workflow]({delivery['url']})"
                f" · Attempt {delivery['run_attempt']}"
            )
            if delivery.get("release_url"):
                overview += f" · [GitHub Release]({delivery['release_url']})"
            if delivery.get("completed_at"):
                overview += " · " + delivery["completed_at"]
            if delivery.get("error"):
                overview += " — " + delivery["error"]
    check = state.get("release_check")
    if check:
        overview += "\n\n### Release Check\n\n"
        if check["status"] == "not_checked":
            overview += "Run `/release_check` to inspect the current release."
        else:
            overview += (
                "⚠️ Pending items"
                if check["status"] == "blocked"
                else "✅ No blocking items"
            )
            overview += f" — [Latest check]({check['url']})\n"
            if check.get("candidate"):
                candidate = check["candidate"]
                overview += (
                    f"\nTag source: `{candidate['ref']}` (`{candidate['sha']}`).\n"
                )
            categories = (
                ("features_without_pr", "Feature Issues without PRs"),
                ("open_features", "Open feature Issues"),
                ("open_bugs", "Open Bugs targeted for this version"),
                ("unmerged_prs", "Open PRs awaiting merge"),
                ("closed_unmerged_prs", "Closed PRs without merge (review)"),
                ("other_version_prs", "PRs assigned to another version (review)"),
                ("version_conflicts", "PR target-version conflicts"),
            )
            for key, title in categories:
                if check.get(key):
                    overview += (
                        "\n- **"
                        + title
                        + ":** "
                        + ", ".join(f"#{n}" for n in check[key])
                    )
            for key, title in (
                ("added", "Added requirements"),
                ("removed", "Removed requirements"),
                ("changed", "Edited requirements"),
                ("metadata", "Edited release fields"),
            ):
                if check["unsynced"].get(key):
                    overview += (
                        "\n- **Pending `/update` — "
                        + title
                        + ":** "
                        + ", ".join(check["unsynced"][key])
                    )
            overview += "\n\nSnapshot of the latest requested check. Run `/release_check` to refresh."
    if state.get("history"):
        overview += (
            "\n\n<details>\n<summary>Scope history</summary>\n\n"
            + "\n".join(state["history"][-8:])
            + "\n\n</details>"
        )
    body, managed = replace_region(body, OVERVIEW_START, OVERVIEW_END, overview)
    return body.rstrip() + "\n\n" + managed + "\n"
