"""Milestone ownership shared by requirements, Bugs, and PRs."""


def milestone_number(issue):
    return issue["milestone"]["number"] if issue.get("milestone") else None


def manual_mode(record, current, human_event=False):
    updated = dict(record)
    if human_event or ("last" in record and current != record["last"]):
        updated.update(mode="manual", last=current)
    return updated
