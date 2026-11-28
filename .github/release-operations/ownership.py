"""Milestone ownership shared by requirements, Bugs, and PRs."""


def milestone_number(issue):
    return issue["milestone"]["number"] if issue.get("milestone") else None
