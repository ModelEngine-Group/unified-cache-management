"""Manage GitHub Development relationships and synchronize their metadata."""

from ownership import milestone_number
from syntax import OperationError


class PullRequestAssociations:
    def link_requested_pull_requests(self, entries):
        targets = {
            entry["number"]: entry["requested_prs"]
            for entry in entries
            if entry["requested_prs"]
        }
        if not targets:
            return
        numbers = {pr for prs in targets.values() for pr in prs}
        prs = self.pull_requests(numbers)
        # The preflight PR records can predate both the links and Issue Milestones.
        for number in numbers:
            self._pull_requests.pop(number, None)
            self._pr_issues.pop(number, None)
        native = self.native_references(targets)
        for issue, requested in targets.items():
            missing = [pr for pr in requested if issue not in native.get(pr, set())]
            if not missing:
                continue
            issue_id = self.issue(issue)["node_id"]
            self._native_issue_snapshot.pop(issue, None)
            try:
                self.api.call(
                    "graphql",
                    "POST",
                    {
                        "query": "mutation($input:AddCloseIssueReferencesInput!){addCloseIssueReferences(input:$input){issue{id}}}",
                        "variables": {
                            "input": {
                                "issueId": issue_id,
                                "pullRequestIds": [
                                    prs[pr]["node_id"] for pr in missing
                                ],
                            }
                        },
                    },
                )
            except OperationError as error:
                refs = ", ".join(f"#{pr}" for pr in missing)
                raise OperationError(
                    f"Could not link {refs} to Issue #{issue}: {error}"
                ) from error
        native = self.native_references(targets)
        self.pull_requests(numbers)
        for issue, requested in targets.items():
            for pr in requested:
                if (
                    issue not in native.get(pr, set())
                    or issue not in self._pr_issues[pr]
                ):
                    raise OperationError(
                        f"Development link for PR #{pr} and Issue #{issue} is not visible yet; retry the command"
                    )

    def native_references(self, numbers):
        owner, name = self.repository.split("/")
        cache = getattr(self, "_native_issue_snapshot", None)
        if cache is None:
            cache = self._native_issue_snapshot = {}
        missing = sorted(set(numbers) - cache.keys())
        for offset in range(0, len(missing), 10):
            batch = missing[offset : offset + 10]
            fields = " ".join(
                f"i{n}:issue(number:{n}){{closedByPullRequestsReferences(first:100,includeClosedPrs:true,excludeUserLinked:false){{nodes{{number repository{{nameWithOwner}}}} pageInfo{{hasNextPage endCursor}}}}}}"
                for n in batch
            )
            data = self.api.call(
                "graphql",
                "POST",
                {
                    "query": "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){"
                    + fields
                    + "}}",
                    "variables": {"owner": owner, "name": name},
                },
            )["data"]["repository"]
            for number in batch:
                issue = data[f"i{number}"]
                if issue is None:
                    raise OperationError(
                        f"Native association Issue #{number} is unavailable"
                    )
                page = issue["closedByPullRequestsReferences"]
                nodes = list(page["nodes"])
                while page["pageInfo"]["hasNextPage"]:
                    page = self.api.call(
                        "graphql",
                        "POST",
                        {
                            "query": "query($owner:String!,$name:String!,$number:Int!,$after:String){repository(owner:$owner,name:$name){issue(number:$number){closedByPullRequestsReferences(first:100,after:$after,includeClosedPrs:true,excludeUserLinked:false){nodes{number repository{nameWithOwner}} pageInfo{hasNextPage endCursor}}}}}",
                            "variables": {
                                "owner": owner,
                                "name": name,
                                "number": number,
                                "after": page["pageInfo"]["endCursor"],
                            },
                        },
                    )["data"]["repository"]["issue"]["closedByPullRequestsReferences"]
                    nodes.extend(page["nodes"])
                cache[number] = {
                    p["number"]
                    for p in nodes
                    if p["repository"]["nameWithOwner"].lower()
                    == self.repository.lower()
                }
        result = {}
        for number in numbers:
            for pr in cache[number]:
                result.setdefault(pr, set()).add(number)
        return result

    def pull_requests(self, numbers):
        owner, name = self.repository.split("/")
        if not hasattr(self, "_pull_requests"):
            self._pull_requests, self._pr_issues = {}, {}
        missing = sorted(set(numbers) - self._pull_requests.keys())
        issue_fields = "number id title state labels(first:100){nodes{name}} milestone{number title} repository{nameWithOwner}"
        for offset in range(0, len(missing), 20):
            batch = missing[offset : offset + 20]
            fields = " ".join(
                f"p{n}:pullRequest(number:{n}){{number id url state mergedAt isDraft milestone{{number title}} closingIssuesReferences(first:100,excludeUserLinked:false){{nodes{{{issue_fields}}} pageInfo{{hasNextPage endCursor}}}}}}"
                for n in batch
            )
            data = self.api.call(
                "graphql",
                "POST",
                {
                    "query": "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){"
                    + fields
                    + "}}",
                    "variables": {"owner": owner, "name": name},
                },
            )["data"]["repository"]
            for number in batch:
                pr = data[f"p{number}"]
                if pr is None:
                    raise OperationError(f"Linked PR #{number} is unavailable")
                page = pr.pop("closingIssuesReferences")
                nodes = list(page["nodes"])
                while page["pageInfo"]["hasNextPage"]:
                    page = self.api.call(
                        "graphql",
                        "POST",
                        {
                            "query": "query($owner:String!,$name:String!,$number:Int!,$after:String){repository(owner:$owner,name:$name){pullRequest(number:$number){closingIssuesReferences(first:100,after:$after,excludeUserLinked:false){nodes{"
                            + issue_fields
                            + "} pageInfo{hasNextPage endCursor}}}}}",
                            "variables": {
                                "owner": owner,
                                "name": name,
                                "number": number,
                                "after": page["pageInfo"]["endCursor"],
                            },
                        },
                    )["data"]["repository"]["pullRequest"]["closingIssuesReferences"]
                    nodes.extend(page["nodes"])
                self._pr_issues[number] = {
                    i["number"]: dict(
                        i,
                        node_id=i["id"],
                        state=i["state"].lower(),
                        labels=i["labels"]["nodes"],
                    )
                    for i in nodes
                    if i["repository"]["nameWithOwner"].lower()
                    == self.repository.lower()
                }
                self._pull_requests[number] = dict(
                    pr,
                    node_id=pr["id"],
                    html_url=pr["url"],
                    state=pr["state"].lower(),
                    merged_at=pr["mergedAt"],
                    draft=pr["isDraft"],
                )
        return {n: self._pull_requests[n] for n in numbers}

    def memberships(self, item):
        self.prefetch_memberships([item])
        return self._project_memberships[item["node_id"]]

    def prefetch_memberships(self, items):
        if self._project_memberships is None:
            self._project_memberships, self._project_status = {}, {}
        missing = sorted(
            {item["node_id"] for item in items} - self._project_memberships.keys()
        )
        selection = """projectItems(first:100,after:$after){
            nodes{id project{id} fieldValueByName(name:"Status"){... on ProjectV2ItemFieldSingleSelectValue{name}}}
            pageInfo{hasNextPage endCursor}}"""
        fields = "... on Issue{" + selection + "} ... on PullRequest{" + selection + "}"
        for offset in range(0, len(missing), 20):
            batch = missing[offset : offset + 20]
            nodes = self.api.graphql(
                "query($ids:[ID!]!,$after:String){nodes(ids:$ids){id " + fields + "}}",
                ids=batch,
                after=None,
            )["nodes"]
            if any(node is None for node in nodes):
                raise OperationError(
                    "Project token cannot access all linked GitHub items"
                )
            for node in nodes:
                item_id = node["id"]
                page = node["projectItems"]
                memberships = {}
                while True:
                    for row in page["nodes"]:
                        memberships[row["project"]["id"]] = row["id"]
                        self._project_status[row["id"]] = (
                            row.get("fieldValueByName") or {}
                        ).get("name")
                    if not page["pageInfo"]["hasNextPage"]:
                        break
                    page = self.api.graphql(
                        "query($id:ID!,$after:String){node(id:$id){" + fields + "}}",
                        id=item_id,
                        after=page["pageInfo"]["endCursor"],
                    )["node"]["projectItems"]
                self._project_memberships[item_id] = memberships

    def sync_pr_projects(self, pr, desired):
        if not desired:
            return
        current = self.memberships(pr)
        for project in sorted(desired - current.keys()):
            item = self.api.graphql(
                """mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
                project=project,
                content=pr["node_id"],
            )["addProjectV2ItemById"]["item"]
            current[project] = item["id"]

    def sync_pull_requests(self, state, issues):
        bugs = {int(number) for number in state["bugs"]}
        native = self.native_references(issues)
        prs = self.pull_requests(native)
        rows, managed_links = {}, {}
        for number, related in sorted(native.items()):
            pr = prs[number]
            links = []
            for item in self._pr_issues[number].values():
                milestone = milestone_number(item)
                labels = {label["name"] for label in item["labels"]}
                if (
                    item["number"] in issues
                    or labels & {"bug", "requirement"}
                    or (milestone is not None and self.checklists(milestone))
                ):
                    links.append(item)
            targets = {milestone_number(item) for item in links}
            complete = None not in targets
            targets.discard(None)
            current = milestone_number(pr)
            if current is None and complete and len(targets) == 1:
                current = next(iter(targets))
                pr["milestone"] = self.api.rest(
                    f"issues/{number}", "PATCH", {"milestone": current}
                )["milestone"]
            state["prs"][str(number)] = {
                "number": number,
                "requirements": sorted(related - bugs),
                "bugs": sorted(related & bugs),
                "merged": bool(pr["merged_at"]),
                "state": pr["state"],
                "draft": pr["draft"],
                "conflict": len(targets) > 1 and current is None,
            }
            managed_links[number] = links
            for issue in related:
                rows.setdefault(issue, []).append(pr)

        bugfix = self.config["projects"].get("Bugfix")
        project_sources = {
            item["number"]: item
            for links in managed_links.values()
            for item in links
            if item["number"] not in bugs
            and not any(label["name"] == "bug" for label in item["labels"])
        }
        self.prefetch_memberships(project_sources.values())
        projects = {}
        for number, links in managed_links.items():
            desired = set()
            for item in links:
                if item["number"] in project_sources:
                    desired.update(self.memberships(item))
                elif bugfix:
                    desired.add(bugfix["id"])
            if desired:
                projects[number] = desired
        self.prefetch_memberships([prs[number] for number in projects])
        for number, desired in projects.items():
            self.sync_pr_projects(prs[number], desired)
        return rows
