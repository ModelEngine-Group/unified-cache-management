"""Read explicit GitHub Development relationships and synchronize their metadata."""

from ownership import milestone_number
from syntax import OperationError, parse_release, patch_pull_requests


class PullRequestAssociations:
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
        cache = getattr(self, "_project_memberships", None)
        if cache is None:
            cache = self._project_memberships = {}
            self._project_status = {}
        if item["node_id"] in cache:
            return cache[item["node_id"]]
        result, cursor = {}, None
        while True:
            page = self.api.graphql(
                """query($id:ID!,$after:String){node(id:$id){
                ... on Issue{projectItems(first:100,after:$after){nodes{id project{id} fieldValueByName(name:"Status"){... on ProjectV2ItemFieldSingleSelectValue{name}}} pageInfo{hasNextPage endCursor}}}
                ... on PullRequest{projectItems(first:100,after:$after){nodes{id project{id} fieldValueByName(name:"Status"){... on ProjectV2ItemFieldSingleSelectValue{name}}} pageInfo{hasNextPage endCursor}}}
                }}""",
                id=item["node_id"],
                after=cursor,
            )["node"]["projectItems"]
            result.update({n["project"]["id"]: n["id"] for n in page["nodes"]})
            for node in page["nodes"]:
                self._project_status[node["id"]] = (
                    node.get("fieldValueByName") or {}
                ).get("name")
            if not page["pageInfo"]["hasNextPage"]:
                cache[item["node_id"]] = result
                return result
            cursor = page["pageInfo"]["endCursor"]

    def sync_pr_projects(self, release, state, pr, record, desired):
        managed = record.setdefault("projects", {})
        if not desired and not managed:
            return
        current = self.memberships(pr)
        for project in desired | set(managed):
            existing = managed.get(project)
            if project not in desired:
                if existing and existing["mode"] == "auto" and project in current:
                    self.api.graphql(
                        """mutation($project:ID!,$item:ID!){deleteProjectV2Item(input:{projectId:$project,itemId:$item}){deletedItemId}}""",
                        project=project,
                        item=current[project],
                    )
                    current.pop(project, None)
                managed.pop(project, None)
                continue
            if existing is None:
                existing = managed[project] = {
                    "mode": "manual" if project in current else "auto",
                    "item": current.get(project),
                }
            if (
                existing["mode"] == "auto"
                and existing.get("item")
                and project not in current
            ):
                existing["mode"] = "manual_removed"
            if existing["mode"] == "auto" and project in current:
                existing["item"] = current[project]
            if existing["mode"] == "auto" and project not in current:
                # Save ownership before an API request whose response can be lost.
                self.save(release, state, f"Synchronizing PR #{pr['number']} Projects")
                item = self.api.graphql(
                    """mutation($project:ID!,$content:ID!){addProjectV2ItemById(input:{projectId:$project,contentId:$content}){item{id}}}""",
                    project=project,
                    content=pr["node_id"],
                )["addProjectV2ItemById"]["item"]
                existing["item"] = item["id"]
                current[project] = item["id"]

    def sync_pull_requests(self, release, state, event=None):
        requirements = {e["number"] for e in state["scope"].values()}
        bugs = {int(n) for n in state["bugs"]} | {
            i["number"] for i in self._target_bugs.get(release, [])
        }
        native = self.native_references(requirements | bugs)
        cleanup = {int(n) for n, p in state["prs"].items() if p.get("projects")}
        self.pull_requests(set(native) | cleanup)
        directory = self.directory()
        managed = (
            {i for row in directory.rows().values() for i in row["issues"]}
            | requirements
            | bugs
        )
        rows, problems = {}, []
        records = {}
        for number in sorted(set(native) | cleanup):
            pr = self._pull_requests[number]
            links = [
                self._bug_items.get(n, i)
                for n, i in self._pr_issues[number].items()
                if n in managed or "bug" in {label["name"] for label in i["labels"]}
            ]
            targets = {milestone_number(i) for i in links}
            complete = None not in targets
            targets.discard(None)
            current = milestone_number(pr)
            first = directory.claim_prs({number}) if number in native else set()
            if number in first and current is None and complete and len(targets) == 1:
                pr["milestone"] = self.api.rest(
                    f"issues/{number}", "PATCH", {"milestone": next(iter(targets))}
                ).get("milestone")
                current = milestone_number(pr)
            related = native.get(number, set())
            record = state["prs"].get(str(number), {})
            record.update(
                requirements=sorted(related & requirements),
                bugs=sorted(related & bugs),
                merged=bool(pr["merged_at"]),
                state=pr["state"],
                draft=pr["draft"],
                last=current,
                target_title=(pr.get("milestone") or {}).get("title"),
                conflict=len(targets) > 1 and current is None,
            )
            state["prs"][str(number)] = record
            if record["conflict"]:
                problems.append(
                    f"PR #{number}: linked Issues have conflicting Milestones; set the PR Milestone manually"
                )
            projects = set()
            if links:
                for issue in links:
                    if "bug" in {label["name"] for label in issue["labels"]}:
                        projects.add(self.config["projects"]["Bugfix"]["id"])
                    else:
                        projects.update(self.memberships(issue))
            self.sync_pr_projects(release, state, pr, record, projects)
            if related or record.get("projects"):
                records[str(number)] = record
            for issue in related & requirements:
                rows.setdefault(issue, []).append(pr)
        state["prs"] = records
        source = self.issue(release)
        try:
            parse_release(source["body"], self.repository)
        except OperationError as error:
            problems.append("Release body requires attention: " + str(error))
            return problems
        patched = patch_pull_requests(source["body"], rows, self.repository)
        if patched != source["body"]:
            if self.issue(release)["body"] != source["body"]:
                raise OperationError(
                    "Release body changed during synchronization; retry /update"
                )
            self.api.rest(f"issues/{release}", "PATCH", {"body": patched})
        return problems
