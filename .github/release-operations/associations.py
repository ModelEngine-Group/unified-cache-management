"""Reconcile PR metadata from current Issue mentions and native Projects."""

from ownership import manual_mode, milestone_number
from syntax import OperationError, parse_release, patch_pull_requests, references


class PullRequestAssociations:
    def native_references(self, numbers, cutoffs=None):
        """Read Development links, retaining closed and merged PRs."""
        owner, name = self.repository.split("/")
        cache = getattr(self, "_native_issue_snapshot", None)
        if cache is None:
            cache = self._native_issue_snapshot = {}
        dates = getattr(self, "_native_pr_dates", None)
        if dates is None:
            dates = self._native_pr_dates = {}
        refs = {}
        for number in sorted(numbers):
            if number not in cache:
                linked, cursor = [], None
                while True:
                    issue = self.api.call(
                        "graphql",
                        "POST",
                        {
                            "query": """query($owner:String!,$name:String!,$number:Int!,$after:String){repository(owner:$owner,name:$name){issue(number:$number){url closedByPullRequestsReferences(first:100,after:$after,includeClosedPrs:true,excludeUserLinked:false){nodes{number createdAt repository{nameWithOwner}} pageInfo{hasNextPage endCursor}}}}}""",
                            "variables": {
                                "owner": owner,
                                "name": name,
                                "number": number,
                                "after": cursor,
                            },
                        },
                    )["data"]["repository"]["issue"]
                    if issue is None:
                        raise OperationError(
                            f"Native association Issue #{number} is unavailable"
                        )
                    page = issue["closedByPullRequestsReferences"]
                    linked.extend(
                        (pr["number"], issue["url"], pr["createdAt"])
                        for pr in page["nodes"]
                        if pr["repository"]["nameWithOwner"].lower()
                        == self.repository.lower()
                    )
                    if not page["pageInfo"]["hasNextPage"]:
                        break
                    cursor = page["pageInfo"]["endCursor"]
                cache[number] = linked
            for pr, url, created in cache[number]:
                dates[pr] = created
                if (cutoffs or {}).get(number) and created < cutoffs[number]:
                    continue
                refs.setdefault(pr, {}).setdefault(number, []).append(url)
        return refs

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

    def discussion_references(self, pr):
        sources = [(pr.get("body") or "", pr["html_url"])]
        for path in (
            f"issues/{pr['number']}/comments",
            f"pulls/{pr['number']}/reviews",
            f"pulls/{pr['number']}/comments",
        ):
            for comment in self.api.pages(path):
                if comment.get("user", {}).get("type") == "Bot":
                    continue
                sources.append(
                    (comment.get("body") or "", comment.get("html_url", pr["html_url"]))
                )
        refs = {}
        for text, url in sources:
            for number in references(text, self.repository):
                refs.setdefault(number, []).append(url)
        return refs

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
        event = event or {}
        target = self.event_pr(event)
        requirements = set()
        catalog = self.release_catalog()
        versions = {
            s["milestone"]["number"]: (n, s)
            for n, s in catalog.items()
            if s.get("initialized")
        }
        for other in catalog.values():
            requirements.update(e["number"] for e in other["scope"].values())
        local_requirements = {e["number"] for e in state["scope"].values()}
        local_bugs = {int(n) for n in state["bugs"]}
        local_bugs.update(
            bug["number"] for bug in getattr(self, "_target_bugs", {}).get(release, [])
        )
        index = self.reference_index(event, release=release, state=state)
        progress = index.status()
        state["reference_index"] = (
            {"complete": True} if progress["complete"] else progress
        )
        native_issues = local_requirements | local_bugs
        if target:
            old = state["prs"].get(str(target), {})
            native_issues &= set(index.references(target)) | set(
                old.get("native_issues", [])
            )
        affected_issue = event.get("issue", {}).get("number")
        if not target and affected_issue in native_issues:
            native_issues = {affected_issue}
        native = self.native_references(
            native_issues,
            {number: self.release_created(state, release) for number in native_issues},
        )
        rows, problems = {}, []
        tracked = {
            int(n)
            for n, record in state["prs"].items()
            if record.get("requirements")
            or record.get("bugs")
            or record.get("projects")
        }
        candidates = (
            index.candidates(local_requirements | local_bugs) | tracked | set(native)
        )
        selected_prs = None
        if target:
            candidates.add(target)
            selected_prs = {target}
        elif affected_issue in local_requirements | local_bugs:
            selected_prs = (
                index.candidates({affected_issue})
                | set(native)
                | {
                    int(n)
                    for n, record in state["prs"].items()
                    if affected_issue
                    in record.get("requirements", []) + record.get("bugs", [])
                }
            )
        for pr_number in sorted(candidates):
            key = str(pr_number)
            if selected_prs is not None and pr_number not in selected_prs:
                old = state["prs"].get(key, {})
                pr = {
                    "number": pr_number,
                    "state": old.get("state", "open"),
                    "merged_at": old.get("merged"),
                    "draft": old.get("draft", False),
                }
                for number in state["prs"].get(key, {}).get("requirements", []):
                    if number in local_requirements:
                        rows.setdefault(number, []).append(pr)
                continue
            created = (
                getattr(self, "_pull_requests", {}).get(pr_number)
                or index.data["prs"].get(key, {}).get("pr")
                or state["prs"].get(key, {})
            ).get("created_at") or getattr(self, "_native_pr_dates", {}).get(pr_number)
            historical = bool(
                created and index.data.get("since") and created < index.data["since"]
            )
            if (
                not index.indexed(pr_number)
                and not index.can_refresh()
                and not historical
            ):
                index.queue(pr_number)
                old = state["prs"].get(key, {})
                for number in old.get("requirements", []):
                    if number in local_requirements:
                        rows.setdefault(number, []).append(
                            {
                                "number": pr_number,
                                "state": old.get("state", "open"),
                                "merged_at": old.get("merged"),
                                "draft": old.get("draft", False),
                            }
                        )
                continue
            pr = self.pull_request(pr_number)
            # Newly discovered Development links can bypass the historical
            # backfill cursor; refresh those PRs before applying their metadata.
            if not index.indexed(pr_number) and index.eligible(pr):
                pr = self.pull_request(pr_number, fresh=True)
                index.refresh(pr)
                index.flush()
            sources = {
                n: list(urls) for n, urls in index.references(pr["number"]).items()
            }
            for number, urls in native.get(pr["number"], {}).items():
                sources.setdefault(number, []).extend(urls)
            if (
                not (set(sources) & (local_requirements | local_bugs))
                and key not in state["prs"]
            ):
                continue
            linked, reqs, bugs = [], [], []
            for number in sources:
                try:
                    item = self.api.rest(f"issues/{number}")
                except OperationError as error:
                    if "404" in str(error):
                        continue
                    raise
                if "pull_request" in item:
                    continue
                floor = self.release_created(state, release)
                if number in requirements and milestone_number(item) in versions:
                    version, owner = versions[milestone_number(item)]
                    floor = max(floor, self.release_created(owner, version))
                if pr["created_at"] < floor:
                    continue
                is_bug = "bug" in {label["name"] for label in item["labels"]}
                if number in requirements or is_bug:
                    linked.append(item)
                    (bugs if is_bug else reqs).append(number)
            current = milestone_number(pr)
            record = state["prs"].setdefault(
                key, {"mode": "auto" if current is None else "manual", "last": current}
            )
            if "pending_milestone" in record:
                pending = record.pop("pending_milestone")
                if current == pending:
                    record["last"] = current
            record = manual_mode(record, current)
            if not pr.get("updated_at") or record.get("updated_at") != pr["updated_at"]:
                record = self.milestone_history(pr["number"], record)
            record["updated_at"] = pr.get("updated_at")
            record.update(
                bugs=bugs,
                requirements=reqs,
                merged=bool(pr.get("merged_at")),
                state=pr["state"],
                draft=pr.get("draft", False),
                sources={str(n): sources[n] for n in reqs + bugs},
                created_at=pr["created_at"],
                native_issues=sorted(
                    set(reqs + bugs) & native.get(pr["number"], {}).keys()
                ),
            )
            state["prs"][key] = record
            targets = {milestone_number(i) for i in linked}
            record["conflict"] = len(targets) > 1 and record["mode"] == "auto"
            if len(targets) > 1:
                if record["conflict"]:
                    problems.append(
                        f"PR #{key}: linked Issues have conflicting Milestones; set the PR Milestone manually"
                    )
            elif record["mode"] == "auto":
                desired = next(iter(targets), None)
                if current != desired:
                    record["pending_milestone"] = desired
                    self.save(release, state, f"Synchronizing PR #{key} Milestone")
                    updated = self.api.rest(
                        f"issues/{key}", "PATCH", {"milestone": desired}
                    )
                    pr["milestone"] = updated.get("milestone")
                    self._pull_requests[pr["number"]] = pr
                    if str(pr["number"]) in index.data["prs"]:
                        index.data["prs"][str(pr["number"])]["pr"]["milestone"] = pr[
                            "milestone"
                        ]
                    if updated.get("updated_at"):
                        record["updated_at"] = updated["updated_at"]
                        index.data["prs"][str(pr["number"])]["pr"]["updated_at"] = (
                            updated["updated_at"]
                        )
                    record.pop("pending_milestone", None)
                record["last"] = desired
            record["target_title"] = next(
                (
                    i["milestone"]["title"]
                    for i in linked
                    if i.get("milestone") and i["milestone"]["number"] == record["last"]
                ),
                (pr.get("milestone") or {}).get("title"),
            )
            projects = set()
            for item in linked:
                if item["number"] in reqs:
                    projects.update(self.memberships(item))
            if bugs:
                projects.add(self.config["projects"]["Bugfix"]["id"])
            self.sync_pr_projects(release, state, pr, record, projects)
            for number in reqs:
                if number in local_requirements:
                    rows.setdefault(number, []).append(pr)
        progress = index.status()
        state["reference_index"] = (
            {"complete": True} if progress["complete"] else progress
        )
        index.flush()
        source = self.issue(release)
        if (
            "### Requirements\n" in source["body"]
            or "### Requirements\r\n" in source["body"]
        ):
            # Validate before writeback; requirement edits stay pending until /update.
            try:
                parse_release(source["body"], self.repository)
            except OperationError as error:
                problems.append("Release body requires attention: " + str(error))
                return problems
            patched = patch_pull_requests(source["body"], rows, self.repository)
            latest = self.issue(release)
            if latest["body"] != source["body"]:
                problems.append(
                    "Release body changed during synchronization; retry /update"
                )
            elif patched != source["body"]:
                self.api.rest(f"issues/{release}", "PATCH", {"body": patched})
        return problems
