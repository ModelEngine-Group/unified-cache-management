"""Repository Issue pinning using the repository credential."""

from syntax import OperationError


class ReleasePinning:
    def release_pin(self, release, state):
        record = state.setdefault("release_pin", {})
        owner, name = self.repository.split("/")
        try:
            if self.issue(release)["state"] == "closed":
                return
            snapshot = self.api.call(
                "graphql",
                "POST",
                {
                    "query": """query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){issue(number:$number){id isPinned} pinnedIssues(first:3){nodes{issue{number} pinnedBy{__typename login}}}}}""",
                    "variables": {"owner": owner, "name": name, "number": release},
                },
            )["data"]["repository"]
            issue = snapshot["issue"]
            existing = next(
                (
                    p
                    for p in snapshot["pinnedIssues"]["nodes"]
                    if p["issue"]["number"] == release
                ),
                None,
            )
            if issue["isPinned"]:
                if not record or (
                    record.get("status") == "pinned"
                    and existing
                    and existing["pinnedBy"]["__typename"] == "User"
                ):
                    record.update(mode="manual", status="pinned")
            elif record.get("status") == "pinned":
                record.update(mode="manual", status="unpinned")
            if record.get("mode") == "manual":
                return
            if issue["isPinned"]:
                record["status"] = "pinned"
                state.pop("pin_warning", None)
                return
            if len(snapshot["pinnedIssues"]["nodes"]) >= 3:
                raise OperationError(
                    "All three repository pin slots are occupied; Release pinning is pending"
                )
            record.update(mode="auto", status="pending_pin")
            self.save(release, state, "Updating Release Issue pin")
            self.api.call(
                "graphql",
                "POST",
                {
                    "query": "mutation($id:ID!){pinIssue(input:{issueId:$id}){issue{id isPinned}}}",
                    "variables": {"id": issue["id"]},
                },
            )
            record["status"] = "pinned"
            state.pop("pin_warning", None)
        except OperationError as error:
            state["pin_warning"] = str(error)
