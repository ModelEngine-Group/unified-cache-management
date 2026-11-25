"""Pin a newly configured Release Checklist."""

from syntax import OperationError


class ReleasePinning:
    def release_pin(self, release):
        owner, name = self.repository.split("/")
        try:
            if self.issue(release)["state"] == "closed":
                return
            snapshot = self.api.call(
                "graphql",
                "POST",
                {
                    "query": """query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){issue(number:$number){id isPinned} pinnedIssues(first:3){nodes{issue{number}}}}}""",
                    "variables": {"owner": owner, "name": name, "number": release},
                },
            )["data"]["repository"]
            issue = snapshot["issue"]
            if issue["isPinned"]:
                return
            if len(snapshot["pinnedIssues"]["nodes"]) >= 3:
                raise OperationError("All three repository pin slots are occupied")
            self.api.call(
                "graphql",
                "POST",
                {
                    "query": "mutation($id:ID!){pinIssue(input:{issueId:$id}){issue{id isPinned}}}",
                    "variables": {"id": issue["id"]},
                },
            )
        except OperationError as error:
            print(f"Could not pin Release Checklist #{release}: {error}")
