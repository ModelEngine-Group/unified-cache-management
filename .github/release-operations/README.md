# Release operations

Create a version using **Release Checklist**. Its structured Markdown requirements
are the release scope. The controller manages a Milestone, flat requirement Issues,
a native Bug Archive under the Release Issue, and PR associations. The complete
body is available in [the example](examples/release-checklist.md).

See the complete [UCM version-management operating guide](../../docs/version-management.md)
for roles, every command, examples, and official-repository deployment steps.

## Input and commands

All template headings and field names use English and ASCII punctuation.
Each requirement starts with a task-list title. The Issue reference stays outside
the collapsible Details so GitHub can show its native title/state presentation.
Title continuation lines use two spaces. Inside Details, fields use two spaces,
and description/PR content uses at least four spaces. Keep assignee, description,
and pull_requests in that order. Titles and descriptions support multiple lines.
The Requirements input starts with a hidden copyable template comment. Copy its
generic requirement block outside the comment and fill it in; the commented
example is excluded from parsing and writeback.

```markdown
- [ ] Connector v2
  initial integration

  <details>
  <summary>Details</summary>

  assignee: @username

  description:
    Implement the connector.

    Acceptance criteria:
    - Pass the integration test.

  pull_requests:

  </details>
```

An existing Issue can also be referenced directly as `- [ ] #123`.
The checkbox is presentation data and is excluded from scope comparisons.
Existing flat lists and summary-title cards remain readable. Conversion preserves
Issue identities and scope revisions, and removes the old generated status badges.
Bot writeback updates the outer Issue reference and the PR field inside Details.

A title without an Issue reference creates a new Issue. Legacy empty placeholders
remain accepted. An existing requirement ends in a native `#123`
reference. Markdown links, bare URLs, and older bracket formats remain accepted
input; writeback normalizes them to a title plus `#123` so GitHub can display
native Issue references. The whole title may be pasted as a Markdown link.
Issue IDs identify synchronized scope. Creation-origin markers recover accepted requests after lost
responses. Requirement parent relationships are managed by developers in GitHub.
The controller retains historical snapshots and existing parent relationships.

| Command | Result |
| --- | --- |
| `/release_init --dry-run` | Validate input and preview object creation/reuse. |
| `/release_init` | Create/reuse Milestone, requirements, and Bug Archive; write Issue links; replace the preview. |
| `/update --dry-run` | Preview differences from the last synchronized scope. |
| `/update` | Synchronize requirement additions/removals, assignees, metadata, and PR associations. |
| `/release_check` | Inspect pending scope edits, feature PR coverage, open requirements/Bugs, unmerged PRs, version conflicts, and inclusion in the release branch. |

Commands require repository write, maintain, or admin permission. Each command
is a separate comment on the Release Issue. Update `reason:` in Change Log for
scope removals. Removal retains the Issue and clears its current release
Milestone when that value still belongs to this version. Existing Issue titles
and descriptions are maintained in the Issue; the release entry stores the
version's plan. Assignee edits, including clearing an assigned owner, synchronize.
Description whitespace at the end of a block is ignored in scope comparisons;
the original Markdown remains intact.

Release Information contains `version`, `branch`, `release_manager`,
`test_lead`, and `release_date`. Roles use one @username or an empty value. The
release manager is assigned to the Release Issue. The roles are recorded
in the Milestone description; `release_date` accepts YYYY-MM-DD or TBD and controls
the Milestone due date. Release checks use the configured branch.

## Release navigation and overview

The Bot maintains two bounded regions in the Issue body. The navigation region
sits directly under Release Information and links the Milestone and Bug Archive.
The Release Overview region is always at the bottom, with one table for requirements
and actual Bugs: Issue, Pull requests, Assignee, State. Issue and PR
cells use native #number references. Lists can expand native references into title
and state cards; the table also retains explicit Issue and PR states.

Region markers start/end with `ucm-release-navigation` and `ucm-release-overview`.
Generated content is excluded from scope parsing and revision comparison. Human
fields, descriptions, custom sections, and checklist edits outside those regions
are retained. Repeated refreshes replace the same regions.

The bottom Bot region contains one latest operation line: a running arrow and
workflow link during execution, a green check on success, or a red cross with a
short reason on failure. Initialization, synchronization, and release checks
use the same region. Nested steps preserve their outer operation.
Preview plans remain in a collapsible panel until formal execution replaces them.

The trusted checkpoint comment contains only machine data and is automatically
collapsed. Existing notification text is removed in place, retaining object maps
and command cursors. Results, links, tables, release checks, and scope
history are displayed in the body. The workflow finalizer marks interrupted
operations from its own run as failed without overwriting another run's status.

## PR and Project synchronization

Formal initialization automatically pins the Release Issue above the repository
Issues list. The Bug Archive remains accessible from the release receipt and
native sub-issue relationship. Dry-runs leave pins untouched. Repeated initialization
or updates retry a pending pin; occupied slots are reported without removing other
Issues. Manual pins/unpins are retained. Closed Release Issues are skipped by automatic pinning.

The Bot adds an `eyes` reaction to an authorized version-command comment when the
controller starts processing it. PR conversation comments with local Issue
references receive the same acknowledgement before reconciliation, including
edits that remove a previous reference. Repeated requests reuse the Bot's reaction.
The reaction means processing has started; execution results remain in the release
receipt. Deleted comments, automated Bot comments, and unrelated conversation
messages receive no acknowledgement. Reaction API failures are logged while the
operation continues. A queued workflow acknowledges when its controller starts.

Reference a local Issue in PR descriptions, conversation comments, review bodies,
or inline review comments, using `#123`, `OWNER/REPO#123`, or a complete Issue URL.
PR events refresh their target's sources; periodic discovery updates a persistent
reference index in bounded batches. Reconciliation deduplicates indexed references
and current Development links. References in automated Bot comments are excluded from inputs.
Execution receipts retain source permalinks.

GitHub's native Development sidebar links are an additional association source.
The controller reads `closedByPullRequestsReferences` using the repository token,
includes manually linked and closed/merged PRs, and combines them with textual
references. Removing a sidebar link preserves any remaining textual relationship;
removing the last source cleans up automatic metadata and the table row. Native
links have the same Milestone/Project ownership and conflict rules as mentions.
Queries are paginated, restricted to this repository, and cached within a run.
A failed native query aborts association reconciliation before cleanup.

Sidebar link/unlink changes are picked up by the 15-minute schedule, `/update`,
`/release_check`, or manual workflow dispatch. PR lifecycle events refresh native links as well.
GitHub Actions does not expose the timeline connected/disconnected activities as
direct workflow triggers, so sidebar edits use these reconciliation entry points.

- A scoped requirement supplies its Milestone and current native Projects.
- A Bug supplies its target fix Milestone and the configured Bugfix Project.
- Multiple requirements supply the union of their Projects.
- Conflicting Milestones require an explicit maintainer choice and appear as pending items
  while automatic ownership is unresolved.
- Maintainer Milestone changes/clears persist. Existing manual Project memberships
  remain; removing a managed membership manually suppresses its automatic re-add.
- Removing the last reference removes automatic associations and the requirement's
  generated PR row. A remaining mention in any source retains the relationship.

Choose requirement Projects in the Issue sidebar. The controller queries native
Project membership and synchronizes associated PRs. `pull_requests:` is generated
inside each release requirement, with native PR references and draft/open/closed/merged status. Changes to
this field alone refresh display without creating scope revisions.

PR creation, body edits, state changes, and ordinary comment creation/edit/deletion
trigger Release operations. Review events trigger a credential-free **Release
review observer**; its completion wakes the trusted default-branch controller.
The observer checks out no code. It supplies a data-only PR-number artifact for
fork events whose workflow-run payload omits the PR. The receiver validates the
source workflow, reads the numeric target, and refreshes that PR. Reviews without
Issue references and automated PR conversation comments skip this path.
`/release_init`, `/update`, manual dispatch, and the 15-minute schedule also
reconcile existing references and subsequent Issue Milestone/Project changes.

## Bugs and release checks

Initialization creates a Bug Archive as a native sub-issue of the Release Issue,
with the source version's Milestone. Use the existing **Bug report** form for
ordinary and version test Bugs. Create version Bugs under the archive, or attach
an existing Bug as its sub-issue. The native parent relationship supplies the
source version. Bugs default to the
source version's Milestone; maintainer retargets/clears persist. Their parent
continues to identify the source version. Legacy Bug source fields remain readable.
Bug and fix PRs enter Bugfix. The Bug status follows implementation, merged fixes,
and maintainer closure. Filter parent Issue for source-version defects and Milestone
for target-version fixes.

Maintainers review implementation and test results, then close completed feature
Issues and resolved Bugs in GitHub. Reopening an Issue brings it back into the
pending list. Test reports and discussion remain in the corresponding Issues.

`/release_check` refreshes existing associations and writes a readable report at
the bottom of the Release Issue:

- Feature Issues without associated PRs, including manually completed Issues.
- Open feature Issues and open Bugs whose target Milestone is this version,
  including Bugs discovered in another version.
- Open PRs awaiting merge, including PRs assigned directly to the Milestone.
- Closed PRs without a merge and related PRs assigned to another version, for review.
- Unresolved PR target-version conflicts and merged commits missing from the
  configured release branch.
- Requirements added, removed, or edited since the last `/update`, plus changed
  Release Information fields.

For example, adding a Store v2 requirement card to the body creates a pending plan edit.
`/update` creates its Issue and applies the Milestone/assignee. Removing a
requirement or editing its planned owner also requires `/update` to apply that
change. `/release_check` reports these differences and preserves the synchronized
scope. A routine Issue/PR state refresh leaves the scope revision unchanged.

Open requirements, open target Bugs, open target PRs, version conflicts, missing
merge commits, and pending plan edits are reported as blocking items. Missing feature PRs and
closed unmerged PRs are review notes; maintainer closure records completion.
The report is a snapshot of the last requested check. Run `/release_check` again
for current results.

Release Check supplies a status report for maintainers and the repository's existing
release publishing process.

## Tag publication and manual reruns

The existing Tag workflow gates `v*` publication on a fresh version check against
the Tag's frozen source commit. `draft/v*` and `nightly/v*` retain their existing
routes. The Tag classifier's canonical version identifies exactly one initialized
Release Issue. Both stable and `vX.Y.ZrcN` Tags use this bridge.

The publisher directly writes Checking/Publishing and its final outcome into the
Release Issue's Release Delivery section. Each Tag has one record keyed by the
source run ID and attempt, with an attempt-specific workflow link. The delivery
status workflow supplements partial rerun starts and cancellation using trusted
publisher lifecycle events. Older events cannot replace a newer attempt, and
ordinary reconciliation only renders the stored delivery status. Rerun-all checks
the scope again; rerunning failed jobs reuses successful prerequisite jobs.

Completion requires a successful publication and a matching `acceptance.json`
from `verify-release-delivery`. Acceptance artifacts include the attempt in their
name and contents. A partial rerun may reuse a successful earlier acceptance from
the same run, Tag and source commit. The bridge adds links to the Release Issue,
Milestone and Bug Archive in a separate GitHub Release region, closes the
Milestone, and records the Release URL and completion time. Release and archive
Issues remain available for later defects. Failure/cancellation records the
outcome and retains the Milestone's current state.

## Deployment and interfaces

The first initialized release creates a closed internal **Release reference
index** Issue, labelled `release-index`. Trusted Bot comments hold compressed
shards and a manifest. A shard retains its committed generation until the
manifest switches, so interrupted writes leave the previous cursor readable.
The label is created automatically when needed.

Each release has an independent discovery window beginning at its Release
Tracking Issue's `created_at`. The initial sweep lists PRs in descending
creation order and stops at the first older PR. Each requirement association
requires `PR.created_at >= Release.created_at`, including references from
comments/reviews and native Development links. A requirement created after an
eligible PR can be linked to that PR. Requirement creation dates do not filter
associations. Later edits to an older PR retain its original creation time.

Subsequent polling refreshes updated PRs inside the selected window. Requirement
additions/removals reuse the fixed release boundary and shared reference shards.
Release readiness waits for its own window and includes target-fix Bug associations.
Explicit Milestone PRs remain delivery inputs regardless of their creation time;
Milestone assignment supplies version ownership, while requirement rows and
Project inheritance follow requirement associations.
The default budget is 10 PR discussions per run, shared by backfill, changed PRs
and newly discovered native links. Older global-backfill cursors are replaced by
release windows; out-of-window history does not delay readiness.
Manual dispatch accepts `reference_batch_size` of 10, 25 or 50. Configure regular
batches with `sync.prs_per_run` and the request ceiling with `sync.max_api_calls`
(default 200). Each run also has a ten-minute controller budget. Completed batches
persist before further reconciliation; retries resume their saved cursor.

Events select affected releases. Periodic runs reconcile active versions and
rotate one archived version, sharing per-run state and Project snapshots.
Unchanged metadata leaves the release checkpoint, body and Bug Project status
untouched. State checkpoints use compressed v2 payloads and retain v1 read support.
Controller tests run in the PR-only **Release operations contracts** workflow.

The checked-in config targets the official repository and its nine organization
Projects, with verified Project, view, and Status IDs. The Projects use the
organization's default private visibility. Configure the Project credential and
repository labels as described in the operating guide before enabling the workflow.

Deploy this directory and its workflows to the default branch. Configure
`config.json.repository`, Project/field IDs, and the `UCM_PROJECT_TOKEN` Actions
secret. Repository metadata uses `GITHUB_TOKEN`; Project API calls use the Project
credential. The workflow runs trusted default-branch code and reads PR content
as data. No PR code build is required by this metadata workflow.

The Tag gate uses repository Issue/PR credentials plus the Project credential.
The metadata receiver uses Actions read for the observer's data-only target.
Delivery reporting uses Actions read, Issues write and Contents write to inspect
the publisher's receipt and maintain GitHub Release links. Gate/report jobs share
the metadata controller's short-write concurrency queue; long builds retain their
existing release concurrency group.

`Controller.initialize_or_update`, `reconcile`, and `release_check`
are Python interfaces. The CLI accepts `--config`, `--event`, and `--event-name`.
Mutations go through `GitHub.rest`/`graphql`. The parser and PR association logic
are separate modules using only the Python standard library.

One Bot result comment stores scope, object IDs, ownership, release-check results, revision
history, and processed command IDs. Formal execution replaces previews. Workflow
concurrency serializes writers. Later events replay pending release commands in
order; a persistent cursor prevents completed history from replaying. Periodic
reconciliation supplements event delivery.

Run contracts with:

```shell
python -m unittest discover -s .github/release-operations -p 'test_*.py' -v
```
