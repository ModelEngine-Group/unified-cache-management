"""Record, retain and clean UCM publications through one deletion lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm_release import cleanup_records as records
from ucm_release import version_config
from ucm_release.cleanup_inventory import ReleaseInventory, collect_catalog
from ucm_release.cleanup_records import (
    RETRY_DELAYS_SECONDS,
    ActiveRelease,
    CleanupError,
    CleanupReport,
    CleanupTarget,
    Resource,
    ResourceFailure,
)
from ucm_release.cleanup_remote import ProductionRemote, RemoteError


def execute_cleanup(
    target: CleanupTarget,
    remote: ProductionRemote,
    *,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> CleanupReport:
    """Delete resources in order, preserving evidence for failures or GHCR skips.

    All source formats must first identify concrete Actions and Release resources.
    Objects holding the full recovery inventory are deleted last. Publication
    status is read again before every phase; active or queued runs defer cleanup.
    Every DELETE requires a readback proving the exact ID/digest is absent.
    GHCR errors are nonfatal: other resources proceed, but recovery holders stay
    available for a later retry. Skipped resources are never reported as deleted.
    """
    remote.ensure_idle(target.run_ids)
    registry_failures = _run_phase(
        remote, target.registry, sleeper=sleeper, fail_resource=fail_resource
    )
    skipped = tuple(
        failure
        for failure in registry_failures
        if failure.resource.reference.startswith("ghcr.io/")
    )
    blocking = tuple(
        failure
        for failure in registry_failures
        if not failure.resource.reference.startswith("ghcr.io/")
    )
    if blocking:
        return CleanupReport(target.tag, False, 1, blocking, skipped)
    early_actions = tuple(
        action for action in target.actions if not action.holds_recovery_data
    )
    stages = (
        early_actions,
        (Resource("git-tag", target.tag, target.tag),),
    )
    for phase, resources in enumerate(stages, start=2):
        remote.ensure_idle(target.run_ids)
        failures = _run_phase(
            remote, resources, sleeper=sleeper, fail_resource=fail_resource
        )
        if failures:
            return CleanupReport(target.tag, False, phase, tuple(failures), skipped)
    remote.ensure_idle(target.run_ids)
    unbacked = tuple(
        release for release in target.releases if not release.holds_recovery_data
    )
    failures = _run_phase(
        remote, unbacked, sleeper=sleeper, fail_resource=fail_resource
    )
    if failures:
        return CleanupReport(target.tag, False, 4, tuple(failures), skipped)
    if skipped:
        return CleanupReport(target.tag, False, None, (), skipped)
    holders = tuple(
        release for release in target.releases if release.holds_recovery_data
    )
    holders += tuple(action for action in target.actions if action.holds_recovery_data)
    failures = _run_phase(remote, holders, sleeper=sleeper, fail_resource=fail_resource)
    return CleanupReport(
        target.tag, not failures, 4 if failures else None, tuple(failures)
    )


def resource_failure(
    resource: Resource, attempts: int, error: BaseException
) -> ResourceFailure:
    return ResourceFailure(resource, attempts, str(error) or type(error).__name__)


def delete_resource_with_retry(
    remote: ProductionRemote,
    resource: Resource,
    *,
    sleeper=time.sleep,
    fail_resource: str | None = None,
) -> ResourceFailure | None:
    """Probe and delete a resource, confirming Release deletion by ID readback."""
    locked_state: object | None = None
    dockerhub = resource.kind in {"dockerhub-index", "dockerhub-member"}
    for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, start=1):
        print(
            f"cleanup resource_type={resource.kind} reference={resource.reference} "
            f"attempt={attempt}/{len(RETRY_DELAYS_SECONDS)} delay={int(delay)}s",
            flush=True,
        )
        if delay:
            sleeper(delay)
        try:
            state = locked_state if locked_state is not None else remote.probe(resource)
            if state is None:
                return None
            # A moved Tag must never redirect a retry to another manifest.
            if dockerhub:
                locked_state = state
            if fail_resource is not None and resource.reference == fail_resource:
                raise RemoteError(
                    f"synthetic HTTP 503 for {resource.reference}", status=503
                )
            try:
                remote.delete(resource, state)
            except RemoteError as error:
                if not error.is_missing:
                    raise
            if not remote.is_absent(resource, state):
                raise RemoteError(
                    f"{resource.kind} still exists after deletion", status=409
                )
            return None
        except RemoteError as error:
            if error.is_missing:
                return None
            if error.is_retryable and attempt < len(RETRY_DELAYS_SECONDS):
                continue
            return resource_failure(resource, attempt, error)
        except CleanupError as error:
            return resource_failure(resource, attempt, error)
    raise AssertionError("resource retry loop exhausted without a result")


def _run_phase(
    remote: ProductionRemote,
    resources: Sequence[Resource],
    *,
    sleeper,
    fail_resource: str | None,
) -> list[ResourceFailure]:
    failures: list[ResourceFailure] = []
    for resource in resources:
        failure = delete_resource_with_retry(
            remote,
            resource,
            sleeper=sleeper,
            fail_resource=fail_resource,
        )
        if failure is not None:
            failures.append(failure)
    return failures


def render_resource_summary(issues: Sequence[ResourceFailure], title: str) -> str:
    """Render failures or nonfatal GHCR skips for the GitHub job summary."""
    if not issues:
        return ""

    def cell(value: object) -> str:
        return " ".join(str(value).split()).replace("|", "\\|")

    lines = [
        f"## UCM release cleanup {title}",
        "",
        "| Resource type | Reference | Final error |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {cell(item.resource.kind)} | {cell(item.resource.reference)} | "
        f"{cell(item.final_error)} |"
        for item in issues
    )
    return "\n".join(lines) + "\n"


def append_resource_summary(
    path: Path | None, issues: Sequence[ResourceFailure], title: str
) -> None:
    summary = render_resource_summary(issues, title)
    if path is not None and summary:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(summary)


def _boolean(value: str) -> bool:
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)


def _add_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--crane", default="crane")
    parser.add_argument(
        "--api-base", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--fail-resource")
    parser.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    parser.add_argument(
        "--dry-run", action="store_true", help="read-only cleanup preview"
    )
    parser.add_argument("--inventory", type=Path, help="reviewed historical inventory")
    parser.add_argument("--report", type=Path, help="write per-version JSON results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    local = commands.add_parser("record", help="save one local publication attempt")
    _add_identity_arguments(local)
    local.add_argument("--release-id", type=int)
    local.add_argument("--plan", type=Path)
    local.add_argument("--output-dir", type=Path, default=Path("out/cleanup"))

    targets = commands.add_parser(
        "record-targets", help="save and verify one publication attempt on its Release"
    )
    targets.add_argument("--tag", required=True)
    targets.add_argument(
        "--release-type", choices=sorted(records.RELEASE_TYPES), required=True
    )
    targets.add_argument("--run-id", type=int, required=True)
    targets.add_argument("--source-sha", required=True)
    targets.add_argument("--run-attempt", type=int)
    targets.add_argument("--release-id", type=int)
    targets.add_argument("--plan", type=Path)
    targets.add_argument("--output-dir", type=Path, default=Path("out/cleanup"))
    _add_remote_arguments(targets)

    validate = commands.add_parser(
        "validate-record", help="verify a local record against a publication event"
    )
    _add_identity_arguments(validate)
    validate.add_argument("--input", type=Path, required=True)

    tag = commands.add_parser("tag", help="clean one exact Tag")
    tag.add_argument("--tag", required=True)
    _add_remote_arguments(tag)

    retention = commands.add_parser(
        "retention", help="clean oldest excess same-type Tags"
    )
    retention.add_argument("--current-tag", help="required for non-Nightly retention")
    retention.add_argument(
        "--release-type", choices=sorted(records.RELEASE_TYPES), required=True
    )
    retention.add_argument("--max-count", type=int, required=True)
    retention.add_argument("--pypi-enabled", type=_boolean, required=True)
    _add_remote_arguments(retention)
    return parser


def _production_remote(arguments: argparse.Namespace) -> ProductionRemote:
    repository = arguments.repository
    if not isinstance(repository, str) or not repository:
        raise CleanupError("--repository or GITHUB_REPOSITORY is required")
    return ProductionRemote(
        repository,
        os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "",
        crane=arguments.crane,
        api_base=arguments.api_base,
    )


def _read_document(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanupError(f"cannot read cleanup input {path}: {error}") from error


def _write_document(path: Path | None, value: object) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_plan(arguments: argparse.Namespace) -> dict[str, Any] | None:
    plan = _read_document(arguments.plan) if arguments.plan is not None else None
    if plan is not None and not isinstance(plan, dict):
        raise CleanupError("cleanup publication plan must be an object")
    return plan


def _run_record(arguments: argparse.Namespace) -> None:
    plan = _read_plan(arguments)
    snapshot = records.new_record(
        arguments.repository,
        arguments.tag,
        arguments.source_sha,
        arguments.run_id,
        arguments.run_attempt,
        release_id=arguments.release_id,
        plan=plan,
    )
    filename = records.record_basename(
        snapshot, arguments.run_attempt, planned=plan is not None
    )
    output = arguments.output_dir / filename
    _write_document(output, snapshot)
    print(f"record saved: {output}")


def _run_validate_record(arguments: argparse.Namespace) -> None:
    record = records.decode_record(
        _read_document(arguments.input),
        arguments.repository,
        asset_name=arguments.input.name,
    )
    expected = (arguments.run_id, arguments.run_attempt)
    bindings = {
        (item["id"], item["attempt"]): item["source_sha"]
        for item in record.get("attempt_sources", [])
    }
    source_sha = bindings.get(expected, record["source_sha"])
    if (
        record["tag"] != arguments.tag
        or source_sha != arguments.source_sha
        or expected not in {(run["id"], run["attempt"]) for run in record["runs"]}
    ):
        raise CleanupError("cleanup record does not match the publication event")
    print(f"record validated: {arguments.input}")


def _run_record_targets(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> None:
    coordinates = version_config.classify_tag(arguments.tag)
    if coordinates["release_type"] != arguments.release_type:
        raise CleanupError("record release type does not match the Tag")
    plan = _read_plan(arguments)
    if arguments.dry_run:
        records.new_record(
            remote.repository,
            arguments.tag,
            arguments.source_sha,
            arguments.run_id,
            arguments.run_attempt or 1,
            release_id=arguments.release_id,
            plan=plan,
        )
        print(f"would-record: {arguments.tag}")
        return
    ReleaseInventory(remote).save_record(
        arguments.tag,
        arguments.run_id,
        arguments.source_sha,
        plan,
        run_attempt=arguments.run_attempt,
        release_id=arguments.release_id,
        output_dir=arguments.output_dir,
    )


def _catalog(
    arguments: argparse.Namespace, remote: ProductionRemote, release_type: str
):
    path = getattr(arguments, "inventory", None)
    history = _read_document(path) if path is not None else None
    return collect_catalog(remote, history, release_type=release_type)


def _record_result(
    record: dict[str, Any],
    status: str,
    reason: str | None = None,
    *,
    target: CleanupTarget | None = None,
):
    result = {
        "tag": record["tag"],
        "status": status,
        "release_ids": record["release_ids"],
        "run_ids": sorted({run["id"] for run in record["runs"]}),
        "registry_references": len(record["resources"]),
    }
    if reason is not None:
        result["reason"] = reason
    if target is not None:
        result.update(
            release_ids=[release.identifier for release in target.releases],
            run_ids=list(target.run_ids),
            registry_references=len(target.registry),
        )
    return result


def _resource_issues(issues: Sequence[ResourceFailure]) -> list[dict[str, Any]]:
    return [
        {
            "kind": issue.resource.kind,
            "reference": issue.resource.reference,
            "attempts": issue.attempts,
            "error": issue.final_error,
        }
        for issue in issues
    ]


def _run_targets(
    arguments: argparse.Namespace,
    remote: ProductionRemote,
    targets: Sequence[dict[str, Any]],
    *,
    blocked: dict[str, str],
    kept: Sequence[dict[str, Any]] = (),
    deferred: Sequence[dict[str, Any]] = (),
) -> list[ResourceFailure]:
    """Resolve, preserve and delete selected versions through one lifecycle."""
    inventory = ReleaseInventory(remote)
    failures: list[ResourceFailure] = []
    skipped: list[ResourceFailure] = []
    results = [_record_result(item, "kept", blocked.get(item["tag"])) for item in kept]
    for record in deferred:
        reason = blocked.get(record["tag"])
        if reason is not None:
            failures.append(
                resource_failure(
                    Resource("release-version", record["tag"]), 1, CleanupError(reason)
                )
            )
        results.append(
            _record_result(
                record,
                "blocked" if reason else "deferred",
                reason or "publication run is active or queued",
            )
        )
    dry_run = getattr(arguments, "dry_run", False)
    for record in targets:
        target = None
        try:
            reason = blocked.get(record["tag"])
            if reason is not None:
                raise CleanupError(reason)
            resolution = inventory.resolve_record(record)
            target = resolution.target
            if dry_run:
                preview_skips = []
                for resource in target.registry:
                    try:
                        remote.probe(resource)
                    except CleanupError as error:
                        if not resource.reference.startswith("ghcr.io/"):
                            raise
                        preview_skips.append(resource_failure(resource, 1, error))
                result = _record_result(record, "would-delete", target=target)
                if preview_skips:
                    result["skipped"] = _resource_issues(preview_skips)
                    skipped.extend(preview_skips)
                results.append(result)
                continue
            inventory.persist_recovery(resolution)
            report = execute_cleanup(
                resolution.target, remote, fail_resource=arguments.fail_resource
            )
        except ActiveRelease as error:
            results.append(
                _record_result(record, "deferred", str(error), target=target)
            )
            continue
        except CleanupError as error:
            failures.append(
                resource_failure(Resource("release-version", record["tag"]), 1, error)
            )
            results.append(_record_result(record, "blocked", str(error), target=target))
            continue
        failures.extend(report.failures)
        skipped.extend(report.skipped)
        result = _record_result(
            record,
            (
                "blocked"
                if report.failures
                else "partial" if report.skipped else "deleted"
            ),
            target=target,
        )
        if report.failures:
            result["failures"] = _resource_issues(report.failures)
        if report.skipped:
            result["skipped"] = _resource_issues(report.skipped)
        results.append(result)
    deferred_versions = any(result["status"] == "deferred" for result in results)
    if failures:
        status = "failed"
    elif dry_run:
        status = "dry-run"
    elif skipped:
        status = "partial"
    elif deferred_versions:
        status = "deferred"
    else:
        status = "complete"
    _write_document(
        getattr(arguments, "report", None),
        {
            "repository": remote.repository,
            "max_count": getattr(arguments, "max_count", None),
            "status": status,
            "results": results,
        },
    )
    summary = getattr(arguments, "summary", None)
    append_resource_summary(
        Path(summary) if summary else None, skipped, "skipped GHCR resources"
    )
    for issue in skipped:
        print(
            f"cleanup skipped resource_type={issue.resource.kind} "
            f"reference={issue.resource.reference}: {issue.final_error}",
            file=sys.stderr,
        )
    for result in results:
        print(
            f"cleanup {result['status']}: {result['tag']}"
            + (f" - {result['reason']}" if "reason" in result else "")
        )
    return failures


def _run_tag(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    release_type = str(version_config.classify_tag(arguments.tag)["release_type"])
    catalog = _catalog(arguments, remote, release_type)
    targets = [
        record
        for record in (*catalog.records, *catalog.orphans)
        if record["tag"] == arguments.tag
    ]
    if not targets:
        raise CleanupError(
            catalog.blocked.get(
                arguments.tag,
                "cannot resolve a proven cleanup target for the exact Tag",
            )
        )
    active = arguments.tag in catalog.active_tags
    return _run_targets(
        arguments,
        remote,
        [] if active else targets,
        blocked=catalog.blocked,
        deferred=targets if active else (),
    )


def _run_retention(
    arguments: argparse.Namespace, remote: ProductionRemote
) -> list[ResourceFailure]:
    options = {
        "release_type": arguments.release_type,
        "current_tag": arguments.current_tag,
        "pypi_enabled": arguments.pypi_enabled,
    }
    selection = records.select_retention([], arguments.max_count, **options)
    if selection.skipped_reason is not None:
        print(selection.skipped_reason)
        _write_document(
            getattr(arguments, "report", None), {"status": "skipped", "results": []}
        )
        return []
    catalog = _catalog(arguments, remote, arguments.release_type)
    selection = records.select_retention(
        catalog.records, arguments.max_count, catalog.active_tags, **options
    )
    if selection.skipped_reason is not None:
        print(selection.skipped_reason)
        _write_document(
            getattr(arguments, "report", None), {"status": "skipped", "results": []}
        )
        return []
    orphan_candidates = [
        record for record in catalog.orphans if record["tag"] not in catalog.active_tags
    ]
    orphan_deferred = [
        record for record in catalog.orphans if record["tag"] in catalog.active_tags
    ]
    return _run_targets(
        arguments,
        remote,
        (*selection.candidates, *orphan_candidates),
        blocked=catalog.blocked,
        kept=selection.kept,
        deferred=(*selection.deferred, *orphan_deferred),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "record":
            _run_record(arguments)
            return 0
        if arguments.command == "validate-record":
            _run_validate_record(arguments)
            return 0
        remote = _production_remote(arguments)
        if arguments.command == "record-targets":
            _run_record_targets(arguments, remote)
            return 0
        failures = (
            _run_tag(arguments, remote)
            if arguments.command == "tag"
            else _run_retention(arguments, remote)
        )
        summary = Path(arguments.summary) if arguments.summary else None
        append_resource_summary(summary, failures, "final failures")
        if failures:
            for failure in failures:
                print(
                    f"{failure.resource.kind} {failure.resource.reference}: {failure.final_error}",
                    file=sys.stderr,
                )
            return 1
        return 0
    except (CleanupError, ValueError, OSError, json.JSONDecodeError) as error:
        _write_document(
            getattr(arguments, "report", None),
            {"status": "failed", "error": str(error)},
        )
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
