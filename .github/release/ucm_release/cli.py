from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from . import (
    builders,
    chart,
    meta,
    plan,
    policy,
    pr,
    problems,
    pypi,
    registry,
    runtime,
    serialization,
    toolkit,
    upstream,
    wheel,
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release", type=Path, default=policy.DEFAULT_RELEASE)


def _write(path: Path, value: object) -> None:
    path.write_bytes(serialization.canonical_bytes(value) + b"\n")


def _publication_context(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    value = serialization.load_json(path)
    if not isinstance(value, dict) or set(value) != {
        "fork_test_pypi",
        "dockerhub_namespace",
    }:
        raise ValueError("publication context fields must be exact")
    if not isinstance(value["fork_test_pypi"], bool):
        raise ValueError("publication context TestPyPI flag must be boolean")
    namespace = value["dockerhub_namespace"]
    if namespace is not None and (not isinstance(namespace, str) or not namespace):
        raise ValueError("publication context Docker Hub namespace is invalid")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ucm_release")
    groups = parser.add_subparsers(dest="group", required=True)

    chart_parser = groups.add_parser("chart")
    chart_actions = chart_parser.add_subparsers(dest="action", required=True)
    chart_prepare = chart_actions.add_parser("prepare")
    chart_prepare.add_argument("--plan", type=Path, required=True)
    chart_prepare.add_argument("--output", type=Path, required=True)
    chart_prepare.set_defaults(
        func=lambda a: chart.prepare_chart(serialization.load_json(a.plan), a.output)
    )

    builders_parser = groups.add_parser("builders")
    builders_actions = builders_parser.add_subparsers(dest="action", required=True)

    builders_discover = builders_actions.add_parser("discover")
    builders_discover.add_argument(
        "--config", type=Path, default=policy.DEFAULT_PLATFORMS
    )
    builders_discover.add_argument("--owner")
    builders_discover.add_argument("--selection", type=Path, required=True)
    builders_discover.add_argument("--output", type=Path, required=True)

    def _cmd_builders_discover(a):
        formal = policy.resolve(platforms_path=a.config)
        result = builders.catalog_from_builds(
            upstream.validate_selection(serialization.load_json(a.selection))[
                "wheel_builds"
            ],
            a.config,
            owner=a.owner,
            formal_policy=formal,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_discover.set_defaults(func=_cmd_builders_discover)

    builders_sync_plan = builders_actions.add_parser("sync-plan")
    builders_sync_plan.add_argument("--catalog", type=Path, required=True)
    builders_sync_plan.add_argument("--existing", type=Path, required=True)
    builders_sync_plan.add_argument("--output", type=Path, required=True)

    def _cmd_builders_sync_plan(a):
        result = builders.compute_sync_plan(
            serialization.load_json(a.catalog), serialization.load_json(a.existing)
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_sync_plan.set_defaults(func=_cmd_builders_sync_plan)

    builders_labels = builders_actions.add_parser("labels")
    builders_labels.add_argument("--builder", type=Path, required=True)
    builders_labels.add_argument("--output", type=Path, required=True)

    def _cmd_builders_labels(a):
        result = builders.builder_labels(serialization.load_json(a.builder))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_labels.set_defaults(func=_cmd_builders_labels)

    builders_finalize = builders_actions.add_parser("finalize")
    builders_finalize.add_argument("--catalog", type=Path, required=True)
    builders_finalize.add_argument("--observations", type=Path, required=True)
    builders_finalize.add_argument("--output", type=Path, required=True)

    def _cmd_builders_finalize(a):
        result = builders.finalize_catalog(
            serialization.load_json(a.catalog), serialization.load_json(a.observations)
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_finalize.set_defaults(func=_cmd_builders_finalize)

    builders_bind_source = builders_actions.add_parser("bind-source")
    builders_bind_source.add_argument("--catalog", type=Path, required=True)
    builders_bind_source.add_argument("--output", type=Path, required=True)

    def _cmd_builders_bind_source(a):
        result = builders.bind_source_catalog(serialization.load_json(a.catalog))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_bind_source.set_defaults(func=_cmd_builders_bind_source)

    builders_scan = builders_actions.add_parser("scan-registry")
    builders_scan.add_argument("--output", type=Path, required=True)

    def _cmd_builders_scan(a):
        result = builders.scan_registry_builders(policy.resolve())
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_scan.set_defaults(func=_cmd_builders_scan)

    upstreams_parser = groups.add_parser("upstreams")
    upstreams_actions = upstreams_parser.add_subparsers(dest="action", required=True)

    upstreams_candidates = upstreams_actions.add_parser("candidates")
    upstreams_candidates.add_argument(
        "--release", type=Path, default=policy.DEFAULT_RELEASE
    )
    upstreams_candidates.add_argument(
        "--release-type", choices=policy.RELEASE_TYPES, default="stable"
    )
    upstreams_candidates.add_argument("--tag-fixture", type=Path)
    upstreams_candidates.add_argument("--pr-default", action="store_true")
    upstreams_candidates.add_argument("--output", type=Path, required=True)

    def _cmd_upstreams_candidates(a):
        result = upstream.resolve_runtime_candidates(
            policy.resolve(a.release, release_type=a.release_type),
            tag_fixture=(
                serialization.load_json(a.tag_fixture) if a.tag_fixture else None
            ),
            pr_default=a.pr_default,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    upstreams_candidates.set_defaults(func=_cmd_upstreams_candidates)

    upstreams_resolve = upstreams_actions.add_parser("resolve")
    upstreams_resolve.add_argument(
        "--release", type=Path, default=policy.DEFAULT_RELEASE
    )
    upstreams_resolve.add_argument(
        "--release-type", choices=policy.RELEASE_TYPES, default="stable"
    )
    upstreams_resolve.add_argument("--candidates", type=Path, required=True)
    upstreams_resolve.add_argument("--runtime-probe", type=Path, required=True)
    upstreams_resolve.add_argument("--tag-fixture", type=Path)
    upstreams_resolve.add_argument("--output", type=Path, required=True)

    def _cmd_upstreams_resolve(a):
        result = upstream.resolve_upstreams(
            policy.resolve(a.release, release_type=a.release_type),
            candidates=serialization.load_json(a.candidates),
            runtime_probe=serialization.load_json(a.runtime_probe),
            tag_fixture=(
                serialization.load_json(a.tag_fixture) if a.tag_fixture else None
            ),
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    upstreams_resolve.set_defaults(func=_cmd_upstreams_resolve)

    plan_parser = groups.add_parser("plan")
    plan_actions = plan_parser.add_subparsers(dest="action", required=True)

    plan_create = plan_actions.add_parser("create")
    plan_create.add_argument("--release", type=Path, default=policy.DEFAULT_RELEASE)
    plan_create.add_argument(
        "--release-type", choices=policy.RELEASE_TYPES, default="stable"
    )
    plan_create.add_argument("--builder-catalog", type=Path, required=True)
    plan_create.add_argument("--runtime-selection", type=Path, required=True)
    plan_create.add_argument(
        "--route", choices=tuple(sorted(plan.ROUTES)), required=True
    )
    plan_create.add_argument("--git-tag")
    plan_create.add_argument("--release-kind", choices=("none", "publish", "draft"))
    plan_create.add_argument("--is-prerelease", choices=("true", "false"))
    plan_create.add_argument("--chart-version")
    plan_create.add_argument("--publication-context", type=Path)
    plan_create.add_argument("--output", type=Path, required=True)

    def _cmd_plan_create(a):
        publication = _publication_context(a.publication_context)
        result = plan.resolve_plan(
            policy.resolve(
                a.release,
                release_type=a.release_type,
                **publication,
            ),
            builder_catalog=serialization.load_json(a.builder_catalog),
            runtime_selection=serialization.load_json(a.runtime_selection),
            route=a.route,
            git_tag=a.git_tag,
            release_kind=a.release_kind,
            is_prerelease=(
                None if a.is_prerelease is None else a.is_prerelease == "true"
            ),
            chart_version=a.chart_version,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    plan_create.set_defaults(func=_cmd_plan_create)

    plan_retag = plan_actions.add_parser("retag-pr")
    plan_retag.add_argument("--plan", type=Path, required=True)
    plan_retag.add_argument("--pr-number", required=True)
    plan_retag.add_argument("--author", required=True)
    plan_retag.add_argument("--run-id", required=True)
    plan_retag.add_argument("--output", type=Path, required=True)

    def _cmd_plan_retag(a):
        result = plan.retag_pr_plan(
            serialization.load_json(a.plan),
            pr_number=a.pr_number,
            author=a.author,
            run_id=a.run_id,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    plan_retag.set_defaults(func=_cmd_plan_retag)

    plan_select = plan_actions.add_parser("select")
    plan_select.add_argument("--plan", type=Path, required=True)
    plan_select.add_argument("--kind", choices=("wheel", "image"), required=True)
    plan_select.add_argument("--id", required=True)
    plan_select.add_argument("--output", type=Path, required=True)

    def _cmd_plan_select(a):
        result = plan.select_task(serialization.load_json(a.plan), a.kind, a.id)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    plan_select.set_defaults(func=_cmd_plan_select)

    wheel_actions = groups.add_parser("wheel").add_subparsers(
        dest="action", required=True
    )
    wheel_prepare = wheel_actions.add_parser("prepare-source")
    wheel_prepare.add_argument("--source-root", type=Path, required=True)
    wheel_prepare.add_argument("--distribution", required=True)
    wheel_prepare.set_defaults(
        func=lambda a: wheel.prepare_wheel_source(a.source_root, a.distribution)
    )

    wheel_record = wheel_actions.add_parser("record-result")
    wheel_record.add_argument("--task", type=Path, required=True)
    wheel_record.add_argument("--wheel", type=Path, required=True)
    wheel_record.add_argument("--output", type=Path, required=True)

    def _cmd_wheel_record(a):
        result = wheel.record_wheel_result(serialization.load_json(a.task), a.wheel)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    wheel_record.set_defaults(func=_cmd_wheel_record)

    meta_actions = groups.add_parser("meta").add_subparsers(
        dest="action", required=True
    )
    meta_source = meta_actions.add_parser("materialize-source")
    meta_source.add_argument("--plan", type=Path, required=True)
    meta_source.add_argument("--output-dir", type=Path, required=True)

    def _cmd_meta_source(a):
        path = meta.materialize_meta_source(
            serialization.load_json(a.plan), a.output_dir
        )
        return {"source": str(path)}

    meta_source.set_defaults(func=_cmd_meta_source)

    meta_result = meta_actions.add_parser("record-result")
    meta_result.add_argument("--plan", type=Path, required=True)
    meta_result.add_argument("--wheel", type=Path, required=True)
    meta_result.add_argument("--output", type=Path, required=True)

    def _cmd_meta_result(a):
        result = meta.record_meta_wheel(serialization.load_json(a.plan), a.wheel)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    meta_result.set_defaults(func=_cmd_meta_result)

    toolkit_parser = groups.add_parser("toolkit")
    toolkit_actions = toolkit_parser.add_subparsers(dest="action", required=True)
    toolkit_source = toolkit_actions.add_parser("prepare-source")
    toolkit_source.add_argument("--plan", type=Path, required=True)
    toolkit_source.add_argument("--source-root", type=Path, required=True)
    toolkit_source.set_defaults(
        func=lambda a: toolkit.prepare_source(
            serialization.load_json(a.plan), a.source_root
        )
    )
    toolkit_result = toolkit_actions.add_parser("record-result")
    toolkit_result.add_argument("--plan", type=Path, required=True)
    toolkit_result.add_argument("--wheel", type=Path, required=True)
    toolkit_result.add_argument("--output", type=Path, required=True)

    def _cmd_toolkit_result(a):
        result = toolkit.record_result(serialization.load_json(a.plan), a.wheel)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    toolkit_result.set_defaults(func=_cmd_toolkit_result)

    pypi_parser = groups.add_parser("pypi")
    pypi_actions = pypi_parser.add_subparsers(dest="action", required=True)

    pypi_publish = pypi_actions.add_parser("publish")
    pypi_publish.add_argument("--release-plan", type=Path, required=True)
    pypi_publish.add_argument("--wheel-results-dir", type=Path, required=True)
    pypi_publish.add_argument("--meta-result", type=Path, required=True)
    pypi_publish.add_argument("--wheel-root", type=Path, required=True)
    pypi_publish.add_argument("--meta-root", type=Path, required=True)
    pypi_publish.add_argument("--toolkit-root", type=Path)
    pypi_publish.add_argument("--repository-url", required=True)
    pypi_publish.add_argument("--attempts", type=int, default=12)
    pypi_publish.add_argument("--interval", type=float, default=5.0)
    pypi_publish.add_argument("--output", type=Path, required=True)

    def _cmd_pypi_publish(a):
        backend_results = [
            serialization.load_json(path)
            for path in sorted(a.wheel_results_dir.rglob("wheel-result.json"))
        ]
        release_plan = serialization.load_json(a.release_plan)
        toolkit_result, _ = toolkit.load_artifact(release_plan, a.toolkit_root)
        publication = pypi.build_publication(
            release_plan,
            backend_results,
            serialization.load_json(a.meta_result),
            toolkit_result,
        )
        if a.repository_url != publication["repository_url"]:
            raise ValueError("PyPI repository URL differs from the authorized plan")
        token = os.environ.get("PYPI_API_TOKEN", "")
        projects = [*publication["backends"], publication["meta"]]
        if publication.get("toolkit") is not None:
            projects.append(publication["toolkit"])
        expected_sha256 = {
            file["filename"]: file["sha256"]
            for project in projects
            for file in project["files"]
        }
        uploader = pypi.make_twine_uploader(
            roots=[a.wheel_root, a.meta_root]
            + ([a.toolkit_root] if a.toolkit_root else []),
            expected_sha256=expected_sha256,
            repository_url=a.repository_url,
            token=token,
        )
        result = pypi.publish(
            publication,
            uploader=uploader,
            fetch=lambda project, version: pypi.fetch_version_json(
                project,
                version,
                json_api_url=publication["json_api"],
            ),
            attempts=a.attempts,
            interval=a.interval,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    pypi_publish.set_defaults(func=_cmd_pypi_publish)

    runtime_parser = groups.add_parser("runtime")
    runtime_actions = runtime_parser.add_subparsers(dest="action", required=True)

    runtime_inspect = runtime_actions.add_parser("inspect")
    runtime_inspect.add_argument("--reference", action="append", default=[])
    runtime_inspect.add_argument("--references-file", type=Path)
    runtime_inspect.add_argument("--output", type=Path, required=True)

    def _cmd_runtime_inspect(a):
        references = list(a.reference)
        if a.references_file is not None:
            references.extend(
                line.strip()
                for line in a.references_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        formal = policy.resolve()
        result = runtime.inspect_runtime_references(
            references,
            products=formal["products"],
            runners=formal["runners"],
            manifest_loader=lambda reference: registry.read_json("manifest", reference),
            config_loader=lambda reference: registry.read_json("config", reference),
            digest_loader=lambda reference: registry.read("digest", reference).strip(),
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    runtime_inspect.set_defaults(func=_cmd_runtime_inspect)

    runtime_aggregate = runtime_actions.add_parser("aggregate")
    runtime_aggregate.add_argument("--inspection", type=Path, required=True)
    runtime_aggregate.add_argument("--probe-dir", type=Path, required=True)
    runtime_aggregate.add_argument("--output", type=Path, required=True)

    def _cmd_runtime_aggregate(a):
        probe_paths = sorted(a.probe_dir.rglob("runtime-probe-raw.json"))
        result = runtime.aggregate_runtime_probes(
            serialization.load_json(a.inspection),
            [serialization.load_json(path) for path in probe_paths],
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    runtime_aggregate.set_defaults(func=_cmd_runtime_aggregate)

    runtime_resolve = runtime_actions.add_parser("resolve")
    runtime_resolve.add_argument("--probe", type=Path, required=True)
    runtime_resolve.add_argument("--builder-registry", type=Path, required=True)
    runtime_resolve.add_argument("--pr-number", required=True)
    runtime_resolve.add_argument("--author", required=True)
    runtime_resolve.add_argument("--run-id", required=True)
    runtime_resolve.add_argument("--output-dir", type=Path, required=True)

    def _cmd_runtime_resolve(a):
        result = pr.resolve_pr_request(
            policy.resolve(),
            serialization.load_json(a.probe),
            serialization.load_json(a.builder_registry),
            pr_number=a.pr_number,
            author=a.author,
            run_id=a.run_id,
        )
        a.output_dir.mkdir(parents=True, exist_ok=True)
        _write(a.output_dir / "pr-resolution.json", result)
        if result["ok"]:
            _write(a.output_dir / "runtime-selection.json", result["selection"])
            _write(a.output_dir / "builder-catalog.json", result["builder_catalog"])
            _write(a.output_dir / "publication.json", result["publication"])
        return result

    runtime_resolve.set_defaults(func=_cmd_runtime_resolve)

    runtime_receipt = runtime_actions.add_parser("receipt")
    runtime_receipt.add_argument("--reference", action="append", default=[])
    runtime_receipt.add_argument("--references-file", type=Path)
    runtime_receipt.add_argument("--stage", action="append", default=[])
    runtime_receipt.add_argument("--inspection", type=Path)
    runtime_receipt.add_argument("--probe", type=Path)
    runtime_receipt.add_argument("--resolution", type=Path)
    runtime_receipt.add_argument("--publication", type=Path)
    runtime_receipt.add_argument("--failure-dir", type=Path)
    runtime_receipt.add_argument("--run-url", default="")
    runtime_receipt.add_argument("--output", type=Path, required=True)
    runtime_receipt.add_argument("--markdown", type=Path)

    def _cmd_runtime_receipt(a):
        references = list(a.reference)
        if a.references_file is not None and a.references_file.is_file():
            references.extend(
                line.strip()
                for line in a.references_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        stage_results = {}
        for raw_stage in a.stage:
            name, separator, result = raw_stage.partition("=")
            if not separator:
                raise ValueError("receipt stage must use name=result")
            stage_results[name] = result
        resolution = (
            serialization.load_json(a.resolution)
            if a.resolution is not None and a.resolution.is_file()
            else None
        )
        builder_matches = resolution.get("builder_matches") if resolution else None
        failures = (
            []
            if builder_matches is not None
            else resolution.get("problems", []) if resolution else []
        )
        external_failures = []
        if a.failure_dir is not None and a.failure_dir.is_dir():
            external_failures = [
                serialization.load_json(path)
                for path in sorted(a.failure_dir.rglob("*-failure.json"))
            ]
        result = runtime.build_receipt(
            requested_refs=references,
            stage_results=stage_results,
            inspection=(
                serialization.load_json(a.inspection)
                if a.inspection is not None and a.inspection.is_file()
                else None
            ),
            runtime_probe=(
                serialization.load_json(a.probe)
                if a.probe is not None and a.probe.is_file()
                else None
            ),
            builder_matches=builder_matches,
            publication=(
                serialization.load_json(a.publication)
                if a.publication is not None and a.publication.is_file()
                else None
            ),
            failures=[*failures, *external_failures],
            run_url=a.run_url,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        if a.markdown is not None:
            a.markdown.parent.mkdir(parents=True, exist_ok=True)
            a.markdown.write_text(
                runtime.render_receipt_markdown(result), encoding="utf-8"
            )
        return result

    runtime_receipt.set_defaults(func=_cmd_runtime_receipt)

    problems_parser = groups.add_parser("problems")
    problems_actions = problems_parser.add_subparsers(dest="action", required=True)
    problems_render = problems_actions.add_parser("render")
    problems_render.add_argument("--selection", type=Path, required=True)
    problems_render.add_argument("--summary", type=Path, required=True)
    problems_render.add_argument("--issue", type=Path, required=True)
    problems_render.add_argument("--action", type=Path, required=True)

    def _cmd_problems_render(a):
        selection = upstream.validate_selection(serialization.load_json(a.selection))
        values = selection["problems"]
        summary = problems.render_actions_summary(values)
        issue = problems.render_rolling_issue(values)
        action = {"action": problems.decide_rolling_issue_action(values)}
        for path in (a.summary, a.issue, a.action):
            path.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text(summary, encoding="utf-8")
        _write(a.issue, issue)
        _write(a.action, action)
        return {**action, "problem_count": len(values)}

    problems_render.set_defaults(func=_cmd_problems_render)

    config = groups.add_parser("config")
    config_actions = config.add_subparsers(dest="action", required=True)
    validate = config_actions.add_parser("validate")
    _paths(validate)
    validate.set_defaults(
        func=lambda a: {
            "schema_version": 6,
            "products": len(policy.load(a.release)["release"]["products"]),
            "backends": len(policy.load(a.release)["platforms"]["backends"]),
        }
    )  # noqa: E501

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    print(_json(result))
    return 0
