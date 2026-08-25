# fmt: off
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import (
    builders,
    compact,
    core,
    policy,
    pr,
    problems,
    registry,
    runtime,
    upstream,
    wheel,
)

catalog_resolution = registry


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release", type=Path, default=core.DEFAULT_RELEASE)
    parser.add_argument("--schema-dir", type=Path, default=core.DEFAULT_SCHEMA_DIR)


def _write(path: Path, value: object) -> None:
    path.write_bytes(core.canonical_bytes(value) + b"\n")


def _crane_output(operation: str, reference: str) -> str:
    completed = None
    last_error = ""
    for attempt in range(1, 4):
        try:
            completed = subprocess.run(
                ["crane", operation, reference],
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            completed = None
            last_error = "timed out after 60 seconds"
        if completed is None:
            if attempt < 3:
                time.sleep(2**attempt)
            continue
        if completed.returncode == 0:
            return completed.stdout
        last_error = completed.stderr.strip() or str(completed.returncode)
        if attempt < 3:
            time.sleep(2**attempt)
    raise ValueError(
        f"crane {operation} failed for {reference}: {last_error or 'unknown error'}"
    )


def _crane_json(operation: str, reference: str) -> object:
    try:
        return json.loads(_crane_output(operation, reference))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"crane {operation} returned malformed JSON for {reference}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ucm_release")
    groups = parser.add_subparsers(dest="group", required=True)

    builders_parser = groups.add_parser("builders")
    builders_actions = builders_parser.add_subparsers(dest="action", required=True)

    builders_discover = builders_actions.add_parser("discover")
    builders_discover.add_argument("--config", type=Path, default=policy.DEFAULT_PLATFORMS)
    builders_discover.add_argument("--owner")
    builders_discover.add_argument("--selection", type=Path, required=True)
    builders_discover.add_argument("--output", type=Path, required=True)

    def _cmd_builders_discover(a):
        formal = policy.resolve(platforms_path=a.config)
        result = builders.catalog_from_selection(
            core.load_json(a.selection),
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
    builders_sync_plan.add_argument(
        "--existing", "--existing-tags", dest="existing", type=Path, required=True
    )
    builders_sync_plan.add_argument("--output", type=Path, required=True)

    def _cmd_builders_sync_plan(a):
        result = builders.compute_sync_plan(
            core.load_json(a.catalog), core.load_json(a.existing)
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_sync_plan.set_defaults(func=_cmd_builders_sync_plan)

    builders_labels = builders_actions.add_parser("labels")
    builders_labels.add_argument("--builder", type=Path, required=True)
    builders_labels.add_argument("--output", type=Path, required=True)

    def _cmd_builders_labels(a):
        result = builders.builder_labels(core.load_json(a.builder))
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
            core.load_json(a.catalog), core.load_json(a.observations)
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    builders_finalize.set_defaults(func=_cmd_builders_finalize)

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
    upstreams_candidates.add_argument("--release", type=Path, default=core.DEFAULT_RELEASE)
    upstreams_candidates.add_argument("--tag-fixture", type=Path)
    upstreams_candidates.add_argument("--pr-default", action="store_true")
    upstreams_candidates.add_argument("--output", type=Path, required=True)

    def _cmd_upstreams_candidates(a):
        result = upstream.resolve_runtime_candidates(
            policy.resolve(a.release),
            tag_fixture=core.load_json(a.tag_fixture) if a.tag_fixture else None,
            pr_default=a.pr_default,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    upstreams_candidates.set_defaults(func=_cmd_upstreams_candidates)

    upstreams_resolve = upstreams_actions.add_parser("resolve")
    upstreams_resolve.add_argument("--release", type=Path, default=core.DEFAULT_RELEASE)
    upstreams_resolve.add_argument("--candidates", type=Path, required=True)
    upstreams_resolve.add_argument("--runtime-probe", type=Path, required=True)
    upstreams_resolve.add_argument("--tag-fixture", type=Path)
    upstreams_resolve.add_argument("--output", type=Path, required=True)

    def _cmd_upstreams_resolve(a):
        result = upstream.resolve_upstreams(
            policy.resolve(a.release),
            candidates=core.load_json(a.candidates),
            runtime_probe=core.load_json(a.runtime_probe),
            tag_fixture=core.load_json(a.tag_fixture) if a.tag_fixture else None,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    upstreams_resolve.set_defaults(func=_cmd_upstreams_resolve)

    compact_parser = groups.add_parser("compact")
    compact_actions = compact_parser.add_subparsers(dest="action", required=True)

    compact_plan = compact_actions.add_parser("plan")
    compact_plan.add_argument("--catalog", type=Path, default=core.DEFAULT_RELEASE)
    compact_plan.add_argument("--schema-dir", type=Path, default=core.DEFAULT_SCHEMA_DIR)
    compact_plan.add_argument("--builder-catalog", type=Path, required=True)
    compact_plan.add_argument("--runtime-selection", type=Path, required=True)
    compact_plan.add_argument("--route", choices=tuple(sorted(compact.ROUTES)), required=True)
    compact_plan.add_argument("--pin-upstream", action="append", default=None)
    compact_plan.add_argument("--git-tag")
    compact_plan.add_argument(
        "--release-kind", choices=("none", "publish", "draft")
    )
    compact_plan.add_argument("--is-prerelease", choices=("true", "false"))
    compact_plan.add_argument("--chart-version")
    compact_plan.add_argument("--output", type=Path, required=True)

    def _cmd_compact_plan(a):
        result = compact.resolve_plan(
            policy.resolve(a.catalog),
            builder_catalog=core.load_json(a.builder_catalog),
            runtime_selection=core.load_json(a.runtime_selection),
            route=a.route,
            pinned_upstreams=a.pin_upstream,
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

    compact_plan.set_defaults(func=_cmd_compact_plan)

    compact_retag = compact_actions.add_parser("retag-pr")
    compact_retag.add_argument("--plan", type=Path, required=True)
    compact_retag.add_argument("--pr-number", required=True)
    compact_retag.add_argument("--author", required=True)
    compact_retag.add_argument("--run-id", required=True)
    compact_retag.add_argument("--output", type=Path, required=True)

    def _cmd_compact_retag(a):
        result = compact.retag_pr_plan(
            core.load_json(a.plan),
            pr_number=a.pr_number,
            author=a.author,
            run_id=a.run_id,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    compact_retag.set_defaults(func=_cmd_compact_retag)

    compact_select = compact_actions.add_parser("select")
    compact_select.add_argument("--plan", type=Path, required=True)
    compact_select.add_argument("--kind", choices=("wheel", "image"), required=True)
    compact_select.add_argument("--id", required=True)
    compact_select.add_argument("--output", type=Path, required=True)

    def _cmd_compact_select(a):
        result = compact.select_task(core.load_json(a.plan), a.kind, a.id)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    compact_select.set_defaults(func=_cmd_compact_select)

    compact_prepare = compact_actions.add_parser("prepare-wheel-source")
    compact_prepare.add_argument("--source-root", type=Path, required=True)
    compact_prepare.add_argument("--distribution", required=True)
    compact_prepare.set_defaults(
        func=lambda a: compact.prepare_wheel_source(a.source_root, a.distribution)
    )

    compact_record = compact_actions.add_parser("record-wheel-result")
    compact_record.add_argument("--task", type=Path, required=True)
    compact_record.add_argument("--wheel", type=Path, required=True)
    compact_record.add_argument("--output", type=Path, required=True)

    def _cmd_compact_record(a):
        result = compact.record_wheel_result(core.load_json(a.task), a.wheel)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result

    compact_record.set_defaults(func=_cmd_compact_record)

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
            manifest_loader=lambda reference: _crane_json("manifest", reference),
            config_loader=lambda reference: _crane_json("config", reference),
            digest_loader=lambda reference: _crane_output("digest", reference).strip(),
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
            core.load_json(a.inspection),
            [core.load_json(path) for path in probe_paths],
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
            core.load_json(a.probe),
            core.load_json(a.builder_registry),
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
            core.load_json(a.resolution)
            if a.resolution is not None and a.resolution.is_file()
            else None
        )
        builder_matches = resolution.get("builder_matches") if resolution else None
        failures = [] if builder_matches is not None else resolution.get("problems", []) if resolution else []
        external_failures = []
        if a.failure_dir is not None and a.failure_dir.is_dir():
            external_failures = [
                core.load_json(path)
                for path in sorted(a.failure_dir.rglob("*-failure.json"))
            ]
        result = runtime.build_receipt(
            requested_refs=references,
            stage_results=stage_results,
            inspection=(
                core.load_json(a.inspection)
                if a.inspection is not None and a.inspection.is_file()
                else None
            ),
            runtime_probe=(
                core.load_json(a.probe)
                if a.probe is not None and a.probe.is_file()
                else None
            ),
            builder_matches=builder_matches,
            publication=(
                core.load_json(a.publication)
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
        selection = upstream.validate_selection(core.load_json(a.selection))
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
    validate.set_defaults(func=lambda a: {'schema_version': 5, 'products': len(policy.load(a.release)['release']['products']), 'backends': len(policy.load(a.release)['platforms']['backends'])})  # fmt: skip  # noqa: E501

    catalog_parser = groups.add_parser("catalog")
    catalog_actions = catalog_parser.add_subparsers(dest="action", required=True)
    catalog_validate = catalog_actions.add_parser("validate")
    catalog_validate.add_argument("--catalog", type=Path, default=core.DEFAULT_RELEASE)
    catalog_validate.add_argument('--schema-dir', type=Path, default=core.DEFAULT_SCHEMA_DIR)  # fmt: skip  # noqa: E501
    catalog_validate.add_argument('--repository-root', type=Path, default=core.REPO_ROOT)  # fmt: skip  # noqa: E501

    def _cmd_catalog_validate(a):
        raw = core.load_yaml(a.catalog)
        if raw.get("kind") == "ucm-release-policy":
            formal = policy.resolve(a.catalog, repository_root=a.repository_root)
            return {
                "kind": "ucm-catalog-validation",
                "schema_version": 1,
                "config_sha256": core.sha256_value(formal),
                "products": len(formal["products"]),
                "backends": len(formal["backends"]),
            }
        catalog = core.load_catalog(
            a.catalog, a.schema_dir, repository_root=a.repository_root
        )
        catalog_resolution.validate_catalog_tag_grammar(catalog)
        return {
            "kind": "ucm-catalog-validation",
            "schema_version": 1,
            "config_sha256": core.sha256_value(catalog),
            "upstream_products": len(catalog["upstream_products"]),
            "compatibility_rules": len(catalog["compatibility"]["rules"]),
        }

    catalog_validate.set_defaults(func=_cmd_catalog_validate)

    catalog_resolve = catalog_actions.add_parser("resolve")
    catalog_resolve.add_argument("--catalog", type=Path, default=core.DEFAULT_RELEASE)
    catalog_resolve.add_argument('--schema-dir', type=Path, default=core.DEFAULT_SCHEMA_DIR)  # fmt: skip  # noqa: E501
    catalog_resolve.add_argument('--lane', choices=('feature-candidate', 'protected-tag'), required=True)  # fmt: skip  # noqa: E501
    catalog_resolve.add_argument("--source-sha", required=True)
    catalog_resolve.add_argument("--builder-catalog", type=Path, required=True)
    catalog_resolve.add_argument("--fixture", type=Path)
    catalog_resolve.add_argument("--pin-upstream", action="append", default=None, metavar="REPO:TAG", help="pin a specific upstream image:tag (PR path; repeatable; skips the registry scan + catalog compatibility gates)")  # fmt: skip  # noqa: E501
    catalog_resolve.add_argument("--output", type=Path, required=True)

    def _cmd_resolve(a):
        release = core.load_catalog(a.catalog, a.schema_dir)
        builder_catalog = core.load_json(a.builder_catalog)
        fixture = core.load_json(a.fixture) if a.fixture else None
        result = catalog_resolution.resolve_catalog(release, builder_catalog=builder_catalog, source_sha=a.source_sha, lane=a.lane, fixture=fixture, pin_upstreams=a.pin_upstream)  # fmt: skip  # noqa: E501
        a.output.parent.mkdir(parents=True, exist_ok=True)
        _write(a.output, result)
        return result
    catalog_resolve.set_defaults(func=_cmd_resolve)

    validate_resolved_plan = catalog_actions.add_parser("validate-resolved-plan")
    validate_resolved_plan.add_argument("--plan", type=Path, required=True)

    def _cmd_validate_resolved_plan(a):
        plan = core.load_json(a.plan)
        registry.validate_resolved_plan(plan)
        return {"kind": "ucm-resolved-plan-validation", "schema_version": 1}

    validate_resolved_plan.set_defaults(func=_cmd_validate_resolved_plan)

    validate_main_loop = catalog_actions.add_parser("validate-main-loop")
    validate_main_loop.add_argument("--plan", type=Path, required=True)
    validate_main_loop.add_argument("--catalog", type=Path, default=core.DEFAULT_RELEASE)
    validate_main_loop.add_argument('--schema-dir', type=Path, default=core.DEFAULT_SCHEMA_DIR)  # fmt: skip  # noqa: E501

    def _cmd_validate_main_loop(a):
        plan = core.load_json(a.plan)
        catalog = core.load_catalog(a.catalog, a.schema_dir)
        counts = registry.validate_main_full_loop_plan(plan, catalog)
        return {"kind": "ucm-main-full-loop-validation", "schema_version": 1, **counts}
    validate_main_loop.set_defaults(func=_cmd_validate_main_loop)

    core_parser = groups.add_parser("core")
    core_actions = core_parser.add_subparsers(dest="action", required=True)
    tag_preflight = core_actions.add_parser("tag-preflight")
    tag_preflight.add_argument('--lane', choices=('feature-candidate', 'protected-tag'), required=True)  # fmt: skip  # noqa: E501
    tag_preflight.add_argument("--resolved-plan", type=Path)
    tag_preflight.add_argument('--catalog-planner', action='store_true', help='resolve current catalog authority only in the initial planning job')  # fmt: skip  # noqa: E501
    _paths(tag_preflight)

    def _cmd_tag_preflight(a):
        if a.catalog_planner:
            if a.resolved_plan is not None:
                raise ValueError('catalog planner mode cannot consume a resolved plan binding')  # fmt: skip  # noqa: E501
            return core.tag_preflight(lane=a.lane, release_path=a.release, schema_dir=a.schema_dir)  # fmt: skip  # noqa: E501
        if a.resolved_plan is None:
            raise ValueError('tag preflight requires a resolved plan')  # fmt: skip  # noqa: E501
        resolved_plan = core.load_json(a.resolved_plan)
        registry.validate_resolved_plan(resolved_plan)
        if resolved_plan['lane'] != a.lane: raise ValueError('tag preflight lane differs from resolved plan')  # noqa: E701,E501
        return core.tag_preflight(lane=a.lane, authority=resolved_plan['source'])  # fmt: skip  # noqa: E501
    tag_preflight.set_defaults(func=_cmd_tag_preflight)

    wheel_parser = groups.add_parser("wheel")
    wheel_actions = wheel_parser.add_subparsers(dest="action", required=True)

    wc_build_config = wheel_actions.add_parser("build-config")
    wc_build_config.add_argument("--task-file", type=Path, required=True)
    wc_build_config.add_argument("--authority-file", type=Path, required=True)
    wc_build_config.add_argument("--output", type=Path, required=True)
    wc_build_config.set_defaults(
        func=lambda a: wheel.build_wheel_config(
            a.task_file, a.authority_file, a.output
        )
    )

    wc_prepare_source = wheel_actions.add_parser("prepare-source")
    wc_prepare_source.add_argument("--build-config", type=Path, required=True)
    wc_prepare_source.add_argument("--source-root", type=Path, required=True)
    wc_prepare_source.set_defaults(
        func=lambda a: wheel.prepare_wheel_source(a.build_config, a.source_root)
    )

    wc_source_context = wheel_actions.add_parser("source-context")
    wc_source_context.add_argument("--repository-root", type=Path, default=core.REPO_ROOT)
    wc_source_context.add_argument("--source-sha", required=True)
    wc_source_context.add_argument("--source-version", required=True)
    wc_source_context.add_argument("--output-dir", type=Path, required=True)
    wc_source_context.set_defaults(
        func=lambda a: wheel.prepare_source_context(
            a.output_dir,
            a.source_sha,
            a.source_version,
            repository_root=a.repository_root,
        )
    )

    wc_production_authority = wheel_actions.add_parser("production-authority")
    wc_production_authority.add_argument("--task", type=Path, required=True)
    wc_production_authority.add_argument(
        "--source-context", type=Path, required=True
    )
    wc_production_authority.add_argument("--source-date-epoch", type=int, required=True)
    wc_production_authority.add_argument("--tool-wheels", type=Path, required=True)
    wc_production_authority.add_argument("--output", type=Path, required=True)
    wc_production_authority.set_defaults(
        func=lambda a: wheel.build_production_authority(
            core.load_json(a.task),
            core.load_json(a.source_context),
            a.source_date_epoch,
            core.load_json(a.tool_wheels),
            output=a.output,
        )
    )

    wc_env = wheel_actions.add_parser("check-environment")
    wc_env.add_argument("--task", type=Path, required=True)
    wc_env.add_argument("--python-executable", type=Path, required=True)

    def _cmd_wheel_check_env(a):
        task = core.load_json(a.task)
        return wheel.check_build_environment(task, python_executable=a.python_executable)  # fmt: skip  # noqa: E501
    wc_env.set_defaults(func=_cmd_wheel_check_env)

    wc_ctx = wheel_actions.add_parser("verify-context")
    wc_ctx.add_argument("--archive", type=Path, required=True)
    wc_ctx.add_argument("--manifest", type=Path, required=True)
    wc_ctx.add_argument("--source-root", type=Path, required=True)
    wc_ctx.add_argument("--commit-payload", type=Path)
    wc_ctx.add_argument("--expected-source-sha")

    def _cmd_wheel_verify_ctx(a):
        return wheel.verify_source_context(a.archive, a.manifest, a.source_root, a.commit_payload, a.expected_source_sha)  # fmt: skip  # noqa: E501
    wc_ctx.set_defaults(func=_cmd_wheel_verify_ctx)

    wc_auth = wheel_actions.add_parser("authority")
    wc_auth.add_argument("--spec-id", required=True)
    wc_auth.add_argument("--source-sha", required=True)
    wc_auth.add_argument("--source-date-epoch", required=True)
    wc_auth.add_argument("--builder-coordinate", required=True)
    wc_auth.add_argument("--wheelhouse", type=Path, required=True)
    wc_auth.add_argument("--source-archive", type=Path, required=True)
    wc_auth.add_argument("--source-commit-payload", type=Path, required=True)
    wc_auth.add_argument("--source-manifest", type=Path, required=True)
    wc_auth.add_argument("--source-root", type=Path, required=True)
    wc_auth.add_argument("--task-file", type=Path, required=True)
    wc_auth.add_argument("--output", type=Path, required=True)

    def _cmd_wheel_authority(a):
        return wheel.build_authority_record(a.output, a.spec_id, a.source_sha, int(a.source_date_epoch), a.builder_coordinate, a.wheelhouse, a.source_archive, a.source_commit_payload, a.source_manifest, a.source_root, a.task_file)  # fmt: skip  # noqa: E501
    wc_auth.set_defaults(func=_cmd_wheel_authority)

    wc_closure = wheel_actions.add_parser("closure")
    wc_closure.add_argument("path", type=Path)
    wc_closure.add_argument("--spec-id", required=True)
    wc_closure.add_argument("--authority-file", type=Path, required=True)
    wc_closure.add_argument("--task-file", type=Path)
    wc_closure.add_argument("--output", type=Path, required=True)

    def _cmd_wheel_closure(a):
        return wheel.audit_dependency_closure(a.path, a.output, a.spec_id, a.authority_file, task_path=a.task_file)  # fmt: skip  # noqa: E501
    wc_closure.set_defaults(func=_cmd_wheel_closure)

    wc_seal = wheel_actions.add_parser("seal")
    wc_seal.add_argument("path", type=Path)
    wc_seal.add_argument("--spec-id", required=True)
    wc_seal.add_argument("--source-sha", required=True)
    wc_seal.add_argument("--build-key", required=True)
    wc_seal.add_argument("--source-date-epoch", required=True)
    wc_seal.add_argument("--authority-file", type=Path, required=True)
    wc_seal.add_argument("--dependency-closure", type=Path, required=True)
    wc_seal.add_argument("--task-file", type=Path)
    wc_seal.add_argument("--output-dir", type=Path, required=True)

    def _cmd_wheel_seal(a):
        return wheel.seal_wheel(a.path, a.output_dir, a.spec_id, a.source_sha, a.build_key, int(a.source_date_epoch), a.authority_file, a.dependency_closure, task_path=a.task_file)  # fmt: skip  # noqa: E501
    wc_seal.set_defaults(func=_cmd_wheel_seal)

    wc_inspect = wheel_actions.add_parser("inspect")
    wc_inspect.add_argument("path", type=Path)
    wc_inspect.add_argument("--spec-id", required=True)
    wc_inspect.add_argument("--expected-sha256", required=True)
    wc_inspect.add_argument('--source-kind', choices=('fixture', 'builder-candidate'), required=True)  # fmt: skip  # noqa: E501
    wc_inspect.add_argument("--task-file", type=Path)
    wc_inspect.add_argument("--release", type=Path, default=core.DEFAULT_RELEASE)
    wc_inspect.add_argument('--schema-dir', type=Path, default=core.DEFAULT_SCHEMA_DIR)  # fmt: skip  # noqa: E501
    wc_inspect.add_argument("--output", type=Path)

    def _cmd_wheel_inspect(a):
        result = wheel.inspect_wheel(a.path, a.spec_id, a.expected_sha256, a.source_kind, task_path=a.task_file, release_path=a.release, schema_dir=a.schema_dir)  # fmt: skip  # noqa: E501
        if a.output is not None:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            _write(a.output, result)
        return result
    wc_inspect.set_defaults(func=_cmd_wheel_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")
    print(_json(result))
    return 0


if __name__ == '__main__': sys.exit(main())  # noqa: E701
# fmt: on
