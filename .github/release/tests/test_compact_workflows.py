"""User-visible Actions contracts for upstream planning and parallel publishing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str) -> dict[str, object]:
    value = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    if True in value and "on" not in value:
        value["on"] = value.pop(True)
    return value


def test_user_facing_release_workflow_names_explain_the_build_lanes() -> None:
    assert _load("release-ucm.yml")["name"] == ("UCM Tag Release · Draft and Formal")
    assert _load("ucm-build-bot.yml")["name"] == (
        "UCM PR Build Robot · Wheel, Image, and Chart"
    )


def test_release_workflow_uses_crane_with_native_fallback_before_plan() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    assert jobs["open-release"]["needs"] == "classify-tag"
    assert "needs" not in jobs["select-runtime-candidates"]
    assert jobs["inspect-runtimes"]["needs"] == "select-runtime-candidates"
    assert jobs["probe-runtimes"]["needs"] == "inspect-runtimes"
    assert jobs["probe-runtimes"]["uses"] == "./.github/workflows/_probe-runtime.yml"
    assert "if" not in jobs["probe-runtimes"]
    assert (
        jobs["probe-runtimes"]["with"]["enabled"] == "${{ matrix.fallback_required }}"
    )
    assert "probe-runtimes.result == 'skipped'" in jobs["resolve-upstreams"]["if"]
    assert (
        jobs["inspect-runtimes"]["outputs"]["has_probe_fallback"]
        == "${{ steps.inspect.outputs.has_probe_fallback }}"
    )
    assert set(jobs["resolve-upstreams"]["needs"]) == {
        "select-runtime-candidates",
        "inspect-runtimes",
        "probe-runtimes",
    }
    assert jobs["sync-builders"]["needs"] == "resolve-upstreams"
    assert set(jobs["plan"]["needs"]) == {"resolve-upstreams", "sync-builders"}
    assert (
        jobs["sync-builders"]["with"]["runtime_selection_artifact"]
        == "${{ needs.resolve-upstreams.outputs.runtime_selection_artifact }}"
    )
    plan_text = yaml.safe_dump(jobs["plan"])
    assert "--runtime-selection" in plan_text
    assert "image_index_matrix" in plan_text
    assert "'{include:[.families[]" in plan_text


def test_pr_gate_keeps_ucm_artifact_builds_in_the_robot_lane() -> None:
    text = (WORKFLOWS / "pull-request.yml").read_text(encoding="utf-8")
    assert "release-catalog-smoke" not in text
    assert "./.github/workflows/release-ucm.yml" not in text


def test_release_workflow_has_staged_publication_jobs() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    publish_jobs = [
        "open-release",
        "publish-release-artifacts",
        "publish-image-members",
        "publish-image-indexes",
        "publish-pypi",
        "publish-chart-oci",
        "update-release-images",
    ]
    assert all(name in jobs for name in publish_jobs)
    assert set(jobs["publish-release-artifacts"]["needs"]) == {
        "plan",
        "open-release",
        "build-wheels",
        "package-chart",
    }
    assert set(jobs["publish-image-members"]["needs"]) == {
        "plan",
        "build-images",
        "publish-release-artifacts",
    }
    assert set(jobs["build-images"]["needs"]) == {
        "plan",
        "build-wheels",
        "publish-release-artifacts",
    }
    assert "publish-release-artifacts.result == 'success'" in jobs["build-images"]["if"]
    assert set(jobs["publish-image-indexes"]["needs"]) == {
        "plan",
        "publish-image-members",
    }
    assert set(jobs["publish-pypi"]["needs"]) == {
        "plan",
        "publish-release-artifacts",
    }
    assert set(jobs["publish-chart-oci"]["needs"]) == {
        "plan",
        "publish-release-artifacts",
    }
    assert set(jobs["update-release-images"]["needs"]) == {
        "plan",
        "open-release",
        "publish-release-artifacts",
        "build-images",
        "publish-image-members",
        "publish-image-indexes",
    }
    for name in (
        "open-release",
        "publish-release-artifacts",
        "update-release-images",
    ):
        assert jobs[name]["env"]["GH_REPO"] == "${{ github.repository }}"


def test_artifact_stage_keeps_release_state_internal() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    text = yaml.safe_dump(jobs["publish-release-artifacts"])
    assert "release.py artifacts" in text
    assert "release.py notes" in text
    assert "release-state.json" in text
    assert "release-manifest.json" not in text
    assert "ucm-release-stage-run-${{ github.run_id }}" in text
    assert "build-images" not in jobs["publish-release-artifacts"]["needs"]
    upload = next(
        step
        for step in jobs["publish-release-artifacts"]["steps"]
        if step.get("name") == "Upload Wheels and Chart"
    )
    upload_lines = upload["run"].splitlines()
    upload_index = upload_lines.index('gh release upload "${tag}" --clobber \\')
    assert upload_lines[upload_index : upload_index + 2] == [
        'gh release upload "${tag}" --clobber \\',
        "  input/wheels/*/*.whl input/chart/*.tgz",
    ]
    assert "SHA256SUMS" not in upload["run"]
    assert "release-manifest.json" not in upload["run"]
    assert 'gh api "/repos/${GH_REPO}/releases/${release_id}"' in upload["run"]
    assert '--repository "${GH_REPO}"' in upload["run"]
    failure_step = jobs["publish-release-artifacts"]["steps"][-1]
    assert failure_step["if"] == "${{ failure() }}"
    assert "artifacts-failed" in failure_step["run"]


def test_public_release_updates_do_not_upload_internal_state() -> None:
    update = next(
        step
        for step in _load("release-ucm.yml")["jobs"]["update-release-images"]["steps"]
        if step.get("id") == "update-release"
    )

    assert "release-state.json" in update["run"]
    assert "release-manifest.json" not in update["run"]
    assert "gh release upload" not in update["run"]


def test_handwritten_release_notes_start_with_status() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    runs = [
        jobs["open-release"]["steps"][0]["run"],
        jobs["report-planning-failure"]["steps"][0]["run"],
        jobs["publish-release-artifacts"]["steps"][0]["run"],
        jobs["publish-release-artifacts"]["steps"][-1]["run"],
        jobs["update-release-images"]["steps"][-1]["run"],
    ]

    for run in runs:
        assert "printf '%s\\n' \\\n  \"Status:" in run
        assert '"# UCM ${tag}"' not in run
        assert "Checksums:" not in run


def test_opened_release_reports_pre_plan_failures() -> None:
    job = _load("release-ucm.yml")["jobs"]["report-planning-failure"]

    assert "always()" in job["if"]
    assert "needs.plan.result != 'success'" in job["if"]
    assert job["env"]["GH_REPO"] == "${{ github.repository }}"
    assert "artifacts-failed" in job["steps"][0]["run"]


def test_image_failure_notes_are_not_overwritten_by_the_fallback() -> None:
    steps = _load("release-ucm.yml")["jobs"]["update-release-images"]["steps"]
    update = next(step for step in steps if step.get("id") == "update-release")
    require = next(
        step
        for step in steps
        if step.get("name") == "Require complete image publication"
    )
    fallback = steps[-1]

    assert "release.status" not in update["run"]
    assert "release.py notes" in update["run"]
    assert 'gh api "/repos/${GH_REPO}/releases/${release_id}"' in update["run"]
    assert "release.status" in require["run"]
    assert "steps.update-release.outcome != 'success'" in fallback["if"]


def test_formal_and_draft_tags_open_before_builds_with_exact_api_lookup() -> None:
    workflow = _load("release-ucm.yml")
    assert workflow["on"]["push"]["tags"] == ["v*", "draft/v*"]
    jobs = workflow["jobs"]
    open_text = yaml.safe_dump(jobs["open-release"])
    open_run = jobs["open-release"]["steps"][0]["run"]
    assert jobs["open-release"]["needs"] == "classify-tag"
    assert "--verify-tag" in open_text
    assert "--paginate --slurp" in open_text
    assert "select(.tag_name == $tag)" in open_run
    assert "wait_for_release" in open_run
    assert "for attempt in $(seq 1 15)" in open_run
    assert "sleep 2" in open_run
    assert open_run.count('-f "tag_name=${tag}"') == 2
    assert "draft=false" in open_text
    assert "build-wheels" not in open_text
    assert "draft/v*" in yaml.safe_dump(jobs["plan"])
    plan_run = next(
        step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan"
    )
    assert "--git-tag" in plan_run
    assert ".git_tag == $tag" in plan_run
    assert ".release_tag = $tag" not in plan_run


def test_every_release_patch_preserves_the_exact_tag_name() -> None:
    text = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    assert text.count("gh api --method PATCH") == text.count('-f "tag_name=${tag}"')


def test_direct_member_receipt_barrier_and_index_matrix_are_unbounded() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    members = jobs["publish-image-members"]
    indexes = jobs["publish-image-indexes"]
    assert "strategy" not in members
    assert "ucm-image-member-receipt-*" in yaml.safe_dump(members)
    assert "one published GHCR receipt per planned image" in yaml.safe_dump(members)
    assert indexes["strategy"]["fail-fast"] is False
    assert "max-parallel" not in indexes["strategy"]
    assert (
        indexes["strategy"]["matrix"]
        == "${{ fromJSON(needs.plan.outputs.image_index_matrix) }}"
    )
    assert indexes["if"] == "${{ needs.plan.outputs.has_image_indexes == 'true' }}"
    text = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    assert "while IFS=$'\\t' read -r image_id" not in text


def test_member_receipt_barrier_rejects_the_wrong_ghcr_reference(
    tmp_path: Path,
) -> None:
    steps = _load("release-ucm.yml")["jobs"]["publish-image-members"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Require one published GHCR receipt per planned image"
    )
    jq_filter = run.split('jq -e -s --slurpfile plan "${plan}" \'', 1)[1].split(
        '\n\' "${receipts[@]}"',
        1,
    )[0]
    plan = {
        "images": [{"id": "image-amd64"}],
        "families": [
            {
                "members": [
                    {
                        "image_id": "image-amd64",
                        "reference": "ghcr.io/release-org/runtime:member-amd64",
                    }
                ]
            }
        ],
    }
    receipt = {
        "kind": "ucm-image-member-receipt",
        "schema_version": 1,
        "id": "image-amd64",
        "status": "published",
        "targets": [
            {
                "channel": "ghcr",
                "reference": "ghcr.io/release-org/runtime:member-amd64",
                "digest": "sha256:" + "1" * 64,
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def validate(value: dict[str, object]) -> subprocess.CompletedProcess[str]:
        receipt_path.write_text(json.dumps(value), encoding="utf-8")
        return subprocess.run(
            [
                "jq",
                "-e",
                "-s",
                "--slurpfile",
                "plan",
                str(plan_path),
                jq_filter,
                str(receipt_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert validate(receipt).returncode == 0
    wrong = json.loads(json.dumps(receipt))
    wrong["targets"][0]["reference"] = "ghcr.io/attacker/wrong:tag"
    assert validate(wrong).returncode != 0


def test_release_index_matrix_generation_fails_closed() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    plan_script = next(
        step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan"
    )

    assert 'index_matrix="$(jq -ce' in plan_script
    assert 'echo "image_index_matrix=${index_matrix}"' in plan_script
    assert 'echo "has_image_indexes=$(jq -r' in plan_script
    assert "image_index_matrix=$(jq" not in plan_script


def test_open_release_is_the_only_job_that_publicizes_release() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    publicizers = []
    for name, job in jobs.items():
        if "draft=false" in yaml.safe_dump(job):
            publicizers.append(name)
    assert publicizers == ["open-release"]
    assert "release-open" in yaml.safe_dump(jobs["open-release"])
    assert "release.py finalize" in yaml.safe_dump(jobs["update-release-images"])


def test_remote_writers_use_environment_and_minimum_permissions() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    for name in (
        "open-release",
        "publish-release-artifacts",
        "publish-image-members",
        "publish-image-indexes",
        "publish-pypi",
        "publish-chart-oci",
        "update-release-images",
    ):
        assert jobs[name]["environment"] == "release-production"
        assert jobs[name]["permissions"]["contents"] in {"read", "write"}
    assert "packages" not in jobs["publish-pypi"]["permissions"]
    assert "release_kind != 'draft'" in jobs["publish-pypi"]["if"]
    assert jobs["publish-chart-oci"]["permissions"]["packages"] == "write"
    assert "packages" not in jobs["publish-image-members"]["permissions"]
    assert jobs["build-images"]["permissions"]["packages"] == "write"


def test_builder_sync_consumes_selection_and_uses_digest_pinned_mirror_only() -> None:
    workflow = _load("sync-builders.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "runtime_selection_artifact",
    }
    text = (WORKFLOWS / "sync-builders.yml").read_text(encoding="utf-8")
    assert "--selection input/upstreams/runtime-selection.json" in text
    assert "matrix.source_image" in text
    assert "matrix.source_image_digest" in text
    assert 'pinned_source="${source_repository}@${SOURCE_IMAGE_DIGEST}"' in text
    assert 'test "cann-${actual_cann}" = "${EXPECTED_RUNTIME}"' in text
    assert 'test "${SOC_VERSION}" = "${EXPECTED_SOC}"' in text
    assert "Dockerfile.builder-mirror" in text
    assert '--build-arg "BASE_IMAGE=${pinned_source}"' in text
    assert "builder-finalize-error.log" in text
    assert "builder-finalize-failure.json" in text
    assert "ucm-builder-finalize-failure-run-${{ github.run_id }}" in text
    assert "matrix.source_repository" not in text
    for forbidden in (
        "source_ref",
        "source_commit",
        "recipe-extend",
        "materialize-recipe",
        "REQUIRE_MOONCAKE",
        "MOONCAKE_TAG",
    ):
        assert forbidden not in text
    assert (
        workflow["jobs"]["build-missing"]["continue-on-error"]
        == "${{ matrix.checks.blocking != true }}"
    )
    assert "builders labels" in text
    assert "builders finalize" in text
    assert 'target_digest="$(docker buildx imagetools inspect "${target}"' in text
    assert 'verified_target="${TARGET_REPOSITORY}@${target_digest}"' in text
    assert '"${verified_target}" bash -c' in text
    assert "docker image inspect" in text
    assert "ucm-builder-verification-${{ matrix.id }}" in text
    assert "candidate-${GITHUB_RUN_ID}" in text
    assert 'imagetools create --tag "${target}" "${candidate}"' in text
    assert workflow["jobs"]["build-missing"]["timeout-minutes"] == 180
    assert set(workflow["jobs"]["finalize"]["needs"]) == {
        "prepare",
        "build-missing",
    }

    release_docker = ROOT / ".github" / "release" / "docker"
    for retired in (
        "Dockerfile.builder",
        "gflags-config.cmake",
        "mooncake_installer.sh",
    ):
        assert not (release_docker / retired).exists()


def test_runtime_probe_is_one_shared_reusable_workflow() -> None:
    workflow = _load("_probe-runtime.yml")

    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "probe_id",
        "runtime_ref",
        "image_reference",
        "platform",
        "accelerator",
        "runner",
        "enabled",
        "retention_days",
    }
    text = (WORKFLOWS / "_probe-runtime.yml").read_text(encoding="utf-8")
    assert "runtime-probe-raw.json" in text
    assert "ucm-runtime-probe-${{ inputs.probe_id }}" in text


def test_wheel_build_records_auditwheel_result_manifest() -> None:
    text = (WORKFLOWS / "_build-wheel.yml").read_text(encoding="utf-8")
    assert '.builder.repository + "@" + .builder.digest' in text
    assert "auditwheel==" in text
    assert "python -m auditwheel -v show" in text
    assert 'show "${wheel}" 2>&1' in text
    assert "compact record-wheel-result" in text
    assert "out/wheel/wheel-result.json" in text


def test_reusable_builds_keep_functional_inputs() -> None:
    expected = {
        "_build-wheel.yml": {"wheel_id", "runner", "plan_artifact", "source_ref"},
        "_build-image.yml": {
            "image_id",
            "runner",
            "plan_artifact",
            "upload_oci",
            "source_ref",
        },
        "_build-release-image.yml": {"image_id", "runner", "plan_artifact"},
        "_build-chart.yml": {"plan_artifact", "source_ref"},
    }
    for filename, inputs in expected.items():
        workflow = _load(filename)
        assert set(workflow["on"]["workflow_call"]["inputs"]) == inputs


def test_compact_wheel_passes_dynamic_python_and_platform_to_build() -> None:
    workflow = (WORKFLOWS / "_build-wheel.yml").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")
    assert "UCM_PYTHON_VERSION" in workflow
    assert "UCM_PYTHON_ABI" in workflow
    assert "UCM_PLATFORM" in workflow
    assert "ARG UCM_PYTHON_VERSION" in dockerfile
    assert "ARG UCM_PYTHON_ABI" in dockerfile
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in dockerfile
    assert 'sysconfig.get_path("scripts")' in dockerfile
    assert 'PATH="${python_scripts}:${PATH}"' in dockerfile


def test_compact_wheel_uses_source_metadata_and_active_a3_arch_handoff() -> None:
    workflow = (WORKFLOWS / "_build-wheel.yml").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "compact prepare-wheel-source" in workflow
    assert "--distribution \"$(jq -r '.dist_name'" in workflow
    assert 'version_path = os.path.join(ROOT_DIR, "version.ini")' in setup_py
    assert 'key == "VLLM_UC_VERSION"' in setup_py
    assert "version=get_package_version()" in setup_py

    combined = workflow + dockerfile + setup_py
    assert "UCM_BUILD_CONFIG" not in combined
    assert "UCM_DIST_NAME" not in combined

    assert "--build-arg \"UCM_CPU_ARCH=$(jq -r '.cpu_arch'" in workflow
    assert "ARG UCM_CPU_ARCH" in dockerfile
    assert 'UCM_BUILD_CPU_ARCH="${UCM_CPU_ARCH}"' in dockerfile
    assert 'os.getenv("UCM_BUILD_CPU_ARCH")' in setup_py
    assert "-DASCEND_ARCH_DIR=" in setup_py


def test_release_build_keeps_gcc_fmt_false_positive_non_fatal() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'CMAKE_BUILD_TYPE_LOWER STREQUAL "release"' in cmake
    assert 'CMAKE_CXX_COMPILER_ID STREQUAL "GNU"' in cmake
    assert "-Wno-error=stringop-overflow" in cmake


def test_chart_consumes_product_smoke_values_from_v4_policy() -> None:
    text = (WORKFLOWS / "_build-chart.yml").read_text(encoding="utf-8")
    assert ".chart.smoke_values[$product]" in text
    assert "huawei.com/Ascend910" in text
    assert "nvidia.com/gpu" in text
    assert "validation_cases" not in text


def test_ucm_build_bot_uses_probe_pipeline_without_hardcoded_capabilities() -> None:
    workflow = _load("ucm-build-bot.yml")
    assert set(workflow["on"]) == {"issue_comment", "workflow_dispatch"}
    assert set(workflow["on"]["workflow_dispatch"]["inputs"]) == {
        "pr_number",
        "subcommand",
        "image_ref",
        "profile",
    }
    jobs = workflow["jobs"]
    assert jobs["inspect-formal-runtimes"]["needs"] == "permission-check"
    assert set(jobs["probe-formal-runtimes"]["needs"]) == {
        "permission-check",
        "inspect-formal-runtimes",
    }
    assert set(jobs["resolve-formal"]["needs"]) == {
        "permission-check",
        "inspect-formal-runtimes",
        "probe-formal-runtimes",
    }
    assert jobs["probe-formal-runtimes"]["strategy"]["fail-fast"] is False
    assert "has_probe_fallback" not in jobs["probe-formal-runtimes"]["if"]
    assert "probe-formal-runtimes.result == 'skipped'" in jobs["resolve-formal"]["if"]
    assert set(
        jobs["inspect-runtimes"]["needs"]
        if isinstance(jobs["inspect-runtimes"]["needs"], list)
        else [jobs["inspect-runtimes"]["needs"]]
    ) == {"permission-check"}
    assert set(jobs["probe-runtimes"]["needs"]) == {
        "permission-check",
        "inspect-runtimes",
    }
    assert set(jobs["resolve-pr-runtimes"]["needs"]) == {
        "permission-check",
        "inspect-runtimes",
        "probe-runtimes",
    }
    assert set(jobs["sync-pr-builders"]["needs"]) == {
        "permission-check",
        "resolve-pr-runtimes",
    }
    assert set(jobs["plan-image"]["needs"]) == {
        "permission-check",
        "resolve-pr-runtimes",
        "sync-pr-builders",
    }
    assert jobs["probe-runtimes"]["strategy"]["fail-fast"] is False
    assert "subcommand == 'image'" in jobs["probe-runtimes"]["if"]
    assert "probe-runtimes.result == 'skipped'" in jobs["resolve-pr-runtimes"]["if"]
    assert "always()" in jobs["build-wheels"]["if"]
    assert "select-plan.result == 'success'" in jobs["build-wheels"]["if"]
    assert "build-wheels.result == 'success'" in jobs["build-images"]["if"]
    text = (WORKFLOWS / "ucm-build-bot.yml").read_text(encoding="utf-8")
    hint = (WORKFLOWS / "ucm-build-hint.yml").read_text(encoding="utf-8")
    assert "runtime inspect" in text
    assert "runtime aggregate" in text
    assert "--pr-default" in (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    assert "builders scan-registry" in text
    assert "runtime resolve" in text
    assert "upstreams candidates" in text
    assert text.count("./.github/workflows/_probe-runtime.yml") == 2
    assert "opaque" in hint
    assert "pep440" not in hint.lower()
    assert "cann900" not in text + hint
    assert "--runtime-selection" in text
    assert "profile_id==$profile" in text


def test_pr_member_and_index_publication_are_separate_dynamic_matrices() -> None:
    jobs = _load("ucm-build-bot.yml")["jobs"]
    members = jobs["publish-pr-image-members"]
    indexes = jobs["publish-pr-image-indexes"]
    assert members["strategy"]["fail-fast"] is False
    assert indexes["strategy"]["fail-fast"] is False
    assert (
        members["strategy"]["matrix"]
        == "${{ fromJSON(needs.select-plan.outputs.image_matrix) }}"
    )
    assert (
        indexes["strategy"]["matrix"]
        == "${{ fromJSON(needs.select-plan.outputs.index_matrix) }}"
    )
    assert "has_indexes == 'true'" in indexes["if"]
    assert "always()" in members["if"]
    assert "build-images.result == 'success'" in members["if"]
    member_step = members["steps"][-1]["run"]
    assert ".families[]" in member_step
    assert ".members[]" in member_step
    assert '"docker://${target_ref}"' in member_step
    assert "${tag}-${arch}" not in member_step
    index_step = indexes["steps"][-1]["run"]
    assert "matrix.members" not in index_step
    assert "jq -er '.[]'" in index_step
    assert '"${member_refs[@]}"' in index_step
    assert "-amd64" not in index_step
    assert "-arm64" not in index_step


def test_pr_receipt_is_always_posted_and_has_no_package_write_permission() -> None:
    receipt = _load("ucm-build-bot.yml")["jobs"]["post-build-receipt"]
    assert "always()" in receipt["if"]
    assert receipt["permissions"] == {
        "actions": "read",
        "contents": "read",
        "pull-requests": "write",
    }
    text = yaml.safe_dump(receipt)
    assert "runtime receipt" in text
    assert "pr-resolution.json" in text
    assert "sync-pr-builders" in receipt["needs"]
    assert "ucm-builder-finalize-failure-run-${{ github.run_id }}" in text
    workflow_text = (WORKFLOWS / "ucm-build-bot.yml").read_text(encoding="utf-8")
    assert "builder=${{ needs.sync-pr-builders.result }}" in workflow_text


def test_bot_control_plane_is_trusted_while_builds_use_pr_source() -> None:
    jobs = _load("ucm-build-bot.yml")["jobs"]
    for name in (
        "inspect-formal-runtimes",
        "resolve-formal",
        "inspect-runtimes",
        "resolve-pr-runtimes",
        "plan-formal",
        "plan-image",
        "post-build-receipt",
    ):
        checkouts = [
            step
            for step in jobs[name]["steps"]
            if step.get("uses") == "actions/checkout@v4.2.2"
        ]
        assert checkouts
        assert all("ref" not in step.get("with", {}) for step in checkouts)
    assert jobs["probe-runtimes"]["uses"] == "./.github/workflows/_probe-runtime.yml"
    assert jobs["probe-formal-runtimes"]["uses"] == (
        "./.github/workflows/_probe-runtime.yml"
    )
    assert (
        jobs["build-wheels"]["with"]["source_ref"]
        == "${{ needs.select-plan.outputs.source_ref }}"
    )
    assert (
        jobs["build-images"]["with"]["source_ref"]
        == "${{ needs.select-plan.outputs.source_ref }}"
    )


def test_bot_all_retags_and_publishes_without_formal_tag_collision() -> None:
    jobs = _load("ucm-build-bot.yml")["jobs"]
    plan_text = yaml.safe_dump(jobs["plan-formal"])
    assert "compact retag-pr" in plan_text
    assert "index_matrix=${indexes}" in plan_text
    assert "publish-pr-image-members" in jobs
    assert "publish-pr-image-indexes" in jobs


def test_runtime_fallback_probe_pulls_only_when_crane_facts_are_missing() -> None:
    probe = _load("_probe-runtime.yml")["jobs"]["probe"]
    uses = [step.get("uses") for step in probe["steps"]]
    assert "jlumbroso/free-disk-space@v1.3.1" in uses
    free_disk = next(
        step
        for step in probe["steps"]
        if step.get("uses") == "jlumbroso/free-disk-space@v1.3.1"
    )
    assert free_disk["if"] == "${{ inputs.enabled }}"
    no_op = next(
        step
        for step in probe["steps"]
        if step.get("name") == "Accept complete Crane config facts"
    )
    assert no_op["if"] == "${{ !inputs.enabled }}"
    upload = next(
        step
        for step in probe["steps"]
        if step.get("uses") == "actions/upload-artifact@v4.6.2"
    )
    assert "always()" in upload["if"]
    assert upload["with"]["path"] == "out/"
    text = (WORKFLOWS / "_probe-runtime.yml").read_text(encoding="utf-8")
    assert 'docker pull --platform "${PLATFORM}" "${IMAGE_REFERENCE}"' in text
    assert "has_numeric_version" in text
    assert 'grep -Eq "[0-9]+\\.[0-9]+"' in text
    assert '! has_numeric_version "${cuda_version}"' in text
    assert '! has_numeric_version "${cann_version}"' in text
    assert "nvcc --version" in text
    assert "version.info" in text
    for workflow_name in ("release-ucm.yml", "ucm-build-bot.yml"):
        workflow_text = (WORKFLOWS / workflow_name).read_text(encoding="utf-8")
        assert "has_probe_fallback" in workflow_text
        assert "probe_matrix.include | length > 0" in workflow_text
        assert "else {include:[.members[0]]}" in workflow_text
    image_build = (WORKFLOWS / "_build-image.yml").read_text(encoding="utf-8")
    assert ".runtime.image_reference" in image_build
    assert '.runtime.repository + ":" + .runtime.tag' not in image_build


def test_runtime_image_checks_wheel_glibc_floor_and_import() -> None:
    workflow = _load("_build-image.yml")
    steps = workflow["jobs"]["build"]["steps"]
    verify_index, verify = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Verify runtime glibc, Python, OS, and UCM import"
    )
    upload_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses") == "actions/upload-artifact@v4.6.2"
    )
    run = verify["run"]
    assert verify_index < upload_index
    assert "input/wheel/wheel-result.json" in run
    assert ".glibc_floor" in run
    assert "EXPECTED_WHEEL_GLIBC_FLOOR" in run
    assert "ldd --version" in run
    assert "parse(sys.argv[1]) >= parse(sys.argv[2])" in run
    assert 'python3 -c "import ucm"' in run
    assert 'EXPECTED_OS_ID}" != linux' in run
    assert 'EXPECTED_OS_VERSION}" != unreported' in run

    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.runtime"
    ).read_text(encoding="utf-8")
    assert "python3 -m pip install" in dockerfile
    assert "python3 -c 'import ucm'" in dockerfile


def test_trusted_tag_image_publishes_directly_without_oci_artifact() -> None:
    workflow = _load("_build-release-image.yml")
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "image_id",
        "runner",
        "plan_artifact",
    }
    job = workflow["jobs"]["publish"]
    assert job["environment"] == "release-production"
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    text = (WORKFLOWS / "_build-release-image.yml").read_text(encoding="utf-8")
    assert "source_ref" not in text
    assert 'test "${GITHUB_EVENT_NAME}" = push' in text
    assert 'test "${GITHUB_REF_TYPE}" = tag' in text
    assert 'skopeo copy "oci-archive:out/image.oci.tar"' in text
    assert "ucm-image-member-receipt-${{ inputs.image_id }}" in text
    assert "path: out/image.oci.tar" not in text
    release = _load("release-ucm.yml")["jobs"]
    assert release["build-images"]["uses"] == (
        "./.github/workflows/_build-release-image.yml"
    )
    assert release["build-validation-images"]["uses"] == (
        "./.github/workflows/_build-image.yml"
    )


def test_cross_job_artifact_names_survive_failed_job_reruns() -> None:
    names = (
        "release-ucm.yml",
        "_probe-runtime.yml",
        "sync-builders.yml",
        "_build-wheel.yml",
        "_build-image.yml",
        "_build-release-image.yml",
        "_build-chart.yml",
        "ucm-build-bot.yml",
    )
    text = "\n".join((WORKFLOWS / name).read_text(encoding="utf-8") for name in names)
    assert "github.run_attempt" not in text
    assert "GITHUB_RUN_ATTEMPT" in text  # candidate Builder tags remain retry-scoped.


def test_a5_issue_is_isolated_to_formal_tag_workflow() -> None:
    issue = _load("release-capability-issue.yml")
    assert issue["on"] == {"push": {"tags": ["v*"]}}
    assert issue["permissions"] == {
        "actions": "read",
        "contents": "read",
        "issues": "write",
    }
    assert issue["concurrency"]["cancel-in-progress"] is True
    assert "issues" not in _load("release-ucm.yml")["permissions"]
    assert "issues" not in _load("ucm-build-bot.yml")["permissions"]
    text = (WORKFLOWS / "release-capability-issue.yml").read_text(encoding="utf-8")
    assert "problems render" in text
    assert "ucm-runtime-selection-run-${run_id}" in text
    assert "upstreams resolve" not in text
    assert "gh issue create" in text
    assert "gh issue close" in text
