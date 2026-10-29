"""User-visible Actions contracts for upstream planning and parallel publishing."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str) -> dict[str, object]:
    value = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    if True in value and "on" not in value:
        value["on"] = value.pop(True)
    return value


def test_runtime_images_install_the_same_run_toolkit_wheel_by_default() -> None:
    dockerfile = (ROOT / ".github/release/docker/Dockerfile.runtime").read_text(
        encoding="utf-8"
    )
    assert "/tmp/${UCM_WHEEL} /tmp/${UCM_TOOLKIT_WHEEL}" in dockerfile
    assert "&& ucm-toolkit list" in dockerfile
    for name in ("_build-image.yml", "_build-release-image.yml"):
        workflow = _load(name)
        steps = next(iter(workflow["jobs"].values()))["steps"]
        download = next(
            step
            for step in steps
            if step.get("with", {}).get("path") == "input/toolkit"
        )
        assert download["with"]["name"] == "ucm-toolkit-run-${{ github.run_id }}"
        build = next(
            step["run"] for step in steps if "Dockerfile.runtime" in step.get("run", "")
        )
        assert 'cp "${toolkit_wheel}"' in build
        assert '--build-arg "UCM_TOOLKIT_WHEEL=' in build
    release = _load("release-ucm.yml")["jobs"]
    assert "build-toolkit" in release["publish-release-artifacts"]["needs"]
    bot = _load("ucm-build-bot.yml")["jobs"]
    assert bot["build-toolkit"]["uses"] == "./.github/workflows/_build-toolkit.yml"
    assert bot["build-toolkit"]["with"] == {
        "source_ref": "${{ needs.select-plan.outputs.source_ref }}",
        "plan_artifact": "${{ needs.select-plan.outputs.plan_artifact }}",
    }
    assert "build-toolkit" in bot["build-images"]["needs"]
    assert "needs.build-toolkit.result == 'success'" in bot["build-images"]["if"]


def test_chart_filters_families_without_a_second_plan_artifact() -> None:
    release = _load("release-ucm.yml")["jobs"]
    plan = release["plan"]
    assert "chart_plan_artifact" not in plan["outputs"]
    assert release["package-chart"]["with"]["plan_artifact"] == (
        "${{ needs.plan.outputs.plan_artifact }}"
    )
    chart = _load("_build-chart.yml")["jobs"]["package"]
    package_step = next(step for step in chart["steps"] if step.get("id") == "package")
    assert "plan=input/plan/release-plan.json" in package_step["run"]
    assert (
        'select(.product_id == "vllm" or .product_id == "vllm-ascend")'
        in package_step["run"]
    )


def test_nightly_schedule_creates_or_reuses_a_tag_then_calls_core_in_same_run() -> None:
    workflow = _load("release-nightly.yml")
    assert workflow["on"]["schedule"] == [{"cron": "0 18 * * *"}]
    assert workflow["concurrency"] == {
        "group": "ucm-nightly-${{ github.repository_id }}",
        "queue": "max",
        "cancel-in-progress": False,
    }
    jobs = workflow["jobs"]
    prepare = jobs["prepare-nightly"]
    assert "github.ref == 'refs/heads/develop'" in prepare["if"]
    assert (
        "github.repository == 'ModelEngine-Group/unified-cache-management'"
        in prepare["if"]
    )
    assert prepare["permissions"] == {"contents": "write"}
    checkout = next(
        step
        for step in prepare["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    run = next(step["run"] for step in prepare["steps"] if step.get("id") == "prepare")
    assert "TZ=Asia/Shanghai date +%Y%m%d" in run
    assert "--next-nightly" in run
    assert "--version-config version.ini" in run
    assert "next_patch_version" not in run
    assert 'git rev-list -n 1 "${candidate}"' in run
    assert "bash .github/release/find-github-release.sh" in run
    assert '"${GH_REPO}" "${candidate}"' in run
    assert ".draft == false and .prerelease == true" in run
    assert 'any(.name == "release-manifest.json")' in run
    assert 'gh api --method POST "/repos/${GH_REPO}/git/refs"' in run
    assert '-f "ref=refs/tags/${selected_tag}"' in run
    assert 'test "$(git rev-list -n 1 "${selected_tag}")" = "${source_sha}"' in run

    release = jobs["release"]
    assert release["needs"] == "prepare-nightly"
    assert release["uses"] == "./.github/workflows/release-ucm.yml"
    assert set(release["with"]) == {
        "git_tag",
        "release_type",
        "version",
        "chart_version",
        "image_version",
        "release_kind",
        "is_prerelease",
        "source_sha",
        "publication_scope",
    }
    assert release["with"]["source_sha"] == (
        "${{ needs.prepare-nightly.outputs.source_sha }}"
    )
    assert "github.event_name == 'workflow_dispatch'" in prepare["if"]
    assert release["with"]["publication_scope"] == (
        "${{ github.repository == 'ModelEngine-Group/unified-cache-management' "
        "&& 'official' || 'fork' }}"
    )
    assert release["secrets"] == {
        "PYPI_API_TOKEN": (
            "${{ github.repository == 'ModelEngine-Group/unified-cache-management' "
            "&& secrets.PYPI_API_TOKEN || '' }}"
        ),
        "TEST_PYPI_API_TOKEN": (
            "${{ github.repository != 'ModelEngine-Group/unified-cache-management' "
            "&& secrets.TEST_PYPI_API_TOKEN || '' }}"
        ),
        "DOCKERHUB_USERNAME": "${{ secrets.DOCKERHUB_USERNAME }}",
        "DOCKERHUB_TOKEN": "${{ secrets.DOCKERHUB_TOKEN }}",
        "RTD_API_TOKEN": "${{ secrets.RTD_API_TOKEN }}",
    }


def test_nightly_wheels_do_not_inherit_the_skipped_builder_sync() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    for name in ("build-wheels", "build-toolkit", "build-meta"):
        assert (
            jobs[name]["if"] == "${{ !cancelled() && needs.plan.result == 'success' }}"
        )
    validation = jobs["validate-wheel-runtimes"]
    assert "!cancelled()" in validation["if"]
    assert "needs.plan.result == 'success'" in validation["if"]
    assert "needs.build-wheels.result == 'success'" in validation["if"]


def test_schema_v9_manifest_is_uploaded_only_after_complete_and_read_back() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    steps = jobs["update-release-images"]["steps"]
    update_index, update = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("id") == "update-release"
    )
    complete_index, complete = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Require complete image publication"
    )
    manifest_index, manifest = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("id") == "publish-manifest"
    )

    assert update_index < complete_index < manifest_index
    assert "release-state.json" in update["run"]
    assert "release-manifest.json" not in update["run"]
    assert "gh release upload" not in update["run"]
    assert not any(step.get("id") == "publish-pypi-receipt" for step in steps)
    assert "release.status" in complete["run"]
    assert "release.py manifest" in manifest["run"]
    assert 'gh release upload "${tag}" --clobber out/release/release-manifest.json' in (
        manifest["run"]
    )
    assert "release-manifest-readback.json" in manifest["run"]
    assert "--output out/release/public-readback" in manifest["run"]
    assert "out/release/public-readback/release-manifest.json" in manifest["run"]
    assert "Accept: application/octet-stream" in manifest["run"]
    assert "cmp out/release/release-manifest.json" in manifest["run"]
    assert "github_release_assets" in manifest["run"]
    assert "--rawfile notes out/release/release-notes.md" in manifest["run"]
    assert "out/release/readback/release-notes.md" in manifest["run"]
    assert "cmp out/release/release-notes.md" in manifest["run"]
    manifest_upload = manifest["run"].index("gh release upload")
    assert (
        manifest_upload
        < manifest["run"].index('-f "tag_name=${tag}"', manifest_upload)
        < manifest["run"].index("release.py notes")
    )

    for name, job in jobs.items():
        if name != "update-release-images":
            assert "gh release upload" not in yaml.safe_dump(job) or (
                name == "publish-release-artifacts"
                and "release-manifest.json" not in yaml.safe_dump(job)
            )


def test_image_failure_notes_are_not_overwritten_by_the_fallback() -> None:
    steps = _load("release-ucm.yml")["jobs"]["update-release-images"]["steps"]
    update = next(step for step in steps if step.get("id") == "update-release")
    manifest = next(step for step in steps if step.get("id") == "publish-manifest")
    require = next(
        step
        for step in steps
        if step.get("name") == "Require complete image publication"
    )
    fallback = steps[-1]

    assert "release.status" not in update["run"]
    assert "release.py notes" not in update["run"]
    assert 'gh api "/repos/${GH_REPO}/releases/${release_id}"' in manifest["run"]
    assert "release.py notes" in manifest["run"]
    assert "release.status" in require["run"]
    assert "steps.publish-manifest.outcome != 'success'" in fallback["if"]
    assert "publication-failed" in fallback["run"]
    assert 'if [ "${RELEASE_TYPE}" = nightly ]; then' in fallback["run"]
    assert "-F draft=true -F prerelease=true" in fallback["run"]


def test_artifact_upload_restores_tag_before_generating_asset_links() -> None:
    steps = _load("release-ucm.yml")["jobs"]["publish-release-artifacts"]["steps"]
    run = next(
        step["run"]
        for step in steps
        if step.get("name") == "Upload backend Wheels, Chart, and Config"
    )

    upload = run.index("gh release upload")
    assert (
        upload
        < run.index('-f "tag_name=${tag}"', upload)
        < run.index("release.py notes")
    )
    assert "out/release/artifact-readback/release-notes.md" in run
    assert 'sub("\\\\n+$"; "")' in run


def test_release_workflow_checks_member_receipts() -> None:
    core = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")

    assert "release.py members" in core


def test_release_state_entrypoints_install_their_dependencies() -> None:
    for name, job in _load("release-ucm.yml")["jobs"].items():
        steps = job.get("steps", [])
        entrypoint = next(
            (
                index
                for index, step in enumerate(steps)
                if "ucm_release/release.py" in step.get("run", "")
            ),
            None,
        )
        if entrypoint is None:
            continue
        installs = "\n".join(
            step.get("run", "")
            for step in steps[:entrypoint]
            if "pip install" in step.get("run", "")
        )
        assert "-r .github/release/requirements.txt" in installs, name


def test_only_open_and_successful_nightly_finalize_can_publicize_release() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    publicizers = []
    for name, job in jobs.items():
        if "draft=false" in yaml.safe_dump(job):
            publicizers.append(name)
    assert publicizers == ["open-release", "update-release-images"]
    assert "release-open" in yaml.safe_dump(jobs["open-release"])
    finalizer = yaml.safe_dump(jobs["update-release-images"])
    assert "release.py finalize" in finalizer
    assert "release.py manifest" in finalizer
    assert "RELEASE_TYPE" in finalizer
    assert "draft=false" in finalizer
    assert "prerelease=true" in finalizer


def test_remote_writers_use_environment_and_minimum_permissions() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    scoped_environment = (
        "${{ inputs.publication_scope == 'official' && "
        "'release-production' || 'fork-preview' }}"
    )
    for name in (
        "open-release",
        "plan",
        "publish-release-artifacts",
        "publish-image-members",
        "publish-image-indexes",
        "publish-pypi",
        "publish-chart-oci",
        "update-release-images",
        "retain-releases",
    ):
        assert jobs[name]["environment"] == scoped_environment
        assert jobs[name]["permissions"]["contents"] in {"read", "write"}
    assert jobs["release-preflight"]["environment"] == scoped_environment
    assert jobs["release-preflight"]["permissions"] == {"contents": "read"}
    for name in ("open-release", "plan"):
        assert jobs[name]["permissions"] == {
            "actions": "read",
            "contents": "write",
        }
    assert "packages" not in jobs["publish-pypi"]["permissions"]
    assert jobs["publish-chart-oci"]["permissions"]["packages"] == "write"
    assert "packages" not in jobs["publish-image-members"]["permissions"]
    assert jobs["build-images"]["permissions"]["packages"] == "write"
    assert jobs["update-release-images"]["permissions"] == {
        "actions": "read",
        "contents": "write",
    }
    assert jobs["retain-releases"]["permissions"] == {
        "actions": "write",
        "contents": "write",
        "packages": "write",
    }

    for name in (
        "publish-image-members",
        "publish-image-indexes",
        "publish-chart-oci",
    ):
        channel_job = yaml.safe_dump(jobs[name])
        assert "release_kind" not in channel_job
        assert "release_type" not in channel_job
        assert "!= draft" not in channel_job
        assert "!= nightly" not in channel_job


def test_publication_scope_and_repository_ownership_fail_closed_at_writers() -> None:
    core = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    child = (WORKFLOWS / "_build-release-image.yml").read_text(encoding="utf-8")
    builders = (WORKFLOWS / "sync-builders.yml").read_text(encoding="utf-8")
    cleanup = (WORKFLOWS / "cleanup-ucm-release.yml").read_text(encoding="utf-8")

    assert ".publication_scope == $scope" in core
    assert ".repository | ascii_downcase" in core
    assert 'test "${GH_REPO}" = "${GITHUB_REPOSITORY}"' in core
    assert "Chart OCI target is outside current owner namespace" in core
    assert "GHCR index target is outside current owner namespace" in core
    assert '.publish.pypi.target == "pypi"' in core
    assert '.publish.pypi.target == "testpypi"' in core
    assert "Docker Hub namespace is outside docker.io" in core
    assert core.count("modelengine-group/unified-cache-management") >= 2

    assert ".publication_scope == $scope" in child
    assert "GHCR member target is outside current owner namespace" in child
    assert "Docker Hub member is outside the planned namespace" in child
    assert '"${namespace}/"*:*)' in child

    assert "[.builders[].target_repository] | all(startswith($prefix))" in builders
    assert "Builder target is outside current owner namespace" in builders
    assert "Builder candidate is outside current owner namespace" in builders

    cleanup_job = _load("cleanup-ucm-release.yml")["jobs"]["cleanup"]
    assert "release-production" in cleanup_job["environment"]
    assert "fork-preview" in cleanup_job["environment"]
    assert "Authenticate to DockerHub when configured" in cleanup
    assert "modelengine-group/unified-cache-management" in cleanup.lower()


def test_exact_wheels_pass_runtime_validation_before_publication() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    validation = jobs["validate-wheel-runtimes"]
    assert set(validation["needs"]) == {"plan", "build-wheels"}
    assert validation["strategy"]["matrix"] == (
        "${{ fromJSON(needs.plan.outputs.pypi_test_matrix) }}"
    )
    run = next(
        step["run"]
        for step in validation["steps"]
        if step.get("name")
        == "Install and validate the local Wheel in its matching Runtime"
    )
    assert run.strip() == (
        "bash .github/release/install-validate-wheel-runtime.sh input/wheel"
    )
    assert "validate-wheel-runtimes" in jobs["publish-release-artifacts"]["needs"]
    assert set(jobs["publish-pypi"]["needs"]) == {
        "plan",
        "publish-release-artifacts",
    }


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


def test_cross_job_artifact_names_survive_failed_job_reruns() -> None:
    names = (
        "release-ucm.yml",
        "_probe-runtime.yml",
        "sync-builders.yml",
        "_build-wheel.yml",
        "_build-image.yml",
        "_build-release-image.yml",
        "_build-chart.yml",
        "_native-wheel-gate.yml",
        "ucm-build-bot.yml",
    )
    text = "\n".join((WORKFLOWS / name).read_text(encoding="utf-8") for name in names)
    for name in names:
        workflow = _load(name)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                artifact = step.get("with", {}).get("name", "")
                if "github.run_attempt" in artifact:
                    # Evidence is scoped to an attempt; downstream build inputs stay run-scoped.
                    assert artifact.startswith("ucm-release-cleanup-")
    assert "GITHUB_RUN_ATTEMPT" in text  # candidate Builder tags remain retry-scoped.


def test_release_ownership_does_not_depend_on_runtime_discovery_success() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    opened = jobs["open-release"]
    assert opened["needs"] == "release-preflight"
    assert opened["if"] == "${{ needs.release-preflight.outputs.enabled == 'true' }}"
    assert jobs["select-runtime-candidates"]["needs"] == "release-preflight"
    assert set(jobs["inspect-runtimes"]["needs"]) == {
        "open-release",
        "select-runtime-candidates",
    }
    steps = opened["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "opened")
    create = run.index('gh release create "${args[@]}"')
    record = run.index("cleanup.py record-targets")
    enabled = run.index('echo "enabled=true"')
    assert create < record < enabled
    assert run.startswith("set -euo pipefail")
    assert 'if [ "${RELEASE_TYPE}" != nightly ]; then' not in run
    assert '--source-sha "${SOURCE_SHA}"' in run
    assert '--run-id "${GITHUB_RUN_ID}"' in run
    assert '--run-attempt "${GITHUB_RUN_ATTEMPT}"' in run
    assert '--release-id "${release_id}"' in run
    assert "--output-dir out/cleanup" in run
    assert "--plan" not in run
    assert opened["env"]["SOURCE_SHA"] == "${{ inputs.source_sha }}"
    assert any(
        ".github/release/requirements.txt" in step.get("run", "") for step in steps
    )
    assert "needs.open-release.result == 'success'" in jobs["plan"]["if"]


def test_plan_records_cleanup_targets_before_enabling_publishers() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    plan = jobs["plan"]
    step = next(step for step in plan["steps"] if step.get("id") == "plan")
    run = step["run"]
    validate = run.index("jq -e --arg tag")
    record = run.index("cleanup.py record-targets")
    outputs = run.index('echo "route=release"')
    assert validate < record < outputs
    assert '.route == "release" and .publish.github_release.enabled == true' in run
    assert 'if [ "${RELEASE_TYPE}" != nightly ] &&' not in run
    assert "--plan out/plan/release-plan.json" in run
    assert '--source-sha "${SOURCE_SHA}"' in run
    assert '--run-id "${GITHUB_RUN_ID}"' in run
    assert '--run-attempt "${GITHUB_RUN_ATTEMPT}"' in run
    assert '--release-id "${RELEASE_ID}"' in run
    assert "--output-dir out/cleanup" in run
    assert step["env"]["RELEASE_ID"] == "${{ needs.open-release.outputs.release_id }}"
    assert run.startswith("set -euo pipefail")
    assert not step.get("continue-on-error", False)
    assert plan["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "GH_REPO": "${{ github.repository }}",
        "SOURCE_SHA": "${{ inputs.source_sha }}",
    }
    assert 'test "${GH_REPO}" = "${GITHUB_REPOSITORY}"' in run
    assert "needs.plan.result == 'success'" in jobs["build-images"]["if"]
    assert "needs.plan.result == 'success'" in jobs["publish-release-artifacts"]["if"]
    # Chart/PyPI outputs are absent if writing the cleanup record fails first.
    assert 'echo "publish_chart_oci=' in run[outputs:]
    assert 'echo "publish_pypi=' in run[outputs:]


def test_retention_runs_after_failed_planning_artifacts_or_finalization() -> None:
    workflow = _load("release-ucm.yml")
    jobs = workflow["jobs"]
    cleanup = jobs["retain-releases"]
    assert cleanup["concurrency"] == {
        "group": (
            "ucm-release-retention-${{ github.repository_id }}-${{ inputs.release_type }}"
        ),
        "queue": "max",
        "cancel-in-progress": False,
    }
    assert workflow["concurrency"] == {
        "group": "ucm-release-${{ github.repository_id }}-${{ inputs.git_tag }}",
        "cancel-in-progress": False,
    }
    # Only validated release inputs gate retention: failed or skipped downstream
    # jobs, and a missing cleanup manifest, do not suppress this job.
    assert " ".join(cleanup["if"].split()) == (
        "${{ always() && inputs.release_type != 'nightly' && "
        "needs.release-preflight.result == 'success' && needs.release-preflight.outputs.enabled == 'true' }}"
    )
    retention_steps = [
        (name, step)
        for name, job in jobs.items()
        for step in job.get("steps", [])
        if "cleanup.py retention" in step.get("run", "")
    ]
    assert len(retention_steps) == 1
    name, retention = retention_steps[0]
    assert name == "retain-releases"
    assert "cleanup.py retention" in retention["run"]
    assert ".release_profile.max_count" in retention["run"]
    assert 'pypi_enabled="$(jq -r' in retention["run"]
    assert 'pypi_enabled="$(jq -er' not in retention["run"]
    assert '<<<"${resolved}"' in retention["run"]
    assert "fork_test_pypi=sys.argv[2]" in retention["run"]
    assert "dockerhub_namespace=sys.argv[3]" in retention["run"]
    assert "release-plan.json" not in yaml.safe_dump(cleanup)
    assert "release-manifest.json" not in yaml.safe_dump(cleanup)
    assert "actions/download-artifact@" not in yaml.safe_dump(cleanup)
    assert cleanup["permissions"] == {
        "actions": "write",
        "contents": "write",
        "packages": "write",
    }
    assert cleanup["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert cleanup["env"]["DOCKERHUB_TOKEN"] == "${{ secrets.DOCKERHUB_TOKEN }}"


def test_retention_waits_for_resource_writers_and_failure_reports() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    pending = list(jobs["retain-releases"]["needs"])
    ancestors = set()
    while pending:
        name = pending.pop()
        if name in ancestors:
            continue
        ancestors.add(name)
        dependencies = jobs[name].get("needs", [])
        pending.extend(
            [dependencies] if isinstance(dependencies, str) else dependencies
        )
    assert {
        "open-release",
        "plan",
        "report-planning-failure",
        "sync-builders",
        "build-images",
        "publish-release-artifacts",
        "publish-image-indexes",
        "publish-pypi",
        "publish-chart-oci",
        "update-release-images",
        "verify-release-docs",
        "verify-release-delivery",
    } <= ancestors
    # Retention failure is reported separately from channel finalization and delivery.
    for job in jobs.values():
        assert "retain-releases" not in job.get("needs", [])


def test_manual_cleanup_reuses_the_release_cleanup_module() -> None:
    workflow = _load("cleanup-ucm-release.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert set(workflow["on"]["workflow_dispatch"]["inputs"]) == {
        "tag",
        "fail_resource",
    }
    job = workflow["jobs"]["cleanup"]
    assert job["permissions"] == {
        "actions": "write",
        "contents": "write",
        "packages": "write",
    }
    text = yaml.safe_dump(job)
    assert "cleanup.py" in text and "--fail-resource" in text
    assert "0 5 15" not in text  # retry policy has one Python owner.
    steps = job["steps"]
    setup_python = next(
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    dependencies = next(
        index
        for index, step in enumerate(steps)
        if ".github/release/requirements.txt" in step.get("run", "")
    )
    delete = next(
        index for index, step in enumerate(steps) if "cleanup.py" in step.get("run", "")
    )
    assert setup_python < dependencies < delete


def test_pr_gate_runs_the_native_wheel_matrix_behind_one_stable_check() -> None:
    workflow = _load("pull-request.yml")
    assert "feature/**" in workflow["on"]["pull_request"]["branches"]
    jobs = workflow["jobs"]
    pre_check = jobs["pre-check"]
    build = jobs["native-wheel-build"]
    gate = jobs["native-wheel-compile"]

    assert pre_check["outputs"]["native_wheel_required"] == (
        "${{ steps.build-impact.outputs.required }}"
    )
    impact = next(
        step for step in pre_check["steps"] if step.get("id") == "build-impact"
    )
    for build_input in (
        "CMakeLists.txt",
        "ucm/*",
        "setup.py",
        "pyproject.toml",
        "version.ini",
        ".github/release/ucm_release/*",
        ".github/release/docker/*",
        ".github/release/requirements/*",
        ".github/release/requirements*.txt",
        "requirements-build.txt",
        ".github/workflows/_build-wheel.yml",
        ".github/workflows/_native-wheel-gate.yml",
    ):
        assert build_input in impact["run"]
    assert build["needs"] == "pre-check"
    assert "native_wheel_required == 'true'" in build["if"]
    assert build["permissions"] == {"contents": "read", "packages": "read"}
    assert build["uses"] == "./.github/workflows/_native-wheel-gate.yml"
    assert build["with"]["source_ref"] == "${{ github.sha }}"
    assert set(gate["needs"]) == {"pre-check", "native-wheel-build"}
    assert gate["if"] == "${{ always() }}"
    assert gate["name"] == "native-wheel-compile"
    gate_text = yaml.safe_dump(gate)
    assert "PRE_CHECK_RESULT" in gate_text
    assert "BUILD_REQUIRED" in gate_text
    assert "BUILD_RESULT" in gate_text
    pc_e2e_if = jobs["test-e2e-pc-a2"]["if"]
    assert "github.repository == 'ModelEngine-Group/unified-cache-management'" in (
        pc_e2e_if
    )
    assert "github.event.pull_request.draft == false" in pc_e2e_if
    sglang_e2e = jobs["test-e2e-sglang-a2"]
    assert sglang_e2e["runs-on"] == ["self-hosted", "npu", "a2"]
    assert set(sglang_e2e["needs"]) == {"lint-and-unit-tests", "test-e2e-pc-a2"}
    assert "always()" in sglang_e2e["if"]
    assert "needs.lint-and-unit-tests.result == 'success'" in sglang_e2e["if"]
    assert "github.repository == 'ModelEngine-Group/unified-cache-management'" in (
        sglang_e2e["if"]
    )
    sglang_build = next(
        step for step in sglang_e2e["steps"] if step.get("name") == "Build"
    )
    assert 'no_proxy="repo.huaweicloud.com,${no_proxy:-}"' in sglang_build["run"]
    sglang_e2e_text = yaml.safe_dump(sglang_e2e)
    assert "Dockerfile.ucm-sglang-ascend.a2-v0.5.18" in sglang_e2e_text
    assert "--privileged" in sglang_e2e_text
    assert "--network=host" in sglang_e2e_text
    assert "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro" in (sglang_e2e_text)
    assert "/usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro" in (
        sglang_e2e_text
    )
    assert "RUNNER_NAME" in sglang_e2e_text
    assert "Host NPU status" in sglang_e2e_text
    assert "torch_npu.npu.is_available()" in sglang_e2e_text
    assert "sglang_online_inference" in sglang_e2e_text
    assert "test_sglang_online_accuracy_from_persisted_cache_a2" in (sglang_e2e_text)

    text = (WORKFLOWS / "pull-request.yml").read_text(encoding="utf-8")
    assert "./.github/workflows/release-ucm.yml" not in text


def test_pr_gate_skips_work_per_job_instead_of_filtering_the_workflow() -> None:
    workflow = _load("pull-request.yml")
    trigger = workflow["on"]["pull_request"]
    # Workflow-level path filtering would leave required checks Pending on a
    # skipped run, so irrelevant work is skipped per job via `if:` instead.
    assert "paths" not in trigger and "paths-ignore" not in trigger
    assert "ready_for_review" in trigger["types"]

    jobs = workflow["jobs"]
    assert jobs["pre-check"]["outputs"]["code_changed"] == (
        "${{ steps.build-impact.outputs.code_changed }}"
    )
    impact = next(
        step for step in jobs["pre-check"]["steps"] if step.get("id") == "build-impact"
    )
    assert "code_changed=${code_changed}" in impact["run"]

    assert jobs["lint-and-unit-tests"]["if"] == (
        "needs.pre-check.outputs.code_changed == 'true'"
    )
    for name in (
        "native-wheel-build",
        "test-e2e-pc-a2",
        "test-e2e-sglang-a2",
    ):
        assert "github.event.pull_request.draft == false" in jobs[name]["if"], name


def test_bilingual_check_runs_without_a_path_filter() -> None:
    trigger = _load("docs-bilingual.yml")["on"]["pull_request"]
    assert "paths" not in trigger and "paths-ignore" not in trigger


def test_version_config_changes_trigger_docs_check_on_push_and_pr() -> None:
    workflow = _load("docs-check.yml")
    for event in ("push", "pull_request"):
        assert "version.ini" in workflow["on"][event]["paths"]


def test_native_wheel_gate_uses_source_builders_without_publication() -> None:
    workflow = _load("_native-wheel-gate.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "source_ref",
        "retention_days",
    }
    assert workflow["permissions"] == {"contents": "read", "packages": "read"}
    jobs = workflow["jobs"]
    assert set(jobs) == {
        "inspect-runtimes",
        "probe-runtimes",
        "resolve-upstreams",
        "plan-wheels",
        "build-wheels",
        "validate-fallback-runtimes",
    }
    assert jobs["probe-runtimes"]["uses"] == ("./.github/workflows/_probe-runtime.yml")
    assert jobs["build-wheels"]["strategy"]["fail-fast"] is False
    assert jobs["build-wheels"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.plan-wheels.outputs.wheel_matrix) }}"
    )
    assert jobs["build-wheels"]["uses"] == "./.github/workflows/_build-wheel.yml"
    assert jobs["build-wheels"]["permissions"] == {
        "contents": "read",
        "packages": "read",
    }
    text = (WORKFLOWS / "_native-wheel-gate.yml").read_text(encoding="utf-8")
    assert "builders bind-source" in text
    assert "plan create" in text and "--route pr" in text
    assert "sync-builders.yml" not in text
    assert "packages: write" not in text
    assert "docker push" not in text
    assert "open-release" not in text


def test_native_wheel_gate_requires_each_planned_fallback_runtime() -> None:
    jobs = _load("_native-wheel-gate.yml")["jobs"]
    plan = jobs["plan-wheels"]
    validation = jobs["validate-fallback-runtimes"]
    plan_step = next(step for step in plan["steps"] if step.get("id") == "plan")
    assert "runtime fallback-matrix" in plan_step["run"]
    assert "has_suffix_fallback" in plan["outputs"]
    assert set(validation["needs"]) == {"plan-wheels", "build-wheels"}
    assert "needs.plan-wheels.result == 'success'" in validation["if"]
    assert "needs.build-wheels.result == 'success'" in validation["if"]
    assert "needs.plan-wheels.outputs.has_suffix_fallback == 'true'" in validation["if"]
    assert validation["strategy"] == {
        "fail-fast": False,
        "matrix": "${{ fromJSON(needs.plan-wheels.outputs.fallback_runtime_matrix) }}",
    }
    assert validation["runs-on"] == "${{ matrix.runner }}"
    assert validation["permissions"] == {"contents": "read"}
    download = next(
        step
        for step in validation["steps"]
        if step.get("with", {}).get("path") == "input/wheel"
    )
    assert download["with"]["name"] == (
        "ucm-wheel-${{ matrix.wheel_id }}-run-${{ github.run_id }}"
    )
    step = next(
        step for step in validation["steps"] if step.get("id") == "runtime-validation"
    )
    assert step["run"].strip() == (
        "bash .github/release/install-validate-wheel-runtime.sh input/wheel"
    )
    assert step["env"]["RUNTIME_IMAGE"] == "${{ matrix.runtime_image }}"
    assert step["env"]["RUNTIME_PLATFORM"] == "linux/${{ matrix.cpu_arch }}"
    assert "continue-on-error" not in validation
    assert all("continue-on-error" not in step for step in validation["steps"])


def test_finalizer_downloads_pypi_receipt_only_when_publication_is_enabled() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    receipt = next(
        step
        for step in jobs["update-release-images"]["steps"]
        if step.get("with", {}).get("name")
        == "ucm-pypi-receipt-run-${{ github.run_id }}"
    )
    assert receipt["if"] == jobs["publish-pypi"]["if"]
    assert "needs.plan.outputs.publish_pypi == 'true'" in receipt["if"]
    # Failed publication still reaches finalization to record the failed channel.
    assert receipt["continue-on-error"] is True


def test_release_image_retries_each_enabled_profile_member_after_verification() -> None:
    steps = _load("_build-release-image.yml")["jobs"]["publish"]["steps"]
    step_names = [step.get("name") for step in steps]
    build = next(
        step for step in steps if step.get("name") == "Build install-only Runtime image"
    )
    verify = next(
        step
        for step in steps
        if step.get("name") == "Verify Runtime glibc, Python, OS, and UCM import"
    )
    reclaim = next(
        step
        for step in steps
        if step.get("name") == "Reclaim BuildKit cache before runtime verification"
    )
    publish = next(
        step
        for step in steps
        if step.get("name") == "Publish verified Profile members and record digests"
    )

    assert (
        step_names.index(build["name"])
        < step_names.index(reclaim["name"])
        < step_names.index(verify["name"])
        < step_names.index(publish["name"])
    )
    disk_cleanup = next(
        step for step in steps if step.get("uses") == "jlumbroso/free-disk-space@v1.3.1"
    )
    assert disk_cleanup["with"]["large-packages"] is True
    assert "docker buildx prune --all --force" in reclaim["run"]
    assert "publish_member()" in publish["run"]
    assert "publish_channel()" in publish["run"]
    assert "for attempt in 1 2 3" in publish["run"]
    assert 'publish_member "${channel}" "${reference}"' in publish["run"]
    assert "docker login ghcr.io" in publish["run"]
    assert "docker login docker.io" in publish["run"]
    assert 'skopeo copy "oci-archive:out/image.oci.tar"' in publish["run"]
    assert "retrying in ${sleep_seconds}s" in publish["run"]
    assert ".publish.ghcr.enabled" in publish["run"]
    assert ".publish.dockerhub.enabled" in publish["run"]
    assert "targets:[{channel:" not in publish["run"]
    assert "for attempt in 1 2 3" not in build["run"]
    assert "retry-registry-command.sh" in build["run"]
    assert "--retry-quay-blob" in build["run"]
    assert 'UCM_REGISTRY_RETRY_DELAYS="10 20"' in build["run"]
    assert "for setup_attempt in 1 2 3" in verify["run"]
    assert "if ! command -v skopeo" in verify["run"]
    assert (
        "Dir::Etc::sourcelist=/etc/apt/sources.list.d/ubuntu.sources" in verify["run"]
    )
    assert "Dir::Etc::sourceparts=-" in verify["run"]
    assert 'sudo apt-get "${apt_options[@]}" update' in verify["run"]
    assert 'sudo apt-get "${apt_options[@]}" install --yes skopeo' in verify["run"]
    assert "docker run --rm --entrypoint sh" in verify["run"]


def test_builder_prefetches_layers_before_registry_push() -> None:
    steps = _load("sync-builders.yml")["jobs"]["build-missing"]["steps"]
    build = next(
        step["run"] for step in steps if step.get("name") == "Build missing Builder"
    )
    prefetch = 'docker buildx build --output type=oci,dest=/dev/null "${build_args[@]}"'
    push = 'docker buildx build --push "${build_args[@]}"'
    assert build.index(prefetch) < build.index(push)


def test_wheel_only_release_downloads_toolkit_without_requiring_chart() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    steps = jobs["publish-release-artifacts"]["steps"]
    downloads = {
        step["with"]["path"]: step
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    }
    assert "if" not in downloads["input/toolkit"]
    assert downloads["input/chart"]["if"] == (
        "${{ needs.plan.outputs.publish_chart_oci == 'true' }}"
    )
    upload = next(
        step["run"]
        for step in steps
        if step.get("name") == "Upload backend Wheels, Chart, and Config"
    )
    assert "artifacts=(input/wheels/*/*.whl input/toolkit/*.whl)" in upload
