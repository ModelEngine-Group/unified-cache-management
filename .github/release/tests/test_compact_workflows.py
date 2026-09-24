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


def test_chart_receives_a_filtered_plan_with_the_standard_plan_filename() -> None:
    release = _load("release-ucm.yml")["jobs"]
    plan = release["plan"]
    assert plan["outputs"]["chart_plan_artifact"] == (
        "${{ steps.plan.outputs.chart_plan_artifact }}"
    )
    chart_upload = next(
        step
        for step in plan["steps"]
        if step.get("with", {}).get("name")
        == "${{ steps.plan.outputs.chart_plan_artifact }}"
    )
    assert chart_upload["with"]["path"] == "out/chart-plan/release-plan.json"
    assert release["package-chart"]["with"]["plan_artifact"] == (
        "${{ needs.plan.outputs.chart_plan_artifact }}"
    )


def test_nightly_schedule_creates_or_reuses_a_tag_then_calls_core_in_same_run() -> None:
    workflow = _load("release-nightly.yml")
    assert workflow["on"]["schedule"] == [{"cron": "0 18 * * *"}]
    assert workflow["concurrency"] == {
        "group": "ucm-nightly-${{ github.repository_id }}",
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
        if step.get("uses") == "actions/checkout@v4.2.2"
    )
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    run = next(step["run"] for step in prepare["steps"] if step.get("id") == "prepare")
    assert "TZ=Asia/Shanghai date +%Y%m%d" in run
    assert "--next-nightly" in run
    assert "--version-config version.ini" in run
    assert "next_patch_version" not in run
    assert 'git rev-list -n 1 "${candidate}"' in run
    assert ".name == $tag and .draft == true" in run
    assert 'startswith("untagged-")' in run
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
    assert release["secrets"] == "inherit"


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


def test_member_receipt_barrier_uses_the_profile_target_validator() -> None:
    core = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    release_module = (
        ROOT / ".github" / "release" / "ucm_release" / "release.py"
    ).read_text(encoding="utf-8")

    assert "release.py members" in core
    assert 'members = commands.add_parser("members")' in release_module
    assert "validate_member_receipts" in release_module
    assert "_validated_receipt_targets" in release_module


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
        "publish-release-artifacts",
        "publish-image-members",
        "publish-image-indexes",
        "publish-pypi",
        "publish-chart-oci",
        "update-release-images",
    ):
        assert jobs[name]["environment"] == scoped_environment
        assert jobs[name]["permissions"]["contents"] in {"read", "write"}
    assert jobs["release-preflight"]["environment"] == scoped_environment
    assert jobs["release-preflight"]["permissions"] == {"contents": "read"}
    assert "packages" not in jobs["publish-pypi"]["permissions"]
    assert jobs["publish-chart-oci"]["permissions"]["packages"] == "write"
    assert "packages" not in jobs["publish-image-members"]["permissions"]
    assert jobs["build-images"]["permissions"]["packages"] == "write"
    assert jobs["update-release-images"]["permissions"] == {
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
    assert "docker run --rm" in run
    assert '-e UCM_WHEEL="/tmp/${wheel_name}"' in run
    assert '"${UCM_WHEEL}"' in run
    assert "validate_wheel_runtime.py" in run
    assert "python3 -m venv" in run
    assert "DEFERRED_EXTERNAL_LIBRARIES" in run
    assert "EXPECTED_RUNTIME_REQUIREMENTS" in run
    assert '.dependencies | select(type == "array")' in run
    assert ".deferred_external_libraries" in run
    assert 'if [[ "${PLATFORM_ARG}" == ascend* ]]' not in run
    assert "python -m pip check" not in run
    assert 're.search(r\\"[0-9]+' in run
    assert "re.search(r'[0-9]+" not in run
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
    assert "github.run_attempt" not in text
    assert "GITHUB_RUN_ATTEMPT" in text  # candidate Builder tags remain retry-scoped.


def test_completed_release_runs_retention_and_manual_cleanup_reuses_the_module() -> (
    None
):
    finalize = _load("release-ucm.yml")["jobs"]["update-release-images"]
    steps = finalize["steps"]
    manifest_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("id") == "publish-manifest"
    )
    retention_index, retention = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("id") == "retention"
    )
    assert manifest_index < retention_index
    assert "cleanup.py retention" in retention["run"]
    assert "release_profile" in retention["run"]
    assert 'pypi_enabled="$(jq -r' in retention["run"]
    assert 'pypi_enabled="$(jq -er' not in retention["run"]
    assert finalize["permissions"] == {
        "actions": "write",
        "contents": "write",
        "packages": "write",
    }

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
    assert jobs["test-e2e-pc-a2"]["if"] == (
        "github.repository == 'ModelEngine-Group/unified-cache-management'"
    )

    text = (WORKFLOWS / "pull-request.yml").read_text(encoding="utf-8")
    assert "./.github/workflows/release-ucm.yml" not in text


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
    assert "publish" not in text.lower()


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
    assert "${RUNNER_TEMP}/ucm-builder-prefetch.log" in build
    assert "${RUNNER_TEMP}/ucm-builder-push.log" in build


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
