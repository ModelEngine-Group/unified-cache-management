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


def test_user_facing_release_workflow_names_explain_the_build_lanes() -> None:
    assert _load("release-tag.yml")["name"] == (
        "UCM Tag Release · Stable, Prerelease, Draft, and Nightly"
    )
    assert _load("release-nightly.yml")["name"] == (
        "UCM Nightly Release · Shanghai 02:00"
    )
    assert _load("release-ucm.yml")["name"] == "UCM Reusable Release Core"
    assert _load("_native-wheel-gate.yml")["name"] == "UCM Native Wheel Gate"
    assert _load("ucm-build-bot.yml")["name"] == (
        "UCM PR Build Robot · Wheel, Image, and Chart"
    )


def test_release_core_is_input_driven_and_uses_crane_before_plan() -> None:
    workflow = _load("release-ucm.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    inputs = workflow["on"]["workflow_call"]["inputs"]
    assert set(inputs) == {
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
    assert all(value["required"] is True for value in inputs.values())
    assert inputs["is_prerelease"]["type"] == "boolean"
    assert all(
        inputs[name]["type"] == "string" for name in set(inputs) - {"is_prerelease"}
    )

    text = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    for event_value in (
        "GITHUB_EVENT_NAME",
        "GITHUB_REF_NAME",
        "GITHUB_REF_TYPE",
        "github.event_name",
        "github.ref_name",
        "github.ref_type",
    ):
        assert event_value not in text

    jobs = workflow["jobs"]
    assert "needs" not in jobs["release-preflight"]
    assert jobs["select-runtime-candidates"]["needs"] == "release-preflight"
    assert (
        "needs.release-preflight.outputs.enabled"
        in jobs["select-runtime-candidates"]["if"]
    )
    assert set(jobs["open-release"]["needs"]) == {
        "release-preflight",
        "select-runtime-candidates",
    }
    assert set(jobs["inspect-runtimes"]["needs"]) == {
        "select-runtime-candidates",
        "open-release",
    }
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
    assert set(jobs["plan"]["needs"]) == {
        "release-preflight",
        "open-release",
        "resolve-upstreams",
        "sync-builders",
    }
    assert (
        jobs["sync-builders"]["with"]["runtime_selection_artifact"]
        == "${{ needs.resolve-upstreams.outputs.runtime_selection_artifact }}"
    )
    candidate_text = yaml.safe_dump(jobs["select-runtime-candidates"])
    resolve_text = yaml.safe_dump(jobs["resolve-upstreams"])
    plan_text = yaml.safe_dump(jobs["plan"])
    plan_run = next(
        step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan"
    )
    profile_run = next(
        step["run"]
        for step in jobs["release-preflight"]["steps"]
        if step.get("id") == "profile"
    )
    assert "--release-type" in candidate_text
    assert "inputs.release_type" in candidate_text
    assert "--release-type" in resolve_text
    assert "inputs.release_type" in resolve_text
    assert "--runtime-selection" in plan_text
    assert "--release-type" in plan_text
    assert "inputs.release_type" in plan_text
    assert ".release_type == $type" in plan_run
    assert "canonical non-local version" in profile_run
    assert "if version <" not in profile_run
    assert "if version >" not in profile_run
    assert '--tag "${RELEASE_TAG}" --classify' in profile_run
    assert "--version-config version.ini" in profile_run
    assert "Docker Hub namespace must be configured" in profile_run
    assert "Docker Hub username and token must be configured together" in profile_run
    assert 'if [ "${dockerhub_requested}" = true ]; then' in profile_run
    assert '[ -z "${DOCKERHUB_NAMESPACE}" ]' in profile_run
    assert '[ -z "${DOCKERHUB_USERNAME}" ]' in profile_run
    assert '[ -z "${DOCKERHUB_TOKEN}" ]' in profile_run
    assert "dockerhub_namespace=sys.argv[3] or None" in profile_run
    assert "fork_test_pypi" in profile_run
    assert jobs["release-preflight"]["env"]["DOCKERHUB_NAMESPACE"] == (
        "${{ vars.DOCKERHUB_NAMESPACE }}"
    )
    assert "dockerhub_namespace" not in jobs["release-preflight"]["outputs"]
    assert "refs/tags/${RELEASE_TAG}^{commit}" in profile_run
    assert 'test "${head_sha}" = "${SOURCE_SHA}"' in profile_run
    assert 'test "${tag_sha}" = "${SOURCE_SHA}"' in profile_run
    for field in (
        "git_tag",
        "release_type",
        "release_kind",
        "version",
        "chart_version",
        "image_version",
        "is_prerelease",
    ):
        assert f".{field}" in profile_run
    assert "image_index_matrix" in plan_text
    assert "'{include:[.families[]" in plan_text
    assert "--publication-context" in plan_run
    assert "needs.release-preflight.outputs.fork_test_pypi" in plan_text
    assert "needs.release-preflight.outputs.dockerhub_namespace" not in plan_text
    plan_step = next(step for step in jobs["plan"]["steps"] if step.get("id") == "plan")
    assert plan_step["env"]["DOCKERHUB_NAMESPACE"] == "${{ vars.DOCKERHUB_NAMESPACE }}"
    assert "dockerhub_namespace" in plan_run
    assert "FORK_DOCKERHUB_NAMESPACE" not in text
    assert "docker.io/${DOCKERHUB_USERNAME,,}" not in text


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
    assert "compact plan" in text and "--route pr" in text
    assert "sync-builders.yml" not in text
    assert "packages: write" not in text
    assert "docker push" not in text
    assert "open-release" not in text
    assert "publish" not in text.lower()


def test_release_workflow_has_staged_publication_jobs() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    publish_jobs = [
        "open-release",
        "build-meta",
        "validate-wheel-runtimes",
        "publish-release-artifacts",
        "publish-image-members",
        "publish-image-indexes",
        "publish-pypi",
        "verify-pypi-installs",
        "publish-chart-oci",
        "update-release-images",
    ]
    assert all(name in jobs for name in publish_jobs)
    assert set(jobs["publish-release-artifacts"]["needs"]) == {
        "plan",
        "open-release",
        "build-wheels",
        "build-meta",
        "validate-wheel-runtimes",
        "package-chart",
    }
    assert set(jobs["publish-image-members"]["needs"]) == {
        "plan",
        "build-images",
        "publish-release-artifacts",
    }
    assert (
        "needs.plan.outputs.publish_images == 'true'"
        in jobs["publish-image-members"]["if"]
    )
    assert set(jobs["build-images"]["needs"]) == {
        "plan",
        "publish-release-artifacts",
    }
    assert jobs["plan"]["outputs"]["publish_images"] == (
        "${{ steps.plan.outputs.publish_images }}"
    )
    assert jobs["plan"]["outputs"]["publish_pypi"] == (
        "${{ steps.plan.outputs.publish_pypi }}"
    )
    assert jobs["plan"]["outputs"]["pypi_disposition"] == (
        "${{ steps.plan.outputs.pypi_disposition }}"
    )
    assert jobs["plan"]["outputs"]["dockerhub_disposition"] == (
        "${{ steps.plan.outputs.dockerhub_disposition }}"
    )
    report_decisions = next(
        step
        for step in jobs["plan"]["steps"]
        if step.get("name") == "Report shared publication decisions"
    )["run"]
    assert ".publish.pypi.requested" in report_decisions
    assert ".publish.pypi.disposition" in report_decisions
    assert ".publish.dockerhub.requested" in report_decisions
    assert ".publish.dockerhub.disposition" in report_decisions
    assert "jq -r '.publish.pypi.requested'" in report_decisions
    assert "jq -r '.publish.pypi.enabled'" in report_decisions
    assert "jq -r '.publish.dockerhub.requested'" in report_decisions
    assert "jq -r '.publish.dockerhub.enabled'" in report_decisions
    assert "jq -er '.publish.pypi.requested'" not in report_decisions
    assert "jq -er '.publish.dockerhub.requested'" not in report_decisions
    assert "GITHUB_STEP_SUMMARY" in report_decisions
    assert "needs.plan.outputs.publish_images == 'true'" in jobs["build-images"]["if"]
    assert "publish-release-artifacts.result == 'success'" in jobs["build-images"]["if"]
    assert set(jobs["publish-image-indexes"]["needs"]) == {
        "plan",
        "publish-image-members",
    }
    assert set(jobs["publish-pypi"]["needs"]) == {
        "plan",
        "publish-release-artifacts",
    }
    assert jobs["publish-pypi"]["if"] == (
        "${{ needs.plan.outputs.publish_pypi == 'true' }}"
    )
    assert jobs["publish-pypi"]["name"] == (
        "PyPI · ${{ needs.plan.outputs.pypi_disposition }}"
    )
    assert set(jobs["verify-pypi-installs"]["needs"]) == {
        "plan",
        "publish-pypi",
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
        "publish-pypi",
        "verify-pypi-installs",
        "publish-chart-oci",
    }
    for name in (
        "open-release",
        "publish-release-artifacts",
        "update-release-images",
    ):
        assert jobs[name]["env"]["GH_REPO"] == "${{ github.repository }}"


def test_nightly_schedule_creates_or_reuses_a_tag_then_calls_core_in_same_run() -> None:
    workflow = _load("release-nightly.yml")
    assert workflow["on"] == {
        "schedule": [{"cron": "0 18 * * *"}],
        "workflow_dispatch": None,
    }
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
    assert release["with"]["publication_scope"] == "official"
    assert release["secrets"] == "inherit"


def test_artifact_stage_keeps_release_state_internal() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    text = yaml.safe_dump(jobs["publish-release-artifacts"])
    assert jobs["publish-release-artifacts"]["name"] == (
        "Release · Publish backend Wheels, Chart, and Config"
    )
    assert "release.py artifacts" in text
    assert "release-state.json" in text
    assert "release-manifest.json" not in text
    state_step = next(
        step
        for step in jobs["publish-release-artifacts"]["steps"]
        if step.get("name") == "Build internal artifacts-ready release state"
    )
    assert '--run-id "${GITHUB_RUN_ID}"' in state_step["run"]
    assert "ucm-release-stage-run-${{ github.run_id }}" in text
    assert "build-images" not in jobs["publish-release-artifacts"]["needs"]
    upload = next(
        step
        for step in jobs["publish-release-artifacts"]["steps"]
        if step.get("name") == "Upload backend Wheels, Chart, and Config"
    )
    assert 'gh release upload "${tag}" --clobber "${artifacts[@]}"' in upload["run"]
    assert ".publish.pypi.enabled" not in upload["run"]
    assert "meta_artifacts=(input/meta/*.whl)" in upload["run"]
    assert 'artifacts+=("${meta_artifacts[@]}")' not in upload["run"]
    assert 'index("ucm_config_example.yaml") != null' in upload["run"]
    assert "SHA256SUMS" not in upload["run"]
    assert "release-manifest.json" not in upload["run"]
    assert "release.py notes" in upload["run"]
    assert 'gh api "/repos/${GH_REPO}/releases/${release_id}"' in upload["run"]
    assert '--repository "${GH_REPO}"' in upload["run"]
    metadata_check = next(
        step
        for step in jobs["publish-release-artifacts"]["steps"]
        if step.get("name") == "Validate all PyPI package metadata without publishing"
    )
    assert "python -m twine check" in metadata_check["run"]
    assert "twine==6.1.0" in text
    failure_step = jobs["publish-release-artifacts"]["steps"][-1]
    assert failure_step["if"] == (
        "${{ failure() && needs.open-release.outputs.enabled == 'true' }}"
    )
    assert "artifacts-failed" in failure_step["run"]


def test_runtime_images_include_the_config_example_at_the_workspace_root() -> None:
    image_path = "/workspace/ucm_config_example.yaml"
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.runtime"
    ).read_text(encoding="utf-8")

    assert f"COPY --chmod=0644 ucm_config_example.yaml {image_path}" in dockerfile
    assert "ENV UCM_CONFIG_FILE" not in dockerfile
    for workflow_name, build_name, verify_name in (
        (
            "_build-release-image.yml",
            "Build install-only Runtime image",
            "Verify Runtime glibc, Python, OS, and UCM import",
        ),
        (
            "_build-image.yml",
            "Build install-only runtime image",
            "Verify runtime glibc, Python, OS, and UCM import",
        ),
    ):
        steps = _load(workflow_name)["jobs"][
            "publish" if workflow_name == "_build-release-image.yml" else "build"
        ]["steps"]
        build = next(step for step in steps if step.get("name") == build_name)
        verify = next(step for step in steps if step.get("name") == verify_name)
        assert (
            "cp examples/ucm_config_example.yaml context/ucm_config_example.yaml"
            in build["run"]
        )
        assert "sha256sum examples/ucm_config_example.yaml" in verify["run"]
        assert "EXPECTED_UCM_CONFIG_SHA256" in verify["run"]
        assert "DEFERRED_EXTERNAL_LIBRARIES" in verify["run"]
        assert "EXPECTED_RUNTIME_REQUIREMENTS" in verify["run"]
        assert '.dependencies | select(type == "array")' in verify["run"]
        assert ".deferred_external_libraries" in verify["run"]
        assert "hashlib.sha256(path.read_bytes())" in verify["run"]
        assert image_path in verify["run"]


def test_schema_v6_manifest_is_uploaded_only_after_complete_and_read_back() -> None:
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
    release_module = (
        ROOT / ".github" / "release" / "ucm_release" / "release.py"
    ).read_text(encoding="utf-8")
    assert "PUBLIC_MANIFEST_SCHEMA_VERSION = 6" in release_module

    for name, job in jobs.items():
        if name != "update-release-images":
            assert "gh release upload" not in yaml.safe_dump(job) or (
                name == "publish-release-artifacts"
                and "release-manifest.json" not in yaml.safe_dump(job)
            )


def test_handwritten_release_notes_start_with_status() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    runs = [
        next(
            step["run"]
            for step in jobs["open-release"]["steps"]
            if step.get("name") == "Create or reuse exact Tag Release"
        ),
        jobs["report-planning-failure"]["steps"][0]["run"],
        jobs["publish-release-artifacts"]["steps"][0]["run"],
        jobs["publish-release-artifacts"]["steps"][-1]["run"],
    ]

    for run in runs:
        assert "printf '%s\\n' \\\n  \"Status:" in run
        assert '"# UCM ${tag}"' not in run
        assert "Checksums:" not in run

    fallback = jobs["update-release-images"]["steps"][-1]["run"]
    assert "release.py notes" in fallback
    assert "Status: \\`publication-failed\\`" in fallback


def test_opened_release_reports_pre_plan_failures() -> None:
    job = _load("release-ucm.yml")["jobs"]["report-planning-failure"]

    assert "always()" in job["if"]
    assert "needs.plan.result != 'success'" in job["if"]
    assert job["env"]["GH_REPO"] == "${{ github.repository }}"
    assert "artifacts-failed" in job["steps"][0]["run"]


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


def test_tag_entry_only_classifies_four_release_types_and_calls_one_core() -> None:
    workflow = _load("release-tag.yml")
    assert workflow["on"] == {"push": {"tags": ["v*", "draft/v*", "nightly/v*"]}}
    assert set(workflow["jobs"]) == {
        "classify-tag",
        "release",
    }
    classify = workflow["jobs"]["classify-tag"]
    assert set(classify["outputs"]) == {
        "git_tag",
        "release_type",
        "release_kind",
        "version",
        "chart_version",
        "image_version",
        "is_prerelease",
        "publication_scope",
    }
    classify_run = next(
        step["run"] for step in classify["steps"] if step.get("id") == "classify"
    )
    assert '--tag "${RELEASE_TAG}" --classify' in classify_run
    assert "--version-config version.ini" in classify_run
    assert "release_type=$(jq -r '.release_type'" in classify_run

    assert "${GITHUB_REPOSITORY,,}" in classify_run
    assert "modelengine-group/unified-cache-management" in classify_run

    release = workflow["jobs"]["release"]
    assert release["name"] == "Run Release core"
    assert release["needs"] == "classify-tag"
    assert "if" not in release
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
    assert release["with"]["source_sha"] == "${{ github.sha }}"
    assert release["with"]["publication_scope"] == (
        "${{ needs.classify-tag.outputs.publication_scope }}"
    )
    assert release["secrets"] == {
        "PYPI_API_TOKEN": (
            "${{ needs.classify-tag.outputs.publication_scope == 'official' "
            "&& secrets.PYPI_API_TOKEN || '' }}"
        ),
        "TEST_PYPI_API_TOKEN": (
            "${{ needs.classify-tag.outputs.publication_scope == 'fork' "
            "&& secrets.TEST_PYPI_API_TOKEN || '' }}"
        ),
        "DOCKERHUB_USERNAME": "${{ secrets.DOCKERHUB_USERNAME }}",
        "DOCKERHUB_TOKEN": "${{ secrets.DOCKERHUB_TOKEN }}",
    }


def test_common_core_validates_and_selects_runtimes_before_opening_release() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    checkout = next(
        step
        for step in jobs["release-preflight"]["steps"]
        if step.get("uses") == "actions/checkout@v4.2.2"
    )
    open_text = yaml.safe_dump(jobs["open-release"])
    open_run = next(
        step["run"]
        for step in jobs["open-release"]["steps"]
        if step.get("name") == "Create or reuse exact Tag Release"
    )
    assert "needs" not in jobs["release-preflight"]
    assert set(jobs["open-release"]["needs"]) == {
        "release-preflight",
        "select-runtime-candidates",
    }
    assert checkout["with"] == {
        "ref": "${{ inputs.source_sha }}",
        "fetch-depth": 0,
        "fetch-tags": True,
        "persist-credentials": False,
    }
    assert "--verify-tag" in open_text
    assert "--paginate --slurp" in open_text
    assert "(.tag_name == $tag) or" in open_run
    assert ".name == $tag and .draft == true" in open_run
    assert 'startswith("untagged-")' in open_run
    assert 'index("release-manifest.json")' in open_run
    assert "(.assets | length) == 0" in open_run
    assert (
        "artifacts-ready|artifacts-failed|images-failed|publication-failed" in open_run
    )
    assert "complete or is not owned by this pipeline" in open_run
    assert open_run.count(".name == $tag") == 1
    assert "wait_for_release" in open_run
    assert "for attempt in $(seq 1 15)" in open_run
    assert "sleep 2" in open_run
    assert open_run.count('-f "tag_name=${tag}"') == 2
    assert "draft=false" in open_text
    assert "build-wheels" not in open_text
    plan_run = next(
        step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan"
    )
    assert "RELEASE_TAG: ${{ inputs.git_tag }}" in yaml.safe_dump(jobs["plan"])
    assert "--git-tag" in plan_run
    assert ".git_tag == $tag" in plan_run
    assert ".release_type == $type" in plan_run
    assert ".release_tag = $tag" not in plan_run


def test_draft_body_patches_preserve_tag_after_asset_uploads() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    artifact_run = next(
        step["run"]
        for step in jobs["publish-release-artifacts"]["steps"]
        if step.get("name") == "Upload backend Wheels, Chart, and Config"
    )
    manifest_run = next(
        step["run"]
        for step in jobs["update-release-images"]["steps"]
        if step.get("id") == "publish-manifest"
    )

    for run in (artifact_run, manifest_run):
        assert "if jq -e '.draft == true'" in run
        assert '-f "tag_name=${tag}"' in run
        assert '-f "body=$(<out/release/release-notes.md)"' in run


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


def test_direct_member_receipt_barrier_and_index_matrix_are_unbounded() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    members = jobs["publish-image-members"]
    indexes = jobs["publish-image-indexes"]
    assert "strategy" not in members
    assert "ucm-image-member-receipt-*" in yaml.safe_dump(members)
    member_run = next(
        step["run"]
        for step in members["steps"]
        if step.get("name") == "Require one complete receipt per planned image"
    )
    assert "release.py members" in member_run
    assert "--plan input/plan/release-plan.json" in member_run
    assert "--receipts input/receipts" in member_run
    assert indexes["strategy"]["fail-fast"] is False
    assert "max-parallel" not in indexes["strategy"]
    assert (
        indexes["strategy"]["matrix"]
        == "${{ fromJSON(needs.plan.outputs.image_index_matrix) }}"
    )
    assert "needs.plan.outputs.publish_images == 'true'" in indexes["if"]
    assert "needs.publish-image-members.result == 'success'" in indexes["if"]
    assert "needs.plan.outputs.has_image_indexes == 'true'" in indexes["if"]
    text = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    assert "while IFS=$'\\t' read -r image_id" not in text


def test_member_receipt_barrier_uses_the_profile_target_validator() -> None:
    core = (WORKFLOWS / "release-ucm.yml").read_text(encoding="utf-8")
    release_module = (
        ROOT / ".github" / "release" / "ucm_release" / "release.py"
    ).read_text(encoding="utf-8")

    assert "release.py members" in core
    assert 'members = commands.add_parser("members")' in release_module
    assert "validate_member_receipts" in release_module
    assert "_validated_receipt_targets" in release_module


def test_release_index_matrix_generation_fails_closed() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    plan_script = next(
        step["run"] for step in jobs["plan"]["steps"] if step.get("id") == "plan"
    )

    assert 'index_matrix="$(jq -ce' in plan_script
    assert ".publish.ghcr.enabled == true" in plan_script
    assert ".publish.dockerhub.enabled == true" in plan_script
    assert " or " in plan_script
    assert 'echo "publish_images=${publish_images}"' in plan_script
    assert 'if [ "${publish_images}" = true ]; then' in plan_script
    assert 'echo "image_index_matrix=${index_matrix}"' in plan_script
    assert 'echo "has_image_indexes=${has_image_indexes}"' in plan_script
    assert "image_index_matrix=$(jq" not in plan_script


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


def test_shared_channel_writers_use_only_effective_enabled_decisions() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    pypi = next(
        step["run"]
        for step in jobs["publish-pypi"]["steps"]
        if step.get("name") == "Publish backend Wheels, then the meta Wheel"
    )
    assert "--attempts 36 --interval 5" in pypi
    dockerhub_index = next(
        step["run"]
        for step in jobs["publish-image-indexes"]["steps"]
        if step.get("name") == "Publish multi-architecture index"
    )
    member = next(
        step["run"]
        for step in _load("_build-release-image.yml")["jobs"]["publish"]["steps"]
        if step.get("name") == "Publish verified Profile members and record digests"
    )

    for script, channel in (
        (pypi, "pypi"),
        (dockerhub_index, "dockerhub"),
        (member, "dockerhub"),
    ):
        assert f".publish.{channel}.enabled" in script
        assert f".publish.{channel}.requested" not in script
        assert f".publish.{channel}.disposition" not in script


def test_builder_sync_consumes_selection_and_uses_digest_pinned_mirror_only() -> None:
    workflow = _load("sync-builders.yml")
    assert set(workflow["on"]) == {"workflow_call"}
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "runtime_selection_artifact",
    }
    text = (WORKFLOWS / "sync-builders.yml").read_text(encoding="utf-8")
    build = next(
        step["run"]
        for step in workflow["jobs"]["build-missing"]["steps"]
        if step.get("name") == "Build missing Builder"
    )
    assert "--selection input/upstreams/runtime-selection.json" in text
    assert "matrix.source_image" in text
    assert "matrix.source_image_digest" in text
    assert 'pinned_source="${source_repository}@${SOURCE_IMAGE_DIGEST}"' in text
    assert "retry-registry-command.sh" in build
    assert '--rate-limit-marker "${target_rate_limit_marker}"' in build
    assert 'docker pull --platform "linux/${CPU_ARCH}" "${verified_target}"' in build
    assert 'docker pull --platform "linux/${CPU_ARCH}" "${pinned_source}"' in build
    assert '[ ! -f "${target_rate_limit_marker}" ] ||' in build
    assert '!= "rate-limit-exhausted"' in build
    assert 'validation_image="${pinned_source}"' in build
    assert 'docker run --pull=never --platform "linux/${CPU_ARCH}" --rm' in build
    assert build.count("docker run --pull=never") == 1
    assert build.index(
        'docker pull --platform "linux/${CPU_ARCH}" "${verified_target}"'
    ) < build.index('docker pull --platform "linux/${CPU_ARCH}" "${pinned_source}"')
    assert build.index(
        'docker pull --platform "linux/${CPU_ARCH}" "${pinned_source}"'
    ) < build.index('docker run --pull=never --platform "linux/${CPU_ARCH}" --rm')
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
    assert '"${validation_image}" bash -c' in text
    assert "--format '{{json .Image}}' >out/verified-target-config-raw.json" in text
    assert "if has($platform) then .[$platform] else . end" in text
    assert "--slurpfile image out/verified-target-config.json" in text
    assert "config:$image[0]" in text
    assert "docker image inspect" not in text
    assert "ucm-builder-verification-${{ matrix.id }}" in text
    assert "candidate-${GITHUB_RUN_ID}" in text
    promotion = 'docker buildx imagetools create --tag "${target}" "${candidate}"'
    promotion_index = build.index(promotion)
    retry_index = build.rfind("retry-registry-command.sh", 0, promotion_index)
    assert 0 < promotion_index - retry_index < 240
    retry_prefix = build[retry_index:promotion_index]
    assert "${RUNNER_TEMP}/ucm-builder-promotion.log" in retry_prefix
    assert "--retry-transport" in retry_prefix
    assert workflow["jobs"]["build-missing"]["timeout-minutes"] == 180
    assert set(workflow["jobs"]["finalize"]["needs"]) == {
        "prepare",
        "build-missing",
    }

    release_docker = ROOT / ".github" / "release" / "docker"
    mirror_lines = [
        line
        for line in (release_docker / "Dockerfile.builder-mirror")
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert mirror_lines == ["ARG BASE_IMAGE", "FROM ${BASE_IMAGE}"]
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
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")
    assert '.builder.repository + "@" + .builder.digest' in text
    requirements = (
        ROOT / ".github" / "release" / "requirements" / "wheel-build.txt"
    ).read_text(encoding="utf-8")
    assert "auditwheel==6.7.0" in requirements
    assert "ucm-python -m auditwheel -v repair" in dockerfile
    assert "--only-plat" in dockerfile
    assert "ucm-python -m auditwheel -v show" in dockerfile
    assert "--strip" not in dockerfile
    assert "compact record-wheel-result" in text
    assert "out/wheel/wheel-result.json" in text


def test_release_builds_reproducible_empty_meta_wheel_from_the_plan() -> None:
    jobs = _load("release-ucm.yml")["jobs"]
    job = jobs["build-meta"]
    assert job["needs"] == "plan"
    build = next(
        step
        for step in job["steps"]
        if step.get("name") == "Build and validate empty meta Wheel"
    )
    assert build["env"]["SOURCE_DATE_EPOCH"] == "315532800"
    assert "compact \\\n  materialize-meta-source" in build["run"]
    assert "record-meta-result" in build["run"]
    assert "ucm-meta-run-${{ github.run_id }}" in yaml.safe_dump(job)


def test_pypi_publication_is_backend_first_meta_last_and_read_back() -> None:
    job = _load("release-ucm.yml")["jobs"]["publish-pypi"]
    publish = next(
        step
        for step in job["steps"]
        if step.get("name") == "Publish backend Wheels, then the meta Wheel"
    )["run"]
    assert "pypi publish" in publish
    assert '.release_type == "stable" or .release_type == "prerelease"' not in publish
    assert "--wheel-root input/wheels --meta-root input/meta" in publish
    assert "--output out/pypi/pypi-receipt.json" in publish
    assert "--skip-existing" not in publish
    adapter = (ROOT / ".github" / "release" / "ucm_release" / "pypi.py").read_text(
        encoding="utf-8"
    )
    assert '"Cache-Control": "no-cache"' in adapter
    assert adapter.index("for filename in missing_backends") < adapter.index(
        "for filename in missing_meta"
    )

    job = _load("release-ucm.yml")["jobs"]["verify-pypi-installs"]
    verify = job
    assert "pypi_test_matrix" in str(verify["strategy"]["matrix"])
    install = next(
        step
        for step in job["steps"]
        if step.get("name") == "Install one extra from the planned Python index"
    )["run"]
    assert "python3 -m venv" in install
    assert ".meta_package.distribution" in install
    assert "ucm-pypi-receipt-run-${{ github.run_id }}" in yaml.safe_dump(job)
    assert ".publish.pypi.json_api" in install
    assert "https://test-files.pythonhosted.org/" in install
    assert "https://files.pythonhosted.org/" in install
    assert ".digests.sha256 == $sha256" in install
    assert 'actual_sha256="$(sha256sum' in install
    assert 'UCM_META_DISTRIBUTION="${meta_distribution}"' in install
    assert 'EXPECTED_META_DISTRIBUTION="${meta_distribution}"' in install
    assert '"${UCM_META_WHEEL}[${UCM_EXTRA}]" "${UCM_BACKEND_WHEEL}"' in install
    assert "--extra-index-url" not in install
    assert "validate_wheel_runtime.py" in install
    assert "DEFERRED_EXTERNAL_LIBRARIES" in install


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


def test_wheel_build_authenticates_to_ghcr_with_read_only_package_access() -> None:
    workflow = _load("_build-wheel.yml")
    assert workflow["permissions"] == {"contents": "read", "packages": "read"}

    steps = workflow["jobs"]["build"]["steps"]
    auth_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Authenticate to GHCR"
    )
    build_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Build native wheel"
    )
    auth = steps[auth_index]

    assert auth_index < build_index
    assert auth["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert "docker login ghcr.io" in auth["run"]
    assert '"${GITHUB_ACTOR}" --password-stdin' in auth["run"]


def test_reusable_builds_keep_functional_inputs() -> None:
    expected = {
        "_build-wheel.yml": {
            "wheel_id",
            "runner",
            "plan_artifact",
            "source_ref",
            "retention_days",
        },
        "_build-image.yml": {
            "image_id",
            "runner",
            "plan_artifact",
            "upload_oci",
            "source_ref",
        },
        "_build-release-image.yml": {
            "image_id",
            "runner",
            "plan_artifact",
            "source_ref",
            "publication_scope",
        },
        "_build-chart.yml": {"plan_artifact", "source_ref", "retention_days"},
    }
    for filename, inputs in expected.items():
        workflow = _load(filename)
        assert set(workflow["on"]["workflow_call"]["inputs"]) == inputs

    core = _load("release-ucm.yml")["jobs"]
    assert core["build-wheels"]["with"]["source_ref"] == "${{ inputs.source_sha }}"
    assert core["build-wheels"]["with"]["retention_days"] == 90
    assert core["package-chart"]["with"]["source_ref"] == "${{ inputs.source_sha }}"
    assert core["package-chart"]["with"]["retention_days"] == 90
    for workflow_name, job_name in (
        ("_build-wheel.yml", "build"),
        ("_build-chart.yml", "package"),
    ):
        text = yaml.safe_dump(_load(workflow_name)["jobs"][job_name])
        assert "scripts/materialize_version.py" in text
        assert "--version" in text
        assert ".version" in text
        assert "GITHUB_EVENT_NAME" not in text
        assert "GITHUB_REF_NAME" not in text


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
    publish = next(
        step
        for step in steps
        if step.get("name") == "Publish verified Profile members and record digests"
    )

    assert (
        step_names.index(build["name"])
        < step_names.index(verify["name"])
        < step_names.index(publish["name"])
    )
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
    assert "for setup_attempt in 1 2 3" in verify["run"]
    assert "sudo apt-get update && sudo apt-get install --yes skopeo" in verify["run"]
    assert "docker run --rm --entrypoint sh" in verify["run"]


def test_compact_wheel_passes_dynamic_python_and_platform_to_build() -> None:
    workflow = (WORKFLOWS / "_build-wheel.yml").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")
    assert "retry-registry-command.sh" in workflow
    assert "${RUNNER_TEMP}/ucm-wheel-target-build.log" in workflow
    assert "${RUNNER_TEMP}/ucm-wheel-source-build.log" in workflow
    assert ".builder.source_image" in workflow
    assert ".builder.source_image_digest" in workflow
    assert 'source_builder="${source_repository}@$(jq -er' in workflow
    assert 'build_wheel "${builder}"' in workflow
    assert 'build_wheel "${source_builder}"' in workflow
    assert 'if [ "${builder}" = "${source_builder}" ]; then' in workflow
    assert '--rate-limit-marker "${rate_limit_marker}"' in workflow
    assert '--rate-limit-scope "${rate_limit_scope}"' in workflow
    assert 'builder_rate_limit_scope="${builder_repository#ghcr.io/}"' in workflow
    assert '"${target_rate_limit_marker}" "${builder_rate_limit_scope}"' in workflow
    assert '[ ! -f "${target_rate_limit_marker}" ] ||' in workflow
    assert '!= "rate-limit-exhausted"' in workflow
    assert "UCM_PYTHON_VERSION" in workflow
    assert "UCM_PYTHON_ABI" in workflow
    assert "UCM_PLATFORM" in workflow
    assert "ARG UCM_PYTHON_VERSION" in dockerfile
    assert "ARG UCM_PYTHON_ABI" in dockerfile
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in dockerfile
    assert 'sysconfig.get_path("scripts")' in dockerfile
    assert 'PATH="${python_scripts}:${PATH}"' in dockerfile


def test_compact_wheel_uses_source_metadata_and_active_ascend_arch_handoff() -> None:
    workflow = (WORKFLOWS / "_build-wheel.yml").read_text(encoding="utf-8")
    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.wheel"
    ).read_text(encoding="utf-8")
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "compact prepare-wheel-source" in workflow
    assert "--distribution \"$(jq -r '.dist_name'" in workflow
    assert 'version_path = os.path.join(ROOT_DIR, "version.ini")' in setup_py
    assert 'key == "UCM_VERSION"' in setup_py
    assert "version=get_package_version()" in setup_py

    combined = workflow + dockerfile + setup_py
    assert "UCM_BUILD_CONFIG" not in combined
    assert "UCM_DIST_NAME" not in combined

    assert "--build-arg \"UCM_CPU_ARCH=$(jq -r '.cpu_arch'" in workflow
    assert "ARG UCM_CPU_ARCH" in dockerfile
    assert 'UCM_BUILD_CPU_ARCH="${UCM_CPU_ARCH}"' in dockerfile
    assert 'os.getenv("UCM_BUILD_CPU_ARCH")' in setup_py
    assert "if is_ascend() and build_cpu_arch:" in setup_py
    assert '"amd64": "x86_64-linux"' in setup_py
    assert '"arm64": "aarch64-linux"' in setup_py
    assert "-DASCEND_ARCH_DIR=" in setup_py


def test_ucm_build_enables_strict_compilation() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'set(FLAGS_PUBLIC "-Wall -Werror ' in cmake
    assert "-Wno-error=stringop-overflow" not in cmake


def test_ascend_drampool_defers_only_shared_library_driver_symbols() -> None:
    cmake = (ROOT / "ucm" / "store" / "dram" / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    option = 'target_link_options(drampool PRIVATE "LINKER:--allow-shlib-undefined")'
    assert cmake.count(option) == 1
    option_index = cmake.index(option)
    guard_index = cmake.rfind("if(UCM_RUNTIME_ASCEND_FAMILY)", 0, option_index)
    endif_index = cmake.index("endif()", option_index)
    assert guard_index < option_index < endif_index


def test_chart_consumes_product_smoke_values_from_v4_policy() -> None:
    text = (WORKFLOWS / "_build-chart.yml").read_text(encoding="utf-8")
    assert ".chart.smoke_values[$product]" in text
    assert '--set-string "images.image=${repository}:${tag}"' in text
    assert "images.engine." not in text
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
    assert not (WORKFLOWS / "ucm-build-hint.yml").exists()
    assert "runtime inspect" in text
    assert "runtime aggregate" in text
    assert "builders scan-registry" in text
    assert "runtime resolve" in text
    assert "upstreams candidates" in text
    assert text.count("./.github/workflows/_probe-runtime.yml") == 2
    assert "opaque" in text
    assert "pep440" not in text.lower()
    assert "cann900" not in text
    assert "## `/ucm-build`" in text
    assert "build receipt" in text
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
    assert "GITHUB_REPOSITORY_OWNER" in member_step
    assert "outside current owner namespace" in member_step
    assert "${tag}-${arch}" not in member_step
    index_step = indexes["steps"][-1]["run"]
    assert "matrix.members" not in index_step
    assert "jq -er '.[]'" in index_step
    assert '"${member_refs[@]}"' in index_step
    assert "GITHUB_REPOSITORY_OWNER" in index_step
    assert "outside current owner namespace" in index_step
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


def test_pr_cleanup_matches_owner_prefixed_runtime_tags() -> None:
    cleanup = (WORKFLOWS / "ucm-pr-cleanup.yml").read_text(encoding="utf-8")

    assert 'regex="(^|-)pr-${pr_number}-"' in cleanup
    assert 'regex="(^|-)pr-[0-9]+-"' in cleanup


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
    assert "python3 /tmp/validate_wheel_runtime.py" in run
    assert ":/tmp/validate_wheel_runtime.py:ro" in run
    assert 'EXPECTED_OS_ID}" != linux' in run
    assert 'EXPECTED_OS_VERSION}" != unreported' in run

    dockerfile = (
        ROOT / ".github" / "release" / "docker" / "Dockerfile.runtime"
    ).read_text(encoding="utf-8")
    assert "python3 -m pip install" in dockerfile
    assert "--index-url https://pypi.org/simple" in dockerfile
    assert "validate_wheel_runtime.py" not in dockerfile
    assert "python3 -c 'import ucm'" not in dockerfile


def test_trusted_tag_image_publishes_directly_without_oci_artifact() -> None:
    workflow = _load("_build-release-image.yml")
    assert set(workflow["on"]["workflow_call"]["inputs"]) == {
        "image_id",
        "runner",
        "plan_artifact",
        "source_ref",
        "publication_scope",
    }
    job = workflow["jobs"]["publish"]
    assert job["environment"] == (
        "${{ inputs.publication_scope == 'official' && "
        "'release-production' || 'fork-preview' }}"
    )
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    text = (WORKFLOWS / "_build-release-image.yml").read_text(encoding="utf-8")
    assert "ref: ${{ inputs.source_ref }}" in text
    assert "GITHUB_EVENT_NAME" not in text
    assert "GITHUB_REF_TYPE" not in text
    assert "GITHUB_REF_NAME" not in text
    assert ".publish.ghcr.enabled" in text
    assert ".publish.dockerhub.enabled" in text
    assert "dockerhub_ref" in text
    assert 'skopeo copy "oci-archive:out/image.oci.tar"' in text
    assert "ucm-image-member-receipt-${{ inputs.image_id }}" in text
    assert "path: out/image.oci.tar" not in text
    release = _load("release-ucm.yml")["jobs"]
    assert release["build-images"]["uses"] == (
        "./.github/workflows/_build-release-image.yml"
    )
    assert release["build-images"]["with"]["source_ref"] == "${{ inputs.source_sha }}"
    assert release["build-images"]["with"]["publication_scope"] == (
        "${{ inputs.publication_scope }}"
    )
    assert "build-validation-images" not in release


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
