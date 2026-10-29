"""Functional contract for the shared compact Release/PR planner."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
TAG_FIXTURE = RELEASE_ROOT / "tests" / "fixtures" / "catalog-registry.json"
sys.path.insert(0, str(RELEASE_ROOT))

builders = importlib.import_module("ucm_release.builders")
chart = importlib.import_module("ucm_release.chart")
cleanup_records = importlib.import_module("ucm_release.cleanup_records")
wheel_ops = importlib.import_module("ucm_release.wheel")
planner = importlib.import_module("ucm_release.plan")
serialization = importlib.import_module("ucm_release.serialization")
policy = importlib.import_module("ucm_release.policy")
upstream = importlib.import_module("ucm_release.upstream")
version_config = importlib.import_module("ucm_release.version_config")


def _fixture_policy(
    release_type: str = "stable",
    repository: str = "release-org/unified-cache-management",
):
    resolved = policy.resolve(
        repository=repository,
        version_override="0.7.60rc1",
        release_type=release_type,
        dockerhub_namespace="docker.io/release-test",
    )
    selectors = {
        "vllm": [{"raw": "0.22.1", "version": "0.22.1", "tag": None}],
        "vllm-ascend": [{"raw": "0.22.1", "version": "0.22.1", "tag": None}],
    }
    resolved["products"] = [
        product for product in resolved["products"] if product["id"] in selectors
    ]
    resolved["runtime_selectors"] = copy.deepcopy(selectors)
    for product in resolved["products"]:
        product["runtime_selectors"] = copy.deepcopy(selectors[product["id"]])
    return resolved


def _inputs(
    release_type: str = "stable",
    repository: str = "release-org/unified-cache-management",
):
    formal = _fixture_policy(release_type, repository)
    selection = upstream.resolve_upstreams(
        formal,
        tag_fixture=serialization.load_json(TAG_FIXTURE),
    )
    return formal, selection, _builder_catalog(formal, selection)


def _builder_catalog(formal, selection):
    catalog = builders.catalog_from_builds(
        selection["wheel_builds"],
        owner=formal["repository"].split("/", 1)[0],
        formal_policy=formal,
    )
    observations = {
        item["id"]: {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": builders.builder_labels(item)},
            },
        }
        for index, item in enumerate(catalog["builders"])
    }
    return builders.finalize_catalog(catalog, observations)


def _plan(
    release_type: str = "stable",
    repository: str = "release-org/unified-cache-management",
):
    formal, selection, catalog = _inputs(release_type, repository)
    return planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
    )


@pytest.fixture
def cann_fallback_plan():
    document = _plan()
    for wheel in document["wheels"]:
        if wheel["backend"].startswith("cann-"):
            wheel["builder"]["source_image"] = wheel["builder"]["source_image"].replace(
                "9.0.1-", "9.0.1-beta.2-"
            )
    return document


def test_fallback_matrix_covers_every_runtime_member(cann_fallback_plan) -> None:
    document = cann_fallback_plan
    image = next(
        image
        for image in document["images"]
        if image["runtime"]["accelerator_runtime"].startswith("cann-")
    )
    second = copy.deepcopy(image)
    second["id"] += "-openeuler"
    second["runtime"]["image_reference"] = (
        "quay.io/ascend/vllm-ascend@sha256:" + "d" * 64
    )
    document["images"].append(second)

    selected = planner.fallback_runtime_matrix(document)["include"]
    expected = [
        image
        for image in document["images"]
        if image["runtime"]["accelerator_runtime"].startswith("cann-")
    ]
    assert [
        (item["wheel_id"], item["runtime_image"], item["runner"]) for item in selected
    ] == [
        (image["wheel_id"], image["runtime"]["image_reference"], image["runner"])
        for image in expected
    ]
    assert all("id" not in item for item in selected)
    assert {item["runtime_cann_version"] for item in selected} == {"9.0.1"}
    assert {item["builder_cann_version"] for item in selected} == {"9.0.1-beta.2"}


@pytest.mark.parametrize("version", ["9.0.1", "9.0.1-beta.2"])
def test_exact_cann_versions_need_no_fallback_matrix(cann_fallback_plan, version):
    document = cann_fallback_plan
    for image in document["images"]:
        if image["runtime"]["accelerator_runtime"].startswith("cann-"):
            image["runtime"]["accelerator_runtime"] = "cann-" + version
    for wheel in document["wheels"]:
        wheel["builder"]["source_image"] = wheel["builder"]["source_image"].replace(
            "9.0.1-beta.2-", version + "-"
        )
    assert planner.fallback_runtime_matrix(document) == {"include": []}


@pytest.mark.parametrize("version", ["9.1.0", "9.0.1-beta.1"])
def test_fallback_matrix_rejects_unapproved_substitution(cann_fallback_plan, version):
    image = next(
        image
        for image in cann_fallback_plan["images"]
        if image["runtime"]["accelerator_runtime"].startswith("cann-")
    )
    image["runtime"]["accelerator_runtime"] = "cann-" + version
    with pytest.raises(ValueError, match="unsupported CANN Builder substitution"):
        planner.fallback_runtime_matrix(cann_fallback_plan)


def _write_wheel_fixture(
    path: Path,
    metadata_tag: str,
    dependencies: tuple[str, ...] = ("wrapt==1.17.2",),
) -> None:
    distribution, version, _build, _tags = parse_wheel_filename(path.name)
    dist_info = f"{str(distribution).replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "\n".join(
                (
                    "Wheel-Version: 1.0",
                    "Generator: ucm-release-test",
                    "Root-Is-Purelib: false",
                    f"Tag: {metadata_tag}",
                    "",
                )
            ),
        )
        wheel.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                (
                    "Metadata-Version: 2.2",
                    f"Name: {distribution}",
                    f"Version: {version}",
                    *(f"Requires-Dist: {dependency}" for dependency in dependencies),
                    "",
                )
            ),
        )


def _wheel_result_task(cpu_arch: str) -> dict[str, str]:
    wheel_arch = {"amd64": "x86_64", "arm64": "aarch64"}[cpu_arch]
    target = f"manylinux_2_28_{wheel_arch}"
    return {
        "id": f"cuda130-cp312-{cpu_arch}",
        "dist_name": "uc-manager-cuda-cu130",
        "wheel_version": "0.7.60rc1",
        "python_abi": "cp312",
        "cpu_arch": cpu_arch,
        "target_platform_tag": target,
        "external_runtime_exclude_patterns": ["libcudart.so.13"],
        "runtime_requirements": ["wrapt==1.17.2"],
        "repair": {
            "tool": "auditwheel",
            "version": "6.7.0",
            "target_platform": target,
            "excluded_patterns": ["libcudart.so.13"],
        },
    }


def _write_auditwheel_report(
    wheel: Path,
    *,
    compatible_platform: str,
    constrained_platform: str | None = None,
    reported_filename: str | None = None,
    glibc_versions: tuple[str, ...] = ("GLIBC_2.17",),
    external_libraries: tuple[str, ...] = ("libcudart.so.13",),
    direct_external_libraries: tuple[str, ...] = ("libcudart.so.13",),
    external_library_paths: dict[str, str | None] | None = None,
) -> Path:
    report = wheel.with_name(wheel_ops.AUDITWHEEL_REPORT)
    direct_external = sorted(set(direct_external_libraries) & set(external_libraries))
    external_paths = external_library_paths or {
        name: None if name in direct_external else f"/vendor/{name}"
        for name in external_libraries
    }
    libraries = json.dumps(external_paths, indent=4)
    dependencies = {soname: [] for soname in external_libraries}
    if direct_external:
        dependencies[direct_external[0]] = sorted(
            set(external_libraries) - set(direct_external)
        )
    elf_trees = {
        "ucm/test-extension.so": {
            "needed": direct_external,
            "libraries": {
                soname: {"needed": required}
                for soname, required in dependencies.items()
            },
        }
    }
    constraint = (
        f'This constrains the platform tag to "{constrained_platform}". In order '
        "to achieve a more compatible tag, you would need to recompile a new Wheel."
        if constrained_platform is not None
        else ""
    )
    report.write_text(
        "\n".join(
            (
                "DEBUG:auditwheel.wheel_abi:full_elftree:",
                json.dumps(elf_trees, sort_keys=True),
                f"{reported_filename or wheel.name} is consistent with the following "
                f'platform tag: "{compatible_platform}".',
                "",
                "The wheel references external versioned symbols in these system-provided shared libraries: "
                f"libc.so.6 with versions {set(glibc_versions)!r}",
                "",
                constraint,
                "",
                "The following external shared libraries are required by the wheel:",
                libraries,
                "",
            )
        ),
        encoding="utf-8",
    )
    return report


def test_plan_projects_only_runtime_referenced_wheel_union() -> None:
    formal, selection, catalog = _inputs()
    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
    )
    runtimes = selection["runtimes"]
    assert {wheel["id"] for wheel in plan["wheels"]} == {
        wheel_id
        for runtime in runtimes
        for wheel_id in runtime["wheel_build_ids"].values()
    }
    assert len(plan["images"]) == sum(
        len(runtime["architectures"]) for runtime in runtimes
    )
    assert len(plan["families"]) == len(runtimes)
    assert sum(bool(item["create_index"]) for item in plan["families"]) == sum(
        len(runtime["architectures"]) > 1 for runtime in runtimes
    )
    assert sorted(len(item["members"]) for item in plan["families"]) == sorted(
        len(runtime["architectures"]) for runtime in runtimes
    )
    assert len({item["id"] for item in plan["families"]}) == len(plan["families"])
    assert len(
        {(item["target_repository"], item["target_tag"]) for item in plan["families"]}
    ) == len(plan["families"])
    assert all(
        item["builder"]["digest"].startswith("sha256:") for item in plan["wheels"]
    )


def _opaque_pin_inputs(
    product_id: str,
    backend: str,
    tags: list[str],
    *,
    version: str = "0.7.60rc1",
    release_type: str = "prerelease",
):
    formal = policy.resolve(
        repository="release-org/unified-cache-management",
        version_override=version,
        release_type=release_type,
        dockerhub_namespace="docker.io/release-test",
    )
    key = version_config.SUPPORTED_VERSION_KEYS[product_id]
    configured_tags = ",".join(f"@{tag}" for tag in tags)
    formal["runtime_selectors"] = version_config.parse(
        f"UCM_VERSION={version}\n{key}={configured_tags}\n"
    )["supported_runtimes"]
    for product in formal["products"]:
        product["runtime_selectors"] = formal["runtime_selectors"][product["id"]]
    product = next(item for item in formal["products"] if item["id"] == product_id)
    source = product["runtime_repository"]
    fixture = serialization.load_json(TAG_FIXTURE)
    fixture["repositories"][source] = {"pages": [{"tags": tags}]}
    source_probes = [
        probe
        for probe in fixture["runtime_probe"]["probes"]
        if probe["backend"] == backend
    ]
    probes = []
    for index, tag in enumerate(tags):
        for source_probe in source_probes:
            probe = copy.deepcopy(source_probe)
            request_id = f"runtime-pin-{index}"
            probe.update(
                probe_id=f"{request_id}-{probe['cpu_arch']}",
                request_id=request_id,
                product_id=product_id,
                repository=source,
                tag=tag,
                runtime_ref=f"{source}:{tag}",
                target_repository=product["target_repository"],
                image_reference=f"{source}@{probe['image_reference'].split('@')[1]}",
            )
            probes.append(probe)
    fixture["runtime_probe"]["probes"] = probes
    return formal, fixture


@pytest.mark.parametrize(
    ("product_id", "backend", "tag"),
    [
        ("vllm-ascend", "cann-a2", "custom-310p"),
        ("sglang", "cann-a3", "custom-cu129-runtime"),
    ],
)
def test_opaque_tags_use_probed_capabilities_instead_of_name_hints(
    product_id, backend, tag
):
    formal, fixture = _opaque_pin_inputs(product_id, backend, [tag])
    selection = upstream.resolve_upstreams(formal, tag_fixture=fixture)
    runtime = selection["runtimes"][0]
    assert (runtime["runtime_tag"], runtime["version"], runtime["channel"]) == (
        tag,
        tag,
        "pinned",
    )
    assert runtime["backend"] == backend


@pytest.fixture
def named_draft_inputs():
    coordinates = version_config.classify_tag("draft/v0.9.0-1-test")
    formal, fixture = _opaque_pin_inputs(
        "vllm",
        "cuda",
        ["Release", "release"],
        version=coordinates["version"],
        release_type="draft",
    )
    selection = upstream.resolve_upstreams(formal, tag_fixture=fixture)
    return {
        "catalog": formal,
        "runtime_selection": selection,
        "builder_catalog": _builder_catalog(formal, selection),
    }, coordinates


def test_named_draft_plan_preserves_identity_and_case_distinct_runtime_pins(
    named_draft_inputs,
):
    inputs, coordinates = named_draft_inputs
    plan = planner.resolve_plan(**inputs, route="release")
    version = coordinates["version"]
    assert plan["git_tag"] == coordinates["git_tag"]
    assert plan["release_kind"] == "draft" and plan["is_prerelease"] is True
    assert plan["version"] == plan["chart"]["app_version"] == version
    assert plan["chart"]["version"] == coordinates["chart_version"]
    assert plan["image_version"] == coordinates["image_version"]
    assert {wheel["wheel_version"] for wheel in plan["wheels"]} == {version}
    assert (
        plan["meta_package"]["version"] == plan["toolkit_package"]["version"] == version
    )
    assert all(
        requirement.endswith(f"=={version}")
        for requirement in plan["meta_package"]["extras"].values()
    )
    families = plan["families"]
    assert {family["runtime"]["tag"] for family in families} == {"Release", "release"}
    assert len({family["id"] for family in families}) == len(families)
    assert len({family["published_reference"] for family in families}) == len(families)
    assert {wheel["id"] for wheel in plan["wheels"]} == set(
        inputs["runtime_selection"]["runtimes"][0]["wheel_build_ids"].values()
    )
    rendered = chart.render_values(
        plan, (ROOT / plan["chart"]["source"] / "values.yaml").read_text()
    )
    for family in families:
        assert f"{family['runtime']['tag']}-ucm-{plan['image_version']}" in rendered


@pytest.mark.parametrize(
    "overrides", [{"chart_version": "1.0.0"}, {"release_kind": "publish"}]
)
def test_named_draft_plan_rejects_inconsistent_publication_coordinates(
    named_draft_inputs, overrides
):
    inputs, _ = named_draft_inputs
    with pytest.raises(ValueError, match="custom Draft Tag coordinates differ"):
        planner.resolve_plan(**inputs, route="release", **overrides)


@pytest.mark.parametrize("restriction", ["blocked-backend", "excluded-variant"])
def test_opaque_pin_rejects_probed_platform_before_builder_resolution(
    monkeypatch, restriction
):
    formal, fixture = _opaque_pin_inputs("vllm-ascend", "cann-a2", ["aaaaa-cc"])
    if restriction == "blocked-backend":
        formal["backends"]["cann-a2"].update(
            status="blocked", reason="Disabled by platform policy"
        )
    else:
        formal["excluded_upstream_variants"]["vllm-ascend"] = ["a2"]

    def unexpected_builder_resolution(*args, **kwargs):
        pytest.fail("a rejected Runtime must not query or resolve Builders")

    monkeypatch.setattr(builders, "resolve_probe_builds", unexpected_builder_resolution)
    with pytest.raises(ValueError, match="probed Runtime.*(blocked|excluded)"):
        upstream.resolve_upstreams(formal, tag_fixture=fixture)


def test_plan_projects_runtime_referenced_distributions_from_platform_policy() -> None:
    plan = _plan()

    assert plan["publish"]["pypi"]["distributions"] == [
        "release-org-uc-manager-cann901-a2",
        "release-org-uc-manager-cann901-a3",
        "release-org-uc-manager-cann901-a5",
        "release-org-uc-manager-cuda-cu129",
        "release-org-ucm-toolkit",
    ]
    assert {item["dist_name"] for item in plan["wheels"]} == set(
        plan["publish"]["pypi"]["distributions"]
    ) - {plan["toolkit_package"]["distribution"]}
    assert plan["meta_package"]["extras"]["toolkit"] == (
        f"{plan['toolkit_package']['distribution']}=={plan['version']}"
    )
    assert plan["meta_package"] == {
        "distribution": "release-org-uc-manager",
        "version": "0.7.60rc1",
        "extras": {
            "cann901-a2": "release-org-uc-manager-cann901-a2==0.7.60rc1",
            "cann901-a3": "release-org-uc-manager-cann901-a3==0.7.60rc1",
            "cann901-a5": "release-org-uc-manager-cann901-a5==0.7.60rc1",
            "cu129": "release-org-uc-manager-cuda-cu129==0.7.60rc1",
            "toolkit": "release-org-ucm-toolkit==0.7.60rc1",
        },
    }
    assert all(
        wheel["target_platform_tag"].startswith("manylinux_")
        and wheel["repair"]["target_platform"] == wheel["target_platform_tag"]
        and wheel["repair"]["excluded_patterns"]
        == wheel["external_runtime_exclude_patterns"]
        and "runtime_deferred_libraries" not in wheel
        for wheel in plan["wheels"]
    )


@pytest.mark.parametrize(
    ("accelerator_runtime", "expected"),
    (("cuda-12.9", "libcudart.so.12"), ("cuda-13.0", "libcudart.so.13")),
)
def test_cuda_external_boundary_follows_accelerator_major(
    accelerator_runtime: str, expected: str
) -> None:
    assert planner._external_runtime_exclude_patterns(
        {"external_runtime_exclude_patterns": ["libcudart.so.{accelerator_major}"]},
        {"accelerator_runtime": accelerator_runtime},
    ) == [expected]


def test_compact_source_accepts_exact_prefixed_distribution(tmp_path: Path) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "uc-manager"\n', encoding="utf-8")

    result = wheel_ops.prepare_wheel_source(
        tmp_path, "supermarioyl-uc-manager-cuda-cu130"
    )

    assert result["distribution"] == "supermarioyl-uc-manager-cuda-cu130"
    assert 'name = "supermarioyl-uc-manager-cuda-cu130"' in project.read_text(
        encoding="utf-8"
    )


def test_plan_rejects_repository_scope_or_runtime_prefix_drift() -> None:
    formal, selection, catalog = _inputs()
    wrong_scope = copy.deepcopy(formal)
    wrong_scope["publication_scope"] = "official"
    with pytest.raises(ValueError, match="scope does not match"):
        planner.resolve_plan(
            wrong_scope,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )

    missing_prefix = copy.deepcopy(formal)
    missing_prefix["runtime_image_tag_prefix"] = ""
    with pytest.raises(ValueError, match="prefix does not match"):
        planner.resolve_plan(
            missing_prefix,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )

    wrong_python_prefix = copy.deepcopy(formal)
    wrong_python_prefix["publish"]["pypi"]["distribution_prefix"] = "other-owner-"
    with pytest.raises(ValueError, match="distribution prefix"):
        planner.resolve_plan(
            wrong_python_prefix,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )

    shared_channel = copy.deepcopy(formal)
    shared_channel["publish"]["pypi"]["enabled"] = True
    with pytest.raises(ValueError, match="publication decision does not match"):
        planner.resolve_plan(
            shared_channel,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )

    wrong_disposition = copy.deepcopy(formal)
    wrong_disposition["publish"]["dockerhub"].update(
        enabled=False,
        disposition="publish",
    )
    with pytest.raises(ValueError, match="publication decision does not match"):
        planner.resolve_plan(
            wrong_disposition,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )


@pytest.mark.parametrize(
    "repository",
    (policy.OFFICIAL_REPOSITORY, "release-org/unified-cache-management"),
)
def test_release_plan_requires_requested_dockerhub_namespace(
    repository: str,
) -> None:
    formal, selection, catalog = _inputs(repository=repository)
    formal["release_profile"]["publish"]["dockerhub"] = True
    formal["publish"]["dockerhub"] = {
        "requested": True,
        "enabled": False,
        "disposition": "scope-skipped",
    }

    with pytest.raises(ValueError, match="publication decision does not match"):
        planner.resolve_plan(
            formal,
            runtime_selection=selection,
            builder_catalog=catalog,
            route="release",
        )


def test_pr_plan_does_not_require_release_dockerhub_configuration() -> None:
    formal, selection, catalog = _inputs()
    formal["release_profile"]["publish"]["dockerhub"] = True
    formal["publish"]["dockerhub"] = {
        "requested": True,
        "enabled": False,
        "disposition": "scope-skipped",
    }

    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )

    assert plan["publish"]["dockerhub"] == formal["publish"]["dockerhub"]


def test_draft_coordinates_are_owned_by_the_plan_contract() -> None:
    formal, selection, catalog = _inputs("draft")

    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="release",
        git_tag="draft/v0.7.62-4",
        release_kind="draft",
        is_prerelease=True,
        chart_version="0.7.62-draft.4",
    )

    assert plan["git_tag"] == "draft/v0.7.62-4"
    assert plan["release_type"] == "draft"
    assert plan["release_kind"] == "draft"
    assert plan["is_prerelease"] is True
    assert plan["chart"]["version"] == "0.7.62-draft.4"


@pytest.mark.parametrize(
    ("release_type", "version", "git_tag", "repository", "prefix"),
    (
        ("prerelease", "0.9.0rc2", "v0.9.0rc2", policy.OFFICIAL_REPOSITORY, ""),
        (
            "draft",
            "0.9.0.dev1",
            "draft/v0.9.0-1",
            "release-org/unified-cache-management",
            "release-org-",
        ),
    ),
    ids=("prerelease-official", "draft-fork"),
)
def test_sglang_versioned_images_have_release_owned_cleanup_coordinates(
    release_type: str,
    version: str,
    git_tag: str,
    repository: str,
    prefix: str,
) -> None:
    formal = policy.resolve(
        repository=repository,
        version_override=version,
        release_type=release_type,
        dockerhub_namespace="docker.io/release-test",
    )
    product = next(item for item in formal["products"] if item["id"] == "sglang")
    selectors = [
        {"raw": value, "version": value, "tag": None} for value in ("0.5.19", "0.5.20")
    ]
    product["runtime_selectors"] = selectors
    formal["products"] = [product]
    formal["runtime_selectors"] = {"sglang": selectors}

    fixture = serialization.load_json(TAG_FIXTURE)
    source = "docker.io/lmsysorg/sglang"
    tags = {"cuda": "v0.5.19-cu129", "cann-a3": "v0.5.20rc1-cann9.0.1-a3"}
    probes = [
        probe
        for probe in fixture["runtime_probe"]["probes"]
        if probe["backend"] in tags
    ]
    for probe in probes:
        tag = tags[probe["backend"]]
        probe.update(
            product_id="sglang",
            repository=source,
            tag=tag,
            runtime_ref=f"{source}:{tag}",
            target_repository=product["target_repository"],
            image_reference=f"{source}@{probe['image_reference'].split('@')[1]}",
        )
    fixture["runtime_probe"]["probes"] = probes
    # Main tags have no probes because they must never become build candidates.
    fixture["repositories"][source] = {
        "pages": [
            {"tags": [*tags.values(), "main-cann9.0.0-910b", "main-cann9.0.0-a3"]}
        ]
    }

    selection = upstream.resolve_upstreams(formal, tag_fixture=fixture)
    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=_builder_catalog(formal, selection),
        route="release",
        git_tag=git_tag,
        release_kind="draft" if release_type == "draft" else "publish",
    )
    assert {item["runtime"]["channel"] for item in plan["families"]} == {
        "stable",
        "rc",
    }
    source_members = {
        (probe["tag"], probe["cpu_arch"]): probe["image_reference"] for probe in probes
    }
    assert {
        (image["runtime"]["tag"], image["cpu_arch"]): image["runtime"][
            "image_reference"
        ]
        for image in plan["images"]
    } == source_members

    owner = repository.split("/", 1)[0].lower()
    cuda_tag = f"{prefix}{tags['cuda']}-ucm-{version}"
    cann_tag = f"{prefix}{tags['cann-a3']}-ucm-{version}"
    families = {family["runtime"]["tag"]: family for family in plan["families"]}
    assert set(families) == set(tags.values())
    cuda_family = families[tags["cuda"]]
    assert cuda_family["target_tag"] == cuda_tag
    assert cuda_family["create_index"] is True
    assert cuda_family["published_reference"] == f"ghcr.io/{owner}/sglang:{cuda_tag}"
    assert {member["reference"] for member in cuda_family["members"]} == {
        f"ghcr.io/{owner}/sglang:{cuda_tag}-{architecture}"
        for architecture in ("amd64", "arm64")
    }
    cann_family = families[tags["cann-a3"]]
    assert cann_family["target_tag"] == cann_tag
    assert cann_family["create_index"] is False
    assert cann_family["published_reference"] == f"ghcr.io/{owner}/sglang:{cann_tag}"

    record = cleanup_records.record_from_plan(
        cleanup_records.record_for_tag(repository, git_tag, "a" * 40, 12345, 1),
        plan,
    )
    expected = set()
    channels = {"ghcr": f"ghcr.io/{owner}/sglang"}
    if release_type == "prerelease":
        channels["dockerhub"] = "docker.io/release-test/sglang"
    for channel, target in channels.items():
        expected.add((f"{channel}-index", f"{target}:{cuda_tag}"))
        expected.add((f"{channel}-member", f"{target}:{cann_tag}"))
        expected.update(
            (f"{channel}-member", f"{target}:{cuda_tag}-{architecture}")
            for architecture in ("amd64", "arm64")
        )
    assert {
        (item["kind"], item["reference"])
        for item in record["resources"]
        if item["kind"] != "chart-oci"
    } == expected


def test_fork_owner_prefix_changes_runtime_and_python_coordinates() -> None:
    formal, selection, builder_catalog = _inputs()
    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=builder_catalog,
        route="release",
    )

    assert all(
        family["target_tag"].startswith("release-org-") for family in plan["families"]
    )
    assert all(
        wheel["dist_name"].startswith("release-org-uc-manager-")
        for wheel in plan["wheels"]
    )
    assert all(
        not builder["target_tag"].startswith("release-org-")
        for builder in builder_catalog["builders"]
    )
    assert plan["chart"]["version"] == "0.7.60-rc.1"


def test_fork_python_coordinates_do_not_depend_on_secret_availability() -> None:
    formal, selection, builder_catalog = _inputs()
    formal = copy.deepcopy(formal)
    # This compact-plan test supplies its own request instead of inheriting defaults.
    formal["release_profile"]["publish"]["pypi"] = True
    formal["publish"]["pypi"].update(
        requested=True,
        enabled=False,
        disposition="scope-skipped",
    )
    disabled = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=builder_catalog,
        route="release",
    )
    enabled_formal = copy.deepcopy(formal)
    enabled_formal["publish"]["pypi"].update(
        enabled=True,
        disposition="publish",
    )
    enabled = planner.resolve_plan(
        enabled_formal,
        runtime_selection=selection,
        builder_catalog=builder_catalog,
        route="release",
    )

    assert disabled["publish"]["pypi"]["enabled"] is False
    assert enabled["publish"]["pypi"]["enabled"] is True
    assert disabled["publish"]["pypi"]["distribution_prefix"] == "release-org-"
    assert enabled["publish"]["pypi"]["distribution_prefix"] == "release-org-"
    assert [wheel["dist_name"] for wheel in disabled["wheels"]] == [
        wheel["dist_name"] for wheel in enabled["wheels"]
    ]
    assert disabled["meta_package"] == enabled["meta_package"]


def test_formal_all_plan_can_be_retagged_for_pr_publication() -> None:
    formal, selection, catalog = _inputs()
    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )
    retagged = planner.retag_pr_plan(
        plan, pr_number=12, author="Release-Author", run_id=345
    )

    assert all(
        family["target_tag"].startswith("release-org-pr-12-release-author-run-345-")
        for family in retagged["families"]
    )
    assert all(
        image["target_tag"]
        == next(
            family["target_tag"]
            for family in retagged["families"]
            if family["id"] == image["family_id"]
        )
        for image in retagged["images"]
    )


def test_single_arch_pr_build_has_one_member_and_no_index() -> None:
    formal, selection, catalog = _inputs()
    selection = copy.deepcopy(selection)
    runtime = next(
        item for item in selection["runtimes"] if item["id"] == "vllm-v0.22.1-cu129"
    )
    runtime["architectures"] = ["amd64"]
    runtime["member_references"] = {"amd64": runtime["member_references"]["amd64"]}
    runtime["wheel_build_ids"] = {"amd64": runtime["wheel_build_ids"]["amd64"]}
    runtime["channel"] = "pinned"
    selection["runtimes"] = [runtime]

    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )

    assert (len(plan["wheels"]), len(plan["images"]), len(plan["families"])) == (
        1,
        1,
        1,
    )
    family = plan["families"][0]
    assert family["create_index"] is False
    assert family["published_reference"] == (
        f"{family['target_repository']}:{family['target_tag']}"
    )
    assert family["members"] == [
        {
            "image_id": "vllm-v0.22.1-cu129-amd64",
            "cpu_arch": "amd64",
            "reference": family["published_reference"],
        }
    ]


def test_multiple_pr_runtime_tags_with_same_capability_reuse_wheels() -> None:
    formal, selection, catalog = _inputs()
    selection = copy.deepcopy(selection)
    first = next(item for item in selection["runtimes"] if item["product_id"] == "vllm")
    second = copy.deepcopy(first)
    second["id"] = "vllm-v0.22.1-cu129-ubuntu2404"
    second["runtime_tag"] = "v0.22.1-cu129-ubuntu2404"
    second["runtime_digest"] = "sha256:" + "d" * 64
    second["os_version"] = "24.04"
    second["target_tag"] = "release-org-v0.22.1-cu129-ubuntu2404-ucm-0.7.60rc1"
    selected = [first, second]
    for runtime in selected:
        runtime["channel"] = "pinned"
    selection["runtimes"] = selected

    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=catalog,
        route="pr",
    )

    assert len(plan["wheels"]) == 2
    assert len(plan["images"]) == 4
    assert len(plan["families"]) == 2


def test_wheel_result_manifest_uses_actual_filename_tags(tmp_path: Path) -> None:
    version = "0.9.0.dev0+draft.1.test"
    task = _wheel_result_task("amd64")
    task["wheel_version"] = version
    wheel = tmp_path / (
        f"uc_manager_cuda_cu130-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
    )
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_28_x86_64")
    report = _write_auditwheel_report(
        wheel,
        compatible_platform="linux_x86_64",
        constrained_platform="manylinux_2_27_x86_64",
    )
    result = wheel_ops.record_wheel_result(task, wheel)
    assert result["task_id"] == "cuda130-cp312-amd64"
    assert result["filename"] == wheel.name
    assert result["version"] == version
    assert result["schema_version"] == 5
    assert result["platform_tags"] == ["manylinux_2_28_x86_64"]
    assert result["auditwheel_platform_tag"] == "linux_x86_64"
    assert result["abi_compatible_platform_tag"] == "manylinux_2_27_x86_64"
    assert result["glibc_versions"] == ["GLIBC_2.17"]
    assert result["glibc_floor"] == "GLIBC_2.17"
    assert result["external_library_roots"] == ["libcudart.so.13"]
    assert result["external_libraries"] == ["libcudart.so.13"]
    assert "runtime_deferred_libraries" not in result
    assert result["deferred_external_libraries"] == []
    assert result["repair"] == task["repair"]
    assert len(result["sha256"]) == 64
    assert result["auditwheel_report"]["filename"] == report.name
    assert result["auditwheel_report"]["text"] == report.read_text(encoding="utf-8")
    assert len(result["auditwheel_report"]["sha256"]) == 64


def test_wheel_result_rejects_an_external_library_outside_configured_roots(
    tmp_path: Path,
) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-" "manylinux_2_28_x86_64.whl"
    )
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_28_x86_64")
    _write_auditwheel_report(
        wheel,
        compatible_platform="manylinux_2_27_x86_64",
        external_libraries=("libcudart.so.13", "libunexpected.so"),
        direct_external_libraries=("libcudart.so.13", "libunexpected.so"),
    )

    with pytest.raises(ValueError, match="outside the provider boundary"):
        wheel_ops.record_wheel_result(task, wheel)


def test_wheel_result_maps_arm64_to_aarch64_only(tmp_path: Path) -> None:
    task = _wheel_result_task("arm64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-" "manylinux_2_28_aarch64.whl"
    )
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_28_aarch64")
    _write_auditwheel_report(
        wheel,
        compatible_platform="linux_aarch64",
        constrained_platform="manylinux_2_26_aarch64",
    )

    result = wheel_ops.record_wheel_result(task, wheel)

    assert result["cpu_arch"] == "arm64"
    assert result["platform_tags"] == ["manylinux_2_28_aarch64"]
    assert result["auditwheel_platform_tag"] == "linux_aarch64"


def test_wheel_result_rejects_report_for_a_different_wheel(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-" "manylinux_2_28_x86_64.whl"
    )
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_28_x86_64")
    _write_auditwheel_report(
        wheel,
        compatible_platform="linux_x86_64",
        constrained_platform="manylinux_2_27_x86_64",
        reported_filename="different-1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    )

    with pytest.raises(ValueError, match="different Wheel"):
        wheel_ops.record_wheel_result(task, wheel)


@pytest.mark.parametrize(
    "filename",
    [
        "wrong-0.7.60rc1-cp312-cp312-manylinux_2_28_x86_64.whl",
        "uc_manager_cuda_cu130-0.7.60rc1-cp311-cp311-manylinux_2_28_x86_64.whl",
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-manylinux_2_28_aarch64.whl",
    ],
)
def test_wheel_result_rejects_distribution_abi_and_arch_drift(
    tmp_path: Path, filename: str
) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / filename
    parsed_tag = "-".join(filename.removesuffix(".whl").rsplit("-", 3)[-3:])
    _write_wheel_fixture(wheel, parsed_tag)
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_27_x86_64")
    with pytest.raises(ValueError):
        wheel_ops.record_wheel_result(task, wheel)


@pytest.mark.parametrize(
    ("task_id", "wheel_platform", "error"),
    [
        ("cuda130-cp312-amd64", "linux_amd64", "must use"),
        ("cuda130-cp312-amd64", "manylinux_2_29_x86_64", "must include"),
    ],
)
def test_wheel_result_rejects_generic_alias_or_wrong_manylinux_target(
    tmp_path: Path,
    task_id: str,
    wheel_platform: str,
    error: str,
) -> None:
    task = _wheel_result_task(task_id.rsplit("-", 1)[-1])
    wheel = tmp_path / (
        f"{task['dist_name'].replace('-', '_')}-{task['wheel_version']}-"
        f"cp312-cp312-{wheel_platform}.whl"
    )
    _write_wheel_fixture(wheel, f"cp312-cp312-{wheel_platform}")
    wheel_arch = "aarch64" if task["cpu_arch"] == "arm64" else "x86_64"
    _write_auditwheel_report(wheel, compatible_platform=f"manylinux_2_27_{wheel_arch}")

    with pytest.raises(ValueError, match=error):
        wheel_ops.record_wheel_result(task, wheel)


def test_wheel_result_accepts_auditwheel_compressed_lower_floor_tag(
    tmp_path: Path,
) -> None:
    task = _wheel_result_task("amd64")
    platforms = "manylinux_2_27_x86_64.manylinux_2_28_x86_64"
    wheel = tmp_path / (f"uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-{platforms}.whl")
    _write_wheel_fixture(wheel, f"cp312-cp312-{platforms}")
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_27_x86_64")

    result = wheel_ops.record_wheel_result(task, wheel)

    assert result["platform_tags"] == [
        "manylinux_2_27_x86_64",
        "manylinux_2_28_x86_64",
    ]


def test_wheel_result_rejects_filename_metadata_tag_drift(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-" "manylinux_2_28_x86_64.whl"
    )
    _write_wheel_fixture(wheel, "cp312-cp312-manylinux_2_27_x86_64")
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_27_x86_64")

    with pytest.raises(ValueError, match="filename and WHEEL metadata"):
        wheel_ops.record_wheel_result(task, wheel)


def test_wheel_result_rejects_backend_dependency_drift(tmp_path: Path) -> None:
    task = _wheel_result_task("amd64")
    wheel = tmp_path / (
        "uc_manager_cuda_cu130-0.7.60rc1-cp312-cp312-" "manylinux_2_28_x86_64.whl"
    )
    _write_wheel_fixture(
        wheel,
        "cp312-cp312-manylinux_2_28_x86_64",
        dependencies=("wrapt==1.17.1",),
    )
    _write_auditwheel_report(wheel, compatible_platform="manylinux_2_27_x86_64")

    with pytest.raises(ValueError, match="dependencies do not match"):
        wheel_ops.record_wheel_result(task, wheel)


@pytest.mark.parametrize(
    ("route", "release_type"), [("pr", "stable"), ("release", "nightly")]
)
def test_wheel_plan_can_build_directly_from_pinned_upstream_builders(
    route: str, release_type: str
) -> None:
    formal, selection, _finalized_catalog = _inputs(release_type)
    desired = builders.catalog_from_builds(
        selection["wheel_builds"],
        owner="release-org",
        formal_policy=formal,
    )

    plan = planner.resolve_plan(
        formal,
        runtime_selection=selection,
        builder_catalog=builders.bind_source_catalog(desired),
        route=route,
    )

    builds = {item["id"]: item for item in selection["wheel_builds"]}
    assert len(plan["wheels"]) == len(builds)
    assert {wheel["id"] for wheel in plan["wheels"]} == {
        item["id"] for item in plan["wheel_matrix"]["include"]
    }
    for wheel in plan["wheels"]:
        build = builds[wheel["id"]]
        backend = formal["backends"][build["backend"]]
        source_repository, _source_tag = build["source_image"].rsplit(":", 1)
        assert wheel["builder"]["repository"] == source_repository
        assert wheel["builder"]["digest"] == build["source_image_digest"]
        assert wheel["build"]["platform_arg"] == backend["platform"]
