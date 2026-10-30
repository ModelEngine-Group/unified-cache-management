"""Registry-only runtime selection and mirror Builder contracts."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RELEASE_ROOT = ROOT / ".github" / "release"
TAG_FIXTURE = RELEASE_ROOT / "tests" / "fixtures" / "catalog-registry.json"
sys.path.insert(0, str(RELEASE_ROOT))

builders = importlib.import_module("ucm_release.builders")
serialization = importlib.import_module("ucm_release.serialization")
policy = importlib.import_module("ucm_release.policy")
upstream = importlib.import_module("ucm_release.upstream")


def test_sglang_main_tag_is_outside_formal_runtime_grammar() -> None:
    assert (
        upstream._parsed_runtime_tag("sglang", "main-cann9.0.0-a3")  # noqa: SLF001
        is None
    )


def _selector(version: str) -> dict[str, str | None]:
    return {
        "raw": version,
        "version": version,
        "tag": None,
    }


def _tag_selector(tag: str) -> dict[str, str | None]:
    return {"raw": f"@{tag}", "version": None, "tag": tag}


def _policy(
    release_type: str = "stable", *, platforms_path: Path | None = None
) -> dict[str, object]:
    resolved = copy.deepcopy(
        policy.resolve(
            repository="release-org/unified-cache-management",
            version_override="0.7.60rc1",
            release_type=release_type,
            platforms_path=platforms_path,
        )
    )
    selectors = {
        "vllm": [_selector("0.22.1")],
        "vllm-ascend": [_selector("0.22.1")],
    }
    resolved["products"] = [
        product for product in resolved["products"] if product["id"] in selectors
    ]
    resolved["runtime_selectors"] = copy.deepcopy(selectors)
    for product in resolved["products"]:
        product["runtime_selectors"] = copy.deepcopy(selectors[product["id"]])
    return resolved


def _fixture() -> dict[str, object]:
    return serialization.load_json(TAG_FIXTURE)


def _sglang_policy(
    selector: dict[str, str | None] | None = None,
) -> dict[str, object]:
    resolved = policy.resolve(
        repository="release-org/unified-cache-management",
        version_override="0.7.60rc1",
    )
    product = next(item for item in resolved["products"] if item["id"] == "sglang")
    product["runtime_selectors"] = [selector or _selector("0.5")]
    resolved["products"] = [product]
    return resolved


def _selection(fixture: dict[str, object] | None = None) -> dict[str, object]:
    registry = fixture or _fixture()
    return upstream.resolve_upstreams(
        _policy(),
        candidates=upstream.resolve_runtime_candidates(_policy(), tag_fixture=registry),
        runtime_probe=registry["runtime_probe"],
        tag_fixture=registry,
    )


def _catalog(selection: dict[str, object] | None = None) -> dict[str, object]:
    return builders.catalog_from_builds(
        (selection or _selection())["wheel_builds"],
        owner="release-org",
        formal_policy=_policy(),
    )


def test_registry_tag_selection_uses_version_ranges_and_all_winner_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [
            _selector("0.23"),
            _selector("0.24"),
        ],
    }

    selected = upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.23.0rc1",
            "v0.23.0",
            "v0.23.0-a3",
            "nightly-releases-v0.24.0rc",
            "nightly-releases-v0.24.0rc-a3",
        ],
    )

    assert selected == [
        {"runtime_tag": "v0.23.0", "version": "0.23.0", "channel": "stable"},
        {"runtime_tag": "v0.23.0-a3", "version": "0.23.0", "channel": "stable"},
        {
            "runtime_tag": "nightly-releases-v0.24.0rc",
            "version": "0.24.0rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.24.0rc-a3",
            "version": "0.24.0rc",
            "channel": "nightly",
        },
    ]


@pytest.mark.parametrize("version", ("0.5.18", "0.5.19", "0.5.20"))
def test_sglang_selector_keeps_cuda_runtime_and_cann_variants(
    version: str,
) -> None:
    product = {"id": "sglang", "runtime_selectors": [_selector(version)]}

    selected = upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            f"v{version}-cu129",
            f"v{version}-cu130",
            f"v{version}-cu129-runtime",
            f"v{version}-cu130-runtime",
            f"v{version}-cann9.0.0-910b",
            f"v{version}-cann9.0.0-a3",
            f"v{version}-cu129-aarch64",
            "v0.5.21-cu130",
        ],
    )

    assert selected == [
        {
            "runtime_tag": f"v{version}-cann9.0.0-910b",
            "version": version,
            "channel": "stable",
        },
        {
            "runtime_tag": f"v{version}-cann9.0.0-a3",
            "version": version,
            "channel": "stable",
        },
        {"runtime_tag": f"v{version}-cu129", "version": version, "channel": "stable"},
        {
            "runtime_tag": f"v{version}-cu129-runtime",
            "version": version,
            "channel": "stable",
        },
        {"runtime_tag": f"v{version}-cu130", "version": version, "channel": "stable"},
        {
            "runtime_tag": f"v{version}-cu130-runtime",
            "version": version,
            "channel": "stable",
        },
    ]


@pytest.mark.parametrize(
    ("formal_tags", "expected"),
    [
        (
            ["v0.5.19-cu129", "v0.5.19-cann9.0.0-a3", "v0.5.20rc1-cu130"],
            {"v0.5.19-cu129": "stable", "v0.5.19-cann9.0.0-a3": "stable"},
        ),
        (
            ["v0.5.19rc1-cu129", "v0.5.19rc2-cu130", "v0.5.19rc2-cann9.0.0-a3"],
            {"v0.5.19rc2-cu130": "rc", "v0.5.19rc2-cann9.0.0-a3": "rc"},
        ),
    ],
    ids=["stable-before-newer-rc", "rc-fallback"],
)
def test_sglang_candidates_select_formal_tags_without_appending_main(
    formal_tags: list[str], expected: dict[str, str]
) -> None:
    candidates = upstream.resolve_runtime_candidates(
        _sglang_policy(),
        tag_loader=lambda _repository: [*formal_tags, "main-cann9.0.0-a3"],
    )

    assert {
        item["runtime_tag"]: item["channel"] for item in candidates["runtimes"]
    } == expected
    assert all("created_at" not in item for item in candidates["runtimes"])


def test_sglang_candidates_fail_when_registry_only_has_main_tags() -> None:
    with pytest.raises(ValueError, match="no published Runtime tag matches"):
        upstream.resolve_runtime_candidates(
            _sglang_policy(),
            tag_loader=lambda _repository: ["main-cann9.0.0-910b", "main-cann9.0.0-a3"],
        )


def test_sglang_explicit_main_tag_is_an_opaque_pin() -> None:
    tag = "main-cann9.0.0-a3"
    candidates = upstream.resolve_runtime_candidates(
        _sglang_policy(_tag_selector(tag)),
        tag_loader=lambda _repository: [tag],
    )
    assert [
        (item["runtime_tag"], item["version"], item["channel"])
        for item in candidates["runtimes"]
    ] == [(tag, tag, "pinned")]


@pytest.mark.parametrize(
    ("version", "channel", "metadata"),
    [
        ("main-cann9.0.0-a3", "main", {}),
        (
            "main-cann9.0.0-a3",
            "main",
            {"created_at": "2026-09-23T00:00:00+00:00"},
        ),
        ("0.0.0.dev20260923000000", "nightly", {}),
    ],
    ids=["main", "main-with-created-at", "legacy-synthetic-nightly"],
)
def test_sglang_candidates_reject_saved_main_documents(
    version: str, channel: str, metadata: dict[str, str]
) -> None:
    candidates = upstream.resolve_runtime_candidates(
        _sglang_policy(), tag_loader=lambda _repository: ["v0.5.19-cu129"]
    )
    tag = "main-cann9.0.0-a3"
    reference = f"docker.io/lmsysorg/sglang:{tag}"
    candidates["runtimes"][0].update(
        runtime_tag=tag,
        runtime_ref=reference,
        version=version,
        channel=channel,
        **metadata,
    )
    candidates["references"] = [reference]

    with pytest.raises(
        ValueError, match="fields must be exact|does not match product grammar"
    ):
        upstream.validate_runtime_candidates(candidates)


@pytest.mark.parametrize("channel", ["main", "nightly", "pinned"])
def test_sglang_selection_rejects_main_channels_but_preserves_pinned(
    channel: str,
) -> None:
    selection = _selection()
    runtime = next(
        item for item in selection["runtimes"] if item["backend"] == "cann-a2"
    )
    tag = f"main-{runtime['accelerator_runtime']}-910b"
    runtime.update(
        product_id="sglang",
        runtime_repository="docker.io/lmsysorg/sglang",
        runtime_tag=tag,
        version=tag if channel != "nightly" else "0.0.0.dev20260923000000",
        channel=channel,
    )
    selection["runtimes"] = [runtime]

    if channel == "pinned":
        validated = upstream.validate_selection(selection)
        assert validated["runtimes"][0]["channel"] == "pinned"
    else:
        with pytest.raises(
            ValueError, match="unsupported.*channel|channel.*unsupported"
        ):
            upstream.validate_selection(selection)


def test_latest_only_selector_uses_the_newest_version() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.23"), _selector("0.24")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        ["v0.23.0", "v0.24.0"],
        latest_only=True,
    ) == [{"runtime_tag": "v0.24.0", "version": "0.24.0", "channel": "stable"}]


@pytest.mark.parametrize(
    ("ranges", "expected_automatic_tags"),
    [
        ([_selector("0.5.18"), _selector("0.5.19")], ["v0.5.19-cu129"]),
        ([], []),
    ],
)
def test_latest_only_preserves_all_pins_with_or_without_numeric_ranges(
    ranges: list[dict[str, str | None]], expected_automatic_tags: list[str]
) -> None:
    pinned_tags = ["custom-build", "v0.5.18rc1-cu129"]
    selectors = ranges + [_tag_selector(tag) for tag in pinned_tags]
    selected = upstream._select_runtime_tags(  # noqa: SLF001
        {"id": "sglang", "runtime_selectors": selectors},
        ["v0.5.19-cu129", *pinned_tags],
        latest_only=True,
    )

    assert [item["runtime_tag"] for item in selected] == [
        *expected_automatic_tags,
        *pinned_tags,
    ]


def test_latest_per_product_narrows_only_the_named_product() -> None:
    release = _policy()
    release["products"].append(
        {
            "id": "sglang",
            "runtime_repository": "docker.io/lmsysorg/sglang",
            "runtime_selectors": [
                _selector("0.5.18"),
                _selector("0.5.19"),
                _selector("0.5.20"),
            ],
        }
    )
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.22.1"],
        "quay.io/ascend/vllm-ascend": ["v0.22.1"],
        "docker.io/lmsysorg/sglang": [
            "v0.5.18-cu129",
            "v0.5.19-cu129",
            "v0.5.20-cu129",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        latest_per_product=["sglang"],
    )

    sglang = [item for item in candidates["runtimes"] if item["product_id"] == "sglang"]
    assert [item["runtime_tag"] for item in sglang] == ["v0.5.20-cu129"]
    # Unnamed products still resolve their full configured selector range.
    assert any(item["product_id"] == "vllm" for item in candidates["runtimes"])


def test_latest_per_product_rejects_an_unknown_product() -> None:
    with pytest.raises(ValueError, match="unknown products"):
        upstream.resolve_runtime_candidates(
            _policy(),
            tag_loader=lambda _repository: [],
            latest_per_product=["vllm-does-not-exist"],
        )


def test_latest_only_rejects_a_malformed_selector_before_comparing() -> None:
    product = {
        "id": "sglang",
        "runtime_selectors": [
            _selector("0.5.19"),
            {"raw": "0.5.20", "tag": None},
        ],
    }

    # The newest selector is picked before the per-selector loop, so it must
    # still fail with the loop's contextual error instead of a bare KeyError.
    with pytest.raises(ValueError, match=r"runtime selector\[1\]: malformed selector"):
        upstream._select_runtime_tags(  # noqa: SLF001
            product,
            ["v0.5.19-cu129"],
            latest_only=True,
        )


def test_latest_only_rejects_an_unparseable_selector_version() -> None:
    product = {
        "id": "sglang",
        "runtime_selectors": [_selector("0.5.19"), _selector("0.5.20.post1")],
    }

    with pytest.raises(
        ValueError, match=r"runtime selector\[1\] version must be a canonical"
    ):
        upstream._select_runtime_tags(  # noqa: SLF001
            product,
            ["v0.5.19-cu129"],
            latest_only=True,
        )


def test_latest_only_rejects_duplicate_numeric_ranges() -> None:
    product = {
        "id": "sglang",
        "runtime_selectors": [_selector("0.5.20"), _selector("0.5.20")],
    }

    with pytest.raises(ValueError, match="the latest selector is ambiguous"):
        upstream._select_runtime_tags(  # noqa: SLF001
            product,
            [],
            latest_only=True,
        )


def test_shared_wheel_capability_is_retained_when_vllm_is_disabled() -> None:
    probe = next(
        item
        for item in _fixture()["runtime_probe"]["probes"]
        if item["product_id"] == "vllm" and item["cpu_arch"] == "amd64"
    )
    sglang_probe = copy.deepcopy(probe)
    sglang_probe.update(
        {
            "product_id": "sglang",
            "runtime_ref": "docker.io/lmsysorg/sglang:v0.5.19-cu129",
            "repository": "docker.io/lmsysorg/sglang",
            "tag": "v0.5.19-cu129",
            "target_repository": "ghcr.io/release-org/sglang",
        }
    )

    release = _policy()
    shared = builders.resolve_probe_builds(
        release, [probe, sglang_probe], tag_fixture=_fixture()
    )
    vllm = next(item for item in release["products"] if item["id"] == "vllm")
    vllm["runtime_selectors"] = []
    retained = builders.resolve_probe_builds(
        release, [sglang_probe], tag_fixture=_fixture()
    )

    assert len(shared) == len(retained) == 1
    assert retained[0]["id"] == shared[0]["id"]
    assert retained[0]["product_id"] == "shared"


def test_explicit_runtime_tag_flows_through_candidate_contract() -> None:
    release = _policy()
    vllm = next(product for product in release["products"] if product["id"] == "vllm")
    vllm["runtime_selectors"] = [_tag_selector("v0.29.0rc1-cu129")]
    tags = {
        "docker.io/vllm/vllm-openai": [
            "v0.29.0rc1-cu129",
            "v0.29.0rc2",
        ],
        "quay.io/ascend/vllm-ascend": ["v0.22.1rc1"],
    }

    candidates = upstream.resolve_runtime_candidates(
        release, tag_loader=lambda repository: tags[repository]
    )

    pinned = next(
        item for item in candidates["runtimes"] if item["product_id"] == "vllm"
    )
    assert pinned["runtime_tag"] == "v0.29.0rc1-cu129"
    assert pinned["version"] == "v0.29.0rc1-cu129"
    assert pinned["channel"] == "pinned"


def test_opaque_candidate_version_must_equal_its_tag() -> None:
    document = upstream.resolve_runtime_candidates(
        _sglang_policy(_tag_selector("aaaaa-cc")),
        tag_loader=lambda _repository: ["aaaaa-cc"],
    )
    document["runtimes"][0]["version"] = "1.0.0"
    with pytest.raises(ValueError, match="pinned Runtime version must equal its tag"):
        upstream.validate_runtime_candidates(document)


def test_patch_selector_does_not_merge_rc_or_other_patch_versions() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.25.1")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        ["v0.25.1", "v0.25.1-a3", "v0.25.1rc1", "v0.25.10"],
    ) == [
        {"runtime_tag": "v0.25.1", "version": "0.25.1", "channel": "stable"},
        {"runtime_tag": "v0.25.1-a3", "version": "0.25.1", "channel": "stable"},
    ]


def test_minor_selector_prefers_channel_before_newer_version() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.27")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.27.0",
            "v0.27.0-a3",
            "v0.27.1rc2",
            "nightly-releases-v0.27.2rc-a3",
        ],
    ) == [
        {"runtime_tag": "v0.27.0", "version": "0.27.0", "channel": "stable"},
        {
            "runtime_tag": "v0.27.0-a3",
            "version": "0.27.0",
            "channel": "stable",
        },
    ]


def test_minor_selector_uses_highest_rc_and_only_its_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.26")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "v0.26.0rc1",
            "v0.26.0rc1-openeuler",
            "v0.26.0rc2",
            "v0.26.0rc2-a3",
            "nightly-releases-v0.26.1rc-a3",
        ],
    ) == [
        {"runtime_tag": "v0.26.0rc2", "version": "0.26.0rc2", "channel": "rc"},
        {
            "runtime_tag": "v0.26.0rc2-a3",
            "version": "0.26.0rc2",
            "channel": "rc",
        },
    ]


def test_minor_selector_uses_highest_nightly_and_only_its_variants() -> None:
    product = {
        "id": "vllm-ascend",
        "runtime_selectors": [_selector("0.25")],
    }

    assert upstream._select_runtime_tags(  # noqa: SLF001
        product,
        [
            "nightly-releases-v0.25.0rc",
            "nightly-releases-v0.25.0rc-a3",
            "nightly-releases-v0.25.1rc",
            "nightly-releases-v0.25.1rc-a3",
            "nightly-releases-v0.25.1rc-openeuler",
        ],
    ) == [
        {
            "runtime_tag": "nightly-releases-v0.25.1rc",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.25.1rc-a3",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
        {
            "runtime_tag": "nightly-releases-v0.25.1rc-openeuler",
            "version": "0.25.1rc",
            "channel": "nightly",
        },
    ]


@pytest.mark.parametrize(
    ("selector", "tags", "message"),
    [
        (
            _selector("0.27.1"),
            [],
            "matches the version range",
        ),
        (
            _tag_selector("v0.27.1"),
            [],
            "not published",
        ),
        (
            {
                "raw": "0.27.1@v0.28.0",
                "version": "0.27.1",
                "tag": "v0.28.0",
            },
            ["v0.28.0"],
            "must use @tag without a version range",
        ),
        (
            _tag_selector("repository:tag"),
            ["repository:tag"],
            "not a valid OCI tag",
        ),
        (
            {"raw": "@v0.27.0", "version": None, "tag": "v0.28.0"},
            ["v0.28.0"],
            "raw selector is inconsistent",
        ),
    ],
)
def test_runtime_selector_fails_closed(
    selector: dict[str, object], tags: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        upstream._select_runtime_tags(  # noqa: SLF001
            {"id": "vllm", "runtime_selectors": [selector]}, tags
        )


def test_direct_tag_selects_only_the_exact_opaque_name() -> None:
    tag = "Release-build"
    selected = upstream._select_runtime_tags(  # noqa: SLF001
        {"id": "vllm", "runtime_selectors": [_tag_selector(tag)]},
        [tag, tag.lower(), f"{tag}-extra"],
    )
    assert selected == [{"runtime_tag": tag, "version": tag, "channel": "pinned"}]


def test_direct_tag_does_not_accept_a_prefix_or_change_case() -> None:
    with pytest.raises(ValueError, match="not published"):
        upstream._select_runtime_tags(  # noqa: SLF001
            {"id": "vllm", "runtime_selectors": [_tag_selector("aaaaa-cc")]},
            ["aaaaa-cc-dd", "AAAAA-CC"],
        )


@pytest.mark.parametrize("first", [_selector("0.30"), _tag_selector("v0.30.0")])
def test_direct_tags_cannot_duplicate_another_selected_runtime(
    first: dict[str, str | None],
) -> None:
    with pytest.raises(ValueError, match="Runtime tag is duplicated"):
        upstream._select_runtime_tags(  # noqa: SLF001
            {
                "id": "vllm",
                "runtime_selectors": [first, _tag_selector("v0.30.0")],
            },
            ["v0.30.0"],
        )


def test_direct_tag_does_not_infer_excluded_variants_from_its_name() -> None:
    tag = "v0.30.0-310p"
    selected = upstream._select_runtime_tags(  # noqa: SLF001
        {"id": "vllm-ascend", "runtime_selectors": [_tag_selector(tag)]},
        [tag],
        excluded_variants=["310p"],
    )
    assert selected == [{"runtime_tag": tag, "version": tag, "channel": "pinned"}]


def test_disabled_products_are_skipped_before_querying_the_registry() -> None:
    enabled_product, tag = "sglang", "custom-build"
    release = policy.resolve(repository="release-org/unified-cache-management")
    for product in release["products"]:
        product["runtime_selectors"] = (
            [_tag_selector(tag)] if product["id"] == enabled_product else []
        )
    enabled = next(
        product for product in release["products"] if product["id"] == enabled_product
    )
    queried = []

    def load_tags(repository: str) -> list[str]:
        queried.append(repository)
        assert repository == enabled["runtime_repository"]
        return [tag]

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=load_tags,
        latest_per_product=[product["id"] for product in release["products"]],
    )

    assert queried == [enabled["runtime_repository"]]
    assert {item["product_id"] for item in candidates["runtimes"]} == {enabled_product}


def test_candidate_resolution_selects_each_minor_independently() -> None:
    release = _policy()
    selectors = {
        "vllm": [_selector("0.26"), _selector("0.27")],
        "vllm-ascend": [
            _selector("0.24"),
            _selector("0.25"),
            _selector("0.26"),
        ],
    }
    for product in release["products"]:
        product["runtime_selectors"] = selectors[product["id"]]
    tags = {
        "docker.io/vllm/vllm-openai": [
            "v0.26.0",
            "v0.26.1",
            "v0.26.1-cu129",
            "v0.27.0rc2",
            "v0.27.1rc1",
        ],
        "quay.io/ascend/vllm-ascend": [
            "nightly-releases-v0.24.0rc",
            "nightly-releases-v0.24.0rc-a3",
            "nightly-releases-v0.25.0rc",
            "nightly-releases-v0.25.1rc",
            "nightly-releases-v0.25.1rc-a3",
            "v0.26.0rc1",
            "v0.26.0rc1-a3",
            "nightly-releases-v0.26.1rc-a3",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release, tag_loader=lambda repository: tags[repository]
    )

    assert candidates["references"] == [
        "docker.io/vllm/vllm-openai:v0.26.1",
        "docker.io/vllm/vllm-openai:v0.26.1-cu129",
        "docker.io/vllm/vllm-openai:v0.27.1rc1",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.24.0rc",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.24.0rc-a3",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.25.1rc",
        "quay.io/ascend/vllm-ascend:nightly-releases-v0.25.1rc-a3",
        "quay.io/ascend/vllm-ascend:v0.26.0rc1",
        "quay.io/ascend/vllm-ascend:v0.26.0rc1-a3",
    ]
    assert {
        (item["product_id"], item["version"], item["channel"])
        for item in candidates["runtimes"]
    } == {
        ("vllm", "0.26.1", "stable"),
        ("vllm", "0.27.1rc1", "rc"),
        ("vllm-ascend", "0.24.0rc", "nightly"),
        ("vllm-ascend", "0.25.1rc", "nightly"),
        ("vllm-ascend", "0.26.0rc1", "rc"),
    }


@pytest.mark.parametrize("excluded", [[], ["310p", "a3"]])
def test_excluded_variant_policy_is_the_runtime_filter_authority(
    tmp_path: Path, excluded: list[str]
) -> None:
    platforms = yaml.safe_load((RELEASE_ROOT / "platforms.yaml").read_text())
    platforms["excluded_upstream_variants"]["vllm-ascend"] = excluded
    path = tmp_path / "platforms.yaml"
    path.write_text(yaml.safe_dump(platforms), encoding="utf-8")
    release = _policy(platforms_path=path)
    probes = _fixture()["runtime_probe"]["probes"]

    # Use implemented Runtime members so this test isolates policy filtering.
    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: [
            probe["tag"] for probe in probes if probe["repository"] == repository
        ],
    )

    for probe in probes:
        expected = not (probe["backend"] == "cann-a3" and "a3" in excluded)
        assert (probe["runtime_ref"] in candidates["references"]) == expected


@pytest.mark.parametrize("variant_files", [{}, {"a2": ["libruntime.so"]}])
def test_variant_file_policy_reaches_builder_checks(
    tmp_path: Path, variant_files: dict[str, list[str]]
) -> None:
    platforms = yaml.safe_load((RELEASE_ROOT / "platforms.yaml").read_text())
    platforms["builder_families"]["ascend"]["variant_required_files"] = variant_files
    path = tmp_path / "platforms.yaml"
    path.write_text(yaml.safe_dump(platforms), encoding="utf-8")
    release = _policy(platforms_path=path)
    selection = upstream.resolve_upstreams(release, tag_fixture=_fixture())

    catalog = builders.catalog_from_builds(
        selection["wheel_builds"], owner="release-org", formal_policy=release
    )

    for builder in catalog["builders"]:
        if builder["accelerator"] == "ascend":
            expected = {"acl.h", *variant_files.get(builder["variant"], [])}
            assert set(builder["checks"]["required_files"]) == expected


def test_excluded_winner_does_not_fall_back_to_an_older_runtime() -> None:
    release = _policy()
    for product in release["products"]:
        product["runtime_selectors"] = [_selector("0.27")]
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.27.0"],
        "quay.io/ascend/vllm-ascend": ["v0.27.0", "v0.27.1-310p"],
    }

    with pytest.raises(ValueError, match="winning Runtime version"):
        upstream.resolve_runtime_candidates(
            release, tag_loader=lambda repository: tags[repository]
        )


def test_product_with_only_blocked_variants_fails_selection(runtime_probe) -> None:
    release = _policy()
    release["backends"][runtime_probe["backend"]].update(
        status="blocked", reason="Disabled by platform policy"
    )
    release["products"] = [
        product
        for product in release["products"]
        if product["id"] == runtime_probe["product_id"]
    ]

    with pytest.raises(ValueError, match="no publishable Runtime"):
        upstream.resolve_runtime_candidates(
            release, tag_loader=lambda repository: [runtime_probe["tag"]]
        )


def test_pr_default_selects_one_latest_ascend_a2_ubuntu_runtime() -> None:
    release = _policy()
    tags = {
        "docker.io/vllm/vllm-openai": ["v0.22.1", "v0.27.1"],
        "quay.io/ascend/vllm-ascend": [
            "v0.22.1rc1",
            "v0.26.0",
            "v0.26.0-a3",
            "v0.26.0-openeuler",
            "nightly-releases-v0.27.0rc",
        ],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        pr_default=True,
    )

    assert candidates["references"] == ["quay.io/ascend/vllm-ascend:v0.26.0"]
    assert candidates["problems"] == []


def test_pr_default_does_not_require_configured_runtime_selectors() -> None:
    release = _policy()
    for product in release["products"]:
        product["runtime_selectors"] = []
    tags = {
        "docker.io/vllm/vllm-openai": [],
        "quay.io/ascend/vllm-ascend": ["v0.27.0"],
    }

    candidates = upstream.resolve_runtime_candidates(
        release,
        tag_loader=lambda repository: tags[repository],
        pr_default=True,
    )

    assert candidates["references"] == ["quay.io/ascend/vllm-ascend:v0.27.0"]


def test_each_runtime_member_has_one_exact_wheel_link() -> None:
    selection = _selection()
    builds = {item["id"]: item for item in selection["wheel_builds"]}

    for runtime in selection["runtimes"]:
        assert set(runtime["architectures"]) == set(runtime["wheel_build_ids"])
        for architecture, wheel_id in runtime["wheel_build_ids"].items():
            build = builds[wheel_id]
            assert build["backend"] == runtime["backend"]
            assert build["accelerator_runtime"] == runtime["accelerator_runtime"]
            assert build["python_abi"] == runtime["python_abi"]
            assert build["cpu_arch"] == architecture


def test_raw_member_digest_changes_only_its_mirror_identity() -> None:
    before = _selection()
    fixture = _fixture()
    reference = "docker.io/pytorch/manylinux2_28-builder:cuda12.9"
    fixture["source_image_members"][reference]["amd64"] = "sha256:" + "e" * 64
    after = _selection(fixture)
    baseline = {item["id"]: item for item in before["wheel_builds"]}
    changed = {item["id"]: item for item in after["wheel_builds"]}

    assert (
        baseline["cu129-cp312-amd64"]["recipe_revision"]
        != changed["cu129-cp312-amd64"]["recipe_revision"]
    )
    assert (
        baseline["cu129-cp312-arm64"]["recipe_revision"]
        == changed["cu129-cp312-arm64"]["recipe_revision"]
    )


def test_single_platform_raw_builder_uses_its_verified_manifest_digest() -> None:
    digest = "sha256:" + "d" * 64
    pinned = "docker.io/pytorch/manylinuxaarch64-builder@" + digest
    seen: list[str] = []

    def manifest(reference: str) -> object:
        seen.append(reference)
        return {"mediaType": "application/vnd.docker.distribution.manifest.v2+json"}

    def config(reference: str) -> object:
        seen.append(reference)
        return {"os": "linux", "architecture": "arm64"}

    resolved = builders._manifest_member_digest(  # noqa: SLF001
        "docker.io/pytorch/manylinuxaarch64-builder:cuda12.9",
        "arm64",
        tag_fixture=None,
        manifest_loader=manifest,
        config_loader=config,
        digest_loader=lambda _reference: digest,
    )

    assert resolved == digest
    assert seen == [pinned, pinned]


def test_single_platform_raw_builder_rejects_wrong_architecture() -> None:
    with pytest.raises(ValueError, match="is not linux/arm64"):
        builders._manifest_member_digest(  # noqa: SLF001
            "docker.io/pytorch/manylinuxaarch64-builder:cuda12.9",
            "arm64",
            tag_fixture=None,
            manifest_loader=lambda _reference: {
                "mediaType": "application/vnd.oci.image.manifest.v1+json"
            },
            config_loader=lambda _reference: {
                "os": "linux",
                "architecture": "amd64",
            },
            digest_loader=lambda _reference: "sha256:" + "e" * 64,
        )


def test_raw_builder_selection_honors_configured_manylinux_policy() -> None:
    fixture = _fixture()
    repository = "quay.io/ascend/manylinux"
    tag = "9.0.1-910b-manylinux_2_28-py3.12"
    fixture["repositories"][repository]["pages"][0]["tags"].append(tag)
    fixture["source_image_members"][f"{repository}:{tag}"] = {
        "amd64": "sha256:" + "f" * 64
    }

    selection = _selection(fixture)
    build = next(
        item
        for item in selection["wheel_builds"]
        if item["backend"] == "cann-a2" and item["cpu_arch"] == "amd64"
    )
    assert build["manylinux"] == "manylinux_2_34"
    assert build["source_image"].endswith("9.0.1-910b-manylinux_2_34-py3.12")


def _cann_inputs(
    versions: list[str],
    requested: str = "9.2.0",
    architecture: str = "amd64",
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    fixture = _fixture()
    probe = copy.deepcopy(
        next(
            item
            for item in fixture["runtime_probe"]["probes"]
            if item["backend"] == "cann-a2"
        )
    )
    probe["accelerator_runtime"] = f"cann-{requested}"
    probe["cpu_arch"] = architecture
    repository = "quay.io/ascend/manylinux"
    tags = [f"{version}-910b-manylinux_2_34-py3.12" for version in versions]
    fixture["repositories"][repository] = {"pages": [{"tags": tags}]}
    for index, tag in enumerate(tags, start=1):
        fixture["source_image_members"][f"{repository}:{tag}"] = {
            "amd64": f"sha256:{index:064x}",
            "arm64": f"sha256:{index + 100:064x}",
        }
    return _policy(), fixture, probe


@pytest.mark.parametrize(
    ("requested", "available", "winner"),
    [
        ("9.2.0", ["9.2.0-beta.2", "9.2.0"], "9.2.0"),
        ("9.2.0", ["9.2.0-beta.1", "9.2.0-beta.2"], "9.2.0-beta.2"),
        ("9.2.0", ["9.2.0-beta.2", "9.2.0-beta.10"], "9.2.0-beta.10"),
        ("9.2.0", ["9.2.0-beta.10", "9.2.0-beta.2"], "9.2.0-beta.10"),
        ("9.2.0", ["9.2.0-beta.10", "9.2.0-rc.1"], "9.2.0-rc.1"),
        ("9.2.0-beta.2", ["9.2.0b2"], "9.2.0b2"),
        (
            "9.2.0-beta.1",
            ["9.2.0-beta.2", "9.2.0-beta.1", "9.2.0"],
            "9.2.0-beta.1",
        ),
    ],
)
def test_cann_selection_prefers_exact_then_latest_same_base_suffix(
    requested: str, available: list[str], winner: str
) -> None:
    formal, fixture, probe = _cann_inputs(available, requested)

    build = builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)[0]

    assert builders.source_cann_version(build["source_image"]) == winner
    assert build["accelerator_runtime"] == f"cann-{requested}"


@pytest.mark.parametrize(
    ("requested", "available"),
    [
        ("9.2.0-beta.1", ["9.2.0-beta.2", "9.2.0"]),
        ("9.2.0", ["9.1.0-beta.9", "9.2.1-beta.1", "9.3.0-beta.1"]),
    ],
)
def test_cann_selection_does_not_relax_explicit_suffix_or_numeric_base(
    requested: str, available: list[str]
) -> None:
    formal, fixture, probe = _cann_inputs(available, requested)

    with pytest.raises(ValueError, match="found 0"):
        builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)


def test_cann_fallback_preserves_target_names_and_records_actual_source_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.2"])
    build = builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)[0]
    catalog = builders.catalog_from_builds(
        [build], owner="release-org", formal_policy=formal
    )
    selected = catalog["builders"][0]

    assert build["id"] == "cann920-a2-cp312-amd64"
    assert build["runtime_variant"] == "cann920-a2"
    assert selected["accelerator_runtime"] == "cann-9.2.0"
    assert selected["checks"]["accelerator_runtime"] == "cann-9.2.0-beta.2"
    labels = builders.builder_labels(selected)
    reopened = builders.registry_builder_record(
        selected["target_repository"],
        selected["target_tag"],
        {"created": "2026-10-14T00:00:00Z", "config": {"Labels": labels}},
    )
    assert reopened["source_image"] == build["source_image"]
    assert reopened["source_image_digest"] == build["source_image_digest"]
    diagnostic = capsys.readouterr()
    assert diagnostic.out == ""
    assert "same-base latest suffix" in diagnostic.err
    assert "9.2.0-beta.2" in diagnostic.err


def test_cann_selection_filters_capability_and_diagnoses_bad_tags_before_ranking(
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.2", "9.2.0-unknown"])
    tags = fixture["repositories"]["quay.io/ascend/manylinux"]["pages"][0]["tags"]
    tags.extend(
        [
            "9.2.0-rc.1-a3-manylinux_2_34-py3.12",
            "9.2.0-rc.1-910b-manylinux_2_34-py3.11",
            "9.2.0-rc.1-910b-manylinux_2_28-py3.12",
            "latest",
        ]
    )

    build = builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)[0]

    assert builders.source_cann_version(build["source_image"]) == "9.2.0-beta.2"
    assert "unparseable CANN version" in capsys.readouterr().err


def test_cann_ranks_only_builder_versions_that_supply_requested_architecture() -> None:
    formal, fixture, probe = _cann_inputs(
        ["9.2.0-beta.1", "9.2.0-beta.2"], architecture="arm64"
    )
    newest = "quay.io/ascend/manylinux:9.2.0-beta.2-910b-manylinux_2_34-py3.12"
    del fixture["source_image_members"][newest]["arm64"]

    build = builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)[0]

    assert builders.source_cann_version(build["source_image"]) == "9.2.0-beta.1"
    assert build["cpu_arch"] == "arm64"


@pytest.mark.parametrize("invalid_member", ["duplicate", "bad-digest"])
def test_cann_architecture_filter_preserves_manifest_errors(
    invalid_member: str,
) -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.2"])
    reference = "quay.io/ascend/manylinux:9.2.0-beta.2-910b-manylinux_2_34-py3.12"
    del fixture["source_image_members"][reference]
    member = {
        "platform": {"os": "linux", "architecture": "amd64"},
        "digest": "sha256:" + "1" * 64,
    }
    members = (
        [member, member]
        if invalid_member == "duplicate"
        else [{**member, "digest": "invalid"}]
    )

    with pytest.raises(ValueError, match="exactly one|invalid.*digest"):
        builders.resolve_probe_builds(
            formal,
            [probe],
            tag_fixture=fixture,
            manifest_loader=lambda _reference: {"manifests": members},
            digest_loader=lambda _reference: "sha256:" + "9" * 64,
        )


def test_cann_missing_architecture_in_real_index_is_ineligible() -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.1", "9.2.0-beta.2"])
    newest = "quay.io/ascend/manylinux:9.2.0-beta.2-910b-manylinux_2_34-py3.12"
    del fixture["source_image_members"][newest]

    build = builders.resolve_probe_builds(
        formal,
        [probe],
        tag_fixture=fixture,
        manifest_loader=lambda _reference: {
            "manifests": [
                {
                    "platform": {"os": "linux", "architecture": "arm64"},
                    "digest": "sha256:" + "2" * 64,
                }
            ]
        },
        digest_loader=lambda _reference: "sha256:" + "9" * 64,
    )[0]

    assert builders.source_cann_version(build["source_image"]) == "9.2.0-beta.1"


def test_cann_equal_version_aliases_remain_ambiguous() -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.2", "9.2.0b2"])

    with pytest.raises(ValueError, match="found 2"):
        builders.resolve_probe_builds(formal, [probe], tag_fixture=fixture)


def test_cann_selection_filters_platform_and_actual_manifest_architecture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    formal, fixture, probe = _cann_inputs(["9.2.0-beta.2", "9.2.0-unknown"])
    repository = "quay.io/ascend/manylinux"
    tags = fixture["repositories"][repository]["pages"][0]["tags"]
    wrong_platform = [
        "9.2.0-rc.1-a3-manylinux_2_34-py3.12",
        "9.2.0-rc.1-910b-manylinux_2_34-py3.11",
        "9.2.0-rc.1-910b-manylinux_2_28-py3.12",
    ]
    tags.extend(wrong_platform)
    for tag in wrong_platform:
        fixture["source_image_members"][f"{repository}:{tag}"] = {
            "amd64": "sha256:" + "8" * 64
        }
    tags.append("9.2.0-rc.1-910b-manylinux_2_34-py3.12")

    build = builders.resolve_probe_builds(
        formal,
        [probe],
        tag_fixture=fixture,
        manifest_loader=lambda _reference: {
            "manifests": [
                {
                    "platform": {"os": "linux", "architecture": "arm64"},
                    "digest": "sha256:" + "2" * 64,
                }
            ]
        },
        digest_loader=lambda _reference: "sha256:" + "9" * 64,
    )[0]

    assert builders.source_cann_version(build["source_image"]) == "9.2.0-beta.2"
    assert "9.2.0-unknown" in capsys.readouterr().err


def test_ascend_builder_floor_can_vary_by_runtime() -> None:
    fixture = _fixture()
    repository = "quay.io/ascend/manylinux"
    source_members = fixture["source_image_members"]
    tags = fixture["repositories"][repository]["pages"][0]["tags"]
    probes = []
    cases = (
        ("cann-a2", "ascend910b1", "910b", "amd64", "5"),
        ("cann-a3", "ascend910_9391", "a3", "arm64", "6"),
    )
    for index, (backend, soc, token, architecture, digest_digit) in enumerate(
        cases, start=1
    ):
        tag = f"9.0.0-{token}-manylinux_2_28-py3.11"
        if tag not in tags:
            tags.append(tag)
        reference = f"{repository}:{tag}"
        source_members.setdefault(reference, {})[architecture] = (
            "sha256:" + digest_digit * 64
        )
        probe = copy.deepcopy(fixture["runtime_probe"]["probes"][index + 1])
        probe.update(
            {
                "product_id": "sglang",
                "runtime_ref": ("docker.io/lmsysorg/sglang:v0.5.19-cann9.0.0-" + token),
                "backend": backend,
                "accelerator_runtime": "cann-9.0.0",
                "soc_version": soc,
                "python_version": "3.11",
                "python_abi": "cp311",
                "cpu_arch": architecture,
            }
        )
        probes.append(probe)

    builds = builders.resolve_probe_builds(_policy(), probes, tag_fixture=fixture)

    assert {
        (
            build["backend"],
            build["cpu_arch"],
            build["manylinux"],
            build["source_image"],
        )
        for build in builds
    } == {
        (
            "cann-a2",
            "amd64",
            "manylinux_2_28",
            f"{repository}:9.0.0-910b-manylinux_2_28-py3.11",
        ),
        (
            "cann-a3",
            "arm64",
            "manylinux_2_28",
            f"{repository}:9.0.0-a3-manylinux_2_28-py3.11",
        ),
    }


def test_runtime_glibc_is_not_required_for_wheel_or_builder_planning() -> None:
    fixture = _fixture()
    for probe in fixture["runtime_probe"]["probes"]:
        probe["glibc_version"] = None

    selection = _selection(fixture)
    catalog = _catalog(selection)

    assert selection["wheel_builds"]
    assert all(runtime["glibc_version"] is None for runtime in selection["runtimes"])
    assert catalog["builders"]


def test_sync_is_append_only_and_registry_records_reopen_exactly() -> None:
    catalog = _catalog()
    first = catalog["builders"][0]
    existing = {first["target_repository"]: [first["target_tag"]]}
    sync = builders.compute_sync_plan(catalog, existing)

    assert len(sync["builders"]) == len(catalog["builders"]) - 1
    assert len(sync["matrix"]["include"]) == len(catalog["builders"])
    assert "deletions" not in sync
    assert all(item["build_mode"] == "mirror" for item in sync["builders"])

    labels = builders.builder_labels(first)
    record = builders.registry_builder_record(
        first["target_repository"],
        first["target_tag"],
        {
            "created": "2026-08-24T00:00:00Z",
            "config": {"Labels": labels},
        },
    )
    reopened = builders.catalog_from_registry_records([record])
    assert reopened["schema_version"] == 3
    assert (
        reopened["builders"][0]["source_image_digest"] == first["source_image_digest"]
    )


def test_final_catalog_binds_checked_labels_and_target_digests() -> None:
    catalog = _catalog()
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

    finalized = builders.finalize_catalog(catalog, observations)

    assert finalized["schema_version"] == 4
    assert all(
        item["target_digest"].startswith("sha256:") for item in finalized["builders"]
    )
    with pytest.raises(ValueError, match="unfinalized Catalog"):
        builders.compute_sync_plan(finalized, {})


@pytest.mark.parametrize("legacy_product_id", ["vllm", "vllm-ascend", "sglang"])
def test_final_catalog_accepts_legacy_product_label_for_shared_builder(
    legacy_product_id: str,
) -> None:
    catalog = _catalog()
    first = catalog["builders"][0]
    observations = {}
    for index, item in enumerate(catalog["builders"]):
        labels = builders.builder_labels(item)
        if item["id"] == first["id"]:
            labels["io.ucm.builder.product_id"] = legacy_product_id
        observations[item["id"]] = {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": labels},
            },
        }

    finalized = builders.finalize_catalog(catalog, observations)

    assert finalized["builders"][0]["product_id"] == "shared"


def test_final_catalog_rejects_unknown_product_label_for_shared_builder() -> None:
    catalog = _catalog()
    first = catalog["builders"][0]
    observations = {}
    for index, item in enumerate(catalog["builders"]):
        labels = builders.builder_labels(item)
        if item["id"] == first["id"]:
            labels["io.ucm.builder.product_id"] = "unrelated-product"
        observations[item["id"]] = {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": labels},
            },
        }

    with pytest.raises(ValueError, match="label product_id differs"):
        builders.finalize_catalog(catalog, observations)


def test_final_catalog_rejects_stale_builder_labels() -> None:
    catalog = _catalog()
    item = catalog["builders"][0]
    labels = builders.builder_labels(item)
    labels["io.ucm.builder.source_image_digest"] = "sha256:" + "f" * 64
    observations = {
        current["id"]: {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {
                    "Labels": (
                        labels
                        if current["id"] == item["id"]
                        else builders.builder_labels(current)
                    )
                },
            },
        }
        for index, current in enumerate(catalog["builders"])
    }

    with pytest.raises(ValueError, match="label source_image_digest differs"):
        builders.finalize_catalog(catalog, observations)


def test_source_binding_pins_upstream_builders_without_mutating_desired_catalog() -> (
    None
):
    desired = _catalog()

    bound = builders.bind_source_catalog(desired)

    assert desired["schema_version"] == 3
    assert bound["schema_version"] == 4
    assert len(bound["builders"]) == len(desired["builders"])
    for original, selected in zip(desired["builders"], bound["builders"], strict=True):
        source_repository, _source_tag = original["source_image"].rsplit(":", 1)
        assert selected["target_repository"] == source_repository
        assert selected["target_tag"] == original["target_tag"]
        assert selected["target_digest"] == original["source_image_digest"]
        assert original["target_repository"] == "ghcr.io/release-org/" + (
            "ucm-builder-vllm"
            if original["accelerator"] == "cuda"
            else "ucm-builder-vllm-ascend"
        )


def test_source_binding_rejects_an_already_finalized_catalog() -> None:
    desired = _catalog()
    observations = {
        item["id"]: {
            "target_digest": f"sha256:{index + 1:064x}",
            "config": {
                "created": "2026-08-24T00:00:00Z",
                "config": {"Labels": builders.builder_labels(item)},
            },
        }
        for index, item in enumerate(desired["builders"])
    }

    with pytest.raises(ValueError, match="desired Catalog schema 3"):
        builders.bind_source_catalog(builders.finalize_catalog(desired, observations))
