from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
pypi = importlib.import_module("ucm_release.pypi")

VERSION = "0.9.1rc1"
CUDA = "uc-manager-cuda-cu130"
CANN = "uc-manager-cann901-a2"
META = "uc-manager"
CUDA_AMD64 = "uc_manager_cuda_cu130-0.9.1rc1-cp312-cp312-manylinux_2_28_x86_64.whl"
CUDA_ARM64 = "uc_manager_cuda_cu130-0.9.1rc1-cp312-cp312-manylinux_2_28_aarch64.whl"
CANN_AMD64 = "uc_manager_cann901_a2-0.9.1rc1-cp312-cp312-manylinux_2_34_x86_64.whl"
META_WHEEL = "uc_manager-0.9.1rc1-py3-none-any.whl"


def _sha(character: str, *, prefix: bool = False) -> str:
    value = character * 64
    return f"sha256:{value}" if prefix else value


def _release_plan() -> dict[str, object]:
    return {
        "kind": "ucm-release-plan",
        "version": VERSION,
        "repository": "ModelEngine-Group/unified-cache-management",
        "publication_scope": "official",
        "publish": {
            "pypi": {
                "enabled": True,
                "index": "https://upload.pypi.org/legacy/",
            }
        },
        "meta_package": {
            "distribution": META,
            "version": VERSION,
            "extras": {
                "cann901-a2": f"{CANN}=={VERSION}",
                "cu130": f"{CUDA}=={VERSION}",
            },
        },
        "wheels": [
            {"id": "cuda-arm64", "dist_name": CUDA, "wheel_version": VERSION},
            {"id": "cann-amd64", "dist_name": CANN, "wheel_version": VERSION},
            {"id": "cuda-amd64", "dist_name": CUDA, "wheel_version": VERSION},
        ],
    }


def _backend_results() -> list[dict[str, object]]:
    return [
        {
            "kind": "ucm-wheel-result",
            "schema_version": 5,
            "task_id": "cuda-amd64",
            "distribution": CUDA,
            "version": VERSION,
            "filename": CUDA_AMD64,
            "sha256": _sha("a"),
        },
        {
            "kind": "ucm-wheel-result",
            "schema_version": 5,
            "task_id": "cuda-arm64",
            "distribution": CUDA,
            "version": VERSION,
            "filename": CUDA_ARM64,
            "sha256": _sha("b"),
        },
        {
            "kind": "ucm-wheel-result",
            "schema_version": 5,
            "task_id": "cann-amd64",
            "distribution": CANN,
            "version": VERSION,
            "filename": CANN_AMD64,
            "sha256": _sha("c"),
        },
    ]


def _meta_result() -> dict[str, object]:
    return {
        "kind": "ucm-meta-result",
        "schema_version": 1,
        "distribution": META,
        "version": VERSION,
        "filename": META_WHEEL,
        "sha256": _sha("d", prefix=True),
        "extras": _release_plan()["meta_package"]["extras"],  # type: ignore[index]
    }


def _publication() -> dict[str, object]:
    return pypi.build_publication(_release_plan(), _backend_results(), _meta_result())


def _document(project: dict[str, object]) -> dict[str, object]:
    urls = [
        {
            "filename": file["filename"],
            "packagetype": "bdist_wheel",
            "digests": {"sha256": file["sha256"].removeprefix("sha256:")},
            "yanked": False,
        }
        for file in project["files"]  # type: ignore[union-attr]
    ]
    info: dict[str, object] = {
        "name": project["project"],
        "version": project["version"],
        "yanked": False,
    }
    if project["role"] == "meta":
        info.update(
            {
                "provides_extra": ["cu130", "cann901-a2"],
                "requires_dist": [
                    f'{CANN}=={VERSION}; extra == "cann901-a2"',
                    f'{CUDA}=={VERSION}; extra == "cu130"',
                ],
            }
        )
    return {"info": info, "urls": urls}


class _SequenceFetcher:
    def __init__(self, values: dict[str, list[dict[str, object] | None]]) -> None:
        self.values = {key: list(items) for key, items in values.items()}

    def __call__(self, project: str, version: str):
        assert version == VERSION
        values = self.values[project]
        if len(values) > 1:
            return values.pop(0)
        return values[0]


def test_publication_is_deterministic_and_groups_architectures() -> None:
    expected = _publication()
    actual = pypi.build_publication(
        _release_plan(), list(reversed(_backend_results())), _meta_result()
    )

    assert actual == expected
    assert [item["project"] for item in actual["backends"]] == [CANN, CUDA]
    assert [file["filename"] for file in actual["backends"][1]["files"]] == [
        CUDA_ARM64,
        CUDA_AMD64,
    ]
    assert actual["meta"]["files"][0]["filename"] == META_WHEEL


def test_publication_rejects_incomplete_results_and_extra_drift() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        pypi.build_publication(_release_plan(), _backend_results()[:-1], _meta_result())

    meta = _meta_result()
    meta["extras"] = {"cu130": f"{CUDA}=={VERSION}"}
    with pytest.raises(ValueError, match="extras differ"):
        pypi.build_publication(_release_plan(), _backend_results(), meta)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("publication_scope", "fork"),
        ("repository", "SuperMarioYL/unified-cache-management"),
        ("pypi_enabled", False),
    ],
)
def test_publication_rejects_non_official_or_disabled_plan(
    field: str, value: object
) -> None:
    plan = _release_plan()
    if field == "pypi_enabled":
        plan["publish"]["pypi"]["enabled"] = value  # type: ignore[index]
    else:
        plan[field] = value

    with pytest.raises(ValueError, match="official enabled PyPI"):
        pypi.build_publication(plan, _backend_results(), _meta_result())


def test_publish_uploads_backends_then_meta_and_returns_receipt() -> None:
    publication = _publication()
    cann, cuda = publication["backends"]
    meta = publication["meta"]
    fetch = _SequenceFetcher(
        {
            CANN: [None, _document(cann)],
            CUDA: [None, _document(cuda)],
            META: [None, None, _document(meta)],
        }
    )
    uploads: list[str] = []

    receipt = pypi.publish(
        publication,
        fetch=fetch,
        uploader=uploads.append,
        sleep=lambda _seconds: None,
        attempts=2,
    )

    assert uploads == [CANN_AMD64, CUDA_ARM64, CUDA_AMD64, META_WHEEL]
    assert receipt["kind"] == "ucm-pypi-receipt"
    assert receipt["status"] == "complete"
    assert receipt["projects"] == [cann, cuda, meta]


def test_publish_skips_existing_same_hash_files() -> None:
    publication = _publication()
    projects = [*publication["backends"], publication["meta"]]
    documents = {project["project"]: _document(project) for project in projects}
    uploads: list[str] = []

    receipt = pypi.publish(
        publication,
        fetch=lambda project, version: documents[project],
        uploader=uploads.append,
        sleep=lambda _seconds: None,
    )

    assert uploads == []
    assert receipt["status"] == "complete"


def test_publish_rejects_meta_that_precedes_missing_backends() -> None:
    publication = _publication()
    meta = publication["meta"]
    documents = {
        CANN: None,
        CUDA: None,
        META: _document(meta),
    }

    with pytest.raises(pypi.PyPIReadbackError, match="meta Wheel is public"):
        pypi.publish(
            publication,
            fetch=lambda project, version: documents[project],
            uploader=lambda _filename: None,
            sleep=lambda _seconds: None,
        )


@pytest.mark.parametrize("failure", ["hash", "extra-file", "yanked", "metadata"])
def test_publish_preflight_fails_before_any_upload(failure: str) -> None:
    publication = _publication()
    cann, cuda = publication["backends"]
    meta = publication["meta"]
    documents = {CANN: _document(cann), CUDA: _document(cuda), META: _document(meta)}
    if failure == "hash":
        documents[CUDA]["urls"][0]["digests"]["sha256"] = _sha("9")  # type: ignore[index]
        error = pypi.PyPIConflictError
    elif failure == "extra-file":
        documents[CUDA]["urls"].append(  # type: ignore[union-attr]
            {
                "filename": "unexpected.whl",
                "packagetype": "bdist_wheel",
                "digests": {"sha256": _sha("9")},
                "yanked": False,
            }
        )
        error = pypi.PyPIReadbackError
    elif failure == "yanked":
        documents[CUDA]["urls"][0]["yanked"] = True  # type: ignore[index]
        error = pypi.PyPIReadbackError
    else:
        documents[META]["info"]["requires_dist"] = []  # type: ignore[index]
        error = pypi.PyPIReadbackError
    uploads: list[str] = []

    with pytest.raises(error):
        pypi.publish(
            publication,
            fetch=lambda project, version: documents[project],
            uploader=uploads.append,
            sleep=lambda _seconds: None,
        )
    assert uploads == []


class _Response:
    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


def test_fetch_retries_transient_http_with_fresh_cache_nonce() -> None:
    urls: list[str] = []

    def open_url(request, *, timeout):
        urls.append(request.full_url)
        if len(urls) == 1:
            raise HTTPError(request.full_url, 503, "unavailable", None, None)
        return _Response({"info": {}, "urls": []})

    result = pypi.fetch_version_json(
        CUDA,
        VERSION,
        attempts=2,
        retry_interval=0,
        open_url=open_url,
        sleep=lambda _seconds: None,
    )

    assert result == {"info": {}, "urls": []}
    nonces = [parse_qs(urlparse(url).query)["ucm_readback"][0] for url in urls]
    assert len(set(nonces)) == 2


def test_twine_uploader_resolves_one_file_and_keeps_token_out_of_args(
    tmp_path: Path, monkeypatch
) -> None:
    wheel = tmp_path / CUDA_AMD64
    wheel.write_bytes(b"wheel")
    invocation = {}

    def run(arguments, **options):
        invocation.update({"arguments": arguments, **options})

    monkeypatch.setattr(pypi.subprocess, "run", run)
    uploader = pypi.make_twine_uploader(
        roots=[tmp_path],
        expected_sha256={CUDA_AMD64: "sha256:" + hashlib.sha256(b"wheel").hexdigest()},
        repository_url="https://upload.pypi.org/legacy/",
        token="secret-token",
    )

    assert uploader(CUDA_AMD64) == wheel
    assert invocation["env"]["TWINE_PASSWORD"] == "secret-token"
    assert "secret-token" not in " ".join(invocation["arguments"])
