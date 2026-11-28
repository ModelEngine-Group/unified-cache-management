"""Exercise the same receipt, retry and registry operations used by Tag releases."""

import hashlib
import importlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
delivery = importlib.import_module("ucm_release.image_delivery")


def test_observations_and_pins_have_different_retry_semantics(tmp_path):
    targets = {"ghcr": "ghcr.io/example/ucm:v1"}
    target = {
        "channel": "ghcr",
        "reference": targets["ghcr"],
        "digest": "sha256:" + "a" * 64,
    }
    receipt = {
        "id": "amd",
        "schema_version": 1,
        "kind": "ucm-image-member-receipt",
        "status": "published",
        "targets": [target],
    }
    root = tmp_path / "history"
    for attempt in (2, 10):
        path = root / "binding/run-1" / f"attempt-{attempt}" / "member-amd.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        receipt = {**receipt, "status": "failed", "targets": []}
    out = tmp_path / "latest"
    delivery.collect(root, "binding", out)
    assert json.loads((out / "member-amd.json").read_text()) == receipt
    assert delivery.receipt_pins(
        delivery.snapshots(root, "binding"), "member", "amd", targets
    ) == {"ghcr": target["digest"]}
    assert not delivery.snapshots(root, "different-plan")


def make_archive(path, arch):
    """A real OCI archive with an architecture-specific config and no filesystem layers."""
    files = {}

    def descriptor(media, payload):
        data = json.dumps(payload).encode()
        digest = hashlib.sha256(data).hexdigest()
        files["blobs/sha256/" + digest] = data
        return {"mediaType": media, "digest": "sha256:" + digest, "size": len(data)}

    config = descriptor(
        "application/vnd.oci.image.config.v1+json",
        {
            "os": "linux",
            "architecture": arch,
            "rootfs": {"type": "layers", "diff_ids": []},
        },
    )
    media = "application/vnd.oci.image.manifest.v1+json"
    manifest = descriptor(
        media, {"schemaVersion": 2, "mediaType": media, "config": config, "layers": []}
    )
    files["index.json"] = json.dumps(
        {"schemaVersion": 2, "manifests": [manifest]}
    ).encode()
    files["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


@pytest.fixture
def registry():
    if not (shutil.which("crane") and shutil.which("skopeo")):
        pytest.skip("local OCI readback requires crane and skopeo")
    with subprocess.Popen(
        ["crane", "registry", "serve", "--address", "127.0.0.1:0"],
        stderr=subprocess.PIPE,
        text=True,
    ) as server:
        try:
            port = re.search(r"port (\d+)", server.stderr.readline()).group(1)
            # The registry logs every blob request. Drain the pipe throughout the test.
            threading.Thread(target=server.stderr.read, daemon=True).start()
            yield f"127.0.0.1:{port}"
        finally:
            server.terminate()


def test_partial_delivery_and_pinned_retry_with_registries(
    registry, tmp_path, monkeypatch
):
    # Redirect both production namespaces into isolated loopback repositories.
    # All digest/manifest reads, uploads, cross-repo copies and index writes are real.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "docker"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/ucm")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "321")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("PUBLICATION_SCOPE", "official")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "v1")
    monkeypatch.setenv("DOCKERHUB_NAMESPACE", "docker.io/example")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "outputs"))
    policy = tmp_path / "policy.json"
    policy.write_text('{"default":[{"type":"insecureAcceptAnything"}]}')
    original_command = delivery.command

    def local(*args):
        args = [
            value.replace("ghcr.io/example/", f"{registry}/ghcr/").replace(
                "docker.io/example/", f"{registry}/hub/"
            )
            for value in args
        ]
        if args[:2] == ["skopeo", "copy"]:
            args = [
                "skopeo",
                "--policy",
                str(policy),
                "copy",
                "--dest-tls-verify=false",
                *args[2:],
            ]
        return original_command(*args)

    monkeypatch.setattr(delivery, "command", local)
    offline = {"dockerhub"}

    def login(channel):
        if channel in offline:
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(delivery, "login", login)
    ref = "ghcr.io/example/ascend:v1"
    members = [
        {"image_id": arch, "cpu_arch": arch, "reference": f"{ref}-{arch}"}
        for arch in ("amd64", "arm64")
    ]
    cuda = {
        "image_id": "cuda-amd",
        "cpu_arch": "amd64",
        "reference": "ghcr.io/example/cuda:v1-amd64",
    }
    plan = {
        "route": "release",
        "repository": "example/ucm",
        "git_tag": "v1",
        "publication_scope": "official",
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {"enabled": True, "namespace": "docker.io/example"},
        },
        "families": [
            {
                "id": "ascend",
                "create_index": True,
                "members": members,
                "published_reference": ref,
            },
            {
                "id": "cuda",
                "create_index": True,
                "members": [
                    cuda,
                    {**cuda, "cpu_arch": "arm64", "image_id": "cuda-arm"},
                ],
                "published_reference": "ghcr.io/example/cuda:v1",
            },
        ],
    }
    plan_path = Path("input/plan/release-plan.json")
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan))

    def execute(action, *args, output="out/history"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "image_delivery",
                action,
                "--history",
                "out/history",
                "--output",
                output,
                *args,
            ],
        )
        delivery.main()

    def member(arch):
        execute("publish", "--kind", "member", "--id", arch, "--archive", f"{arch}.tar")

    def index(name):
        execute("publish", "--kind", "index", "--id", name)

    for arch in ("amd64", "arm64"):
        make_archive(Path(f"{arch}.tar"), arch)
        with pytest.raises(ValueError, match="registry unavailable"):
            member(arch)
    execute("collect", output="input/members")
    with pytest.raises(ValueError, match="registry unavailable"):
        index("ascend")
    with pytest.raises(ValueError, match="missing verified architecture"):
        index("cuda")
    initial = delivery.registry_digest(ref)
    assert initial
    assert delivery.registry_digest("docker.io/example/ascend:v1") is None
    assert delivery.registry_digest("ghcr.io/example/cuda:v1") is None
    pins = [delivery.registry_digest(m["reference"]) for m in members]
    # The original archives have gone away, as they would on a fresh runner.
    for arch in ("amd64", "arm64"):
        Path(f"{arch}.tar").unlink()
    offline.clear()
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    for arch in ("amd64", "arm64"):
        execute("prepare", "--kind", "member", "--id", arch)
        assert "reuse=true" in Path("outputs").read_text()
        member(arch)
    execute("collect", output="input/members")
    index("ascend")
    assert delivery.registry_digest(ref) == initial
    for m, digest in zip(members, pins):
        assert delivery.registry_digest(m["reference"]) == digest
        assert (
            delivery.registry_digest(m["reference"].replace("ghcr.io", "docker.io"))
            == digest
        )
    assert delivery.registry_digest("docker.io/example/ascend:v1")
    # A later run of the same Tag uses receipts bundled with its frozen plan.
    shutil.move("out/history", plan_path.parent / "prior-image-receipts")
    monkeypatch.setenv("GITHUB_RUN_ID", "322")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    member("amd64")
    assert delivery.registry_digest(members[0]["reference"]) == pins[0]
    # A foreign write must remain visible and must never be overwritten by retry.
    local("crane", "tag", delivery.pinned(ref, pins[1]), "v1-amd64")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    with pytest.raises(ValueError, match="existing digest"):
        member("amd64")
    assert delivery.registry_digest(members[0]["reference"]) == pins[1]


def test_existing_member_without_receipt_cannot_be_overwritten(monkeypatch, tmp_path):
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(delivery, "registry_digest", lambda reference: digest)
    with pytest.raises(ValueError, match="proven digest None"):
        delivery.publish_member(
            "ghcr.io/example/ucm:v1", None, None, tmp_path / "missing"
        )


@pytest.mark.parametrize(
    "target", ["ghcr.io/another/repo:v1", "docker.io/another/repo:v1"]
)
def test_publisher_rejects_foreign_plan_coordinates(monkeypatch, target):
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/ucm")
    plan = {
        "families": [{"id": "f", "create_index": True, "published_reference": target}],
        "publish": {"ghcr": {"enabled": True}, "dockerhub": {"enabled": False}},
    }
    with pytest.raises(ValueError, match="owner namespace"):
        delivery.publication_targets(plan, "index", "f")


def test_publisher_rejects_substituted_docker_hub_namespace(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/ucm")
    monkeypatch.setenv("DOCKERHUB_NAMESPACE", "docker.io/example")
    plan = {
        "families": [
            {
                "id": "f",
                "create_index": True,
                "published_reference": "ghcr.io/example/repo:v1",
            }
        ],
        "publish": {
            "ghcr": {"enabled": True},
            "dockerhub": {"enabled": True, "namespace": "docker.io/other"},
        },
    }
    with pytest.raises(ValueError, match="configured Docker Hub namespace"):
        delivery.publication_targets(plan, "index", "f")
