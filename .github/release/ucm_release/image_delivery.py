"""Publish verified OCI members and indexes, retaining immutable evidence for retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import release, runtime


def command(*args, password=None):
    result = subprocess.run(
        args, input=password, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip() or f"{args[0]} exited {result.returncode}"
        )
    return result.stdout.strip()


def registry_digest(reference):
    try:
        digest = command("crane", "digest", reference)
    except RuntimeError as error:
        # A failed authentication or unreadable registry must never authorize a push.
        if re.search(
            r"\b(?:MANIFEST_UNKNOWN|NAME_UNKNOWN):", str(error).splitlines()[-1]
        ):
            return None
        raise
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"invalid registry digest for {reference}: {digest!r}")
    return digest


def pinned(reference, digest):
    return reference.rsplit(":", 1)[0] + "@" + digest


def login(channel):
    if channel == "ghcr":
        host, user, secret = "ghcr.io", "GITHUB_ACTOR", "GH_TOKEN"
    else:
        host, user, secret = "docker.io", "DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"
    command(
        "docker",
        "login",
        host,
        "--username",
        os.environ[user],
        "--password-stdin",
        password=os.environ[secret] + "\n",
    )


def snapshot_binding(plan_path):
    # The frozen plan includes the repository, Tag, profile and exact member coordinates.
    return (
        os.environ["GITHUB_SHA"]
        + "-"
        + hashlib.sha256(plan_path.read_bytes()).hexdigest()
    )


def snapshots(root, binding, filename="*.json"):
    """Iterate immutable observations in run/attempt order; never merge a stale success."""
    result = []
    for path in root.glob(f"**/{binding}/run-*/attempt-*/{filename}"):
        attempt = int(path.parent.name.removeprefix("attempt-"))
        run_id = int(path.parent.parent.name.removeprefix("run-"))
        result.append(((run_id, attempt), path))
    return [path for _, path in sorted(result)]


def collect(root, binding, output):
    # A newer failed receipt replaces the observable state. Older digests are still
    # available to the publisher for ownership checks, even after a failed retry.
    output.mkdir(parents=True, exist_ok=True)
    for path in snapshots(root, binding):
        (output / path.name).write_bytes(path.read_bytes())


def read_targets(path, kind, identity, targets):
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict) or (receipt.get("kind"), receipt.get("id")) != (
        f"ucm-image-{kind}-receipt",
        identity,
    ):
        raise ValueError(f"unexpected publication receipt: {path}")
    return release.validated_receipt_targets(receipt, targets, str(path))


def receipt_pins(paths, kind, identity, targets):
    pins = {}
    for path in paths:
        for item in read_targets(path, kind, identity, targets):
            channel, digest = item["channel"], item["digest"]
            if pins.setdefault(channel, digest) != digest:
                raise ValueError(f"conflicting prior digests for {item['reference']}")
    return pins


def publish_member(reference, known_digest, source, archive):
    """An existing tag can only be reused with evidence from this exact release."""
    current = registry_digest(reference)
    if current is not None:
        if current != known_digest:
            raise ValueError(
                f"{reference}: existing digest {current}, proven digest {known_digest}"
            )
        return current
    if source:
        source_ref, source_digest = source
        if known_digest and source_digest != known_digest:
            raise ValueError(
                f"{reference}: recovery source differs from its recorded digest"
            )
        command("crane", "copy", pinned(source_ref, source_digest), reference)
        expected = source_digest
    else:
        if known_digest:
            raise ValueError(f"{reference}: recorded content is unavailable")
        if not archive.is_file():
            raise ValueError(
                f"{reference}: no verified archive or recoverable registry source"
            )
        # The digest file names the actual manifest written by Skopeo, including
        # its selected platform when BuildKit exported an OCI index.
        digest_file = archive.with_suffix(".digest")
        command(
            "skopeo",
            "copy",
            "--retry-times",
            "2",
            "--digestfile",
            str(digest_file),
            "oci-archive:" + str(archive),
            "docker://" + reference,
        )
        expected = digest_file.read_text().strip()
    actual = registry_digest(reference)
    if actual is None or actual != expected:
        raise ValueError(f"{reference}: expected {expected}, read back {actual}")
    return actual


def index_members(family, plan, members_dir, channel):
    """Select exactly one verified digest per planned platform in this registry."""
    selected = {}
    missing = []
    for member in family["members"]:
        identity = member["image_id"]
        receipt = members_dir / f"member-{identity}.json"
        if (
            member["reference"].rsplit(":", 1)[0]
            != family["published_reference"].rsplit(":", 1)[0]
        ):
            raise ValueError(
                f"{member['reference']}: member outside its planned index repository"
            )
        expected = runtime.image_publication_targets(plan, member["reference"])
        observed = (
            read_targets(receipt, "member", identity, expected)
            if receipt.is_file()
            else []
        )
        target = next((t for t in observed if t["channel"] == channel), None)
        platform = ("linux", member["cpu_arch"])
        if platform in selected:
            raise ValueError(f"duplicate platform {platform} for {family['id']}")
        if target is None:
            missing.append(member["cpu_arch"])
            continue
        ref, digest = target["reference"], target["digest"]
        if registry_digest(ref) != digest:
            raise ValueError(f"{ref}: member tag differs from its receipt")
        selected[platform] = pinned(ref, digest)
    if missing:
        raise ValueError("missing verified architecture(s): " + ", ".join(missing))
    return selected


def verify_index(reference, digest, platforms):
    document = json.loads(command("crane", "manifest", pinned(reference, digest)))
    expected = {platform: ref.rsplit("@", 1)[1] for platform, ref in platforms.items()}
    actual = {}
    for entry in document.get("manifests", []):
        platform = entry.get("platform", {})
        key = platform.get("os"), platform.get("architecture")
        if key in actual:
            raise ValueError(f"{reference}: duplicate index platform {key}")
        actual[key] = entry["digest"]
    if not expected or actual != expected:
        raise ValueError(
            f"{reference}: index platforms or digests differ from verified members"
        )


def publish_index(reference, known_digest, platforms):
    current = registry_digest(reference)
    if current is not None:
        if current != known_digest:
            raise ValueError(
                f"{reference}: existing index digest {current}, proven digest {known_digest}"
            )
        verify_index(reference, current, platforms)
        return current
    if known_digest:
        # Verify the frozen index before making its tag visible again.
        verify_index(reference, known_digest, platforms)
        command(
            "crane", "tag", pinned(reference, known_digest), reference.rsplit(":", 1)[1]
        )
    else:
        args = ["crane", "index", "append", "-t", reference]
        for (system, arch), member in platforms.items():
            config = json.loads(command("crane", "config", member))
            if (config["os"], config["architecture"]) != (system, arch):
                raise ValueError(
                    f"{member}: image does not match planned platform {system}/{arch}"
                )
            args.extend(["-m", member])
        command(*args)
    digest = registry_digest(reference)
    if digest is None or (known_digest and known_digest != digest):
        raise ValueError(f"{reference}: cannot verify published index digest")
    verify_index(reference, digest, platforms)
    return digest


def publication_targets(plan, kind, identity):
    if kind == "member":
        matches = [
            (family, member["reference"])
            for family in plan["families"]
            for member in family["members"]
            if member["image_id"] == identity
        ]
    else:
        matches = [
            (family, family["published_reference"])
            for family in plan["families"]
            if family["id"] == identity and family["create_index"]
        ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one planned {kind} for {identity!r}, found {len(matches)}"
        )
    family, reference = matches[0]
    targets = runtime.image_publication_targets(plan, reference)
    if not targets:
        raise ValueError("image publication is disabled")
    owner = os.environ["GITHUB_REPOSITORY"].split("/")[0].lower()
    if not reference.startswith(f"ghcr.io/{owner}/"):
        raise ValueError("image target outside repository owner namespace")
    hub = plan["publish"]["dockerhub"].get("namespace")
    if "dockerhub" in targets and (
        hub != os.environ.get("DOCKERHUB_NAMESPACE")
        or not re.fullmatch(r"docker\.io/[a-z0-9][a-z0-9_.-]{0,127}", hub or "")
    ):
        raise ValueError("image target outside configured Docker Hub namespace")
    return family, targets


def connect_registries(targets, pins):
    """Keep healthy channels available and find pinned sources for cross-registry repair."""
    failures = []
    sources = []
    # Authenticate independently. A broken Docker Hub login cannot suppress GHCR.
    available = {}
    for channel, ref in targets.items():
        try:
            login(channel)
            if channel in pins:
                digest = pins[channel]
                if registry_digest(pinned(ref, digest)) == digest:
                    sources.append((ref, digest))
            available[channel] = ref
        except (RuntimeError, ValueError) as error:
            failures.append(f"{ref}: {error}")
    return available, sources, failures


def deliver(args, plan, binding):
    kind, identity = args.kind, args.id
    family, targets = publication_targets(plan, kind, identity)
    filename = f"{kind}-{identity}.json"
    out = (
        args.output
        / binding
        / ("run-" + os.environ["GITHUB_RUN_ID"])
        / ("attempt-" + os.environ["GITHUB_RUN_ATTEMPT"])
        / filename
    )
    receipt = {
        "kind": f"ucm-image-{kind}-receipt",
        "schema_version": 1,
        "id": identity,
        "status": "failed",
        "targets": [],
    }
    release._write_json(out, receipt)
    history = snapshots(args.history, binding, filename)
    history += snapshots(args.plan.parent / "prior-image-receipts", binding, filename)
    pins = receipt_pins(history, kind, identity, targets)
    if args.command == "prepare":
        reuse = bool(pins) and set(pins) == set(targets)
        if pins and not reuse:
            # A partial publication can skip the build only if its bytes are
            # still accessible. A registry outage must not strand the other one.
            _, sources, _ = connect_registries(targets, pins)
            reuse = bool(sources)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            print(f"reuse={'true' if reuse else 'false'}", file=output)
        return

    available, sources, failures = connect_registries(
        targets, pins if kind == "member" else {}
    )
    for channel, ref in available.items():
        try:
            if kind == "index":
                platforms = index_members(family, plan, args.members, channel)
                digest = publish_index(ref, pins.get(channel), platforms)
            else:
                source = next((s for s in sources if s[1] == pins.get(channel)), None)
                if not source and sources:
                    source = sources[0]
                digest = publish_member(ref, pins.get(channel), source, args.archive)
            receipt["targets"].append(
                {"channel": channel, "reference": ref, "digest": digest}
            )
            release._write_json(out, receipt)
        except (RuntimeError, ValueError, OSError) as error:
            failures.append(f"{ref}: {error}")
    if failures:
        raise ValueError("\n".join(failures))
    receipt["status"] = "published"
    release._write_json(out, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "prepare", "publish"))
    parser.add_argument(
        "--plan", type=Path, default=Path("input/plan/release-plan.json")
    )
    parser.add_argument("--history", type=Path, default=Path("input/history"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=Path, default=Path("input/members"))
    parser.add_argument("--archive", type=Path, default=Path("out/image.oci.tar"))
    parser.add_argument("--kind", choices=("member", "index"))
    parser.add_argument("--id")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    if (
        plan["route"] != "release"
        or plan["publication_scope"] != os.environ["PUBLICATION_SCOPE"]
        or plan["repository"].lower() != os.environ["GITHUB_REPOSITORY"].lower()
        or (
            os.environ.get("GITHUB_REF_TYPE") == "tag"
            and plan["git_tag"] != os.environ["GITHUB_REF_NAME"]
        )
    ):
        raise ValueError("release plan does not belong to this publication")
    binding = snapshot_binding(args.plan)
    if args.command == "collect":
        collect(args.history, binding, args.output)
    else:
        if not args.id or not args.kind:
            parser.error("--id and --kind are required for this operation")
        deliver(args, plan, binding)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        sys.exit(str(error))
