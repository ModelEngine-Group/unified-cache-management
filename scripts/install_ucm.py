#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Install published UCM wheels without importing UCM.

The flow is: probe the target, discover compatible published backends, choose
a backend for its Toolkit, then invoke pip. PyPI owns package facts; selection functions
own policy. Keep both here so downloading this one file remains sufficient.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

if sys.version_info < (3, 10):  # noqa: UP036 - standalone interpreter check
    sys.exit("UCM: run install_ucm.py with Python 3.10 or newer.")

try:
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.tags import Tag, sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version
except ImportError:
    sys.exit("UCM: this Python needs pip with its bundled packaging library.")


# These are publication/Toolkit interfaces, not lists of available versions.
META_PACKAGE = "uc-manager"
PYPI_INDEX = "https://pypi.org/simple"
NON_BACKEND_PACKAGES = {f"{META_PACKAGE}-toolkit"}
CANN_ROOT_VARIABLES = ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
CUDA_ROOT_VARIABLES = ("CUDA_HOME", "CUDA_PATH")
ASCEND_SOC_FAMILIES = (
    (r"ascend910b\d*", "a2"),
    (r"ascend910_93\d+", "a3"),
)


@dataclass(frozen=True)
class AcceleratorRuntime:
    """An extra family and Toolkit coordinates (CANN: three parts; CUDA: two)."""

    family: str
    toolkit_version: tuple[int, ...]

    @property
    def version_text(self) -> str:
        return ".".join(map(str, self.toolkit_version))

    @property
    def extra(self) -> str:
        """Encode the release publisher's Toolkit coordinates without guessing ranges."""
        version = "".join(map(str, self.toolkit_version))
        return (
            f"cu{version}" if self.family == "cuda" else f"cann{version}-{self.family}"
        )


@dataclass(frozen=True)
class TargetEnvironment:
    """Interpreter and runtime facts, independent of published UCM versions."""

    markers: dict[str, str]
    tags: tuple[Tag, ...]
    accelerator: AcceleratorRuntime | None = None

    def supports_python(self, requires_python: str | None) -> bool:
        return SpecifierSet(requires_python or "").contains(
            self.markers["python_full_version"], prereleases=True
        )

    def to_json(self) -> dict:
        return {
            "markers": self.markers,
            "tags": [str(tag) for tag in self.tags],
            "accelerator": (
                {
                    "family": self.accelerator.family,
                    "toolkit_version": list(self.accelerator.toolkit_version),
                }
                if self.accelerator
                else None
            ),
        }

    @classmethod
    def from_json(cls, data: dict) -> "TargetEnvironment":
        """Load probe facts without filling missing fields from the resolver host."""
        if not isinstance(data, dict):
            raise TypeError("runtime probe must be a JSON object")
        markers = data["markers"]
        if (
            not isinstance(markers, dict)
            or not set(default_environment()).issubset(markers)
            or any(not isinstance(value, str) for value in markers.values())
            or markers["sys_platform"] != "linux"
        ):
            raise ValueError(
                "runtime markers must contain the complete Linux probe environment"
            )
        raw_tags = data["tags"]
        if (
            not isinstance(raw_tags, list)
            or not raw_tags
            or any(
                not isinstance(tag, str) or len(tag.split("-")) != 3 for tag in raw_tags
            )
        ):
            raise ValueError(
                "runtime tags must be a non-empty array of wheel tag strings"
            )
        accelerator = data["accelerator"]
        if accelerator is not None:
            if not isinstance(accelerator, dict):
                raise ValueError("runtime accelerator must be an object or null")
            family = accelerator["family"]
            version = accelerator["toolkit_version"]
            if (
                not isinstance(family, str)
                or re.fullmatch(r"cuda|a\d+", family) is None
                or not isinstance(version, list)
                or len(version) != (2 if family == "cuda" else 3)
                or any(type(part) is not int or part < 0 for part in version)
            ):
                raise ValueError(
                    "runtime accelerator must contain a family and Toolkit coordinates"
                )
            accelerator = AcceleratorRuntime(family, tuple(version))
        return cls(
            markers, tuple(Tag(*tag.split("-")) for tag in raw_tags), accelerator
        )


@dataclass(frozen=True)
class BackendCandidate:
    """A metadata-declared backend with a wheel compatible with the target."""

    extra: str
    package_name: str
    package_version: Version
    wheel_filename: str

    @property
    def requirement(self) -> str:
        return f"{self.package_name}=={self.package_version}"


def diagnostic(message: str) -> None:
    print(f"UCM: {message}", file=sys.stderr)


def runtime_version(value: str, source: str) -> tuple[int, int, int]:
    """Toolkit build suffixes do not change the published backend coordinates."""
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+)|\.RC\d+)?(?:\D|$)", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot read a Toolkit version from {source}: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def consistent_version(sources: list[tuple[str, str]]) -> tuple[int, ...] | None:
    versions = {runtime_version(value, source) for source, value in sources}
    if len(versions) > 1:
        raise ValueError(f"conflicting Toolkit metadata: {sources}")
    return next(iter(versions)) if versions else None


def toolkit_roots(names: tuple[str, ...], defaults: list[Path]) -> list[Path]:
    configured = [Path(os.environ[name]) for name in names if os.environ.get(name)]
    return list(dict.fromkeys(path.resolve() for path in (configured or defaults)))


def cann_version() -> tuple[int, ...] | None:
    roots = toolkit_roots(
        CANN_ROOT_VARIABLES,
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
        ],
    )
    sources = []
    for root in roots:
        # Only Toolkit metadata: recursive scans also find driver/ops versions.
        paths = [root / "version.cfg", root / "version.info"]
        for pattern in (
            "*-linux/ascend_toolkit_install.info",
            "*-linux/version.info",
            "*-linux/version.cfg",
        ):
            paths.extend(sorted(root.glob(pattern)))
        for path in paths:
            if not path.is_file():
                continue
            for line in path.read_text().splitlines():
                match = re.match(
                    r"\s*(?:version|cann_version)\s*[:=]\s*[\"']?([^\s\"']+)",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    sources.append((str(path), match[1]))
    version = consistent_version(sources)
    if version is None and (
        any(root.exists() for root in roots)
        or any(os.environ.get(key) for key in CANN_ROOT_VARIABLES)
    ):
        raise ValueError(
            f"CANN Toolkit metadata was not found in {roots}; specify --extra"
        )
    return version


def cuda_version() -> tuple[int, ...] | None:
    configured = any(os.environ.get(key) for key in CUDA_ROOT_VARIABLES)
    nvcc = shutil.which("nvcc")
    defaults = [Path("/usr/local/cuda")]
    if nvcc:
        defaults.append(Path(nvcc).resolve().parent.parent)
    roots = toolkit_roots(CUDA_ROOT_VARIABLES, defaults)
    sources = []
    for root in roots:
        metadata = root / "version.json"
        legacy = root / "version.txt"
        if metadata.is_file():
            data = json.loads(metadata.read_text())
            sources.append((str(metadata), data["cuda"]["version"]))
        elif legacy.is_file():
            match = re.search(r"CUDA Version\s+(\S+)", legacy.read_text())
            if not match:
                raise ValueError(f"cannot read CUDA Toolkit metadata: {legacy}")
            sources.append((str(legacy), match[1]))
        elif (root / "bin/nvcc").is_file():
            result = subprocess.run(
                [str(root / "bin/nvcc"), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
            if not match:
                raise ValueError(
                    f"cannot read CUDA Toolkit version from {root / 'bin/nvcc'}"
                )
            sources.append((str(root / "bin/nvcc"), match[1]))
    version = consistent_version(sources)
    if version is None and (configured or any(root.exists() for root in roots)):
        raise ValueError(
            f"CUDA Toolkit metadata was not found in {roots}; specify --extra"
        )
    return version[:2] if version else None


def detect_accelerator() -> AcceleratorRuntime:
    cann, cuda = cann_version(), cuda_version()
    if cann and cuda:
        raise ValueError("both CANN and CUDA Toolkits were found; specify --extra")
    if cuda:
        return AcceleratorRuntime(family="cuda", toolkit_version=cuda)
    if cann:
        soc = os.environ.get("SOC_VERSION", "").casefold()
        for pattern, family in ASCEND_SOC_FAMILIES:
            if re.fullmatch(pattern, soc):
                return AcceleratorRuntime(family=family, toolkit_version=cann)
        raise ValueError(
            f"cannot identify A2/A3 from SOC_VERSION={soc!r}; specify --extra"
        )
    raise ValueError("no CANN or CUDA Toolkit was identified; specify --extra")


def probe_environment(automatic: bool) -> TargetEnvironment:
    markers = default_environment()
    architecture = platform.machine().lower()
    libc, glibc = platform.libc_ver()
    if sys.platform != "linux" or libc != "glibc" or not glibc:
        raise ValueError(
            f"requires Linux with glibc; found {sys.platform}, {architecture}, {libc} {glibc}"
        )
    os_release = platform.freedesktop_os_release()
    diagnostic(
        f"Python {markers['python_full_version']}, {architecture}, glibc {glibc}, {os_release.get('PRETTY_NAME', 'Linux')}"
    )
    accelerator = detect_accelerator() if automatic else None
    if accelerator:
        diagnostic(f"detected {accelerator.family} Toolkit {accelerator.version_text}")
    # Wheel tags, rather than an architecture allowlist, determine availability.
    return TargetEnvironment(markers, tuple(sys_tags()), accelerator)


class PyPI:
    """Official publication metadata, cached only for this invocation."""

    def __init__(self) -> None:
        self._project_files: dict[str, list[dict]] = {}
        self._wheel_metadata: dict[str, Message] = {}

    def read(self, url: str, accept: str = "application/octet-stream") -> bytes:
        try:
            with urlopen(
                Request(url, headers={"Accept": accept}), timeout=30
            ) as response:
                return response.read()
        except HTTPError:
            # The index caller distinguishes an absent project from a failure.
            raise
        except OSError as error:
            raise RuntimeError(f"cannot read {url}: {error}") from error

    def files(self, project: str) -> list[dict]:
        name = canonicalize_name(project)
        if name not in self._project_files:
            url = f"{PYPI_INDEX}/{quote(name, safe='')}/"
            try:
                data = self.read(url, "application/vnd.pypi.simple.v1+json")
            except HTTPError as error:
                if error.code != 404:
                    raise
                self._project_files[name] = []
            else:
                self._project_files[name] = json.loads(data)["files"]
        return self._project_files[name]

    def wheel_metadata(self, artifact: dict) -> Message:
        """Read PEP 658 metadata, or METADATA inside a wheel without a sidecar."""
        url = artifact["url"]
        if url not in self._wheel_metadata:
            sidecar = artifact.get("core-metadata") or artifact.get(
                "data-dist-info-metadata"
            )
            if sidecar:
                data = self.read(url + ".metadata")
                hashes = sidecar if isinstance(sidecar, dict) else {}
            else:
                data = self.read(url)
                hashes = artifact.get("hashes", {})
            if (
                hashes.get("sha256")
                and hashlib.sha256(data).hexdigest() != hashes["sha256"]
            ):
                raise ValueError(f"PyPI metadata/wheel hash mismatch: {url}")
            if not sidecar:
                with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                    names = [
                        name
                        for name in wheel.namelist()
                        if name.endswith(".dist-info/METADATA")
                    ]
                    if len(names) != 1:
                        raise ValueError(f"expected one wheel METADATA: {url}")
                    data = wheel.read(names[0])
            self._wheel_metadata[url] = BytesParser().parsebytes(data)
        return self._wheel_metadata[url]


def compatible_wheels(
    files: list[dict], project: str, version: Version, environment: TargetEnvironment
) -> list[dict]:
    """Order published wheels by this interpreter's supported-tag preference."""
    ranks = {tag: index for index, tag in enumerate(environment.tags)}
    candidates = []
    for artifact in files:
        if artifact.get("yanked") or not artifact["filename"].endswith(".whl"):
            continue
        name, wheel_version, _, tags = parse_wheel_filename(artifact["filename"])
        if name != canonicalize_name(project) or wheel_version != version:
            continue
        matching = tags.intersection(ranks)
        if matching and environment.supports_python(artifact.get("requires-python")):
            candidates.append(
                (min(ranks[tag] for tag in matching), artifact["filename"], artifact)
            )
    return [artifact for _, _, artifact in sorted(candidates)]


def usable_wheel(
    pypi: PyPI, project: str, version: Version, environment: TargetEnvironment
) -> tuple[dict, Message] | None:
    """Return the preferred compatible file and its verified core metadata."""
    for artifact in compatible_wheels(
        pypi.files(project), project, version, environment
    ):
        metadata = pypi.wheel_metadata(artifact)
        if (
            canonicalize_name(metadata["Name"] or "") != canonicalize_name(project)
            or Version(metadata["Version"]) != version
        ):
            raise ValueError(
                f"wheel metadata disagrees with filename: {artifact['filename']}"
            )
        if environment.supports_python(metadata.get("Requires-Python")):
            return artifact, metadata
    return None


def backend_requirement(
    metadata: Message, extra: str, markers: dict[str, str]
) -> tuple[str, Version] | None:
    """Follow the meta package's backend pin; never construct a backend name."""
    requirements = []
    for raw in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(raw)
        name = canonicalize_name(requirement.name)
        if (
            name.startswith(f"{META_PACKAGE}-")
            and name not in NON_BACKEND_PACKAGES
            and (
                requirement.marker is None
                or requirement.marker.evaluate({**markers, "extra": extra})
            )
        ):
            requirements.append(requirement)
    if not requirements:
        return None
    if len(requirements) != 1:
        raise ValueError(f"extra {extra!r} must declare exactly one UCM backend")
    requirement = requirements[0]
    pins = list(requirement.specifier)
    if (
        requirement.url
        or requirement.extras
        or len(pins) != 1
        or pins[0].operator != "=="
        or "*" in pins[0].version
    ):
        raise ValueError(
            f"extra {extra!r} does not pin a backend version: {requirement}"
        )
    return canonicalize_name(requirement.name), Version(pins[0].version)


def candidate_versions(files: list[dict], requested: str) -> list[Version]:
    """Try the exact request, or published non-dev versions newest first (RC too)."""
    if requested != "latest":
        return [Version(requested)]
    versions = {
        parse_wheel_filename(artifact["filename"])[1]
        for artifact in files
        if artifact["filename"].endswith(".whl") and not artifact.get("yanked")
    }
    return sorted(
        (version for version in versions if not version.is_devrelease), reverse=True
    )


def compatible_backends(
    pypi: PyPI, metadata: Message, environment: TargetEnvironment, extra: str
) -> list[BackendCandidate]:
    """Discover the requested backend(s) from one release's dependency metadata."""
    candidates = []
    for declared_extra in metadata.get_all("Provides-Extra", []):
        declared_extra = canonicalize_name(declared_extra)
        if extra == "auto":
            if (
                environment.accelerator is None
                or declared_extra != environment.accelerator.extra
            ):
                continue
        elif declared_extra != extra:
            continue

        pin = backend_requirement(metadata, declared_extra, environment.markers)
        if pin is None:
            continue
        package_name, package_version = pin
        wheel = usable_wheel(pypi, package_name, package_version, environment)
        if wheel is None:
            continue
        artifact, _metadata = wheel
        candidates.append(
            BackendCandidate(
                extra=declared_extra,
                package_name=package_name,
                package_version=package_version,
                wheel_filename=artifact["filename"],
            )
        )
    return candidates


def resolve(
    pypi: PyPI, environments: list[TargetEnvironment], version: str, extra: str
) -> dict:
    """Find the newest release and backend usable by every target environment."""
    if not environments:
        raise ValueError("at least one runtime environment is required")
    if extra == "auto" and any(target.accelerator is None for target in environments):
        raise ValueError(
            "automatic backend selection requires a Toolkit; specify --extra"
        )
    for candidate_version in candidate_versions(pypi.files(META_PACKAGE), version):
        selected = []
        published_version = str(candidate_version)
        for environment in environments:
            meta_wheel = usable_wheel(
                pypi, META_PACKAGE, candidate_version, environment
            )
            if meta_wheel is None:
                break
            _artifact, metadata = meta_wheel
            candidates = compatible_backends(pypi, metadata, environment, extra)
            if not candidates:
                break
            selected.append(candidates[0])
            published_version = str(Version(metadata["Version"]))
        if (
            len(selected) != len(environments)
            or len({(backend.extra, backend.requirement) for backend in selected}) != 1
        ):
            diagnostic(
                f"skip {candidate_version}: no common compatible backend wheel for {extra}"
            )
            continue

        backend = selected[0]
        diagnostic(
            f"selected {published_version} / {backend.extra} for {len(environments)} runtime(s)"
        )
        selection = {
            "version": published_version,
            "extra": backend.extra,
            "requirement": f"{META_PACKAGE}[{backend.extra}]=={published_version}",
            "backend_requirement": backend.requirement,
        }
        if len(environments) == 1:
            selection["wheel"] = backend.wheel_filename
        else:
            selection["wheels"] = [
                {
                    "architecture": environment.markers["platform_machine"],
                    "python": environment.markers["python_full_version"],
                    "wheel": backend.wheel_filename,
                }
                for environment, backend in zip(environments, selected)
            ]
        return selection
    targets = ", ".join(
        f"{target.markers['platform_machine']} Python {target.markers['python_full_version']}"
        + (f" {target.accelerator.extra}" if target.accelerator else "")
        for target in environments
    )
    raise ValueError(
        f"no published, non-yanked UCM wheel matches version={version}, extra={extra}, "
        f"all runtime targets ({targets}) and their ABI/glibc tags"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install published UCM wheels for this Python interpreter, architecture and Toolkit."
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="latest published release, including RC (default), or an exact version",
    )
    parser.add_argument(
        "--extra", default="auto", help="auto (default) or one published backend extra"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resolve",
        action="store_true",
        help="print the selection as JSON without installing",
    )
    action.add_argument(
        "--probe",
        action="store_true",
        help="print this runtime's environment as JSON without network access",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        action="append",
        default=[],
        help="use a --probe JSON file for --resolve; repeat for a common selection",
    )
    parser.add_argument(
        "--report", help="pass an installation report path to pip --report"
    )
    args = parser.parse_args()
    if not args.version or not args.extra or "," in args.extra:
        parser.error("--version must be non-empty and --extra must select one backend")
    if args.version != "latest":
        Version(args.version)
    if args.runtime and not args.resolve:
        parser.error(
            "--runtime requires --resolve; installation must run in the target environment"
        )
    if args.report and (args.resolve or args.probe):
        parser.error("--report is only available when installing")
    extra = canonicalize_name(args.extra)
    if args.probe:
        print(json.dumps(probe_environment(extra == "auto").to_json()))
        return
    environments = (
        [
            TargetEnvironment.from_json(json.loads(path.read_text()))
            for path in args.runtime
        ]
        if args.runtime
        else [probe_environment(extra == "auto")]
    )
    selection = resolve(PyPI(), environments, args.version, extra)
    if args.resolve:
        print(json.dumps(selection))
        return
    diagnostic(f"installing {selection['requirement']} with {sys.executable}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            *(["--report", args.report] if args.report else []),
            selection["requirement"],
        ],
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        diagnostic(f"cannot read {error.url}: {error}")
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        diagnostic(f"command failed (exit {error.returncode}): {error.cmd}")
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
    ) as error:
        diagnostic(str(error))
        sys.exit(1)
