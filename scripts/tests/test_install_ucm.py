"""Exercise the standalone installer's published-package contract without hardware."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "install_ucm.py"
SPEC = importlib.util.spec_from_file_location("ucm_installer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = installer
SPEC.loader.exec_module(installer)


def environment(architecture="x86_64", family="a2", runtime=(9, 1, 0)):
    markers = installer.default_environment()
    markers.update(
        python_full_version="3.12.4",
        python_version="3.12",
        sys_platform="linux",
        platform_system="Linux",
        platform_machine=architecture,
    )
    from pip._vendor.packaging.tags import Tag

    tags = [
        Tag("cp312", "cp312", f"manylinux_2_{floor}_{architecture}")
        for floor in range(34, 16, -1)
    ] + [Tag("py3", "none", "any")]
    return installer.TargetEnvironment(
        markers=markers,
        tags=tuple(tags),
        accelerator=installer.AcceleratorRuntime(family, runtime),
    )


class Catalog:
    def __init__(self):
        self.projects = {}
        self.metadata = {}

    def add(self, name, version, *, extras=(), dependencies=(), architecture=None):
        tag = (
            f"cp312-cp312-manylinux_2_34_{architecture}"
            if architecture
            else "py3-none-any"
        )
        filename = f"{name.replace('-', '_')}-{version}-{tag}.whl"
        artifact = {
            "filename": filename,
            "url": f"https://files.pythonhosted.org/{filename}",
            "requires-python": ">=3.10",
            "yanked": False,
        }
        metadata = Message()
        metadata["Name"] = name
        metadata["Version"] = version
        metadata["Requires-Python"] = ">=3.10"
        for extra in extras:
            metadata["Provides-Extra"] = extra
        for requirement in dependencies:
            metadata["Requires-Dist"] = requirement
        self.projects.setdefault(name, []).append(artifact)
        self.metadata[artifact["url"]] = metadata
        return artifact

    def release(self, version, extras, architectures=("x86_64", "aarch64")):
        dependencies = []
        for extra in extras:
            name = (
                f"uc-manager-cuda-{extra}"
                if extra.startswith("cu")
                else f"uc-manager-{extra}"
            )
            dependencies.append(f'{name}=={version}; extra == "{extra}"')
            for arch in architectures:
                self.add(name, version, architecture=arch)
        self.add("uc-manager", version, extras=extras, dependencies=dependencies)

    def files(self, project):
        return self.projects.get(project, [])

    def wheel_metadata(self, artifact):
        return self.metadata[artifact["url"]]


@pytest.fixture
def catalog():
    result = Catalog()
    for version in ("0.7.0", "0.8.0"):
        result.release(
            version, ["cann901-a2", "cann910-a2", "cann910-a3", "cu129", "cu130"]
        )
    return result


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
@pytest.mark.parametrize(
    "family,runtime,expected",
    [
        ("a2", (8, 3, 0), None),
        ("a2", (9, 0, 1), "cann901-a2"),
        ("a2", (9, 0, 2), None),
        ("a2", (9, 1, 0), "cann910-a2"),
        ("a2", (10, 2, 0), None),
        ("a3", (9, 0, 1), None),
        ("a3", (9, 1, 0), "cann910-a3"),
        ("cuda", (11, 8), None),
        ("cuda", (12, 9), "cu129"),
        ("cuda", (12, 10), None),
        ("cuda", (13, 0), "cu130"),
        ("cuda", (14, 0), None),
    ],
)
def test_latest_requires_exact_toolkit_and_family(
    catalog, arch, family, runtime, expected
):
    if expected is None:
        with pytest.raises(ValueError, match="no published"):
            installer.resolve(
                catalog, [environment(arch, family, runtime)], "latest", "auto"
            )
        return
    result = installer.resolve(
        catalog, [environment(arch, family, runtime)], "latest", "auto"
    )
    assert result["version"] == "0.8.0"
    assert result["extra"] == expected
    assert result["wheel"].endswith(f"_{arch}.whl")


def test_latest_requires_a_common_release_for_all_architectures(catalog):
    catalog.release("0.9.0rc1", ["cann910-a2"])
    catalog.release("0.9.0rc2", ["cann910-a2"], architectures=("aarch64",))
    targets = [environment("x86_64"), environment("aarch64")]
    result = installer.resolve(catalog, targets, "latest", "auto")
    assert result["version"] == "0.9.0rc1"
    assert result["extra"] == "cann910-a2"
    assert result["wheels"] == [
        {
            "architecture": arch,
            "python": "3.12.4",
            "wheel": f"uc_manager_cann910_a2-0.9.0rc1-cp312-cp312-manylinux_2_34_{arch}.whl",
        }
        for arch in ("x86_64", "aarch64")
    ]
    assert "wheel" not in result
    with pytest.raises(ValueError, match="all runtime targets"):
        installer.resolve(catalog, targets, "0.9.0rc2", "auto")


def test_latest_skips_release_without_exact_toolkit(catalog):
    catalog.release("0.9.0rc1", ["cann901-a2"])
    assert (
        installer.resolve(catalog, [environment()], "latest", "auto")["version"]
        == "0.8.0"
    )


def test_runtime_group_requires_the_same_extra_and_backend_pin(catalog):
    with pytest.raises(ValueError, match="no published"):
        installer.resolve(
            catalog, [environment(), environment(family="a3")], "latest", "auto"
        )
    meta = catalog.metadata[catalog.projects["uc-manager"][1]["url"]]
    del meta["Requires-Dist"]
    meta["Requires-Dist"] = (
        'uc-manager-cann910-a2==0.8.0; extra == "cann910-a2" and platform_machine == "x86_64"'
    )
    meta["Requires-Dist"] = (
        'uc-manager-cann910-a2==0.7.0; extra == "cann910-a2" and platform_machine == "aarch64"'
    )
    with pytest.raises(ValueError, match="no published"):
        installer.resolve(
            catalog, [environment(), environment("aarch64")], "0.8.0", "auto"
        )


def test_new_release_and_backend_use_published_dependencies(catalog):
    catalog.add("uc-manager-next-backend", "2.0.1", architecture="x86_64")
    catalog.add(
        "uc-manager",
        "2.0.0",
        extras=["cann1020-a2"],
        dependencies=['uc-manager-next-backend==2.0.1; extra == "cann1020-a2"'],
    )
    result = installer.resolve(
        catalog, [environment(runtime=(10, 2, 0))], "latest", "auto"
    )
    assert result["requirement"] == "uc-manager[cann1020-a2]==2.0.0"
    assert result["backend_requirement"] == "uc-manager-next-backend==2.0.1"


def test_published_wheel_drives_architecture_support(catalog):
    catalog.release("2.0.0", ["cann910-a2"], architectures=("ppc64le",))
    result = installer.resolve(catalog, [environment("ppc64le")], "latest", "auto")
    assert result["version"] == "2.0.0"
    assert result["wheel"].endswith("_ppc64le.whl")


def test_explicit_rc_and_extra_are_honored(catalog):
    catalog.release("0.9.0rc1", ["cu129"])
    result = installer.resolve(catalog, [environment()], "0.9.0rc1", "cu129")
    assert result["requirement"] == "uc-manager[cu129]==0.9.0rc1"
    assert result["backend_requirement"] == "uc-manager-cuda-cu129==0.9.0rc1"


def test_latest_includes_rc_in_version_order_but_excludes_dev(catalog):
    for version in ("0.9.0rc1", "0.9.0rc10", "0.9.0rc2", "0.10.0.dev20260921"):
        catalog.release(version, ["cann910-a2"])
    result = installer.resolve(catalog, [environment()], "latest", "auto")
    assert result["requirement"] == "uc-manager[cann910-a2]==0.9.0rc10"

    catalog.release("0.9.0", ["cann910-a2"])
    result = installer.resolve(catalog, [environment()], "latest", "auto")
    assert result["version"] == "0.9.0"


def test_latest_rc_still_requires_a_compatible_non_yanked_wheel(catalog):
    catalog.release("0.9.0rc1", ["cann910-a2"], architectures=("aarch64",))
    assert (
        installer.resolve(catalog, [environment()], "latest", "auto")["version"]
        == "0.8.0"
    )
    assert (
        installer.resolve(catalog, [environment("aarch64")], "latest", "auto")[
            "version"
        ]
        == "0.9.0rc1"
    )
    catalog.projects["uc-manager"][-1]["yanked"] = True
    assert (
        installer.resolve(catalog, [environment("aarch64")], "latest", "auto")[
            "version"
        ]
        == "0.8.0"
    )


def test_equivalent_version_input_reports_the_published_version(catalog):
    result = installer.resolve(catalog, [environment()], "0.7", "cann910-a2")
    assert result["version"] == "0.7.0"
    assert result["requirement"] == "uc-manager[cann910-a2]==0.7.0"


@pytest.mark.parametrize(
    "failure",
    [
        "yanked",
        "python",
        "abi",
        "architecture",
        "glibc",
        "sdist",
        "metadata-python",
        "missing",
    ],
)
def test_incompatible_latest_falls_back_but_exact_version_fails(catalog, failure):
    files = catalog.projects["uc-manager-cann910-a2"]
    for artifact in list(files):
        if "-0.8.0-" not in artifact["filename"]:
            continue
        if failure == "yanked":
            artifact["yanked"] = "bad release"
        elif failure == "python":
            artifact["requires-python"] = ">=3.13"
        elif failure == "metadata-python":
            catalog.metadata[artifact["url"]].replace_header(
                "Requires-Python", ">=3.13"
            )
        elif failure == "missing":
            files.remove(artifact)
        else:
            before, after = {
                "abi": ("cp312-cp312", "cp311-cp311"),
                "architecture": ("x86_64", "ppc64le"),
                "glibc": ("manylinux_2_34", "manylinux_2_40"),
                "sdist": (".whl", ".tar.gz"),
            }[failure]
            artifact["filename"] = artifact["filename"].replace(before, after)
    assert (
        installer.resolve(catalog, [environment()], "latest", "cann910-a2")["version"]
        == "0.7.0"
    )
    with pytest.raises(ValueError, match="no published"):
        installer.resolve(catalog, [environment()], "0.8.0", "cann910-a2")


def test_missing_extra_and_legacy_metadata_do_not_install_empty_meta(catalog):
    catalog.add("uc-manager", "3.0.0")
    assert (
        installer.resolve(catalog, [environment()], "latest", "auto")["version"]
        == "0.8.0"
    )
    for version, extra in (("3.0.0", "auto"), ("0.8.0", "toolkit"), ("8.0.0", "auto")):
        with pytest.raises(ValueError, match="no published"):
            installer.resolve(catalog, [environment()], version, extra)


def test_dependency_markers_use_target_environment(catalog):
    meta = catalog.metadata[catalog.projects["uc-manager"][1]["url"]]
    del meta["Requires-Dist"]
    meta["Requires-Dist"] = (
        'uc-manager-cann910-a2==0.8.0; extra == "cann910-a2" and platform_machine == "aarch64"'
    )
    assert installer.resolve(catalog, [environment("aarch64")], "0.8.0", "cann910-a2")
    with pytest.raises(ValueError, match="no published"):
        installer.resolve(catalog, [environment()], "0.8.0", "cann910-a2")


def test_yanked_meta_wheel_is_never_selected(catalog):
    catalog.projects["uc-manager"][1]["yanked"] = True
    assert (
        installer.resolve(catalog, [environment()], "latest", "auto")["version"]
        == "0.7.0"
    )


@pytest.mark.parametrize(
    "value,expected",
    [("9.1.0.0.123", (9, 1, 0)), ("8.3.RC1", (8, 3, 0)), ("12.9", (12, 9, 0))],
)
def test_toolkit_versions(value, expected):
    assert installer.runtime_version(value, "test") == expected


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    root = tmp_path / "misleading-cann-99.0.0"
    root.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(root))
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.setenv("CANN_VERSION", "88.0.0")
    return root


def test_cann_uses_toolkit_metadata_not_path_env_or_driver(toolkit):
    metadata = toolkit / "aarch64-linux/ascend_toolkit_install.info"
    metadata.parent.mkdir()
    metadata.write_text("version=9.1.0\n")
    driver = toolkit / "driver/version.info"
    driver.parent.mkdir()
    driver.write_text("version=25.0.0\n")
    assert installer.cann_version() == (9, 1, 0)


def test_missing_and_conflicting_cann_metadata_fail(toolkit):
    with pytest.raises(ValueError, match="metadata was not found"):
        installer.cann_version()
    (toolkit / "version.cfg").write_text("version=9.1.0\n")
    (toolkit / "version.info").write_text("version=9.0.1\n")
    with pytest.raises(ValueError, match="conflicting"):
        installer.cann_version()


@pytest.mark.parametrize("metadata", ["json", "legacy", "nvcc"])
def test_cuda_uses_toolkit_without_device(tmp_path, monkeypatch, metadata):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CUDA_VERSION", "99.0")
    if metadata == "json":
        (tmp_path / "version.json").write_text(
            json.dumps({"cuda": {"version": "12.9.1"}})
        )
    elif metadata == "legacy":
        (tmp_path / "version.txt").write_text("CUDA Version 12.9.1\n")
    else:
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin/nvcc").touch()

        def run(command, **kwargs):
            assert command == [str(tmp_path / "bin/nvcc"), "--version"]
            return subprocess.CompletedProcess(
                command, 0, stdout="Cuda compilation tools, release 12.9, V12.9.86"
            )

        monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.cuda_version() == (12, 9)


@pytest.mark.parametrize(
    "soc,family",
    [("Ascend910B1", "a2"), ("ascend910b4", "a2"), ("ascend910_9391", "a3")],
)
def test_soc_detection(monkeypatch, soc, family):
    monkeypatch.setattr(installer, "cann_version", lambda: (9, 1, 0))
    monkeypatch.setattr(installer, "cuda_version", lambda: None)
    monkeypatch.setenv("SOC_VERSION", soc)
    assert installer.detect_accelerator() == installer.AcceleratorRuntime(
        family, (9, 1, 0)
    )


@pytest.mark.parametrize(
    "cann,cuda,soc",
    [
        (None, None, ""),
        ((9, 1, 0), (12, 9), "ascend910b1"),
        ((9, 1, 0), None, "unknown"),
    ],
)
def test_ambiguous_accelerator_reports_explicit_extra(monkeypatch, cann, cuda, soc):
    monkeypatch.setattr(installer, "cann_version", lambda: cann)
    monkeypatch.setattr(installer, "cuda_version", lambda: cuda)
    monkeypatch.setenv("SOC_VERSION", soc)
    with pytest.raises(ValueError, match="specify --extra"):
        installer.detect_accelerator()


@pytest.mark.parametrize("distro", ["Ubuntu 22.04", "openEuler 24.03"])
@pytest.mark.parametrize("arch", ["x86_64", "aarch64", "ppc64le"])
def test_probe_without_accelerator_with_explicit_extra(monkeypatch, distro, arch):
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: arch)
    monkeypatch.setattr(installer.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(
        installer.platform, "freedesktop_os_release", lambda: {"PRETTY_NAME": distro}
    )
    monkeypatch.setattr(installer, "sys_tags", lambda: iter(environment(arch).tags))
    monkeypatch.setattr(
        installer,
        "detect_accelerator",
        lambda: pytest.fail("explicit extra must not need hardware detection"),
    )
    assert installer.probe_environment(False).accelerator is None


@pytest.mark.parametrize(
    "system,arch,libc",
    [
        ("darwin", "aarch64", ("", "")),
        ("linux", "x86_64", ("musl", "1.2")),
    ],
)
def test_unsupported_platform_fails(monkeypatch, system, arch, libc):
    monkeypatch.setattr(installer.sys, "platform", system)
    monkeypatch.setattr(installer.platform, "machine", lambda: arch)
    monkeypatch.setattr(installer.platform, "libc_ver", lambda: libc)
    with pytest.raises(ValueError, match="requires Linux"):
        installer.probe_environment(False)


@pytest.mark.parametrize(
    "error,expected_type",
    [
        (URLError("offline"), RuntimeError),
        (HTTPError("url", 503, "unavailable", {}, None), HTTPError),
    ],
)
def test_network_errors_are_not_missing_candidates(monkeypatch, error, expected_type):
    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(installer, "urlopen", unavailable)
    with pytest.raises(expected_type):
        installer.PyPI().files("uc-manager")


def test_missing_project_is_empty_but_missing_wheel_metadata_is_an_error(monkeypatch):
    def missing(request, **kwargs):
        raise HTTPError(request.full_url, 404, "not found", {}, None)

    monkeypatch.setattr(installer, "urlopen", missing)
    pypi = installer.PyPI()
    assert pypi.files("uc-manager-missing") == []
    with pytest.raises(HTTPError):
        pypi.wheel_metadata(
            {"url": "https://files.pythonhosted.org/missing.whl", "core-metadata": True}
        )


@pytest.mark.parametrize("sidecar", [True, False])
def test_read_actual_wheel_metadata_or_sidecar(monkeypatch, sidecar):
    data = b"Name: uc-manager\nVersion: 0.7.0\nProvides-Extra: cu129\n"
    if not sidecar:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as wheel:
            wheel.writestr("uc_manager-0.7.0.dist-info/METADATA", data)
        data = buffer.getvalue()
    artifact = {"url": "https://files.pythonhosted.org/example.whl"}
    artifact["core-metadata" if sidecar else "hashes"] = {
        "sha256": hashlib.sha256(data).hexdigest()
    }

    def read(self, url):
        assert url == artifact["url"] + (".metadata" if sidecar else "")
        return data

    monkeypatch.setattr(installer.PyPI, "read", read)
    metadata = installer.PyPI().wheel_metadata(artifact)
    assert metadata.get_all("Provides-Extra") == ["cu129"]
    artifact["core-metadata" if sidecar else "hashes"]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="hash mismatch"):
        installer.PyPI().wheel_metadata(artifact)


@pytest.mark.parametrize("resolve_only", [True, False])
def test_cli_resolve_and_install_contract(monkeypatch, catalog, capsys, resolve_only):
    monkeypatch.setattr(installer, "PyPI", lambda: catalog)
    monkeypatch.setattr(installer, "probe_environment", lambda automatic: environment())
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setattr(installer.sys, "executable", "/chosen/python")
    monkeypatch.setattr(
        installer.sys,
        "argv",
        ["-", "--extra", "cu129"] + (["--resolve"] if resolve_only else []),
    )
    commands = []

    def run(command, **kwargs):
        assert installer.os.environ["PIP_INDEX_URL"] == "https://mirror.example/simple"
        assert kwargs == {"check": True}
        commands.append(command)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.main()
    if resolve_only:
        result = json.loads(capsys.readouterr().out)
        assert set(result) == {
            "version",
            "extra",
            "requirement",
            "backend_requirement",
            "wheel",
        }
        assert commands == []
    else:
        assert commands == [
            [
                "/chosen/python",
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "uc-manager[cu129]==0.8.0",
            ]
        ]


def test_probe_round_trip_without_network_or_install(monkeypatch, capsys):
    target = environment("aarch64")
    monkeypatch.setattr(installer, "probe_environment", lambda automatic: target)
    monkeypatch.setattr(
        installer, "PyPI", lambda: pytest.fail("probe must not access PyPI")
    )
    monkeypatch.setattr(installer.sys, "argv", ["-", "--probe"])
    installer.main()
    data = json.loads(capsys.readouterr().out)
    assert data["accelerator"] == {"family": "a2", "toolkit_version": [9, 1, 0]}
    assert data["tags"][0] == "cp312-cp312-manylinux_2_34_aarch64"
    assert installer.TargetEnvironment.from_json(data) == target


def test_runtime_probe_does_not_fill_missing_markers_from_host():
    data = environment().to_json()
    del data["markers"]["platform_machine"]
    with pytest.raises(ValueError, match="complete Linux probe"):
        installer.TargetEnvironment.from_json(data)


def test_cli_resolves_runtime_files_without_probing_the_host(
    monkeypatch, catalog, capsys, tmp_path
):
    args = ["-", "--resolve"]
    for arch in ("x86_64", "aarch64"):
        path = tmp_path / f"{arch}.json"
        path.write_text(json.dumps(environment(arch).to_json()))
        args.extend(["--runtime", str(path)])
    monkeypatch.setattr(installer.sys, "argv", args)
    monkeypatch.setattr(installer, "PyPI", lambda: catalog)
    monkeypatch.setattr(
        installer,
        "probe_environment",
        lambda automatic: pytest.fail("host is not a target"),
    )
    installer.main()
    selection = json.loads(capsys.readouterr().out)
    assert selection["version"] == "0.8.0"
    assert [wheel["architecture"] for wheel in selection["wheels"]] == [
        "x86_64",
        "aarch64",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--runtime", "runtime.json"],
        ["--resolve", "--report", "report.json"],
        ["--probe", "--report", "report.json"],
    ],
)
def test_cli_rejects_installation_with_runtime_files_or_reports_in_read_only_modes(
    monkeypatch, args
):
    monkeypatch.setattr(installer.sys, "argv", ["-", *args])
    monkeypatch.setattr(
        installer, "PyPI", lambda: pytest.fail("invalid options must not access PyPI")
    )
    with pytest.raises(SystemExit) as error:
        installer.main()
    assert error.value.code == 2


def test_cli_passes_report_path_to_pip(monkeypatch, catalog, tmp_path):
    report = str(tmp_path / "install report.json")
    monkeypatch.setattr(installer.sys, "argv", ["-", "--report", report])
    monkeypatch.setattr(installer, "PyPI", lambda: catalog)
    monkeypatch.setattr(installer, "probe_environment", lambda automatic: environment())
    commands = []
    monkeypatch.setattr(
        installer.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    installer.main()
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "--report",
            report,
            "uc-manager[cann910-a2]==0.8.0",
        ]
    ]


def test_downloaded_file_starts_with_only_pip(tmp_path):
    python = tmp_path / "venv/bin/python"
    subprocess.run(
        [sys.executable, "-m", "venv", str(python.parent.parent)], check=True
    )
    script = tmp_path / "install_ucm.py"
    script.write_text(SCRIPT.read_text())
    result = subprocess.run(
        [str(python), str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--resolve" in result.stdout
    assert "ucm" not in installer.__dict__
