# Integrate the wheel installer

Use UCM's standalone Python file `scripts/install_ucm.py` to install a published wheel in a user environment or a downstream project's image build. UCM owns environment detection and package selection; the integrating project only downloads the script, passes parameters, and chooses when to run it.

The script needs Python 3.10 or newer and pip in that interpreter. It uses pip's bundled packaging library and does not need a checkout, an existing UCM installation, a GPU/NPU, or Docker. Installation targets use Linux and glibc; architecture availability is determined by **actually published, compatible wheel tags**, not an architecture allowlist. AMD64/ARM64 on Ubuntu and openEuler are covered by the installation checks. For example, a package's `Requires-Python: >=3.10` does not make its `cp312` backend wheel usable with Python 3.10.

## Obtain one file

Pin the installer revision separately from the UCM package version. Replace `INSTALLER_REF` with the full commit SHA of the installer you have reviewed. The pinned script can still select newly published compatible packages at build time.

```bash
INSTALLER_REPOSITORY=ModelEngine-Group/unified-cache-management
INSTALLER_REF='<full-commit-sha>'
curl --fail --location \
  "https://raw.githubusercontent.com/${INSTALLER_REPOSITORY}/${INSTALLER_REF}/scripts/install_ucm.py" \
  --output install_ucm.py
python3 install_ucm.py
```

The default is `--version latest --extra auto`. `latest` means the highest published UCM version with compatible meta and backend wheels and exact Toolkit publication coordinates for the current environment, including prereleases such as RC. Development/nightly versions and yanked files are excluded. Versions follow PEP 440 ordering: `0.7.0 < 0.8.0rc1 < 0.8.0rc2 < 0.8.0`. Source tags do not determine availability, and the installer never builds from source.

## Choose parameters

```bash
# Use another interpreter and its pip configuration.
/path/to/python3.12 install_ucm.py

# Fix the UCM version and published backend extra.
python3 install_ucm.py --version 0.7.0 --extra cann910-a2

# Resolve without installing anything.
python3 install_ucm.py --resolve > ucm-selection.json
```

An exact version may be a published prerelease. It is never replaced with another version if unavailable. Yanked files are excluded even for an exact version. `--extra` accepts one published backend extra, such as `cann910-a3` or `cu129`; it is not a list of optional features such as `toolkit`.

Run the file with `python3` or the explicit path of the interpreter you want to use, such as `/path/to/python3.12`. Both detection and installation use that interpreter; the script invokes pip through `sys.executable -m pip install --only-binary=:all:` with a pinned requirement and inherits pip's index, proxy, certificate and other configuration. Discovery always reads official PyPI. If a configured mirror lacks the selected packages, pip fails; the installer does not change indexes or select an older version to hide the failure. Ordinary dependency resolution remains pip's responsibility.

`--resolve` writes one JSON object to stdout. Its stable fields are:

| Field | Meaning |
| --- | --- |
| `version` | Selected UCM meta-package version |
| `extra` | Selected backend extra |
| `requirement` | Pinned `uc-manager[extra]==version` requirement |
| `backend_requirement` | Backend distribution and version from the published dependency declaration |
| `wheel` | Compatible backend wheel filename |

Diagnostics go to stderr. Failures return a nonzero exit code. This JSON records the selection; it does not lock every transitive dependency or prove hardware compatibility.

## Environment and backend selection

Python/ABI, CPU architecture and glibc are checked against wheel tags and `Requires-Python`. The operating-system name is diagnostic information, not an allowlist. The installer reads available versions, extras, dependencies and wheel metadata from official PyPI; it does not consult UCM's vLLM-Ascend support declarations.

For Ascend, configure the active Toolkit environment before running the installer. `ASCEND_HOME_PATH` and `ASCEND_TOOLKIT_HOME` locate it; otherwise the script checks the standard Toolkit locations. The version comes from Toolkit installation metadata, never an image tag, a directory name, `CANN_VERSION`, driver metadata or `npu-smi`. `SOC_VERSION` identifies A2 (`ascend910b…`) or A3 (`ascend910_93…`). Missing or unknown SoC information requires an explicit extra.

For CUDA, `CUDA_HOME` or `CUDA_PATH` identifies the active Toolkit. Otherwise the script checks `/usr/local/cuda` and the Toolkit containing `nvcc` on `PATH`. It reads Toolkit version metadata, using that Toolkit's `nvcc --version` when needed. It does not use `nvidia-smi` or the driver's advertised CUDA capability.

Automatic selection fails if neither Toolkit is identifiable, metadata conflicts, or both accelerator families are present. An explicit extra bypasses automatic Toolkit/SoC detection; Python, architecture, glibc and wheel availability are still checked. This is useful for device-less build environments whose intended backend is known to the caller.

For each candidate UCM version, automatic selection requires **exact Toolkit publication coordinates** and, for Ascend, the same A2/A3 family. CANN compares major/minor/patch; CUDA compares major/minor. Toolkit build suffixes are not compared. If a matching backend is missing, `latest` tries older UCM versions; an exact UCM version fails. The installer never substitutes a nearby Toolkit version or crosses CUDA major versions.

`AcceleratorRuntime.extra` encodes coordinates as names such as `cann910-a2` or `cu129` and compares them with the extras declared on PyPI. It does not guess version digits by decoding compact names. The meta package's `Requires-Dist` determines the actual backend distribution and version. Toolkit paths, SoC family mappings and extra encoding are publication interfaces; adding a released version does not require a version allowlist.

Exact selection does not establish native-library, engine-interface or hardware compatibility. The installer does not install or change Toolkits, drivers, torch or inference engines; pip still resolves ordinary package dependencies.

## Share one selection across architectures

In each target image, export the environment with the Python that will install UCM:

```bash
# Run separately in the ARM64 and AMD64 target environments, saving each file.
python3 install_ucm.py --probe > runtime-arm64.json
python3 install_ucm.py --probe > runtime-amd64.json
```

`--probe` does not access the network or install packages. Its JSON contains complete PEP 508 `markers`, wheel `tags` as strings in interpreter preference order, and an `accelerator` object such as `{"family":"a2","toolkit_version":[9,1,0]}`. An explicit `--extra` skips Toolkit detection and produces `accelerator: null`.

On the build coordinator, resolve the target environments for the same backend together:

```bash
python3 install_ucm.py --resolve \
  --runtime runtime-amd64.json --runtime runtime-arm64.json > ucm-selection.json
```

`--runtime` may be repeated and is allowed only with `--resolve`. The resolver neither probes nor installs into its host environment. It chooses the highest UCM version and one backend extra whose wheels are compatible with every target. A missing wheel never causes one target to select a different version. A2 and A3 are different backends and cannot share one automatic selection group.

A single target retains the five output fields documented above. Multiple targets replace `wheel` with `wheels`, preserving the other four fields. Each entry, in input order, contains `architecture` (for example `aarch64`), `python` (the full version, for example `3.12.4`) and `wheel` (the filename). Pass the selected `version` and `extra` to each target environment:

```bash
python3 install_ucm.py --version 0.8.0rc1 --extra cann910-a2 --report pip-report.json
```

`--report` is available only during installation and passes its path directly to `pip --report`, producing standard pip JSON. The backend entry's `metadata.name/version`, `download_info.url` and `download_info.archive_info.hashes.sha256` can verify the installed artifact. When the same version is already installed, pip may produce no new `install` entry. Check installed distribution versions in that case; a missing download record does not prove a hash was checked during this invocation. The installer does not force reinstallation to generate a report.

## Call from a Docker build

Download a reviewed installer revision before building, then copy that single file into the build context:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UCM_VERSION=latest
ARG UCM_EXTRA=auto
COPY install_ucm.py /tmp/install_ucm.py
RUN python3 /tmp/install_ucm.py --version "${UCM_VERSION}" --extra "${UCM_EXTRA}" \
    && rm /tmp/install_ucm.py
```

The base image must provide Python/pip and, for automatic detection, readable Toolkit metadata and the intended Ascend `SOC_VERSION`. Set up its normal Toolkit environment first if necessary. No device mounts are needed for selection or installation.

Motor and other callers retain their own build arguments, cache policy, architecture builds and manifest assembly. They should not copy the selection rules. Independent `latest` executions can select different versions. For one shared version, collect `--probe` output, resolve with repeated `--runtime` inputs, then pass the selected version/extra to each build. A cached Docker layer does not re-run discovery; manage cache invalidation in the calling project.

The root-level UCM `install.sh` remains a post-wheel installation hook; it is unrelated to this standalone entry point.

## Verification boundaries

Run script tests without an installed UCM or accelerator:

```bash
python3 -m py_compile scripts/install_ucm.py
python3 -m pytest -q scripts/tests
```

The installer CI separately installs real published wheels in AMD64/ARM64 Ubuntu/openEuler Ascend environments and a CUDA Toolkit environment, without accelerator devices. It checks installed distribution versions and the wheel in the pip report against the selection, and resolves the two architecture probes together. These checks do not constitute native-library or inference acceptance.

In a prepared target environment, explicitly check versions and imports after installation:

```bash
python3 -c 'from importlib.metadata import version; print(version("uc-manager"))'
python3 -c 'import ucm; from ucm.store.pipeline import ucmpipelinestore; print(ucm.__file__)'
```

Check the distribution named by `backend_requirement` separately. To test native linking, load the libraries used by your deployment, for example:

```python
import ctypes
import os
from pathlib import Path
import ucm

for relative in ("cache/libcachestore.so", "posix/libposixstore.so"):
    ctypes.CDLL(str(Path(ucm.__file__).parent / "store" / relative),
                mode=os.RTLD_NOW | os.RTLD_LOCAL)
```

Native linking requires the corresponding runtime libraries. Real GPU/NPU acceptance additionally requires hardware, drivers, the target engine and a UCM cache save/load or inference workload. Report script tests, package installation, native loading and hardware execution as separate results.

## Release pipeline integration

Official releases exercise this same installer in the existing dynamic PyPI verification matrix, after publishing backend wheels and then the meta package. The release version is fixed for this check; the installer detects the target Toolkit/SoC, and its selected version, extra, backend and wheel must match the build plan and publication receipt. Existing native-library and toolkit smoke checks still run afterward.

Fork releases retain isolated TestPyPI package names and the existing installation check. The release metadata tests build the actual meta wheel and verify that the standalone installer understands its dependency declarations. The standalone installer workflow separately tests current public packages on both architectures; it does not publish packages or replace the release manifest.
