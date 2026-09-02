# UCM Registry-driven release automation

The active pipeline builds UCM Wheels and the Helm Chart from actual published
runtime images. Stable, Prerelease, Draft, and Nightly Profiles independently
select the five publication channels and retention limits. The pipeline never
reads vLLM or vLLM-Ascend source branches to decide versions or Builder
capabilities.

## Maintained policy

The human-maintained release authorities are:

- `version.ini`: the UCM base version plus the exact supported vLLM and
  vLLM-Ascend Runtime selectors for this source version;
- `release.yaml`: runtime repositories, runners, fixed publication addresses,
  four fully expanded Release Profiles, retention, and Chart smoke inputs;
- `platforms.yaml`: raw Builder registries, excluded variants, Builder checks,
  and supported or blocked UCM backends;
- `requirements/wheel-build.txt` and `requirements/wheel-runtime.txt`: exact
  Python dependencies.

Each product selector is a literal Registry-tag keyword such as `0.27.1` or
`0.25.1rc`. A bare keyword expands every legal tag variant containing that
delimited keyword. It does not use PEP 440 normalization, so `0.25.1` does not
match `0.25.1rc` or `0.25.10`. An explicit `keyword@tag` binding such as
`0.25.1rc@nightly-releases-v0.25.1rc-a3` selects only that exact published Tag;
the Tag must contain the keyword. Exact bindings are reported as pinned and are
not expanded. Missing keywords, missing explicit Tags, and selector sets that
produce no publishable Runtime fail the release. All four Release Profiles
consume the same selectors; Profiles no longer trim the Runtime matrix. 310P is
filtered and A5 is reported as blocked.

This keyword rule applies only to `UCM_SUPPORTED_VLLM_VERSIONS` and
`UCM_SUPPORTED_VLLM_ASCEND_VERSIONS`. `UCM_VERSION` remains a canonical PEP
440 package version because it drives Wheel, Chart, and Release coordinates.

For example:

```ini
UCM_SUPPORTED_VLLM_VERSIONS=0.26.0,0.27.1,0.28.0
UCM_SUPPORTED_VLLM_ASCEND_VERSIONS=0.24.0rc,0.25.1rc,0.26.0rc
```

## Registry-only flow

```text
Runtime Registry Tags
  -> version selection
  -> manifest/member + Crane config inspection
  -> native fallback probe only for missing capability fields
  -> compatible raw Builder Registry match
  -> digest-pinned label-only Builder mirror
  -> Wheel union
  -> optional runtime images and Chart
```

Crane reads each immutable member manifest/config to obtain CUDA/CANN,
SOC/backend, Python ABI, and CPU architecture without downloading
image layers. Only a member missing Python, CUDA/CANN, or SOC metadata is pulled
and probed on its native architecture. Runtime glibc does not gate planning:
the Wheel's actual GLIBC floor comes from `auditwheel`, then the final
install-only Runtime image verifies its glibc floor, installs the Wheel, and
runs `import ucm` before publication. Every runtime member resolves exactly one
Wheel ID. Zero or ambiguous matches fail instead of guessing.

The explicit `-openeuler` Runtime Tag suffix is retained as an OS hint for
Release mapping and final-image validation. OS hints never participate in the
Builder or Wheel capability key.

CUDA raw Builders come from the configured PyTorch manylinux repositories.
Ascend raw Builders come from `quay.io/ascend/manylinux`. Both use
`docker/Dockerfile.builder-mirror`; the mirror adds labels but installs no
software. Builder identity is based on the raw platform member digest and
checked capability, not a vLLM version or Git ref. The official pipeline does
not download, build, or identify Mooncake.

Every mirror's remote OCI config and labels are compared with the desired
catalog, and its final manifest digest is recorded. The mirror is normally
pulled and re-probed on its native architecture. If bounded retries prove that
the mirror pull is still rate-limited, the capability probe uses the exact raw
platform member recorded by `source_image@source_image_digest`; other failures
remain terminal. Wheel builds follow the same rule: prefer the mirror digest,
fall back only when the Builder repository itself remains rate-limited, and
never consume a mutable Builder Tag.

## Tag and Release modes

The workflow accepts:

| Git Tag | Selected Profile | Wheel version | Chart version | GitHub Release mode when enabled |
| --- | --- | --- | --- | --- |
| `vX.Y.Z` | Stable | `X.Y.Z` | `X.Y.Z` | public Release |
| `vX.Y.ZrcN` | Prerelease | `X.Y.ZrcN` | `X.Y.Z-rc.N` | public prerelease |
| `draft/vX.Y.Z` | Draft | `X.Y.Z.dev0` | `X.Y.Z-draft.0` | Draft |
| `draft/vX.Y.Z-N` | Draft | `X.Y.Z.devN` | `X.Y.Z-draft.N` | Draft |
| `nightly/vX.Y.Z-YYYYMMDD-N` | Nightly | `X.Y.Z.devYYYYMMDDNNN` | `X.Y.Z-nightly.YYYYMMDD.N` | public prerelease after success |

The tagged source owns the package model, and public versions must be canonical
and non-local. Before opening or reusing a Release, the reusable core requires
the Tag's complete `X.Y.Z` base to match `UCM_VERSION`, then reclassifies the
Tag and checks its release type, visibility, prerelease flag, Chart/image
versions, target commit, requested source SHA, and checked-out commit. A Draft
Tag always remains Draft. Exact Releases API lookup requires one Release for the
Tag; duplicate Release records fail closed.

At 02:00 Asia/Shanghai (`18:00` UTC), `release-nightly.yml` reads the `X.Y.Z`
base from `version.ini` and creates the next dated Nightly Tag from `develop`.
An incomplete same-SHA Nightly Tag is reused; an existing Tag is never moved.
Because a `GITHUB_TOKEN` Tag creation does not
recursively trigger another workflow, the same scheduled Run calls the common
`release-ucm.yml` reusable core directly. Manual `nightly/*` Tag pushes use the
same core through `release-tag.yml`.

Supported Release Tags in both the official repository and Forks invoke that
same Release Core through one caller. The caller passes the repository-derived
publication scope and conditionally forwards the matching Python index Secret:
`PYPI_API_TOKEN` only for the official repository and `TEST_PYPI_API_TOKEN`
only for Forks. Docker Hub credentials remain explicit for either scope, while
publication jobs bind to the corresponding Environment.

A fork publishes any Profile-enabled GitHub Release, GHCR, and Chart OCI outputs
under its own identity.

When `github_release` is enabled, publication updates the same Release in
stages:

1. version, Tag, Profile, and Fork target preflight passes, then every configured
   Runtime selector resolves to published Registry Tags;
2. `release-open` records the in-progress Release;
3. every repaired Wheel passes one matching native-architecture Runtime before
   any Release asset or PyPI upload;
4. backend Wheels, the example config, and the Chart are uploaded and the state
   is `artifacts-ready` while any enabled channel remains; the empty meta Wheel
   remains an internal Actions artifact;
5. image members/indexes, PyPI, and Chart OCI complete and are read back;
6. the final state becomes `complete`, `images-failed`, or
   `publication-failed`.

`release-state.json` remains the rich internal staging file in the
`ucm-release-stage-run-<run>` Actions artifact. Only after all enabled channels
succeed, a compact public `release-manifest.json` schema 6 is uploaded and read
back. It records only the Tag/type/Actions Run, Chart OCI reference, Runtime
member/index references, and GitHub Release asset names needed for cleanup.

If image publication is disabled while other channels remain enabled, the image
stages are skipped and publication continues only through those enabled
channels. If every channel is disabled, validation completes without an
external write. If requested image publication fails, already published
artifacts remain usable and the public Release is marked `images-failed`. OCI
archives are not uploaded to GitHub Release. PyPI and Docker Hub follow the
selected Release Profile in `release.yaml`, which is the sole authority for
channel switches. Official releases use production targets. Forks use TestPyPI
when `TEST_PYPI_API_TOKEN` exists. Docker Hub publication in either scope
requires `DOCKERHUB_NAMESPACE` plus both Docker Hub credentials. Missing
requested Docker Hub configuration fails preflight, while a Profile-disabled
channel stays disabled.
Missing backend Wheels are uploaded and read back first, the meta Wheel is
uploaded last, and exact extras metadata plus all filenames, versions, and
SHA256 digests are persisted in the internal `pypi-receipt.json` Actions
artifact. It is not attached to the public GitHub Release. Fresh-environment
install validation downloads the exact receipt-bound meta/backend URLs,
verifies their SHA256 digests, and uses production PyPI only for ordinary
dependencies; index ordering never chooses the UCM distributions. GitHub
Release notes omit Python-index installation commands for both official and
Fork releases.

Python distribution names are repository-owned and deterministic. The official
repository publishes canonical `uc-manager*` names. A Fork always prefixes the
same family with its losslessly normalized GitHub owner, independent of whether
credentials are present. For example, `SuperMarioYL` builds
`supermarioyl-uc-manager` and `supermarioyl-uc-manager-cuda-cu130`. The Python
import remains `ucm`.

## Official release setup

This section applies only to
`ModelEngine-Group/unified-cache-management`. The Tag workflow identifies that
repository as `publication_scope=official`.

Publication has two separate inputs:

1. the selected Profile in `release.yaml` requests the channel with
   `publish.<channel>: true`;
2. the repository contains the Secret or Variable required by that channel.

A Secret never turns on a Profile-disabled channel. Conversely, an enabled
official channel does not silently fall back to another target when its
configuration is missing.

### 1. Configure the official repository

Open **Settings -> Secrets and variables -> Actions** in the official repository.
Add tokens and credentials under **Repository secrets**, and add the Docker Hub
target under **Repository variables**:

| Name | GitHub kind | Required when | Value |
| --- | --- | --- | --- |
| `PYPI_API_TOKEN` | Repository Secret | Profile has `pypi: true` | Complete production PyPI API token authorized for every planned `uc-manager*` project. |
| `DOCKERHUB_USERNAME` | Repository Secret | Profile has `dockerhub: true` | Docker ID authorized for the configured namespace. |
| `DOCKERHUB_TOKEN` | Repository Secret | Profile has `dockerhub: true` | Docker Hub access token with Read & Write permission; add Delete permission when cleanup must remove tags. |
| `DOCKERHUB_NAMESPACE` | Repository Variable | Profile has `dockerhub: true` | Full namespace such as `docker.io/official-org`, without a repository name or trailing slash. |

The workflow reads the three credentials through `secrets.*` and reads only
`DOCKERHUB_NAMESPACE` through `vars.*`. Do not store tokens in Variables. Even
though the Docker Hub username is not sensitive, it must remain a Secret because
the workflow does not read `vars.DOCKERHUB_USERNAME`.

Under **Settings -> Actions -> General**, allow **Read and write permissions**
when repository or organization policy otherwise restricts `GITHUB_TOKEN`.

The equivalent GitHub CLI commands are:

```bash
repo=ModelEngine-Group/unified-cache-management
gh secret set PYPI_API_TOKEN --repo "${repo}"
gh secret set DOCKERHUB_USERNAME --repo "${repo}"
gh secret set DOCKERHUB_TOKEN --repo "${repo}"
gh variable set DOCKERHUB_NAMESPACE --repo "${repo}" \
  --body docker.io/official-org
```

Each command prompts for the value without writing it to the repository. Omit
the PyPI command while `pypi` is disabled, and omit both Docker Hub commands
plus the namespace command while `dockerhub` is disabled.

Release jobs still reference the Environment name `release-production`. You do
not need to create or populate it for repository-level configuration: GitHub
[creates an unconfigured referenced Environment automatically](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
Create it manually only when deployment reviewers, wait timers, or
Environment-level isolation are required; do not duplicate the same credential
at both repository and Environment levels.

### 2. Confirm the official targets

Official PyPI always uses `https://upload.pypi.org/legacy/` and publishes the
canonical `uc-manager*` distribution names. Never put a TestPyPI token in
`PYPI_API_TOKEN`.

Docker Hub always uses the exact `DOCKERHUB_NAMESPACE` value. It is not derived
from the GitHub owner or `DOCKERHUB_USERNAME`, and `release.yaml` contains no
fallback namespace. Image repository basenames and tags still come from the
frozen Release Plan.

### 3. Check the effective decision

Before pushing a formal Tag, inspect the selected Profile in `release.yaml`.
The Plan summary and frozen `release-plan.json` must show the same `requested`
value.

Official PyPI resolves as follows:

| Profile `pypi` | `PYPI_API_TOKEN` | Result |
| --- | --- | --- |
| `false` | absent or present | channel is disabled and the Secret is unused |
| `true` | present and authorized | publication and public readback run |
| `true` | absent | the enabled publication job fails |

Official Docker Hub resolves as follows:

| Profile `dockerhub` | Namespace + Secret pair | Result |
| --- | --- | --- |
| `false` | absent, partial, or complete | channel is disabled and the values are unused |
| `true` | all three configured and authorized | preflight passes; member and index publication run |
| `true` | namespace, username, or token missing | preflight fails before Registry discovery |

An invalid or unauthorized supplied credential can still fail authentication or
readback; `present` does not mean it has been accepted by the external service.

## Fork preview setup

This section applies to every repository other than the canonical official
repository. Fork PyPI and Docker Hub publication requires both a request from
the selected Release Profile and the corresponding Fork credential. It never
reads the official PyPI credential and it does not change the publication
targets of `ModelEngine-Group/unified-cache-management`.

### 1. Configure the Fork repository

Open **Settings -> Secrets and variables -> Actions** in the Fork. Add the
credentials as **Repository secrets** and the Docker Hub target as a
**Repository variable**:

| Name | GitHub kind | Required when | Value |
| --- | --- | --- | --- |
| `TEST_PYPI_API_TOKEN` | Repository Secret | Profile has `pypi: true` and TestPyPI publication is wanted | Account-scoped TestPyPI API token authorized for every Fork-owned `<owner>-uc-manager*` project. |
| `DOCKERHUB_USERNAME` | Repository Secret | Profile has `dockerhub: true` | Docker ID authorized for the configured namespace. |
| `DOCKERHUB_TOKEN` | Repository Secret | Profile has `dockerhub: true` | Docker Hub access token with Read & Write permission; add Delete permission when cleanup must remove tags. |
| `DOCKERHUB_NAMESPACE` | Repository Variable | Profile has `dockerhub: true` | Full namespace such as `docker.io/my-org`, without a repository name or trailing slash. |

For Fork scope, the shared Tag caller forwards the TestPyPI Secret and both
Docker Hub Secrets to the reusable Release Core. It does not forward
`PYPI_API_TOKEN`. The equivalent GitHub CLI commands are:

```bash
fork=OWNER/unified-cache-management
gh secret set TEST_PYPI_API_TOKEN --repo "${fork}"
gh secret set DOCKERHUB_USERNAME --repo "${fork}"
gh secret set DOCKERHUB_TOKEN --repo "${fork}"
gh variable set DOCKERHUB_NAMESPACE --repo "${fork}" \
  --body docker.io/my-org
```

Each `gh secret set` command prompts for its value. Omit the TestPyPI command
when that publication is not wanted, and omit all three Docker Hub settings
while `dockerhub` is disabled.

Fork jobs still reference the Environment name `fork-preview`, but
repository-level configuration does not require you to create or populate it.
GitHub creates an unconfigured referenced Environment automatically. Create it
manually only when approval or Environment-level isolation is required, and do
not duplicate the same credential at both levels.

Also open **Settings -> Actions -> General** and make sure Actions is enabled. If
repository or organization policy restricts `GITHUB_TOKEN` writes, allow
**Read and write permissions**. The workflow itself requests scoped `contents`
and `packages` writes for the Fork's Release, GHCR, and Chart OCI.
See [GitHub's Actions settings guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).

The TestPyPI endpoints are fixed by policy: upload uses
`https://test.pypi.org/legacy/`, package installation uses
`https://test.pypi.org/simple/`, and normal Python dependencies still come from
`https://pypi.org/simple/`. No endpoint or distribution-prefix Variable is
required; the prefix is derived from `github.repository_owner` through the
frozen repository identity. TestPyPI file readback waits for up to approximately
three minutes because upload acceptance can precede JSON/Simple Index visibility;
production PyPI keeps the shorter one-minute window.

Keep the production `PYPI_API_TOKEN` only in the official repository. Do not
copy it into the Fork or `fork-preview`.

### 2. Understand what becomes enabled

Credentials only make a Fork target available; they do not override the
selected Release Profile. PyPI resolves as follows:

| Profile `pypi` | `TEST_PYPI_API_TOKEN` | Effective decision |
| --- | --- | --- |
| `false` | absent or present | `requested=false`, `enabled=false`, `disabled` |
| `true` | absent | `requested=true`, `enabled=false`, `scope-skipped` |
| `true` | present | `requested=true`, `enabled=true`, `publish` |

Docker Hub uses the same explicit three-value contract as official publication:

| Profile `dockerhub` | Fork Docker Hub configuration | Result |
| --- | --- | --- |
| `false` | absent, partial, or complete | `requested=false`, `enabled=false`, `disabled`; values are unused |
| `true` | namespace, username, and token all configured | preflight passes and Docker Hub publication runs |
| `true` | any of the three values missing | preflight fails before Registry discovery |

`DOCKERHUB_NAMESPACE` must match `docker.io/<account-or-org>`. It has no
`{owner}` or username-derived fallback. Credentials by themselves never change
a `false` Profile switch to `true`; read the current switch values from
`release.yaml` rather than inferring them from this setup section.

When requested, GHCR, Chart OCI, and the Fork's own GitHub Release use the
Fork's GitHub identity and do not depend on the external credentials above.

### 3. Validate Profile decisions with a Fork Draft

First update and commit `version.ini`, including `UCM_VERSION` and both
supported Runtime selector lists. Use a fresh Draft sequence because published
Tag, Release, and OCI coordinates are immutable:

```bash
base="$(PYTHONPATH=.github/release python -c \
  'from pathlib import Path; from ucm_release.version_config import load; print(load(Path("version.ini"))["ucm_base_version"])')"
tag="draft/v${base}-1"  # replace 1 with an unused sequence for this base
git tag -a "${tag}" -m "Fork publication validation ${tag}"
git push origin "${tag}"
```

This run validates only the channels requested by the selected Draft Profile.
In the currently committed `release.yaml`, Draft disables both PyPI and Docker
Hub, so this example proves that neither channel writes externally; it does not
prove that TestPyPI or Docker Hub credentials are valid. Credential validation
requires a fresh Fork Tag whose selected Profile explicitly requests the target
channel.

In the `Plan Wheels, Images, and Chart` summary, compare each channel with the
selected Release Profile in `release.yaml`. Every `requested` value must match
that Profile exactly. For PyPI and Docker Hub, `enabled` and `disposition` then
follow the credential table above; the frozen `release-plan.json` records the
same decisions, the TestPyPI target, and any configured Docker Hub namespace.

After completion, verify all of the following:

- every channel with `enabled=true` was published and read back at its planned
  target;
- every channel with `enabled=false` performed no external write;
- GHCR member tags and multi-architecture indexes, when enabled, match the
  references recorded in `release-manifest.json`;
- the Fork GitHub Release, when enabled, contains backend Wheels, the Chart,
  config, and manifest, but not the empty meta Wheel or internal receipt;
- no project or image was written to production PyPI or the official Docker Hub
  namespace.

### Common setup failures

- Preflight reports incomplete Docker Hub configuration while the channel is
  requested: configure `DOCKERHUB_NAMESPACE`, `DOCKERHUB_USERNAME`, and
  `DOCKERHUB_TOKEN` together.
- Preflight rejects the Docker Hub namespace: use exactly
  `docker.io/<account-or-org>` with no repository name or trailing slash.
- TestPyPI returns an authorization error: use the complete account-scoped
  TestPyPI token, including its prefix, and verify that it can publish every
  planned `<owner>-uc-manager*` project. Production PyPI ownership does not
  grant TestPyPI ownership.
- TestPyPI reports an existing filename with different bytes: do not reuse or
  move the Tag; publish a new version.
- GitHub Release, GHCR, or Chart writes return `403`: recheck the Fork's Actions
  policy and `GITHUB_TOKEN` Read and write setting before changing external
  credentials.

## Retention and cleanup

`max_count: -1` disables retention. Finite retention considers only successful
same-type Releases carrying an exact schema-v6 manifest, never the current Tag;
old Releases without that manifest are skipped rather than guessed. When PyPI
is enabled for a finite Profile, retention is skipped with an explicit reason.

Cleanup is manifest-driven and retryable through
`cleanup-ucm-release.yml(tag=...)`. Each resource is probed before up to three
attempts with waits of 0, 5, and 15 seconds. Registry resources are removed
first, followed by the associated Actions Run, Git Tag, and all exact-tag
GitHub Releases. A failed phase blocks later phases while other resources in
that phase are still attempted. GHCR package versions carrying any non-target
Tag are refused. Re-running cleanup is idempotent because missing resources are
treated as already removed.

Every published Runtime image also contains the same Tag's example config at
`/workspace/ucm_config_example.yaml`. It is not selected automatically; callers
must explicitly point `UCM_CONFIG_FILE` at it when they want to use the example.

## Wheel contract

Internal and OCI architectures use `amd64` and `arm64`; Wheel tags use
`x86_64` and `aarch64`. The Builder repairs every backend Wheel to its planned
platform:

```text
manylinux_2_28_x86_64  # CUDA
manylinux_2_28_aarch64
manylinux_2_34_x86_64  # CANN
manylinux_2_34_aarch64
```

`external_runtime_exclude_patterns` declares the Runtime-provider boundary:
CANN uses one `/usr/local/Ascend/*` path pattern, while unresolved CUDA uses a
versioned `libcudart` SONAME pattern. The concrete direct roots and complete
version-specific closure are derived from auditwheel's ELF graph. Every direct
external library must match the boundary, and every transitive external library
must be reachable from a discovered root. UCM-owned libraries such as
`libmetrics.so` must already be present in the Wheel; an unplanned auditwheel
graft fails the build. Each backend artifact contains:

- the Wheel;
- `wheel-result.json` schema 5;
- `auditwheel-repair.txt`;
- `auditwheel-show.txt`.

The result verifies the filename against WHEEL metadata, records the repair
target and exclude patterns, compatible ABI floor, GLIBC symbols/floor,
dynamically discovered external roots and closure, the unresolved non-root
libraries that Runtime validation may defer, and the audit report digest.
Deferred libraries are derived from unresolved non-root nodes in the verified
provider closure; they are not maintained as a SONAME list. A direct external
outside the provider boundary, a UCM-owned external, or an external unreachable
from the discovered roots still fails. A separate `py3-none-any` `uc-manager`
meta Wheel maps the release's dynamic extras to exact backend distribution
versions. It is published only through PyPI or TestPyPI and is never a GitHub
Release asset.
The staged Release validates the standalone report again before uploading the
Wheel. Draft notes show filenames and architecture labels without embedding
GitHub's rotating `untagged-*` asset URLs; downloads remain in the Release
Assets section. Published Releases keep direct asset links.

## PR behavior

An ordinary PR selects one latest Stable vLLM-Ascend A2 Ubuntu runtime, falling
back to RC and release-nightly. Its actual architecture members determine the
Wheel and image tasks. Explicit `/ucm-build image <repository:tag>` accepts one
runtime reference, inspects its OCI config, pulls only metadata-incomplete
members for fallback probing, and publishes PR-scoped GHCR tags. `/ucm-build
all` retains the complete formal matrix behavior.

## Artifact names

Artifacts are scoped by run and overwritten by failed-job reruns:

- `ucm-runtime-candidates-run-<run>`;
- `ucm-runtime-inspection-run-<run>`;
- `ucm-runtime-selection-run-<run>`;
- `ucm-builder-catalog-run-<run>`;
- `ucm-release-plan-run-<run>`;
- `ucm-wheel-<wheel-id>-run-<run>`;
- `ucm-chart-run-<run>`;
- `ucm-image-<image-id>-run-<run>` for PR Robot trust-boundary handoff only;
- image member/index receipt artifacts;
- `ucm-release-stage-run-<run>` containing internal `release-state.json`.

When image publication is enabled, trusted Tag builds keep the verified OCI
archive on the native build runner, copy it directly to GHCR, and upload only
the small member receipt. They do not retain multi-gigabyte OCI archives as
Actions artifacts. PR Robot builds keep the separate read-only build to trusted
publisher artifact boundary.

## Verification

```bash
python -m pytest -q .github/release/tests
ruff check .github/release scripts/materialize_version.py ucm/__init__.py
black --check .github/release scripts/materialize_version.py ucm/__init__.py
pre-commit run actionlint --all-files --hook-stage manual
git diff --check
```

Local checks are preflight only. Forward-compatible matrix and staged Release
acceptance must be demonstrated by GitHub Actions on `feature/cicd_v5`, with
run URL/SHA/job/artifact evidence and Registry/Release readback.
