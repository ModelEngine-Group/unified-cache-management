# UCM Registry-driven release automation

The active pipeline builds UCM Wheels and the Helm Chart from actual published
runtime images. Stable, Prerelease, Draft, and Nightly Profiles independently
select the five publication channels and retention limits. The pipeline never
reads vLLM, vLLM-Ascend or SGLang source branches to decide versions or Builder
capabilities.


## Module ownership

`version_config` owns version parsing, Tag classification and materialization;
`policy` loads the current release/platform configuration and resolves publication
channels. `registry` reads OCI facts, `runtime` owns probe contracts and image
coordinates, `upstream` selects Runtime versions, and `builders` resolves raw
Builders and synchronizes checked mirrors. `plan` only combines these results
into build tasks. `wheel` and `meta` prepare and record their own artifacts.

`release` aggregates publication state and renders release notes. `manifest`
owns the public Schema 10 contract shared with documentation and cleanup, while
retaining Schema 9 compatibility. `cleanup` supplies CLI parsing and the
top-level cleanup flow. The package import has no CLI
side effects; `python -m ucm_release` dispatches through `__main__`.

Cleanup responsibilities have explicit boundaries:

| Module | Responsibility |
| --- | --- |
| `cleanup` | Own the common CLI, deletion phases, retries, activity checks and confirmed absence. |
| `cleanup_records` | Decode supported records and manifests, project exact resources, and select excess versions. |
| `cleanup_inventory` | Discover versions, resolve trusted evidence without writes, and explicitly save attempt snapshots or recovered plans. |
| `cleanup_remote` | Own GitHub/registry access, transport errors, resource mutation and readback. |

Wire-format readers preserve the two existing cleanup formats. Both produce
`CleanupTarget` values with explicit Actions and Release recovery holders;
only `cleanup.execute_cleanup()` owns destructive phase ordering. Nightly and
Draft use the same discovery, selection, resolution and execution operations;
release type selects the profile and ordering rule, rather than a separate
cleanup implementation.

Use `plan create/select/retag-pr`, `wheel prepare-source/record-result`, and
`meta materialize-source/record-result`. The Catalog planner, old Wheel authority
commands and parameter aliases are not supported. Existing Sphinx documentation,
manual `scripts/build_*.sh` packaging and root Dockerfiles remain independent.

## Maintained policy

The human-maintained release authorities are:

- `version.ini`: the UCM base version plus the exact supported vLLM,
  vLLM-Ascend and SGLang Runtime selectors for this source version;
- `release.yaml`: runtime repositories, runners, fixed publication addresses,
  four fully expanded Release Profiles, retention, and Chart smoke inputs;
- `platforms.yaml`: raw Builder registries, excluded variants, Builder checks,
  and supported or blocked UCM backends;
- Root `pyproject.toml`: package runtime dependencies and build-system requirements;
- Root `requirements-build.txt`: exact Wheel Builder tool versions, validated
  against `build-system.requires` before creating the release plan;
- `.github/release/requirements.txt`: release script dependencies. Workflows
  install this file; shared build tool pins come from `requirements-build.txt`.

The release plan reads runtime dependencies directly from `project.dependencies`.
The built Wheel metadata must match those declarations before publication.

Each product selector is a canonical `X.Y` Minor range, an exact `X.Y.Z`
Patch range, or a fixed `@Tag`. Every range is resolved independently from
parsed Registry tags:
vLLM and SGLang choose `stable` before `rc`; vLLM-Ascend also allows a
`nightly` fallback. The selector then chooses the highest complete version in
that channel and expands every legal variant of only that version and channel.
A newer RC never displaces a stable version in the same range, and missing
variants are never backfilled from another version or channel. SGLang
`main-cann*` and `nightly-*` tags are not automatic release candidates.

An explicit `@Tag` such as `@nightly-releases-v0.30.0` selects that exact published
Tag from the product's configured upstream repository. Write only the Tag after
`@`, not a repository address or digest. Any valid OCI Tag is accepted when it
exists in that repository: names such as `nightly`, `latest`, or `my-build`
do not have to encode a version. Matching is exact; wildcard selectors are not
supported. A fixed candidate records the Tag itself as `version` and uses
`channel: pinned`. Its hardware and Runtime
capabilities come from image inspection and, when needed, native probing; these
facts must identify a supported platform and a compatible Builder. Name-based
version, channel, and variant candidate filters do not discard explicit pins.
The pin is not expanded to other Tags, but its supported CPU architectures are
still built. The old `version@Tag` syntax is rejected; for example, replace
`0.30@v0.30.0-a3` with `@v0.30.0-a3`.

Ranges and fixed Tags can be mixed in one comma-separated list without spaces.
Overlapping ranges such as `0.30,0.30.0`, repeated fixed Tags, and a range plus a
pin that resolve to the same Tag are errors. A configured range with no match,
a missing explicit Tag, or selectors that produce no publishable Runtime also
fail the release. All four Release Profiles consume the same selectors.
Automatic ranges retain the configured variant exclusions and do not fall back
when a selected backend is blocked. Fixed Tags bypass these name-based candidate
filters, while platform support and Builder compatibility are checked against
their inspected capabilities. `--latest-per-product` narrows only the numeric
ranges for the named products; every explicitly pinned Tag is retained.

These selector rules apply to `UCM_SUPPORTED_VLLM_VERSIONS`,
`UCM_SUPPORTED_VLLM_ASCEND_VERSIONS`, and `UCM_SUPPORTED_SGLANG_VERSIONS`.
Deleting, commenting out, or assigning an empty value to one of these variables
disables that product's automatic Runtime selection and build tasks. Disabled
products are skipped before querying their Runtime Registry. Shared Wheels and
Builders required by another enabled product are retained. At least one product
must have valid selectors; disabling all three fails configuration parsing.
Canonical output and version materialization omit disabled product assignments.

Upstream candidate channels are independent of UCM's Stable, Prerelease, Draft,
and Nightly Profiles: the Profile controls which artifacts are built and
published, without widening the candidate set. `UCM_VERSION` is a canonical
PEP 440 package version with an `X.Y.Z` release tuple. Builds materialize the
complete version selected by the Release Tag, including a named Draft's local
identity as described below.

For example (fixed Tags must already exist in the configured repositories):

```ini
UCM_SUPPORTED_VLLM_VERSIONS=0.30,@v0.31.0
UCM_SUPPORTED_VLLM_ASCEND_VERSIONS=@nightly-releases-v0.30.0
UCM_SUPPORTED_SGLANG_VERSIONS=
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
image layers. Only a member with missing or conflicting Python, CUDA/CANN, or
SOC metadata is pulled and probed on its native architecture. Runtime glibc does not gate planning:
the Wheel's actual GLIBC floor comes from `auditwheel`, then the final
install-only Runtime image verifies its glibc floor, installs the Wheel, and
runs `import ucm` before publication. Every runtime member resolves exactly one
Wheel ID. Zero or ambiguous matches fail instead of guessing.

Ascend Builder selection first filters SoC, Python ABI, native architecture and
the configured manylinux policy. An exact complete CANN version wins. Only a
bare Runtime version such as `9.2.0` may fall back to the highest parseable
suffixed version with the same `X.Y.Z`, for example `9.2.0-beta.2`. Version
ordering uses `packaging.Version`, not Registry order or creation time. A Runtime
that explicitly reports a suffix requires that exact version. A selected
Builder's verification or build failure never triggers an older-version retry.

`accelerator_runtime` continues to describe the target Runtime requirement and
therefore the Wheel's existing identity. The complete source Builder version is
owned by `source_image`, with its native member pinned by `source_image_digest`;
`checks.accelerator_runtime` is derived from that source for actual Toolkit
verification. Ascend PRs select the current upstream source before reusing a
checked GHCR mirror with the same source, digest and recipe revision.

The Native Wheel Gate installs and loads each suffix-fallback Wheel in every
matching Runtime member before passing. It reuses the release installation
validator without publishing images. These checks cover glibc, dependencies and
native loading; they do not exercise NPU operations or FFTS transfers.

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

| Git Tag | Selected Profile | Wheel version | Chart version | Image version | GitHub Release mode when enabled |
| --- | --- | --- | --- | --- | --- |
| `vX.Y.Z` | Stable | `X.Y.Z` | `X.Y.Z` | `X.Y.Z` | public Release |
| `vX.Y.ZrcN` | Prerelease | `X.Y.ZrcN` | `X.Y.Z-rc.N` | `X.Y.ZrcN` | public prerelease |
| `draft/v0.9.0` | Draft | `0.9.0.dev0` | `0.9.0-draft.0` | `0.9.0.dev0` | Draft |
| `draft/v0.9.0-1` | Draft | `0.9.0.dev1` | `0.9.0-draft.1` | `0.9.0.dev1` | Draft |
| `draft/v0.9.0-test` | Draft | `0.9.0.dev0+draft.test` | `0.9.0-draft.test` | `0.9.0.dev0.draft.test` | Draft |
| `draft/v0.9.0-1-test` | Draft | `0.9.0.dev0+draft.1.test` | `0.9.0-draft.1.test` | `0.9.0.dev0.draft.1.test` | Draft |
| `nightly/vX.Y.Z-YYYYMMDD-N` | Nightly | `X.Y.Z.devYYYYMMDDNNN` | `X.Y.Z-nightly.YYYYMMDD.N` | `X.Y.Z.devYYYYMMDDNNN` | public prerelease after success |

An omitted Draft suffix still means `.dev0`; a numeric suffix must be a positive
integer without leading zeros. A named suffix contains lowercase ASCII letters,
digits and single separating hyphens, with at least one letter. Purely numeric
segments may not have leading zeros: `test-0` and `1-test` are valid, while
`test-01`, `-0`, `-01`, uppercase letters, periods, underscores, empty segments
and leading/trailing hyphens are rejected. Inputs are not silently normalized.

The complete named suffix identifies the build. Wheel versions preserve it in
the `+draft.` local namespace; Chart `appVersion`, Meta/Toolkit versions and exact
dependencies retain the complete Wheel version. Only these generated named Draft
local versions are allowed for new releases. Other local forms are not enabled.
The namespace avoids the known historical `+g<commit>` format, without claiming
to exclude every manually created historical artifact coordinate.

Automatic release image tags use `<upstream-tag>-ucm-<UCM-image-version>` for
every product, with the existing repository-owner prefix on Forks. The UCM image
version is the OCI-compatible package version selected by the Git Tag, such as
`0.9.0rc2` or `0.9.0.dev1`. A local `+` is replaced with `.`, so a named Draft
uses `0.9.0.dev0.draft.test`. This keeps the label distinct from the
multi-architecture member suffixes `-amd64` and
`-arm64`; their index and single-architecture images use the bare tag. SGLang
does not add a creation-time suffix. The final tag, including owner prefix and
architecture suffix, must fit the OCI 128-character limit; it is never truncated.

The tagged source owns the package model. Versions must be canonical and local
versions are restricted to the named Draft mapping above. Before opening or
reusing a Release, the reusable core requires
the Tag's complete `X.Y.Z` base to match `UCM_VERSION`, then reclassifies the
Tag and checks its release type, visibility, prerelease flag, Chart/image
versions, target commit, requested source SHA, and checked-out commit. A Draft
Tag always remains Draft. Exact Releases API lookup requires one Release for the
Tag; duplicate Release records fail closed.

Named Drafts cannot request PyPI or TestPyPI publication because those indexes
do not accept local versions. Preflight checks the Draft profile's `pypi`
request before building or creating a Release, even when a Fork has no token
and would otherwise skip that channel. It does not silently turn the channel
off; the upload layer also keeps its existing local-version rejection.
See the [PyPA local-version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/#local-version-identifiers).

To install a named Draft from downloaded/local Wheels, collect the matching
Meta, backend and Toolkit Wheels plus dependencies in `wheelhouse`, then use
the complete version. For example, with the official distribution names:

```bash
python -m pip install --no-index --find-links ./wheelhouse \
  'uc-manager[cu130]==0.9.0.dev0+draft.test'
```

On a Fork, use its existing owner-prefixed distribution name instead, for
example `supermarioyl-uc-manager[cu130]==0.9.0.dev0+draft.test`. Named labels do
not imply creation order. Draft retention keeps the existing Release ordering;
new named Draft cleanup records must match the complete version, while old
records retain historical local-version read compatibility.

At 02:00 Asia/Shanghai (`18:00` UTC),
`release-nightly.yml` reads the `X.Y.Z`
base from `version.ini` and creates the next dated Nightly Tag from `develop`.
An incomplete same-SHA Nightly Tag is reused; an existing Tag is never moved.
Scheduled runs are restricted to the official repository. Maintainers can also
dispatch the workflow on a Fork's `develop` branch to validate the same Nightly
path; publication scope is derived from the repository and stays `fork` there.
Because a `GITHUB_TOKEN` Tag creation does not
recursively trigger another workflow, the same scheduled Run calls the common
`release-ucm.yml` reusable core directly. Manual `nightly/*` Tag pushes use the
same core through `release-tag.yml`.

Nightly publishes backend Wheels to GitHub Release, followed by the cleanup
manifest. It builds directly from digest-pinned upstream Builders without
syncing or pushing Builder images to GHCR. Runtime image builds, Chart packaging,
and PyPI, GHCR, Docker Hub, and Chart OCI publication are disabled. Native Wheel
builds and matching-Runtime installation checks still run before publication.
These decisions follow the selected Profile's channel switches: disabling
`ghcr` bypasses Builder synchronization, and disabling `chart_oci` skips Chart
packaging and its GitHub Release asset. The plan retains Chart metadata for
every release type.

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
4. backend Wheels (plus the example config and Chart when `chart_oci` is enabled)
   are uploaded and the state
   is `artifacts-ready` while any enabled channel remains; the empty meta Wheel
   remains an internal Actions artifact;
5. image members/indexes, PyPI, and Chart OCI complete and are read back;
6. the final state becomes `complete`, `images-failed`, or
   `publication-failed`.

Chart packaging runs `python -m ucm_release chart prepare` against the release
plan. It copies the source Chart, fills `images.image` with the latest stable
vLLM CUDA image, and adds commented alternatives for every runtime family and
architecture beside that setting. The upstream default CUDA tag is preferred;
otherwise the highest CUDA and OS versions break ties. Both the default and
alternatives prefer Docker Hub when enabled, then GHCR, using the same address
mapping as publication. The source values and their comments are preserved.
Without stable CUDA candidates the default remains empty; PR and image-disabled
plans leave the source values unchanged. Packaging reads the values back from
the archive and renders the CUDA profile without an image override. Planned
addresses become usable only after their existing publication checks succeed.

`release-state.json` remains the rich internal staging file in the
`ucm-release-stage-run-<run>` Actions artifact. Only after all enabled channels
succeed, a public `release-manifest.json` Schema 10 is uploaded and read back.
The pure `ucm_release.manifest` module generates and validates this contract
for publication, cleanup and documentation. It records Python package identity,
extras and published index URLs, backend Wheels, Runtime image families, Chart
and exact GitHub Release assets. When Chart publication is disabled, `chart` is
`null`; publication, cleanup and documentation omit Chart operations while
retaining the published Wheels. Meta Wheels and publication receipts remain
internal; neither is required as a public Release attachment. Enumeration skips
unsupported manifests; exact unsupported Tag operations fail without migration.

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
install validation resolves the requested extra through pip, then checks the
selected UCM filenames and SHA256 digests against the receipt. Fork installs
use TestPyPI for the UCM packages and enable production PyPI for ordinary
dependencies. GitHub
Release notes show version-pinned installation commands only after a complete
Python-index publication receipt is available. Each Runtime capability then
shows one meta-package extra in the Wheel column. Official PyPI commands use the
default index; Fork commands include both index URLs. Package
names, versions, and extras come from the receipt. Without a complete receipt,
the Wheel column keeps its architecture-specific GitHub Release links.

Release notes preserve the existing body and append pipeline status and artifact
tables in a section delimited by `<!-- ucm-release:begin -->` and
`<!-- ucm-release:end -->`. Later stages, failure reports, and reruns replace only
that section, preserving manual notes before and after it. Existing text without
these markers is kept as-is; it is never inferred to be disposable pipeline output.
Keep manually maintained content outside the markers.

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
production PyPI uses the same bounded readback window.

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

Named Draft local versions are an exception: `pypi: true` is a configuration
error regardless of token availability. Use a numeric Draft or a normal
prerelease when Python-index publication is required.

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

First update and commit `version.ini`, including `UCM_VERSION` and at least one
supported Runtime selector list. Use a fresh Draft sequence because published
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

`max_count: -1` disables automatic retention. Nightly counts every unique
Nightly Tag, including successful Releases, failed or cancelled Drafts, legacy
manifests and orphaned Releases. It keeps the newest configured number by the
Tag's date and numeric sequence, using the base version as the tie-breaker.
Duplicate Releases and run attempts count once. Existing objects determine the
quota; a current Tag that was never created does not reserve a slot. Active,
queued and waiting runs are protected even when their version is over quota,
and the report explicitly records the deferral until a terminal sweep.
Ordinary Draft retention also counts successful and failed versions without
requiring a public manifest. Tags count once, in GitHub Release creation order
rather than the tagged commit's date. Current and active publications are
protected without substituting newer deletion candidates. Immutable PyPI
publication still disables retention for its Profile.

All release types use `record-targets` to persist small
`release-cleanup-<run>-<attempt>-<opened|planned>.json` snapshots with kind
`ucm-release-targets` and schema version 1. Each snapshot binds the repository,
Tag, source SHA, source run and exact release-only registry targets; its filename
identifies the run attempt and publication stage. Opening is independent of
Runtime discovery, and planning records all enabled targets before publishers
can start. Snapshots are immutable: reruns preserve earlier attempts and cleanup
combines their resources. Python writes the snapshot to `--output-dir` before
uploading it to the exact Release and verifying the readback. An `always()`
artifact upload saves the local copy even if the remote write fails.

Nightly preparation and the Nightly Tag caller use the local `record` command
before a Release exists. Their artifact preserves the initial Tag/SHA binding
when preparation fails or is cancelled. The independent `retain-releases` job
runs after success or failure for non-Nightly publications; Nightly uses the
separate lifecycle sweep described below.

Historical failures can recover an exact repository-owned Actions plan and
persist it before deleting its run. A plan-less early failure is recoverable
when all-attempt job results prove every release publisher was skipped.
Missing, expired or conflicting evidence otherwise reports a cleanup failure
while the version remains in the count. Non-Nightly `--dry-run` resolves the
same read-only target as actual cleanup, including pending historical evidence.
Actual execution explicitly saves and reads back that evidence before deletion;
the preview does not upload recovery assets or delete resources.

Builders, base runtimes and PyPI packages are never cleanup targets. Existing
Nightly `release-cleanup.json` records and `ucm-nightly-cleanup-*` artifacts remain
readable alongside the new snapshots and `ucm-release-cleanup-*` artifacts.
The public Schema 10 manifest continues to require complete publication. Schema 9
manifests remain readable, and the internal cleanup reader also supports proven
legacy Schema 6 records without weakening the public contract.

`nightly-cleanup.yml` runs after trusted Nightly or Nightly-Tag workflows finish,
including failure and cancellation. It uses default-branch code and shares a
repository lifecycle concurrency group with both publishing entrypoints and
manual Nightly cleanup, with `queue: max` and `cancel-in-progress: false`. The
common release core retains its separate Tag-level group. Non-Nightly cleanup
remains available through `cleanup-ucm-release.yml(tag=...)`.

Cleanup resolves immutable identities before mutation and uses recoverable
phases: registry, Actions run, Tag, then Releases holding the recovery inventory.
If no live Release attachment covers the full inventory, the artifact-holding
run is also retained until the last phase. Each resource is probed and deleted
with up to three attempts, waiting 0, 5 and 15 seconds, followed by an exact
ID/digest readback. Only confirmed absence is a successful deletion. GHCR errors,
including denied access, a last tagged version, and versions with non-target
Tags, are recorded as nonfatal skips after the applicable retries. Cleanup
continues with other registry targets, non-recovery Actions runs, the Tag and
non-recovery Releases. Releases or runs holding the recovery inventory remain
available for retry. The job succeeds if GHCR skips are its only errors; it never
falls back to deleting an entire package. Other failed phases preserve later
recovery holders and fail the job; independent resources and versions continue
to be attempted. Missing or conflicting publication ownership remains a blocker.

Preview the configured quota without deleting:

```bash
python .github/release/ucm_release/cleanup.py retention \
  --repository ModelEngine-Group/unified-cache-management \
  --release-type nightly --max-count 7 --pypi-enabled false \
  --dry-run --report out/cleanup/report.json
```

A reviewed offline inventory can be supplied with
`--inventory /path/to/reviewed-inventory.json`. It restores exact resource
ownership for older failed Drafts and already deleted GitHub objects; these
registry-only leftovers are cleaned separately from the live quota. Keep this
recovery evidence until registry cleanup is confirmed. No references are guessed
from a prefix.
The JSON report distinguishes kept, would-delete, deleted, partial, deferred and
blocked versions. A `partial` result lists each GHCR skip with its reference,
attempt count and final error; the Actions summary also lists these warnings.
Dry-run records GHCR probe errors as skips without remote writes. Non-GHCR API
errors or an unresolved publication keep the result failed. Retained recovery
objects remain in the quota. Manual `tag` cleanup can retry them even after the
Tag is gone; retention retries them when they are selected as excess versions.
Reruns are idempotent when resources have already been removed.

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

The ordinary PR Native Wheel Gate reads the configured Runtime selectors and
builds their amd64 Wheel matrix. It narrows SGLang's numeric ranges to the newest
range while retaining every fixed `@Tag`; release builds use all configured
ranges and supported architectures. Explicit `/ucm-build image <repository:tag>`
accepts one runtime reference, inspects its OCI config, pulls only metadata-incomplete
members for fallback probing, and publishes PR-scoped GHCR tags. Explicit image
requests use the `pinned` channel independently of automatic release candidate
selection, so a SGLang `main-cann*` image can still be requested this way.
`/ucm-build all` retains the complete formal matrix behavior.

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

Candidates or selection artifacts containing SGLang `main` images are no longer
accepted. Regenerate discovery and selection with the updated code before
planning; rerunning only the failed Plan job reuses the unsupported artifacts.

When image publication is enabled, trusted Tag builds keep the verified OCI
archive on the native build runner, copy it directly to GHCR, and upload only
the small member receipt. They do not retain multi-gigabyte OCI archives as
Actions artifacts. PR Robot builds keep the separate read-only build to trusted
publisher artifact boundary.

## Verification

The release suite checks planning, artifact validation, publication, and recovery.
Runtime engine patch routing belongs in `test/suites/Unit`, outside this suite.
Prefer observable behavior over copied dependency pins, internal function names,
or shell command spelling. Workflow checks should protect publication ordering,
resource ownership, and failure handling.

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

## Toolkit and complete release delivery

The same release plan now includes a portable `ucm-toolkit` Wheel and the exact
`toolkit` extra on the meta package. Forks use the existing owner prefix on both
packages and publish to TestPyPI. Toolkit sources and runtime resources are
bundled; dev-sandbox compilation remains an explicit user action.

`_build-toolkit.yml` is shared by PR checks and releases. Index verification
checks standalone Toolkit, the Toolkit-only meta extra and each backend combined
with Toolkit. Pip reports must identify files from the publication receipt.

Stable and prerelease runs require `RTD_PROJECT_EN`, `RTD_PROJECT_ZH` and
`RTD_API_TOKEN` before building. The projects must use the release repository,
English and Simplified Chinese respectively, and the Chinese project must be a
translation of the English project. For Fork validation, both projects' default
branch is `feature/docs_v2`. The workflow waits for RC and Latest
builds, requires the RC build to match the release source SHA, and verifies public
version roots and manifests. Latest keeps the project default-branch source.

The published Chart is downloaded from GitHub Release and OCI and compared with
the original package digest. Every packaged default/candidate image is checked
against publication receipts and Registry architecture metadata. Final acceptance
is stored only as `ucm-release-acceptance-run-<run_id>` in Actions artifacts. A docs
or delivery-check failure can be retried without rebuilding published packages.
