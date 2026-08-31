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
`UCM_SUPPORTED_VLLM_ASCEND_VERSIONS`. `VLLM_UC_VERSION` remains a canonical PEP
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

| Git Tag | Wheel version | Chart version | GitHub Release |
| --- | --- | --- | --- |
| `vX.Y.Z` | `X.Y.Z` | `X.Y.Z` | public Release |
| `vX.Y.ZrcN` | `X.Y.ZrcN` | `X.Y.Z-rc.N` | public prerelease |
| `draft/vX.Y.Z` | `X.Y.Z.dev0` | `X.Y.Z-draft.0` | Draft |
| `draft/vX.Y.Z-N` | `X.Y.Z.devN` | `X.Y.Z-draft.N` | Draft |
| `nightly/vX.Y.Z-YYYYMMDD-N` | `X.Y.Z.devYYYYMMDDNNN` | `X.Y.Z-nightly.YYYYMMDD.N` | public prerelease after success |

The tagged source owns the package model, and public versions must be canonical
and non-local. Before opening or reusing a Release, the reusable core requires
the Tag's complete `X.Y.Z` base to match `VLLM_UC_VERSION`, then reclassifies the
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

Official and fork `v*` Tags invoke that same Release Core. The two mutually
exclusive callers isolate production secrets: the official caller inherits
production credentials, while the fork uses only its `fork-preview` environment.
A fork still publishes its own GitHub Release, GHCR images, and Chart OCI
artifacts.

Publication then updates the same Release in stages:

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

If image publication is disabled, Tag Releases stop after Wheels, the example
config, and Chart publication. If requested image publication fails, those
artifacts remain usable and the public Release is marked `images-failed`. OCI
archives are not uploaded to GitHub Release. PyPI and Docker Hub follow the
selected Release Profile. Official releases use only production targets. Forks
use TestPyPI when `TEST_PYPI_API_TOKEN` exists, and a custom Docker Hub namespace
when both Docker Hub credentials exist; absent Fork credentials remain
`scope-skipped`, while a Profile-disabled channel stays disabled.
Missing backend Wheels are uploaded and read back first, the meta Wheel is
uploaded last, and exact extras metadata plus all filenames, versions, and
SHA256 digests are persisted in `pypi-receipt.json`.

## Fork preview setup

Fork publication is opt-in by credential presence. It never reads the official
PyPI credential and it does not change the publication targets of the canonical
`ModelEngine-Group/unified-cache-management` repository.

### 1. Create the GitHub Environment

In the fork repository, open **Settings -> Environments** and create an
environment named exactly `fork-preview`. Store the Fork channel Secrets in
that environment when possible. A repository-level Actions Secret is also
supported for `TEST_PYPI_API_TOKEN`: the Tag caller forwards only that named
credential and never forwards `PYPI_API_TOKEN` to the Fork Release call. Keep
the Docker Hub credential pair in `fork-preview`.

Also open **Settings -> Actions -> General** and make sure Actions is enabled. If
repository or organization policy restricts `GITHUB_TOKEN` writes, allow
**Read and write permissions**. The workflow itself requests scoped `contents`
and `packages` writes for the Fork's Release, GHCR, and Chart OCI.
See [GitHub's Actions settings guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository).

| Name | GitHub kind | Required | Value |
| --- | --- | --- | --- |
| `TEST_PYPI_API_TOKEN` | Secret (environment recommended; repository supported) | TestPyPI only | A TestPyPI API token with permission to publish the planned `uc-manager*` projects. TestPyPI accounts and tokens are separate from production PyPI; follow the [TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/). |
| `DOCKERHUB_USERNAME` | Environment Secret | Docker Hub only | The Docker ID whose token can write to the selected personal or organization namespace. |
| `DOCKERHUB_TOKEN` | Environment Secret | Docker Hub only | A Docker Hub personal access token with Read & Write permission; also grant Delete when retention or `cleanup-ucm-release.yml` must remove Docker Hub tags. Do not use the account password; follow the [Docker access-token guide](https://docs.docker.com/security/access-tokens/). |
| `FORK_DOCKERHUB_NAMESPACE` | Variable | Optional | Full namespace such as `docker.io/my-org`, without a trailing slash. If omitted, the pipeline uses `docker.io/<DOCKERHUB_USERNAME>`. |

The equivalent GitHub CLI commands are below. Each `gh secret set` command
prompts for the value without writing it to the repository; omit the channels
you do not want to enable:

```bash
fork=OWNER/unified-cache-management
gh secret set TEST_PYPI_API_TOKEN --env fork-preview --repo "${fork}"
gh secret set DOCKERHUB_USERNAME --env fork-preview --repo "${fork}"
gh secret set DOCKERHUB_TOKEN --env fork-preview --repo "${fork}"
gh variable set FORK_DOCKERHUB_NAMESPACE --env fork-preview \
  --repo "${fork}" --body docker.io/my-org
```

The TestPyPI endpoints are fixed by policy: upload uses
`https://test.pypi.org/legacy/`, package installation uses
`https://test.pypi.org/simple/`, and normal Python dependencies still come from
`https://pypi.org/simple/`. No endpoint Variable is required.

Keep the production `PYPI_API_TOKEN` only in the official
`release-production` environment. Do not copy it into `fork-preview`.

Docker Hub credentials are a pair. Configuring only the username or only the
token, or setting `FORK_DOCKERHUB_NAMESPACE` without both credentials, fails the
Release preflight before Registry discovery or GitHub Release creation. The
configured user or token must have write access when the namespace names an
organization.

### 2. Understand what becomes enabled

Credentials only make a Fork target available; the selected Release Profile
must also request that channel:

| `fork-preview` configuration | Stable / Prerelease / Draft | Nightly |
| --- | --- | --- |
| No external credentials | TestPyPI and Docker Hub are `scope-skipped` | Both remain disabled |
| `TEST_PYPI_API_TOKEN` only | Publish and read back TestPyPI | Disabled by Profile |
| Docker Hub username + token | Publish and read back Docker Hub | Disabled by Profile |
| All three Secrets | Publish and read back both targets | Both remain disabled |

GHCR, Chart OCI, and the Fork's own GitHub Release continue to use the Fork's
GitHub identity in every combination. `FORK_DOCKERHUB_NAMESPACE` changes only
the Docker Hub destination; image repository basenames and tags still come from
the frozen Release Plan.

### 3. Run one Fork Draft validation

First update and commit `version.ini`, including `VLLM_UC_VERSION` and both
supported Runtime selector lists. Use a fresh Draft sequence because Python
index filenames are immutable and the cleanup workflow does not delete
TestPyPI releases:

```bash
base="$(PYTHONPATH=.github/release python -c \
  'from pathlib import Path; from ucm_release.version_config import load; print(load(Path("version.ini"))["ucm_base_version"])')"
tag="draft/v${base}-1"  # replace 1 with an unused sequence for this base
git tag -a "${tag}" -m "Fork publication validation ${tag}"
git push origin "${tag}"
```

In the `Plan Wheels, Images, and Chart` summary, confirm the requested channels
show the expected effective decisions:

```text
PyPI: requested=true enabled=true disposition=publish scope=fork
Docker Hub: requested=true enabled=true disposition=publish scope=fork
```

If only one credential set was configured, the other channel should remain
`scope-skipped`. The frozen `release-plan.json` must also record
`.publish.pypi.target == "testpypi"` and the configured Docker Hub namespace.
After completion, verify all of the following:

- each backend project and the final empty `uc-manager` meta package are visible
  on TestPyPI, and the fresh-environment extra installation job passed;
- Docker Hub member tags and multi-architecture indexes exist under the planned
  namespace and match the digests recorded in `release-manifest.json`;
- the Fork GitHub Release still contains backend Wheels, the Chart, config,
  receipts, and manifest, but not the empty meta Wheel;
- no project or image was written to production PyPI or the official Docker Hub
  namespace.

### Common setup failures

- Preflight reports an incomplete Docker Hub credential pair: add or remove both
  `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` together.
- Preflight rejects the Docker Hub namespace: use exactly
  `docker.io/<account-or-org>` with no repository name or trailing slash.
- TestPyPI returns an authorization error: use the complete TestPyPI token,
  including its prefix, and verify that it can publish every planned
  `uc-manager*` project. Production PyPI ownership does not grant TestPyPI
  ownership.
- TestPyPI reports an existing filename with different bytes: do not reuse or
  move the Tag; create a new Draft sequence so the derived `.devN` version is
  new.
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
