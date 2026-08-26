# UCM Registry-driven release automation

The active pipeline builds UCM Wheels and the Helm Chart from actual published
runtime images. Stable, Prerelease, Draft, and Nightly Profiles independently
select the five publication channels and retention limits. The pipeline never
reads vLLM or vLLM-Ascend source branches to decide versions or Builder
capabilities.

## Maintained policy

The human-maintained release authorities are:

- `release.yaml`: runtime repositories, inclusive version windows, runners,
  fixed publication addresses, four fully expanded Release Profiles, and Chart
  smoke inputs;
- `platforms.yaml`: raw Builder registries, excluded variants, Builder checks,
  and supported or blocked UCM backends;
- `requirements/wheel-build.txt` and `requirements/wheel-runtime.txt`: exact
  Python dependencies.

`minimum_version` is required per product. `maximum_version` is optional and,
when present, inclusive. Registry Tags are grouped by major/minor. Each line
selects its highest Stable version, otherwise its highest formal RC, otherwise
its release-nightly version. All real variants of that selected version remain
eligible; 310P is filtered and A5 is reported as blocked. A Profile
`max_minor_versions` of `-1` keeps all actual minor groups; a positive value
keeps the first N existing minor groups in ascending order without filling gaps.

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

Every existing or newly built mirror is re-probed on its native architecture,
its OCI labels are compared with the desired catalog, and its final manifest
digest is recorded. Wheel builds consume `repository@digest`, never a mutable
Builder Tag.

## Tag and Release modes

The workflow accepts:

| Git Tag | Wheel version | Chart version | GitHub Release |
| --- | --- | --- | --- |
| `vX.Y.Z` | `X.Y.Z` | `X.Y.Z` | public Release |
| `vX.Y.ZrcN` | `X.Y.ZrcN` | `X.Y.Z-rc.N` | public prerelease |
| `draft/vX.Y.Z` | `X.Y.Z.dev0` | `X.Y.Z-draft.0` | Draft |
| `draft/vX.Y.Z-N` | `X.Y.Z.devN` | `X.Y.Z-draft.N` | Draft |
| `nightly/vX.Y.Z-YYYYMMDD-N` | `X.Y.Z.devYYYYMMDDNNN` | `X.Y.Z-nightly.YYYYMMDD.N` | public prerelease after success |

For a formal `v*` Tag, the Release is created or reused and made public
immediately. A manually pre-opened public Release is valid. A Draft Tag always
remains Draft. Exact Releases API lookup requires one Release for the Tag;
duplicate Release records fail closed.

At 02:00 Asia/Shanghai (`18:00` UTC), `release-nightly.yml` reads the highest
strict `vX.Y.Z` Stable Tag, advances its patch, and creates the next dated
Nightly Tag from `develop`. An incomplete same-SHA Nightly Tag is reused; an
existing Tag is never moved. Because a `GITHUB_TOKEN` Tag creation does not
recursively trigger another workflow, the same scheduled Run calls the common
`release-ucm.yml` reusable core directly. Manual `nightly/*` Tag pushes use the
same core through `release-tag.yml`.

Publication then updates the same Release in stages:

1. `release-open`: no artifacts are required yet;
2. all Wheels, the example config, and the Chart are uploaded and the state is
   `artifacts-ready` while any enabled channel remains;
3. image members/indexes, PyPI, and Chart OCI complete and are read back;
4. the final state becomes `complete`, `images-failed`, or
   `publication-failed`.

`release-state.json` remains the rich internal staging file in the
`ucm-release-stage-run-<run>` Actions artifact. Only after all enabled channels
succeed, a compact public `release-manifest.json` schema 6 is uploaded and read
back. It records only the Tag/type/Actions Run, Chart OCI reference, Runtime
member/index references, and GitHub Release asset names needed for cleanup.

If image publication is disabled, Tag Releases stop after Wheels, the example
config, and Chart publication. If requested image publication fails, those
artifacts remain usable and the public Release is marked `images-failed`. OCI
archives are not uploaded to GitHub Release. Draft and Nightly channel behavior
comes only from their Profiles; there are no type-specific PyPI or DockerHub
prohibitions in publication code.

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
`x86_64` and `aarch64`. Official files currently use the honest generic tags:

```text
linux_x86_64
linux_aarch64
```

They must not be renamed to `manylinux_*` without a successful portability
repair. Each Wheel artifact contains:

- the Wheel;
- `wheel-result.json` schema 2;
- `auditwheel-show.txt`.

The result verifies the filename against WHEEL metadata, records platform Tags,
GLIBC symbols/floor, external libraries, and embeds the audit report digest.
The staged Release validates the standalone report again before uploading the
Wheel.

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
