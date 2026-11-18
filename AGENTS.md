# AGENTS.md

This file records the conventions for changing code in the UCM repository: what each kind of change
needs to be verified, how code should be written, what to test, and how to submit and review a change.
Installation, deployment and user configuration are documented under `docs/` and are not repeated here.

## 1. Project layout

UCM (Unified Cache Manager) sits between LLM inference engines and storage and manages the life cycle
and persistence of KV Cache.

- `ucm/store/`: storage backends, including `posix`, `nfsstore`, `mooncakestore`, `ds3fs`, `dram`,
  `asu`, `pipeline`, `compress`, `delegator`.
- `ucm/shared/`: shared components such as logging, metrics, memory pools, routing and device
  transport.
- `ucm/integration/`: inference engine adapters for vLLM, SGLang and MindIE.
- `toolkit/`: the `ucm-toolkit` command line tool.
- `test/`: pytest suites and test configuration.
- `kv_semantics/`: KV semantics components, built only when `BUILD_UCM_ASU=ON`.
- `docs/docs-site/`: the MkDocs documentation site. English is the source language and the Chinese
  pages are maintained alongside it.

Before changing a module, read the module and its callers, and follow the existing implementation and
error handling. Do not infer behavior from directory names alone.

## 2. Determine the scope of verification first

Choose the verification according to the kind of change. Do not run a full build or the full E2E suite
by default.

| Location or kind of change | Native build required | Suggested verification |
| --- | --- | --- |
| `ucm/integration/**/*.py`, `ucm/*.py` | no | run the related pytest; without a matching test, do an import and a minimal smoke test |
| `toolkit/**` | no | `python -m pytest toolkit/tests -q` |
| `test/**` | no | run the tests involved in this change |
| `docs/**`, `*.md` | no | `python -m pytest -q docs/docs-site/tests` |
| `ucm/**/cc`, `**/CMakeLists.txt`, `setup.py`, `version.ini` | yes | build and run CTest |
| connector or KV layout behavior in `ucm/integration/**` | usually not | run `ucm-toolkit run model-check` first, then verify on a device as the change requires |
| `.github/release/**` | no | `python -m pytest -q .github/release/tests` |

In CI, a change touching `ucm/*`, `CMakeLists.txt`, `setup.py`, `pyproject.toml`, `MANIFEST.in`,
`version.ini` or `scripts/materialize_version.py` triggers the native Wheel build matrix. A
documentation-only or test-only change does not trigger that matrix, so there is no need to run it
locally to be safe.

The Python part of `ucm` imports without the native extension. When only Python files change, do not
run `pip install -e .`: that command recompiles the C++ part. Install only when the `.so` is genuinely
missing or the installation flow itself needs verification.

### C++ build

The C++ unit tests mainly cover `ucm/shared` and `ucm/store`, and can be built against the simulated
runtime:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_UNIT_TESTS=ON \
  -DRUNTIME_ENVIRONMENT=simu

cmake --build build -j
cd build && ctest -C Debug --output-on-failure
```

To run a single case:

```bash
ctest -R <CaseName> -V
```

`simu` does not need an accelerator. The full CI matrix also enables `BUILD_UCM_ASU`. Verification that
involves real devices, device copies or communication belongs in the matching E2E environment.

## 3. Coding conventions

Formatting and static checks use the tools the repository already provides:

```bash
pip install -r requirements-lint.txt
bash format.sh
```

`format.sh` runs the pre-commit checks, which include black, isort, codespell and actionlint. Format
only the files involved in this change.

For C++:

```bash
clang-format-20 -style=file -i <files>
```

Other conventions:

- Python uses Black's default line width of 88 columns; C++ uses 100 columns.
- Python docstrings use the Google style. Test modules usually need no docstring.
- Comments explain design reasons and non-obvious constraints; do not restate the code line by line.
- Imports go at the top of the file. For a circular dependency, fix the dependency rather than hiding
  it behind a function-level import.
- Catch specific exception types; do not use a bare `except` or silently ignore an exception.
- Reading and writing text files specifies `encoding="utf-8"` explicitly.
- Logging uses `ucm/logger.py`; do not use `print` in place of the project logger.
- C++ manages resources with RAII and expresses ownership with `std::unique_ptr` where possible. Follow
  the error handling of the surrounding module: `ucm/store` reports errors through `Status` rather than
  exceptions.
- C++ naming follows the surrounding module: `namespace UC::<Module>`, PascalCase for types and
  methods, a trailing underscore for member variables.
- The build enables `-Wall -Werror`. Fix new warnings; do not bypass them by widening a suppression or
  lowering a compiler option.
- New source files carry the MIT license header, copied from a file in the same directory.

## 4. Testing

### What tests should cover

- A new behavior comes with a matching test; a changed behavior updates the relevant test.
- A bug fix adds a regression test. Where the environment allows, show the result before the fix
  failing and after the fix passing.
- Formatting-only, comment-only and documentation-only changes need no functional test.
- Cover the change itself. Do not add tests unrelated to the change in order to raise coverage; the
  repository currently has no unified coverage gate.

### Test locations and naming

Python tests:

- Logic that does not depend on an inference engine, a model, a device or a network service goes in
  `test/suites/Unit/`.
- Tests that depend on an engine, a model, a device or a service endpoint go in `test/suites/E2E/`.
- File names use `test_*.py`, test classes use `Test*`, test functions use `test_*`.
- A unit test must not import an inference engine or require a device; if it does, it belongs in E2E.

C++ tests:

- `ucm/store` tests go in `ucm/store/test/case/<area>/<unit>_test.cc`.
- `ucm/shared` tests go in the corresponding directory under `ucm/shared/test/case/`.
- Fixtures use `class UC<Feature>Test : public ::testing::Test`.
- Shared resources are initialized and released in `SetUpTestSuite` / `TearDownTestSuite`.
- `gtest_discover_tests` registers every `TEST` as its own CTest item, so a single test runs with
  `ctest -R <Fixture>.<Test> -V`.
- `ucm/store` collects test files with a glob, so a new `.cc` requires running the CMake configure step
  again.

### pytest markers

| Marker | Argument | Purpose |
| --- | --- | --- |
| `stage` | `0`, `1`, `2`, `3` | unit, smoke, regression and release respectively |
| `feature` | feature name, for example `online_inference` | filters by feature |
| `platform` | `gpu`, `npu` | filters by device platform |
| `gpu_mem` | MB | free GPU memory requested by a CUDA test; the framework adds a 30% margin |
| `gpu_count` | count | number of GPUs requested by a CUDA test; sets `CUDA_VISIBLE_DEVICES` |

`conftest.py` currently uses only `stage`, `feature` and `platform` for filtering. `gpu_mem` and
`gpu_count` are resource request markers, not filter conditions.

Points to note when using them:

- `--stage=2+` selects stage 2 and above; `--feature=a,b` accepts several features.
- A test that lacks the marker being filtered on is deselected by pytest directly, with no error and
  no skip entry.
- The cases under `test/suites/Unit/` have no markers yet, so `--stage=0` does not select them. Run
  unit tests by passing the directory or file directly.
- If a filtered run reports `0 collected` or exits with code 5, no test actually ran, and the run is
  not evidence of anything passing.
- `gpu_mem` and `gpu_count` apply to CUDA tests only. The CUDA fixture locks an available GPU with
  `flock`, and the test fails when no GPU satisfies the request. On NPU these two markers do no
  resource locking; `--platform=npu` is what selects NPU cases.

### Test quality

- Assertions verify actual behavior, such as return values, persisted data, error codes or resource
  statistics. Verifying only that a mock was called, or that no exception was raised, is usually not
  enough.
- Prefer one behavior per test. Use `pytest.mark.parametrize` for similar scenarios.
- Avoid depending on fixed execution time, test order, or shared state that is not reset.
- Do not work around a failure by adding `skip` / `xfail`, retrying, widening a tolerance or deleting
  an assertion.

### Test result capture

`test/conftest.py` writes test results into `test_case_info` through `@export_vars`.

A new result capture decorator must be the innermost decorator and return the following structure:

- `_name`: the test item name;
- `_data`: a mapping of scalar fields; or
- `_proj`: a list of rows.

## 5. Connector and KV layout verification

Without a device, `ucm-toolkit run model-check` can verify the connector and the KV layout first. It
builds the engine configuration, `KVCacheSpec`, grouping and block table, and performs dump/load in
host memory.

```bash
# Ascend layout simulation: needs vLLM-Ascend and a read-only mount of the CANN driver libraries;
# no /dev/davinci* is required
ucm-toolkit run model-check --platform ascend-sim --ascend-soc A2 --layout-only \
  --model /path/to/model --storage-backends /path/to/ucm_store

# CUDA layout simulation: needs a CUDA build of vLLM
ucm-toolkit run model-check --platform cuda-sim --cuda-capability 9.0 \
  --model /path/to/model

# CPU environment: needs an official vLLM CPU build
ucm-toolkit run model-check --platform cpu --model /path/to/model
```

These modes require UCM to be built with `PLATFORM=simu`. Dropping `--layout-only` runs the full
dump/load comparison.

`ascend-sim` and `cuda-sim` come from `dev_connector` (PR #1579); the `cpu` mode predates them.

The check verifies layout, grouping, block ids and the data round trip, and cannot replace verification
on real devices. It does not cover device copy kernels, stream/event ordering, concurrency races,
performance, or the TP/PP greater than 1 case. When reporting test results, list the parts that were not
verified.

E2E tests currently run on upstream self-hosted NPU runners. A fork PR does not run them. Without
hardware, state which simulated checks actually ran and which behaviors remain unverified, and do not
describe an unrun test as passing.

## 6. Documentation

The documentation site is under `docs/docs-site/`. English is the source language, so update the
matching Chinese file when changing a document:

- `docs/docs-site/docs/en/**`
- `docs/docs-site/docs/zh/**`

Paths under the two directories must correspond one to one, otherwise the `Docs · Bilingual files`
check fails.

After changing a document, run:

```bash
python -m pytest -q docs/docs-site/tests
```

## 7. Git, commits and PRs

`develop` is the default branch. PRs are squash-merged, so the PR title becomes the final commit
subject.

PR title format:

```text
[Prefix] Imperative summary
```

The prefix uses these fixed values: `[Feat]` `[Bugfix]` `[Opt]` `[Build]` `[CI]` `[Doc]` `[Test]`
`[Misc]`. Combine them when a change spans categories, for example `[Feat][Test]`. Do not invent a new
prefix. Keep the summary imperative, without a trailing period, and within 72 characters where
possible.

Commit and PR notes:

- One PR focuses on one logical change. Do not mix in unrelated renaming, formatting or dependency
  upgrades.
- Do not add `Signed-off-by`; the project does not require a DCO.
- `Co-authored-by:` is for real human co-authors, not for marking a tool.
- Fill in the `Purpose`, `Modifications`, `Test` and `Documentation` sections of the PR template.
- State the linked issue and the tests actually run, with their results. When a problem cannot be
  reproduced, describe what was observed instead of claiming a reproduction.
- The PR branch must contain the latest target branch. CI checks whether `origin/<base>` is an ancestor
  of the current branch.
- `develop` and `release` are protected: at least one CODEOWNERS approval is required, the author cannot
  approve their own PR, and CI must pass.
- CI runs `pre-check`, `lint-and-unit-tests`, the native Wheel build when required, per-engine image
  builds, and E2E on upstream runners. The Wheel build is slow; do not push repeatedly to retry it when
  nothing changed.

## 8. Issues

Issues use the templates the repository provides. The type numbers are:

- `100`: documentation
- `110`: user story
- `200`: installation
- `300`: usage question
- `400`: bug
- `500`: feature request
- `600`: new model
- `700`: performance discussion
- `800`: other
- `900`: release checklist

A bug report includes environment information, reproduction steps, expected result, actual result and
relevant logs. New backend, engine or model support, and design-level changes, are discussed in an
issue first.

## 9. Code review

A review looks at the problems in this diff that affect correctness, compatibility, security or
maintainability.

- One review usually carries one to three substantive comments. A PR typically contains three or four
  real defects; too many comments bury the ones that need action.
- Each comment points at a specific location, states the problem and the reason, and gives a concrete
  suggestion where one exists.
- Label the severity: `blocking:`, `question:`, `nit:`. A `nit:` does not use Request changes.
- Formatting, import order, line width and unused imports belong to the automated tools. Do not write
  them into review comments.
- Do not summarize the diff file by file. Unrelated suggestions outside the diff belong in a separate
  issue, not in this review.
- Submit findings as a single review instead of scattered comments; every comment notifies all
  watchers.
- Approve when the change improves the code. Do not require it to be perfect in one pass, and state
  explicitly when a suggestion does not block.
- A review started by an agent is advisory: it does not use Request changes, does not count as an
  approval, and is not posted on someone else's PR without the maintainer asking for it. Review a
  revision once; do not resubmit a review on every push.

### Commenting on issues and PRs

- Answer the question that was asked. Do not post "+1" and do not restate the issue.
- Do not reopen a decision the maintainers have already made; raise it again only with new evidence.
- Reproduce, bisect or read the code yourself before asking a maintainer to do it.
- Do not use Update branch, repeated review requests or thread bumps to draw attention.
- Disagree with the technical claim, not with the person.
- A comment drafted with a tool is cut down before posting until its author can explain every sentence
  of it.

## 10. Working with an agent

- Small, well-defined changes can be implemented directly.
- A change that spans modules, or touches a public interface, a default configuration or a persisted
  format, starts with a scope and a plan.
- When the request is ambiguous and different readings lead to different implementations, ask first and
  do not widen the scope on your own.
- Handle only what this request covers. Note an adjacent problem if you find one, but do not fix it
  unasked.
- Do not push, force push, open a PR or comment on an issue or PR without being asked to.
- When the thing being changed has no matching test, say so; do not invent a new test strategy.
- At the end of a task, state which files changed, which commands ran and their results, and which
  checks did not run and why. For a performance-related change, provide comparable before and after
  data.
- When a test was not run, was skipped, or lacked hardware verification, say so; do not describe it as
  passing.

### Attribution and responsibility

The person submitting the change is responsible for what is finally submitted and must be able to
explain and maintain every line of it, however it was produced.

- Do not add `Signed-off-by`; the project does not require a DCO.
- Use `Co-authored-by:` only for human co-authors, never to mark a tool.
- When disclosure is wanted, put one line in the PR description; the form already seen in this
  repository is `Generated with <tool>`. Do not invent an `Assisted-by:` trailer: no merged PR in this
  repository uses one.
- Mechanical help such as formatting, spelling and renaming needs no disclosure; substantial generated
  content does.
- Judge a change by its diff, its tests and its evidence, not by whether it looks AI-written.

## 11. Change boundaries

Confirm the impact and compatibility before changing any of the following:

- `UcmKVStoreBaseV1`
- the persisted block layout
- default configuration values
- public interfaces and data formats

Other notes:

- Do not commit model weights, cache directories, build output, credentials or customer data.
- Do not modify third-party vendored code directly (`**/vendor/`).
- Do not add a third-party dependency without confirmation.
- Do not edit generated content by hand. The `BEGIN/END GENERATED KV` regions in the following files are
  maintained by the generation script:
  - `examples/metrics/metrics_configs.yaml`
  - `ucm/default_metrics_config.py`

After changing the generation source, run:

```bash
python kv_semantics/metrics/tools/generate_kv_metrics.py --sync-ucm
python kv_semantics/metrics/tools/generate_kv_metrics.py --check
```

## 12. Related documents

- [Contributing](docs/docs-site/docs/en/developer-guide/contribute.md)
- [Architecture](docs/docs-site/docs/en/developer-guide/architecture.md)
- [Request lifecycle](docs/docs-site/docs/en/developer-guide/request-lifecycle.md)
- [Extending store](docs/docs-site/docs/en/developer-guide/extending-store.md)
- [Add metrics](docs/docs-site/docs/en/developer-guide/add-metrics.md)
- [Build from source](docs/docs-site/docs/en/developer-guide/build_from_source.md)
- [Test README](test/README.md)
