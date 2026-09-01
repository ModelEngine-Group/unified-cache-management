# UCM Toolkit Design Document

## Background

UCM is a KV cache offload plugin built on the vLLM KV connector. The `toolkit`
directory serves as a tool subproject of UCM, used to uniformly manage multiple testing, diagnostic, and auxiliary tools, providing a consistent entry point, CLI,
documentation, build, and packaging approach.

The tools currently planned for inclusion are:

- NIC load monitoring tool: currently integrated via `toolkit/src/nic_monitor/nic_monitor_pro.sh`, used for passive physical NIC traffic monitoring.
- `dev-sandbox`: currently exists under `toolkit/src/dev-sandbox`, a CMake-based C++17
  test project for testing D2D, D2H, H2D and other copy bandwidths on CUDA / Ascend platforms.
- POSIX AIO test script: the POSIX AIO script in the code repository needs to be unified under the toolkit's CLI entry point.

## Design Goals

- Provide a unified command entry point, for example `ucm-toolkit`.
- Provide consistent `--help` output and parameter style.
- Provide unified tool list, build, run, diagnostic, and clean commands.
- Provide a unified build entry for tools that require compilation, without breaking the original tool's own build semantics.
- Isolate different tools behind adapters to facilitate adding new tools later.
- Centralize the toolkit usage documentation under the `toolkit` directory.
- Support native command parameter passthrough for tools, avoiding the unified CLI blocking advanced usage.

## Non-Goals

- Do not merge all tool implementations into one giant Python file.
- Do not immediately rewrite `dev-sandbox`. The first phase only wraps its existing CMake build and binary run approach.
- Do not forcibly interfere with `dev-sandbox`'s backend detection logic at the toolkit layer.
- Do not introduce heavyweight Python CLI dependencies in the first version.
- Do not require all native tool parameters to be converted into toolkit standard parameters in the first version.

## Recommended Directory Structure

```text
toolkit/
  pyproject.toml
  README.md
  DESIGN.md
  ucm_toolkit/
    __init__.py
    cli.py
    registry.py
    runner.py
    errors.py

    commands/
      __init__.py
      build.py
      run.py
      list.py
      doctor.py
      clean.py

    tools/
      __init__.py
      dev_sandbox/
        __init__.py
        adapter.py
        README.md
      posix_aio/
        __init__.py
        adapter.py
        README.md
      nic_monitor/
        __init__.py
        adapter.py
        README.md

  src/
    dev-sandbox/
      ...

  build/
    dev-sandbox/            # optional unified build directory, not the default dev-sandbox directory

  dist/
```

Where:

- `ucm_toolkit/` is the Python CLI and tool orchestration layer.
- `ucm_toolkit/tools/*/adapter.py` is the integration layer for each tool.
- `toolkit/src/` holds the actual source code of each tool.
- `toolkit/src/dev-sandbox/` remains as the existing CMake project.
- `toolkit/build/` and `toolkit/dist/` are generated directories, recommended to be added to `.gitignore`.
- `toolkit/build/` can serve as a unified build output directory explicitly specified by the user, but for tools with existing native projects like `dev-sandbox`,
  the default should follow the original project convention, which is using `toolkit/src/dev-sandbox/build`.

## Overall Architecture

toolkit uses a lightweight Python package as the unified orchestration layer. Python is responsible for:

- CLI parameter parsing.
- Tool registration and discovery.
- Build command orchestration.
- Subprocess execution.
- Error message formatting.
- Documentation entry and help information.

Specific tools are integrated through adapters. The CLI core does not directly concern itself with how each tool builds or runs internally, but rather finds the corresponding tool object through the registry, then calls the object's `build`, `run`, `doctor` and other methods.

Tool paths, default build directory, binary relative paths, script paths, etc. are all configuration belonging to the tool object itself, and no separate `config.py` is extracted. This way the registry is the source of truth for tools, avoiding duplication between the registry and config configurations.

The conceptual structure is as follows:

```text
ucm-toolkit
  |
  +-- commands/build.py
  |     |
  |     +-- registry.get("dev-sandbox").build(args)
  |
  +-- commands/run.py
  |     |
  |     +-- registry.get("dev-sandbox").run(args)
  |     +-- registry.get("posix-aio").run(args)
  |     +-- registry.get("nic-monitor").run(args)
  |
  +-- commands/doctor.py
        |
        +-- registry.get(...).doctor(args)
```

## Tool Adapter Interface

Each tool adapter is recommended to provide a unified interface, and centralize paths as class fields or instance fields within the tool class:

```python
class ToolAdapter:
    name: str
    aliases: list[str]
    description: str
    buildable: bool

    source_dir: str | None
    build_dir: str | None
    binary_relpath: str | None
    script_path: str | None
    subcommands: dict[str, str]

    def add_build_args(self, parser): ...
    def build(self, args): ...

    def add_run_args(self, parser): ...
    def run(self, args): ...

    def doctor(self, args): ...
```

This way, when adding a new tool, you only need to add a new adapter and register it in the registry, without major changes to the CLI main flow. The paths required by the tool also follow the adapter itself, and are no longer placed in a global config file.

Example:

```python
class DevSandboxTool(ToolAdapter):
    name = "dev-sandbox"
    aliases = ["dev_sandbox"]
    buildable = True
    source_dir = "toolkit/src/dev-sandbox"
    build_dir = "toolkit/src/dev-sandbox/build"
    subcommands = {
        "copy": "module/copy/copy",
        "trans": "module/trans/trans",
        "aio": "module/aio/aio",
    }


class PosixAioTool(ToolAdapter):
    name = "posix-aio"
    buildable = False
    script_path = "ucm/store/test/e2e/posixstore_aio_test.py"
```

If executing:

```bash
ucm-toolkit build dev-sandbox --build-dir toolkit/build/dev-sandbox/release
```

then the build adapter updates `DevSandboxTool.build_dir` after a successful build. The "update" here is not just modifying the current process memory, but writing back to the Python source file that defines the tool class, so that the next execution of `ucm-toolkit run dev-sandbox copy/trans/aio` can also read the new build directory. `copy/trans/aio` as subfunctions of `dev-sandbox` read the build directory from `DevSandboxTool.build_dir`.

## Core Module Design

### `cli.py`

`cli.py` is the sole command entry point for `ucm-toolkit`, responsible for parsing user input and dispatching to specific commands or tool adapters. It does not implement specific tool logic, nor does it directly assemble the underlying commands of each tool.

Main responsibilities:

- Build the top-level argparse parser.
- Parse top-level commands, for example `list`, `doctor`, `build`, `run`, `clean`.
- Handle general help such as `ucm-toolkit --help`, `ucm-toolkit run --help`.
- Dispatch `build dev-sandbox` to the `dev-sandbox` tool object in the registry.
- Dispatch `run dev-sandbox copy ...` to the `dev-sandbox` tool object in the registry.
- Catch `ToolkitError` and uniformly convert it to command-line error output and exit codes.

Suggested functions:

```python
def main(argv: list[str] | None = None) -> int:
    ...


def build_parser() -> argparse.ArgumentParser:
    ...


def handle_list(args: argparse.Namespace) -> int:
    ...


def handle_doctor(args: argparse.Namespace) -> int:
    ...


def handle_build(args: argparse.Namespace) -> int:
    ...


def handle_run(argv: list[str]) -> int:
    ...


def handle_clean(args: argparse.Namespace) -> int:
    ...
```

The `run` command requires special handling. `cli.py` only parses out the first-level tool name, and the arguments after the first-level tool name should be passed as-is to the corresponding tool object. For `dev-sandbox`, `copy/trans/aio` are its own subfunctions, parsed further by `DevSandboxTool.run()`.

Example:

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K
```

`cli.py` should only extract:

```text
tool = "dev-sandbox"
tool_args = ["copy", "-t", "host_to_device_ce", "-s", "16K"]
```

The first-level subcommand of `tool_args` is interpreted by `DevSandboxTool.run()`; the arguments after the subcommand are entirely determined by the corresponding binary.

### `registry.py`

`registry.py` is the source of truth for tools, responsible for holding all tool objects, tool aliases, tool paths, and tool metadata. Since `config.py` is no longer designed, tool paths, script paths, binary paths, build directories, etc. should all be held by the tool objects in the registry.

Main responsibilities:

- Define the tool base class or protocol.
- Define the registration method for each tool object.
- Provide tool name and alias lookup.
- Provide the tool list.
- Hold tool path fields.
- Provide controlled field update capability, for example updating `DevSandboxTool.build_dir`.

Suggested structure:

```python
class ToolAdapter:
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    buildable: bool = False

    source_dir: str | None = None
    build_dir: str | None = None
    binary_relpath: str | None = None
    script_path: str | None = None
    subcommands: dict[str, str] = {}

    def build(self, args: argparse.Namespace) -> int:
        raise ToolNotBuildableError(self.name)

    def run(self, tool_args: list[str]) -> int:
        raise NotImplementedError

    def doctor(self) -> int:
        return 0
```

Suggested registry API:

```python
def register(tool: ToolAdapter) -> None:
    ...


def get(name: str) -> ToolAdapter:
    ...


def list_tools() -> list[ToolAdapter]:
    ...


def update_tool_field(tool_name: str, field_name: str, value: str) -> None:
    ...
```

`list_tools()` only returns top-level tools, used for `ucm-toolkit list`. For example, it only lists `dev-sandbox`, `posix-aio`, `nic-monitor`, not `copy/trans/aio`.

Suggested built-in tools:

```text
dev-sandbox  buildable tool, holds source_dir and build_dir
posix-aio    Python script tool, holds script_path
nic-monitor     passive physical NIC traffic monitoring tool
```

`copy/trans/aio` are not registered as top-level tools in the registry, but exposed only as subfunctions of `dev-sandbox`.

`update_tool_field()` should be a controlled update, not an arbitrary file writing tool. Suggested restrictions:

- Only allow updating whitelisted fields, for example `build_dir`.
- Only allow updating fields of tool classes explicitly registered in the registry/adapter.
- Validate that the field value is a plain string path before updating.
- Throw `ConfigUpdateError` or the renamed `RegistryUpdateError` on update failure.

### `runner.py`

`runner.py` is responsible for uniformly executing external commands. It wraps `subprocess`, so that build, run, doctor and other logic do not need to repeatedly handle command printing, cwd, env, exit codes, and exception conversion.

Main responsibilities:

- Check whether an external command exists.
- Execute commands and preserve stdout/stderr output as-is.
- Throw a unified error when a command fails.
- Normalize the command display format for users to easily copy and reproduce.

Suggested functions:

```python
def command_exists(command: str) -> bool:
    ...


def format_command(cmd: list[str]) -> str:
    ...


def run_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    ...


def check_command(
    cmd: list[str],
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    ...
```

Usage example:

```python
check_command(["cmake", "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], cwd=source_dir)
check_command(["cmake", "--build", build_dir, "-j", str(jobs)], cwd=source_dir)
run_command([copy_binary, *tool_args])
```

`runner.py` should not know about specific tools like `dev-sandbox`, `copy`, `posix-aio`, it only performs generic command execution.

### `errors.py`

`errors.py` defines the unified exception types for the toolkit. This way each module is only responsible for throwing explicit errors, and `cli.py` handles the unified printing.

Suggested exception hierarchy:

```python
class ToolkitError(Exception):
    exit_code: int = 1


class UnknownToolError(ToolkitError):
    ...


class ToolNotBuildableError(ToolkitError):
    ...


class BuildDirNotFoundError(ToolkitError):
    ...


class BinaryNotFoundError(ToolkitError):
    ...


class ScriptNotFoundError(ToolkitError):
    ...


class CommandNotFoundError(ToolkitError):
    ...


class CommandFailedError(ToolkitError):
    ...


class RegistryUpdateError(ToolkitError):
    ...
```

`cli.py` uniformly catches:

```python
try:
    return dispatch(args)
except ToolkitError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return exc.exit_code
```

Error messages should be as actionable as possible, for example:

```text
error: dev-sandbox binary not found: toolkit/src/dev-sandbox/build/module/copy/copy
hint: run `ucm-toolkit build dev-sandbox` first
```

Module relationships:

```text
cli.py
  -> registry.py
      -> tool adapter
          -> runner.py
          -> errors.py
```

`errors.py` can be referenced by all modules, but does not depend on other toolkit modules.

## CLI Design

After installation, a unified entry point is provided:

```bash
ucm-toolkit --help
```

The suggested top-level commands are as follows:

```bash
ucm-toolkit list
ucm-toolkit doctor [TOOL]
ucm-toolkit build TOOL [options]
ucm-toolkit run TOOL [options]
ucm-toolkit clean [TOOL]
```

Examples:

```bash
ucm-toolkit list
ucm-toolkit doctor
ucm-toolkit doctor dev-sandbox
ucm-toolkit build dev-sandbox --build-type Release --jobs 16
ucm-toolkit build dev-sandbox --cmake-arg -DCUDA_ROOT=/usr/local/cuda
ucm-toolkit build dev-sandbox --build-dir toolkit/build/dev-sandbox/release
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
ucm-toolkit run dev-sandbox trans --help
ucm-toolkit run dev-sandbox aio --help
ucm-toolkit run posix-aio --help
ucm-toolkit run nic-monitor --help
```

First-level commands such as `build`, `run`, `doctor`, `clean` are followed only by top-level tool names. `copy/trans/aio` are not exposed as top-level tools, but as run subfunctions of `dev-sandbox`. For compiled artifacts like `dev-sandbox copy/trans/aio` that already have native command-line arguments, the arguments after the subfunction name should be passed through as-is to the underlying binary by default.

## dev-sandbox Build Design

### Current dev-sandbox Build Behavior

`dev-sandbox` is currently a standalone CMake project, and the build method in its README is:

```bash
cmake -B build
cmake --build build -j
```

That is, if the original command is executed in the `toolkit/src/dev-sandbox` directory, the default build directory is:

```text
toolkit/src/dev-sandbox/build
```

After the build is complete, the `copy` executable is usually located at:

```text
toolkit/src/dev-sandbox/build/module/copy/copy
```

Other CMake artifacts are also located in the same build directory, for example:

```text
toolkit/src/dev-sandbox/build/module/trans/trans
toolkit/src/dev-sandbox/build/module/aio/aio
```

Its `cmake/DetectRuntime.cmake` automatically detects the runtime backend. The current logic is roughly:

1. Attempt to detect CUDA.
2. If CUDA is not available, attempt to detect Ascend.
3. If neither CUDA nor Ascend is available, fall back to CPU simulation mode.

No explicit platform options like the following were found in the current CMake configuration:

```text
-DPLATFORM=cuda
-DUCM_TOOLKIT_PLATFORM=cuda
-DRUNTIME_BACKEND=ascend
```

That is, the original `dev-sandbox` project currently does not support directly passing in `cuda` or `ascend` to select the backend.

However, it supports influencing runtime root detection through CMake cache variables, for example:

```text
-DCUDA_ROOT=/usr/local/cuda
-DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

These parameters are not explicit platform selectors, but can be passed through to CMake as existing CMake parameters of the original project.

These cache variables are read in `toolkit/src/dev-sandbox/cmake/DetectRuntime.cmake`:

- `CUDA_ROOT`: reads `CACHE{CUDA_ROOT}` first, then reads `CUDA_HOME`, `CUDA_PATH`, and the default path `/usr/local/cuda`.
- `ASCEND_ROOT`: reads `CACHE{ASCEND_ROOT}` first, then reads `ASCEND_HOME`, `ASCEND_TOOLKIT_HOME`, and the default path `/usr/local/Ascend/ascend-toolkit/latest`.

Therefore, the `--cmake-arg -DCUDA_ROOT=...` in toolkit is not a new business parameter, but appends the CMake cache parameter passed by the user as-is to the CMake configure command.

### toolkit's Principles for dev-sandbox

toolkit should not interfere with `dev-sandbox`'s backend selection by default. By default, toolkit should execute CMake commands equivalent to the original project, only providing a unified CLI entry point.

Default build behavior:

```bash
cd toolkit/src/dev-sandbox
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
```

This preserves `dev-sandbox`'s original auto-detection behavior.

By default, if the user does not pass `--build-dir`, toolkit should use:

```text
toolkit/src/dev-sandbox/build
```

If the user explicitly passes `--build-dir`, toolkit uses the user-specified directory, for example:

```bash
ucm-toolkit build dev-sandbox \
  --build-dir toolkit/build/dev-sandbox/release
```

toolkit does not save external state files such as "last build directory". The default build directory should be placed as a `dev-sandbox` tool class field in the adapter, for example:

```python
class DevSandboxTool(ToolAdapter):
    build_dir = "toolkit/src/dev-sandbox/build"
```

When `ucm-toolkit build dev-sandbox` is run without `--build-dir`, this default value is used. When the user passes `--build-dir`, the build command uses the user-specified path and synchronously writes back to the Python file defining `DevSandboxTool`, updating `DevSandboxTool.build_dir`. After that, when the user runs `ucm-toolkit run dev-sandbox copy/trans/aio`, there is no need to pass the build directory again; it is uniformly read from the tool class field.

Note: modifying class variables only in memory cannot persist across CLI processes, so an explicit source code update function needs to be provided here, for example:

```python
def update_tool_field(tool_name: str, field_name: str, value: str) -> None:
    ...
```

This function only allows updating controlled simple string fields in the registry/adapter, such as `build_dir`, to avoid arbitrary rewriting of Python files.

If the user needs to pass parameters supported by the original CMake project, they can use the generic passthrough parameter:

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda
```

Or:

```bash
ucm-toolkit build dev-sandbox \
  --cmake-arg -DASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest
```

The first version does not recommend forcing `--platform cuda|ascend` in `ucm-toolkit build dev-sandbox`, because this would imply that toolkit can control `dev-sandbox`'s backend selection, but the current original CMake project does not have such explicit capability.

### Optional CLI Parameters

The `dev-sandbox` build adapter is recommended to support:

```bash
ucm-toolkit build dev-sandbox \
  --build-type Release \
  --jobs 16 \
  --cmake-arg -DCUDA_ROOT=/usr/local/cuda
```

Parameter meanings:

- `--build-type`: passes `-DCMAKE_BUILD_TYPE=...`, default `Release`.
- `--jobs`: passed to `cmake --build ... -j`.
- `--build-dir`: overrides `DevSandboxTool.build_dir`; when not provided, the current value of the field is used.
- `--cmake-arg`: can be passed repeatedly, passed as-is to the CMake configure stage.

If `dev-sandbox` later adds an explicit backend parameter itself, for example:

```text
-DUCM_TOOLKIT_PLATFORM=cuda|ascend|simu
```

then toolkit would add:

```bash
ucm-toolkit build dev-sandbox --platform cuda
```

and translate it into CMake parameters supported by the original project. This translation is not done at the current stage.

## dev-sandbox Run Design

After the `dev-sandbox` build is complete, toolkit should locate and run its binary artifacts, for example:

- `module/copy/copy`
- `module/trans/trans`
- `module/aio/aio`

Among them, `copy` is the key integration target for the first phase.

For these compiled artifacts, toolkit's run strategy should be "top-level tool dispatch + subfunction pure proxy":

- toolkit is only responsible for locating the binary.
- toolkit is only responsible for providing a build hint when the binary does not exist.
- toolkit does not redefine existing parameters of the binary.
- toolkit does not re-wrap `copy`'s `-t/-s/-n/-i/-d` into `--case/--size/...`.
- In `ucm-toolkit run dev-sandbox copy`, the arguments after `copy` are parsed by the `copy` binary itself.
- `copy/trans/aio` are only shown as subfunctions in `ucm-toolkit run dev-sandbox --help`, and do not appear in `ucm-toolkit list`.

Binary lookup rules:

1. `ucm-toolkit run dev-sandbox copy/trans/aio` does not provide a `--build-dir` parameter.
2. The run stage uniformly reads `DevSandboxTool.build_dir`.
3. If the user wants to switch the build directory, they should first execute `ucm-toolkit build dev-sandbox --build-dir <path>` to update the tool class field.

toolkit does not need to install build artifacts to a separate `bin` directory, nor does it need to copy binaries. Wherever the build artifacts are generated, they are looked up from there at run time.

Example:

```bash
ucm-toolkit run dev-sandbox copy -t host_to_device_ce -s 16K -n 512 -i 128 -d 8
```

Equivalent to executing:

```bash
toolkit/src/dev-sandbox/build/module/copy/copy \
  -t host_to_device_ce \
  -s 16K \
  -n 512 \
  -i 128 \
  -d 8
```

### dev-sandbox Help Design

`ucm-toolkit run dev-sandbox copy --help` should display the native help of the underlying binary as much as possible, rather than maintaining a toolkit-side parameter description that is prone to becoming outdated.

Suggested strategy:

- `ucm-toolkit run dev-sandbox --help`: shows the available subfunctions of `dev-sandbox`, for example `copy/trans/aio`.
- `ucm-toolkit run dev-sandbox copy --help`: locates the `copy` binary and triggers its native help output.
- `ucm-toolkit run dev-sandbox trans --help`: locates the `trans` binary and triggers its native help output.
- `ucm-toolkit run dev-sandbox aio --help`: locates the `aio` binary and triggers its native help output.
- `ucm-toolkit run --help`: only shows toolkit's general run rules and the list of top-level runnable tools.

Currently, `trans` and `aio` already have `-h` or `--help` style help parameters. Currently, `copy` internally has an `ArgsParser::Help()` function, but has not registered `-h/--help` as a formal help parameter that returns success; `--help` falls into the unknown argument branch, prints Usage, and exits with a failure code. Therefore, the first version can display the native Usage by executing `copy`'s help path; a more ideal follow-up improvement is to add native `-h/--help` to the `copy` binary, so that `ucm-toolkit run dev-sandbox copy --help` can be fully equivalently forwarded to `copy --help` and get a success exit code.

## POSIX AIO Integration Design

The POSIX AIO script path is fixed at:

```text
ucm/store/test/e2e/posixstore_aio_test.py
```

POSIX AIO is integrated through a stable command:

```bash
ucm-toolkit run posix-aio
```

The current script parameters exist as global variables at the beginning of the file:

```python
worker_number = 1
shard_size = 8 * 1024 * 1024
shard_number = 1
block_number = 64
dump_epoch_number = 32
load_epoch_number = 32
storage_backends = ["./build/data"]
```

When toolkit integrates this, it should preserve these default values while providing CLI parameter override entry points:

```bash
ucm-toolkit run posix-aio \
  --worker-number 1 \
  --shard-size 8388608 \
  --shard-number 1 \
  --block-number 64 \
  --dump-epoch-number 32 \
  --load-epoch-number 32 \
  --storage-backend ./build/data
```

Parameter design:

- `--worker-number`: overrides `worker_number`.
- `--shard-size`: overrides `shard_size`, in bytes.
- `--shard-number`: overrides `shard_number`.
- `--block-number`: overrides `block_number`.
- `--dump-epoch-number`: overrides `dump_epoch_number`.
- `--load-epoch-number`: overrides `load_epoch_number`.
- `--storage-backend`: overrides `storage_backends`, can be passed multiple times.

To implement this capability, a small compatibility modification to `posixstore_aio_test.py` is recommended:

- Preserve the default values at the beginning of the file.
- Add `parse_args()`, using existing global variables as default values.
- Add `main()`, updating the runtime configuration based on argparse results.
- Keep the original default behavior when directly executing `python ucm/store/test/e2e/posixstore_aio_test.py`.

This way, `ucm-toolkit run posix-aio` can call the script through subprocess and pass in user-specified parameters; the habit of directly running the original script is also not broken.

If the script path changes later, only `posix_aio/adapter.py` needs to be modified, without changing the command entry point facing the user.

## NIC Load Monitoring Integration Design

The NIC load monitoring tool currently only supports passive monitoring, not active stress testing. The current integration script is:

```text
toolkit/src/nic_monitor/nic_monitor_pro.sh
```

Stable command entry point:

```bash
ucm-toolkit run nic-monitor
```

The script supports two modes:

- `fg [interval_sec]`: foreground dynamic display, default 2-second refresh.
- `bg [duration_hours] [interval_sec]`: background daemon monitoring, default 12 hours.

Examples:

```bash
ucm-toolkit run nic-monitor fg
ucm-toolkit run nic-monitor fg 5
ucm-toolkit run nic-monitor bg
ucm-toolkit run nic-monitor bg 24 5
ucm-toolkit run nic-monitor bg 24 5 --log-dir /mnt/test/net_log
ucm-toolkit run nic-monitor bg 24 5 --stat-cycle-seconds 600
```

The adapter uses `bash` to call the script and forward arguments. This tool does not design active traffic generation, active stress testing, or traffic generation capabilities. The underlying script uses `ethtool` to read NIC information and statistics counters, and requires root or sudo permissions at run time.

Background logs are written to the `net_log` directory under the current command execution directory by default. This can be overridden with `--log-dir`. The background stage statistics cycle defaults to 3600 seconds, which can be overridden with `--stat-cycle-seconds`.

## Packaging Design

`ucm-toolkit` should be installed as a standalone Python package, not installed by default with the main `uc-manager` package, to avoid interfering with the main package's installation, dependencies, and release process.

`toolkit/pyproject.toml` is recommended to define a standalone package:

```toml
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ucm-toolkit"
requires-python = ">=3.10"
dynamic = ["version"]

[project.scripts]
ucm-toolkit = "ucm_toolkit.cli:main"
```

The first version is recommended to use only the Python standard library:

- `argparse`
- `pathlib`
- `subprocess`
- `os`
- `sys`
- `shutil`

This reduces additional dependencies in CUDA, Ascend, container, and CI environments.

## Documentation Design

`toolkit/README.md` serves as the main entry document, recommended to include:

- Project introduction
- Installation method
- Quick start
- General CLI patterns
- Buildable tools
- `dev-sandbox` build and copy bandwidth testing
- POSIX AIO testing
- NIC load monitoring
- FAQ

Each tool adapter directory keeps its own `README.md` for recording the tool's native commands, parameters, and examples.

The CLI's `--help` and the examples in the README should be kept in sync. When adding new CLI parameters, the corresponding documentation should be updated synchronously.

## Error Handling

toolkit should fail early and provide actionable error messages:

- Tool name does not exist.
- Executing build on a non-buildable tool.
- `cmake` does not exist.
- Build directory does not exist.
- Build artifact does not exist.
- Native script path does not exist.
- Subprocess execution failed.

When a subprocess fails, the failed command and exit code should be printed for users to reproduce.

## Implementation Order

1. Add the Python package skeleton and `ucm-toolkit --help`.
2. Add the registry and adapter interfaces.
3. Add the `list`, `doctor`, `build`, `run`, `clean` commands.
4. Integrate `dev-sandbox` default CMake build, preserving the original auto-detection behavior.
5. Add `--cmake-arg` passthrough for `dev-sandbox` build.
6. Integrate `dev-sandbox` run subfunctions, supporting `ucm-toolkit run dev-sandbox copy/trans/aio ...`.
7. Integrate POSIX AIO, and add parameter override entry points to the script that preserve default values.
8. Integrate the NIC load passive monitoring script.
9. Add `toolkit/README.md` and README for each tool.

## Confirmed Decisions

- The POSIX AIO script path is `ucm/store/test/e2e/posixstore_aio_test.py`.
- POSIX AIO preserves the script's existing default values while providing CLI parameter override entry points.
- `dev-sandbox` build artifacts are not installed to a separate bin directory; wherever they are built, they are looked up from there at run time.
- toolkit does not introduce additional state files to record the build directory.
- `config.py` is no longer designed; tool paths, script paths, binary paths, and build directories are all held by tool objects in the registry.
- The default value of `DevSandboxTool.build_dir` is `toolkit/src/dev-sandbox/build`.
- Only `ucm-toolkit build dev-sandbox` supports `--build-dir`; after specifying and a successful build, it synchronously writes back to `DevSandboxTool.build_dir` in the Python source file.
- `ucm-toolkit run dev-sandbox copy/trans/aio` does not support `--build-dir`, and only reads the build directory from `DevSandboxTool.build_dir`.
- `ucm-toolkit list` only shows top-level tools, not `copy/trans/aio`; these subfunctions are exposed through `ucm-toolkit run dev-sandbox --help`.
- Currently, no `--platform cuda|ascend` semantics are added in toolkit; unless the `dev-sandbox` original project itself supports explicit backend selection parameters in the future.
- NIC load monitoring currently only supports passive monitoring, not active stress testing.
- `ucm-toolkit` is installed as a standalone package, not installed by default with the main `uc-manager` package.
