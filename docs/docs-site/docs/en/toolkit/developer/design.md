# Toolkit architecture and extension

UCM Toolkit gives prechecks, metric inspection and native performance tests one command entry point. Tools have different dependencies: the CLI dispatches commands, while each Adapter owns its parameters, resource paths and execution. Toolkit is a separate Python package; see [installation and usage](../index.md).

## How one command runs

For `ucm-toolkit run dev-sandbox copy ...`:

1. `cli.main()` initializes built-in registration and recognizes `run`.
2. `commands/run.py` resolves the tool name or alias and passes remaining arguments to its Adapter.
3. `DevSandboxTool` resolves the copy binary from `build_dir` and `subcommands` and forwards native arguments.
4. `runner.py` launches the subprocess and returns its exit code. The CLI turns `ToolkitError` into a readable error and exit status.

The top-level CLI does not interpret copy cases or device settings. In dev-sandbox friendly mode, its Adapter maps `--model-type`, `--iodirect` and `--sdma` to an existing Ascend copy case and forwards remaining arguments.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` and `commands/` | Parse the top-level action, select a tool and return its result |
| `registry.py` | Own tool objects/aliases, define `ToolAdapter` and resolve repository-relative paths |
| `tools/<tool>/` | Own tool parameters, dependency checks, build, execution and cleanup |
| `runner.py` | Check commands, run subprocesses and propagate failures |
| `errors.py` | Define readable CLI errors and exit codes |

`list` shows top-level tools and `doctor` checks their environments. Only buildable tools enter `build`, currently dev-sandbox. Internal subcommands are exposed through the tool's own help.

## Build directories and runtime resources

`DevSandboxTool` reads CMake/C++ sources from the installed package and runs configure and build. Outputs default to a user cache directory isolated by installation and Toolkit version. An explicit `--build-dir` is resolved against the working directory and saved as an absolute path only after a successful build.

`run`, `doctor`, and `clean` share that user state without editing Adapter source files. `clean --dry-run` displays the target; cleanup verifies that it is a CMake build of this dev-sandbox.

## Add a tool

1. Implement `ToolAdapter` under `tools/` with a unique name, aliases and description. Keep tool-specific paths and rules in that module.
2. Implement `run(tool_args)` and `doctor()`. For compilation, declare `buildable=True` and implement build arguments and `build()`. Clean only actual generated outputs.
3. Import and register an instance in `init_builtin_tools()`. Registry rejects duplicate names and aliases.
4. Call the runner with argument lists. Keep tool semantics in the Adapter rather than coupling top-level CLI branches to one tool.
5. Update the user guide and verify list, help, doctor and run exit codes, plus build/cleanup when applicable.

See `tools/nic_monitor/` for script execution and `tools/dev_sandbox/adapter.py` for native builds. A tool need not support every lifecycle action; use existing errors for unsupported operations.

## Maintenance and verification

Focus unit checks on dispatch, argument forwarding and errors. Native builds and bandwidth measurements require the corresponding hardware and runtime; successful CLI dispatch is separate evidence. See [precheck](../user/precheck.md) for thresholds and results and [dev sandbox](dev-sandbox.md) for native development.
