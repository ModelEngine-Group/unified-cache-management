# UCM Toolkit

Command-line utilities for environment checks, metrics, network monitoring,
storage benchmarks and device transfer benchmarks.

```sh
pip install ucm-toolkit
ucm-toolkit list
ucm-toolkit run precheck --help
```

The same-version UCM meta package also supports `uc-manager[toolkit]` and
combined backend extras such as `uc-manager[cu130,toolkit]`.

The package includes dev-sandbox sources. Compile them for your local SDK with
`ucm-toolkit build dev-sandbox`; compilation does not run during pip installation.
Network monitoring requires Linux and ethtool. Storage bandwidth benchmarks
require an installed UCM native backend and numpy. The basic CLI does not install
a CUDA or Ascend backend automatically.
