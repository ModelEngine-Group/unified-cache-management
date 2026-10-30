# SPDX-License-Identifier: MIT
"""Device-less declarations that every interpreter of a run needs.

vLLM inspects a model's architecture in a **fresh interpreter**
(``subprocess.run([sys.executable, "-m", "vllm.model_executor.models.registry"])``),
and that child calls ``load_general_plugins()`` before anything else. A
monkey patch applied in the parent does not travel to it, so a child on a
machine with no accelerator repeats the device query the parent avoided.

Everything a child needs therefore lives here, in a module that imports
without torch, vLLM or UCM, and is reached through a generated
``sitecustomize.py``: Python's ``site`` imports that name at startup, before
the ``-m`` module runs. The parent applies the same functions directly, so
there is one implementation rather than two that can drift.

Nothing here touches KV-cache specs, grouping, shapes, strides or block IDs;
it declares which chip is being simulated.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

ARCH_ENV = "TRITON_ASCEND_ARCH"
SOC_ENV = "SOC_VERSION"
# Set by a launcher so the children of its run know what to declare.
PLATFORM_ENV = "UCM_MODEL_CHECK_SIM_PLATFORM"
# Marks an interpreter that already ran the declarations.
APPLIED_ENV = "UCM_MODEL_CHECK_SIM_APPLIED"

_DRIVER_MODULE = "triton.backends.ascend.driver"
# Both reach the device on an Ascend build. ``get_arch`` decides the triton
# target; ``get_aicore_num`` is only a kernel launch parameter.
_DEVICE_LOOKUPS = ("get_arch", "get_aicore_num")


class BootstrapError(RuntimeError):
    """A declaration that the caller asked for could not be made."""


def declared_ascend_arch() -> str:
    """The triton Ascend arch to simulate, from the environment only.

    ``TRITON_ASCEND_ARCH`` wins when set; otherwise it is derived from the
    ``SOC_VERSION`` the image declares. Nothing is read from a device.
    """

    declared = os.environ.get(ARCH_ENV, "").strip()
    if declared:
        return declared
    soc_version = os.environ.get(SOC_ENV, "").strip()
    prefix = "ascend"
    if not soc_version.lower().startswith(prefix):
        raise BootstrapError(
            f"no Ascend arch to declare: set {ARCH_ENV} (e.g. Ascend910B1) or "
            f"provide {SOC_ENV}. Without an NPU, triton cannot read the arch "
            "from the device."
        )
    return "Ascend" + soc_version[len(prefix) :].upper()


def triton_version() -> str:
    try:
        return str(getattr(importlib.import_module("triton"), "__version__", "?"))
    except ImportError:
        return "not installed"


def declare_ascend_arch(arch: str) -> list[str]:
    """Declare ``arch`` everywhere triton would otherwise query the device.

    The environment override is set because older triton honours it (3.2.2);
    newer builds log "host environment overrides are ignored" and query the
    device anyway (3.6.0+dev), so the driver's own lookups are replaced too.
    The owning class is found by searching for the lookups rather than by
    name, because it differs between versions -- ``NPUUtils`` on
    vllm-ascend 0.30.0rc2.

    Returns the replaced ``Class.method`` names. Raises ``BootstrapError``
    when triton's Ascend backend is present but owns no arch lookup, so a
    renamed lookup is reported instead of letting the device query through.
    """

    os.environ[ARCH_ENV] = arch
    try:
        driver_module = importlib.import_module(_DRIVER_MODULE)
    except ImportError:
        # No Ascend backend: the environment override is all there is to do.
        return []

    replaced: list[str] = []
    for name, candidate in vars(driver_module).items():
        if not isinstance(candidate, type):
            continue
        for lookup in _DEVICE_LOOKUPS:
            if not hasattr(candidate, lookup):
                continue
            if lookup == "get_arch":
                setattr(candidate, lookup, lambda self, _arch=arch: _arch)
            else:
                setattr(candidate, lookup, lambda self: 1)
            replaced.append(f"{name}.{lookup}")
    if not any(entry.endswith(".get_arch") for entry in replaced):
        raise BootstrapError(
            f"{_DRIVER_MODULE} owns no arch lookup to declare (triton "
            f"{triton_version()}). Without it, triton queries the device and "
            "fails with 'cann_get_soc_name failed' / 'get_arch returned "
            "NULL'. Check that module for the class resolving the target arch "
            "in this version."
        )
    return replaced


def apply(platform: str | None = None) -> list[str]:
    """Make this interpreter's declarations. Safe to call more than once.

    Used by the parent directly and by the generated ``sitecustomize`` in
    every child. A child must not be killed by a failure here -- that would
    replace a clear error with an unrelated one -- so it warns on stderr and
    leaves the original failure to surface where it happens.
    """

    platform = platform or os.environ.get(PLATFORM_ENV, "")
    if platform not in ("ascend", "cuda"):
        return []
    replaced: list[str] = []
    if platform == "ascend":
        replaced = declare_ascend_arch(declared_ascend_arch())
    # cuda needs no child-side declaration today: its blocker is vLLM's
    # platform probe, which a child that only inspects a model class does not
    # reach. If one shows up, it belongs here next to the Ascend branch.
    os.environ[APPLIED_ENV] = platform
    return replaced


# Directory name used for the generated sitecustomize, inside the caller's
# chosen parent directory.
BOOTSTRAP_DIRNAME = "ucm-model-check-bootstrap"

_SITECUSTOMIZE = """\
# Generated by ucm-toolkit model-check. Declares the simulated accelerator in
# every interpreter of one run, including the fresh ones vLLM starts to inspect
# a model architecture, which load plugins before anything else.
import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLKIT_PATH = {toolkit_path!r}

if _TOOLKIT_PATH and _TOOLKIT_PATH not in sys.path:
    sys.path.insert(0, _TOOLKIT_PATH)

try:
    from ucm_toolkit.tools.model_check import sim_bootstrap

    sim_bootstrap.apply()
except Exception as exc:  # never break an unrelated interpreter
    sys.stderr.write(
        "ucm-model-check bootstrap could not declare the simulated device: "
        "{{}}: {{}}\\n".format(type(exc).__name__, exc)
    )

# Chain to any sitecustomize this file shadowed, so the environment keeps its
# own startup behaviour. It is loaded by path and executed here: re-importing
# the name would take this module out of sys.modules while ``site`` is still
# inside its own ``import sitecustomize``, and every interpreter would then
# report "Error in sitecustomize ... KeyError: 'sitecustomize'" on stderr.
_shadowed = None
for _entry in sys.path:
    if not _entry:
        continue
    if os.path.abspath(_entry) == _SELF_DIR:
        continue
    _candidate = os.path.join(_entry, "sitecustomize.py")
    if os.path.isfile(_candidate):
        _shadowed = _candidate
        break

if _shadowed is not None:
    try:
        with open(_shadowed, "rb") as _handle:
            _code = compile(_handle.read(), _shadowed, "exec")
        exec(_code, {{"__name__": "sitecustomize", "__file__": _shadowed}})
    except Exception as exc:  # never break an unrelated interpreter
        sys.stderr.write(
            "ucm-model-check bootstrap could not run the sitecustomize it "
            "shadowed: {{}}: {{}}\\n".format(type(exc).__name__, exc)
        )

sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _SELF_DIR]
"""


def write_child_bootstrap(parent_directory: str | os.PathLike[str]) -> Path:
    """Write the ``sitecustomize.py`` a child interpreter will pick up.

    Returns the directory to put on ``PYTHONPATH``. The toolkit's own location
    is embedded so the child can import this module even when the toolkit is
    not installed into the interpreter's site-packages.
    """

    directory = Path(parent_directory) / BOOTSTRAP_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    toolkit_path = str(Path(__file__).resolve().parents[3])
    (directory / "sitecustomize.py").write_text(
        _SITECUSTOMIZE.format(toolkit_path=toolkit_path), encoding="utf-8"
    )
    return directory


def install_child_bootstrap(platform: str, parent_directory: str | os.PathLike[str]):
    """Arrange for every child interpreter to repeat this run's declarations.

    Sets ``PLATFORM_ENV`` so the child knows what to declare and prepends the
    generated directory to ``PYTHONPATH`` so ``site`` finds its
    ``sitecustomize`` first.
    """

    directory = write_child_bootstrap(parent_directory)
    os.environ[PLATFORM_ENV] = platform
    existing = os.environ.get("PYTHONPATH", "")
    entries = [str(directory)] + [p for p in existing.split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)
    return directory
