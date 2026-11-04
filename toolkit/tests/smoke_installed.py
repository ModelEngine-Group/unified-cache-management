"""Exercise a built wheel from outside the source checkout (python -I)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import ucm_toolkit
from ucm_toolkit import registry
from ucm_toolkit.resources import resource_path, source_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-native", action="store_true")
    args = parser.parse_args()
    assert importlib.metadata.version(args.distribution) == args.version
    assert ucm_toolkit.__version__ == args.version
    assert source_root() is None, "smoke must use the installed wheel"
    registry.init_builtin_tools()
    assert {tool.name for tool in registry.list_tools()} == {
        "precheck",
        "metrics-view",
        "posix-aio",
        "nic-monitor",
        "dev-sandbox",
    }
    for name in ("dev-sandbox", "nic_monitor_pro.sh", "posixstore_aio_test.py"):
        assert resource_path(name).exists()
    package = Path(ucm_toolkit.__file__).parent
    for name in (
        "tools/precheck/precheck.defaults.json",
        "tools/metrics_view/configs/connector.json",
    ):
        json.loads((package / name).read_text())
    sources = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package.rglob("*.py")
    }
    command = [str(Path(sys.executable).with_name("ucm-toolkit"))]
    with tempfile.TemporaryDirectory(prefix="ucm-toolkit-smoke-") as directory:
        environment = dict(os.environ, XDG_CACHE_HOME=str(Path(directory) / "cache"))
        environment.pop("PYTHONPATH", None)

        def run(*arguments):
            subprocess.run(
                [*command, *arguments], cwd=directory, env=environment, check=True
            )

        run("--help")
        run("list")
        for tool in registry.list_tools():
            run("run", tool.name, "--help")
        run("run", "metrics-view", "list-configs")
        run("run", "posix-aio", "--dry-run")
        if args.build_native:
            run(
                "build",
                "dev-sandbox",
                "--jobs",
                "2",
                "--build-dir",
                str(Path(directory) / "native"),
                "--cmake-arg=-DCUDA_ROOT=" + str(Path(directory) / "no-cuda-sdk"),
                "--cmake-arg=-DASCEND_ROOT=" + str(Path(directory) / "no-ascend-sdk"),
            )
            run("doctor", "dev-sandbox")
            run(
                "run",
                "dev-sandbox",
                "copy",
                "-t",
                "host_to_anonymous_memcpy",
                "-s",
                "1K",
                "-n",
                "1",
                "-i",
                "1",
                "-d",
                "1",
            )
            run(
                "run",
                "dev-sandbox",
                "trans",
                "-t",
                "H2D",
                "-H",
                "normal",
                "-D",
                "normal",
                "-M",
                "memcpy",
                "-s",
                "1024",
                "-n",
                "1",
                "-i",
                "1",
                "-d",
                "1",
            )
            run("clean", "dev-sandbox", "--dry-run")
            assert (Path(directory) / "native/CMakeCache.txt").is_file()
            run("clean", "dev-sandbox")
            assert not (Path(directory) / "native").exists()
    assert sources == {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
    }
    print(f"Installed {args.distribution} {args.version}: smoke passed")


if __name__ == "__main__":
    main()
