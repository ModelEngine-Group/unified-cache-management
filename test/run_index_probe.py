"""Compile/run the CPU index probe against one explicit native simu build."""
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = Path(__file__).resolve().parents[1]
if not args.build.is_absolute() or not args.output.is_absolute():
    parser.error("absolute paths required")
flags = (args.build / "ucm/store/test/CMakeFiles/ucmstore.test.dir/flags.make").read_text()
includes = next(line.split("=", 1)[1] for line in flags.splitlines() if line.startswith("CXX_INCLUDES"))
command = ["c++", "-std=c++17", "-O3", "-Wall", "-Werror", "-pthread", *shlex.split(includes),
           str(source / "test/prefetch_index_probe.cc"), "-o", str(args.output)]
for name in ("libcachestore.so", "libinfra_logger.a", "libgmock.a", "libgtest.a", "libspdlog.a", "libfmt.a"):
    matches = list(args.build.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous library {name}: {matches}")
    command.append(str(matches[0]))
    if name.endswith(".so"):
        command.append(f"-Wl,-rpath,{matches[0].parent}")
subprocess.run(command, cwd=source, check=True)
subprocess.run([str(args.output)], cwd=source, check=True)
