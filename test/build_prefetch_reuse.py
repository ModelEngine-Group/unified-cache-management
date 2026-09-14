"""Build this worktree with existing locked dependencies and isolated outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--runtime", choices=("simu", "ascend-a3"), required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    if not args.build.is_absolute() or not args.log.is_absolute():
        parser.error("build and log paths must be absolute")
    if args.install and args.runtime != "ascend-a3":
        parser.error("only the A3 build may install into this source worktree")
    env = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory",
               GIT_CONFIG_VALUE_0="*")
    env["LD_LIBRARY_PATH"] = ":".join(
        entry for entry in env.get("LD_LIBRARY_PATH", "").split(":") if entry
    )
    def git(*command):
        return subprocess.check_output(["git", *command], cwd=source, env=env)

    def source_fingerprint():
        files = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        return {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in files.decode().split("\0")
            if name and (source / name).is_file()
            and (Path(name).suffix in {".py", ".h", ".hpp", ".cc", ".cpp", ".cmake"}
                 or Path(name).name == "CMakeLists.txt")
        }

    before = source_fingerprint()
    commands = [[
        "cmake", "-S", str(source), "-B", str(args.build),
        "-DCMAKE_BUILD_TYPE=Release", f"-DRUNTIME_ENVIRONMENT={args.runtime}",
        "-DBUILD_UCM_SPARSE=OFF", "-DDOWNLOAD_DEPENDENCE=OFF",
        "-DBUILD_UNIT_TESTS=" + ("ON" if args.runtime == "simu" else "OFF"),
        f"-DCMAKE_INSTALL_PREFIX={source}",
    ], ["cmake", "--build", str(args.build), "--parallel", "8"]]
    if args.install:
        commands.append(["cmake", "--install", str(args.build), "--component", "ucm"])
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=source, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    if source_fingerprint() != before:
        raise RuntimeError("source files changed while building; rebuild the final tree")
    manifest = {
        "source": str(source), "runtime": args.runtime,
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        "source_files": before,
        "libraries": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.build.rglob("*.so") if path.is_file()
        },
    }
    args.log.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"built": str(args.build), "log": str(args.log)}))


if __name__ == "__main__":
    main()
