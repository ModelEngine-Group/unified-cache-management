"""Run Python tests against a separate UCM build without installing its libraries."""

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    args, pytest_args = parser.parse_known_args()
    if not args.build.is_absolute():
        parser.error("build path must be absolute")
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source))

    class BuildExtensions(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("ucm."):
                stem = args.build.joinpath(*fullname.split("."))
                for suffix in importlib.machinery.EXTENSION_SUFFIXES:
                    extension = Path(str(stem) + suffix)
                    if extension.is_file():
                        print(f"UCM_TEST_EXTENSION {fullname}={extension}", flush=True)
                        return importlib.util.spec_from_file_location(fullname, extension)
                # A few native CMake targets install out of their build
                # subdirectory (for example infra/logger -> infra).
                matches = [candidate
                           for suffix in importlib.machinery.EXTENSION_SUFFIXES
                           for candidate in args.build.rglob(stem.name + suffix)]
                if len(matches) == 1:
                    print(f"UCM_TEST_EXTENSION {fullname}={matches[0]}", flush=True)
                    return importlib.util.spec_from_file_location(fullname, matches[0])
                if matches:
                    raise ImportError(f"ambiguous build extensions for {fullname}: {matches}")
            return None

    sys.meta_path.insert(0, BuildExtensions())
    import pytest

    print(f"UCM_TEST_SOURCE {source}", flush=True)
    raise SystemExit(pytest.main(pytest_args))


if __name__ == "__main__":
    main()
