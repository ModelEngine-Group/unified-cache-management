"""Check that pip resolved the exact published files recorded by the release."""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def verify(report, receipt, projects):
    expected = {item["project"]: item for item in receipt["projects"]}
    selected = set(projects)
    if not selected <= expected.keys():
        raise ValueError(
            "requested install projects are absent from the release receipt"
        )
    observed = set()
    for installed in report["install"]:
        name = canonical(installed["metadata"]["name"])
        if name not in expected:
            continue
        if name not in selected or name in observed:
            raise ValueError(f"unexpected release package installed: {name}")
        observed.add(name)
        package = expected[name]
        if installed["metadata"]["version"] != package["version"]:
            raise ValueError(f"pip installed the wrong version: {name}")
        download = installed["download_info"]
        filename = unquote(urlparse(download["url"]).path.rsplit("/", 1)[-1])
        digest = download["archive_info"]["hashes"]["sha256"]
        if {"filename": filename, "sha256": f"sha256:{digest}"} not in package["files"]:
            raise ValueError(
                f"pip installed a file outside the release receipt: {name}"
            )
    if observed != selected:
        raise ValueError(
            f"pip did not resolve all selected packages: {sorted(selected - observed)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--project", action="append", required=True)
    args = parser.parse_args()
    verify(
        json.loads(args.report.read_text()),
        json.loads(args.receipt.read_text()),
        args.project,
    )
    print("Index installation matches the published release files")


if __name__ == "__main__":
    main()
