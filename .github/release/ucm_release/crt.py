"""Correct the false x86-64-v2 property in manylinux_2_34 startup objects."""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

_GNU_PROPERTY_X86_FEATURE_1_AND = 0xC0000002
_GNU_PROPERTY_X86_ISA_1_NEEDED = 0xC0008002
_X86_64_BASELINE = 1
_X86_64_V2 = 2

_CET_PROPERTY = struct.pack("<III", _GNU_PROPERTY_X86_FEATURE_1_AND, 4, 3)
_FALSE_ISA_PROPERTY = struct.pack(
    "<III",
    _GNU_PROPERTY_X86_ISA_1_NEEDED,
    4,
    _X86_64_BASELINE | _X86_64_V2,
)
_BASELINE_ISA_PROPERTY = struct.pack(
    "<III",
    _GNU_PROPERTY_X86_ISA_1_NEEDED,
    4,
    _X86_64_BASELINE,
)


def _validate_startup_object(source: Path, data: bytes) -> None:
    if source.name not in {"crt1.o", "Scrt1.o"}:
        raise ValueError(f"unsupported startup object: {source.name}")
    if (
        len(data) < 20
        or data[:6] != b"\x7fELF\x02\x01"
        or struct.unpack_from("<H", data, 16)[0] != 1
        or struct.unpack_from("<H", data, 18)[0] != 62
    ):
        raise ValueError(f"startup object is not an x86-64 relocatable ELF: {source}")
    if b".note.gnu.property\0" not in data:
        raise ValueError(f"startup object has no GNU property section: {source}")
    if _CET_PROPERTY not in data:
        raise ValueError(f"startup object has no expected CET property: {source}")


def normalize_manylinux_crt(source: Path, destination: Path) -> None:
    """Copy one known AlmaLinux 9 CRT and correct only its false ISA value."""

    source = Path(source)
    destination = Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("startup object normalization requires a separate destination")

    data = source.read_bytes()
    _validate_startup_object(source, data)
    occurrences = data.count(_FALSE_ISA_PROPERTY)
    if occurrences != 1:
        raise ValueError(
            "startup object must contain exactly one baseline-plus-v2 ISA property; "
            f"found {occurrences}: {source}"
        )

    normalized = data.replace(
        _FALSE_ISA_PROPERTY,
        _BASELINE_ISA_PROPERTY,
        1,
    )
    if len(normalized) != len(data) or _CET_PROPERTY not in normalized:
        raise ValueError(
            f"startup object property normalization was not exact: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(normalized)
    shutil.copymode(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        normalize_manylinux_crt(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
