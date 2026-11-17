from __future__ import annotations

import importlib
import os
import struct
import sys
from pathlib import Path

import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))
crt = importlib.import_module("ucm_release.crt")


def _x86_64_crt(*, isa: int = 3) -> bytes:
    data = bytearray(64)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 1, 62)
    data.extend(b".note.gnu.property\0")
    data.extend(struct.pack("<III", 0xC0000002, 4, 3))
    data.extend(struct.pack("<III", 0xC0008002, 4, isa))
    return bytes(data)


def test_normalize_manylinux_crt_keeps_cet_and_changes_only_the_false_isa(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crt1.o"
    destination = tmp_path / "baseline" / "crt1.o"
    source.write_bytes(_x86_64_crt())
    source.chmod(0o640)

    crt.normalize_manylinux_crt(source, destination)

    actual = destination.read_bytes()
    assert len(actual) == len(source.read_bytes())
    assert struct.pack("<III", 0xC0000002, 4, 3) in actual
    assert struct.pack("<III", 0xC0008002, 4, 3) not in actual
    assert struct.pack("<III", 0xC0008002, 4, 1) in actual
    assert os.stat(destination).st_mode & 0o777 == 0o640


def test_normalize_manylinux_crt_rejects_an_unexpected_isa_property(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Scrt1.o"
    source.write_bytes(_x86_64_crt(isa=1))

    with pytest.raises(ValueError, match="exactly one baseline-plus-v2"):
        crt.normalize_manylinux_crt(source, tmp_path / "baseline" / "Scrt1.o")


def test_normalize_manylinux_crt_rejects_non_x86_64_elf(tmp_path: Path) -> None:
    source = tmp_path / "crt1.o"
    data = bytearray(_x86_64_crt())
    struct.pack_into("<H", data, 18, 183)
    source.write_bytes(data)

    with pytest.raises(ValueError, match="x86-64 relocatable ELF"):
        crt.normalize_manylinux_crt(source, tmp_path / "baseline" / "crt1.o")
