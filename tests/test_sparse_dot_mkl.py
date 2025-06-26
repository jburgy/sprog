"""See https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Pypi-mkl-devel-does-not-contain-symlinks/m-p/1641864."""  # noqa: E501

# ruff: noqa: S101,S603

import os
import shutil
import subprocess
import sys
from ctypes.util import _is_elf  # pyright: ignore[reportAttributeAccessIssue]
from pathlib import Path
from typing import cast

import pytest
from packaging.version import parse


@pytest.fixture
def lib_mkl() -> Path:
    """Guess correct libmkl_rt.so version."""
    return max(
        Path(sys.prefix, sys.platlibdir).glob("libmkl_rt.so.*"),
        key=lambda path: parse(path.name.rpartition(".")[-1]),
    )


def test_find_lib_gcc(lib_mkl: Path) -> None:
    """Understand why :func:`ctypes.util._findLib_gcc` doesn't work."""
    output = subprocess.check_output(
        [
            cast("str", shutil.which("gcc")),
            "-Wl,-t",
            "-Wl,-shared",
            f"-Wl,-L,{lib_mkl.parent}",
            "-o",
            os.devnull,
            "-l",
            f":{lib_mkl.name}",
        ],
    )
    name = os.fsencode(lib_mkl.name)
    found = next(
        os.fsdecode(line)
        for line in output.splitlines()
        if line.endswith(name) and _is_elf(line)
    )
    assert Path(found) == lib_mkl


def test_find_lib_ld(lib_mkl: Path) -> None:
    """Understand why :func:`ctypes.util._findLib_ld` doesn't work."""
    output = subprocess.check_output(
        [
            cast("str", shutil.which("ld")),
            "-t",
            "-shared",
            "-L",
            str(lib_mkl.parent),
            "-o",
            os.devnull,
            "-l",
            f":{lib_mkl.name}",
        ],
    )
    name = os.fsencode(lib_mkl.name)
    found = next(
        os.fsdecode(line)
        for line in output.splitlines()
        if line.endswith(name) and _is_elf(line)
    )
    assert Path(found) == lib_mkl
