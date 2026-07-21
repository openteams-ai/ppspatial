"""Plain C ABI shared-library tests for the distance kernels.

Intentionally separate from tests/test_distance.py (interpreted mode). These
build the package shared library, load it with ctypes, and call the exported
`pp_*` C symbols directly — the same path a C / Rust / Go / Julia caller
would take. Each native result is checked against the interpreted result, so
the two execution modes are proven to agree.

The distance kernels are @guvectorize `(d),(d)->()` functions, so their C ABI
takes POST array views (`__pp_array`, spec §9.2), not bare doubles — unlike
ppspecial's scalar @vectorize kernels.
"""

from __future__ import annotations

import ctypes
import json
import shutil
from pathlib import Path

import pytest

import ppspatial

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


class PPArray(ctypes.Structure):
    """Mirror of the generated __pp_array struct (spec §9.2)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset_bytes", ctypes.c_int64),
    ]


def _pp_array(values):
    """Wrap a list of floats as a 1-D __pp_array; returns (struct, keepalives)."""
    n = len(values)
    buf = (ctypes.c_double * n)(*values)
    shape = (ctypes.c_int64 * 1)(n)
    strides = (ctypes.c_int64 * 1)(8)  # float64 = 8 bytes
    arr = PPArray(
        data=ctypes.cast(buf, ctypes.c_void_p),
        ndim=1,
        shape=shape,
        strides=strides,
        offset_bytes=0,
    )
    return arr, (buf, shape, strides)


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    from postpyc.build import build_file

    out_dir = tmp_path_factory.mktemp("ppspatial-native-abi")
    lib_path = build_file(
        Path(ppspatial.__file__),
        output=out_dir / "ppspatial.so",
        emit_header=True,
        emit_manifest=True,
    )
    return {
        "path": lib_path,
        "lib": ctypes.CDLL(str(lib_path)),
        "header": lib_path.with_suffix(".h").read_text(),
        "manifest": json.loads(lib_path.with_suffix(".json").read_text()),
    }


def _call_distance(lib, name, a_vals, b_vals):
    """Call a `(d),(d)->()` distance kernel through its stable pp_* C symbol."""
    fn = getattr(lib, f"pp_{name}")
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(PPArray),
        ctypes.POINTER(PPArray),
        ctypes.POINTER(PPArray),
        ctypes.c_int64,
    ]
    a, _ka = _pp_array(a_vals)
    b, _kb = _pp_array(b_vals)
    out, keep = _pp_array([0.0])
    out_buf = keep[0]
    fn(ctypes.byref(a), ctypes.byref(b), ctypes.byref(out), len(a_vals))
    return out_buf[0]


DISTANCES = ["euclidean", "sqeuclidean", "cityblock", "chebyshev"]

CASES = [
    ([0.0, 0.0], [3.0, 4.0]),
    ([-1.0, 5.0], [2.0, 1.0]),
    ([1.0, 2.0, 3.0], [4.0, 0.0, 1.0]),
    ([7.0], [2.0]),          # d=1 single element
    ([3.0, -4.0], [3.0, -4.0]),  # identical points -> 0
]


@pytest.mark.parametrize("name", DISTANCES)
@pytest.mark.parametrize("a,b", CASES, ids=lambda p: str(p))
def test_native_matches_interpreted(native_artifact, name, a, b):
    """The compiled C ABI kernel agrees with the interpreted kernel."""
    interpreted = getattr(ppspatial, name)(a, b)
    native = _call_distance(native_artifact["lib"], name, a, b)
    assert native == pytest.approx(interpreted, rel=1e-15, abs=1e-15)


def test_header_declares_distance_exports(native_artifact):
    header = native_artifact["header"]
    for name in DISTANCES:
        assert f"void pp_{name}(__pp_array* a, __pp_array* b, __pp_array* out" in header


def test_manifest_describes_gufunc_abi(native_artifact):
    manifest = native_artifact["manifest"]
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "ppspatial"

    exports = {entry["name"]: entry for entry in manifest["exports"]}
    for name in DISTANCES:
        assert exports[name]["c_symbol"] == f"pp_{name}"
        assert exports[name]["kind"] == "ufunc"
        assert exports[name]["ufunc"]["signature"] == "(d),(d)->()"
