"""NumPy ufunc extension tests for the distance kernels.

Builds the `ppspatial_native` CPython extension (postpyc ext_module output),
imports it, and checks that the registered gufuncs (a) broadcast over batches
of point-pairs via the `(d),(d)->()` core signature and (b) agree with the
interpreted kernels. Separate from tests/test_native_distance.py (plain C ABI
via ctypes) and tests/test_distance.py (interpreted).
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

import ppspatial

np = pytest.importorskip("numpy")
cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")

DISTANCES = ["euclidean", "sqeuclidean", "cityblock", "chebyshev"]


@pytest.fixture(scope="module")
def native_ext(tmp_path_factory):
    from postpyc.build import build_file

    out_dir = tmp_path_factory.mktemp("ppspatial-native-ext")
    ext_path = build_file(
        Path(ppspatial.__file__),
        ext_module=True,
        module_name="ppspatial_native",
    )
    target = out_dir / ext_path.name
    ext_path.replace(target)

    spec = importlib.util.spec_from_file_location("ppspatial_native", str(target))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", DISTANCES)
def test_ext_registers_ufunc(native_ext, name):
    assert hasattr(native_ext, name), f"{name} not registered in ppspatial_native"


@pytest.mark.parametrize("name", DISTANCES)
def test_ext_scalar_matches_interpreted(native_ext, name):
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 0.0, 1.0])
    got = getattr(native_ext, name)(a, b)
    expected = getattr(ppspatial, name)(list(a), list(b))
    assert float(got) == pytest.approx(expected, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize("name", DISTANCES)
def test_ext_broadcasts_over_batch(native_ext, name):
    # Stack of 4 point-pairs; the (d),(d)->() signature loops over the last axis.
    A = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 12.0], [-1.0, 5.0]])
    B = np.array([[3.0, 4.0], [4.0, 5.0], [0.0, 0.0], [2.0, 1.0]])
    got = getattr(native_ext, name)(A, B)
    assert got.shape == (4,)
    for i in range(A.shape[0]):
        expected = getattr(ppspatial, name)(list(A[i]), list(B[i]))
        assert got[i] == pytest.approx(expected, rel=1e-15, abs=1e-15)
