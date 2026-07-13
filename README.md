# ppspatial

Spatial algorithms and distance kernels, in POST Python.

`ppspatial` reimplements `scipy.spatial` in
[POST Python](https://github.com/openteams-ai/postpython) — every kernel is
fully-typed Python that runs under the standard CPython interpreter **and**
compiles ahead-of-time to native code (a plain C shared library and a NumPy
ufunc extension module) with the POST Python reference compiler.

It is part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time as the compiler's proving ground.
Primary compiler pressure this package generates: pairwise-distance gufuncs
and fixed-size quaternion kernels.

## What a kernel looks like

Each distance is a `@guvectorize` kernel over the layout signature
`(d),(d)->()` — two coordinate vectors of equal length in, one scalar
distance out. The runtime broadcasts each kernel over batches of points
automatically.

```python
from postyp import Array, Float64
from postpyc import guvectorize
from postpyc.math import sqrt


@guvectorize([], "(d),(d)->()")
def euclidean(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    acc: Float64 = 0.0
    diff: Float64 = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        acc += diff * diff
    out[0] = sqrt(acc)
```

## Implemented

Point-to-point distances, `(d),(d)->()` — mirroring
`scipy.spatial.distance`:

| Function | Metric |
|---|---|
| `euclidean` | L2 — `sqrt(sum((a-b)**2))` |
| `sqeuclidean` | squared L2 — `sum((a-b)**2)` |
| `cityblock` | L1 / Manhattan — `sum(|a-b|)` |
| `chebyshev` | L-infinity — `max(|a-b|)` |

Results are exact up to floating-point rounding (not approximations), so
tests compare against hardcoded reference values with tight tolerances.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for full status. Next up: `minkowski`,
`cosine`, `correlation`, `cdist` (`(n,d),(m,d)->(n,m)`), and `Rotation`
quaternion ops. Blocked on compiler capabilities: `pdist`'s condensed form
(its output length `n(n-1)/2` is a computed core dimension), and
`KDTree`/`ConvexHull`/`Delaunay`.

## Usage

```python
from ppspatial import euclidean, sqeuclidean, cityblock, chebyshev

euclidean([0.0, 0.0], [3.0, 4.0])   # 5.0
cityblock([0.0, 0.0], [3.0, 4.0])   # 7.0
```

When the optional compiled `ppspatial_native` extension is installed next to
the package, the pure-Python functions are transparently replaced by native
NumPy ufuncs at import time (see `ppspatial/__init__.py`).

## Development

The package depends on `postpyc` and `postyp` (declared as PyPI version
dependencies). A C compiler is required to build native code. Compiler
verification during development uses a local
[postpython](https://github.com/openteams-ai/postpython) checkout on `main`
(PostSciPy working rule #3).

Using the pixi workspace defined in `pyproject.toml`:

```bash
pixi run -e dev test           # interpreted-mode test suite
pixi run -e dev build-native   # compile each kernel module to a .so + report
pixi run -e dev build-prefix   # emit the libppspatial lib/include/share layout
pixi run -e dev build-ext      # build ppspatial_native (NumPy-ufunc extension)
```

To verify native builds against a local postpython checkout on `main`:

```bash
PYTHONPATH=/path/to/postpython python scripts/build_native.py
```

## Distribution

Following the PostSciPy policy, `ppspatial` ships **pure Python source only**
to PyPI (`py3-none-any`) — no binary wheels, ever. Compiled artifacts come
through environment package managers (conda/pixi, nix) as a `libppspatial` +
`ppspatial` split, or you compile locally yourself (never automatically at
install or import time). See postpython's `docs/distribution.md` for the full
policy.

## Working rules

- Pure POST Python: no compiler-specific escape hatches; every kernel runs
  interpreted and compiled.
- `scipy` is the reference, never a runtime dependency. Tests may use it
  optionally; deterministic hardcoded reference values are preferred.
- Compiler gaps go upstream as postpython issues with reproducers, not silent
  workarounds.
- Verify against a postpython checkout on `main`.
- Document accuracy targets and reference sources per function.

The full rules and definition of done live in the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
