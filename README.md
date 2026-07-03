# ppspatial

Spatial algorithms and distance kernels, in POST Python.

`ppspatial` reimplements `scipy.spatial` in
[POST Python](https://github.com/openteams-ai/postpython) — every kernel is
fully-typed Python that runs under the standard CPython interpreter **and**
compiles ahead-of-time to native code (a plain C shared library and a NumPy
ufunc extension module) with the POST Python reference compiler.

Status: **Planning** — this repository is scaffolding, ready for an agent or
contributor to claim. It is part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time as the compiler's proving ground.

Primary compiler pressure this package generates: pairwise-distance gufuncs, fixed-size quaternion kernels.

## Start here

1. Read the [POST Python spec](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
   and the [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
   (package map, working rules, capability matrix).
2. Copy the layout of [ppspecial](https://github.com/openteams-ai/ppspecial),
   the exemplar package: `ppspatial/` sources, `tests/`,
   `scripts/build_native.py`, `scripts/build_ext.py`, a pixi workspace with
   `test` / `build-native` / `build-ext` tasks, a git dependency on
   postpython, and a `ROADMAP.md` tracking targets and upstream requests.
3. Start with a slice from "Compiles today" below; land it as a small PR with
   tests in both execution modes.

## First slices

### Compiles today

- Distance kernels `(d),(d)->()`: euclidean, sqeuclidean, cityblock, chebyshev, minkowski, cosine, correlation
- `cdist` as `(n,d),(m,d)->(n,m)`
- `transform.Rotation` core ops on `Shape[4]` quaternions: compose, inverse, `apply` `(4),(3)->(3)`

### Blocked on compiler capabilities

File these as [postpython issues](https://github.com/openteams-ai/postpython/issues)
with minimal reproducers when you start on them — the filing is part of the
work and drives the compiler roadmap.

- `pdist` condensed form — output length n(n-1)/2 is a computed core dimension (same gufunc limit as ppsignal's convolve); use the `(n,d)->(n,n)` square form meanwhile
- `KDTree` — recursive node structs; `ConvexHull`/`Delaunay` — qhull-scale geometry, likely permanent foreign code

## Working rules (summary)

- Pure POST Python: no compiler-specific escape hatches; every kernel runs
  interpreted and compiled.
- `scipy` is the reference, never a runtime dependency. Tests may use it
  optionally; prefer deterministic hardcoded reference values.
- Compiler gaps go upstream as postpython issues with reproducers, not
  silent workarounds.
- Verify against a postpython checkout on `main`.
- Document accuracy targets and reference sources per function.

The full rules and the definition of done live in the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
