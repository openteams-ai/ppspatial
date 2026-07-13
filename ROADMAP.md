# ppspatial Roadmap

`ppspatial` reimplements `scipy.spatial` in POST Python, as part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
Every kernel must run interpreted under CPython **and** compile with the
postpyc reference compiler. `scipy` is the reference only — never a runtime
dependency.

Primary compiler pressure this package generates: pairwise-distance
gufuncs, fixed-size quaternion kernels.

## Status legend

- `Done`: implemented and verified (interpreted; native where noted).
- `Active`: currently being worked on.
- `Ready`: scoped, ready to pick up.
- `Blocked`: waiting on a postpython compiler/spec capability.
- `Later`: intentionally deferred.

## Target 0: Baseline package health

Status: `Active`

- Interpreted tests pass: `python -m pytest tests/`
- Native build passes with a postpython checkout on `main`:
  `PYTHONPATH=/path/to/postpython python scripts/build_native.py`
- Extension module builds: `python scripts/build_ext.py`
- Package-manager layout builds: `pixi run build-prefix` (emits the
  `libppspatial` `lib/`/`include/`/`share/` layout under `dist/prefix`).

### Dependency note

`postpyc`/`postyp` are declared as ordinary PyPI version dependencies
(`>=0.3.0`) for installation. Following the distribution policy, no binary
wheels are shipped — PyPI carries pure source only (`py3-none-any`). Compiler
verification during development uses a local postpython checkout on `main`
via `PYTHONPATH` (see the native-build command above). The roadmap's "git
dependency" wording and the eventual `libpp*`/`pp*` conda split are pending
postpython [#14](https://github.com/openteams-ai/postpython/issues/14); until
then, PyPI version deps + a `main` checkout is the supported path (matches
ppspecial).

## Target 1: Point-to-point distances `(d),(d)->()`

Status: `Active`

| Function | Status | Notes |
|---|---|---|
| `euclidean` | `Done` | L2; exact reference values in tests |
| `sqeuclidean` | `Done` | L2 squared, no sqrt |
| `cityblock` | `Done` | L1 / Manhattan |
| `chebyshev` | `Done` | L-infinity |
| `minkowski` | `Ready` | needs a scalar exponent parameter `p`; `acc += |diff|**p`, `out = acc**(1/p)` |
| `cosine` | `Ready` | `1 - dot(a,b)/(|a||b|)` |
| `correlation` | `Ready` | cosine on mean-centered vectors |

## Target 2: Pairwise distances

Status: `Ready`

- `cdist` as `(n,d),(m,d)->(n,m)` — distance logic wrapped in two outer loops.

## Target 3: Rotations

Status: `Ready`

- `Rotation` core ops on `Shape[4]` quaternions: `compose`, `inverse`,
  `apply` `(4),(3)->(3)`.

## Blocked on postpython compiler capabilities

File each as a [postpython issue](https://github.com/openteams-ai/postpython/issues)
with a minimal reproducer when work starts — the filing is part of the work
(working rule #2).

| Item | Blocker | Workaround |
|---|---|---|
| `pdist` condensed form | output length `n(n-1)/2` is a *computed* core dimension; the gufunc layout signature only takes symbolic input dims | use the square `(n,d)->(n,n)` form meanwhile |
| `KDTree` | recursive node structs | `Later` — likely needs `@dataclass`→struct support |
| `ConvexHull` / `Delaunay` | qhull-scale computational geometry | `Later` — likely permanent foreign code |

## Working rules (summary)

See the [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
for the full list.

1. Pure POST Python — no compiler-specific escape hatches; runs interpreted
   and compiled.
2. Compiler gaps go upstream as postpython issues with reproducers, not
   silent workarounds.
3. Verify against a postpython checkout on `main`.
4. `scipy` is the reference, never a runtime dependency; prefer deterministic
   hardcoded reference values in tests.
5. Follow the ppspecial layout.
6. Accuracy is a deliverable — document targets and reference sources.
7. Small, reviewable landings — one function family per PR, both execution
   modes verified.
