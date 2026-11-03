# Patches carried on top of `er3t`/`shdom`

`software/er3t` and `software/shdom` are git submodules pinned to Hong
Chen's own public GitHub repos, on his `libera-l2-fixes` branches (each
branched off `release/v0.2.0-alpha.1`). These are **real, direct commits
on Hong's repos** -- not a vendored/hand-patched copy, and not a runtime
monkeypatch layer applied by this project. That's a change from how this
worked earlier in development: a previous version of this repo carried
the same three fixes as runtime patches in `l2/er3t_compat.py`, applied
via `apply_patches()` at the top of a few scripts. Once Hong gave us
collaborator access and suggested making Libera-specific branches
directly on his own repos, we moved the fixes there instead and deleted
`er3t_compat.py` (2026-11-02) -- there is no diff left to carry in this
repo at all; `software/er3t` and `software/shdom` are simply pinned past
the commits that contain these fixes.

All four fixes below are, in Hong's own words, "generally applicable
improvements" he intends to eventually merge into his `master`/mainline.
Once that happens and this repo's pin is bumped past the merge, the
corresponding `libera-l2-fixes` branch is no longer needed and the pin
can move to whatever branch/tag Hong designates at that point.

## 1. `shd_run` always wraps the SHDOM call in `mpirun` (`er3t`)

Upstream only wrapped the call in `mpirun -np Ncpu` when `Ncpu > 1`; at
`Ncpu=1` it ran the raw `shdom` binary directly, no MPI runtime at all.
This project's `shdom` build is MPI-linked (`make mpi`, no separate
serial binary), so that path silently produced a single NaN ray plus
SHDOM's un-overwritten `-1` fill value for everything else -- not a real
solve. Fixed on `libera-l2-fixes` by always routing through `mpirun`,
`Ncpu=1` included; a genuinely single-process run still hits SHDOM's own
internal serial early-out, it just gets the MPI environment it needs to
run at all. (`er3t/rtm/shd/shd_run.py`)

## 2. `shdom_ng` gets an open (non-periodic) boundary-condition option (`er3t`)

Upstream hardcoded `BCFLAG=0` (periodic in X & Y) inside `nml_init()`,
with no way to override it. `libera-l2-fixes` adds a `bcflag=` constructor
argument to `shdom_ng` (default `0`, matching upstream's prior hardcoded
behavior) so callers can opt into open boundaries. (`er3t/rtm/shd/shdom.py`)

**Current status in this repo:** `run_tile_rt.py` does **not** currently
pass `bcflag=` and runs with the default (periodic, `BCFLAG=0`). We tried
`bcflag=3` (open in X & Y) on 2026-10-22 to work around a suspected
tile-edge artifact, but found it had zero effect under `mp_mode="mpi"` --
SHDOM's own MPI domain-decomposition forces periodic boundaries
regardless of `BCFLAG` in that mode. The override is available in the
pinned `shdom_ng` for anyone who wants to experiment with `mp_mode="py"`
or `"sh"`, but it is not presently wired into this project's orchestration
code.

## 3. No hardcoded Earthdata token; longer download timeout (`er3t`)

Upstream's `get_token_earthdata()` fell back to a live-looking hardcoded
bearer token when `EARTHDATA_TOKEN` wasn't set in the environment. Fixed
on `libera-l2-fixes` to return `None` with a clear warning instead --
**every user sets their own `EARTHDATA_TOKEN` in their own shell rc file;
nothing is ever hardcoded in this repository.** Bundled in the same fix:
`curl --max-time` raised from 60.0 to 1800.0 seconds across all three
request variants in `get_command_earthdata()`, since downloads were
timing out at 60s. (`er3t/util/daac.py`)

**Security note:** the token that was hardcoded upstream is a real,
live-looking credential, and it is still visible in the history of
Hong's public `master`/`release/v0.2.0-alpha.1` branches (removing it
on `libera-l2-fixes` doesn't erase it from earlier history). Hong should
rotate/revoke that Earthdata token.

## 4. Command-line argument buffer overflow in `shdTool_put` (`shdom`)

`src/tools/shdTool_put.f90` declared its argument buffer as
`character(len=1200)`, too short for some of the long command lines
SHDOM's batch tooling constructs. Widened to `character(len=5000)` on
`libera-l2-fixes`, matching a fix already silently carried in LASP
Nucleus's vendored copy of this file but never upstreamed or documented
before now.

## 5. `h5py`/`netcdf4` floors loosened to coexist with `libera-utils` (`er3t`)

`er3t`'s `pyproject.toml` declared `h5py>=3.15.1` and `netcdf4>=1.7.3`,
both introduced incidentally in the commit that migrated `er3t` from
setuptools to `uv` (checked: er3t's actual usage of both libraries is
long-stable API -- plain `h5py.File(...)`, `netCDF4.Dataset(...)`,
`set_auto_maskandscale()` -- nothing version-gated at either floor).
Both floors made `er3t` impossible to install alongside `libera-utils`
in the same environment: every published version of `libera-utils`
depends on `lasp-curryer`, and every published version of `lasp-curryer`
caps `h5py<3.14.0` and `netcdf4<1.7.3` -- so `er3t`'s floors and
`lasp-curryer`'s ceilings never overlap, regardless of which versions of
either package are chosen. Loosened on `libera-l2-fixes` to `h5py>=3.0`
and `netcdf4>=1.6` (matching `lasp-curryer`'s own floors -- still real
floors, just not artificially high ones) so the two can coexist. Until
this lands on Hong's mainline, `libera_cam_l2`'s own `pyproject.toml`
also carries `[tool.uv] override-dependencies` entries for both as a
belt-and-suspenders fix for anyone who hasn't yet picked up the
`libera-l2-fixes` pin.

## 6. `install.sh` missing a shebang (`er3t`)

`software/er3t/install.sh` (fetches er3t's public reference data --
solar spectra, absorption coefficients, phase functions, standard
atmosphere profiles including `afglus.dat`, from a public Google Drive
link via `gdown`) had no `#!` line. Harmless under bash/zsh (which fall
back to `/bin/sh` for a shebang-less script), but a real footgun under
tcsh: tcsh's own heuristic for a shebang-less script is to run it as
*csh*, not sh, whenever the first character is `#` -- which this file's
first line (`# pack data`) is. The result when Sebastian ran
`./install.sh` from tcsh was a confusing `usage: conda ...` / `Invalid
null command` error with nothing to do with the actual data-fetch
logic. Fixed by adding `#!/usr/bin/env bash` as the first line, so
`./install.sh` behaves the same regardless of the caller's interactive
shell (`bash install.sh` also always works, shebang or not).

## When to retire a patch / bump the pin

If Hong merges one of these into his own mainline, check whether the pin
can move past it; once all patches above are merged upstream,
`software/er3t` and `software/shdom` can be re-pinned to his regular
release branch/tag and this file can be deleted.
