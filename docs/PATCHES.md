# Patches carried on top of `er3t`/`shdom`

`software/er3t` and `software/shdom` are git submodules pinned to Hong
Chen's own public GitHub repos at `release/v0.2.0-alpha.1`, completely
unmodified -- there is no diff against upstream anywhere in those two
directories. Everything this project needs beyond what upstream
provides lives in `l2/er3t_compat.py`, applied once at runtime via
`apply_patches()` (called at the top of `run_tile_rt.py`). This file
exists so the *why* isn't buried in that module's comments alone.

## 1. `shd_run` always wraps the SHDOM call in `mpirun`

Upstream only wraps the call in `mpirun -np Ncpu` when `Ncpu > 1`; at
`Ncpu=1` it runs the raw `shdom` binary directly, no MPI runtime at
all. This repo's `shdom` build is MPI-linked (`make mpi`, no separate
serial binary), so that path silently produced a single NaN ray plus
SHDOM's un-overwritten `-1` fill value for everything else -- not a
real solve. Fixed by always routing through `mpirun`, `Ncpu=1`
included; a genuinely single-process run still hits SHDOM's own
internal serial early-out, it just gets the MPI environment it needs
to run at all.

Mechanics: this is a full replacement of `shd_run.__init__`, mutating
the class object in place (not reassigning a module name), because
`shdom.py` imports the class itself via `from er3t.rtm.shd import
shd_run` -- patching only takes effect if the actual object both
modules reference is mutated.

## 2. `shdom_ng` gets an open (non-periodic) boundary condition option

Upstream hardcodes `BCFLAG=0` (periodic in X & Y) with no way to
override it. Periodic BC isn't physically right for a single isolated
tile cut from a larger scene -- its edges have no real relationship to
wrap into each other -- and was a suspected contributor to a tile-edge
artifact found during development. `PatchedShdomNG` (a small subclass,
not a full-method replacement -- `nml_init()` only builds an in-memory
dict, so the subclass calls the original via `super()` and then fixes
one value) adds a `bcflag=` constructor argument; `run_tile_rt.py`
passes `bcflag=3` (open in X & Y) for single-tile runs.

## 3. No hardcoded Earthdata token; longer download timeout

Upstream's `get_token_earthdata()` fell back to a live-looking
hardcoded bearer token when `EARTHDATA_TOKEN` wasn't set. Fixed to
return `None` with a clear warning instead -- **every user sets their
own `EARTHDATA_TOKEN` in their own shell rc file; nothing is ever
hardcoded in this repository.** Bundled in the same fix: `curl
--max-time` raised from 60.0 to 1800.0 seconds across all three
request variants in `get_command_earthdata()`, since downloads were
timing out at 60s.

## When to retire a patch

Each override carries a signature/version guard that fails loudly if
a future `software/er3t` pin bump changes the internals it targets. If
you bump the pin and a guard trips, check whether Hong's new version
already includes the corresponding fix -- if so, delete that patch
from `er3t_compat.py` rather than carrying dead code. All three of
these are intended to be upstreamed into Hong's actual repos; once
that happens and the pin moves past it, this file (and the
corresponding function in `er3t_compat.py`) should shrink accordingly.
