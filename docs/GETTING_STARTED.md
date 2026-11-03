# Getting started

## 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:LASP-Libera/libera_cam_l2.git
cd libera_cam_l2
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This pulls `software/er3t` and `software/shdom` at the exact commits this repo is pinned to -- Hong Chen's own public GitHub repos, on his `libera-l2-fixes` branches (see `.gitmodules`). Don't expect (or make) any local changes inside `software/`; the fixes this project needs beyond Hong's `release/v0.2.0-alpha.1` are real commits on those branches, not a local patch layer -- see `docs/PATCHES.md` for what they are and why.

## 2. Build SHDOM

Requires a Fortran compiler (`gfortran`) and an MPI implementation
(`openmpi` or similar) -- `shdom` here is always MPI-linked, no
separate serial build.

```bash
cd software/shdom
mkdir -p build && make mpi
cd ../..
```

Then set `SHDOM_EXE` and put the new binaries on `PATH`:

```bash
# bash/zsh:
export SHDOM_EXE="$(pwd)/software/shdom/build/bin/shdom"
export PATH="$(pwd)/software/shdom/build/bin:$PATH"

# csh/tcsh:
setenv SHDOM_EXE "`pwd`/software/shdom/build/bin/shdom"
setenv PATH "`pwd`/software/shdom/build/bin:$PATH"
```

## 3. Python environment

This project uses [`uv`](https://docs.astral.sh/uv/) -- the same tool
the Dockerfile uses, so your local environment and the containerized
one are built the same way.

### 3a. `pyhdf` needs the HDF4 C library first

`pyhdf` (an `er3t`/`cam_l2` dependency) compiles a C extension against
the old NASA HDF4 library -- a system library, not something `uv`/pip
installs for you. If this step is skipped, `uv sync` fails while
building `pyhdf` with `fatal error: 'hdf.h' file not found`.

If you already have a conda environment with a working `pyhdf` in it
(common on LASP machines -- check with `conda env list`, then
`ls <env-path>/include/hdf.h`), just point at it:

```tcsh
setenv INCLUDE_DIRS "<env-path>/include"
setenv LIBRARY_DIRS "<env-path>/lib"
```
```bash
export INCLUDE_DIRS="<env-path>/include"
export LIBRARY_DIRS="<env-path>/lib"
```

Otherwise, get HDF4 from conda-forge into a small, dedicated
environment -- `-p` with a path under your home directory avoids
touching (or needing write access to) any existing/shared conda
install, and avoids the large, unrelated update cascade that
installing into an existing env can trigger:

```bash
conda create -p ~/.conda-envs/hdf4-libs -c conda-forge hdf4 -y
```

then point `INCLUDE_DIRS`/`LIBRARY_DIRS` at `~/.conda-envs/hdf4-libs/{include,lib}`
the same way as above. (Homebrew no longer packages `hdf4` as of
2026, so `brew install hdf4` will not work.)

**This also needs a runtime fix, every time, not just at build time.**
`pyhdf`'s compiled extension looks up `libmfhdf.0.dylib` via `@rpath`
on macOS, and the default fallback search does not include wherever
HDF4 actually lives -- so `import pyhdf` (or anything importing it,
like `er3t`) fails with `Library not loaded: @rpath/libmfhdf.0.dylib`
unless the linker is told where to look, in every shell, not only
during `uv sync`. Put this in your shell rc file (`~/.cshrc` for
csh/tcsh), using the same `<env-path>` as above:

```tcsh
setenv DYLD_FALLBACK_LIBRARY_PATH "<env-path>/lib"
```
```bash
export DYLD_FALLBACK_LIBRARY_PATH="<env-path>/lib"
```

(On Linux, the equivalent is `LD_LIBRARY_PATH` rather than
`DYLD_FALLBACK_LIBRARY_PATH` -- not yet verified against this repo's
own environment, since development so far has been on macOS.)

### 3b. Build the environment

```bash
pip install uv   # if you don't already have it
uv sync
```
```tcsh
source .venv/bin/activate.csh
```
```bash
source .venv/bin/activate
```

`uv sync` installs `cam_l2` and the pinned, editable `er3t` submodule
on its own (via `[tool.uv.sources]` in `pyproject.toml`) -- no
separate `uv pip install -e ...` step needed.

Sanity-check:

```bash
python -c "import er3t; import cam_l2; import pyhdf; print('er3t + cam_l2 + pyhdf OK')"
which shdom || echo "put software/shdom/build/bin on PATH, or set SHDOM_EXE"
```

### 3c. Fetch `er3t`'s reference data

`er3t` ships its own data installer for a public reference-data bundle
it needs at runtime (solar spectra, absorption coefficients, phase
functions, standard atmosphere profiles) -- none of this is Libera- or
LASP-specific, and none of it is in this repo. Without it, any real RT
step (e.g. `rt_shdom()` in `run_tile_rt.py`) fails looking for a file
like `software/er3t/er3t/data/atmmod/afglus.dat`.

```bash
cd software/er3t
./install.sh        # needs gdown, already installed into .venv by step 3b
cd ../..
```

Downloads a tarball from a public Google Drive link and extracts it
into `software/er3t/er3t/data/` -- expect this to take a while (no
published size; budget for it being large, same spirit as the DITL2
camera L1B above).

`./install.sh` works the same from any shell as of
`libera-l2-fixes` -- earlier versions had no `#!` line, which tcsh in
particular mis-handled (treated it as a csh script instead of running
it with bash, since tcsh's shebang-less-script heuristic keys off the
first character being `#`). If you ever see a stray `usage: conda ...`
/ `Invalid null command` error instead of the expected gdown/tar
output, run `bash install.sh` explicitly instead of `./install.sh`.

## 4. Earthdata token

Only needed if you'll use `--proxy-date` or let BRDF auto-fetch
actually download something; not needed for a plain run against
already-cached files.

**Every user sets their own token in their own shell rc file --
nothing is ever hardcoded in this repository.**

```bash
# bash/zsh (~/.bashrc or ~/.zshrc):
export EARTHDATA_TOKEN="your-token-here"

# csh/tcsh (~/.cshrc):
setenv EARTHDATA_TOKEN "your-token-here"
```

Get a token at
https://ladsweb.modaps.eosdis.nasa.gov/learn/download-files-using-laads-daac-tokens

## 5. External data (not in this repository)

### 5a. Real mission data (DITL2)

Not in this repository, by design -- the real camera/radiometer L1B and
ERA5 reanalysis files for this orbit run into the tens of GB (the camera
L1B alone is ~90 GB), so they're never pushed to git. `data/DITL2/` is
gitignored for exactly this reason (the small synthetic fixtures under
`data/l2_cam_sample/` are unaffected and stay tracked).

`build_input_manifest.py` (and therefore `run_full_pipeline.py`, which
calls it when no `--input-manifest` is given) looks for three files at
fixed default paths:

```
data/DITL2/L1B/LIBERA_L1B_CAM_V0-3-2_20260712T001717_20260712T040151_R26281034503.nc
data/DITL2/L1B/LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_R26281034503.nc
data/DITL2/ERA5/39cfa975852f21f218547a3706f8c128.nc
```

Get the team's copy of these (ask Sebastian or whoever ran this DITL2
orbit -- not distributed via this repo), then either:

- **symlink them in** (recommended -- avoids a second ~90 GB copy on
  disk):
  ```bash
  ln -s /path/to/your/DITL2 data/DITL2
  ```
- place real copies at those same paths instead, or
- pass `--cam-l1b`/`--rad-l1b`/`--era5` explicitly to
  `build_input_manifest.py` (or to `run_full_pipeline.py`, which
  forwards any flag it doesn't itself define) to point at wherever you
  actually keep them.

The first run that needs a manifest computes an MD5 of each file to
validate it -- expect the ~90 GB camera L1B to take a while the first
time (chunked/streamed, so it won't blow up your RAM, but it still has
to read all 90 GB off disk). Reruns reuse a cached checksum (keyed on
each file's path/size/mtime) from `l2/data/.checksum_cache.json`
and are near-instant; delete that file, or pass `--no-checksum-cache`,
to force a fresh hash.

The resulting `LIBERA_INPUT_MANIFEST_*.json` (written under
`l2/data/` by default) records each file's **absolute path on your
machine**, so it's inherently per-machine, non-shareable output -- that,
plus the checksum cache, is why `l2/data/` is also gitignored.

### 5b. cam_l2 reference data (surface mask, BRDF climatology)

`run_tile_rt.tile()` reads a few things through `cam_l2`'s own
`data/sample/input.yml` -- the EASE2 land/coast surface mask, MERRA2
ocean-wind reanalysis, and (as a fallback only, overridden whenever
`run_full_pipeline.py` auto-fetches or is given `--brdf-land`) a sample
VJ143C1 land BRDF file. `input.yml` itself is tiny and **is** tracked in
this repo (`data/sample/input.yml`), but the files it points at --
`data/sample/{surface_mask,cam_l1b,brdf_parms}/` -- run to ~700 MB
combined and are gitignored for the same reason DITL2 is.

Get these from Hong's `l2_cam` repo (`data/sample/` there has the same
layout) and symlink the three subdirectories in, same pattern as DITL2:

```bash
ln -s /path/to/l2_cam/data/sample/surface_mask data/sample/surface_mask
ln -s /path/to/l2_cam/data/sample/cam_l1b      data/sample/cam_l1b
ln -s /path/to/l2_cam/data/sample/brdf_parms   data/sample/brdf_parms
```

**TODO (Sebastian, 2026-11-03):** of these three, only `surface_mask/`
(~63 MB: a NSIDC EASE-Grid 2.0 land/coast mask + its lat/lon
georeference grid, likely NSIDC-0772 or a close relative -- exact
dataset ID/URL still to be confirmed) is actually fixed, standard,
orbit-independent reference data -- unlike DITL2, it never changes
between runs. For now it's distributed the same manual way as DITL2
(ask a team member, symlink it in), but since it's static and public,
it's a good candidate for its own small fetch-on-install step later
(same spirit as `software/er3t/install.sh`'s public-data download),
rather than something every new user has to be handed by hand.
`cam_l1b/` and `brdf_parms/` are likely unneeded for the real-tile
pipeline (see the `--brdf-land` override and the real-`cam_l1b`-passed-
directly points raised in chat) and may be droppable once that's
confirmed.

### 5c. The `grain` leap-second library

`libcam.util.util.jpss_tai_to_utc()` (TAI->UTC conversion for obs-ID
logs and FITS housekeeping -- not yet exercised by this repo's own
pipeline, but needed once obs-ID checking is ported in) uses the
`grain` library for a precise, leap-second-aware conversion, falling
back to a flat 37-second offset (correct from 2017 onward, i.e. today)
if `grain` isn't importable. Unlike DITL2/cam_l2 data, `grain.py` and
`leap-seconds.list` are tiny (under 15 KB combined) and **are** vendored
as real committed copies at `py/grain.py` and `py/leap-seconds.list` --
so, unlike everything else in this section, there's nothing a new user
needs to do here; it's already self-contained. (If `grain`'s canonical
source in the `camera` project is ever updated -- e.g. a new leap
second -- that copy needs a manual re-sync; it won't happen
automatically.)

## 6. Run the pipeline

```bash
python l2/run_full_pipeline.py --proxy-date 2024-07-12
```

See `l2/run_full_pipeline.py --help` for the full set of flags
(`--pad-to-orbit`, `--allow-ancillary-date-mismatch`,
`--no-brdf-auto-fetch`, etc.) -- the interface is unchanged from
`libera-cam-proxy`'s version. `--l2-cam-root` still exists (several
scripts use it to locate `software/er3t`, `software/shdom`, and
`src/`), but its default now points at this repo's own root, so you
normally never need to pass it -- only override it if you've moved
things around.

## 7. (Optional) Pick a different tile

`tile_pick.yml` (tracked in this repo) already has a hand-picked
100x100km tile for the 2024-07-12 DITL2 orbit -- the one
`run_full_pipeline.py --proxy-date 2024-07-12` uses by default. You
only need this section if you want to pick a *different* tile.

`l2/pick_tile_overview.py` is an interactive orbit quicklook: it loads
the RAD-4CH radiometer track and ERA5 windspeed for one orbit, reduces
it to one near-nadir point per ~5.5 s scan cycle, and shows three
Orthographic map views plus SZA/Latitude/windspeed panels, color-coded
by land/ocean. Click any panel (a map, or one of the three bottom
time-series plots) to pick a tile -- the figure closes itself right
after the click, and the pick is written out as yml.

**Needs a real display and an interactive matplotlib backend -- run it
on your own machine (not a plain SSH session), and do not set
`matplotlib.use('Agg')`.**

```bash
python l2/pick_tile_overview.py
```

With no arguments it defaults to this repo's own DITL2 files and
surface mask (same convention `build_input_manifest.py` uses), so once
section 5's external data is symlinked in, it should just work. To try
it without overwriting the real `tile_pick.yml`:

```bash
python l2/pick_tile_overview.py --out-yml /tmp/tile_pick_test.yml
```

See `python l2/pick_tile_overview.py --help` for the rest of the flags
(`--bin-seconds`, `--no-land-ocean` if you don't have the surface mask
symlinked in, `--rad-l1b`/`--era5` to point at a different orbit).

## 8. Docker

```bash
docker build -t libera_cam_l2 .
docker run --cpus=4.0 -e EARTHDATA_TOKEN -v /path/to/real/data:/app/data libera_cam_l2
```
