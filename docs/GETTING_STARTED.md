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

## 5. Real mission data

Not in this repository (`data/` beyond the small samples under
`data/l2_cam_sample/` is gitignored -- real L1B/MODIS/BRDF data runs
into the tens of GB). Get it from the team.

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

## 7. Docker

```bash
docker build -t libera_cam_l2 .
docker run --cpus=4.0 -e EARTHDATA_TOKEN -v /path/to/real/data:/app/data libera_cam_l2
```
