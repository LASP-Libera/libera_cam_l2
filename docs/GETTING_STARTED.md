# Getting started

## 1. Clone with submodules

**Status (2026-11-02): `software/shdom` is not in this repo yet.** `hong-chen/shdom` on GitHub isn't public yet -- Sebastian has asked Hong to make it accessible (he already did this for `er3t`). Until then, only the `er3t` submodule is pinned here; step 2 below (build SHDOM) can't be completed. Everything else -- `libcam/`, the `cam_l2` scripts, the patch layer -- is in place and worth exploring/setting up now.

```bash
git clone --recurse-submodules git@github.com:LASP-Libera/libera_cam_l2.git
cd libera_cam_l2
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

This pulls `software/er3t` at the exact commit this repo is pinned to -- unmodified, straight from Hong Chen's own public GitHub repo. `software/shdom` will be added as a pinned submodule the same way once it's public. Don't expect (or make) any local changes inside `software/`; everything this project needs beyond upstream lives in `l2/er3t_compat.py` -- see `docs/PATCHES.md`.

## 2. Build SHDOM

Requires a Fortran compiler (`gfortran`) and an MPI implementation
(`openmpi` or similar) -- `shdom` here is always MPI-linked, no
separate serial build.

```bash
cd software/shdom
mkdir -p build && make mpi
cd ../..
export SHDOM_EXE="$(pwd)/software/shdom/build/bin/shdom"
export PATH="$(pwd)/software/shdom/build/bin:$PATH"
```

## 3. Python environment

This project uses [`uv`](https://docs.astral.sh/uv/) -- the same tool
the Dockerfile uses, so your local environment and the containerized
one are built the same way.

```bash
pip install uv   # if you don't already have it
uv sync
uv pip install -e software/er3t
uv pip install -e .
source .venv/bin/activate
```

Sanity-check:

```bash
python -c "import er3t; import cam_l2; print('er3t + cam_l2 OK')"
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
