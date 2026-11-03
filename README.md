# libera_cam_l2

Libera WFOV Camera L2 processing: cloud detection and cloud fraction,
consolidated from `libera-cam-proxy`'s L2 development work and Hong
Chen's `cam_l2` package.

**Architecture, in one paragraph:** `er3t` and `shdom` are pulled in as
git submodules pinned to Hong Chen's own public GitHub repos
(`software/er3t`, `software/shdom`), on his `libera-l2-fixes` branches
(each branched off `release/v0.2.0-alpha.1`) -- real, direct commits on
Hong's own repos, not a vendored/hand-patched copy and not a runtime
override layer in this project. See `docs/PATCHES.md` for exactly what
those fixes are, why, and when each can be retired. `src/cam_l2` is
Hong's own Libera-specific processing code (cloud detection, BRDF,
etc.), forked in directly. `libcam/` is the camera geometry /
pixel-mask support library, forked in from `libera-cam-proxy`.

Getting started: see `docs/GETTING_STARTED.md`.

Real mission/ancillary data (L1B granules, MODIS, BRDF, etc.) is not
in this repository -- get it from the team. An `EARTHDATA_TOKEN` in
your own shell environment is needed if you'll fetch BRDF files
yourself; see `docs/GETTING_STARTED.md`.
