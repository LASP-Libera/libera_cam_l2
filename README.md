# libera_cam_l2

Libera WFOV Camera L2 processing: cloud detection and cloud fraction,
consolidated from `libera-cam-proxy`'s L2 development work and Hong
Chen's `cam_l2` package.

**Architecture, in one paragraph:** `er3t` and `shdom` are pulled in as
pinned, *unmodified* dependencies straight from Hong Chen's own public
GitHub repos (`software/er3t`, `software/shdom` -- git submodules, pinned
to `release/v0.2.0-alpha.1`), not vendored/hand-edited copies. Three
small fixes this project needs on top of them live entirely in
`l2/er3t_compat.py` as runtime overrides, applied once at import time
-- `software/er3t` itself is never modified. See `docs/PATCHES.md` for
exactly what those three fixes are and why. `src/cam_l2` is Hong's own
Libera-specific processing code (cloud detection, BRDF, etc.), forked
in directly since it isn't published anywhere outside LASP's internal
Nucleus server. `libcam/` is the camera geometry / pixel-mask support
library, forked in from `libera-cam-proxy`.

Getting started: see `docs/GETTING_STARTED.md`.

Real mission/ancillary data (L1B granules, MODIS, BRDF, etc.) is not
in this repository -- get it from the team. An `EARTHDATA_TOKEN` in
your own shell environment is needed if you'll fetch BRDF files
yourself; see `docs/GETTING_STARTED.md`.
