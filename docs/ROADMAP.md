# Roadmap / future work

Where this stands as of 2026-11-03: the pipeline runs end to end for one
hand-picked 100x100km tile (`tile_pick.yml`) on one DITL2 orbit -- SHDOM RT,
cloud-fraction retrieval, QA plots, CF-compliant output.

**This is not yet a working orbit-level pipeline, and dockerizing it does
not change that.** `--pad-to-orbit` only *reshapes* that one tile's result
into a full-orbit-length output array (fill values everywhere else) -- it
does **not** loop the RT/retrieval over every tile in an orbit. Looping
over tiles has not been implemented at all yet. Once this repo is
dockerized (item 4), the SDC team can run/test whatever is in it at that
point -- but "dockerized" and "done" are different milestones: this
pipeline isn't done, in the sense of being a real operational orbit-level
product, until item 1 (whole-orbit looping) actually exists.

Roughly in order of how soon each is likely to matter:

1. **Whole-orbit processing -- not started.** Extend from one hand-picked
   tile to actually looping the existing per-tile pipeline
   (`run_tile_rt.tile()`, cloud-fraction retrieval, CF-CAM output) across
   every tile in an orbit, rather than one tile picked by hand each time.
   This is the core missing piece -- nothing below substitutes for it,
   and the pipeline isn't a complete orbit-level product without it.

2. **VIIRS cross-check.** Compare this pipeline's cloud detection and cloud
   fraction against VIIRS' own operational cloud mask / CF retrieval for
   the same scene, eventually across a whole orbit. Needs real VIIRS
   imagery wired into the pipeline as an input, not just used as a proxy
   radiance source the way `fetch_proxy_scene.py` uses it today.

3. **Operational-mode coverage -- only CT exercised so far.** Everything
   run/verified through this pipeline to date has been in the camera's CT
   (cross-track, `Operational_Mode` 128) mode. The camera also has other
   operational modes -- at least RAPS and ET, and possibly others -- that
   this pipeline has not been run against or adapted for at all yet;
   extending (and testing) the pipeline for those modes is unstarted.

4. **Dockerize for operations.** The `Dockerfile` here is a start; getting
   it into an actual operational pipeline is a separate integration task.
   This is what lets the SDC team run/test the pipeline themselves -- but
   it's orthogonal to item 1: dockerizing the current single-tile pipeline
   doesn't make it orbit-capable, it just makes whatever exists at the
   time easier to run elsewhere.

5. **ATBD documentation.** Write up the algorithm (current approach, and
   eventually the angular-index and camera-time-CF approaches below) in
   ATBD form.

6. **Readability / cleanup pass.** A fair amount of this code was ported
   and patched under time pressure (see `docs/PATCHES.md` and the
   `apps/l2` -> `l2` path cleanup) -- worth a pass to make the structure
   and naming more self-explanatory for someone encountering it fresh.

7. **New algorithm development:**
   - Incremental: improve the existing cloud-detection / cloud-fraction
     approach (current standard algorithm).
   - New direction: an angular-index-based approach, of particular
     interest in the Arctic (where SZA/VZA geometry and surface
     conditions make the current approach weakest).
   - Another new direction: the so-called "camera time CF" algorithm --
     yet to be developed; no design specifics recorded here yet.

Not yet captured as GitHub issues -- this is a starting list for
discussion, not a committed schedule.
