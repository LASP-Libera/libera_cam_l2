#!/usr/bin/env python3
"""Single-tile clear-sky RT benchmark pipeline for the Libera WFOV camera.

Three timed phases, run from the command line (see `--help`):

    chop  -- locate the requested overpass (site) in the L1B granule: an
             exact camera-pixel/radiometer-footprint match to the named
             site's target (lat, lon), plus the camera-frame time window
             around it.
    tile  -- define the 100x100 km (configurable) tile centered there,
             crop the land/ocean mask + VJ143C1 land BRDF + MERRA-2 ocean
             wind onto it (reusing Hong Chen's cam_l2.cld functions
             unchanged -- see "What this reuses from Hong's code" below),
             and reduce the tile's camera pixels to ONE tile-mean
             SZA/SAA/VZA/VAA quadruplet.
    rt    -- (a) a no-atmosphere, BRDF-only reflectance map (pure kernel
             algebra, brdf_kernels.py, no SHDOM solve), and (b) a full
             SHDOM clear-sky-with-atmosphere run (Hong's run_sim_rad
             pattern, sensor_type='camera3'), both over the same tile
             grid using the ONE tile-mean angle quadruplet from `tile`.

Each phase prints its own wall-clock time, so a single-tile run here is
the basis for extrapolating the cost of looping this over a whole
orbit (or more) of 100x100 km tiles later.

What this reuses from Hong's code (cam_l2, in --l2-cam-root), unchanged
--------------------------------------------------------------------
  cam_l2.cld.read_surface_mask   -- EASE2 3 km land/ocean/ice mask crop
  cam_l2.cld.read_brdf_parms     -- VJ143C1 land BRDF + MERRA-2 ocean wind crop
  cam_l2.cld.gen_brdf_shdom      -- combined brdf[Nx,Ny,7] array (same
                                     land=3/ocean=1/ice=4 flag convention)
  er3t.pre.{atm,abs,sfc,cld}, er3t.rtm.shd.{shd_sfc_2d,shd_atm_1d,
      shd_atm_3d,shdom_ng}       -- identical SHDOM orchestration to
                                     Hong's run_sim_rad, just parameterized
                                     (dx/dy, altitude, date, sensor grid)
                                     instead of hardcoded.
These are domain-size-agnostic (they only look at data['extent']), so
they work unchanged for a small 100x100 km tile instead of Hong's much
larger full-swath domain.

What this does NOT reuse from Hong's code
------------------------------------------
  cam_l2.cld.read_cam_l1b -- written for the VIIRS-shaped sample L1B
      test files (var_rad/var_lon/... HDF5 paths via h5py). Our real
      Libera CAM L1B granules have a different shape entirely (per-frame
      2-D rasters, MaskRetriever component membership, stride-selected
      pixels) -- that's apps/l2/elevation_sweep_timing.py's job
      (pool_stripe_pixels et al.), reused here instead, from the OTHER
      repo (--libera-cam-root). This also means data['sza_cam'] etc. in
      this script are the same per-CAMERA-PIXEL L1B fields Hong's code
      uses (not the radiometer's own -- still placeholder -999 in this
      test file -- copies), just pooled/stride-selected rather than a
      full raster.

Deliberate design choice -- ONE averaged VZA/VAA (and SZA/SAA) per tile
------------------------------------------------------------------------
Hong's camera3 SHDOM calls pass a full per-camera-pixel VZA/VAA raster
(because his domain spans an entire swath, where the angles genuinely
vary a lot). Here, by construction, the tile is small enough (<=100 km,
well under the ~10 deg angular-spread threshold Sebastian and I agreed
caps the error from treating VZA/VAA/SZA/SAA as constant across it) that
ONE tile-mean value of each angle stands in for the whole tile. SZA/SAA
have no choice anyway -- shdom_ng's solar_zenith_angle/solar_azimuth_angle
are scalar-only; there is no array form in its signature, so a SINGLE
solar beam per domain is a structural SHDOM/plane-parallel-grid
limitation, not a design choice. VZA/VAA genuinely CAN vary per sensor
position in SHDOM (sensor_zenith_angles/sensor_azimuth_angles are
array-capable) -- we are choosing not to use that here, not forced not
to. See "SHDOM sensor_type options" below for the tradeoffs of that
choice, for discussion with Hong.

SHDOM sensor_type options (for Sebastian/Hong's discussion)
------------------------------------------------------------
  "camera3"    (used here, matches Hong): one output value per
               arbitrary (x, y, vza, vaa) ray; ray-TRACED backward from
               each position (Fortran shdRad.f90 uses cos(pi - vza), the
               photon-arrival convention flipped 180 deg -- a backward
               ray-tracer convention, not a bug). Lets us put a full
               (Nx, Ny) raster of OUTPUT positions in one call even
               though every one of them shares the same (vza, vaa) --
               i.e. we get a whole reflectance MAP per tile, "for free",
               from a single-angle assumption. One SHDOM solve, N rays.
  "radiometer" (not used; the capability Sebastian asked about): ONE
               fixed ground footprint, but N simultaneous (vza, vaa)
               pairs from that ONE solved radiative field (mu=cos(vza)
               directly, the physical upward-photon convention -- NOT
               flipped like camera3). This is the natural fit for a
               future multi-angle-per-revisit design (e.g. caching
               several view geometries at the one tile location across
               repeat overpasses), but gives only one spatial point per
               call, not a map -- would need one call per output pixel
               to build a tile-wide image, which is likely much more
               expensive than one camera3 call with N rays.
  "camera1"/"camera2": full framing-camera / pushbroom-style regular
               angular grids from one viewpoint -- not a good fit here
               since our "viewpoint" is deliberately a single averaged
               angle, not a real optical camera geometry.
  SZA/SAA always single-valued regardless of sensor_type (solver
  limitation, see above).

Azimuth convention: both solar_azimuth_angle and sensor_azimuth_angles
are passed here as raw compass bearings (0=north, clockwise), exactly
as Hong's run_sim_rad does -- er3t.rtm.shd.shdom.py's shdom_ng.__init__
converts them internally (er3t.rtm.shd.cal_shd_saa / cal_shd_vaa) before
handing them to the Fortran namelist, so no manual conversion is needed
or should be applied here.

Units/caveat on the two output panels: the no-atmosphere panel is a
true dimensionless BRDF reflectance (pure kernel algebra). The
with-atmosphere panel is SHDOM's own 'radiance' target output, converted
to an "apparent reflectance" via rho = pi*L / (F0*cos(sza)) using a
PLACEHOLDER top-of-atmosphere solar irradiance constant (F0_555NM below)
-- NOT yet checked against whatever solar-flux normalization Hong's
own abs_rep/atm setup assumes internally. The relative "with vs without
atmosphere" comparison (e.g. "a little brighter") is valid regardless of
that constant; do not trust the absolute reflectance scale of the
with-atmosphere panel until that is confirmed against Hong's convention.
"""

import argparse
import datetime
import os
import shutil
import sys
import time

import numpy as np


# ── repo bootstrap ──────────────────────────────────────────────────────
# this file lives in <libera-cam-root>/apps/l2/, same as
# elevation_sweep_timing.py, which it imports directly as a sibling
# module (same established-this-session convention: no cross-script
# imports otherwise, see that file's own docstring).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_LIBERA_CAM_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _LIBERA_CAM_ROOT not in sys.path:
    sys.path.insert(0, _LIBERA_CAM_ROOT)

import elevation_sweep_timing as est  # noqa: E402

import brdf_kernels  # noqa: E402


FILL_VAL = -999.0
EARTH_R_KM = 6371.0

# ════════════════════════════════════════════════════════════════════════
# TODO -- open items, not yet resolved. Printed at the start of every run
# (see main()) so these don't stay buried in a docstring.
# ════════════════════════════════════════════════════════════════════════
TODO_BANNER = """
############################################################################
# TODO / OPEN ITEMS -- discuss with Hong before trusting absolute numbers:
#
# 1. OCEAN GLINT MISMATCH RISK (brdf_kernels.py):
#    The no-atmosphere panel's ocean reflectance is OUR OWN plain-Python
#    Cox-Munk (1954) glint implementation. It has NOT been checked against
#    SHDOM's own internal Fortran glint routine (triggered by the ocean
#    flag=1 in the BRDF array), which the with-atmosphere panel uses
#    instead. These are two independently-written pieces of physics for
#    the same thing -- they could disagree (mean-square-slope fit,
#    whitecap correction, refractive index) for reasons that have NOTHING
#    to do with the atmosphere. Land (RTLSR) is much lower risk -- it's a
#    single standardized MODIS/VIIRS formula.
#
# 2. F0_555NM (below) is a PLACEHOLDER solar constant used to convert the
#    with-atmosphere SHDOM radiance into an "apparent reflectance" -- not
#    yet checked against Hong's own abs_rep/atm solar-flux convention.
#    The relative "brighter with atmosphere" comparison is fine regardless;
#    don't trust the ABSOLUTE reflectance scale of that panel yet.
############################################################################
"""

# TODO(Sebastian): confirm against Hong's abs_rep solar-flux convention
# (see TODO_BANNER item 2 above) before trusting the ABSOLUTE scale of the
# with-atmosphere reflectance panel.
F0_555NM = 1.826  # W m^-2 nm^-1, placeholder TOA solar irradiance @555nm

# Two confirmed test sites (found this session via full-granule scans of
# the RAD-4CH file's Latitude/Longitude/Solar_Zenith_Surface fields --
# see run with --site custom to target any other (lat, lon)).
SITE_PRESETS = {
    "ocean": dict(
        label="Equatorial Pacific (ocean), daylit ascending equator crossing",
        seed_lat=0.0, seed_lon=-164.934,
        note="SZA~41.8 deg; pure ocean (EASE2 mask all-ocean in a "
             "~123x123 km neighborhood)",
    ),
    "kyushu": dict(
        label="Southern Kyushu, Japan (land/ocean coastline), highest-sun "
              "coastline crossing in the granule",
        seed_lat=30.637, seed_lon=131.302,
        note="SZA~8.675 deg; mixed coastline (EASE2 mask: 1361 ocean + "
             "83 land cells in a ~110x110 km neighborhood). CONFIRMED "
             "UNUSABLE in this granule (2026-10, Sebastian): this preset "
             "was picked by minimum-SZA alone, with no VZA constraint, and "
             "it turns out the RADIOMETER only saw this exact ground point "
             "at VZA=86.77 deg (near-limb) during this pass -- the real "
             "near-nadir ground track at that time was ~2600-2770 km away, "
             "over open Pacific. At that VZA the camera has either a huge "
             "pixel-count explosion (far side of the swath) or, as it does "
             "here, exactly ZERO camera pixels (FSW-bug-affected side, "
             "cutoff ~63-64 deg VZA) -- not a timing/geolocation bug. See "
             "'kuril' below for the replacement site, found via "
             "find_site.py's global VZA-constrained scan. Kept here for "
             "the record, not for use.",
    ),
    "kuril": dict(
        label="Kuril Islands chain (North Pacific land/ocean coastline), "
              "found via find_site.py global scan after 'kyushu' proved "
              "unreachable in this granule",
        seed_lat=46.2611, seed_lon=151.0258,
        note="SZA~29.58 deg, radiometer/camera VZA~32.28 deg (well clear "
             "of the ~63-64 deg FSW cutoff on either side); mixed "
             "coastline (EASE2 mask: 1383 ocean + 27 land cells in a "
             "~110x110 km neighborhood). Chosen over find_site.py's own "
             "lowest-SZA pick (SZA=28.6, VZA=6.1 deg, but only 1 land "
             "cell) for a more usable land/ocean balance (Sebastian, "
             "2026-10). seed_lat/lon is the exact matched RAD-4CH sample "
             "already, so chop() needs no --max-match-vza-deg here.",
    ),
}


def _log_phase(name):
    """Decorator-ish helper: call as `t0 = _phase_start(name)` ... then
    `_phase_end(name, t0)`; kept as two calls (not a context manager) so
    the printed start/end lines bracket arbitrary multi-step phase
    bodies without an extra indent level."""
    print("\n=== [%s] starting ===" % name)
    return time.time()


def _phase_end(name, t0):
    dt = time.time() - t0
    print("=== [%s] done in %.2f s ===" % (name, dt))
    return dt


def haversine_bearing(lat0, lon0, lat, lon):
    """Great-circle distance [km] and initial bearing [rad, 0=north,
    clockwise] from (lat0, lon0) to (lat, lon). Same formula used
    throughout this session's exploration scripts."""
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lat), np.radians(lon)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    dist = 2 * EARTH_R_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    y = np.sin(dlon) * np.cos(latr)
    x = np.cos(lat0r) * np.sin(latr) - np.sin(lat0r) * np.cos(latr) * np.cos(dlon)
    return dist, np.arctan2(y, x)


def footprint_axes_km(vza_deg, altitude_km, half_fov_deg=np.degrees(np.arctan(10.0 / 831.16)),
                      length_vza_cap_deg=None):
    """Representative radiometer-footprint ellipse full-axes [km] at the
    given VZA, for the QA panel only (not used anywhere in the RT
    physics). Same geometry as plot_footprint_maps.py/check_equator_
    1000W.py, generalized to a configurable altitude.

    `length_vza_cap_deg` (added 2026-10-22, per Sebastian's explicit
    request: "we should still account for radiometer footprints that go
    beyond 80, all the way to limb" -- excluding them outright, as the
    first fix did, was unsatisfactory): the elongation factor
    1/cos(VZA) that sets the ellipse LENGTH diverges as VZA -> 90 deg --
    a real, known limitation of this simple bowtie model, not specific
    to this instrument. Width (`w`, from the slant-range lookup table)
    does NOT have this problem -- it stays physically bounded all the
    way to the limb -- so only the length's cosine denominator is
    capped here, at `length_vza_cap_deg` if given (None preserves the
    original uncapped behavior for existing callers that only ever see
    near-nadir VZA, e.g. the tile-mean-geometry call below). This keeps
    every footprint IN the record, with a large-but-bounded (not
    infinite/absurd) ellipse past the cap, rather than silently
    dropping real limb footprints the way plain exclusion did."""
    phi_grid = np.linspace(1e-6, np.radians(40), 200000)
    RE, Rs = EARTH_R_KM, EARTH_R_KM + altitude_km
    r_grid = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * np.cos(phi_grid))
    vza_grid = np.degrees(np.arcsin(np.clip(Rs * np.sin(phi_grid) / r_grid, -1, 1)))
    order = np.argsort(vza_grid)
    vza_sorted, r_sorted = vza_grid[order], r_grid[order]
    R = np.interp(vza_deg, vza_sorted, r_sorted)
    theta0 = np.radians(half_fov_deg)
    w = 2.0 * R * np.tan(theta0)
    vza_for_length = (np.minimum(vza_deg, length_vza_cap_deg)
                       if length_vza_cap_deg is not None else vza_deg)
    l = w / np.cos(np.radians(vza_for_length))
    return w, l


def footprint_fill_fraction(pooled, footprint_info, min_neighbors=4,
                            occupancy_radius_factor=0.75):
    """Exposure-gap / fill-fraction diagnostic (added 2026-10-22, per
    Sebastian's request: "quantif[y] (1) the gap and (2) ... the % fill
    of pixels that we WOULD have for a given radiometer footprint if
    camera pixels in the gap had been populated ... vs. the pixels that
    were populated"), using his own suggested cheap approach: "[f]rom
    those camera pixels we DO have, we can cheaply estimate where else
    we WOULD have had camera pixels for a regular grid with 'local'
    {dx,dy}s."

    Purely geometric/empirical -- no camera angular model, no PSF
    convolution (flagged by Sebastian as possibly too compute-heavy for
    a first cut; this is the cheaper alternative, revisit with real PSF
    areal overlap later if it proves too coarse):

      1. For real camera pixels (pooled['in_box']) near each footprint,
         estimate the LOCAL pixel pitch as the median nearest-neighbor
         distance between actually-present pixels. This already reflects
         off-nadir radial/azimuthal stretching implicitly (real pixels
         really are more sparse there) with no separate geometry model.
      2. Lay a regular grid at that local pitch over the footprint
         ellipse's own bounding box; a grid cell "would have" a pixel if
         its center falls inside the ellipse (the fill-fraction
         denominator).
      3. A grid cell counts as "populated" if a real camera pixel lies
         within occupancy_radius_factor * local_pitch of its center.
      4. fill_fraction = populated / would_have.

    Also reports gap_km_median (typical local spacing -- the "pixel
    pitch") and gap_km_max (the single largest real nearest-neighbor
    gap found near this footprint -- the "how big is the hole" number),
    per footprint record, as the gap-size diagnostic.

    Mutates and returns footprint_info; adds fill_fraction/gap_km_median/
    gap_km_max to every dict in footprint_info['records'] (None where
    there isn't enough nearby real data to estimate a pitch)."""
    from scipy.spatial import cKDTree

    x_all, y_all = pooled["x_km"], pooled["y_km"]
    in_box = pooled["in_box"]
    x_box, y_box = x_all[in_box], y_all[in_box]
    records = footprint_info.get("records", [])
    if x_box.size < min_neighbors:
        for rec in records:
            rec["fill_fraction"] = None
            rec["gap_km_median"] = None
            rec["gap_km_max"] = None
        return footprint_info

    box_tree = cKDTree(np.column_stack([x_box, y_box]))

    for rec in records:
        cx, cy = rec["east_km"], rec["north_km"]
        a_km, b_km = rec["length_km"] / 2.0, rec["width_km"] / 2.0
        ang = np.radians(rec["angle_deg"])
        bbox_half_x = np.sqrt((b_km * np.cos(ang)) ** 2 + (a_km * np.sin(ang)) ** 2)
        bbox_half_y = np.sqrt((b_km * np.sin(ang)) ** 2 + (a_km * np.cos(ang)) ** 2)

        # Generous padded neighborhood (1.5x the ellipse bbox) purely for
        # estimating the local pixel pitch -- wider than the ellipse itself
        # so the pitch estimate isn't starved right at the footprint edge.
        search_r = 1.5 * max(bbox_half_x, bbox_half_y)
        nearby_idx = box_tree.query_ball_point([cx, cy], r=search_r)
        if len(nearby_idx) < min_neighbors:
            rec["fill_fraction"] = None
            rec["gap_km_median"] = None
            rec["gap_km_max"] = None
            continue

        pts = np.column_stack([x_box[nearby_idx], y_box[nearby_idx]])
        local_tree = cKDTree(pts)
        nn_dist, _ = local_tree.query(pts, k=2)
        nn_dist = nn_dist[:, 1]
        nn_dist = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
        if nn_dist.size == 0:
            rec["fill_fraction"] = None
            rec["gap_km_median"] = None
            rec["gap_km_max"] = None
            continue
        pitch = float(np.median(nn_dist))
        gap_max = float(np.max(nn_dist))
        rec["gap_km_median"] = round(pitch, 4)
        rec["gap_km_max"] = round(gap_max, 4)
        if pitch <= 0:
            rec["fill_fraction"] = None
            continue

        nx_grid = max(1, int(np.ceil(2 * bbox_half_x / pitch)))
        ny_grid = max(1, int(np.ceil(2 * bbox_half_y / pitch)))
        gx = cx - bbox_half_x + (np.arange(nx_grid) + 0.5) * pitch
        gy = cy - bbox_half_y + (np.arange(ny_grid) + 0.5) * pitch
        GX, GY = np.meshgrid(gx, gy)
        dx, dy = GX - cx, GY - cy
        # un-rotate into the ellipse's own (width=x=b_km, height=y=a_km)
        # frame -- same matplotlib Ellipse(angle=angle_deg) convention used
        # throughout this module (angle = -VAA).
        xl = dx * np.cos(ang) + dy * np.sin(ang)
        yl = -dx * np.sin(ang) + dy * np.cos(ang)
        inside_ellipse = (xl / b_km) ** 2 + (yl / a_km) ** 2 <= 1.0
        would_have = int(inside_ellipse.sum())
        if would_have == 0:
            rec["fill_fraction"] = None
            continue
        grid_pts = np.column_stack([GX[inside_ellipse], GY[inside_ellipse]])
        occ_dist, _ = box_tree.query(grid_pts, k=1)
        populated = int((occ_dist <= occupancy_radius_factor * pitch).sum())
        rec["fill_fraction"] = round(populated / would_have, 4)

    valid_ff = [r["fill_fraction"] for r in records if r["fill_fraction"] is not None]
    footprint_info["fill_fraction_median"] = (round(float(np.median(valid_ff)), 4)
                                               if valid_ff else None)
    footprint_info["fill_fraction_min"] = (round(float(np.min(valid_ff)), 4)
                                            if valid_ff else None)
    return footprint_info


def read_rad4ch_full_geometry(rad_l1b):
    """Read the FULL (unwindowed, every sample in the file) per-footprint
    geometry arrays from a real RAD-4CH L1B file. Factored out of
    footprints_feeding_tile() 2026-10-23 (item D, optional full-orbit
    padding of the CF-CAM output -- Sebastian: "the file itself is
    lengthened to match the full L1B radiometer input file's length")
    so write_cf_cam_output.py's padding code reads this EXACT same way
    instead of risking a second, possibly-inconsistent transcription of
    the epoch-sensitive logic below.

    NOTE (bug found 2026-10-22, first real run -- gave 0 footprints for a
    tile with 18013 valid camera pixels, obviously wrong): an earlier
    version of this function read radiometer_time via raw netCDF4
    (Dataset(...).variables[...][:]), which does NOT apply CF decoding --
    that gives nanoseconds since the file's own epoch (1958-01-01), while
    t_center_ns (chop()'s match["t_ns"]) comes from xr.open_dataset(),
    which DOES auto-decode via the variable's units/calendar attrs into
    absolute datetime64[ns] (numpy's own epoch, 1970-01-01) before
    chop() casts it to int64. Comparing those two was comparing values
    ~12 years (the 1958-1970 gap) apart -- the search window matched
    nothing, ever. Same bug *category* as pick_tile_overview.py's
    decode_radiometer_time() 12-year offset (2026-10-21), just introduced
    fresh here via a different code path (raw netCDF4 vs. xarray) instead
    of double-decoding. Fixed by reading this file the exact same way
    chop() does (xarray, not netCDF4) so the two epochs can never
    disagree again.

    Returns a dict of full-length 1-D arrays, in the file's own
    RADIOMETER_TIME order, with NO filtering/windowing at all: t_ns
    (int64, 1970 epoch, CF-decoded), lat, lon, vza, vaa (float64,
    always-present fields), alt, sza, raa, saa (float64; Added
    2026-10-22 for the CF-CAM output writer's Geolocation/Geometry
    pass-through block -- all four are REAL fields in the RAD-4CH file
    used here (V0-7-0) -- confirmed via a direct netCDF4 variable-name
    check -- even though the 2026-06-24 spec sheet flags
    Solar_Azimuth_Surface_WRT_North as "PENDING, requires RAD-4CH to add
    it first": RAD-4CH has since gained it. Read defensively anyway
    (fall back to an all-FILL_VAL array) in case an older RAD-4CH file
    without them is ever used here)."""
    import xarray as xr
    ds = xr.open_dataset(rad_l1b)
    t_ns_all = np.asarray(ds["radiometer_time"].values, dtype=np.int64)
    lat_all = np.asarray(ds["Latitude"].values, dtype=np.float64)
    lon_all = np.asarray(ds["Longitude"].values, dtype=np.float64)
    vza_all = np.asarray(ds["Viewing_Zenith_Surface"].values, dtype=np.float64)
    vaa_all = np.asarray(ds["Viewing_Azimuth_Surface_WRT_North"].values, dtype=np.float64)

    def _read_optional(name):
        if name in ds:
            return np.asarray(ds[name].values, dtype=np.float64)
        return np.full(t_ns_all.shape, FILL_VAL, dtype=np.float64)
    alt_all = _read_optional("Altitude")
    sza_all = _read_optional("Solar_Zenith_Surface")
    raa_all = _read_optional("Relative_Azimuth_Surface")
    saa_all = _read_optional("Solar_Azimuth_Surface_WRT_North")
    ds.close()
    return dict(t_ns=t_ns_all, lat=lat_all, lon=lon_all, vza=vza_all,
               vaa=vaa_all, alt=alt_all, sza=sza_all, raa=raa_all, saa=saa_all)


def footprints_feeding_tile(rad_l1b, lat0, lon0, tile_km, t_center_ns, window_s,
                             altitude_km, length_vza_cap_deg=80.0,
                             verbose=True):
    """Which real RAD-4CH radiometer footprints does this one tile's RT
    solve actually feed -- i.e. for how many CF-CAM output records
    (one per radiometer footprint, per the approved spec) can
    `cam_radiance_footprint` be populated from THIS tile's computed
    radiance field, vs. needing an adjacent tile too?

    This is a DIFFERENT question from tile()'s own pixel pooling (how
    many camera pixels fall in the tile box) and from
    plot_box_angle_vza.py's "camera pixels per radiometer footprint"
    panel (how many camera pixels feed ONE given footprint, for the
    eventual real PSF convolution) -- added 2026-10-22 after Sebastian
    asked "how many radiance footprints does this tile feed" and
    pointed out the existing per-footprint-pixel-count work answers a
    related but different question.

    Footprint ellipse geometry matches plot_footprint_maps.py exactly
    (20 km nadir diameter, constant solid angle via footprint_axes_km(),
    oriented along the true radial bearing -- angle=-VAA, the +90deg
    version was a bug fixed 2026-10-13, see that script's docstring).
    `window_s` should be the same half-window passed to chop() for this
    tile's match, so the footprint candidates searched here are drawn
    from the same radiometer samples chop()/tile() already used.

    Classification (via each ellipse's own rotated axis-aligned bounding
    box vs. the tile box -- not an exact ellipse/rectangle polygon
    intersection, but good enough to flag real candidates):
      'inside' -- ellipse bounding box entirely within the tile box;
                  this tile alone can fully populate this footprint's
                  cam_radiance_footprint.
      'edge'   -- ellipse overlaps the tile box but pokes outside it;
                  this is the stitching case Sebastian flagged
                  2026-10-22 (needs pixels from an adjacent tile too) --
                  identified here, NOT yet resolved/stitched.
    Footprints with no overlap at all are dropped.

    `length_vza_cap_deg` (revised 2026-10-22 per Sebastian's explicit
    pushback on the first version of this fix, which EXCLUDED high-VZA
    footprints outright: "we should still account for radiometer
    footprints that go beyond 80, all the way to limb (just shy of
    90)"): footprint_axes_km()'s length = width / cos(VZA) diverges as
    VZA -> 90 deg (a real, known limitation of the simple bowtie-effect
    model, not specific to this instrument -- CERES-type footprint
    models are routinely only quoted up to ~70-80 deg for the same
    reason; width does NOT have this problem and is never capped).
    Confirmed on this run's real data: a VZA=89.8 deg sample ~3257 km
    from the tile center got an UNCAPPED modeled ellipse length of
    25760 km. Rather than excluding these samples (which silently drops
    real limb footprints from the CF-CAM accounting -- not acceptable),
    footprint_axes_km() now caps the cosine term feeding the LENGTH
    calculation at this VZA, so every footprint stays in `records` with
    a large-but-bounded ellipse past the cap; `vza_model_capped` flags
    which records that applies to, and `n_vza_model_capped` counts them,
    so the limb-geometry approximation stays visible without dropping
    any footprint from the record.

    Returns a dict with per-footprint `records` (lat, lon, local
    east_km/north_km, vza_deg, vaa_deg, width_km, length_km, angle_deg,
    t_ns, classification, vza_model_capped) plus
    `n_inside`/`n_edge`/`n_vza_model_capped` counts.
    """
    full = read_rad4ch_full_geometry(rad_l1b)
    t_ns_all, lat_all, lon_all = full["t_ns"], full["lat"], full["lon"]
    vza_all, vaa_all = full["vza"], full["vaa"]
    alt_all, sza_all, raa_all, saa_all = (full["alt"], full["sza"],
                                          full["raa"], full["saa"])

    in_window = np.abs(t_ns_all - t_center_ns) <= window_s * 1e9
    good = (in_window
            & np.isfinite(lat_all) & (lat_all > FILL_VAL)
            & np.isfinite(lon_all) & (lon_all > FILL_VAL)
            & np.isfinite(vza_all) & (vza_all > FILL_VAL)
            & np.isfinite(vaa_all) & (vaa_all > FILL_VAL))
    lat, lon, vza, vaa = lat_all[good], lon_all[good], vza_all[good], vaa_all[good]
    alt, sza, raa, saa = alt_all[good], sza_all[good], raa_all[good], saa_all[good]
    t_sub = t_ns_all[good]
    if lat.size == 0:
        if verbose:
            print("  footprints_feeding_tile: no valid RAD-4CH samples in "
                  "+/-%.0f s window -- nothing to classify" % window_s)
        return dict(records=[], n_inside=0, n_edge=0, n_vza_model_capped=0,
                    tile_km=tile_km, lat0=lat0, lon0=lon0, window_s=window_s)

    vza_capped = vza > length_vza_cap_deg
    n_vza_model_capped = int(vza_capped.sum())

    dist_km, bearing_rad = haversine_bearing(lat0, lon0, lat, lon)
    east_km = dist_km * np.sin(bearing_rad)
    north_km = dist_km * np.cos(bearing_rad)

    half_km = tile_km / 2.0
    w_km, l_km = footprint_axes_km(vza, altitude_km, length_vza_cap_deg=length_vza_cap_deg)
    a_km, b_km = l_km / 2.0, w_km / 2.0  # semi-major (radial/height), semi-minor (tangential/width)
    ang_rad = np.radians(-vaa)           # orientation fix: angle = -VAA, not 90-VAA

    # Rotated-ellipse axis-aligned bounding-box half-extents (exact formula;
    # sx=semi-axis along local x pre-rotation = b, sy=along local y = a).
    half_x = np.sqrt((b_km * np.cos(ang_rad)) ** 2 + (a_km * np.sin(ang_rad)) ** 2)
    half_y = np.sqrt((b_km * np.sin(ang_rad)) ** 2 + (a_km * np.cos(ang_rad)) ** 2)

    bbox_xmin, bbox_xmax = east_km - half_x, east_km + half_x
    bbox_ymin, bbox_ymax = north_km - half_y, north_km + half_y

    overlaps = ((bbox_xmax >= -half_km) & (bbox_xmin <= half_km)
                & (bbox_ymax >= -half_km) & (bbox_ymin <= half_km))
    fully_inside = (overlaps & (bbox_xmin >= -half_km) & (bbox_xmax <= half_km)
                    & (bbox_ymin >= -half_km) & (bbox_ymax <= half_km))

    records = []
    for i in np.where(overlaps)[0]:
        records.append(dict(
            lat=float(lat[i]), lon=float(lon[i]),
            east_km=float(east_km[i]), north_km=float(north_km[i]),
            vza_deg=float(vza[i]), vaa_deg=float(vaa[i]),
            width_km=float(w_km[i]), length_km=float(l_km[i]),
            angle_deg=float(-vaa[i]), t_ns=int(t_sub[i]),
            classification=("inside" if fully_inside[i] else "edge"),
            vza_model_capped=bool(vza_capped[i]),
            # Pass-through geolocation/geometry for the CF-CAM output
            # writer -- FILL_VAL (-999.0) where RAD-4CH didn't have the
            # field (see _read_optional above).
            altitude_m=float(alt[i]), sza_deg=float(sza[i]),
            raa_deg=float(raa[i]), saa_deg=float(saa[i]),
        ))

    n_inside = int(fully_inside.sum())
    n_edge = int(overlaps.sum() - n_inside)
    if verbose:
        cap_note = (" (%d of these beyond VZA %.0f deg use a length-capped "
                    "limb-geometry approximation)"
                    % (n_vza_model_capped, length_vza_cap_deg)
                    if n_vza_model_capped else "")
        print("  radiometer footprints feeding this %g km tile: %d fully "
              "inside, %d straddling the edge (need an adjacent tile too, "
              "not yet stitched) -- %d candidate samples scanned in "
              "+/-%.0f s window%s" % (tile_km, n_inside, n_edge, lat.size,
                                      window_s, cap_note))

    return dict(records=records, n_inside=n_inside, n_edge=n_edge,
                n_vza_model_capped=n_vza_model_capped, tile_km=tile_km,
                lat0=lat0, lon0=lon0, window_s=window_s)


def _ellipse_membership(x_km, y_km, center_east_km, center_north_km,
                        length_km, width_km, angle_deg):
    """Boolean mask: which (x_km, y_km) points fall inside the footprint
    ellipse at (center_east_km, center_north_km), length/width/angle_deg
    (same matplotlib Ellipse convention used throughout this module,
    angle=-VAA). Factored out 2026-10-22 from footprint_fill_fraction()'s
    inline version (kept that function's own copy untouched, to avoid any
    risk of regressing its already-validated behavior) so the CF-CAM
    output writer below can reuse the exact same geometry rather than a
    second, possibly-inconsistent transcription of it."""
    a_km, b_km = length_km / 2.0, width_km / 2.0
    ang = np.radians(angle_deg)
    dx, dy = x_km - center_east_km, y_km - center_north_km
    xl = dx * np.cos(ang) + dy * np.sin(ang)
    yl = -dx * np.sin(ang) + dy * np.cos(ang)
    return (xl / b_km) ** 2 + (yl / a_km) ** 2 <= 1.0


def radiance_and_counts_for_footprints(data, rad, pooled, footprint_info,
                                       cloud_flag=None, require_proxy_mask=True):
    """Per-footprint `cam_radiance_footprint` (RT-proxy) and camera-pixel
    counts, for every 'inside' record in footprint_info -- added
    2026-10-22 for the CF-CAM output writer, per Sebastian: "if we use
    the 1-tile code, then we can actually just add one more step ...
    where the calculated radiances are now just poured into the
    radiometer footprints that are fully within the tile (no 'edge')."
    Deliberately 'edge' records are left alone (classification=='inside'
    is the only one dev-step-1's output covers -- no multi-tile
    stitching yet, by explicit instruction: "I don't want to get into
    any multi-tile work just yet").

    cam_radiance_footprint_rt: nanmean(rad) over every (lon_land,
    lat_land) RT/BRDF grid cell whose real-geographic position falls
    inside the footprint's ellipse -- `rad` is SHDOM's per-cell radiance
    (same grid `_panel_radiance`/`_panel_thin_atm_map` plot), so this is
    literally "pour the RT radiance field into the footprint," at
    whatever resolution the RT grid itself has (~5.5 km VJ143C1-native
    cells -- coarser than a real camera pixel, a known dev-step-1
    approximation, not real PSF convolution). Units: SAME native units
    as `rad`/run_dev_step1.py's `cam_radiance_footprint`
    (W m^-2 sr^-1 nm^-1) -- the CF-CAM NetCDF writer converts to the
    spec's W m^-2 sr^-1 um^-1 (x1000) at write time, not here.

    n_camera_pixels_total/valid: counts of pooled['in_box']/['in_tile']
    camera pixels whose (x_km, y_km) position falls inside the same
    ellipse -- NOTE an approximation: `rad`'s grid cells use a
    haversine-great-circle (east_km, north_km) (matching how
    footprints_feeding_tile() itself placed the ellipses), while
    pooled['x_km']/['y_km'] use tile()'s own flat/equirectangular
    local-km convention. The two agree to a small fraction of one RT
    grid cell at this ~100 km tile scale, which is why this is treated
    as acceptable for dev-step-1 rather than reprojecting one onto the
    other -- revisit if a future, tighter accounting needs better than
    that.

    cloud_flag (added 2026-10-23, Sebastian: "the footprints in green
    could get small cloud fraction numbers for those ones that HAVE
    clouds in them"): optional boolean (or 0/1-castable) array, same
    shape as `rad`, True/1 where run_dev_step1.py's e-fold cloud
    threshold flagged that RT grid cell as a cloud candidate (see
    cloud_threshold_efold() and run_dev_step1.py's own cloud-detection
    block). When given, every 'inside' record also gets
    cam_cloud_fraction_rt: the mean of cloud_flag over the same
    ellipse-membership grid cells cam_radiance_footprint_rt is averaged
    over (None if no qualifying cell, same convention). None (default)
    leaves cam_cloud_fraction_rt as None on every record, unchanged
    behavior for any existing caller.

    cam_radiance_footprint_measured (added 2026-10-23, item C, Sebastian:
    "what should be sent to the output file is not the RT radiance...
    the 'measured' (L1X average footprint by footprint) goes into the
    footprint-level radiance output"): nanmean of pooled['cam_rad'] (the
    REAL per-pixel camera Radiance L1X carries -- same field
    bin_measured_radiance_to_grid() bins onto the RT grid for the QA
    plots) over every raw camera pixel whose (x_km, y_km) position
    (tile()'s own flat/equirectangular convention, same as
    n_camera_pixels_total/valid below -- NOT grid_east_km/north_km)
    falls inside the footprint's ellipse AND passed tile()'s own
    in_tile selection. require_proxy_mask=True (default, matching
    bin_measured_radiance_to_grid()'s own default) additionally
    restricts to pooled['cam_proxy_mask']==1 pixels when that field is
    present, for the same reason: pixels the proxy crop never actually
    overwrote still carry the original dark/placeholder L1B value, not
    a real measurement, and would silently contaminate this average.
    None (no qualifying pixel, or pooled has no 'cam_rad' at all -- a
    plain L1B, not an L1X file) rather than 0.0 or NaN, so a downstream
    writer can tell "no real measurement available" from "measured
    zero". Native units match pooled['cam_rad'] itself (same convention
    cam_radiance_footprint_rt uses for `rad`).

    Mutates and returns footprint_info; adds cam_radiance_footprint_rt,
    cam_radiance_footprint_n_cells, cam_radiance_footprint_measured,
    cam_radiance_footprint_measured_n_pixels, n_camera_pixels_total,
    n_camera_pixels_valid, cam_cloud_fraction_rt to every 'inside'
    record (left as None/0 for 'edge' records, which this dev-step-1
    writer skips downstream)."""
    lon_land, lat_land = data["lon_land"], data["lat_land"]
    lat0, lon0 = pooled["lat0"], pooled["lon0"]
    grid_dist_km, grid_bearing_rad = haversine_bearing(lat0, lon0, lat_land, lon_land)
    grid_east_km = grid_dist_km * np.sin(grid_bearing_rad)
    grid_north_km = grid_dist_km * np.cos(grid_bearing_rad)

    x_box, y_box = pooled["x_km"], pooled["y_km"]
    in_box, in_tile = pooled["in_box"], pooled["in_tile"]
    cam_rad = pooled.get("cam_rad")
    cam_proxy_mask = pooled.get("cam_proxy_mask")

    for rec in footprint_info.get("records", []):
        if rec["classification"] != "inside":
            rec["cam_radiance_footprint_rt"] = None
            rec["cam_radiance_footprint_n_cells"] = 0
            rec["cam_radiance_footprint_measured"] = None
            rec["cam_radiance_footprint_measured_n_pixels"] = 0
            rec["n_camera_pixels_total"] = 0
            rec["n_camera_pixels_valid"] = 0
            rec["cam_cloud_fraction_rt"] = None
            continue
        in_ellipse_grid = _ellipse_membership(
            grid_east_km, grid_north_km, rec["east_km"], rec["north_km"],
            rec["length_km"], rec["width_km"], rec["angle_deg"])
        cell_vals = rad[in_ellipse_grid]
        cell_vals = cell_vals[np.isfinite(cell_vals)]
        rec["cam_radiance_footprint_rt"] = (float(np.mean(cell_vals))
                                            if cell_vals.size else None)
        rec["cam_radiance_footprint_n_cells"] = int(cell_vals.size)

        if cloud_flag is not None:
            cf_vals = np.asarray(cloud_flag, dtype=float)[in_ellipse_grid]
            cf_vals = cf_vals[np.isfinite(cf_vals)]
            rec["cam_cloud_fraction_rt"] = (float(np.mean(cf_vals))
                                            if cf_vals.size else None)
        else:
            rec["cam_cloud_fraction_rt"] = None

        in_ellipse_box = _ellipse_membership(
            x_box, y_box, rec["east_km"], rec["north_km"],
            rec["length_km"], rec["width_km"], rec["angle_deg"])
        rec["n_camera_pixels_total"] = int((in_ellipse_box & in_box).sum())
        rec["n_camera_pixels_valid"] = int((in_ellipse_box & in_tile).sum())

        if cam_rad is not None:
            sel_meas = in_ellipse_box & in_tile
            if require_proxy_mask and cam_proxy_mask is not None:
                sel_meas = sel_meas & (cam_proxy_mask > 0)
            meas_vals = cam_rad[sel_meas]
            meas_vals = meas_vals[np.isfinite(meas_vals)]
            rec["cam_radiance_footprint_measured"] = (
                float(np.mean(meas_vals)) if meas_vals.size else None)
            rec["cam_radiance_footprint_measured_n_pixels"] = int(meas_vals.size)
        else:
            rec["cam_radiance_footprint_measured"] = None
            rec["cam_radiance_footprint_measured_n_pixels"] = 0

    return footprint_info


def bin_measured_radiance_to_grid(data, pooled, require_proxy_mask=True,
                                  verbose=True):
    """Bin pooled's per-pixel "measured" camera Radiance
    (pooled['cam_rad'] -- an apps/l2/build_l1x.py "L1X"-only field, see
    tile()'s MEASURED_RADIANCE_FIELDS / pool_stripe_pixels' extra_fields)
    onto the SAME (lon_land, lat_land) RT/BRDF grid rad/refl_app/
    refl_noatm live on -- one mean per grid cell -- so the RT-calculated
    and real-measured radiance fields can be differenced/scatter-plotted
    cell-for-cell. Added 2026-10-22, Sebastian: "in addition to the RT-
    calculated TOA radiances, we also see the 'measured' radiances, as
    well as a difference ... a scatter plot of 'RT' vs. 'measured'."

    Each measured pixel is assigned to its NEAREST grid cell by a
    KDTree lookup in LOCAL KM space -- grid cell centers are converted
    to (east_km, north_km) via haversine_bearing() exactly as
    radiance_and_counts_for_footprints() already does (so this carries
    the exact same, already-documented, small mismatch against pooled's
    own flat/equirectangular x_km/y_km: a small fraction of one ~5.5 km
    RT cell at this tile scale -- see that function's docstring).

    require_proxy_mask=True (default) additionally restricts to
    pooled['cam_proxy_mask']==1 pixels, when that field is present --
    i.e. pixels build_l1x.py actually overwrote from the MODIS/VIIRS
    proxy scene. Pixels OUTSIDE that mask (other stripe components, or
    nadir-stripe pixels the downloaded proxy crop never reached) still
    carry the ORIGINAL dark/placeholder L1B value, not a real
    measurement -- binning those in would silently contaminate the
    "measured" comparison with near-zero noise that has nothing to do
    with the scene.

    Returns (rad_meas, n_px_per_cell): both shaped like data["lon_land"]
    / rad; NaN (rad_meas) / 0 (n_px_per_cell) in any cell no qualifying
    pixel landed in. Returns (None, None) -- with a printed explanation,
    not an error -- if pooled has no 'cam_rad' at all (a plain L1B, not
    an L1X file): callers should treat this as "measured comparison not
    available for this input" and skip it.
    """
    cam_rad = pooled.get("cam_rad")
    if cam_rad is None:
        if verbose:
            print("  bin_measured_radiance_to_grid: camera L1B has no "
                 "'Radiance' field pooled (not an L1X file?) -- measured-"
                 "radiance comparison skipped")
        return None, None

    sel = pooled["in_tile"].copy()
    cam_proxy_mask = pooled.get("cam_proxy_mask")
    if require_proxy_mask:
        if cam_proxy_mask is not None:
            sel = sel & (cam_proxy_mask > 0)
        elif verbose:
            print("  bin_measured_radiance_to_grid: camera L1B has "
                 "'Radiance' but no 'L1X_Proxy_Mask' field -- cannot "
                 "restrict to real proxy pixels, using ALL in-tile pixels "
                 "as-is (may include dark/placeholder L1B values)")

    n_sel = int(sel.sum())
    if n_sel == 0:
        if verbose:
            print("  bin_measured_radiance_to_grid: 0 qualifying camera "
                 "pixels in the tile box -- measured-radiance comparison "
                 "skipped")
        return None, None

    from scipy.spatial import KDTree

    lon_land, lat_land = data["lon_land"], data["lat_land"]
    lat0, lon0 = pooled["lat0"], pooled["lon0"]
    grid_dist_km, grid_bearing_rad = haversine_bearing(lat0, lon0, lat_land, lon_land)
    grid_east_km = grid_dist_km * np.sin(grid_bearing_rad)
    grid_north_km = grid_dist_km * np.cos(grid_bearing_rad)

    grid_shape = grid_east_km.shape
    tree = KDTree(np.column_stack([grid_east_km.ravel(), grid_north_km.ravel()]))
    px_pts = np.column_stack([pooled["x_km"][sel], pooled["y_km"][sel]])
    _, flat_idx = tree.query(px_pts)

    n_cells = grid_east_km.size
    rad_sum = np.zeros(n_cells)
    rad_n = np.zeros(n_cells, dtype=np.int64)
    np.add.at(rad_sum, flat_idx, cam_rad[sel])
    np.add.at(rad_n, flat_idx, 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        rad_mean = np.where(rad_n > 0, rad_sum / np.maximum(rad_n, 1), np.nan)
    rad_meas = rad_mean.reshape(grid_shape)
    n_px_per_cell = rad_n.reshape(grid_shape)

    if verbose:
        n_cells_hit = int((n_px_per_cell > 0).sum())
        print("  bin_measured_radiance_to_grid: %d qualifying camera pixel(s) "
             "binned into %d/%d RT grid cell(s) (mean %.1f px/cell where hit)"
             % (n_sel, n_cells_hit, n_cells,
                n_sel / n_cells_hit if n_cells_hit else 0.0))

    return rad_meas, n_px_per_cell


def cloud_threshold_efold(diff, bins=60, scale=np.e):
    """Adaptive reflectance-difference cloud threshold -- Libera ATBD
    Camera L2 CDR draft, Figure 6 (p.8): "The adaptive thresholding
    algorithm determines a threshold as the reflectance where the PDF
    has fallen off to 1/e of its peak value on the RHS of the peak, and
    detects all pixels exceeding that threshold as cloudy. Versions of
    this algorithm use 2 or 3 instead of e, with little difference in
    skill." Same technique as Hong's l2_cam/src/cam_l2/cld.py
    (cal_threshold_efold/_std/_gauss: rad_diff = rad_cam - rad_rtm,
    cld_msk = rad_diff > threshold) -- reimplemented here independently
    (Sebastian, 2026-10-23: "replicate something similar rather than
    simply importing"), not a call into l2_cam.

    `diff` is a "measured - RT" array (reflectance or radiance, any
    units -- same convention as run_dev_step1.py's measured_vs_rt dict
    and ATBD's R_diff = R_obs - R_sfc), already restricted to the cells
    worth histogramming (finite, both sides present). PDF-izes it
    (np.histogram(..., density=True), ATBD Figure 6's y-axis), finds the
    peak (x_max, f_max), then walks the PDF rightward (toward brighter/
    more-cloud-like values -- `hist[n_max:]`, i.e. FROM the peak TOWARD
    the right tail, not literally "left") until it falls to f_max/scale
    -- that reflectance is x_e, and `threshold = x_e`: cells with
    diff > threshold are the cloud candidates.

    Returns a dict (x_max, f_max, x_e, f_e, threshold, scale, bin_edges,
    hist) -- bin_edges/hist (the PDF itself) are included so a caller
    can redraw the exact same histogram the threshold was computed from,
    rather than recomputing it with different binning. Returns None if
    `diff` has fewer than 2 finite values (not enough to histogram)."""
    diff = np.asarray(diff, dtype=np.float64).ravel()
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return None

    hist, bin_edges = np.histogram(diff, bins=bins, density=True)
    bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    n_max = int(np.argmax(hist))
    x_max, f_max = float(bin_center[n_max]), float(hist[n_max])

    f_target = f_max / scale
    hist_rhs = hist[n_max:]
    n_e = int(np.argmin(np.abs(hist_rhs - f_target)))
    x_e, f_e = float(bin_center[n_max:][n_e]), float(hist_rhs[n_e])

    return dict(x_max=x_max, f_max=f_max, x_e=x_e, f_e=f_e,
               threshold=x_e, scale=float(scale),
               bin_edges=bin_edges, hist=hist)


def solar_glint_offset_km(sza_deg, saa_deg, altitude_km):
    """Expected ocean sun-glint ('specular point') location, as an
    (east_km, north_km) offset from the subsatellite point -- flat-
    Earth, flat-ocean approximation (the same geometry behind MODIS/
    VIIRS sunglint flagging). For a satellite looking straight down at
    its subsatellite point, the specular point lies at horizontal
    distance r = altitude * tan(SZA), in the solar azimuth's own
    direction (equivalently: as seen FROM the specular point, the
    satellite appears at the same zenith angle as the sun, exactly
    180 deg opposite its azimuth -- standard mirror-reflection
    geometry). Requested 2026-10 by Sebastian to sanity-check whether
    an observed bright ocean pixel could plausibly be real sun glint:
    at typical mid-latitude SZA this point is many hundreds of km from
    a near-nadir-viewing tile like this pipeline's, so a bright pixel
    found THERE is essentially never glint.
    """
    r_km = altitude_km * np.tan(np.radians(sza_deg))
    saa_rad = np.radians(saa_deg)
    east_km = r_km * np.sin(saa_rad)
    north_km = r_km * np.cos(saa_rad)
    return east_km, north_km, r_km


def _mark_glint_direction(ax, east_km, north_km, r_km, half_km, color="#d9480f",
                           to_data_xy=None, transform_kw=None):
    """Mark the expected sun-glint point on a panel: a direct cross if
    it happens to fall within the plotted tile, otherwise a short arrow
    pointing toward it from the panel's edge (labeled with distance),
    since the real glint spot is usually far outside a near-nadir tile.
    Uses axes-fraction placement for the off-tile case so this works
    identically on a plain Cartesian (east/north, km) axes and on a
    north-up PlateCarree GeoAxes, whose on-screen directions match the
    same (east=right, north=up) convention.

    `to_data_xy`, if given, maps (east_km, north_km) -> this axes' own
    data coordinates for the direct-marker case (e.g. km offsets ->
    lon/lat for the map panels); defaults to identity (km offsets are
    already the data coordinates, as on the local-km panels).
    `transform_kw`, if given, is passed as transform=... to that plot
    call (e.g. ccrs.PlateCarree() for a cartopy GeoAxes)."""
    if abs(east_km) <= half_km and abs(north_km) <= half_km:
        x, y = to_data_xy(east_km, north_km) if to_data_xy else (east_km, north_km)
        kw = dict(transform=transform_kw) if transform_kw is not None else {}
        ax.plot(x, y, "x", color=color, ms=11, mew=2.2, zorder=7, **kw)
        ax.annotate("expected glint", (x, y), color=color, fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points", zorder=7)
        return
    bearing_rad = np.arctan2(east_km, north_km)
    ux, uy = np.sin(bearing_rad), np.cos(bearing_rad)
    tx = (0.5 / abs(ux)) if abs(ux) > 1e-9 else np.inf
    ty = (0.5 / abs(uy)) if abs(uy) > 1e-9 else np.inf
    t = min(tx, ty) * 0.90
    tip = (0.5 + t * ux, 0.5 + t * uy)
    base = (0.5 + (t - 0.12) * ux, 0.5 + (t - 0.12) * uy)
    ax.annotate("", xy=tip, xytext=base, xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=7)
    ax.annotate("expected glint\n~%.0f km away" % r_km, xy=tip, xycoords="axes fraction",
                color=color, fontsize=6.5, ha="center",
                va=("bottom" if uy >= 0 else "top"), zorder=7)


# ── phase 1: chop ───────────────────────────────────────────────────────

def chop(rad_l1b, cam_l1b, seed_lat, seed_lon, window_s, scan_pad_s=400.0,
         max_match_vza_deg=None, verbose=True):
    """Find the RAD-4CH sample nearest (seed_lat, seed_lon), and the
    camera-frame index range within +/-window_s of it. Returns a dict
    with the matched sample's own lat/lon/time/SZA (as a direct
    cross-check against the seed values this site preset was found
    with) plus (first_frame, max_frames) for pool_stripe_pixels.

    scan_camera_frame_times' own per-frame geo_report pre-pass costs
    real wall-clock time per frame (measured ~0.06 s/frame on a 2026
    DITL2-size CAM L1B granule) -- cheap for a handful of frames, but
    NOT cheap integrated over an entire ~2000-frame granule (~2 min).
    Since we already know the target time from the RAD-4CH match, there
    is no need to scan the whole granule to find the nearby camera
    frames: CAMERA_TIME and radiometer_time share one absolute
    nanosecond-since-Unix-epoch clock (confirmed this session), so a
    plain searchsorted on the (cheap, whole-array) CAMERA_TIME coordinate
    gives a good first guess, and only a +/-scan_pad_s window around it
    is ever handed to scan_camera_frame_times. scan_pad_s should stay
    comfortably larger than --window-s (default pad is 4x the default
    window) -- if it isn't, the frame range returned below would be
    silently clipped to the scanned edge; this is checked for and
    reported loudly, not silently trusted.
    """
    import xarray as xr

    ds = xr.open_dataset(rad_l1b)
    lat = np.asarray(ds["Latitude"].values, dtype=np.float64)
    lon = np.asarray(ds["Longitude"].values, dtype=np.float64)
    sza = np.asarray(ds["Solar_Zenith_Surface"].values, dtype=np.float64)
    t_ns = np.asarray(ds["radiometer_time"].values, dtype=np.int64)

    good = np.isfinite(lat) & np.isfinite(lon) & (lat > FILL_VAL) & (lon > FILL_VAL)

    # CONFIRMED (Sebastian, 2026-10, kyushu zero-pixel investigation):
    # chop() used to match on the RADIOMETER's ground lat/lon alone, with no
    # constraint on the radiometer's own viewing angle. RAD-4CH, like CERES,
    # sweeps telescope elevation across a wide swath, so "closest ground
    # sample to the seed" can be an extremely oblique view -- confirmed for
    # the 'kyushu' site: Viewing_Zenith_Surface=86.77 deg at the match.
    # By design the camera co-sweeps the radiometer's full elevation range
    # (Sebastian) -- camera pixel count per radiometer footprint stays near
    # a constant ~500-700 out to ~VZA 60 deg, then at a well-defined VZA
    # threshold (~63-64 deg) does ONE of two things depending on which side
    # of the swath: explodes to 1000s of pixels (the other side, out past
    # ~80 deg -- extreme near-limb footprint distortion, not a bug) or
    # collapses to exactly zero (the FSW-bug-affected side). 86.77 deg is
    # well past that threshold on the collapsed side here -- the camera
    # simply has NO pixels there at all, which is the entire explanation
    # for the zero-pooled-pixels failure (nothing to do with where the
    # camera's own near-nadir track happened to be during this pass -- that
    # was a red herring). --max-match-vza-deg restricts the match to
    # samples within a VZA where the camera reliably has coverage on
    # EITHER side (comfortably below the ~63-64 deg cutoff -- Sebastian
    # suggests ~55 deg), instead of the nearest ground sample at any angle.
    vza_field = None
    for _candidate in ("Viewing_Zenith_Surface",):
        if _candidate in ds:
            vza_field = np.asarray(ds[_candidate].values, dtype=np.float64)
            break
    if max_match_vza_deg is not None:
        if vza_field is None:
            raise SystemExit(
                "chop: --max-match-vza-deg given but RAD-4CH file has no "
                "Viewing_Zenith_Surface field to filter on")
        good = good & (vza_field <= max_match_vza_deg)
        if not good.any():
            raise SystemExit(
                "chop: no RAD-4CH sample within %g deg of nadir anywhere in "
                "this granule -- raise --max-match-vza-deg or pick a "
                "different granule/site" % max_match_vza_deg)

    dist_km, _ = haversine_bearing(seed_lat, seed_lon, lat[good], lon[good])
    idx_good = np.where(good)[0]
    i_best = idx_good[np.argmin(dist_km)]

    match = dict(
        lat=float(lat[i_best]), lon=float(lon[i_best]),
        sza=float(sza[i_best]), t_ns=int(t_ns[i_best]),
        dist_to_seed_km=float(dist_km.min()),
    )
    if vza_field is not None:
        match["match_vza"] = float(vza_field[i_best])
    if verbose:
        print("  matched RAD-4CH sample: lat=%.4f lon=%.4f (%.2f km from "
              "requested site), SZA=%.3f deg, time=%s UTC%s"
              % (match["lat"], match["lon"], match["dist_to_seed_km"],
                 match["sza"],
                 datetime.datetime.utcfromtimestamp(match["t_ns"] / 1e9),
                 ("" if "match_vza" not in match else
                  ", radiometer VZA=%.2f deg" % match["match_vza"])))
        if max_match_vza_deg is not None:
            print("  (match restricted to RAD-4CH samples with VZA <= %g deg)"
                  % max_match_vza_deg)
        # Report whatever other viewing-geometry fields the RAD-4CH file has
        # for the matched sample, for a quick sanity check either way.
        try:
            var_names = list(ds.data_vars)
        except AttributeError:
            var_names = []  # e.g. test_harness's mocked dataset -- skip, not fatal
        vza_like = [v for v in var_names
                    if any(s in v.lower() for s in
                           ("zenith", "view", "scan", "subsat", "sub_sat",
                            "nadir", "azimuth"))
                    and v not in ("Solar_Zenith_Surface",)]
        if var_names:
            print("  RAD-4CH viewing-geometry fields present: %s"
                  % (vza_like or "none found"))
            for v in vza_like:
                try:
                    val = float(np.asarray(ds[v].values, dtype=np.float64)[i_best])
                    print("    %s at matched sample = %.3f" % (v, val))
                except Exception as exc:
                    print("    %s at matched sample: could not read (%r)" % (v, exc))

    cam_ds = est.open_granule(cam_l1b)
    n_cam_all = int(cam_ds.sizes["CAMERA_TIME"])
    cam_time_all = np.asarray(cam_ds["CAMERA_TIME"].values, dtype=np.int64)
    i_approx = int(np.searchsorted(cam_time_all, match["t_ns"]))
    cadence_s = float(np.median(np.diff(cam_time_all[max(0, i_approx - 20):i_approx + 20]))) / 1e9 \
        if n_cam_all > 1 else 5.5
    pad_frames = max(10, int(scan_pad_s / max(cadence_s, 0.1)))
    f0 = max(0, i_approx - pad_frames)
    f1 = min(n_cam_all, i_approx + pad_frames)
    if verbose:
        print("  camera cadence ~%.2f s/frame near the match; scanning frames "
              "[%d, %d) of %d total (+/-%g s pad) instead of the whole granule"
              % (cadence_s, f0, f1, n_cam_all, scan_pad_s))

    scan_idx, scan_time_ns = est.scan_camera_frame_times(
        cam_l1b, first_frame=f0, max_frames=f1 - f0)
    if scan_idx.size == 0:
        raise SystemExit(
            "chop: no geolocated camera frames found in [%d, %d) around the "
            "match -- try a larger --scan-pad-s" % (f0, f1))

    first_frame, max_frames = est.narrow_frame_range(
        scan_idx, scan_time_ns, t_ref=match["t_ns"],
        win_t0=-window_s, win_t1=window_s)
    if first_frame == f0 or (first_frame + max_frames) == f1:
        print("  WARNING: the +/-%g s camera-frame window touches the edge of "
              "the +/-%g s scan pad -- widen --scan-pad-s and re-run to make "
              "sure no usable frames were missed." % (window_s, scan_pad_s))
    match["first_frame"] = first_frame
    match["max_frames"] = max_frames
    if verbose:
        print("  camera frame window: first_frame=%d, max_frames=%d "
              "(+/-%g s around the match)" % (first_frame, max_frames, window_s))
    return match


# ── phase 2: tile ───────────────────────────────────────────────────────

def _fix_find_nearest_miscategorized_cells(data, verbose=False):
    """Downstream workaround (Sebastian, 2026-10) for a confirmed bug in
    find_nearest (er3t/util/util.py, vendored in l2_cam/software/er3t --
    reported to Hong, not fixed here): it compares a candidate nearest-
    neighbor SOURCE index against indices_xy.size, the number of QUERY
    points, instead of data.size, the number of SOURCE points, and
    discards an otherwise-valid in-range match whenever those two numbers
    happen to coincide. That bogus fill-value (-999) then gets picked up
    by gen_brdf_shdom's (~logic_ocean & ~logic_land) catch-all and
    mislabeled as ice/snow.

    This does NOT touch l2_cam's vendored er3t code. It re-derives the
    TRUE nearest-neighbor mask value for every tile cell gen_brdf_shdom
    flagged as ice (flag==4), using the same distance-limit test
    find_nearest itself uses but without the buggy disjunct, and patches
    data['brdf'] in place -- reusing the exact same windspeed/VJ143C1
    interpolation calls gen_brdf_shdom itself makes -- for any cell that
    should really be ocean or land. A cell that's genuinely too far from
    any source pixel (a real data gap) is left as ice, unchanged.

    Returns the number of cells patched (0 if nothing needed fixing).
    """
    import er3t.util
    from scipy.spatial import KDTree

    flag = data["brdf"][..., 6]
    logic_ice = (flag == 4.0)
    if not logic_ice.any():
        return 0

    lon_out, lat_out = data["lon_land"], data["lat_land"]
    sfc_x = np.asarray(data["lon_sfc"], dtype=np.float64).ravel()
    sfc_y = np.asarray(data["lat_sfc"], dtype=np.float64).ravel()
    sfc_v = np.asarray(data["sfc"]).ravel()

    tree = KDTree(np.transpose(np.vstack((sfc_x, sfc_y))))
    dist, src_idx = tree.query(
        np.transpose(np.vstack((lon_out.ravel(), lat_out.ravel()))))
    dist = dist.reshape(lon_out.shape)
    src_idx = src_idx.reshape(lon_out.shape)

    dx = np.zeros_like(lon_out, dtype=np.float32)
    dy = np.zeros_like(lat_out, dtype=np.float32)
    dx[1:, ...] = lon_out[1:, ...] - lon_out[:-1, ...]
    dx[0, ...] = dx[1, ...]
    dy[..., 1:] = lat_out[..., 1:] - lat_out[..., :-1]
    dy[..., 0] = dy[..., 1]
    dist_limit = np.sqrt((dx * 5) ** 2 + (dy * 5) ** 2)

    true_mask = sfc_v[src_idx].astype(np.int64)
    true_mask[dist > dist_limit] = -999  # genuinely too far -- leave as ice

    to_fix = logic_ice & (true_mask != 4) & (true_mask != -999)
    if not to_fix.any():
        return 0

    to_ocean = to_fix & (true_mask == -1)
    to_land = to_fix & (true_mask == 0)
    n_fixed = 0

    if to_ocean.any():
        ws_full = er3t.util.find_nearest(
            data["lon_ocean"], data["lat_ocean"], data["windspeed_ocean"],
            lon_out, lat_out, Ngrid_limit=5, fill_value=0.0)
        for idx in zip(*np.where(to_ocean)):
            data["brdf"][idx + (0,)] = float(ws_full[idx])
            data["brdf"][idx + (1,)] = 0.01
            # slots 2/3/4 unused for ocean -- match gen_brdf_shdom's own
            # convention for a fresh ocean cell (left at the -999 fill
            # value, not the stale ice-kernel numbers this cell had before)
            data["brdf"][idx + (2,)] = -999.0
            data["brdf"][idx + (3,)] = -999.0
            data["brdf"][idx + (4,)] = -999.0
            data["brdf"][idx + (6,)] = 1.0
            if verbose:
                print("    fixed grid %s: ice -> ocean (windspeed=%.2f m/s)"
                      % (idx, data["brdf"][idx + (0,)]))
        n_fixed += int(to_ocean.sum())

    if to_land.any():
        logic_valid_src = ((data["fiso_land"] >= 0.0) & (data["fgeo_land"] >= 0.0)
                            & (data["fvol_land"] >= 0.0))
        if logic_valid_src.any():
            lon_l, lat_l = data["lon_land"], data["lat_land"]
            fiso_full = er3t.util.find_nearest(
                lon_l[logic_valid_src], lat_l[logic_valid_src],
                data["fiso_land"][logic_valid_src], lon_out, lat_out,
                Ngrid_limit=50, fill_value=0.0)
            fgeo_full = er3t.util.find_nearest(
                lon_l[logic_valid_src], lat_l[logic_valid_src],
                data["fgeo_land"][logic_valid_src], lon_out, lat_out,
                Ngrid_limit=50, fill_value=0.0)
            fvol_full = er3t.util.find_nearest(
                lon_l[logic_valid_src], lat_l[logic_valid_src],
                data["fvol_land"][logic_valid_src], lon_out, lat_out,
                Ngrid_limit=50, fill_value=0.0)
            for idx in zip(*np.where(to_land)):
                data["brdf"][idx + (0,)] = float(fiso_full[idx])
                data["brdf"][idx + (1,)] = float(fgeo_full[idx])
                data["brdf"][idx + (2,)] = float(fvol_full[idx])
                data["brdf"][idx + (3,)] = -999.0
                data["brdf"][idx + (4,)] = -999.0
                data["brdf"][idx + (6,)] = 3.0
                if verbose:
                    print("    fixed grid %s: ice -> land (fiso=%.3f fgeo=%.3f "
                          "fvol=%.3f)" % (idx, fiso_full[idx], fgeo_full[idx],
                                          fvol_full[idx]))
            n_fixed += int(to_land.sum())
        else:
            print("    WARNING: %d cell(s) should be land per the corrected "
                  "mask, but no valid VJ143C1 land BRDF exists anywhere in "
                  "this domain to interpolate from -- left as ice (flag=4), "
                  "flagging for manual review." % int(to_land.sum()))

    return n_fixed


def _era5_windspeed_field(era5_path, lon_grid, lat_grid, time_utc):
    """Real ERA5 10 m windspeed sampled onto an arbitrary (lon_grid,
    lat_grid) array -- same shape out as in, nearest-neighbor in space
    (0.25 deg ERA5 grid) and time (hourly, same approach as
    pick_tile_overview.py's sample_era5_at()).

    Added 2026-10-22 so tile()'s ocean BRDF can use the actual ERA5 file
    from the input manifest instead of the hardcoded MERRA2 file baked
    into l2_cam's sample/input.yml -- see docs/L1B_TO_L2_ROADMAP.md.

    Fixed 2026-10-22 (same day, before first real run -- Sebastian caught
    it): the original version of this helper sampled ONE point at the
    tile center and tile() broadcast that single scalar over every ocean
    cell in the tile, silently discarding real spatial structure in the
    wind field. read_brdf_parms()'s own data['windspeed_ocean'] is a
    genuine 2D field on its own data['lon_ocean']/data['lat_ocean'] grid
    (cropped from the MERRA2 source at ITS native resolution, not a
    single value) -- cld.py:270 -- so the ERA5 replacement needs to
    preserve that, not flatten it. This version takes the caller's own
    lon/lat grid and returns a same-shaped windspeed field, one ERA5
    nearest-neighbor lookup per grid cell. A 100 km tile can span several
    ERA5 grid cells (0.25 deg ~= 25-28 km at mid-latitudes), so this is
    not a cosmetic difference."""
    import netCDF4
    ds = netCDF4.Dataset(era5_path, "r")
    era5_lat = np.asarray(ds.variables["latitude"][:])
    era5_lon = np.asarray(ds.variables["longitude"][:])
    # ERA5's valid_time carries real CF units ('seconds since 1970-01-01',
    # confirmed directly against this file's own attrs) -- plain int64
    # seconds-since-epoch arithmetic below, no decode_cf double-application
    # risk (see pick_tile_overview.py's decode_radiometer_time() fix).
    era5_time_s = np.asarray(ds.variables["valid_time"][:], dtype=np.int64)
    u10 = np.asarray(ds.variables["u10"][:])
    v10 = np.asarray(ds.variables["v10"][:])
    ds.close()

    target_s = (np.datetime64(time_utc) - np.datetime64("1970-01-01T00:00:00")) \
        / np.timedelta64(1, "s")
    i_time = int(np.argmin(np.abs(era5_time_s - target_s)))

    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    lon_grid_0_360 = np.mod(lon_grid, 360.0)
    era5_lon_0_360 = np.mod(era5_lon, 360.0)

    # Nearest-neighbor index per grid point. The caller's grid here is the
    # tile-cropped ocean-BRDF grid (a handful of cells), not the full ERA5
    # domain, so a direct broadcast against the full ERA5 lat/lon axes is
    # cheap and makes no assumption about uniform grid spacing.
    i_lat = np.argmin(np.abs(lat_grid.ravel()[:, None] - era5_lat[None, :]), axis=1)
    i_lon = np.argmin(np.abs(lon_grid_0_360.ravel()[:, None] - era5_lon_0_360[None, :]), axis=1)

    u = u10[i_time][i_lat, i_lon]
    v = v10[i_time][i_lat, i_lon]
    return np.hypot(u, v).reshape(lon_grid.shape)


# "Measured" radiance fields (added 2026-10-22): present only in an
# apps/l2/build_l1x.py "L1X" file, absent (silently) in a plain,
# real/dark camera L1B -- see pool_stripe_pixels' extra_fields and
# bin_measured_radiance_to_grid() below.
MEASURED_RADIANCE_FIELDS = {"cam_rad": "Radiance", "cam_proxy_mask": "L1X_Proxy_Mask"}


# VIIRS/JPSS-1 (NOAA-20), which VJ143C1 land BRDF comes from, launched
# 2017-11-18 -- no real VJ143C1 data can exist for any earlier date, full
# stop, regardless of which sample file happens to be on disk.
_VIIRS_JPSS1_LAUNCH_DATE = datetime.date(2017, 11, 18)

# How far a fixed ancillary sample's own embedded date may sit from the
# real overpass/proxy date before we call it a mismatch rather than normal
# day-to-day/short-term variation. Land BRDF and ocean wind both vary on
# timescales well under a year (seasonal vegetation/ice state, synoptic
# wind), so even this is generous, not a precision requirement.
_ANCILLARY_DATE_MISMATCH_DAYS = 30


def _check_ancillary_date_compatibility(cfg, overpass_dt, brdf_land_overridden,
                                        era5_overridden, verbose=True):
    """Sebastian, 2026-10-23, after the first real run against a 2014 proxy
    date gave near-zero measured-vs-RT correlation on an otherwise-working
    pipeline: "the code does not throw an error message... I wonder what
    surface reflectance it grabbed... the dates of ALL the proxy data have
    to be mutually compatible. Can we build in a check somewhere that
    ensures this?"

    l2_cam's land BRDF (VJ143C1) and ocean windspeed (MERRA2) inputs are
    both FIXED SAMPLE FILES baked into l2_cam/data/sample/input.yml (one
    specific date each, not fetched per-date) -- unless brdf_land_path=/
    era5_path= override them, EVERY run uses those same two fixed
    snapshots regardless of which date is actually being simulated. That
    silently feeds a real overpass a fixed-sample surface reflectance that
    may be from an unrelated year/season, with nothing printed to say so.

    Revised 2026-10-23 (same day, Sebastian: "I just want to make the good
    case work again... gradually fix the remaining date mismatch"): there
    is no real per-date VIIRS BRDF fetch wired up AT ALL yet, so literally
    EVERY run (any date) hits SOME mismatch against the fixed 2021-05-18
    sample -- hard-stopping on all of them would make
    allow_ancillary_date_mismatch a flag you'd need on every single
    invocation, which defeats the point. So severity is now split:
    'fatal' only for the categorically-impossible case (overpass predates
    VIIRS/JPSS-1's own launch, so no real match could EVER exist, in
    principle, not just today); everything else that merely doesn't match
    TODAY's fixed sample is 'warning' -- real, worth seeing every time, but
    not blocking, since blocking it would currently block every run.

    Returns a list of {'severity': 'fatal'|'warning', 'message': str}
    dicts (empty if everything's within tolerance or already overridden
    with real data). Does not raise -- the caller decides what to do with
    'fatal' entries."""
    import re
    issues = []

    if not brdf_land_overridden:
        land_fname = os.path.basename(cfg["input"]["brdf"]["f_land"]["fname"])
        m = re.search(r"\.A(\d{4})(\d{3})\.", land_fname)
        if m:
            year, doy = int(m.group(1)), int(m.group(2))
            brdf_date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
            overpass_date = overpass_dt.date()
            if overpass_date < _VIIRS_JPSS1_LAUNCH_DATE:
                issues.append({"severity": "fatal", "message":
                    "Land BRDF source %s is dated %s, but the overpass/proxy "
                    "date is %s -- VIIRS/JPSS-1 (NOAA-20), which VJ143C1 comes "
                    "from, did not launch until %s, so NO real date-matched "
                    "VJ143C1 BRDF can EVER exist for this date. l2_cam's fixed "
                    "sample is being silently substituted regardless; do not "
                    "trust land-surface reflectance for this run."
                    % (land_fname, brdf_date.isoformat(), overpass_date.isoformat(),
                      _VIIRS_JPSS1_LAUNCH_DATE.isoformat())})
            else:
                delta_days = abs((overpass_date - brdf_date).days)
                if delta_days > _ANCILLARY_DATE_MISMATCH_DAYS:
                    issues.append({"severity": "warning", "message":
                        "Land BRDF source %s is dated %s, %d day(s) from the "
                        "overpass/proxy date %s -- l2_cam has no per-date "
                        "VIIRS BRDF fetch wired up, so its fixed sample is "
                        "used regardless of the requested date. Land-surface "
                        "reflectance for this run reflects %s conditions, not "
                        "%s; treat RT-vs-measured disagreement over land with "
                        "that in mind."
                        % (land_fname, brdf_date.isoformat(), delta_days,
                          overpass_date.isoformat(), brdf_date.isoformat(),
                          overpass_date.isoformat())})
        elif verbose:
            print("  (ancillary-date check: could not parse a date out of "
                 "land BRDF filename %s -- skipping that check)" % land_fname)

    if not era5_overridden:
        ocean_fname = os.path.basename(cfg["input"]["brdf"]["f_ocean"]["fname"])
        m = re.search(r"\.(\d{4})(\d{2})(\d{2})\.nc4?$", ocean_fname)
        if m:
            ocean_date = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            delta_days = abs((overpass_dt.date() - ocean_date).days)
            if delta_days > _ANCILLARY_DATE_MISMATCH_DAYS:
                issues.append({"severity": "warning", "message":
                    "Ocean windspeed source %s is dated %s, %d day(s) from "
                    "the overpass/proxy date %s -- no era5_path override was "
                    "given, so l2_cam's fixed MERRA2 sample is used as-is. "
                    "Real synoptic wind speed on the actual date can differ "
                    "a lot from this; pass --era5 (already the default in "
                    "run_dev_step1.py) to use real date-matched wind instead."
                    % (ocean_fname, ocean_date.isoformat(), delta_days,
                      overpass_dt.date().isoformat())})
        elif verbose:
            print("  (ancillary-date check: could not parse a date out of "
                 "ocean windspeed filename %s -- skipping that check)" % ocean_fname)

    return issues


def tile(cam_l1b, match, tile_km, dx_km, dy_km, l2_cam_root,
         stride=1, max_vza_deg=10.0, components=("nadir",), verbose=True,
         era5_path=None, margin_cells=0, brdf_land_path=None,
         allow_ancillary_date_mismatch=False):
    """Pool camera pixels in the time window, select the ones within the
    tile box AND within max_vza_deg of nadir, reduce to tile-mean
    SZA/SAA/VZA/VAA, and crop Hong's land/ocean mask + BRDF/wind
    ancillary data onto the tile extent. Returns (data, pooled) where
    `data` is Hong-cld.py-shaped (ready for gen_brdf_shdom/run) and
    `pooled` carries the raw per-pixel arrays used for the QA plot.

    components defaults to ("nadir",) ONLY -- deliberately narrower than
    pool_stripe_pixels' own default of ("nadir", "forward", "cross").
    "forward" and "cross" are the WFOV camera's other, genuinely oblique
    viewing stripes (large VZA by design, not a bug in them) -- pooling
    all three together was the main reason an earlier version of this
    function saw a VZA range of 0.007-64 deg instead of the expected few
    degrees. The single-tile-mean-angle design this whole script relies
    on only makes sense restricted to the nadir stripe in the first
    place; max_vza_deg below is kept anyway as a cheap secondary sanity
    check (catches any remaining outliers, e.g. at the nadir stripe's own
    edges or scan-pattern artifacts), not the primary mechanism.

    margin_cells (added 2026-10-22, Sebastian's workaround for the
    periodic-BC/MPI-forced-periodic SHDOM artifact tracked in
    docs/L1B_TO_L2_ROADMAP.md's "Hong input" section): if > 0, requests
    this many EXTRA VJ143C1-native (0.05 deg) grid rows/cols of
    surface/BRDF data beyond the tile_km box on every side, purely so
    the RT SOLVE runs on a bigger domain than the tile of interest --
    the pooled-camera-pixel tile-box selection (in_box/in_tile, pooled's
    x_km/y_km) is NOT affected, it is still exactly the real tile_km box.
    Caller is responsible for running rt_shdom()/rt_noatm() on this
    padded `data`, then cropping the result back down with
    crop_tile_margin() before using/plotting/saving it -- tile() itself
    does not crop anything (the padding needs to survive through the
    RT call, which is the whole point).

    era5_path (added 2026-10-22): if given, overrides the ocean BRDF's
    windspeed field -- otherwise read from the hardcoded MERRA2 file in
    l2_cam's sample/input.yml -- with real ERA5 u10/v10-derived values,
    one per cell of read_brdf_parms()'s own cropped ocean grid
    (data['lon_ocean']/data['lat_ocean']), i.e. spatially variable across
    the tile, not a single constant (a 100 km tile can span several 0.25
    deg ERA5 cells). None (default) leaves the original MERRA2-derived
    field untouched.

    brdf_land_path (added 2026-10-22): if given, a real VJ143C1 (or
    VNP43C1) land BRDF file -- e.g. from apps/l2/fetch_brdf.py -- used
    INSTEAD of l2_cam's own hardcoded sample/brdf_parms/VJ143C1...h5
    (which is dated 2021-05-18, a fixed date unrelated to any real
    tile's own date). Unlike era5_path (which overrides read_brdf_parms'
    OUTPUT field after the fact), this overrides which file
    read_brdf_parms() itself opens, since the BRDF kernel weights
    (fiso/fvol/fgeo) have no simple post-hoc substitute the way a
    scalar windspeed does. None (default) leaves l2_cam's own stale
    sample file in use, unchanged.
    """
    lat0, lon0 = match["lat"], match["lon"]
    half_km = tile_km / 2.0

    per_frame = est.pool_stripe_pixels(
        cam_l1b, first_frame=match["first_frame"],
        max_frames=match["max_frames"], stride=stride,
        components=components, verbose=verbose,
        extra_fields=MEASURED_RADIANCE_FIELDS)
    if not per_frame:
        raise SystemExit("tile: no usable camera frames in the matched window "
                          "-- widen --window-s")

    cam_lat = np.concatenate([f["lat"] for f in per_frame])
    cam_lon = np.concatenate([f["lon"] for f in per_frame])
    frame_id = np.concatenate(
        [np.full(len(f["lat"]), f["frame"], dtype=np.int64) for f in per_frame])

    # "Measured" radiance (added 2026-10-22, for the RT-vs-measured
    # comparison Sebastian asked for once an L1X file is available) --
    # present only when cam_l1b is an apps/l2/build_l1x.py "L1X" file
    # (pool_stripe_pixels' extra_fields mechanism silently omits these
    # keys from EVERY per_frame entry for a plain, non-proxy L1B, so
    # "cam_rad" in per_frame[0] is a reliable per-granule presence
    # check, not per-pixel). cam_proxy_mask flags which of THOSE pixels
    # were actually overwritten by the proxy projection (vs. left as the
    # original dark/placeholder L1B value) -- see build_l1x.py's
    # L1X_Proxy_Mask variable.
    cam_rad = (np.concatenate([f["cam_rad"] for f in per_frame])
              if "cam_rad" in per_frame[0] else None)
    cam_proxy_mask = (np.concatenate([f["cam_proxy_mask"] for f in per_frame])
                      if "cam_proxy_mask" in per_frame[0] else None)
    angle_keys = [k for k in ("sza", "vza", "saa", "vaa") if k in per_frame[0]]
    missing = [k for k in ("sza", "vza", "saa", "vaa") if k not in angle_keys]
    if missing:
        print("  WARNING: camera L1B has no %s field(s); tile-mean angle(s) "
              "for %s cannot be computed" % (missing, missing))
    angles = {k: np.concatenate([f[k] for f in per_frame]) for k in angle_keys}

    dist_km, bearing = haversine_bearing(lat0, lon0, cam_lat, cam_lon)
    x_km = dist_km * np.sin(bearing)
    y_km = dist_km * np.cos(bearing)
    in_box = (np.abs(x_km) <= half_km) & (np.abs(y_km) <= half_km)
    n_in_box = int(in_box.sum())
    if n_in_box == 0:
        # diagnostic (Sebastian, 2026-10, BRDF list item #7 / kyushu site):
        # a zero-overlap failure is ambiguous on its own -- it could mean
        # "right time window, just needs a slightly bigger box/window" or
        # "the camera's nadir stripe was never anywhere near this site in
        # this pass at all" (e.g. the RAD-4CH match that chop() found was
        # seen by the radiometer off-nadir/from a different crossing than
        # whatever the camera's narrow nadir stripe was looking at during
        # this time window -- chop() only matches on the RADIOMETER's own
        # lat/lon, with no guarantee the CAMERA was pointed there too).
        # Report exactly how far off the pooled pixels actually are, and
        # in which direction, instead of leaving that to guesswork.
        i_near = int(np.argmin(dist_km))
        vza_note = ""
        if "vza" in angles:
            vza_note = "  VZA of that closest pixel: %.2f deg." % angles["vza"][i_near]
        print("  DIAGNOSTIC [zero pixels in tile box]: %d pooled nadir-component "
              "pixel(s) across %d frame(s) span lat %.4f..%.4f, lon %.4f..%.4f. "
              "Closest pooled pixel to the requested tile center (lat0=%.4f "
              "lon0=%.4f) is %.1f km away at bearing %.1f deg (frame %d, "
              "lat=%.4f lon=%.4f) -- i.e. the camera's nadir stripe was "
              "centered that far from this site during the scanned window, "
              "not just outside a %g km box.%s"
              % (cam_lat.size, len(per_frame), cam_lat.min(), cam_lat.max(),
                 cam_lon.min(), cam_lon.max(), lat0, lon0, dist_km[i_near],
                 np.degrees(bearing[i_near]) % 360.0, int(frame_id[i_near]),
                 cam_lat[i_near], cam_lon[i_near], tile_km, vza_note))
        # second pass (Sebastian, 2026-10): don't assume WHY the lon span
        # above is physically impossible (-180..180 in ~190 s of real
        # ground-track motion) -- check it. Report the facts and let them
        # settle it: does the bad (near-dateline-wraparound) longitude
        # correlate with high VZA (consistent with a near-limb/component-
        # boundary effect), or do even LOW-VZA, nominally-near-nadir
        # pixels show the same bad longitude (a distinct, more basic
        # geolocation problem, not explained by viewing angle at all)?
        if "vza" in angles:
            vza = angles["vza"]
            pct = np.nanpercentile(vza, [0, 25, 50, 75, 90, 99, 100])
            print("  DIAGNOSTIC: VZA distribution of all %d pooled pixels "
                  "[min,25,50,75,90,99,max] = [%s] deg"
                  % (cam_lat.size, ", ".join("%.2f" % p for p in pct)))

            bad_lon = np.abs(cam_lon) > 170.0
            n_bad = int(bad_lon.sum())
            print("  DIAGNOSTIC: %d of %d pooled pixels (%.1f%%) have "
                  "|lon| > 170 deg (dateline-wraparound-suspect)."
                  % (n_bad, cam_lat.size, 100.0 * n_bad / cam_lat.size))
            if n_bad > 0:
                pct_bad = np.nanpercentile(vza[bad_lon], [0, 50, 100])
                print("    -- their VZA [min,median,max] = [%.2f, %.2f, %.2f] deg"
                      % tuple(pct_bad))
            n_good = int((~bad_lon).sum())
            if n_good > 0:
                pct_good = np.nanpercentile(vza[~bad_lon], [0, 50, 100])
                print("    -- the other %d pixels' VZA [min,median,max] = "
                      "[%.2f, %.2f, %.2f] deg (these are the ones actually "
                      "near lon=%.1f; closest-to-tile distance among them: "
                      "%.1f km)"
                      % (n_good, pct_good[0], pct_good[1], pct_good[2], lon0,
                         float(np.min(dist_km[~bad_lon]))))

            clean = vza <= max_vza_deg
            n_clean = int(clean.sum())
            if n_clean == 0:
                print("  DIAGNOSTIC: 0 of %d pooled pixels have VZA <= %.1f "
                      "deg in this window." % (cam_lat.size, max_vza_deg))
            else:
                i_clean_near = int(np.argmin(np.where(clean, dist_km, np.inf)))
                print("  DIAGNOSTIC: %d of %d pooled pixels have VZA <= "
                      "%.1f deg; the closest of those to the tile center is "
                      "%.1f km away (frame %d, lat=%.4f lon=%.4f, VZA=%.2f deg)."
                      % (n_clean, cam_lat.size, max_vza_deg,
                         dist_km[i_clean_near], int(frame_id[i_clean_near]),
                         cam_lat[i_clean_near], cam_lon[i_clean_near],
                         vza[i_clean_near]))
        raise SystemExit("tile: zero pooled camera pixels fall inside the "
                          "%g x %g km tile box -- widen --window-s or check "
                          "the site" % (tile_km, tile_km))

    in_tile = in_box
    if "vza" in angles:
        near_nadir = angles["vza"] <= max_vza_deg
        in_tile = in_box & near_nadir
        n_in_tile = int(in_tile.sum())
        if verbose:
            print("  --max-vza-deg %.1f filter: kept %d/%d pooled pixels in "
                  "the tile box (excluded repeat far-off-nadir camera views "
                  "of the same ground tile)" % (max_vza_deg, n_in_tile, n_in_box))
        if n_in_tile == 0:
            raise SystemExit(
                "tile: zero pooled camera pixels within %g km AND within "
                "%.1f deg of nadir -- the camera never saw this tile from "
                "near-overhead in the scanned window. Either widen "
                "--window-s/--scan-pad-s to catch the actual overhead pass, "
                "or raise --max-vza-deg (at the cost of the single-tile-mean-"
                "angle approximation getting worse)." % (tile_km, max_vza_deg))
    else:
        n_in_tile = n_in_box

    tile_mean = {k: float(np.nanmean(angles[k][in_tile])) for k in angle_keys}
    tile_range = {k: (float(np.nanmin(angles[k][in_tile])),
                       float(np.nanmax(angles[k][in_tile]))) for k in angle_keys}
    if verbose:
        print("  tile box: %g x %g km around (lat0=%.4f, lon0=%.4f); "
              "%d pooled camera pixels inside, from %d frame(s)"
              % (tile_km, tile_km, lat0, lon0, n_in_tile,
                 len(set(frame_id[in_tile].tolist()))))
        for k in angle_keys:
            lo, hi = tile_range[k]
            print("    tile-mean %-4s = %8.3f deg  (range %7.3f .. %7.3f)"
                  % (k.upper(), tile_mean[k], lo, hi))

    # degrees-per-km at this latitude, for the extent handed to Hong's
    # read_surface_mask/read_brdf_parms (same equirectangular approx as
    # Hong's own lonlat_to_xy, just inverted to go km -> degrees)
    dlat_per_km = 1.0 / 111.32
    dlon_per_km = 1.0 / (111.32 * max(0.05, np.cos(np.radians(lat0))))
    half_deg_lat = half_km * dlat_per_km
    half_deg_lon = half_km * dlon_per_km

    # margin_cells padding (2026-10-22): widen ONLY the extent handed to
    # read_surface_mask/read_brdf_parms, in native VJ143C1 CMG grid units
    # (0.05 deg -- same constant already noted below at "true spacing is
    # VJ143C1's native 0.05 deg CMG grid"), so the cropped lon_land/
    # lat_land/brdf grid comes back margin_cells cells bigger on every
    # side than the real tile. half_km itself (pooled-camera-pixel tile
    # box, in_box/in_tile) is deliberately left untouched above -- the
    # padding is for the RT domain only.
    _VJ143C1_GRID_DEG = 0.05
    half_deg_lat_padded = half_deg_lat + margin_cells * _VJ143C1_GRID_DEG
    half_deg_lon_padded = half_deg_lon + margin_cells * _VJ143C1_GRID_DEG

    overpass_dt = datetime.datetime.utcfromtimestamp(match["t_ns"] / 1e9)
    utc_minute = overpass_dt.hour * 60.0 + overpass_dt.minute + overpass_dt.second / 60.0

    data = {
        "extent": [lon0 - half_deg_lon_padded, lon0 + half_deg_lon_padded,
                   lat0 - half_deg_lat_padded, lat0 + half_deg_lat_padded],
        "utc_cam": np.array([utc_minute]),
        "margin_cells": margin_cells,
        "tile_km_core": tile_km,
    }

    sys.path.insert(0, os.path.join(l2_cam_root, "src"))
    from cam_l2.util import read_cfg
    from cam_l2.common import fdir_data
    from cam_l2.cld import read_surface_mask, read_brdf_parms, gen_brdf_shdom

    cfg = read_cfg(os.path.join(fdir_data, "sample", "input.yml"))

    if brdf_land_path is not None:
        _brdf_land_abs = os.path.abspath(brdf_land_path)
        _brdf_land_rel = os.path.relpath(_brdf_land_abs, start=fdir_data)
        if verbose:
            print("  brdf_land_path override: using %s (relative to l2_cam's own data/ dir: %s) instead of l2_cam's hardcoded sample BRDF file (%s)"
                 % (_brdf_land_abs, _brdf_land_rel,
                    cfg["input"]["brdf"]["f_land"]["fname"]))
        cfg["input"]["brdf"]["f_land"]["fname"] = _brdf_land_rel

    class _NullLogger:
        def info(self, *a, **k):
            if verbose:
                print("    [cam_l2] " + (a[0] if a else ""))

        def error(self, *a, **k):
            print("    [cam_l2 ERROR] " + (a[0] if a else ""))

    data = read_surface_mask(cfg, _NullLogger(), data=data)
    data = read_brdf_parms(cfg, data=data)

    if era5_path is not None:
        era5_windspeed_field = _era5_windspeed_field(
            era5_path, data['lon_ocean'], data['lat_ocean'], overpass_dt)
        _valid_ws = data['windspeed_ocean'] > -900  # read_brdf_parms' own fill-value convention
        if verbose:
            _old = data['windspeed_ocean'][_valid_ws]
            _new = era5_windspeed_field[_valid_ws]
            _old_range = ("%.2f..%.2f m/s" % (_old.min(), _old.max())
                         if _old.size else "n/a (no ocean cells in this tile)")
            _new_range = ("%.2f..%.2f m/s" % (_new.min(), _new.max())
                         if _new.size else "n/a (no ocean cells in this tile)")
            print("  ERA5 override: ocean windspeed = %s across tile, %d cell(s) "
                  "(replacing MERRA2-derived %s)"
                  % (_new_range, int(_valid_ws.sum()), _old_range))
        data['windspeed_ocean'] = np.where(_valid_ws, era5_windspeed_field,
                                           data['windspeed_ocean'])

    _date_issues = _check_ancillary_date_compatibility(
        cfg, overpass_dt, brdf_land_path is not None, era5_path is not None,
        verbose=verbose)
    if _date_issues:
        _fatal = [i for i in _date_issues if i["severity"] == "fatal"]
        _label = "FATAL" if _fatal else "WARNING"
        print("\n" + "#" * 78)
        print("# ANCILLARY-DATA DATE MISMATCH (%s) -- see tile()'s "
             "_check_ancillary_date_compatibility() docstring" % _label)
        print("#" * 78)
        for _issue in _date_issues:
            print("# [%s] " % _issue["severity"].upper()
                 + _issue["message"].replace("\n", "\n# "))
        print("#" * 78)
        if _fatal and not allow_ancillary_date_mismatch:
            raise SystemExit(
                "Refusing to continue: at least one ancillary-data mismatch "
                "above is categorically impossible to be right (not just "
                "today's best-available data), not merely a precision "
                "limitation -- pass allow_ancillary_date_mismatch=True "
                "(run_dev_step1.py: --allow-ancillary-date-mismatch) if you "
                "want this run anyway (e.g. deliberately, for dev use).")

    # --- diagnostic (Sebastian, 2026-10): gen_brdf_shdom's "ice" branch is a
    # catch-all (neither ocean mask==-1 nor land mask==0/valid-kernels), not
    # an explicit ice-mask test. A tile cell where find_nearest cannot locate
    # a source land/ocean-mask pixel within Ngrid_limit (raw mask falls back
    # to fill_value=-999) would silently become "ice" with the hardcoded
    # bright snow/ice BRDF below -- with no warning today. Recompute the same
    # nearest-neighbor lookup here (identical inputs/params to what
    # gen_brdf_shdom does internally) to check for that and report
    # WHERE/WHY before it runs. Wrapped defensively -- this is a diagnostic
    # add-on, not allowed to crash the real pipeline (or the mocked test
    # harness, which has no real er3t/scipy to import) on its own.
    try:
        import er3t.util
        from scipy.spatial import KDTree
        _sfc_x = np.asarray(data["lon_sfc"], dtype=np.float64).ravel()
        _sfc_y = np.asarray(data["lat_sfc"], dtype=np.float64).ravel()
        _raw_mask = er3t.util.find_nearest(
            data["lon_sfc"], data["lat_sfc"], data["sfc"],
            data["lon_land"], data["lat_land"], Ngrid_limit=5, fill_value=-999)
        _raw_mask = np.int_(_raw_mask)
        _logic_fill = (_raw_mask == -999)
        _logic_other = (~_logic_fill) & (_raw_mask != 0) & (_raw_mask != -1)
        if verbose:
            print("  diagnostic: raw land/ocean mask covers lon %.4f..%.4f "
                  "lat %.4f..%.4f (tile needs lon %.4f..%.4f lat %.4f..%.4f)"
                  % (_sfc_x.min(), _sfc_x.max(), _sfc_y.min(), _sfc_y.max(),
                     data["lon_land"].min(), data["lon_land"].max(),
                     data["lat_land"].min(), data["lat_land"].max()))
        if _logic_fill.any() or _logic_other.any():
            print("  WARNING [ice-mask catch-all]: %d cell(s) found NO "
                  "source mask pixel within Ngrid_limit=5 (raw mask=-999 "
                  "fill value); %d cell(s) carry a mask code other than the "
                  "expected 0=land/-1=ocean. ALL of these will be silently "
                  "treated as ice/snow (fiso=0.986) by gen_brdf_shdom's "
                  "catch-all logic. Affected cells:"
                  % (int(_logic_fill.sum()), int(_logic_other.sum())))
            _sfc_tree = KDTree(np.transpose(np.vstack((_sfc_x, _sfc_y))))
            # find_nearest's own dist_limit test, replicated exactly (same
            # dtypes/edge handling as er3t.util.find_nearest) -- if the true
            # nearest neighbor turns out to be well inside this radius, the
            # fill-in is NOT a "no data nearby" case and the bug is inside
            # find_nearest's own index bookkeeping (indices_xy>=data.size /
            # ==-1), not a real data gap.
            _xo, _yo = data["lon_land"], data["lat_land"]
            _dx = np.zeros_like(_xo, dtype=np.float32)
            _dy = np.zeros_like(_yo, dtype=np.float32)
            _dx[1:, ...] = _xo[1:, ...] - _xo[:-1, ...]
            _dx[0, ...] = _dx[1, ...]
            _dy[..., 1:] = _yo[..., 1:] - _yo[..., :-1]
            _dy[..., 0] = _dy[..., 1]
            _dist_limit = np.sqrt((_dx * 5) ** 2 + (_dy * 5) ** 2)
            for _idx in zip(*np.where(_logic_fill | _logic_other)):
                _dist, _nn = _sfc_tree.query(
                    [data["lon_land"][_idx], data["lat_land"][_idx]])
                print("    grid %s: raw mask=%d  lon=%.4f lat=%.4f  nearest "
                      "raw-mask pixel %.4f deg away (source index %d of %d)  "
                      "find_nearest's own dist_limit there = %.4f deg -> "
                      "%s"
                      % (_idx, _raw_mask[_idx], data["lon_land"][_idx],
                         data["lat_land"][_idx], _dist, _nn, _sfc_x.size,
                         _dist_limit[_idx],
                         "distance test alone would PASS (bug is in "
                         "find_nearest's index bookkeeping, not a real data "
                         "gap)" if _dist <= _dist_limit[_idx] else
                         "distance test alone would FAIL (genuinely too far "
                         "-- a real data gap)"))
        else:
            print("  diagnostic: no fill-value or unexpected mask codes "
                  "found -- every tile cell matched a real land(0)/ocean(-1) "
                  "mask pixel.")
    except Exception as exc:
        print("  (ice-mask catch-all diagnostic skipped: %r)" % (exc,))

    data = gen_brdf_shdom(data=data)

    # --- workaround for the confirmed find_nearest bug (see the function's
    # docstring, and the ice-mask-catch-all diagnostic above) -- reported
    # to Hong, not fixed in his vendored er3t code; this corrects our own
    # BRDF arrays in the meantime without touching his files.
    try:
        _n_fixed = _fix_find_nearest_miscategorized_cells(data, verbose=verbose)
        if _n_fixed:
            print("  WORKAROUND APPLIED: re-classified %d cell(s) that the "
                  "find_nearest bug above had mislabeled as ice -- BRDF "
                  "arrays patched in place (ice-mask-catch-all workaround)."
                  % _n_fixed)
    except Exception as exc:
        print("  (find_nearest miscategorization workaround skipped: %r)" % (exc,))

    nx, ny = data["lon_land"].shape
    if verbose:
        print("  tile surface/BRDF grid: %d x %d cells (dx=dy=%.2f km "
              "nominal label; true spacing is VJ143C1's native 0.05 deg "
              "CMG grid, ~5.57 km at the equator, kept isotropic per "
              "Sebastian's decision)" % (nx, ny, dx_km))

    data["tile_mean_angles"] = tile_mean
    data["tile_overpass_utc"] = overpass_dt
    data["dx_km"] = dx_km
    data["dy_km"] = dy_km

    pooled = dict(cam_lat=cam_lat, cam_lon=cam_lon, frame_id=frame_id,
                  x_km=x_km, y_km=y_km, in_tile=in_tile, in_box=in_box,
                  angles=angles, lat0=lat0, lon0=lon0,
                  cam_rad=cam_rad, cam_proxy_mask=cam_proxy_mask)
    return data, pooled


def crop_tile_margin(data, margin_cells, extra_arrays=None):
    """Undo tile()'s `margin_cells` padding (2026-10-22 workaround for the
    periodic-BC/MPI-forced-periodic SHDOM artifact -- see
    docs/L1B_TO_L2_ROADMAP.md "Hong input" for the full writeup). Call
    this AFTER rt_shdom()/rt_noatm() have run on the padded `data` --
    cropping before the RT call would defeat the whole point (the
    padding has to be there during the solve, so the forced-periodic
    boundary artifact lands in the discarded margin, not in the tile of
    interest).

    Crops every per-cell array in `data` whose leading (nx, ny) shape
    matches `data['lon_land']`'s (the padded grid) down to the inner
    `margin_cells`-smaller region on every side -- this picks up
    lon_land/lat_land/fiso_land/fvol_land/fgeo_land/brdf (all built from
    the same VJ143C1-native crop in Hong's read_brdf_parms/
    gen_brdf_shdom) automatically, with no hardcoded key list.
    lon_sfc/lat_sfc/sfc (the finer native land-mask grid) and
    lon_ocean/lat_ocean/windspeed_ocean (the coarser MERRA2/ERA5 wind
    grid) are both genuinely DIFFERENT shapes in Hong's own code, so
    they're correctly left untouched here -- they're fully consumed by
    gen_brdf_shdom() already and aren't used again except in our own raw
    diagnostic .npz dump, where they'll still show the padded extent
    (cosmetic only, not a correctness issue for any science output).

    Returns (data_cropped, extra_cropped) -- a NEW dict (does not mutate
    `data`) plus `extra_arrays` (e.g. (rad, refl_app, refl_noatm) from
    rt_shdom()/rt_noatm(), which share the padded grid's shape but live
    outside `data`) cropped the same way, in the same order. If
    `margin_cells` is 0 (the default -- no padding requested), returns
    `data`/`extra_arrays` unchanged.
    """
    if margin_cells <= 0:
        return data, extra_arrays

    m = margin_cells
    shape0 = data["lon_land"].shape
    out = dict(data)
    n_cropped = 0
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[:2] == shape0:
            out[k] = v[m:-m, m:-m, ...]
            n_cropped += 1
    out["margin_cells"] = 0  # cropped away now -- this IS the real tile

    extra_cropped = None
    if extra_arrays is not None:
        extra_cropped = tuple(
            arr[m:-m, m:-m, ...] if isinstance(arr, np.ndarray) and arr.shape[:2] == shape0
            else arr
            for arr in extra_arrays)

    print("  crop_tile_margin: discarded %d-cell margin on every side "
          "(%d padded field(s) in `data` cropped from %s down to %s)"
          % (m, n_cropped, shape0, out["lon_land"].shape))

    return out, extra_cropped


# ── phase 3: rt ──────────────────────────────────────────────────────────

def rt_noatm(data):
    """No-atmosphere, BRDF-only reflectance map -- pure kernel algebra,
    no SHDOM solve. See brdf_kernels.py."""
    tm = data["tile_mean_angles"]
    raz = tm["vaa"] - tm["saa"]  # cos(raz)/sin(raz)^2 only ever used -> sign-independent
    refl, flag = brdf_kernels.tile_noatm_reflectance(
        tm["sza"], tm["vza"], raz, data["brdf"])
    return refl, flag


def rt_shdom(data, l2_cam_root, wavelength_nm, altitude_km, Niter=10,
             Ncpu=4, solver="3D", overwrite=True, out_root="tmp-data",
             surface_mode="brdf", atm_levels=None,
             cloud_cot=0.0, cloud_cer=10.0, cloud_cth=4.0, cloud_cgt=2.0,
             vza_grid=None, vaa_grid=None):
    """Full SHDOM run, Hong's run_sim_rad pattern, parameterized (dx/dy,
    altitude, date, single tile-mean VZA/VAA broadcast across a full
    (Nx, Ny) output raster -- see module docstring). Clear-sky by default
    (cloud_cot=0.0) -- the whole existing dev-step-1 pipeline relies on
    that default and is unaffected by the cloud_cot/_cer/_cth/_cgt
    kwargs below. Pass cloud_cot>0 for a single uniform cloud layer
    across the whole domain (added 2026-10-23 for
    apps/l2/run_angular_sweep.py's EVO clear-vs-cloudy angular-dependence
    study -- see that script).

    vza_grid/vaa_grid (added 2026-10-23, same EVO script, same day --
    Sebastian: "I thought SHDOM could handle several angles in a single
    solve"): optional (nx, ny)-shaped arrays of PER-CELL view zenith/
    azimuth angle [deg], used in place of the default single tile-mean
    angle broadcast across every cell. Correct and cheap to do in one
    solve: SHDOM's expensive step is the 3D radiative-transfer solve
    itself (Niter iterations over the fixed atmosphere/surface/cloud/
    sun geometry), NOT the number of distinct output view angles --
    camera3's per-cell sensor_zenith_angles/sensor_azimuth_angles/
    sensor_xpos/sensor_ypos (below) already support a genuinely
    different angle (and correspondingly different ray-origin parallax)
    per output cell; this script just never exploited that before,
    always calling np.full((nx, ny), tm["vza"]) instead. Physically,
    this is also exactly how the real WFOV camera itself works: one
    wide-FOV snapshot simultaneously observes MANY different ground
    cells at MANY different view angles, not one point revisited at
    different times -- so assigning each grid cell its own requested
    angle mirrors the real instrument's own geometry, not an
    approximation of it. Only changes behavior when passed explicitly;
    None (default) keeps the original tm["vza"]/tm["vaa"] broadcast,
    unchanged for every existing caller.

    surface_mode="brdf" (default): SHDOM sees the real per-cell BRDF,
    exactly as data["brdf"] (ocean/land/ice, via gen_brdf_shdom) specifies
    it -- unchanged pipeline behavior.

    surface_mode="lambertian" (BRDF diagnostic list item #5): SHDOM
    instead sees a per-cell ISOTROPIC albedo. The albedo chosen is each
    cell's own actual-BRDF reflectance at the tile's mean sun/view
    geometry -- i.e. exactly brdf_kernels.tile_noatm_reflectance()'s
    output, the same quantity rt_noatm() returns for the no-atmosphere
    panel. This is the fair, apples-to-apples "Lambertian counterpart":
    at THIS ONE geometry the two surfaces reflect identically by
    construction (a Lambertian surface's reflectance is its albedo at
    any angle, and we set albedo = BRDF reflectance at this angle), so
    the no-atmosphere panel is unchanged, and the no-atmosphere vs.
    SHDOM comparison still isolates "what the atmosphere adds." The
    SHDOM run itself WILL differ between the two modes, though: SHDOM's
    diffuse surface illumination integrates over many incidence/exit
    angles, not just the one tile-mean direction, so a sharply-peaked
    BRDF (sun glint, the land hotspot) feeds back differently into the
    multiple-scattering solution than an isotropic surface carrying the
    same single-angle value would. That difference IS the experiment.
    Reflectance can exceed 1 for a strongly peaked BRDF (e.g. near
    glint) at the evaluation geometry; since a Lambertian albedo must be
    <= 1, any such cells are clipped and reported (itself diagnostic:
    it marks where the Lambertian stand-in cannot fully represent the
    real BRDF's peak)."""
    er3t_dir = os.path.join(l2_cam_root, "software", "er3t")
    if er3t_dir not in sys.path:
        sys.path.insert(0, er3t_dir)
    shdom_exe = os.path.join(l2_cam_root, "software", "shdom", "build", "bin", "shdom")
    if "SHDOM_EXE" not in os.environ:
        if os.path.exists(shdom_exe):
            os.environ["SHDOM_EXE"] = shdom_exe
        else:
            raise SystemExit(
                "SHDOM_EXE is not set and no compiled SHDOM binary was found "
                "at %s -- build it (cd %s/software/shdom && make mpi) or set "
                "SHDOM_EXE yourself before running this script."
                % (shdom_exe, l2_cam_root))
    import er3t  # noqa: E402  (after SHDOM_EXE is set -- er3t.common reads it at import time)

    tm = data["tile_mean_angles"]
    nx, ny = data["lon_land"].shape
    dx, dy = data["dx_km"], data["dy_km"]

    # BUG FIX (2026-10-22 -- surfaced by Sebastian noticing "the top right
    # plot seems to contain TWO tiles side by side" in the QA figure):
    # data["dx_km"] is the VJ143C1 native grid's EQUATORIAL km-per-0.05-deg
    # value (that IS what "nominal" means in the "tile surface/BRDF grid"
    # message printed above) -- not the true physical east-west spacing
    # away from the equator, where a degree of LONGITUDE covers less
    # ground than at the equator (meridians converge toward the poles).
    # This tile sits at ~60 deg latitude, where cos(60)~0.5 -- a degree of
    # longitude there is only ~half as many km as at the equator. The
    # SHDOM domain (surface grid + sensor ray-position/parallax calc
    # below) was built from the UNCORRECTED nominal dx in BOTH
    # directions, so its modeled east-west physical extent was literally
    # ~2x too wide at this latitude -- confirmed directly from this run's
    # own SHDOM property file: NX=37, nominal dx=5.5 -> domain width
    # written as 203.5 km, vs. a true physical width of ~103.6 km.
    # North-south (dy) needs NO such correction -- a degree of latitude
    # is ~111.32 km everywhere on a sphere, to first order, independent
    # of latitude. Fixed by applying the same equirectangular cos(lat)
    # factor tile()'s own dlon_per_km already applies going the OTHER
    # way (km -> degrees, to size the tile's lon/lat extent in the first
    # place) -- this is just that same conversion, applied consistently
    # here too, to go from degree-grid-cell-count back to real km.
    lat_ref = float(np.mean(data["lat_land"]))
    dx_physical = dx * max(0.05, np.cos(np.radians(lat_ref)))

    # namespaced by surface_mode (not just wavelength) so a lambertian run
    # never reads back a stale brdf-mode sfc/rad cache or vice versa --
    # default "brdf" keeps the original path unchanged. (Doesn't fix the
    # broader tmp-data caching footgun across *sites* -- see list item #9.)
    fdir = f"{out_root}/run_tile_rt/{wavelength_nm:04.0f}"
    if surface_mode != "brdf":
        fdir = f"{fdir}_{surface_mode}"
    os.makedirs(fdir, exist_ok=True)

    # atm_levels override (Sebastian, 2026-10): added for a decisive
    # SHDOM-vs-SHDOM atmosphere-attribution test (see
    # rt_shdom_vacuum_test.py) -- passing an extremely thin level range
    # (e.g. [0.0, 0.001] km) gives atm_atmmod() a near-zero actual air
    # column (same standard-atmosphere density profile, just sampled over
    # ~1 m of path length instead of 60 km), which drives both Rayleigh
    # scattering and gas absorption optical depth to ~0 -- a "vacuum"
    # atmosphere using the EXACT SAME code path (surface BRDF dispatch,
    # solver settings, sensor geometry) as a real run, so any difference
    # from the real-atmosphere run isolates what the atmosphere itself
    # adds, with no separate/independently-written comparison model
    # involved. Default (None) is the original fixed 0-60 km profile --
    # unchanged pipeline behavior.
    levels = (np.asarray(atm_levels, dtype=np.float64) if atm_levels is not None
              else np.concatenate((np.arange(0.0, 10.0, 2.0), np.arange(10.0, 60.1, 10.0))))
    atm0 = er3t.pre.atm.atm_atmmod(levels=levels, fname=f"{fdir}/atm.pk", overwrite=overwrite)
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength_nm, fname=f"{fdir}/abs.pk",
                                target="coarse", atm_obj=atm0, overwrite=overwrite)

    if surface_mode == "lambertian":
        raz = tm["vaa"] - tm["saa"]  # same convention as rt_noatm()
        albedo, _flag = brdf_kernels.tile_noatm_reflectance(
            tm["sza"], tm["vza"], raz, data["brdf"])
        n_clipped = int((albedo > 1.0).sum())
        if n_clipped:
            print("  surface-mode=lambertian: %d cell(s) had actual-BRDF "
                  "reflectance > 1.0 at the tile-mean geometry (max %.3f) -- "
                  "clipped to 1.0 for the Lambertian albedo (a Lambertian "
                  "surface cannot represent a reflectance peak above 1; this "
                  "undersells how bright those cells really are)."
                  % (n_clipped, float(np.nanmax(albedo))))
        albedo = np.clip(np.nan_to_num(albedo, nan=0.0), 0.0, 1.0)
        sfc_dict = {"dx": dx_physical, "dy": dy, "alb": albedo}
    elif surface_mode == "brdf":
        sfc_dict = {"dx": dx_physical, "dy": dy, "mixed_brdf": data["brdf"]}
    else:
        raise ValueError("rt_shdom: unknown surface_mode %r (expected 'brdf' or "
                          "'lambertian')" % (surface_mode,))
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=f"{fdir}/sfc.pk", overwrite=overwrite)

    # REVERTED (Sebastian, 2026-10): a previous version of this fix scaled
    # cth/cgt down to "stay within atm_levels" when atm_levels overrides to
    # a thin vacuum range -- wrong, and confirmed to crash
    # (er3t.pre.cld.cld_gen.cld_gen_cop.process(), "zero-size array to
    # reduction operation minimum"). cld_gen_cop builds its OWN small,
    # self-contained internal atm_atmmod purely to place the (cot=0.0,
    # i.e. physically inert either way) cloud layer -- it reuses atm0 only
    # for atm0.fname_atmmod (a cache path), never atm0's actual levels/
    # range. Shrinking cth/cgt/dz together made that internal mini-
    # atmosphere's own np.arange(h_bottom, h_top, dz) come up with too few
    # points (h_top-h_bottom << dz), not an atm_levels-range mismatch at
    # all. Since cot=0.0 everywhere makes this cloud layer a true no-op
    # regardless of where it sits, the fix is simply to leave cth/cgt/dz at
    # their original fixed values unconditionally, independent of
    # atm_levels.
    cld0 = er3t.pre.cld.cld_gen_cop(
        fname=f"{fdir}/cld.pk",
        # cloud_cot/_cer/_cth/_cgt (added 2026-10-23, Sebastian's EVO
        # angular-dependence sweep -- apps/l2/run_angular_sweep.py): a
        # single uniform cloud layer across the whole domain, same
        # cld_gen_cop() call as always -- only the four numbers are now
        # caller-settable. Defaults (cot=0.0) keep every EXISTING caller's
        # behavior byte-for-byte unchanged (a true physical no-op,
        # regardless of cer/cth/cgt, same as before this change); cer's
        # default changed from the old inert placeholder 1.0 to a more
        # physically realistic 10.0 um, harmless while cot=0.0 but
        # sensible the moment a caller turns cot on.
        cot=np.array([cloud_cot]).reshape((1, 1)),
        cer=np.array([cloud_cer]).reshape((1, 1)),
        cth=np.array([cloud_cth]).reshape((1, 1)),
        cgt=np.array([cloud_cgt]).reshape((1, 1)),
        dz=2.0,
        extent_xy=[0.0, sfc0.data["dx"]["data"] * sfc0.data["nx"]["data"],
                   0.0, sfc0.data["dy"]["data"] * sfc0.data["ny"]["data"]],
        atm_obj=atm0, overwrite=overwrite)

    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0,
                                      fname=f"{fdir}/shdom-sfc_{wavelength_nm:04.0f}.txt",
                                      overwrite=overwrite)
    atm1d0 = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0,
                                      fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm3d0 = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0,
                                      fname=f"{fdir}/shdom-prp.txt",
                                      fname_atm_1d=atm1d0.fname, overwrite=overwrite)

    # one output ray per surface grid cell, ALL sharing the one tile-mean
    # (vza, vaa) -- the deliberate single-angle-per-tile design (see
    # module docstring).
    #
    # BUG FIX (Sebastian, 2026-10): this used to build xpos/ypos from
    # plain grid-cell-center positions ((ii+0.5)*dx, (jj+0.5)*dy), with NO
    # correction for the fact that SHDOM's camera3 sensor backward-traces
    # each ray from a point at SENSOR ALTITUDE, not from the ground
    # target -- the two are offset horizontally by altitude*tan(vza) in
    # the view-azimuth direction (parallax). At this tile's VZA (~30-48
    # deg) and our ~831 km altitude that offset is ~480-920 km -- bigger
    # than even a 500x500 km tile -- so every ray was launched from a
    # wildly wrong position. This is exactly the correction Hong's own
    # production code applies (cam_l2.cld.lonlat_to_xy(), confirmed by
    # reading his actual run_sim_rad() source, 2026-10); we had never
    # ported it. Matching his convention exactly: reference origin is the
    # (0,0) grid cell's own lon/lat (not a domain corner), local (x,y) is
    # a plain equirectangular offset, then the parallax term is
    # SUBTRACTED.
    #
    # SECOND BUG FIX (2026-10-22): the comment here used to claim "no
    # cos(lat) term -- fine here since dx/dy are already isotropic by
    # construction" -- that was WRONG, and is exactly the same east-west
    # scale error as the dx_physical fix just above (isotropic in DEGREES
    # on the native VJ143C1 grid is not isotropic in KM away from the
    # equator). Added the same cos(lat_ref) factor to the longitude term
    # here for consistency with dx_physical above -- both must agree, or
    # the surface grid and the sensor ray positions would disagree about
    # where things physically are.
    lon0 = data["lon_land"][0, 0]
    lat0 = data["lat_land"][0, 0]
    xpos0 = (data["lon_land"] - lon0) * (6378.1370 * np.pi) / 180.0 * np.cos(np.radians(lat_ref))
    ypos0 = (data["lat_land"] - lat0) * (6378.1370 * np.pi) / 180.0
    if vza_grid is not None or vaa_grid is not None:
        if vza_grid is None or vaa_grid is None:
            raise ValueError("rt_shdom: vza_grid and vaa_grid must both be "
                             "given together, or both left None")
        vza = np.asarray(vza_grid, dtype=np.float64)
        vaa = np.asarray(vaa_grid, dtype=np.float64)
        if vza.shape != (nx, ny) or vaa.shape != (nx, ny):
            raise ValueError("rt_shdom: vza_grid/vaa_grid shape %r/%r does "
                             "not match this tile's (nx, ny)=(%d, %d)"
                             % (vza.shape, vaa.shape, nx, ny))
    else:
        vza = np.full((nx, ny), tm["vza"])
        vaa = np.full((nx, ny), tm["vaa"])
    xpos = -altitude_km * np.tan(np.radians(vza)) * np.sin(np.radians(vaa)) + xpos0
    ypos = -altitude_km * np.tan(np.radians(vza)) * np.cos(np.radians(vaa)) + ypos0

    # BUG FIX (Sebastian, 2026-10): rt_shdom()'s own `out_root` default
    # ("tmp-data", relative to cwd) is NEVER wired to main()'s --out-dir --
    # main() calls rt_shdom() without an out_root= argument at all, so
    # EVERY run of this script, regardless of site/tile-size/--out-dir,
    # has been reading and writing the exact same shared working directory:
    # tmp-data/run_tile_rt/<wavelength>/rad_<solver>/. shdom_ng() is called
    # below with force=False, which (per er3t.rtm.shd.shdom.py's nml_init:
    # "if self.overwrite and self.force: INSAVEFILE='NONE' else:
    # INSAVEFILE=self.fnames_sav[ig]") tells SHDOM's Fortran solver to
    # RESTORE its internal radiative-field state from whatever
    # shdom-sav_g-*.sHdOmNG-sav checkpoint already happens to sit in that
    # directory -- left over from the PREVIOUS call to this function,
    # regardless of whether that call used a different domain size,
    # geometry, or BRDF. Confirmed happening in practice: the 500 km
    # verification run's own console log showed "Message [_restoreState]:
    # State restored from .../rad_3d/shdom-sav_g-000.sHdOmNG-sav.iproc-0000"
    # followed by convergence in just 2 iterations -- consistent with the
    # solver starting from an already-converged, but stale and mismatched,
    # prior state rather than doing a genuine fresh solve of the current
    # inputs. This plausibly explains the ocean reflectance coming out
    # essentially independent of windspeed (diagnosed 2026-10): the output
    # may be dominated by inherited stale state, not the current per-cell
    # ocean BRDF. Hong's own production code also passes force=False
    # (cam_l2.cld.run_sim_rad(), confirmed by reading his source) with the
    # SAME restore-from-save-file behavior -- but his launcher
    # (run_local.sh) runs `rm -rf tmp-data` before every single invocation,
    # so he never has a stale checkpoint sitting there to restore from. We
    # had no equivalent cleanup. Fix: wipe this call's own SHDOM working
    # directory before every run, so INSAVEFILE never points at a file left
    # over from a previous, unrelated call -- matching Hong's actual
    # operational discipline instead of relying on force=True (which would
    # also work, but this is the more surgical fix and doesn't change
    # shdom_ng's own default call convention, which still matches Hong's).
    rad_dir = "%s/rad_%s" % (fdir, solver.lower())
    shutil.rmtree(rad_dir, ignore_errors=True)

    shd0 = er3t.rtm.shd.shdom_ng(
        date=datetime.datetime(data["tile_overpass_utc"].year,
                                data["tile_overpass_utc"].month,
                                data["tile_overpass_utc"].day),
        atm_1ds=[atm1d0], atm_3ds=[atm3d0], surface=sfc_2d,
        Ng=abs0.Ng, Nmu=8, Nphi=16, Niter=Niter,
        sol_acc=1.0e-4, split_acc=1.0e-4, target="radiance",
        solar_zenith_angle=tm["sza"], solar_azimuth_angle=tm["saa"],
        sensor_type="camera3",
        sensor_zenith_angles=vza, sensor_azimuth_angles=vaa,
        sensor_xpos=xpos, sensor_ypos=ypos, sensor_altitude=altitude_km,
        sensor_dx=dx_physical, sensor_dy=dy,
        fdir=rad_dir, solver=solver,
        Ncpu=Ncpu, mp_mode="mpi", overwrite=True, force=False,
        # bcflag: back to the shdom_ng default (0, periodic), matching
        # Hong's own cld.py usage -- Sebastian's call, 2026-10-22.
        #
        # We tried bcflag=3 (open in X & Y) on 2026-10-22 to work around
        # a concrete, reproducible edge-row artifact (one edge row's TOA
        # output came back nearly identical to the OPPOSITE edge's real
        # surface signal -- true open-ocean reflectance ~0.002 at the
        # south edge read back at TOA as ~0.345, matching true land
        # ~0.35-0.38 at the north edge -- while the very next row in was
        # completely normal: a hard, isolated jump, not a gradually
        # decaying physical coupling). That override turned out to have
        # ZERO effect: `rt_shdom()` runs with `mp_mode="mpi"`, and
        # SHDOMNG's own MPI domain-decomposition setup
        # (`shd_mpi__map()` in shdMpi.f90) unconditionally forces
        # periodic BC in both directions whenever more than one process
        # is used, regardless of the requested BCFLAG ("Multi-processor
        # BC takes precedence over open BCs ... multiple processor
        # requires a periodic full domain"). Forcing truly serial
        # (Ncpu=1) to let bcflag actually take effect then hit a SEPARATE
        # SHDOMNG bug: `sensor_type="camera3"` (CAMID=3) is only
        # implemented in the MPI ray-radiance routine
        # (`shd_mpi__computeRadRay`), not the serial one
        # (`shd_rad__computeRadRay`), so a serial run just returns
        # garbage (one NaN ray, then `-1.0` sentinel for everything
        # else) -- not a real RT solution at all.
        # Net: with the current compiled SHDOMNG binary there is no way
        # to get a correctly-computed, non-periodic V3 camera-ray
        # result. Reverted to the regular/default (periodic, Ncpu=4,
        # mp_mode="mpi") configuration -- same as Hong's own cld.py --
        # while we wait to hear back from him (see docs/
        # L1B_TO_L2_ROADMAP.md "Hong input" section for the full
        # writeup and the email sent). See also the
        # `rt_domain_margin_cells` padding workaround below, which
        # sidesteps the artifact by discarding the outer edge rows/cols
        # instead of trying to change the boundary condition at all.
    )

    rad = np.fromfile(f"{shd0.fnames_out[0]}.sHdOmNG-out", dtype="<f4").reshape((nx, ny), order="C")
    refl_app = np.pi * rad / (F0_555NM * np.cos(np.radians(tm["sza"])))
    return rad, refl_app


# ── phase 4: plot (2x2, per Sebastian's final layout) ────────────────────

FRAME_COLORS = ["#4C78A8", "#E4572E", "#54A24B", "#EECA3B", "#9D6CAF",
                "#17A2A2", "#E377C2", "#8C6D31", "#B4B4B4", "#1F1F1F"]


def _map_panel(ax, lon_grid, lat_grid, values, title, vmin, vmax, cmap="viridis"):
    if est.HAS_CARTOPY:
        import cartopy.crs as ccrs
        mesh = ax.pcolormesh(lon_grid, lat_grid, values, transform=ccrs.PlateCarree(),
                              cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        # Constrain the GeoAxes to the tile's own small extent (good
        # practice regardless, and confirmed harmless) before adding
        # coastlines/gridlines -- otherwise the axes default to the
        # projection's full native extent. (NOTE: the actual root cause of
        # the 2026-10 map-panel blowup turned out to be a units bug in
        # elevation_sweep_timing.add_scale_bar -- see that function's own
        # fix -- not the missing set_extent here. Keeping this anyway since
        # it's still the correct thing to do.)
        lon_min, lon_max = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
        lat_min, lat_max = float(np.nanmin(lat_grid)), float(np.nanmax(lat_grid))
        pad_lon = max(0.02, 0.08 * max(lon_max - lon_min, 1e-6))
        pad_lat = max(0.02, 0.08 * max(lat_max - lat_min, 1e-6))
        ax.set_extent([lon_min - pad_lon, lon_max + pad_lon,
                       lat_min - pad_lat, lat_max + pad_lat], crs=ccrs.PlateCarree())
        est._add_coastlines_if_available(ax)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.7)
        gl.top_labels = gl.right_labels = False
        est.add_scale_bar(ax, use_cartopy=True)
    else:
        mesh = ax.pcolormesh(lon_grid, lat_grid, values, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading="auto")
        mean_lat = float(np.nanmean(lat_grid))
        ax.set_aspect(1.0 / max(0.15, np.cos(np.radians(mean_lat))))
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        est.add_scale_bar(ax, use_cartopy=False)
    ax.set_title(title, fontsize=10.5)
    return mesh


def plot_output(data, pooled, refl_noatm, refl_shdom, out_png, site_label,
                 altitude_km, wavelength_nm, surface_mode="brdf"):
    import matplotlib.pyplot as plt

    tm = data["tile_mean_angles"]
    lon_grid, lat_grid = data["lon_land"], data["lat_land"]

    vmin = float(np.nanmin([np.nanmin(refl_noatm), np.nanmin(refl_shdom)]))
    vmax = float(np.nanmax([np.nanmax(refl_noatm), np.nanmax(refl_shdom)]))

    glint_east_km, glint_north_km, glint_r_km = solar_glint_offset_km(
        tm["sza"], tm["saa"], altitude_km)
    glint_half_km = data.get("tile_half_km", 50.0)
    print("  expected ocean sun-glint spot: %.1f km away at bearing %.1f deg "
          "(tile half-width %.1f km -- %s)"
          % (glint_r_km, tm["saa"] % 360.0, glint_half_km,
             "falls WITHIN this tile" if max(abs(glint_east_km), abs(glint_north_km)) <= glint_half_km
             else "well outside this tile"))

    use_cartopy = est.HAS_CARTOPY
    fig = plt.figure(figsize=(14.5, 14.5))
    # dedicated thin row for the shared top-row colorbar, so it never
    # collides with the bottom row's own titles/axes.
    gs = fig.add_gridspec(3, 2, height_ratios=[10, 0.5, 9], hspace=0.38, wspace=0.25)
    if use_cartopy:
        import cartopy.crs as ccrs
        proj = est.choose_cartopy_projection(lat_grid.ravel(), lon_grid.ravel())
        ax00 = fig.add_subplot(gs[0, 0], projection=proj)
        ax01 = fig.add_subplot(gs[0, 1], projection=proj)
    else:
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, :])
    ax10 = fig.add_subplot(gs[2, 0])
    # ax11 repurposed (Sebastian, 2026-10) from a local-km VZA scatter to a
    # geographic wind-speed map -- needs the same projection as ax00/ax01
    # since it's now plotted on the lon/lat tile grid, not local km.
    if use_cartopy:
        ax11 = fig.add_subplot(gs[2, 1], projection=proj)
    else:
        ax11 = fig.add_subplot(gs[2, 1])

    m0 = _map_panel(ax00, lon_grid, lat_grid, refl_noatm,
                    "No atmosphere (BRDF-only) reflectance", vmin, vmax)
    m1 = _map_panel(ax01, lon_grid, lat_grid, refl_shdom,
                    "SHDOM clear-sky, with atmosphere\n(apparent reflectance -- see units caveat in script)",
                    vmin, vmax)
    cbar = fig.colorbar(m1, cax=cax, orientation="horizontal")
    cbar.set_label("reflectance [unitless]")

    # expected sun-glint direction/location, marked on all four panels (see
    # solar_glint_offset_km's docstring) -- the map panels need the km
    # offset converted to lon/lat first (same linear approx tile() itself
    # uses), the local-km panels take it directly.
    lat0_pt, lon0_pt = pooled["lat0"], pooled["lon0"]
    dlat_per_km = 1.0 / 111.32
    dlon_per_km = 1.0 / (111.32 * max(0.05, np.cos(np.radians(lat0_pt))))

    def _km_to_lonlat(e_km, n_km):
        return lon0_pt + e_km * dlon_per_km, lat0_pt + n_km * dlat_per_km

    _map_transform = ccrs.PlateCarree() if use_cartopy else None
    for _ax in (ax00, ax01):
        _mark_glint_direction(_ax, glint_east_km, glint_north_km, glint_r_km,
                              glint_half_km, to_data_xy=_km_to_lonlat,
                              transform_kw=_map_transform)

    # bottom-left: local-km, radiometer location + camera stripes by frame
    x_km, y_km, frame_id = pooled["x_km"], pooled["y_km"], pooled["frame_id"]
    in_tile = pooled["in_tile"]
    half_km = max(np.nanmax(np.abs(x_km[in_tile])), np.nanmax(np.abs(y_km[in_tile])), 1.0)
    half_km = data.get("tile_half_km", half_km)

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    uniq_frames = sorted(set(frame_id[in_tile].tolist()))
    frame_color_map = {fr: FRAME_COLORS[i % len(FRAME_COLORS)] for i, fr in enumerate(uniq_frames)}
    for fr in uniq_frames:
        m = in_tile & (frame_id == fr)
        ax10.scatter(x_km[m], y_km[m], s=2.0, color=frame_color_map[fr],
                     alpha=0.55, linewidths=0, zorder=2)
    fw, fl = footprint_axes_km(tm["vza"], altitude_km)
    e = mpatches.Ellipse((0, 0), width=fw, height=fl, angle=-tm["vaa"],
                         fill=False, edgecolor="#c0392b", linewidth=1.8,
                         alpha=0.9, zorder=4)
    ax10.add_patch(e)
    ax10.plot(0, 0, "+", color="#c0392b", ms=10, mew=2, zorder=5)
    ax10.set_xlim(-half_km, half_km)
    ax10.set_ylim(-half_km, half_km)
    ax10.set_aspect("equal")
    ax10.axhline(0, color="#dddddd", lw=0.5, zorder=0)
    ax10.axvline(0, color="#dddddd", lw=0.5, zorder=0)
    ax10.set_xlabel("east-west [km]")
    ax10.set_ylabel("north-south [km]")
    ax10.set_title("Camera pixels by frame (%d frames) + radiometer location"
                   % len(uniq_frames), fontsize=10.5)
    _mark_glint_direction(ax10, glint_east_km, glint_north_km, glint_r_km, half_km)
    handles10 = [
        Line2D([0], [0], marker=".", linestyle="none", markerfacecolor="0.4",
              markeredgecolor="0.4", markersize=8,
              label="camera pixels (color = source frame, %d frames)" % len(uniq_frames)),
        Line2D([0], [0], marker="+", linestyle="none", color="#c0392b", markersize=10,
              label="radiometer location (ellipse: representative, tile-mean VZA/VAA --\n"
                    "this file's own per-footprint VZA is an unpopulated placeholder)"),
    ]
    ax10.legend(handles=handles10, fontsize=6.5, loc="upper right", framealpha=0.9)

    # bottom-right: geographic map of ocean wind speed (Sebastian, 2026-10 --
    # was "camera pixels colored by VZA"; swapped out to diagnose the
    # two-brightness-tier pattern seen in the no-atmosphere ocean BRDF panel
    # -- Cox-Munk glint reflectance is a strong function of wind speed
    # (rougher sea = broader/dimmer glint peak, calmer sea = narrower/
    # brighter), so a real wind-speed gradient across the tile is a
    # plausible, non-bug explanation worth checking directly rather than
    # guessing from the reflectance pattern alone). brdf[...,0] is
    # windspeed for ocean cells (flag==1) only -- it means something else
    # (fiso) for land, so non-ocean cells are masked to NaN here.
    sfc_flag_grid = data["brdf"][..., 6]
    windspeed_grid = np.where(sfc_flag_grid == 1.0, data["brdf"][..., 0], np.nan)
    if np.isfinite(windspeed_grid).any():
        ws_vmin = float(np.nanmin(windspeed_grid))
        ws_vmax = float(np.nanmax(windspeed_grid))
        m_ws = _map_panel(ax11, lon_grid, lat_grid, windspeed_grid,
                          "Ocean wind speed (MERRA-2, drives Cox-Munk glint)",
                          ws_vmin, ws_vmax, cmap="cividis")
        fig.colorbar(m_ws, ax=ax11, fraction=0.046, pad=0.04, label="wind speed [m/s]")
        _mark_glint_direction(ax11, glint_east_km, glint_north_km, glint_r_km,
                              glint_half_km, to_data_xy=_km_to_lonlat,
                              transform_kw=_map_transform)
    else:
        ax11.text(0.5, 0.5, "no ocean cells in this tile", ha="center", va="center",
                 transform=ax11.transAxes)
        ax11.set_title("Ocean wind speed (MERRA-2, drives Cox-Munk glint)", fontsize=10.5)

    mode_note = "" if surface_mode == "brdf" else "  [SHDOM surface-mode=%s]" % surface_mode
    fig.suptitle(
        "%s -- %g nm, tile-mean SZA=%.2f SAA=%.2f VZA=%.2f VAA=%.2f deg%s"
        % (site_label, wavelength_nm, tm["sza"], tm["saa"], tm["vza"], tm["vaa"], mode_note),
        fontsize=13, y=0.995)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("  wrote %s" % out_png)

    # Sanity check: at figsize=(14.5,14.5) and dpi=150, a correctly-scoped
    # figure should be on the order of ~2000x2000 px. A bug in a cartopy
    # map panel (e.g. a units mistake feeding an absurd length into
    # ax.plot()/ax.text(), or a GeoAxes extent defaulting to near-global)
    # can blow fig.savefig(bbox_inches="tight") out to tens or hundreds of
    # thousands of pixels wide while the actual content stays tiny in one
    # corner (seen 2026-10 on the ocean-site run: 102154x2019 px, root
    # cause was a units bug in elevation_sweep_timing.add_scale_bar).
    # Catch that immediately rather than silently shipping an unusable
    # image -- cheap insurance against the next version of this class of
    # bug, now that this exact one is fixed.
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # we need to inspect a bad/huge image too, not reject it
        with Image.open(out_png) as im:
            w, h = im.size
        if max(w, h) > 8000:
            print("  WARNING: saved figure is %dx%d px -- expected roughly "
                  "2000x2000 at dpi=150. Check the cartopy map-panel code "
                  "(_map_panel / add_scale_bar) for a units or extent bug."
                  % (w, h))
    except ImportError:
        pass
    except Exception as exc:
        print("  (post-save image-size sanity check skipped: %s)" % exc)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=list(SITE_PRESETS) + ["custom"], required=True)
    p.add_argument("--seed-lat", type=float, help="required if --site custom")
    p.add_argument("--seed-lon", type=float, help="required if --site custom")
    p.add_argument("--cam-l1b", required=True, help="path to the CAM L1B granule")
    p.add_argument("--rad-l1b", required=True, help="path to the RAD-4CH L1B granule")
    p.add_argument("--l2-cam-root",
                   default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="path to this repo's own root (contains src/, software/er3t, "
                        "software/shdom, data/) -- defaults to this repo, override only "
                        "if relocated")
    p.add_argument("--tile-km", type=float, default=100.0)
    p.add_argument("--max-vza-deg", type=float, default=10.0,
                   help="secondary sanity check: exclude any pooled camera "
                        "pixel farther than this from nadir before averaging "
                        "(see tile()'s docstring -- --components is the main "
                        "angle-range control, this just catches stragglers)")
    p.add_argument("--components", default="nadir",
                   help="comma-separated camera stripe component(s) to pool "
                        "(choices: nadir, forward, cross). Default is nadir "
                        "ONLY -- forward/cross are genuinely oblique stripes "
                        "and will blow out the VZA range if included; only "
                        "widen this if you deliberately want multi-angle "
                        "coverage, with --max-vza-deg raised to match.")
    p.add_argument("--dx", type=float, default=5.5, help="surface/BRDF grid spacing, km")
    p.add_argument("--dy", type=float, default=5.5)
    p.add_argument("--altitude-km", type=float, default=831.16, help="Libera altitude")
    p.add_argument("--wavelength", type=float, default=555.0, help="nm")
    p.add_argument("--window-s", type=float, default=90.0,
                   help="camera-frame search window, +/- seconds around the matched "
                        "radiometer sample")
    p.add_argument("--scan-pad-s", type=float, default=400.0,
                   help="camera-frame pre-pass window, +/- seconds around the matched "
                        "sample (keep well above --window-s; see chop()'s docstring)")
    p.add_argument("--max-match-vza-deg", type=float, default=None,
                   help="restrict the RAD-4CH seed match to samples the radiometer "
                        "itself saw within this many degrees of VZA, instead of the "
                        "closest ground sample at any viewing angle. RAD-4CH sweeps "
                        "telescope elevation across a wide swath, co-swept by the "
                        "camera by design -- but past a VZA of ~63-64 deg the camera's "
                        "known FSW reach bug makes one side of the swath collapse to "
                        "ZERO pixels (the other side instead balloons to 1000s due to "
                        "near-limb footprint distortion). A match beyond that threshold "
                        "on the bad side gives tile() zero camera pixels no matter how "
                        "wide --window-s/--tile-km are. ~55 deg keeps the match "
                        "comfortably clear of that cutoff on either side; see chop()'s "
                        "docstring.")
    p.add_argument("--stride", type=int, default=1,
                   help="camera-pixel stride passed to pool_stripe_pixels (1 = every "
                        "pixel; this is a small tile, so no need to thin like the "
                        "full-swath diagnostics do)")
    p.add_argument("--ncpu", type=int, default=4)
    p.add_argument("--niter", type=int, default=10)
    p.add_argument("--solver", default="3D")
    p.add_argument("--surface-mode", choices=["brdf", "lambertian"], default="brdf",
                   help="BRDF diagnostic list item #5: 'brdf' feeds SHDOM the real "
                        "per-cell BRDF (RTLSR/RTLSR-Jiao/Cox-Munk, same as always). "
                        "'lambertian' instead gives SHDOM a per-cell ISOTROPIC albedo "
                        "equal to that same cell's actual BRDF reflectance evaluated "
                        "at this tile's own mean sun/view geometry (see rt_shdom()'s "
                        "docstring for why this -- and not some other albedo -- is "
                        "the fair 'Lambertian counterpart' to compare against). Only "
                        "changes the SHDOM (with-atmosphere) run -- the no-atmosphere "
                        "panel is mathematically identical in both modes by "
                        "construction, so it is left alone.")
    p.add_argument("--margin-cells", type=int, default=2,
                   help="pad the RT/BRDF domain this many extra VJ143C1-"
                        "native grid cells beyond --tile-km on every side, "
                        "then discard that margin from the result before "
                        "using/plotting/saving it (crop_tile_margin()). "
                        "Workaround for the SHDOM periodic-BC artifact that "
                        "shows up when running multi-process (see "
                        "docs/L1B_TO_L2_ROADMAP.md 'Hong input' section) -- "
                        "default changed to 2 on 2026-10-22 to match "
                        "run_dev_step1.py (verified against real data); "
                        "pass --margin-cells 0 explicitly for the regular/ "
                        "un-padded tile.")
    p.add_argument("--out-dir", default="output_l2")
    return p.parse_args()


def main():
    print(TODO_BANNER)
    args = parse_args()
    t_total0 = time.time()

    if args.site == "custom":
        if args.seed_lat is None or args.seed_lon is None:
            raise SystemExit("--site custom requires --seed-lat and --seed-lon")
        seed_lat, seed_lon, site_label = args.seed_lat, args.seed_lon, "custom site"
    else:
        preset = SITE_PRESETS[args.site]
        seed_lat, seed_lon, site_label = preset["seed_lat"], preset["seed_lon"], preset["label"]
        print("site preset %r: %s (%s)" % (args.site, site_label, preset["note"]))

    timings = {}

    t0 = _log_phase("chop")
    match = chop(args.rad_l1b, args.cam_l1b, seed_lat, seed_lon, args.window_s,
                scan_pad_s=args.scan_pad_s,
                max_match_vza_deg=args.max_match_vza_deg)
    timings["chop"] = _phase_end("chop", t0)

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())
    t0 = _log_phase("tile")
    data, pooled = tile(args.cam_l1b, match, args.tile_km, args.dx, args.dy,
                        args.l2_cam_root, stride=args.stride,
                        max_vza_deg=args.max_vza_deg, components=components,
                        margin_cells=args.margin_cells)
    data["tile_half_km"] = args.tile_km / 2.0
    timings["tile"] = _phase_end("tile", t0)

    t0 = _log_phase("rt (no atmosphere)")
    refl_noatm, sfc_flag = rt_noatm(data)
    timings["rt_noatm"] = _phase_end("rt (no atmosphere)", t0)

    t0 = _log_phase("rt (SHDOM, with atmosphere, surface-mode=%s)" % args.surface_mode)
    rad_shdom, refl_shdom = rt_shdom(data, args.l2_cam_root, args.wavelength,
                                     args.altitude_km, Niter=args.niter,
                                     Ncpu=args.ncpu, solver=args.solver,
                                     surface_mode=args.surface_mode)
    timings["rt_shdom"] = _phase_end("rt (SHDOM, with atmosphere)", t0)

    # crop_tile_margin() (2026-10-22): see run_dev_step1.py's own call site
    # for the full rationale -- a no-op when margin_cells=0 (the default).
    data, (rad_shdom, refl_shdom, refl_noatm, sfc_flag) = crop_tile_margin(
        data, args.margin_cells,
        extra_arrays=(rad_shdom, refl_shdom, refl_noatm, sfc_flag))

    # DIAGNOSTIC (temporary -- 2026-10, tracking down the localized bright
    # cell(s) Sebastian spotted in the reflectance maps): dump the raw
    # per-cell brdf/flag arrays and both reflectance maps to .npz so they
    # can be inspected numerically rather than eyeballed off the PNG.
    mode_suffix = "" if args.surface_mode == "brdf" else "_%s" % args.surface_mode
    # bug fix (Sebastian, 2026-10): args.out_dir was never created before this
    # np.savez call -- every prior run happened to reuse an --out-dir that
    # already existed (the default "output_l2", or one from an earlier run),
    # masking this. plot_output() below already creates its own out_png's
    # parent dir; do the same here so a brand-new --out-dir works on its own.
    os.makedirs(args.out_dir, exist_ok=True)
    diag_npz = os.path.join(args.out_dir, "diag_%s_%04.0fnm%s.npz"
                            % (args.site, args.wavelength, mode_suffix))
    np.savez(diag_npz, brdf=data["brdf"], lon_land=data["lon_land"], lat_land=data["lat_land"],
             refl_noatm=refl_noatm, refl_shdom=refl_shdom, sfc_flag=sfc_flag,
             # raw land/ocean-mask inputs too (Sebastian, 2026-10) -- so the
             # find_nearest ice-mask-catch-all bug (step 1 of the BRDF
             # diagnostic list) can be replayed and instrumented offline,
             # without re-running the whole chop/tile/SHDOM pipeline.
             lon_sfc=data["lon_sfc"], lat_sfc=data["lat_sfc"], sfc=data["sfc"])
    print("  wrote diagnostic arrays to %s" % diag_npz)
    flag = data["brdf"][..., 6]
    print("  surface flag counts: ocean(1)=%d land(3)=%d ice(4)=%d other/unset=%d"
         % (int((flag == 1).sum()), int((flag == 3).sum()), int((flag == 4).sum()),
            int((~np.isin(flag, [1, 3, 4])).sum())))
    i0 = np.unravel_index(np.nanargmax(refl_noatm), refl_noatm.shape)
    i1 = np.unravel_index(np.nanargmax(refl_shdom), refl_shdom.shape)
    print("  no-atm max: %.5f at grid cell %s (lon=%.4f lat=%.4f) -- brdf row there: %s"
         % (refl_noatm[i0], i0, data["lon_land"][i0], data["lat_land"][i0], data["brdf"][i0]))
    print("  shdom  max: %.5f at grid cell %s (lon=%.4f lat=%.4f) -- brdf row there: %s"
         % (refl_shdom[i1], i1, data["lon_land"][i1], data["lat_land"][i1], data["brdf"][i1]))

    t0 = _log_phase("plot")
    out_png = os.path.join(args.out_dir, "tile_rt_%s_%04.0fnm%s.png"
                           % (args.site, args.wavelength, mode_suffix))
    plot_output(data, pooled, refl_noatm, refl_shdom, out_png, site_label,
               args.altitude_km, args.wavelength, surface_mode=args.surface_mode)
    timings["plot"] = _phase_end("plot", t0)

    print("\n=== timing summary ===")
    for k, v in timings.items():
        print("  %-24s %8.2f s" % (k, v))
    print("  %-24s %8.2f s" % ("TOTAL", time.time() - t_total0))
    print("\nmean no-atmosphere reflectance: %.4f" % np.nanmean(refl_noatm))
    print("mean with-atmosphere apparent reflectance: %.4f" % np.nanmean(refl_shdom))
    print("(brighter-with-atmosphere expected at this wavelength/geometry per "
          "Sebastian's prior -- check the sign/magnitude of the difference above)")
    # DIAGNOSTIC (Sebastian, 2026-10): the tile-wide mean above averages over
    # whatever land/ocean/ice mix this site happens to have -- a surface-type-
    # dependent effect (plausible: ocean specular/Cox-Munk vs land RTLSR
    # respond very differently to added diffuse skylight) would be invisible
    # in one pooled number. Break it out by surface flag (1=ocean, 3=land,
    # 4=ice) so a real anomaly can be told apart from "this site happens to
    # be land-heavy/ocean-heavy and the aggregate hides a sane per-type story".
    for _flag, _name in ((1.0, "ocean"), (3.0, "land"), (4.0, "ice")):
        _m = (sfc_flag == _flag)
        _n = int(_m.sum())
        if _n == 0:
            continue
        _noatm_mean = float(np.nanmean(refl_noatm[_m]))
        _shdom_mean = float(np.nanmean(refl_shdom[_m]))
        _sign = "brighter" if _shdom_mean > _noatm_mean else "DIMMER"
        print("  %-5s (n=%4d): no-atm=%.4f  with-atm=%.4f  (%s with atmosphere, "
              "%+.1f%%)" % (_name, _n, _noatm_mean, _shdom_mean, _sign,
                             100.0 * (_shdom_mean - _noatm_mean) / _noatm_mean))


if __name__ == "__main__":
    main()
