"""Elevation-sweep timing diagnostic, built 2026-10-07 for today's SDC meeting.

Three stacked panels, all sharing one x-axis (radiometer time, seconds since
the first usable camera frame):

  1. Radiometer Elevation [deg] vs. time -- one full scan sweep (negative
     peak -> positive peak -> negative peak again).
  2. For every radiometer sample in that sweep, the time of its nearest
     camera exposure (same nearest-frame-by-time argmin logic as
     nadir_radiometer_overlap.py's test_overlap), plotted against the
     radiometer's own time. A 1:1 reference line shows where the two would
     coincide; camera cadence (~5 s) vs. radiometer sampling shows up as a
     staircase around that line.
  3. For every radiometer sample, the distance (km) from its own footprint
     to the nearest pooled camera pixel in that assigned frame.

The shaded band on all three panels marks the camera's own active window
(its first to last usable, geolocated frame) -- the scan sweep is picked out
of the FULL radiometer timeline (whatever its duration), not just that
window, since the sweep itself is a property of the instrument's scan
mechanism and may well extend beyond the short burst of usable camera
frames. Panels 2 and 3 make that mismatch visible directly: outside the
camera's window there is only ever one candidate frame (an edge frame), so
the nearest-exposure time flattens out and the pixel distance grows.

Why "Elevation": that is the RAD-4CH field name, confirmed in
apps/l1b/inspect_l1b_rad.py's own field list (Az + Elevation panel). Not the
radiometer's Viewing_Zenith_Surface/Solar_Zenith_Surface (identically -999,
untouched here, per standing instruction).

Operational_Mode=128, NOT 132 (reverted -- see below): docs/NEXT.md's own
2026-10-07 measured entry calls Operational_Mode 128 "CT" explicitly, for
both the camera's cam_obs_id and this exact radiometer file ("14,326 are CT
(Operational_Mode 128)" out of 127,104 samples in the camera's window), with
real, scene-varying SW radiance -- not the flat/near-zero signature
space-view would show. apps/l1b/inspect_l1b_rad.py's own _MODE_DESC dict
labels 128 "space-view", which an earlier version of this docstring trusted
and used to switch the default to 132 -- that was wrong; _MODE_DESC appears
to be mislabeled (a separate fix for that file, not made here).

What's actually going on, confirmed directly from a completed run's own CSV
(2026-10-07): within Operational_Mode=128, Elevation and geo_valid
(Latitude/Longitude present) are almost perfectly correlated -- every
geo_valid sample in that run had Elevation in [-62.0, +61.7] deg, and every
non-geo_valid sample was outside it (up to +-178 deg). That is exactly the
expected behaviour of a cross-track scanner: mode 128 covers the WHOLE scan
head rotation, Earth view only while the ray still intersects the surface
(roughly the +-60..70 deg window Sebastian expected), and the rest of the
same mode/rotation is pointed past the limb into space with no Earth
lat/lon -- not a different mode, not bad data, just not Earth-view. So "one
full elevation scan sweep, horizon to horizon and back, within reasonable
bounds for CT" = the (near-)contiguous run of geo_valid samples, not the
generic local min/max/min of the raw Elevation telemetry (find_full_sweeps
below, now superseded by find_earth_view_sweeps for sweep *selection* --
kept, still unit-tested, in case raw-telemetry extrema are ever wanted for
something else).

Picking "the" sweep (find_earth_view_sweeps): group geo_valid samples into
runs, bridging internal dropouts up to --max-gap-s (default: max(3x the
median geo_valid sample spacing, 1.0 s) -- this granule's own geo_valid
samples run at ~0.01 s, so the plain 3x-median figure alone (~0.03 s) would
NOT bridge its two ~0.6 s dropouts right at the turning points, where the
Earth-intersection computation briefly fails at near-grazing incidence; the
1 s floor fixes that without risking bridging two genuinely different
overpasses, which are minutes apart, not sub-second). A gap bigger than
--max-gap-s starts a new run instead of bridging it
-- that guards against stitching two different overpasses together. Each
run's own start/end sample is the horizon crossing and its interior max is
the overpass peak. By default the sweep with the most overlap with the camera's own
active window is picked (falling back to the nearest one in time if none
overlaps) and every candidate is printed with its index so --sweep-index
can override the pick.

Duplicates pool_stripe_pixels/haversine helpers from
apps/l2/nadir_radiometer_overlap.py rather than importing it -- no precedent
in this repo for one apps/ script importing another (established
2026-10-06).

Maps (new 2026-10-07): the full-sweep and nadir-zoom footprint/camera maps
now render on a real cartopy map (south/north polar stereographic poleward
of 60 deg latitude, same as this granule's -56..-82 deg track -- plain
PlateCarree otherwise), auto-picked by choose_cartopy_projection from the
data's own mean latitude. Falls back to a plain cosine(latitude)-corrected
lon/lat scatter if cartopy isn't installed (same optional-dependency
convention as apps/l1b/inspect_l1b_rad.py's own HAS_CARTOPY guard), and
skips coastlines (gridlines only) if the Natural Earth shapefile isn't
already cached locally, rather than trying to download it.

A third map, also new: a 3-panel direction/frame diagnostic
(--diag-half-width-km, default 50 -> 100 km-wide panels each), at nadir and
at +/- --diag-offset-km (default 500 km) off nadir along the sweep. Colors
two things plot_footprint_camera_map doesn't: footprint color = sweep
direction (rising vs. falling elevation, i.e. before vs. after the sweep's
own peak), camera-pixel color = which single camera frame it came from.
One projection is chosen from all the data so the 3 panels are directly
comparable.

Two more additions (2026-10-07), both answering "is the top row's dot
pattern a selection artifact, and what does the camera's own geometry look
like there": the diagnostic above gained a SECOND row, same 3 columns, that
plots every pooled stripe-component camera pixel of the relevant frame(s)
in view -- not just the ones within neighbor_radius_km (20 km) of some
matched footprint, which is what the first row (and plot_footprint_camera_
map) actually show. The striped/scalloped look of the top row is exactly
that 20 km-per-footprint patch selection, not a rendering artifact -- the
bottom row makes the full pooled coverage directly visible for comparison.
That pooled set is itself already restricted to specific stripe components
(CT_STRIPE_COMPONENTS) at --stride (default 2), so even the bottom row is
not literally every camera pixel in the frame.

A separate new plot, plot_angle_diagnostic, is the camera's own per-pixel
viewing/solar geometry at the same 3 zoom columns: 3 rows (Solar Zenith
Angle, Viewing Zenith Angle, Relative Azimuth Angle), each camera pixel
colored by its own L1B Solar_Zenith_Surface / Viewing_Zenith_Surface /
Relative_Azimuth_Surface value (these ARE present in the CAM L1B file as
per-pixel 2-D arrays, per libcam/l2/domains.py -- a different, per-camera-
pixel copy of the same-named quantities from the radiometer's per-footprint
version, which is why this is safe to plot even though Sebastian asked
(standing instruction) to hold off on the RADIOMETER's own copies of these
fields, identically -999 in this file, until he shares his idea for that
gap). Footprints are drawn as plain position markers in this plot, not
angle-colored, for exactly that reason. A row is skipped (with a console
note) if the granule has no such variable at all, or flagged loudly if a
field comes back as an all-identical placeholder (per domains.py's own
documented "all-zero when L1B geometry is not yet populated" case).

One more plot, towards sizing RT domains (2026-10-07): plot_angle_
timeseries, a stacked SZA/VZA/RAZ-vs-radiometer-time view, one panel per
angle, covering the WHOLE sweep window (not just 3 zoom columns). For
every footprint, footprint_patch_angle_stats finds the closest --patch-
radius-km-radius patch of pooled camera pixels around its nearest match
(default 12.5 -> a 25 km-diameter patch) and computes that angle's mean/
min/max across the patch; the plot draws the mean as a line and shades
the min..max spread, so the SPATIAL variability (the shaded band's width)
and the TEMPORAL/stepwise variability (the mean jumping every time a
footprint's nearest camera frame changes, every ~5 s -- background color
bands mark which frame is currently nearest, same convention as the
diagnostic plot) are both visible in one picture, which is the basis for
deciding how large an RT domain can be before its own angles stop being
"close enough to constant."

Usage
-----
    python apps/l2/elevation_sweep_timing.py <CAM_L1B.nc> --rad <RAD-4CH.nc>
    python apps/l2/elevation_sweep_timing.py <CAM_L1B.nc> --rad <RAD-4CH.nc> --sweep-index 2
"""

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    _root = os.path.abspath(__file__)
    for _ in range(3):
        _root = os.path.dirname(_root)
    sys.path.insert(0, _root)

from libcam.mask.retrieve_mask import MaskRetriever, MASK_BITS, BIT_ANY_COMPONENT, BIT_UNEXPLAINED

# ── optional cartopy (mirrors apps/l1b/inspect_l1b_rad.py's own guard -- same
# repo convention: cartopy is an optional [full]-extra dependency, so this
# script must still run without it, just with plain lon/lat scatter maps) ──
try:
    import cartopy.crs as ccrs
    import cartopy as _cartopy
    HAS_CARTOPY = True
    _ne_110m_shp = os.path.join(
        _cartopy.config["data_dir"],
        "shapefiles", "natural_earth", "physical", "ne_110m_coastline.shp")
    HAS_NE_DATA = os.path.exists(_ne_110m_shp)
except ImportError:
    HAS_CARTOPY = False
    HAS_NE_DATA = False

FILL_VAL = -999.0
EARTH_R_KM = 6371.0

MASK_FIELDS = ["valid_pixel_mask", "Camera_Mask"]
LAT_FIELDS = ["Latitude", "Terrain_Corrected_Latitude"]
LON_FIELDS = ["Longitude", "Terrain_Corrected_Longitude"]
CT_STRIPE_COMPONENTS = ("nadir", "forward", "cross")

# per-pixel camera L1B angle fields (added 2026-10-07 for plot_angle_
# diagnostic) -- same variable names as the radiometer's own per-footprint
# copies (see libcam/l2/domains.py), but here read per-CAMERA-pixel, as a
# 2-D (NX, NY) array per frame. Not present in every CAM L1B file/schema, and
# may be an unpopulated all-identical placeholder even when present -- both
# cases are checked for and reported, never silently trusted.
ANGLE_FIELDS = {
    "sza": "Solar_Zenith_Surface",
    "vza": "Viewing_Zenith_Surface",
    "raz": "Relative_Azimuth_Surface",
    # added 2026-10-13 for the multi-frame per-box SZA/SAA/VZA/VAA range
    # analysis (box_multiframe_angles.py) -- same per-pixel-over-frame
    # pooling pool_stripe_pixels already does for sza/vza/raz, just two
    # more L1B fields, confirmed present in DITL2's CAM L1B as
    # (CAMERA_TIME, CAMERA_PIXEL_COUNT_X, CAMERA_PIXEL_COUNT_Y) float32.
    "saa": "Solar_Azimuth_Surface_WRT_North",
    "vaa": "Viewing_Azimuth_Surface_WRT_North",
}


# ─── duplicated from nadir_radiometer_overlap.py (see docstring) ───────────

def _first(ds, names):
    for n in names:
        if n in ds:
            return n
    return None


def open_granule(path):
    import xarray as xr
    return xr.open_dataset(path)


def _fpa(a):
    return np.asarray(a).T


def frame_mask(ds, i, field):
    a = _fpa(ds[field].isel(CAMERA_TIME=i).values)
    return a.astype(bool) if a.dtype == bool else (a != 0)


def frame_headers(ds):
    n = int(ds.sizes.get("CAMERA_TIME", 0))

    def col(name, default=0):
        if name not in ds:
            return np.full(n, default)
        return ds[name].values

    def col_first(names, default=0):
        for name in names:
            if name in ds:
                return ds[name].values
        return np.full(n, default)

    # "Camera_Operational_Mode" (older camera L1B, e.g. V0-2-8RC1) was
    # renamed "Camera_Observation_ID" in newer camera L1B versions --
    # confirmed 2026-10-13 against DITL2's V0-3-2 (Camera_Observation_ID
    # == 128 for every frame checked, and 128 is already a known
    # OBS_ID_MODE entry, so this is a straight rename, not a new mode).
    # Trying both keeps old and new granules working unchanged.
    mode = col_first(("Camera_Operational_Mode", "Camera_Observation_ID"), -1)
    pmid = col("pixel_mask_id", -1)
    az = col("Azimuth", 0.0)
    out = []
    for i in range(n):
        m = mode[i]
        out.append(dict(
            cam_obs_id=int(m) if np.isfinite(m) else -1,
            pixel_mask_id=int(pmid[i]),
            az=float(az[i]) if np.isfinite(az[i]) else 0.0,
            image_count=int(col("image_count", 0)[i]),
        ))
    return out


def geo_report(ds, n, frames=None):
    lat_f, lon_f = _first(ds, LAT_FIELDS), _first(ds, LON_FIELDS)
    have = np.zeros(n, dtype=bool)
    if lat_f is None or lon_f is None:
        return have, lat_f, lon_f
    for i in (range(n) if frames is None else frames):
        a = ds[lat_f].isel(CAMERA_TIME=i).values
        have[i] = bool(np.isfinite(a).any() and np.nanmax(np.abs(a)) > 0)
    return have, lat_f, lon_f


def haversine_matrix_km(lat_a, lon_a, lat_b, lon_b):
    """All-pairs: (N,) x (M,) -> (N, M) km. Broadcasts; keep N*M modest."""
    p1 = np.radians(lat_a)[:, None]
    p2 = np.radians(lat_b)[None, :]
    dphi = np.radians(lat_b)[None, :] - np.radians(lat_a)[:, None]
    dlmb = np.radians(lon_b)[None, :] - np.radians(lon_a)[:, None]
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def pool_stripe_pixels(granule, components=CT_STRIPE_COMPONENTS, max_frames=None,
                       first_frame=0, stride=2, table_dir=None, verbose=True,
                       extra_fields=None):
    """Returns per_frame: list of dicts frame, time_ns, lat, lon, comp_code,
    plus sza/vza/raz (per-pixel Solar_Zenith_Surface / Viewing_Zenith_Surface
    / Relative_Azimuth_Surface, same per-pixel selection and order as lat/
    lon) whenever those L1B variables are present in the granule -- added
    2026-10-07 for plot_angle_diagnostic. A key is simply absent from every
    dict if the granule has no such variable at all (checked once, not
    per-frame, since variable presence is a property of the file/schema).
    Otherwise identical to nadir_radiometer_overlap.py's function of the
    same name (minus the nadir-only companion list, not needed here).

    extra_fields (added 2026-10-22, for run_tile_rt.tile()'s "measured"-
    vs-RT radiance comparison): an optional {output_key: varname} dict of
    MORE per-pixel fields to pool, pulled with the exact same selection/
    orientation/stride as lat/lon/angle_vars above. Each key is simply
    absent from every dict (same convention as the angle fields) if
    `varname` isn't in the granule at all -- e.g. a plain, non-proxy L1B
    has no "Radiance"/"L1X_Proxy_Mask" field yet, so a caller requesting
    those gets them back only once run against an apps/l2/build_l1x.py
    "L1X" file."""
    ds = open_granule(granule)
    mask_f = _first(ds, MASK_FIELDS)
    if mask_f is None:
        raise SystemExit("no pixel-mask field in %s" % granule)

    angle_vars = {k: v for k, v in ANGLE_FIELDS.items() if v in ds}
    missing_angle_vars = [v for v in ANGLE_FIELDS.values() if v not in ds]
    if missing_angle_vars and verbose:
        print("  (camera L1B file has no %s variable -- that angle "
             "diagnostic row will be skipped)" % ", ".join(missing_angle_vars))

    extra_fields = extra_fields or {}
    extra_vars = {k: v for k, v in extra_fields.items() if v in ds}
    missing_extra_vars = [v for v in extra_fields.values() if v not in ds]
    if missing_extra_vars and verbose:
        print("  (camera L1B file has no %s variable -- that extra "
             "per-pixel field will be skipped, e.g. a plain L1B with no "
             "L1X proxy data yet)" % ", ".join(missing_extra_vars))

    n_all = int(ds.sizes["CAMERA_TIME"])
    f0 = max(0, min(first_frame, n_all - 1))
    n = n_all if max_frames is None else min(f0 + max_frames, n_all)
    frames = list(range(f0, n))
    hdrs = frame_headers(ds)
    have_geo, lat_f, lon_f = geo_report(ds, n_all, frames)
    if lat_f is None or lon_f is None:
        raise SystemExit("granule has no latitude/longitude field at all")

    cam_time = ds["CAMERA_TIME"].values
    r = MaskRetriever(table_dir=table_dir, verbose=False)
    comp_bits = [int(MASK_BITS[c]) for c in components]
    combined_bit = 0
    for b in comp_bits:
        combined_bit |= b

    per_frame = []
    n_attempted = 0
    n_identified_total = 0
    n_px_total = 0
    for i in frames:
        res = r.process_frame(frame_mask(ds, i, mask_f), header=hdrs[i],
                              index=i, az=hdrs[i]["az"])
        n_attempted += 1
        if not (res.status.ok and have_geo[i] and res.membership is not None):
            if verbose:
                reason = (getattr(res.status, "reason", None)
                         or getattr(res.status, "message", None)
                         or ("not geolocated" if not have_geo[i] else "no membership"))
                print("  frame %3d: SKIPPED (mask retrieval failed: %s)" % (i, reason))
            continue
        lat = _fpa(ds[lat_f].isel(CAMERA_TIME=i).values)
        lon = _fpa(ds[lon_f].isel(CAMERA_TIME=i).values)
        nb = res.membership.astype(np.uint32)

        # "% identified" diagnostic (added 2026-10-13, after the
        # Camera_Operational_Mode -> Camera_Observation_ID rename silently
        # zeroed out every pixel for a whole granule with no visible
        # symptom other than an empty per_frame list downstream): fraction
        # of the WHOLE frame (not just this call's selected components)
        # MaskRetriever could assign to ANY known object, named component
        # or "unexplained" -- a direct canary for exactly that kind of
        # silent coverage loss. Carried through as a per-pixel array (one
        # constant value, same length as this frame's selected pixels) so
        # it rides along for free through footprint_patch_angle_stats'
        # existing per-pixel mean/min/max machinery as just another
        # angle-like key -- see its angle_keys argument at the call site.
        identified = (nb & int(BIT_ANY_COMPONENT | BIT_UNEXPLAINED)) != 0
        pct_identified = 100.0 * float(identified.sum()) / identified.size
        n_identified_total += int(identified.sum())
        n_px_total += identified.size

        is_sel = (nb & combined_bit) != 0
        rows, cols = np.where(is_sel[::stride, ::stride])
        rows, cols = rows * stride, cols * stride
        la, lo = lat[rows, cols], lon[rows, cols]
        good = np.isfinite(la) & np.isfinite(lo) & (la > FILL_VAL) & (lo > FILL_VAL)
        if good.any():
            entry = dict(frame=i, time_ns=int(cam_time[i]),
                        lat=la[good], lon=lo[good],
                        pct_identified=np.full(int(good.sum()), pct_identified))
            for k, varname in angle_vars.items():
                av = _fpa(ds[varname].isel(CAMERA_TIME=i).values)
                entry[k] = av[rows, cols][good]
            for k, varname in extra_vars.items():
                ev = _fpa(ds[varname].isel(CAMERA_TIME=i).values)
                entry[k] = ev[rows, cols][good]
            per_frame.append(entry)
        if verbose:
            print("  frame %3d: %6d px in {%s} (post-stride, geolocated), "
                 "%.1f%% of frame identified to some object"
                 % (i, int(good.sum()), ",".join(components), pct_identified))
    if verbose and n_attempted:
        print("  mask retrieval: %d/%d attempted frames usable, %.1f%% of "
             "pixels identified overall (whole-frame, all objects)"
             % (len(per_frame), n_attempted,
                100.0 * n_identified_total / n_px_total if n_px_total else 0.0))
    return per_frame


def scan_camera_frame_times(granule, first_frame=0, max_frames=None):
    """Cheap pre-pass over CAMERA_TIME + geolocation presence only -- NOT the
    expensive MaskRetriever.process_frame call pool_stripe_pixels needs for
    pixel-mask membership. Added 2026-10-07: Sebastian pointed out run() was
    processing every frame in the whole granule through MaskRetriever even
    though only a handful of frames near the picked sweep window ever get
    used downstream. This lets run() get the camera time window (t_ref/
    camera_t0/camera_t1 -- needed up front, before a sweep can even be
    picked, since pick_sweep needs the camera's active-window overlap)
    without paying for the full expensive pass, and then narrow the frame
    range handed to the real pool_stripe_pixels call (see narrow_frame_
    range) to just the frames that can possibly be used.

    Returns (frame_idx, time_ns): both 1-D int64 arrays, same length,
    time-sorted (CAMERA_TIME is monotonic and frames are walked in order),
    covering only frames with have_geo True in [first_frame, first_frame +
    max_frames) -- i.e. the frames pool_stripe_pixels could possibly use,
    though it may still reject a few of these for other reasons (a
    MaskRetriever failure, or zero selected+geolocated pixels after stride/
    fill-value filtering)."""
    ds = open_granule(granule)
    n_all = int(ds.sizes["CAMERA_TIME"])
    f0 = max(0, min(first_frame, n_all - 1))
    n = n_all if max_frames is None else min(f0 + max_frames, n_all)
    frames = list(range(f0, n))
    have_geo, lat_f, lon_f = geo_report(ds, n_all, frames)
    if lat_f is None or lon_f is None:
        raise SystemExit("granule has no latitude/longitude field at all")
    cam_time = ds["CAMERA_TIME"].values
    idx = np.array([i for i in frames if have_geo[i]], dtype=np.int64)
    return idx, cam_time[idx].astype(np.int64)


def narrow_frame_range(scan_idx, scan_time_ns, t_ref, win_t0, win_t1):
    """Given scan_camera_frame_times' (scan_idx, scan_time_ns) -- every
    geolocated candidate frame in the full requested range, time-sorted --
    and a time window [win_t0, win_t1] (seconds since t_ref, same convention
    as everywhere else in this script), return (first_frame, max_frames) for
    the real pool_stripe_pixels call: the smallest CONTIGUOUS original-
    frame-index range provably wide enough to contain the globally-nearest
    usable frame for every radiometer sample in the window.

    A bracketing technique, not a heuristic cadence-based pad guess: find
    the candidate frames bracketing the window via searchsorted, then
    extend by exactly one index on each side -- so even a sample sitting
    right at the window edge, whose nearest usable frame happens to fall
    just outside it, is still covered."""
    scan_idx = np.asarray(scan_idx)
    t_s = (np.asarray(scan_time_ns, dtype=np.float64) - t_ref) / 1e9
    lo = int(np.searchsorted(t_s, win_t0, side="left"))
    hi = int(np.searchsorted(t_s, win_t1, side="right")) - 1
    lo = max(0, lo - 1)
    hi = min(len(t_s) - 1, hi + 1)
    if hi < lo:
        # window missed every candidate frame entirely (shouldn't normally
        # happen -- pick_sweep only picks sweeps overlapping the camera
        # window) -- fall back to the whole scanned range rather than an
        # inverted/empty slice.
        lo, hi = 0, len(t_s) - 1
    first = int(scan_idx[lo])
    last = int(scan_idx[hi])
    return first, last - first + 1


def load_radiometer_elevation(rad_path, mode=128):
    """Elevation timeline for the requested Operational_Mode, independent of
    geolocation validity (a dropout in Latitude/Longitude shouldn't erase an
    otherwise-good elevation/timing sample -- see geo_valid)."""
    import xarray as xr
    ds = xr.open_dataset(rad_path)

    def _get(name):
        return np.asarray(ds[name].values) if name in ds else None

    time = np.asarray(ds["radiometer_time"].values, dtype=np.int64)
    elevation = _get("Elevation")
    if elevation is None:
        raise SystemExit(
            "no 'Elevation' field in %s -- available fields: %s"
            % (rad_path, ", ".join(sorted(ds.data_vars))))
    lat, lon = _get("Latitude"), _get("Longitude")
    mode_arr = _get("Operational_Mode")

    valid = np.isfinite(elevation) & (elevation > FILL_VAL)
    if mode_arr is not None:
        valid &= (mode_arr == mode)

    n = int(valid.sum())
    out = dict(time=time[valid], elevation=elevation[valid])
    out["lat"] = lat[valid] if lat is not None else np.full(n, np.nan)
    out["lon"] = lon[valid] if lon is not None else np.full(n, np.nan)
    out["geo_valid"] = (np.isfinite(out["lat"]) & np.isfinite(out["lon"]) &
                       (out["lat"] > FILL_VAL) & (out["lon"] > FILL_VAL))
    return out


# ─── pure-numpy analysis (unit-tested on synthetic data) ───────────────────

def local_extrema(x):
    """Indices of local minima/maxima in a 1-D series (plateaus handled by
    forward-filling zero-diffs with the last nonzero direction). Returns
    (indices, kinds) with kinds[i] in {"min", "max"}, in index order."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=object)
    d = np.diff(x)
    sign = np.sign(d)
    filled = np.zeros_like(sign)
    last = 0.0
    for i, s in enumerate(sign):
        if s == 0:
            filled[i] = last
        else:
            filled[i] = s
            last = s
    idx, kinds = [], []
    for i in range(1, len(filled)):
        if filled[i - 1] > 0 and filled[i] < 0:
            idx.append(i)
            kinds.append("max")
        elif filled[i - 1] < 0 and filled[i] > 0:
            idx.append(i)
            kinds.append("min")
    return np.array(idx, dtype=int), np.array(kinds, dtype=object)


def find_full_sweeps(elevation, time_s, max_gap_s=None):
    """Every consecutive (local min, local max, local min) triple -- one full
    back-and-forth scan -- as a list of dicts, rejecting any triple whose
    internal sample spacing exceeds max_gap_s anywhere (guards against
    bridging a mode-switch data gap). max_gap_s=None: 3x the median spacing
    of the whole series."""
    elevation = np.asarray(elevation, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    idx, kinds = local_extrema(elevation)
    if max_gap_s is None:
        dt = np.diff(time_s)
        max_gap_s = 3.0 * float(np.median(dt)) if dt.size else float("inf")

    sweeps = []
    for i in range(len(idx) - 2):
        if kinds[i] == "min" and kinds[i + 1] == "max" and kinds[i + 2] == "min":
            s, p, e = int(idx[i]), int(idx[i + 1]), int(idx[i + 2])
            seg_gaps = np.diff(time_s[s:e + 1])
            if seg_gaps.size and seg_gaps.max() > max_gap_s:
                continue
            sweeps.append(dict(
                start_idx=s, peak_idx=p, end_idx=e,
                start_t=float(time_s[s]), peak_t=float(time_s[p]), end_t=float(time_s[e]),
                start_val=float(elevation[s]), peak_val=float(elevation[p]),
                end_val=float(elevation[e])))
    return sweeps


def find_earth_view_sweeps(elevation, time_s, geo_valid, max_gap_s=None):
    """The real "one full elevation scan sweep, horizon to horizon and back"
    -- NOT find_full_sweeps' generic local-min/max/min hunt through the raw
    Elevation telemetry, which also catches the much wider excursions (up to
    +-170ish deg) that happen later in the same Operational_Mode, once the
    scan head rotates past the limb into space with no Earth intersection.

    Two-stage, both needed (confirmed on the real granule, 2026-10-07):

    1. Group the geo_valid samples (the ones with a real Earth
       Latitude/Longitude) into sessions -- one continuous stretch of Earth
       viewing -- bridging any internal gap up to max_gap_s (None: max(3x
       the median geo_valid sample spacing, 1.0 s); the 1 s floor matters
       since this granule samples at ~0.01 s but has ~0.6 s dropouts right
       at individual turning points, which a plain 3x-median figure alone,
       ~0.03 s, would not bridge). A bigger gap starts a new session.

    2. WITHIN each session, run local_extrema (the same min/max/min hunt
       find_full_sweeps uses) on the elevation values AT the geo_valid
       samples themselves, to split a long session into its individual
       back-and-forth cycles. This second stage is NOT optional: a first
       version of this function treated the whole session as one sweep,
       which was wrong once a run was checked against the real granule --
       the radiometer oscillates continuously (one ~6-9 s cycle after
       another, no space-view dip in between) for the entire ~130 s
       camera-active window, with the Earth-view envelope itself (the
       +-170ish deg excursion seen briefly at the very start of the
       granule) a one-off, not a per-cycle thing. Treating that whole
       130 s session as "one sweep" silently produced a sweep containing
       ~20 real cycles, which the elevation panel plotted fine (it's just
       the data) but which fed ~20 cycles' worth of footprints into the
       footprint/camera maps at once -- a dense, unreadable band instead of
       one cycle's actual-sized set of links.

    Same dict shape as find_full_sweeps (start/peak/end _idx/_t/_val) so
    pick_sweep works on either unchanged."""
    elevation = np.asarray(elevation, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    geo_valid = np.asarray(geo_valid, dtype=bool)
    idxs = np.where(geo_valid)[0]
    if idxs.size == 0:
        return []
    if max_gap_s is None:
        dt = np.diff(time_s[idxs])
        max_gap_s = max(3.0 * float(np.median(dt)), 1.0) if dt.size else float("inf")

    breaks = np.where(np.diff(time_s[idxs]) > max_gap_s)[0]
    sessions = np.split(idxs, breaks + 1)

    sweeps = []
    for session in sessions:
        if session.size < 3:
            continue
        sub_elev = elevation[session]
        rel_idx, kinds = local_extrema(sub_elev)
        if rel_idx.size < 3:
            continue
        for i in range(len(rel_idx) - 2):
            if kinds[i] == "min" and kinds[i + 1] == "max" and kinds[i + 2] == "min":
                s = int(session[rel_idx[i]])
                p = int(session[rel_idx[i + 1]])
                e = int(session[rel_idx[i + 2]])
                sweeps.append(dict(
                    start_idx=s, peak_idx=p, end_idx=e,
                    start_t=float(time_s[s]), peak_t=float(time_s[p]), end_t=float(time_s[e]),
                    start_val=float(elevation[s]), peak_val=float(elevation[p]),
                    end_val=float(elevation[e])))
    return sweeps


def pick_sweep(sweeps, window_t0, window_t1):
    """Index of the sweep with the most overlap with [window_t0, window_t1];
    if none overlaps, the one nearest in time. None if sweeps is empty."""
    if not sweeps:
        return None

    def overlap(sw):
        return min(sw["end_t"], window_t1) - max(sw["start_t"], window_t0)

    ov = [overlap(sw) for sw in sweeps]
    best = int(np.argmax(ov))
    if ov[best] > 0:
        return best

    def gap(sw):
        if sw["end_t"] < window_t0:
            return window_t0 - sw["end_t"]
        if sw["start_t"] > window_t1:
            return sw["start_t"] - window_t1
        return 0.0

    gaps = [gap(sw) for sw in sweeps]
    return int(np.argmin(gaps))


def nearest_frame_assignment(times_ns, frame_times_ns):
    """argmin-nearest-frame assignment, identical convention to
    nadir_radiometer_overlap.py's test_overlap. Returns (nearest_frame_idx,
    dt_s) where dt_s = (sample_time - assigned_frame_time) / 1e9."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    frame_times_ns = np.asarray(frame_times_ns, dtype=np.int64)
    nearest = np.argmin(np.abs(times_ns[:, None] - frame_times_ns[None, :]), axis=1)
    dt_s = (times_ns - frame_times_ns[nearest]) / 1e9
    return nearest, dt_s


def distances_to_assigned_frame(lat, lon, geo_valid, nearest_frame_idx,
                                frame_lat_list, frame_lon_list):
    """Nearest pooled-pixel distance (km) in each sample's assigned frame.
    NaN where geo_valid is False (no footprint position to measure from) --
    left as a visible gap rather than silently dropped, since a geolocation
    dropout is itself worth seeing on the plot."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    nearest_frame_idx = np.asarray(nearest_frame_idx)
    n = len(lat)
    out = np.full(n, np.nan)
    idxs = np.where(np.asarray(geo_valid, dtype=bool))[0]
    if idxs.size == 0:
        return out
    sub_nf = nearest_frame_idx[idxs]
    for fi in np.unique(sub_nf):
        sel = idxs[sub_nf == fi]
        flat, flon = frame_lat_list[fi], frame_lon_list[fi]
        if len(flat) == 0:
            continue
        d = haversine_matrix_km(lat[sel], lon[sel], flat, flon)
        out[sel] = d.min(axis=1)
    return out


def footprint_patch_angle_stats(lat, lon, geo_valid, nearest_frame_idx, per_frame,
                                match_radius_km=50.0, patch_radius_km=12.5,
                                angle_keys=("sza", "vza", "raz")):
    """For every geo_valid footprint whose nearest pooled camera pixel (in its
    own assigned frame) is within match_radius_km, gather every pooled pixel
    within patch_radius_km of that one nearest pixel -- "the closest
    2*patch_radius_km-diameter patch of pixels belonging to this footprint",
    requested 2026-10-07 as the basis for an RT-domain sizing analysis -- and
    compute each available angle field's mean/min/max across that patch.
    Same two-stage nearest-then-neighbor-radius structure as
    footprint_camera_links, just reduced to summary stats per footprint
    instead of the full neighbor point lists (which plot_angle_timeseries
    doesn't need).

    Returns a dict {key: {"mean": arr, "min": arr, "max": arr}}, one entry
    per key in angle_keys that per_frame actually carries (per pool_stripe_
    pixels' own per-dataset ANGLE_FIELDS presence check) -- absent entirely
    if none are available. Every array is the same length as lat, NaN
    wherever geo_valid is False, the match_radius_km gate isn't met, or (for
    a key the granule otherwise has) that particular frame's patch happens
    to be empty."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    nearest_frame_idx = np.asarray(nearest_frame_idx)
    n = len(lat)
    have_keys = [k for k in angle_keys if per_frame and k in per_frame[0]]
    out = {k: dict(mean=np.full(n, np.nan), min=np.full(n, np.nan),
                  max=np.full(n, np.nan)) for k in have_keys}
    if not have_keys:
        return out
    idxs = np.where(np.asarray(geo_valid, dtype=bool))[0]
    if idxs.size == 0:
        return out
    sub_nf = nearest_frame_idx[idxs]
    for fi in np.unique(sub_nf):
        sel = idxs[sub_nf == fi]
        f = per_frame[int(fi)]
        flat, flon = f["lat"], f["lon"]
        if len(flat) == 0:
            continue
        d = haversine_matrix_km(lat[sel], lon[sel], flat, flon)
        j = np.argmin(d, axis=1)
        dmin = d[np.arange(len(sel)), j]
        for k in np.where(dmin <= match_radius_km)[0]:
            s = int(sel[k])
            jc = int(j[k])
            dn = haversine_matrix_km(np.array([flat[jc]]), np.array([flon[jc]]),
                                     flat, flon)[0]
            patch = dn <= patch_radius_km
            if not patch.any():
                continue
            for key in have_keys:
                vals = f[key][patch]
                out[key]["mean"][s] = float(np.mean(vals))
                out[key]["min"][s] = float(np.min(vals))
                out[key]["max"][s] = float(np.max(vals))
    return out


def _contiguous_runs(values):
    """List of (start_idx, end_idx_inclusive) for runs of consecutive equal
    values in a 1-D sequence -- used to shade "which camera frame is
    currently the nearest match" bands on plot_angle_timeseries. NaN-safe
    in the sense that a NaN never merges with its neighbor (NaN != NaN),
    so it just becomes its own 1-sample run rather than crashing or
    silently joining an adjacent run."""
    runs = []
    n = len(values)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n + 1):
        if i == n or values[i] != values[start]:
            runs.append((start, i - 1))
            start = i
    return runs


def footprint_camera_links(lat, lon, geo_valid, nearest_frame_idx, frame_lat_list,
                           frame_lon_list, match_radius_km=50.0, neighbor_radius_km=20.0,
                           t_s=None, peak_t=None, frame_t_s=None):
    """For every geolocated footprint whose nearest pooled camera pixel (in its
    own assigned frame) is within match_radius_km, return a dict carrying the
    footprint position, that single nearest ("center") pixel, and every pooled
    pixel in the same frame within neighbor_radius_km of the center pixel --
    a small visual "patch" around it, since one camera pixel is far smaller
    than one radiometer footprint and a lone point would understate the
    camera's actual local coverage. Grouped per-frame (like
    distances_to_assigned_frame) to keep the haversine matrices small.

    t_s/peak_t/frame_t_s are optional, same-length-as-lat per-sample arrays
    (frame_t_s) or a single sweep-peak time (peak_t) for two extra diagnostic
    fields Sebastian asked for 2026-10-07, used only to color the map, not
    the matching itself:
      - direction: "rising" if this footprint's own time is at or before the
        sweep's peak time, else "falling" -- requires both t_s and peak_t.
      - frame_t_s: the nearest camera exposure time assigned to this
        footprint, carried through verbatim -- requires frame_t_s."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    nearest_frame_idx = np.asarray(nearest_frame_idx)
    links = []
    idxs = np.where(np.asarray(geo_valid, dtype=bool))[0]
    if idxs.size == 0:
        return links
    sub_nf = nearest_frame_idx[idxs]
    for fi in np.unique(sub_nf):
        sel = idxs[sub_nf == fi]
        flat, flon = frame_lat_list[fi], frame_lon_list[fi]
        if len(flat) == 0:
            continue
        d = haversine_matrix_km(lat[sel], lon[sel], flat, flon)
        j = np.argmin(d, axis=1)
        dmin = d[np.arange(len(sel)), j]
        for k in np.where(dmin <= match_radius_km)[0]:
            s = int(sel[k])
            jc = int(j[k])
            center_lat, center_lon = float(flat[jc]), float(flon[jc])
            dn = haversine_matrix_km(np.array([center_lat]), np.array([center_lon]),
                                     flat, flon)[0]
            nbr = dn <= neighbor_radius_km
            link = dict(
                fp_lat=float(lat[s]), fp_lon=float(lon[s]),
                center_lat=center_lat, center_lon=center_lon,
                nbr_lat=flat[nbr], nbr_lon=flon[nbr], dist_km=float(dmin[k]))
            if t_s is not None:
                link["t_s"] = float(t_s[s])
                if peak_t is not None:
                    link["direction"] = "rising" if t_s[s] <= peak_t else "falling"
            if frame_t_s is not None:
                link["frame_t_s"] = float(frame_t_s[s])
            links.append(link)
    return links


def choose_cartopy_projection(lats, lons):
    """Auto-pick a cartopy projection for this data's location, the way
    Sebastian asked (2026-10-07): south/north polar stereographic poleward
    of 60 deg -- where plain lon/lat scaling badly distorts shape and
    distance anyway, and a polar view is what you want to look at -- plain
    PlateCarree otherwise. central_longitude is centered on the data so the
    view isn't needlessly cropped/off-center."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    mean_lat = float(np.nanmean(lats)) if lats.size else 0.0
    mean_lon = float(np.nanmean(lons)) if lons.size else 0.0
    if mean_lat <= -60.0:
        return ccrs.SouthPolarStereo(central_longitude=mean_lon)
    if mean_lat >= 60.0:
        return ccrs.NorthPolarStereo(central_longitude=mean_lon)
    return ccrs.PlateCarree(central_longitude=mean_lon)


_NE_WARNED = False


def _add_coastlines_if_available(ax):
    """ax.coastlines() needs the Natural Earth shapefile cached locally --
    cartopy otherwise tries to download it, which fails outright on a
    network-restricted machine. Skip it (gridlines alone still show scale
    and orientation) rather than crash, and say so once, not once per map."""
    global _NE_WARNED
    if not HAS_NE_DATA:
        if not _NE_WARNED:
            print("  (no cached Natural Earth coastline data found -- maps "
                 "will show gridlines only, no coastlines)")
            _NE_WARNED = True
        return
    try:
        ax.coastlines(resolution="110m", linewidth=0.5, color="0.4", zorder=0)
    except Exception as exc:
        print("  (ax.coastlines() failed, skipping: %s)" % exc)


def _nice_scale_length_km(width_km):
    """Round a raw target length down/up to a 'nice' number for a map scale
    bar (1/2/5 * 10^n, plus a few quarter-steps), picking the step closest
    to -- but not exceeding by much -- 25% of the panel's current width."""
    target = max(width_km, 1e-6) * 0.25
    steps = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000)
    for step in steps:
        if step >= target:
            return step
    return steps[-1]


def add_scale_bar(ax, use_cartopy, loc=(0.06, 0.08)):
    """Draw a simple horizontal distance bar + label near the given
    axes-fraction corner, sized to a 'nice' round number close to 25% of
    the panel's own current width -- requested 2026-10-07 for orientation
    on the cartopy maps (no lon/lat gridline labels are legible at a
    glance on a polar-stereographic panel the way a plain lon/lat plot's
    axis ticks are). Reads the axes' CURRENT limits, so call this after
    all plotting and any set_extent/set_xlim call, not before."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if use_cartopy:
        import cartopy.crs as ccrs
        # BUG FIX (2026-10, Sebastian): this branch previously assumed
        # GeoAxes limits are "already meters" for ANY cartopy projection.
        # That's true for metre-based projections like the polar
        # stereographic ones used at high latitude, but ccrs.PlateCarree's
        # native axes units are DEGREES, not meters. For the equatorial/
        # mid-latitude PlateCarree tiles (ocean, kyushu), dividing degrees
        # by 1000 underestimated the true width by ~1e5x, so
        # _nice_scale_length_km picked its smallest step (1 km) and then
        # bar_len = 1 * 1000.0 = 1000 -- one thousand PROJECTION UNITS,
        # i.e. 1000 degrees, in an axes only ~1 degree wide. Drawing that
        # (via transform=ax.projection) put the bar/label ~1000 degrees
        # off-tile, which is exactly what blew fig.savefig(bbox_inches=
        # "tight") out to >100000 px wide. Fix: use the same degrees-to-km
        # conversion as the non-cartopy branch below for PlateCarree;
        # keep the original meters-based math only for projections that
        # actually use meters (e.g. the polar stereographic ones).
        if isinstance(ax.projection, ccrs.PlateCarree):
            mean_lat = 0.5 * (y0 + y1)
            width_km = abs(x1 - x0) * 111.0 * max(0.15, np.cos(np.radians(mean_lat)))
            bar_km = _nice_scale_length_km(width_km)
            bar_len = bar_km / (111.0 * max(0.15, np.cos(np.radians(mean_lat))))
        else:
            # metre-based cartopy projections (e.g. polar stereographic):
            # axes limits genuinely are in meters.
            width_km = abs(x1 - x0) / 1000.0
            bar_km = _nice_scale_length_km(width_km)
            bar_len = bar_km * 1000.0
        transform = ax.projection
    else:
        # plain lon/lat axes: convert the bar's km length to degrees of
        # longitude at this panel's mean latitude (same cos(lat) scaling
        # used elsewhere in this file for the fallback maps).
        mean_lat = 0.5 * (y0 + y1)
        width_km = abs(x1 - x0) * 111.0 * max(0.15, np.cos(np.radians(mean_lat)))
        bar_km = _nice_scale_length_km(width_km)
        bar_len = bar_km / (111.0 * max(0.15, np.cos(np.radians(mean_lat))))
        transform = ax.transData
    bx0 = min(x0, x1) + loc[0] * abs(x1 - x0)
    by0 = min(y0, y1) + loc[1] * abs(y1 - y0)
    ax.plot([bx0, bx0 + bar_len], [by0, by0], "-", color="0.15", lw=2.5,
            solid_capstyle="butt", transform=transform, zorder=6)
    ax.text(bx0 + bar_len / 2.0, by0 + 0.02 * abs(y1 - y0), "%d km" % bar_km,
           ha="center", va="bottom", fontsize=8, color="0.15",
           transform=transform, zorder=6)
    # drawing the bar must not itself expand the axes' own autoscale limits
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def plot_footprint_camera_map(links, out_png, title, center=None, half_width_km=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    all_lat = [lk["fp_lat"] for lk in links] + [lk["center_lat"] for lk in links]
    all_lon = [lk["fp_lon"] for lk in links] + [lk["center_lon"] for lk in links]

    use_cartopy = HAS_CARTOPY and len(all_lat) > 0
    if use_cartopy:
        proj = choose_cartopy_projection(all_lat, all_lon)
        data_crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection=proj))
        for lk in links:
            ax.plot(lk["nbr_lon"], lk["nbr_lat"], ".", ms=3, color="#2a78d6",
                    alpha=0.55, zorder=2, transform=data_crs)
            ax.plot([lk["fp_lon"], lk["center_lon"]], [lk["fp_lat"], lk["center_lat"]],
                    "-", lw=0.6, color="0.6", zorder=1, transform=data_crs)
            ax.plot(lk["fp_lon"], lk["fp_lat"], "*", ms=9, color="#d6336c", zorder=3,
                    transform=data_crs)
        _add_coastlines_if_available(ax)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.7)
        gl.top_labels = gl.right_labels = False

        if center is not None and half_width_km is not None:
            clat, clon = center
            cx, cy = proj.transform_point(clon, clat, data_crs)
            w = half_width_km * 1000.0
            ax.set_extent([cx - w, cx + w, cy - w, cy + w], crs=proj)
            ax.plot(clon, clat, "+", ms=14, mew=2, color="0.3", zorder=4,
                    transform=data_crs)
    else:
        # fallback for a machine without cartopy installed: plain lon/lat
        # scatter with a cosine(latitude) aspect correction instead of a
        # real map projection.
        fig, ax = plt.subplots(figsize=(9, 9))
        for lk in links:
            ax.plot(lk["nbr_lon"], lk["nbr_lat"], ".", ms=3, color="#2a78d6",
                    alpha=0.55, zorder=2)
            ax.plot([lk["fp_lon"], lk["center_lon"]], [lk["fp_lat"], lk["center_lat"]],
                    "-", lw=0.6, color="0.6", zorder=1)
            ax.plot(lk["fp_lon"], lk["fp_lat"], "*", ms=9, color="#d6336c", zorder=3)
        mean_lat = float(np.mean(all_lat)) if all_lat else 0.0
        ax.set_aspect(1.0 / max(0.15, np.cos(np.radians(mean_lat))))
        if center is not None and half_width_km is not None:
            clat, clon = center
            dlat = half_width_km / 111.0
            dlon = half_width_km / (111.0 * max(0.15, np.cos(np.radians(clat))))
            ax.set_xlim(clon - dlon, clon + dlon)
            ax.set_ylim(clat - dlat, clat + dlat)
            ax.plot(clon, clat, "+", ms=14, mew=2, color="0.3", zorder=4)
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

    add_scale_bar(ax, use_cartopy)
    ax.set_title(title, fontsize=11)
    handles = [
        Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#d6336c",
              markeredgecolor="#d6336c", markersize=10, label="radiometer footprint"),
        Line2D([0], [0], marker=".", linestyle="none", markerfacecolor="#2a78d6",
              markeredgecolor="#2a78d6", markersize=6,
              label="camera pixels (<=20 km of nearest match)"),
    ]
    if center is not None:
        handles.append(Line2D([0], [0], marker="+", linestyle="none", color="0.3",
                              markersize=10, label="nadir reference"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("  wrote %s" % out_png)


def pick_offset_centers(lat, lon, geo_valid, t_s, nadir_t, nadir_lat, nadir_lon,
                        offset_km):
    """Find two reference points on either side of the sweep's own nadir
    crossing (nadir_t/nadir_lat/nadir_lon), each the geolocated sample
    closest to offset_km great-circle distance away -- one from samples
    BEFORE nadir_t, one from samples AFTER it. Mirrors a cross-track sweep's
    geometry: before/after in time is before/after in cross-track distance
    on either side of nadir, which is what "500 km off nadir, both
    directions" means physically.

    Returns ((lat, lon, achieved_km), (lat, lon, achieved_km)), with either
    side None if that side has no geolocated samples at all (e.g. nadir
    sits right at the edge of the padded window) -- achieved_km lets the
    caller note it when the window is too short to actually reach
    offset_km, rather than silently mislabeling a closer point as "500 km
    away"."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    geo_valid = np.asarray(geo_valid, dtype=bool)
    t_s = np.asarray(t_s, dtype=float)
    idxs = np.where(geo_valid)[0]
    if idxs.size == 0:
        return None, None

    def best(mask):
        sub = idxs[mask]
        if sub.size == 0:
            return None
        dd = haversine_matrix_km(np.array([nadir_lat]), np.array([nadir_lon]),
                                 lat[sub], lon[sub])[0]
        k = int(np.argmin(np.abs(dd - offset_km)))
        return float(lat[sub[k]]), float(lon[sub[k]]), float(dd[k])

    before = best(t_s[idxs] < nadir_t)
    after = best(t_s[idxs] > nadir_t)
    return before, after


def plot_footprint_camera_diagnostic(links, out_png, centers, panel_titles,
                                     half_width_km=50.0, per_frame=None):
    """Three side-by-side zoom maps (2*half_width_km wide each) at the given
    centers -- meant for (before-nadir, nadir, after-nadir), but any three
    will do -- with two extra diagnostics on top of plot_footprint_camera_map's
    view, both requested 2026-10-07: footprints colored by sweep direction
    (rising vs. falling elevation, from footprint_camera_links' "direction"
    field) and camera pixels colored by which single camera frame they came
    from (its "frame_t_s" field) -- at this zoom level there are normally
    only one or two distinct frames in view, so a small discrete legend
    covers them. One projection is chosen from ALL the links' (and, when
    given, per_frame's) positions so every panel is directly comparable.

    per_frame (added 2026-10-07, optional, backward compatible): pool_
    stripe_pixels' own per_frame list, with a "frame_t_s" key set on each
    entry by the caller (run() does this -- same convention as links'
    "frame_t_s", so the two rows' frame colors match exactly). When given,
    adds a SECOND row per panel: every pooled stripe-component pixel of
    that frame in view, not just the ones within neighbor_radius_km of some
    matched footprint (links' own small "patch" around its nearest match).
    Lets you directly compare the patch-only view (top row, what plot_
    footprint_camera_map also shows) against the camera's actual full
    pooled-pixel coverage (bottom row) at the same zoom -- the top row's
    striped/scalloped look is exactly the per-footprint 20 km-patch
    selection, not a rendering artifact; the bottom row is not itself
    "every camera pixel" either, since pool_stripe_pixels already restricts
    to specific stripe components at --stride (see its own docstring)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # direction is a two-way polarity (rising vs. falling elevation), so it
    # gets the diverging blue<->red pair; frame identity is a separate,
    # unordered categorical dimension and gets its own slots from the same
    # validated palette, deliberately skipping blue/red so the two
    # dimensions never share a hue (marker shape -- star vs. dot -- already
    # tells them apart, but a shared color would still read as "the same
    # thing" at a glance).
    DIRECTION_COLORS = {"rising": "#2a78d6", "falling": "#e34948"}
    DIRECTION_LABELS = {"rising": "footprint, rising elevation",
                        "falling": "footprint, falling elevation"}
    FRAME_PALETTE = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    frame_vals = {lk["frame_t_s"] for lk in links if "frame_t_s" in lk}
    if per_frame:
        frame_vals |= {f["frame_t_s"] for f in per_frame if "frame_t_s" in f}
    frame_vals = sorted(frame_vals)
    frame_colors = {v: FRAME_PALETTE[i % len(FRAME_PALETTE)]
                    for i, v in enumerate(frame_vals)}
    if len(frame_vals) > len(FRAME_PALETTE):
        print("  (%d distinct camera frames in view, more than this palette's "
             "%d slots -- some frames will share a color)"
             % (len(frame_vals), len(FRAME_PALETTE)))

    all_lat = [lk["fp_lat"] for lk in links] + [lk["center_lat"] for lk in links]
    all_lon = [lk["fp_lon"] for lk in links] + [lk["center_lon"] for lk in links]
    if per_frame:
        all_lat += [v for f in per_frame for v in f["lat"]]
        all_lon += [v for f in per_frame for v in f["lon"]]
    use_cartopy = HAS_CARTOPY and len(all_lat) > 0
    nrows = 2 if per_frame else 1
    if use_cartopy:
        proj = choose_cartopy_projection(all_lat, all_lon)
        data_crs = ccrs.PlateCarree()
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6.5 * nrows),
                                 subplot_kw=dict(projection=proj))
    else:
        proj = data_crs = None
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6.5 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    mean_lat_all = float(np.mean(all_lat)) if all_lat else 0.0

    def _finish_panel(ax, center):
        if use_cartopy:
            _add_coastlines_if_available(ax)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.7)
            gl.top_labels = gl.right_labels = False
            if center is not None:
                clat, clon = center
                cx, cy = proj.transform_point(clon, clat, data_crs)
                w = half_width_km * 1000.0
                ax.set_extent([cx - w, cx + w, cy - w, cy + w], crs=proj)
                ax.plot(clon, clat, "+", ms=14, mew=2, color="0.3", zorder=4,
                        transform=data_crs)
                add_scale_bar(ax, use_cartopy)
            else:
                ax.text(0.5, 0.5, "no geolocated sample\non this side of nadir",
                       ha="center", va="center", transform=ax.transAxes, fontsize=9)
        else:
            ax.set_aspect(1.0 / max(0.15, np.cos(np.radians(mean_lat_all))))
            if center is not None:
                clat, clon = center
                dlat = half_width_km / 111.0
                dlon = half_width_km / (111.0 * max(0.15, np.cos(np.radians(clat))))
                ax.set_xlim(clon - dlon, clon + dlon)
                ax.set_ylim(clat - dlat, clat + dlat)
                ax.plot(clon, clat, "+", ms=14, mew=2, color="0.3", zorder=4)
                add_scale_bar(ax, use_cartopy)
            else:
                ax.text(0.5, 0.5, "no geolocated sample\non this side of nadir",
                       ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")

    for c, (center, title) in enumerate(zip(centers, panel_titles)):
        ax = axes[0, c]
        kw = dict(transform=data_crs) if use_cartopy else {}
        for lk in links:
            fc = frame_colors.get(lk.get("frame_t_s"), "#2a78d6")
            dc = DIRECTION_COLORS.get(lk.get("direction"), "#d6336c")
            ax.plot(lk["nbr_lon"], lk["nbr_lat"], ".", ms=4, color=fc,
                    alpha=0.6, zorder=2, **kw)
            ax.plot([lk["fp_lon"], lk["center_lon"]], [lk["fp_lat"], lk["center_lat"]],
                    "-", lw=0.5, color="0.7", zorder=1, **kw)
            ax.plot(lk["fp_lon"], lk["fp_lat"], "*", ms=9, color=dc, zorder=3, **kw)
        _finish_panel(ax, center)
        ax.set_title(title, fontsize=10.5)
        if c == 0:
            ax.annotate("patch only (<=20 km of\nnearest footprint match)",
                       xy=(-0.14, 0.5), xycoords="axes fraction", rotation=90,
                       ha="center", va="center", fontsize=9)

        if per_frame:
            ax2 = axes[1, c]
            for f in per_frame:
                fc = frame_colors.get(f.get("frame_t_s"), "#2a78d6")
                ax2.plot(f["lon"], f["lat"], ".", ms=4, color=fc, alpha=0.5,
                        zorder=2, **kw)
            for lk in links:
                dc = DIRECTION_COLORS.get(lk.get("direction"), "#d6336c")
                ax2.plot(lk["fp_lon"], lk["fp_lat"], "*", ms=9, color=dc, zorder=3, **kw)
            _finish_panel(ax2, center)
            if c == 0:
                ax2.annotate("full pooled-pixel\ncoverage, this frame",
                           xy=(-0.14, 0.5), xycoords="axes fraction", rotation=90,
                           ha="center", va="center", fontsize=9)

    handles = [Line2D([0], [0], marker="*", linestyle="none", color=DIRECTION_COLORS[d],
                      markersize=10, label=DIRECTION_LABELS[d])
              for d in ("rising", "falling")]
    for v in frame_vals:
        handles.append(Line2D([0], [0], marker=".", linestyle="none",
                              color=frame_colors[v], markersize=8,
                              label="camera pixels, frame t=%.1f s" % v))
    handles.append(Line2D([0], [0], marker="+", linestyle="none", color="0.3",
                          markersize=10, label="zoom center"))
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
              fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Footprint direction and camera-frame diagnostics, "
                "%d km zoom%s" % (int(2 * half_width_km),
                                  " -- patch vs. full pooled-pixel coverage"
                                  if per_frame else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("  wrote %s" % out_png)


ANGLE_SPECS = (
    ("sza", "Solar Zenith Angle", "Blues"),
    ("vza", "Viewing Zenith Angle", "Oranges"),
    ("raz", "Relative Azimuth Angle", "viridis"),
)


def plot_angle_diagnostic(per_frame, fp_lat, fp_lon, out_png, centers,
                          panel_titles, half_width_km=50.0,
                          angle_specs=ANGLE_SPECS):
    """3 rows (Solar Zenith Angle, Viewing Zenith Angle, Relative Azimuth
    Angle) x 3 columns (same before/nadir/after zoom centers as
    plot_footprint_camera_diagnostic) -- requested 2026-10-07 alongside its
    full-pooled-coverage row. Each camera pixel in view (every pooled
    stripe-component pixel, same set as that diagnostic's bottom row, NOT
    restricted to a footprint's 20 km patch) is colored by its own per-
    pixel L1B angle value, one shared colorbar per row (vmin/vmax from the
    actual observed data, printed to the console, since this plot has no
    prior art in the repo to assume a 0-90/0-180 convention from).

    Radiometer footprints are plotted as plain gray position markers in
    EVERY panel here, never colored by their own angle fields -- Sebastian
    asked to hold off on using the radiometer's own Solar_Zenith_Surface /
    Viewing_Zenith_Surface (identically -999 in this file) until he shares
    his idea for that gap; the camera's own per-pixel copies of the same-
    named fields are a different L1B variable (see ANGLE_FIELDS/pool_
    stripe_pixels) and are not affected by that gap, which is why this plot
    is safe to build from them alone.

    A row is skipped (with a console note, and no row drawn) if the
    granule's CAM L1B file has no such variable at all; the whole plot is
    skipped (no file written) if none of the three are available."""
    import matplotlib.pyplot as plt

    have_rows = [(k, label, cmap) for k, label, cmap in angle_specs
                if per_frame and k in per_frame[0]]
    if not have_rows:
        print("  (no camera angle fields available in this granule -- "
             "skipping the angle diagnostic plot entirely)")
        return
    nrows = len(have_rows)

    fp_lat = np.asarray(fp_lat, dtype=float)
    fp_lon = np.asarray(fp_lon, dtype=float)
    all_lat = list(fp_lat) + [v for f in per_frame for v in f["lat"]]
    all_lon = list(fp_lon) + [v for f in per_frame for v in f["lon"]]
    use_cartopy = HAS_CARTOPY and len(all_lat) > 0
    if use_cartopy:
        proj = choose_cartopy_projection(all_lat, all_lon)
        data_crs = ccrs.PlateCarree()
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6.2 * nrows),
                                 subplot_kw=dict(projection=proj))
    else:
        proj = data_crs = None
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6.2 * nrows))
    if nrows == 1:
        axes = axes[None, :]

    mean_lat_all = float(np.mean(all_lat)) if all_lat else 0.0
    kw = dict(transform=data_crs) if use_cartopy else {}

    for r, (key, label, cmap) in enumerate(have_rows):
        vals_all = np.concatenate([f[key] for f in per_frame])
        vmin, vmax = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
        print("  %s: observed range %.2f .. %.2f deg across %d pooled "
             "camera pixels%s" % (label, vmin, vmax, vals_all.size,
                                  " (WARNING: all-identical -- looks like an "
                                  "unpopulated L1B placeholder, not real "
                                  "geometry)" if vmax <= vmin else ""))
        if vmax <= vmin:
            vmax = vmin + 1.0
        sm = None
        for c, (center, title) in enumerate(zip(centers, panel_titles)):
            ax = axes[r, c]
            for f in per_frame:
                sm = ax.scatter(f["lon"], f["lat"], c=f[key], cmap=cmap,
                                vmin=vmin, vmax=vmax, s=6, linewidths=0,
                                zorder=2, **kw)
            if fp_lat.size:
                ax.plot(fp_lon, fp_lat, "x", ms=5, mew=1.0, color="0.25",
                       zorder=3, **kw)

            if use_cartopy:
                _add_coastlines_if_available(ax)
                gl = ax.gridlines(draw_labels=(c == 0), linewidth=0.3,
                                  color="0.7", alpha=0.7)
                gl.top_labels = gl.right_labels = False
            else:
                ax.set_aspect(1.0 / max(0.15, np.cos(np.radians(mean_lat_all))))

            if center is not None:
                clat, clon = center
                if use_cartopy:
                    cx, cy = proj.transform_point(clon, clat, data_crs)
                    w = half_width_km * 1000.0
                    ax.set_extent([cx - w, cx + w, cy - w, cy + w], crs=proj)
                    ax.plot(clon, clat, "+", ms=13, mew=2, color="0.15",
                           zorder=4, transform=data_crs)
                else:
                    dlat = half_width_km / 111.0
                    dlon = half_width_km / (111.0 * max(0.15, np.cos(np.radians(clat))))
                    ax.set_xlim(clon - dlon, clon + dlon)
                    ax.set_ylim(clat - dlat, clat + dlat)
                    ax.plot(clon, clat, "+", ms=13, mew=2, color="0.15", zorder=4)
                add_scale_bar(ax, use_cartopy)
            else:
                ax.text(0.5, 0.5, "no geolocated sample\non this side of nadir",
                       ha="center", va="center", transform=ax.transAxes, fontsize=9)
            if r == 0:
                ax.set_title(title, fontsize=10.5)
            if c == 0:
                ax.annotate(label, xy=(-0.14, 0.5), xycoords="axes fraction",
                           rotation=90, ha="center", va="center", fontsize=10.5)
        if sm is not None:
            cbar = fig.colorbar(sm, ax=list(axes[r, :]), fraction=0.015, pad=0.01)
            cbar.set_label("%s [deg]" % label, fontsize=8.5)

    fig.suptitle("Camera per-pixel viewing/solar geometry -- %d km zoom\n"
                "(x = radiometer footprint position only, not angle-colored "
                "-- see docstring)" % int(2 * half_width_km), fontsize=12)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("  wrote %s" % out_png)


ANGLE_TS_SPECS = (
    # same hues as palette.md's categorical slots 1/2/3 (blue/orange/aqua) --
    # these are 3 independent panels, not series sharing one axis, but
    # reusing the validated slot order keeps this consistent with the rest
    # of the file's color story. Units are now part of the label itself
    # (see plot_angle_vs_elevation/plot_angle_timeseries) since a later,
    # non-degree panel (pct_identified) was added at the call site and a
    # hardcoded "[deg]" suffix would mislabel it.
    ("sza", "Solar Zenith Angle\n[deg]", "#2a78d6"),
    ("vza", "Viewing Zenith Angle\n[deg]", "#eb6834"),
    ("raz", "Relative Azimuth Angle\n[deg]", "#1baf7a"),
)


def plot_angle_timeseries(t_s, nearest_frame_t_s, stats, out_png, camera_t0,
                          camera_t1, patch_km, match_radius_km,
                          angle_specs=ANGLE_TS_SPECS):
    """One stacked panel per available angle (Solar Zenith, Viewing Zenith,
    Relative Azimuth) vs. radiometer time t_s -- requested 2026-10-07,
    towards sizing RT domains, to visualize BOTH of Sebastian's two stated
    sources of angle variability at once in a single picture:

      - SPATIAL: the shaded min..max band is the spread across the closest
        patch_km-diameter patch of pooled camera pixels matched to each
        footprint (within match_radius_km) -- footprint_patch_angle_stats'
        own "mean"/"min"/"max" arrays, plotted directly.
      - TEMPORAL/stepwise: the camera only updates every ~5 s, so the patch
        -- and therefore its mean angle -- jumps each time a footprint's
        nearest_frame_idx switches to a new camera frame, rather than
        changing continuously with the radiometer's own much faster
        sampling. The background color bands mark which camera frame is
        currently the nearest match (same FRAME_PALETTE/frame_t_s
        convention as plot_footprint_camera_diagnostic), so that stepwise
        jump is visible directly, not just inferable from the x-axis.

    A panel is skipped (console note, no crash) if that angle key isn't in
    `stats` at all (the granule has no such L1B variable -- see pool_
    stripe_pixels' ANGLE_FIELDS). NaN gaps in a panel (footprint unmatched
    or not geolocated) show as a visible break in the line/band, same
    convention as distances_to_assigned_frame elsewhere in this file."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    have = [(k, label, color) for k, label, color in angle_specs if k in stats]
    if not have:
        print("  (no camera angle fields available -- skipping the angle "
             "time series plot entirely)")
        return

    t_s = np.asarray(t_s, dtype=float)
    nearest_frame_t_s = np.asarray(nearest_frame_t_s, dtype=float)
    FRAME_PALETTE = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    frame_vals = sorted(set(nearest_frame_t_s.tolist()))
    frame_colors = {v: FRAME_PALETTE[i % len(FRAME_PALETTE)]
                   for i, v in enumerate(frame_vals)}
    if len(frame_vals) > len(FRAME_PALETTE):
        print("  (%d distinct camera frames in this window, more than this "
             "palette's %d slots -- some frames will share a color)"
             % (len(frame_vals), len(FRAME_PALETTE)))

    order = np.argsort(t_s)
    t_sorted = t_s[order]
    nf_sorted = nearest_frame_t_s[order]
    runs = _contiguous_runs(list(nf_sorted))

    fig, axes = plt.subplots(len(have), 1, figsize=(11, 3.0 * len(have)), sharex=True)
    axes = [axes] if len(have) == 1 else list(axes)

    for ax in axes:
        ax.axvspan(camera_t0, camera_t1, color="#2a78d6", alpha=0.05, zorder=0)
        for s, e in runs:
            fv = nf_sorted[s]
            ax.axvspan(t_sorted[s], t_sorted[e], color=frame_colors.get(fv, "0.7"),
                      alpha=0.12, zorder=0, lw=0)

    for ax, (key, label, color) in zip(axes, have):
        mean_v, min_v, max_v = stats[key]["mean"], stats[key]["min"], stats[key]["max"]
        ax.fill_between(t_s, min_v, max_v, color=color, alpha=0.25, lw=0,
                       label="patch min..max")
        ax.plot(t_s, mean_v, ".-", ms=3, lw=0.8, color=color, label="patch mean")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    axes[-1].set_xlabel("Radiometer time [s] (t=0 at the first usable camera frame)")
    if t_s.size:
        pad_view = 0.03 * (t_s.max() - t_s.min())
        axes[-1].set_xlim(t_s.min() - pad_view, t_s.max() + pad_view)

    handles = [Line2D([0], [0], color=frame_colors[v], lw=6, alpha=0.35,
                      label="frame t=%.1f s is nearest match" % v)
              for v in frame_vals]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)),
              fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Camera per-pixel angle stats vs. radiometer time -- %d "
                "km-diameter patch around each footprint's nearest camera-"
                "pixel match (<=%d km)" % (int(patch_km), int(match_radius_km)),
                fontsize=11.5)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("  wrote %s" % out_png)


def plot_angle_vs_elevation(sweep_data, out_png, patch_km, match_radius_km,
                            angle_specs=ANGLE_TS_SPECS):
    """One stacked panel per available angle (Solar Zenith, Viewing Zenith,
    Relative Azimuth) vs. the RADIOMETER's own scan Elevation angle, instead
    of time -- requested 2026-10-07 as the next step after plot_angle_
    timeseries (which is kept in this file, just no longer called by
    default), once Sebastian pointed out that what actually matters for
    sizing a 2D (cross-track x along-track) RT domain is seeing BOTH:

      - swath-position (cross-track) variability: how much the patch angle
        stats change as Elevation sweeps from one side to the other WITHIN
        one sweep -- the same mean line + min..max shaded band plot_angle_
        timeseries drew vs. time, just re-parameterized by Elevation. Since
        Elevation is NOT monotonic across a whole sweep (it runs e.g.
        -62 -> +62 -> -62 deg), each sample is first split by `direction`
        (rising/falling half of the back-and-forth scan, same concept as
        footprint_camera_links' own direction field) and each half is
        sorted by Elevation before fill_between, so both halves draw as
        clean monotonic-x curves instead of a self-crossing mess.
      - along-track (satellite progression) variability: how much the SAME
        Elevation value's patch angle stats differ between one sweep and
        the next -- shown by overlaying len(sweep_data) consecutive sweeps
        (one color each, reusing FRAME_PALETTE) on the same axes. A visible
        gap between two sweeps' curves at the same Elevation is purely
        along-track drift, since cross-track position is held fixed by
        construction (same Elevation -> same scan angle -> same point
        across the swath, modulo the satellite having moved).

    Each sweep_data entry is a dict with:
      label      -- legend label, e.g. "sweep 12 (t=123.4-130.1 s)"
      elev       -- 1-D array, that sweep's padded-window Elevation samples
      direction  -- same-length array of "rising"/"falling", purely so each
                    half can be sorted into a monotonic-x curve
      stats      -- footprint_patch_angle_stats' own {key: {mean,min,max}}
                    return dict for that sweep's window

    Linestyle encodes direction (solid=rising, dashed=falling); color
    encodes which sweep. A panel is skipped (console note) if no sweep in
    sweep_data carries that angle key at all."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    have = [(k, label, c) for k, label, c in angle_specs
           if any(k in sd["stats"] for sd in sweep_data)]
    if not have:
        print("  (no camera angle fields available -- skipping the angle "
             "vs. elevation plot entirely)")
        return

    FRAME_PALETTE = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
    sweep_colors = {i: FRAME_PALETTE[i % len(FRAME_PALETTE)] for i in range(len(sweep_data))}
    if len(sweep_data) > len(FRAME_PALETTE):
        print("  (%d sweeps overlaid, more than this palette's %d slots -- "
             "some sweeps will share a color)" % (len(sweep_data), len(FRAME_PALETTE)))

    fig, axes = plt.subplots(len(have), 1, figsize=(11, 3.2 * len(have)), sharex=True)
    axes = [axes] if len(have) == 1 else list(axes)

    for ax, (key, label, _color) in zip(axes, have):
        for si, sd in enumerate(sweep_data):
            if key not in sd["stats"]:
                continue
            elev = np.asarray(sd["elev"], dtype=float)
            direction = np.asarray(sd["direction"])
            mean_v = np.asarray(sd["stats"][key]["mean"])
            min_v = np.asarray(sd["stats"][key]["min"])
            max_v = np.asarray(sd["stats"][key]["max"])
            color = sweep_colors[si]
            for dirn, ls in (("rising", "-"), ("falling", "--")):
                m = direction == dirn
                if not m.any():
                    continue
                order = np.argsort(elev[m])
                ex = elev[m][order]
                ax.fill_between(ex, min_v[m][order], max_v[m][order],
                               color=color, alpha=0.18, lw=0)
                ax.plot(ex, mean_v[m][order], ls, ms=3, lw=1.1, color=color)
        ax.set_ylabel(label, fontsize=9)

    axes[-1].set_xlabel("Radiometer scan Elevation [deg]")

    sweep_handles = [Line2D([0], [0], color=sweep_colors[i], lw=4,
                            label=sd["label"]) for i, sd in enumerate(sweep_data)]
    dir_handles = [Line2D([0], [0], color="0.3", lw=1.3, ls="-", label="rising"),
                  Line2D([0], [0], color="0.3", lw=1.3, ls="--", label="falling")]
    fig.legend(handles=sweep_handles + dir_handles, loc="lower center",
              ncol=min(4, len(sweep_handles) + 2), fontsize=8,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Camera per-pixel angle stats vs. radiometer scan Elevation "
                "-- %d km-diameter patch around each footprint's nearest "
                "camera-pixel match (<=%d km), %d consecutive sweep(s) "
                "overlaid" % (int(patch_km), int(match_radius_km), len(sweep_data)),
                fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("  wrote %s" % out_png)


# ─── output ─────────────────────────────────────────────────────────────────

def write_sweep_csv(t_s, elev, nearest_frame_t_s, dt_s, dist_km, out_csv):
    with open(out_csv, "w") as fh:
        fh.write("# Elevation-sweep timing diagnostic.\n")
        fh.write("# rad_time_s: radiometer time, seconds since the first usable "
                 "camera frame (t=0). nearest_frame_time_s / dt_s: the camera "
                 "frame argmin-assigned to this radiometer sample and the gap "
                 "(same convention as nadir_radiometer_overlap.py's "
                 "test_overlap). dist_km: nearest pooled camera pixel distance "
                 "in that frame (blank where geolocation was invalid).\n")
        fh.write("rad_time_s,elevation_deg,nearest_frame_time_s,dt_s,dist_km\n")
        for i in range(len(t_s)):
            dk = "" if np.isnan(dist_km[i]) else "%.4f" % dist_km[i]
            fh.write("%.6f,%.4f,%.6f,%.4f,%s\n"
                    % (t_s[i], elev[i], nearest_frame_t_s[i], dt_s[i], dk))
    return out_csv


def plot_sweep(t_s, elev, nearest_frame_t_s, dist_km, camera_t0, camera_t1,
              n_frames, sw, out_png, rad_mode=128):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for ax in axes:
        ax.axvspan(camera_t0, camera_t1, color="#2a78d6", alpha=0.08, zorder=0)

    axes[0].plot(t_s, elev, ".-", ms=2, lw=0.6, color="#2a78d6")
    axes[0].axhline(0, color="0.6", lw=0.5)
    # the Earth-view envelope this run's own geo_valid samples occupied --
    # context for why the sweep is bounded here and not at the raw
    # telemetry's much wider (space-view) extremes.
    env_lo = min(sw["start_val"], sw["end_val"])
    env_hi = sw["peak_val"]
    axes[0].axhspan(env_lo, env_hi, color="#2ca02c", alpha=0.06, zorder=0)
    axes[0].set_ylabel("Elevation [deg]")
    title = ("Radiometer elevation -- one full Earth-view sweep (t=%.1f to %.1f s, "
            "%+.1f -> %+.1f -> %+.1f deg)\nsweep = the geolocated (geo_valid) run, "
            "not raw-telemetry extrema -- Operational_Mode %d continues past the "
            "horizon into space-view"
            % (sw["start_t"], sw["end_t"], sw["start_val"], sw["peak_val"],
               sw["end_val"], rad_mode))
    axes[0].set_title(title, fontsize=10.5)

    axes[1].plot(t_s, nearest_frame_t_s, ".", ms=2.5, color="#eb6834")
    axes[1].plot(t_s, t_s, "--", lw=0.6, color="0.5", label="1:1 (radiometer time)")
    axes[1].set_ylabel("Nearest camera\nexposure time [s]")
    axes[1].legend(fontsize=8, loc="upper left")

    axes[2].plot(t_s, dist_km, ".", ms=2.5, color="#d6336c")
    axes[2].set_ylabel("Nearest camera\npixel distance [km]")
    axes[2].set_xlabel("Radiometer time [s] (t=0 at the first usable camera frame)")
    axes[2].set_ylim(0, 50)

    axes[0].text(0.01, 0.04,
                "shaded band: camera-active window (%d usable frames)" % n_frames,
                transform=axes[0].transAxes, fontsize=7.5, color="#2a78d6", va="bottom")

    # axvspan(camera_t0, camera_t1, ...) above participates in autoscale, and the
    # camera-active window (tens to hundreds of seconds) is normally far wider than
    # the plotted sweep (a few seconds) -- without an explicit xlim the real data
    # gets squeezed into a sliver at the left edge. Zoom to the actual data instead;
    # sharex=True propagates this to all three panels.
    pad_view = 0.03 * (t_s.max() - t_s.min())
    axes[-1].set_xlim(t_s.min() - pad_view, t_s.max() + pad_view)

    fig.suptitle("Elevation-sweep timing vs. camera availability", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    print("  wrote %s" % out_png)


def run(granule, rad_path, out_dir="output_l2", rad_mode=128,
        components=CT_STRIPE_COMPONENTS, max_frames=None, first_frame=0,
        stride=2, table_dir=None, sweep_index=None, pad_frac=0.1,
        max_gap_s=None, match_radius_km=50.0, neighbor_radius_km=20.0,
        zoom_half_width_km=250.0, diag_half_width_km=50.0,
        diag_offset_km=500.0, patch_radius_km=12.5, n_sweeps=2):
    os.makedirs(out_dir, exist_ok=True)
    from libcam.env_check import report as _env_report
    _interp, _pkgs = _env_report()
    _missing = [n for n, (ok, _d) in _pkgs.items() if not ok]
    if _missing:
        print("env check: MISSING %s in %s -- `python3 -m libcam.env_check` "
              "for details. A step below that degrades gracefully on one of "
              "these (e.g. ADM blob detection on scipy) will silently skip "
              "real work rather than fail, so its output can look like a "
              "data finding when it is actually this."
              % (", ".join(_missing), _interp["executable"]))
    stem = os.path.splitext(os.path.basename(granule))[0]

    # Loud and up front -- a silent fallback to plain lon/lat scatter maps
    # (no error, no crash, just the old-style plot) is exactly what Sebastian
    # hit 2026-10-07 when cartopy wasn't actually importable in his active
    # environment despite being listed in setup.py's [full] extra. Printing
    # this plainly means that cause shows up in the log instead of having to
    # be diagnosed after the fact from "the map doesn't look like a map."
    if HAS_CARTOPY:
        print("Cartopy available -- maps will use a real projection "
             "(coastlines %s)." % ("available" if HAS_NE_DATA else
                                   "NOT cached locally, skipping"))
    else:
        print("Cartopy NOT importable in this environment -- maps will fall "
             "back to a plain lon/lat scatter (cosine-latitude corrected), "
             "not a real map projection. `pip install cartopy` (it's in "
             "this repo's setup.py [full] extra) to fix this.")

    # Cheap pass first (added 2026-10-07): get the camera time window WITHOUT
    # running MaskRetriever on every frame in the granule, so a sweep can be
    # picked (pick_sweep needs the camera's active-window overlap) before
    # paying for the expensive pool_stripe_pixels call -- which is then
    # narrowed to just the frames that can matter for the picked sweep(s),
    # instead of the whole granule, most of which is never displayed. See
    # scan_camera_frame_times/narrow_frame_range docstrings for the "why".
    print("Scanning camera frame timestamps/geolocation (cheap pass -- no "
         "mask retrieval yet) ...")
    scan_idx, scan_time_ns = scan_camera_frame_times(
        granule, first_frame=first_frame, max_frames=max_frames)
    if scan_idx.size == 0:
        raise SystemExit("no geolocated camera frames in the requested range")
    t_ref = int(scan_time_ns.min())
    camera_t0 = 0.0
    camera_t1 = float((scan_time_ns.max() - t_ref) / 1e9)
    print("  %d candidate camera frames (geolocated), spanning %.1f s "
         "(t=0 at the first one)" % (scan_idx.size, camera_t1))

    print("Loading radiometer elevation timeline (Operational_Mode=%d) ..." % rad_mode)
    rad = load_radiometer_elevation(rad_path, mode=rad_mode)
    order = np.argsort(rad["time"])
    for k in ("time", "elevation", "lat", "lon", "geo_valid"):
        rad[k] = rad[k][order]
    print("  %d samples with valid Elevation in this mode (%d also geolocated)"
         % (len(rad["time"]), int(rad["geo_valid"].sum())))
    time_s_full = (rad["time"].astype(np.float64) - t_ref) / 1e9

    print("Finding Earth-view sweeps (geo_valid runs, not raw-telemetry extrema) ...")
    sweeps = find_earth_view_sweeps(rad["elevation"], time_s_full, rad["geo_valid"],
                                    max_gap_s=max_gap_s)
    print("  %d Earth-view sweeps found" % len(sweeps))
    for i, sw in enumerate(sweeps):
        overlaps = sw["start_t"] <= camera_t1 and sw["end_t"] >= camera_t0
        marker = "  <-- overlaps camera window" if overlaps else ""
        print("    [%2d] t=%9.1f .. %9.1f s (dur %6.1f s)  elevation %+6.1f -> "
             "%+6.1f -> %+6.1f deg%s"
             % (i, sw["start_t"], sw["end_t"], sw["end_t"] - sw["start_t"],
                sw["start_val"], sw["peak_val"], sw["end_val"], marker))

    if sweep_index is None:
        chosen = pick_sweep(sweeps, camera_t0, camera_t1)
        if chosen is None:
            raise SystemExit(
                "no Earth-view sweep found -- no geo_valid samples at all in "
                "this mode; check --rad-mode matches the mode the scan "
                "actually runs in (this granule: 128)")
        print("  auto-picked sweep [%d] (best overlap with the camera window; "
             "override with --sweep-index N)" % chosen)
    else:
        chosen = sweep_index
    sw = sweeps[chosen]

    # Consecutive sweeps carried through to the elevation-based multi-sweep
    # angle plot (--n-sweeps, default 2, added 2026-10-07 to see along-track
    # as well as cross-track angle variability in one picture). Every OTHER
    # output below (CSV, elevation_sweep.png, footprint maps, diagnostics)
    # is still based on `sw` (the single primary/chosen sweep) only,
    # unchanged from before.
    n_sweeps = max(1, int(n_sweeps))
    chosen_idxs = list(range(chosen, min(chosen + n_sweeps, len(sweeps))))
    if len(chosen_idxs) < n_sweeps:
        print("  only %d sweep(s) available from the chosen index onward "
             "(requested --n-sweeps %d) -- the elevation angle plot will "
             "overlay just those" % (len(chosen_idxs), n_sweeps))

    # Narrow the expensive pool_stripe_pixels call to the union of every
    # chosen sweep's own padded window, via narrow_frame_range's bracketing
    # technique (provably contains the globally-nearest usable frame for
    # every sample in that union -- no heuristic cadence-based pad guess).
    windows = []
    for si in chosen_idxs:
        s_i = sweeps[si]
        pad_i = pad_frac * (s_i["end_t"] - s_i["start_t"])
        windows.append((s_i["start_t"] - pad_i, s_i["end_t"] + pad_i))
    union_t0 = min(w[0] for w in windows)
    union_t1 = max(w[1] for w in windows)
    narrow_first, narrow_count = narrow_frame_range(
        scan_idx, scan_time_ns, t_ref, union_t0, union_t1)
    print("Pooling camera pixels for components {%s}, frames %d..%d "
         "(narrowed from %d candidate frames using the %d chosen sweep(s)' "
         "window [%.1f, %.1f] s) ..."
         % (",".join(components), narrow_first, narrow_first + narrow_count - 1,
            scan_idx.size, len(chosen_idxs), union_t0, union_t1))
    per_frame = pool_stripe_pixels(granule, components=components,
                                   max_frames=narrow_count, first_frame=narrow_first,
                                   stride=stride, table_dir=table_dir)
    if not per_frame:
        raise SystemExit("no usable camera frames in the narrowed range")
    frame_times_ns = np.array([f["time_ns"] for f in per_frame], dtype=np.int64)
    print("  %d usable camera frames in the narrowed range" % len(per_frame))
    # t_ref is fixed from the earlier full cheap scan, NOT recomputed from
    # this narrowed per_frame -- keeps "seconds since first usable camera
    # frame" labeling consistent no matter how much narrowing happened.
    # Same "seconds since t_ref" convention as nearest_frame_t_s/links' own
    # frame_t_s below, set here so plot_footprint_camera_diagnostic's second
    # (full-coverage) row colors every frame consistently with the top row,
    # even a frame with zero matched footprints of its own.
    for f in per_frame:
        f["frame_t_s"] = (f["time_ns"] - t_ref) / 1e9

    for key, varname in ANGLE_FIELDS.items():
        if not per_frame or key not in per_frame[0]:
            continue
        allvals = np.concatenate([f[key] for f in per_frame])
        if allvals.size and np.allclose(allvals, allvals[0]):
            print("  (WARNING: camera %s is a constant %.3f deg across every "
                 "pooled pixel -- looks like an unpopulated L1B placeholder, "
                 "not real geometry; its angle-diagnostic row will be flat/"
                 "meaningless)" % (varname, float(allvals[0])))

    pad = pad_frac * (sw["end_t"] - sw["start_t"])
    win_t0, win_t1 = sw["start_t"] - pad, sw["end_t"] + pad
    sel = np.where((time_s_full >= win_t0) & (time_s_full <= win_t1))[0]
    print("  %d radiometer samples in the padded sweep window [%.1f, %.1f] s"
         % (len(sel), win_t0, win_t1))

    t_s = time_s_full[sel]
    elev = rad["elevation"][sel]
    lat, lon, geo_valid = rad["lat"][sel], rad["lon"][sel], rad["geo_valid"][sel]
    times_ns_sel = rad["time"][sel]

    nearest_frame_idx, dt_s = nearest_frame_assignment(times_ns_sel, frame_times_ns)
    nearest_frame_t_s = (frame_times_ns[nearest_frame_idx].astype(np.float64) - t_ref) / 1e9

    frame_lat_list = [f["lat"] for f in per_frame]
    frame_lon_list = [f["lon"] for f in per_frame]
    dist_km = distances_to_assigned_frame(lat, lon, geo_valid, nearest_frame_idx,
                                          frame_lat_list, frame_lon_list)

    out_csv = write_sweep_csv(t_s, elev, nearest_frame_t_s, dt_s, dist_km,
                              os.path.join(out_dir, "%s_elevation_sweep.csv" % stem))
    print("  wrote %s" % out_csv)
    plot_sweep(t_s, elev, nearest_frame_t_s, dist_km, camera_t0, camera_t1,
              len(per_frame), sw, os.path.join(out_dir, "%s_elevation_sweep.png" % stem),
              rad_mode=rad_mode)

    print("Linking footprints to nearby camera pixels (match <= %.0f km, "
         "neighbor patch <= %.0f km) ..." % (match_radius_km, neighbor_radius_km))
    links = footprint_camera_links(lat, lon, geo_valid, nearest_frame_idx,
                                   frame_lat_list, frame_lon_list,
                                   match_radius_km=match_radius_km,
                                   neighbor_radius_km=neighbor_radius_km,
                                   t_s=t_s, peak_t=sw["peak_t"],
                                   frame_t_s=nearest_frame_t_s)
    print("  %d/%d geolocated footprints have a camera match within %.0f km"
         % (len(links), int(geo_valid.sum()), match_radius_km))

    plot_footprint_camera_map(
        links, os.path.join(out_dir, "%s_footprint_camera_map_full.png" % stem),
        title="Radiometer footprints vs. nearby camera pixels -- full sweep "
             "(t=%.1f-%.1f s)" % (sw["start_t"], sw["end_t"]))

    print("Computing per-footprint camera-patch angle stats vs. scan "
         "Elevation (%d km-diameter patch, match <= %.0f km) across %d "
         "sweep(s) ..." % (int(2 * patch_radius_km), match_radius_km, len(chosen_idxs)))
    sweep_data = []
    for si, (t0_i, t1_i) in zip(chosen_idxs, windows):
        s_i = sweeps[si]
        sel_i = np.where((time_s_full >= t0_i) & (time_s_full <= t1_i))[0]
        t_s_i = time_s_full[sel_i]
        elev_i = rad["elevation"][sel_i]
        lat_i, lon_i, geo_valid_i = rad["lat"][sel_i], rad["lon"][sel_i], rad["geo_valid"][sel_i]
        nearest_frame_idx_i, _ = nearest_frame_assignment(rad["time"][sel_i], frame_times_ns)
        stats_i = footprint_patch_angle_stats(
            lat_i, lon_i, geo_valid_i, nearest_frame_idx_i, per_frame,
            match_radius_km=match_radius_km, patch_radius_km=patch_radius_km,
            angle_keys=("sza", "vza", "raz", "pct_identified"))
        direction_i = np.where(t_s_i <= s_i["peak_t"], "rising", "falling")
        sweep_data.append(dict(
            label="sweep [%d] (t=%.1f-%.1f s)" % (si, s_i["start_t"], s_i["end_t"]),
            elev=elev_i, direction=direction_i, stats=stats_i))
    # plot_angle_timeseries (time on the x-axis, one sweep only) is kept
    # in this file -- still correct, just no longer called by default here --
    # in favor of this elevation-based, multi-sweep version, requested
    # 2026-10-07 specifically to compare swath-position (cross-track, within
    # one sweep's Elevation range) and along-track (between sweeps, at the
    # same Elevation) angle variability side by side.
    plot_angle_vs_elevation(
        sweep_data, os.path.join(out_dir, "%s_camera_angle_vs_elevation.png" % stem),
        patch_km=2 * patch_radius_km, match_radius_km=match_radius_km,
        angle_specs=ANGLE_TS_SPECS + (
            ("pct_identified", "% pixels identified\n(mask retrieval coverage)", "#6b6b6b"),))

    nadir_candidates = np.where(geo_valid)[0]
    if nadir_candidates.size:
        nadir_i = nadir_candidates[np.argmin(np.abs(elev[nadir_candidates]))]
        nadir_ref = (float(lat[nadir_i]), float(lon[nadir_i]))
        nadir_t = float(t_s[nadir_i])
        plot_footprint_camera_map(
            links, os.path.join(out_dir, "%s_footprint_camera_map_zoom%dkm.png"
                                % (stem, int(2 * zoom_half_width_km))),
            title="Radiometer footprints vs. nearby camera pixels -- "
                 "%d km zoom around nadir" % int(2 * zoom_half_width_km),
            center=nadir_ref, half_width_km=zoom_half_width_km)

        print("Building the %d km-zoom direction/frame diagnostic (nadir +/- "
             "%.0f km) ..." % (int(2 * diag_half_width_km), diag_offset_km))
        before, after = pick_offset_centers(lat, lon, geo_valid, t_s, nadir_t,
                                            nadir_ref[0], nadir_ref[1], diag_offset_km)
        centers, titles = [], []
        for label, pt in (("before", before), ("nadir", (nadir_ref[0], nadir_ref[1], 0.0)),
                         ("after", after)):
            if pt is None:
                centers.append(None)
                titles.append("~%.0f km %s nadir -- not available (window too "
                              "short on this side)" % (diag_offset_km, label))
                print("    %s side: no geolocated sample at all -- panel skipped"
                     % label)
                continue
            plat, plon, achieved_km = pt
            centers.append((plat, plon))
            if label == "nadir":
                titles.append("Nadir crossing (t=%.1f s)" % nadir_t)
            else:
                titles.append("~%.0f km %s nadir (achieved %.0f km)"
                              % (diag_offset_km, label, achieved_km))
                if abs(achieved_km - diag_offset_km) > 0.25 * diag_offset_km:
                    print("    %s side: window only reaches %.0f km, not the "
                         "requested %.0f km" % (label, achieved_km, diag_offset_km))
        plot_footprint_camera_diagnostic(
            links, os.path.join(out_dir, "%s_footprint_camera_diag%dkm.png"
                                % (stem, int(2 * diag_half_width_km))),
            centers, titles, half_width_km=diag_half_width_km, per_frame=per_frame)

        print("Building the %d km-zoom camera angle diagnostic (SZA/VZA/RAZ) ..."
             % int(2 * diag_half_width_km))
        plot_angle_diagnostic(
            per_frame, lat[geo_valid], lon[geo_valid],
            os.path.join(out_dir, "%s_footprint_camera_angles%dkm.png"
                        % (stem, int(2 * diag_half_width_km))),
            centers, titles, half_width_km=diag_half_width_km)
    else:
        print("  no geolocated sample in this sweep to anchor the nadir zoom map "
             "-- skipped")

    return sweeps, chosen


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("granule")
    p.add_argument("--rad", required=True)
    p.add_argument("--out", default="output_l2")
    p.add_argument("--rad-mode", type=int, default=128,
                   help="RAD-4CH Operational_Mode to select (default: 128, "
                       "'CT' per docs/NEXT.md's own measured 2026-10-07 "
                       "finding -- covers the whole scan-head rotation, not "
                       "just the Earth-viewing part of it; geo_valid "
                       "separates the two, see find_earth_view_sweeps)")
    p.add_argument("--components", default=",".join(CT_STRIPE_COMPONENTS),
                   help="camera pixel pool for the distance panel (default: "
                       "%(default)s, same as nadir_radiometer_overlap.py v2)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--first-frame", type=int, default=0)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--table-dir", default=None)
    p.add_argument("--sweep-index", type=int, default=None,
                   help="pick a specific sweep from the printed candidate "
                       "list instead of the auto-picked one")
    p.add_argument("--pad-frac", type=float, default=0.1,
                   help="context padding before/after the sweep, as a "
                       "fraction of its duration (default 0.1)")
    p.add_argument("--max-gap-s", type=float, default=None,
                   help="start a new Earth-view sweep instead of bridging a "
                       "geo_valid dropout bigger than this many seconds "
                       "(default: max(3x the median geo_valid sample "
                       "spacing, 1.0 s) -- the floor matters since this "
                       "granule samples at ~0.01 s but its turning-point "
                       "dropouts are ~0.6 s)")
    p.add_argument("--match-radius-km", type=float, default=50.0,
                   help="max footprint-to-nearest-camera-pixel distance to "
                       "plot on the footprint/camera maps (default 50)")
    p.add_argument("--neighbor-radius-km", type=float, default=20.0,
                   help="radius around each matched pixel to also show as a "
                       "local camera-pixel patch (default 20)")
    p.add_argument("--zoom-half-width-km", type=float, default=250.0,
                   help="half-width of the nadir-centered zoom map, i.e. "
                       "250 -> a 500 km-wide view (default 250)")
    p.add_argument("--diag-half-width-km", type=float, default=50.0,
                   help="half-width of each of the 3 direction/frame "
                       "diagnostic zoom panels, i.e. 50 -> a 100 km-wide "
                       "view (default 50)")
    p.add_argument("--diag-offset-km", type=float, default=500.0,
                   help="how far off nadir (great-circle distance, both "
                       "directions) the diagnostic's two off-nadir panels "
                       "aim for (default 500)")
    p.add_argument("--patch-radius-km", type=float, default=12.5,
                   help="radius of the camera-pixel patch each footprint's "
                       "angle stats (mean/min/max) are computed over for "
                       "the angle-vs-elevation plot, i.e. 12.5 -> a 25 km-"
                       "diameter patch (default 12.5)")
    p.add_argument("--n-sweeps", type=int, default=2,
                   help="number of consecutive Earth-view sweeps (starting "
                       "at the picked one) to overlay on the camera-angle-"
                       "vs-elevation plot, to compare along-track as well "
                       "as cross-track angle variability (default 2)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    comps = tuple(c.strip() for c in a.components.split(",") if c.strip())
    run(a.granule, a.rad, out_dir=a.out, rad_mode=a.rad_mode, components=comps,
        max_frames=a.max_frames, first_frame=a.first_frame, stride=a.stride,
        table_dir=a.table_dir, sweep_index=a.sweep_index, pad_frac=a.pad_frac,
        max_gap_s=a.max_gap_s, match_radius_km=a.match_radius_km,
        neighbor_radius_km=a.neighbor_radius_km,
        zoom_half_width_km=a.zoom_half_width_km,
        diag_half_width_km=a.diag_half_width_km,
        diag_offset_km=a.diag_offset_km,
        patch_radius_km=a.patch_radius_km,
        n_sweeps=a.n_sweeps)
