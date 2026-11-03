"""
l2/proxy_radiance.py

Dev-step-2, stage 2 of the "L1X" proxy-radiance plan: project a cropped
MODIS/VIIRS scene (l2/inspect_proxy_scene.py) onto the REAL per-
pixel geolocation of REAL camera frames (not the synthetic libcam.camera
Platform model gcp_proxy.py validates against), using the same
ellipsoidal-Gaussian PSF (Platform.mod2cam(sampling="psf")) that proxy
work already uses.

Key finding (2026-10-22): Platform.pixel_spacing() -- which sizes the
PSF -- is computed purely from Platform.cam_lon/cam_lat via finite
differences; it only falls back to the synthetic geomap() if those are
still None. So we build a bare Platform(), overwrite .cam_lon/.cam_lat
with one real camera frame's real geolocation (from the actual CAM L1B
granule, via l2/elevation_sweep_timing.py's open_granule() -- same
reader run_tile_rt.py's chop()/tile() use), and mod2cam() projects onto
real pixel positions with real-geometry PSF sizing, no synthetic model
involved. mod2cam() works on ONE frame at a time (output shape is the
camera's fixed (FPA_NX, FPA_NY) detector grid) -- a 100 km tile spans
several real frames, so this module loops the single-frame projection
over every geolocated frame in the tile's time window whose footprint
actually overlaps the tile box (--single-frame falls back to the
original one-frame proof-of-mechanism path, closest in time to the
matched radiometer sample).

Key finding #2 (2026-10-23, after the first real multi-frame run came
back with ~122 overlapping frames instead of the ~10 dev-step-1 sees,
and a washed-out, gap-free coastline): the camera's "Earth-view" per
frame is NOT a narrow nadir footprint -- its lens model's FOV extends to
VZA_LIMIT=63 deg (libcam/common.py), which at 824 km altitude is right
at the Earth limb, so a single frame genuinely geolocates most of its
full (FPA_NX, FPA_NY) grid (confirmed: 78% Earth-view in the earlier
single-frame test). But in this ("Cross-Track") observing mode the
camera does not actually expose that whole grid every frame -- only a
sparse, precessing pattern of pixels is read out (named "nadir" /
"forward" / "cross" stripe components, plus an ADM ring pattern), which
is exactly what l2/elevation_sweep_timing.py's pool_stripe_pixels()
already decodes per frame via libcam.mask.retrieve_mask.MaskRetriever
and the granule's own pixel-mask field (MASK_FIELDS). Using raw
isfinite(lat/lon) as "this pixel was measured this frame" (what this
module did before) is wrong by a huge margin -- it treats the camera's
entire near-horizon-to-near-horizon FOV as valid data every frame, when
only the narrow stripe component(s) actually were. stripe_mask_for_frame()
/ restrict_to_stripe() below reuse MaskRetriever exactly as
pool_stripe_pixels() does (same CT_STRIPE_COMPONENTS, same bitmask
membership test) to NaN out everything outside the real exposed stripe
before projection, which is also what should recover the gaps between
consecutive stripes that a real camera image shows and the old full-FPA
version washed out.

The MODIS/VIIRS source crop is also tightened to the tile's own bounding
box before projection: ModisL1B.crop() (l2/inspect_proxy_scene.py)
keeps full scanlines -- the entire cross-track swath width -- for any
row with >=1 pixel inside the requested box, so its raw output can span
far more longitude than the tile itself (confirmed: ~40 deg of longitude
for a tile that is ~1.8 deg wide at this latitude). That extra margin
is correctness-neutral for mod2cam() (its PSF search radius is a few
km, so the distant extra points are never within range) but wasteful to
carry through PSF projection on every frame in a loop, so it is masked
down to the exact box once, up front.

The next stage (not yet implemented here) splices each kept frame's
projected radiance into a cropped copy of the real CAM L1B -- leaving
its mask/angles/geolocation/everything else untouched -- to build the
actual "L1X" file the L2 algorithm will read in place of L1B.

Usage
-----
    # loop over every frame in the tile's time window (default)
    python l2/proxy_radiance.py

    # original single-frame proof of mechanism (closest-in-time frame
    # only), for comparison/debugging
    python l2/proxy_radiance.py --single-frame
"""
import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_tile_rt                        # noqa: E402  (for chop())
import elevation_sweep_timing as est       # noqa: E402  (open_granule(), frame geolocation)
import inspect_proxy_scene as ips          # noqa: E402  (MODIS crop, tile_pick loading, tile_bbox)

DEFAULT_CAM_L1B = "data/DITL2/L1B/LIBERA_L1B_CAM_V0-3-2_20260712T001717_20260712T040151_R26281034503.nc"
DEFAULT_RAD_L1B = "data/DITL2/L1B/LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_R26281034503.nc"
DEFAULT_MODIS_GEO = "l2/data/proxy_imagery/MYD03.A2014193.0305.061.2018053133301.hdf"
DEFAULT_MODIS_L1B = "l2/data/proxy_imagery/MYD02HKM.A2014193.0305.061.2018054015044.hdf"

DEFAULT_WINDOW_S = 300.0     # same default as l2/run_dev_step1.py
DEFAULT_SCAN_PAD_S = 500.0   # same default as l2/run_dev_step1.py
DEFAULT_COMPONENTS = est.CT_STRIPE_COMPONENTS   # ("nadir", "forward", "cross")


def get_frame_latlon(cam_l1b, frame_idx, ds=None):
    """Real per-pixel geolocation for ONE camera frame, as a (FPA_NX,
    FPA_NY) pair of (lat, lon) arrays -- same field names/transpose/fill-
    value convention l2/elevation_sweep_timing.py's own geo_report()
    and pool_stripe_pixels() use, so this lines up with the rest of the
    pipeline's idea of "this frame's geolocation". Off-Earth/fill pixels
    (FILL_VAL=-999, or already-NaN) come back as NaN -- Platform.mod2cam()
    already filters on np.isfinite(cam_lon/cam_lat) internally, but raw
    fill values are finite numbers, so without this they would silently
    be treated as real (lat,lon)=(-999,-999)-ish positions instead of
    being excluded.

    Pass an already-open ``ds`` (est.open_granule(cam_l1b)) when calling
    this for many frames from the same granule in a loop, to avoid
    reopening the file every time."""
    if ds is None:
        ds = est.open_granule(cam_l1b)
    lat_f = est._first(ds, est.LAT_FIELDS)
    lon_f = est._first(ds, est.LON_FIELDS)
    if lat_f is None or lon_f is None:
        raise SystemExit("%s has no latitude/longitude field at all "
                         "(looked for %s / %s)"
                         % (cam_l1b, est.LAT_FIELDS, est.LON_FIELDS))

    lat = est._fpa(ds[lat_f].isel(CAMERA_TIME=frame_idx).values).astype(np.float64)
    lon = est._fpa(ds[lon_f].isel(CAMERA_TIME=frame_idx).values).astype(np.float64)

    bad = (~np.isfinite(lat)) | (~np.isfinite(lon)) | (lat <= est.FILL_VAL) | (lon <= est.FILL_VAL)
    lat[bad] = np.nan
    lon[bad] = np.nan
    return lat, lon


def project_frame_psf(cam_lat, cam_lon, src_lon, src_lat, src_rad):
    """Run the ellipsoidal-Gaussian PSF projection
    (libcam.camera.Platform.mod2cam) for ONE real camera frame, given its
    real (FPA_NX, FPA_NY) geolocation (cam_lat, cam_lon) and a source
    imagery crop (src_lon/src_lat/src_rad, any shape -- mod2cam flattens
    them). Returns the (FPA_NX, FPA_NY) projected radiance array (NaN
    where no source data was within range or the camera pixel itself is
    off-Earth)."""
    from libcam.camera import Platform
    import libcam.common as C

    if cam_lat.shape != (C.FPA_NX, C.FPA_NY):
        raise SystemExit(
            "frame lat/lon shape %s does not match the camera's fixed FPA "
            "shape (%d, %d) -- check the _fpa() transpose convention"
            % (cam_lat.shape, C.FPA_NX, C.FPA_NY))

    n_valid_cam = int(np.isfinite(cam_lat).sum())
    if n_valid_cam == 0:
        raise SystemExit(
            "This frame has NO geolocated (Earth-view) pixels at all -- "
            "pick a different frame_idx (this one is probably pointed at "
            "space, see docs/L1B_TO_L2_ROADMAP.md's Camera_Observation_ID "
            "notes).")

    platform = Platform()
    platform.cam_lon = cam_lon
    platform.cam_lat = cam_lat

    return platform.mod2cam(src_lon, src_lat, src_rad, sampling="psf")


def stripe_mask_for_frame(ds, frame_idx, mask_f, header, retriever,
                          components=DEFAULT_COMPONENTS):
    """Run the SAME per-frame WFOV pixel-mask retrieval
    l2/elevation_sweep_timing.py's pool_stripe_pixels() uses -- the
    camera's real, sparse per-exposure pixel-selection pattern (named
    "nadir"/"forward"/"cross" stripe components, out of
    libcam.mask.retrieve_mask's MaskRetriever) -- but return the full
    (FPA_NX, FPA_NY) boolean is-this-pixel-actually-selected-this-exposure
    mask, rather than pool_stripe_pixels()'s own flattened, post-stride
    (lat, lon) output. We need the full-FPA shape here so the unselected
    pixels can be NaN'd out of cam_lat/cam_lon before Platform.mod2cam()
    -- which needs a regular (FPA_NX, FPA_NY) grid to size its PSF via
    pixel_spacing()'s np.gradient, so a flattened array of just the
    selected pixels would not work there.

    ``retriever`` must be a single libcam.mask.retrieve_mask.MaskRetriever
    instance reused (in increasing frame-index order) across every frame
    of one run -- it holds a rolling buffer/cache the tier-2/3 refits use,
    exactly as pool_stripe_pixels() reuses one retriever across its own
    frame loop.

    Returns ``(is_sel, status)``: ``is_sel`` is None (check ``status`` for
    why) when the mask retrieval itself failed for this frame -- a
    different, earlier failure mode than "this frame's stripe doesn't
    overlap the tile", and reported separately by callers."""
    obs = est.frame_mask(ds, frame_idx, mask_f)
    res = retriever.process_frame(obs, header=header, index=frame_idx,
                                  az=header["az"])
    if not (res.status.ok and res.membership is not None):
        return None, res.status

    combined_bit = 0
    for c in components:
        combined_bit |= int(est.MASK_BITS[c])
    nb = res.membership.astype(np.uint32)
    is_sel = (nb & combined_bit) != 0
    return is_sel, res.status


def restrict_to_stripe(cam_lat, cam_lon, ds, frame_idx, mask_f, hdrs,
                       retriever, components=DEFAULT_COMPONENTS):
    """Apply stripe_mask_for_frame()'s selection to one frame's already-
    read (cam_lat, cam_lon): NaN out everything outside the real,
    per-exposure selected pixel pattern (see module docstring, "Key
    finding #2") instead of trusting raw isfinite(lat/lon), which is true
    for almost the whole FPA (this camera's FOV reaches VZA_LIMIT=63 deg,
    not just a narrow nadir patch) regardless of which pixels this
    exposure actually measured. Returns ``(cam_lat, cam_lon, status)``;
    ``(None, None, status)`` if the mask retrieval itself failed for this
    frame."""
    is_sel, status = stripe_mask_for_frame(ds, frame_idx, mask_f,
                                           hdrs[frame_idx], retriever,
                                           components=components)
    if is_sel is None:
        return None, None, status
    return np.where(is_sel, cam_lat, np.nan), np.where(is_sel, cam_lon, np.nan), status


def tighten_source_crop(modis, lon_min, lon_max, lat_min, lat_max, verbose=True):
    """Mask inspect_proxy_scene.inspect_crop()'s returned lon/lat/rad
    (2D arrays, one full-scanline crop) down to the tile's own bounding
    box. See the module docstring for why ModisL1B.crop() hands back more
    than the box asks for. Returns flattened (lon, lat, rad) arrays
    restricted to the exact box, with non-finite radiance already
    dropped (mod2cam() flattens its source arrays internally anyway, so
    there is no shape to preserve)."""
    lon = np.asarray(modis["lon"])
    lat = np.asarray(modis["lat"])
    rad = np.asarray(modis["rad"])

    inside = ((lon >= lon_min) & (lon <= lon_max) &
             (lat >= lat_min) & (lat <= lat_max) & np.isfinite(rad))
    n_before = rad.size
    n_after = int(inside.sum())
    if verbose:
        print("  tightened MODIS source crop to the exact tile box: "
             "%d / %d pixels kept (%.1f%%)"
             % (n_after, n_before, 100.0 * n_after / n_before if n_before else 0.0))
    if n_after == 0:
        raise SystemExit(
            "Tightening the MODIS crop to the exact tile box left ZERO "
            "pixels -- the tile box and the downloaded MODIS crop may not "
            "actually overlap (check --tile-pick lat/lon against the "
            "granule fetch_proxy_scene.py found).")
    return lon[inside], lat[inside], rad[inside]


def pick_one_frame(cam_l1b, rad_l1b, lat0, lon0, window_s, scan_pad_s,
                   verbose=True):
    """Same chop() l2/run_dev_step1.py uses to find the tile's real
    camera-frame window, then the single frame inside it closest in time
    to the matched radiometer sample -- the original stage-2 single-
    frame proof (--single-frame); project_tile_frames() below loops over
    every frame in the window instead of picking just one."""
    match = run_tile_rt.chop(rad_l1b, cam_l1b, lat0, lon0, window_s,
                             scan_pad_s=scan_pad_s, verbose=verbose)
    scan_idx, scan_time_ns = est.scan_camera_frame_times(
        cam_l1b, first_frame=match["first_frame"], max_frames=match["max_frames"])
    if scan_idx.size == 0:
        raise SystemExit(
            "No geolocated camera frames in the matched window at all -- "
            "widen --window-s/--scan-pad-s.")
    i_closest = int(np.argmin(np.abs(scan_time_ns - match["t_ns"])))
    frame_idx = int(scan_idx[i_closest])
    if verbose:
        import datetime
        offset_s = (int(scan_time_ns[i_closest]) - match["t_ns"]) / 1e9
        print("  picked frame %d, %.1f s from the matched radiometer sample "
             "(%s)" % (frame_idx, offset_s,
                      datetime.datetime.utcfromtimestamp(scan_time_ns[i_closest] / 1e9)))
    return frame_idx, match


def project_tile_frames(cam_l1b, rad_l1b, lat0, lon0, window_s, scan_pad_s,
                        tile_bbox_deg, src_lon, src_lat, src_rad,
                        components=DEFAULT_COMPONENTS, table_dir=None,
                        verbose=True):
    """Loop version of pick_one_frame() + project_frame_psf(): find the
    tile's real camera-frame window via chop() (same as the single-frame
    path), then for EVERY geolocated frame in that window, restrict its
    geolocation to the camera's real per-exposure stripe pattern
    (restrict_to_stripe() -- see module docstring, "Key finding #2") and
    run the PSF projection only if that restricted set has at least one
    pixel inside the tile's own lat/lon box -- not just the one frame
    closest in time to the matched radiometer sample, and not the raw
    full-FPA geolocation (which is populated far beyond what this
    exposure actually measured). Frames are skipped, and counted
    separately, for three distinct reasons: the pixel-mask retrieval
    itself failed for that frame; the retrieval succeeded but the
    selected stripe pixels don't reach the tile box; or they do, but the
    PSF projection still finds zero matches (box edge cases, gaps in the
    source crop).

    Returns ``(cam_rad_by_frame, match, combined)``:
      - cam_rad_by_frame: {frame_idx: (FPA_NX, FPA_NY) cam_rad array},
        one entry per kept frame -- what the next stage (L1X writer)
        will splice into a cropped copy of the real CAM L1B.
      - match: the chop() match dict (same as pick_one_frame() returns).
      - combined: {"lon", "lat", "rad"} flattened 1D arrays of every
        camera pixel that got a value, concatenated across all kept
        frames -- for aggregate stats/plotting without having to keep
        every frame's full (FPA_NX, FPA_NY) cam_lat/cam_lon around."""
    lon_min, lon_max, lat_min, lat_max = tile_bbox_deg

    match = run_tile_rt.chop(rad_l1b, cam_l1b, lat0, lon0, window_s,
                             scan_pad_s=scan_pad_s, verbose=verbose)
    scan_idx, scan_time_ns = est.scan_camera_frame_times(
        cam_l1b, first_frame=match["first_frame"], max_frames=match["max_frames"])
    if scan_idx.size == 0:
        raise SystemExit(
            "No geolocated camera frames in the matched window at all -- "
            "widen --window-s/--scan-pad-s.")

    if verbose:
        print("  %d geolocated camera frame(s) in the matched window "
             "(camera frames [%d, %d)) -- that's a raw geolocation-"
             "presence count, before the per-exposure stripe mask narrows "
             "it down"
             % (scan_idx.size, match["first_frame"],
                match["first_frame"] + match["max_frames"]))

    ds = est.open_granule(cam_l1b)  # opened once, reused for every frame below

    mask_f = est._first(ds, est.MASK_FIELDS)
    if mask_f is None:
        raise SystemExit(
            "No pixel-mask field (%s) in %s -- can't restrict the "
            "projection to the camera's real per-exposure stripe pattern."
            % (est.MASK_FIELDS, cam_l1b))
    hdrs = est.frame_headers(ds)
    # One retriever, reused in increasing frame-index order across the
    # whole loop -- it holds a rolling cache the tier-2/3 refits use, same
    # as pool_stripe_pixels()'s own single `r = MaskRetriever(...)`.
    retriever = est.MaskRetriever(table_dir=table_dir, verbose=False)

    cam_rad_by_frame = {}
    plot_lon, plot_lat, plot_rad = [], [], []
    n_skipped_mask_fail = 0
    n_skipped_no_overlap = 0
    n_skipped_zero_psf = 0

    for frame_idx in scan_idx.tolist():
        cam_lat, cam_lon = get_frame_latlon(cam_l1b, frame_idx, ds=ds)
        cam_lat, cam_lon, status = restrict_to_stripe(
            cam_lat, cam_lon, ds, frame_idx, mask_f, hdrs, retriever,
            components=components)
        if cam_lat is None:
            n_skipped_mask_fail += 1
            if verbose:
                print("  frame %d: SKIPPED (pixel-mask retrieval failed: %s)"
                     % (frame_idx, getattr(status, "message", "") or
                                  getattr(status, "code", "unknown")))
            continue

        inside = (np.isfinite(cam_lat) & np.isfinite(cam_lon) &
                 (cam_lon >= lon_min) & (cam_lon <= lon_max) &
                 (cam_lat >= lat_min) & (cam_lat <= lat_max))
        n_inside = int(inside.sum())
        if n_inside == 0:
            n_skipped_no_overlap += 1
            continue

        cam_rad = project_frame_psf(cam_lat, cam_lon, src_lon, src_lat, src_rad)
        finite = np.isfinite(cam_rad)
        n_finite = int(finite.sum())
        if n_finite == 0:
            n_skipped_zero_psf += 1
            if verbose:
                print("  frame %d: %d stripe pixel(s) inside the tile box, "
                     "but the PSF projection found 0 camera pixels with a "
                     "value (skipped)" % (frame_idx, n_inside))
            continue

        cam_rad_by_frame[frame_idx] = cam_rad
        plot_lon.append(cam_lon[finite])
        plot_lat.append(cam_lat[finite])
        plot_rad.append(cam_rad[finite])

        if verbose:
            print("  frame %d: %d stripe pixel(s) inside the tile box -> "
                 "%d camera pixels got a value (mean=%.4f)"
                 % (frame_idx, n_inside, n_finite, float(np.nanmean(cam_rad))))

    if verbose:
        print("\n  %d / %d frame(s) kept, %d skipped (pixel-mask retrieval "
             "failed), %d skipped (no stripe pixels inside the tile box), "
             "%d skipped (0 PSF matches despite overlap)"
             % (len(cam_rad_by_frame), scan_idx.size, n_skipped_mask_fail,
                n_skipped_no_overlap, n_skipped_zero_psf))

    if not cam_rad_by_frame:
        raise SystemExit(
            "No frame in the matched window had any stripe pixels inside "
            "the tile box -- check --tile-pick lat/lon, --window-s/"
            "--scan-pad-s, and --components.")

    combined = dict(
        lon=np.concatenate(plot_lon),
        lat=np.concatenate(plot_lat),
        rad=np.concatenate(plot_rad),
    )
    return cam_rad_by_frame, match, combined


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-pick", default=ips.DEFAULT_TILE_PICK)
    p.add_argument("--cam-l1b", default=DEFAULT_CAM_L1B)
    p.add_argument("--rad-l1b", default=DEFAULT_RAD_L1B)
    p.add_argument("--window-s", type=float, default=DEFAULT_WINDOW_S)
    p.add_argument("--scan-pad-s", type=float, default=DEFAULT_SCAN_PAD_S)
    p.add_argument("--modis-geo", default=DEFAULT_MODIS_GEO)
    p.add_argument("--modis-l1b", default=DEFAULT_MODIS_L1B)
    p.add_argument("--tile-km", type=float, default=ips.DEFAULT_TILE_KM)
    p.add_argument("--band", type=int, default=ips.MODIS_BAND)
    p.add_argument("--components", default=",".join(DEFAULT_COMPONENTS),
                  help="comma-separated pixel-mask stripe component(s) to "
                       "keep as this exposure's real measured pixels "
                       "(default: %(default)s, same as "
                       "elevation_sweep_timing.CT_STRIPE_COMPONENTS)")
    p.add_argument("--table-dir", default=None,
                  help="MaskRetriever calibration table directory "
                       "(default: None, same default pool_stripe_pixels() "
                       "and MaskRetriever itself use)")
    p.add_argument("--single-frame", action="store_true",
                  help="original stage-2 proof of mechanism: project just "
                       "the one frame closest in time to the matched "
                       "radiometer sample, instead of looping over every "
                       "frame in the tile's time window")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    lat0, lon0 = ips.load_tile_pick(args.tile_pick)
    print("Tile pick (%s): lat=%.4f lon=%.4f" % (args.tile_pick, lat0, lon0))

    print("\nLoading MODIS crop (%s, %s) ..."
         % (os.path.basename(args.modis_geo), os.path.basename(args.modis_l1b)))
    modis = ips.inspect_crop(args.modis_geo, args.modis_l1b, lat0, lon0,
                             args.tile_km, band=args.band, verbose=True)

    lon_min, lon_max, lat_min, lat_max = ips.tile_bbox(lat0, lon0, args.tile_km)
    src_lon, src_lat, src_rad = tighten_source_crop(
        modis, lon_min, lon_max, lat_min, lat_max)

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())

    if args.single_frame:
        print("\nFinding this tile's real camera-frame window (chop()) ...")
        frame_idx, match = pick_one_frame(args.cam_l1b, args.rad_l1b, lat0, lon0,
                                          args.window_s, args.scan_pad_s)

        print("\nReading frame %d's real geolocation from %s ..."
             % (frame_idx, os.path.basename(args.cam_l1b)))
        cam_lat, cam_lon = get_frame_latlon(args.cam_l1b, frame_idx)
        n_geoloc = int(np.isfinite(cam_lat).sum())
        print("  %d / %d pixels geolocated (Earth-view) in this frame (%.1f%%) "
             "-- the camera's full FOV reaches VZA_LIMIT=63 deg, so this is "
             "NOT yet restricted to what this exposure actually measured"
             % (n_geoloc, cam_lat.size, 100.0 * n_geoloc / cam_lat.size))

        print("\nRestricting to this exposure's real pixel-mask stripe "
             "(%s) ..." % ",".join(components))
        ds = est.open_granule(args.cam_l1b)
        mask_f = est._first(ds, est.MASK_FIELDS)
        if mask_f is None:
            raise SystemExit(
                "No pixel-mask field (%s) in %s -- can't restrict to the "
                "camera's real per-exposure stripe pattern."
                % (est.MASK_FIELDS, args.cam_l1b))
        hdrs = est.frame_headers(ds)
        retriever = est.MaskRetriever(table_dir=args.table_dir, verbose=False)
        cam_lat, cam_lon, status = restrict_to_stripe(
            cam_lat, cam_lon, ds, frame_idx, mask_f, hdrs, retriever,
            components=components)
        if cam_lat is None:
            raise SystemExit("Pixel-mask retrieval failed for frame %d: %s"
                             % (frame_idx, status))
        n_valid = int(np.isfinite(cam_lat).sum())
        print("  %d / %d pixels are this exposure's real, measured stripe "
             "pixels (%.1f%% of the frame, %.1f%% of what was geolocated)"
             % (n_valid, cam_lat.size, 100.0 * n_valid / cam_lat.size,
                100.0 * n_valid / n_geoloc if n_geoloc else 0.0))

        print("\nRunning PSF projection (Platform.mod2cam) ...")
        cam_rad = project_frame_psf(cam_lat, cam_lon, src_lon, src_lat, src_rad)

        finite = np.isfinite(cam_rad)
        n_finite = int(finite.sum())
        print("\nProjected camera-frame radiance:")
        print("  %d / %d camera pixels got a value (%.1f%% of the full FPA, "
             "%.1f%% of the %d geolocated pixels in this frame)"
             % (n_finite, cam_rad.size, 100.0 * n_finite / cam_rad.size,
                100.0 * n_finite / n_valid if n_valid else 0.0, n_valid))
        if n_finite:
            print("  min=%.4f  mean=%.4f  max=%.4f  [W m-2 sr-1 um-1]  "
                 "(source MODIS crop, tightened: min=%.4f mean=%.4f max=%.4f)"
                 % (float(np.nanmin(cam_rad)), float(np.nanmean(cam_rad)),
                    float(np.nanmax(cam_rad)), float(np.nanmin(src_rad)),
                    float(np.nanmean(src_rad)), float(np.nanmax(src_rad))))
        else:
            print("  WARNING: zero camera pixels got a value -- the MODIS "
                 "crop and this frame's footprint may not actually overlap, "
                 "or the PSF search radius found nothing nearby.")

        if args.plot:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            sc0 = axes[0].scatter(src_lon, src_lat, c=src_rad, s=3, cmap="viridis")
            axes[0].set_title("MODIS source crop (tightened to tile box)")
            plt.colorbar(sc0, ax=axes[0])
            im1 = axes[1].imshow(cam_rad.T, origin="lower", cmap="viridis")
            axes[1].set_title("Projected camera frame %d" % frame_idx)
            plt.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            plt.show()
        return

    print("\nLooping over every geolocated camera frame in the tile's time "
         "window, restricted to this exposure's real (%s) pixel-mask "
         "stripe ..." % ",".join(components))
    cam_rad_by_frame, match, combined = project_tile_frames(
        args.cam_l1b, args.rad_l1b, lat0, lon0, args.window_s, args.scan_pad_s,
        (lon_min, lon_max, lat_min, lat_max), src_lon, src_lat, src_rad,
        components=components, table_dir=args.table_dir)

    all_rad = combined["rad"]
    print("\nAggregate projected camera radiance across %d kept frame(s):"
         % len(cam_rad_by_frame))
    print("  %d total camera pixels with a value  min=%.4f  mean=%.4f  "
         "max=%.4f  [W m-2 sr-1 um-1]  (source MODIS crop, tightened: "
         "min=%.4f mean=%.4f max=%.4f)"
         % (all_rad.size, float(np.nanmin(all_rad)), float(np.nanmean(all_rad)),
            float(np.nanmax(all_rad)), float(np.nanmin(src_rad)),
            float(np.nanmean(src_rad)), float(np.nanmax(src_rad))))

    if args.plot:
        import matplotlib.pyplot as plt
        vmin = min(float(np.nanmin(src_rad)), float(np.nanmin(all_rad)))
        vmax = max(float(np.nanmax(src_rad)), float(np.nanmax(all_rad)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        sc0 = axes[0].scatter(src_lon, src_lat, c=src_rad, s=3, cmap="viridis",
                              vmin=vmin, vmax=vmax)
        axes[0].plot(lon0, lat0, "r+", markersize=14, markeredgewidth=2,
                    label="tile center")
        axes[0].set_title("MODIS source crop (tightened to tile box)")
        axes[0].legend(loc="upper right")
        plt.colorbar(sc0, ax=axes[0])
        sc1 = axes[1].scatter(combined["lon"], combined["lat"], c=combined["rad"],
                              s=3, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].plot(lon0, lat0, "r+", markersize=14, markeredgewidth=2,
                    label="tile center")
        axes[1].set_title("Projected camera pixels, %d frame(s)"
                          % len(cam_rad_by_frame))
        axes[1].legend(loc="upper right")
        plt.colorbar(sc1, ax=axes[1])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
