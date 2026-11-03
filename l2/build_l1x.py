"""
l2/build_l1x.py

Dev-step-2, stage 3 of the "L1X" proxy-radiance plan (Sebastian,
2026-10-23): write the actual "L1X" file -- a cropped copy of the real
CAM L1B, covering just the exposures in the tile's chop() time window,
with the Radiance variable overwritten ONLY at pixels belonging to the
camera's real "nadir" stripe component (l2/proxy_radiance.py's
restrict_to_stripe(), via libcam.mask.retrieve_mask.MaskRetriever) AND
covered by the downloaded MODIS/VIIRS proxy scene. Every other pixel --
non-nadir components (forward/cross/adm/raps/limb/unexplained), and
nadir pixels the MODIS crop doesn't reach -- keeps its original L1B
value verbatim (dark/pre-launch placeholder, not realistic, but left
untouched, per Sebastian's explicit instruction: "the others can remain
what they were in the original L1B file").

Everything that finds, downloads and projects the proxy radiance is
unchanged from l2/proxy_radiance.py (imported, not duplicated) --
this script adds the "copy L1B, splice in only the selected-component
pixels, write a new file" step on top, plus one thing proxy_radiance.py
never needed: a UNITS conversion. Caught independently while building
this script, not something Sebastian flagged: the real CAM L1B's own
"Radiance" variable documents its units as W/(m^2*sr*nm) (per
NANOmeter), but MODIS/VIIRS L1B radiance is W/(m^2*sr*um) (per
MICROmeter) -- 1 um = 1000 nm, so writing a MODIS value in unconverted
would be 1000x too large, silently, without tripping the variable's own
valid_range=[0,1000] check (a typical MODIS value of ~100-450 divided by
1000 looks like a perfectly ordinary in-range camera radiance if nobody
converts it). See MODIS_RADIANCE_UM_TO_CAMERA_NM below -- applied once,
explicitly, at the point of writing, same "flag it, don't silently
assume it" convention this codebase already uses elsewhere (see
write_cf_cam_output.py's radiance_source/libera_psf attribute flagging).

Also adds an explicit per-pixel provenance variable, L1X_Proxy_Mask (1
where Radiance was actually overwritten this run, 0 everywhere else,
including every pixel of every exposure whose selected-component stripe
never reached the tile) -- so nothing downstream has to re-run the mask
retrieval just to know which pixels in this file are real proxy data
versus leftover L1B placeholder, and so a plot (--plot here) can show
only the pixels that will actually matter to L2, per Sebastian's
instruction not to display the untouched elements.

Usage
-----
    python l2/build_l1x.py
    python l2/build_l1x.py --plot   # show only the overwritten pixels
    python l2/build_l1x.py --components nadir,forward  # widen later
"""
import argparse
import datetime
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import elevation_sweep_timing as est       # noqa: E402  (open_granule())
import inspect_proxy_scene as ips          # noqa: E402  (MODIS crop, tile_pick, tile_bbox)
import proxy_radiance as pr                # noqa: E402  (project_tile_frames(), tighten_source_crop())

# See module docstring. 1 um = 1000 nm, so W/(m^2 sr um) -> W/(m^2 sr nm)
# divides by 1000.
MODIS_RADIANCE_UM_TO_CAMERA_NM = 1.0e-3

# Sebastian, 2026-10-23: nadir stripe only for now; forward/cross/adm/
# raps/limb stay as whatever the original L1B had (not populated yet).
DEFAULT_COMPONENTS = ("nadir",)
DEFAULT_OUT_DIR = "l2/data/l1x"


def l1x_path_for(cam_l1b, out_dir=DEFAULT_OUT_DIR):
    """LIBERA_L1B_CAM_...nc -> <out_dir>/LIBERA_L1X_CAM_...nc (falls back
    to appending _L1X before the extension if "L1B" isn't in the name)."""
    base = os.path.basename(cam_l1b)
    if "L1B" in base:
        base = base.replace("L1B", "L1X")
    else:
        stem, ext = os.path.splitext(base)
        base = stem + "_L1X" + ext
    return os.path.join(out_dir, base)


def splice_into_l1b_crop(ds, first_frame, max_frames, cam_rad_by_frame,
                         components, verbose=True):
    """Crop ``ds`` to camera frames [first_frame, first_frame+max_frames),
    load it into memory, and overwrite the Radiance variable's values at
    exactly the pixels project_tile_frames() produced a finite value for
    (already restricted to ``components`` and to pixels the proxy source
    crop actually covers -- see l2/proxy_radiance.py). Every other
    pixel of every frame in the crop -- including every pixel of a frame
    not in ``cam_rad_by_frame`` at all -- is left exactly as it was in
    the original L1B.

    cam_rad_by_frame's arrays are in the "_fpa" orientation (see
    elevation_sweep_timing._fpa(): a transpose of the on-disk
    (CAMERA_PIXEL_COUNT_X, CAMERA_PIXEL_COUNT_Y) layout), matching
    get_frame_latlon()'s own convention -- transposed back (".T", self-
    inverse for a square array) here before writing into ds_out, whose
    Radiance variable is in the native on-disk orientation.

    Returns ``(ds_out, proxy_mask, n_overwritten_px)``: proxy_mask is the
    same (n, X, Y) boolean array now stored as the uint8 L1X_Proxy_Mask
    variable; n_overwritten_px is its total pixel count, across every
    kept frame, for the caller's own summary print."""
    f0, f1 = first_frame, first_frame + max_frames
    ds_out = ds.isel(CAMERA_TIME=slice(f0, f1)).load()

    orig_rad = ds_out["Radiance"].values   # native (n, X, Y) orientation
    new_rad = orig_rad.copy()
    proxy_mask = np.zeros(orig_rad.shape, dtype=bool)

    n_overwritten_px = 0
    for i, frame_idx in enumerate(range(f0, f1)):
        if frame_idx not in cam_rad_by_frame:
            continue
        cam_rad_native = cam_rad_by_frame[frame_idx].T   # undo _fpa()'s .T
        sel = np.isfinite(cam_rad_native)
        if not sel.any():
            continue
        converted = (cam_rad_native[sel] * MODIS_RADIANCE_UM_TO_CAMERA_NM
                   ).astype(new_rad.dtype)
        new_rad[i][sel] = converted
        proxy_mask[i][sel] = True
        n_overwritten_px += int(sel.sum())

    ds_out["Radiance"] = ds_out["Radiance"].copy(data=new_rad)
    ds_out["Radiance"].attrs["l1x_proxy_note"] = (
        "Pixels where L1X_Proxy_Mask==1 are MODIS/VIIRS-proxy-derived "
        "(l2/build_l1x.py, components=%s); every other pixel is the "
        "ORIGINAL LIBERA L1B value, unmodified (dark/pre-launch "
        "placeholder, not a realistic radiance)." % ",".join(components))

    ds_out["L1X_Proxy_Mask"] = ds_out["Radiance"].copy(data=proxy_mask.astype(np.uint8))
    ds_out["L1X_Proxy_Mask"].attrs = dict(
        long_name="L1X MODIS/VIIRS proxy-radiance pixel flag",
        description=(
            "1 where this pixel's Radiance was overwritten from the "
            "MODIS/VIIRS proxy projection (real camera pixel-mask "
            "component(s) %s, restricted to pixels the downloaded proxy "
            "scene actually covers); 0 everywhere else, including every "
            "pixel of any exposure in this file whose selected-"
            "component stripe never reached the tile -- those keep the "
            "original LIBERA L1B value untouched." % ",".join(components)),
    )

    if verbose:
        print("  %d exposure(s) in this L1X crop, %d with >=1 overwritten "
             "pixel, %d pixel(s) overwritten in total"
             % (f1 - f0, len(cam_rad_by_frame), n_overwritten_px))
        if n_overwritten_px:
            ov = new_rad[proxy_mask]
            orig_ov = orig_rad[proxy_mask]
            print("  overwritten pixels: new Radiance min=%.6f mean=%.6f "
                 "max=%.6f [W m-2 sr-1 nm-1]  (original L1B value there: "
                 "min=%.6f mean=%.6f max=%.6f)"
                 % (float(ov.min()), float(ov.mean()), float(ov.max()),
                    float(orig_ov.min()), float(orig_ov.mean()), float(orig_ov.max())))

    return ds_out, proxy_mask, n_overwritten_px


def build_l1x(cam_l1b, rad_l1b, lat0, lon0, window_s, scan_pad_s,
             tile_bbox_deg, src_lon, src_lat, src_rad,
             components=DEFAULT_COMPONENTS, table_dir=None, verbose=True):
    """Run proxy_radiance.project_tile_frames() (unchanged) to get the
    per-frame projected radiance, then splice it into a cropped copy of
    the real CAM L1B (splice_into_l1b_crop()). Returns ``(ds_out,
    cam_rad_by_frame, match, combined)`` -- the last three exactly as
    project_tile_frames() itself returns them."""
    cam_rad_by_frame, match, combined = pr.project_tile_frames(
        cam_l1b, rad_l1b, lat0, lon0, window_s, scan_pad_s, tile_bbox_deg,
        src_lon, src_lat, src_rad, components=components, table_dir=table_dir,
        verbose=verbose)

    # Crop to the SPAN BRACKETING the kept frames (min..max+1), not the
    # full +/-window_s chop() search window: project_tile_frames() often
    # has to search a wide time window (deliberately -- the window's job
    # is just to make sure it doesn't miss anything) to find the handful
    # of exposures whose stripe actually reaches a 100 km tile, and most
    # of that window has nothing to do with this tile at all. Sebastian,
    # 2026-10-23: L1X's exposures should be "all exposures in that tile"
    # (the ~10-20 that matter, with the occasional one left untouched
    # in between because its stripe missed this time), not every exposure
    # chop() merely considered.
    f0 = min(cam_rad_by_frame)
    f1 = max(cam_rad_by_frame) + 1
    if verbose:
        print("\nBuilding L1X: cropping %s to camera frames [%d, %d) -- "
             "the span bracketing the %d exposure(s) whose stripe actually "
             "reached the tile (not the full +/-%.0fs chop() search window)"
             % (os.path.basename(cam_l1b), f0, f1, len(cam_rad_by_frame),
                window_s))
    ds = est.open_granule(cam_l1b)
    ds_out, proxy_mask, n_overwritten_px = splice_into_l1b_crop(
        ds, f0, f1 - f0, cam_rad_by_frame, components, verbose=verbose)

    ds_out.attrs["l1x_tile_lat"] = float(lat0)
    ds_out.attrs["l1x_tile_lon"] = float(lon0)
    ds_out.attrs["l1x_components"] = ",".join(components)
    ds_out.attrs["l1x_source_cam_l1b"] = os.path.basename(cam_l1b)
    ds_out.attrs["l1x_radiance_unit_conversion"] = (
        "MODIS/VIIRS source radiance (W m-2 sr-1 um-1) * %g -> camera "
        "Radiance unit (W m-2 sr-1 nm-1)" % MODIS_RADIANCE_UM_TO_CAMERA_NM)
    ds_out.attrs["l1x_generated_utc"] = (
        datetime.datetime.now(datetime.timezone.utc).isoformat())

    return ds_out, cam_rad_by_frame, match, combined


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-pick", default=ips.DEFAULT_TILE_PICK)
    p.add_argument("--cam-l1b", default=pr.DEFAULT_CAM_L1B)
    p.add_argument("--rad-l1b", default=pr.DEFAULT_RAD_L1B)
    p.add_argument("--window-s", type=float, default=pr.DEFAULT_WINDOW_S)
    p.add_argument("--scan-pad-s", type=float, default=pr.DEFAULT_SCAN_PAD_S)
    p.add_argument("--modis-geo", default=pr.DEFAULT_MODIS_GEO)
    p.add_argument("--modis-l1b", default=pr.DEFAULT_MODIS_L1B)
    p.add_argument("--tile-km", type=float, default=ips.DEFAULT_TILE_KM)
    p.add_argument("--band", type=int, default=ips.MODIS_BAND)
    p.add_argument("--components", default=",".join(DEFAULT_COMPONENTS),
                  help="comma-separated pixel-mask component(s) to "
                       "overwrite from the proxy scene (default: "
                       "%(default)s -- everything else keeps its "
                       "original L1B value)")
    p.add_argument("--table-dir", default=None)
    p.add_argument("--out", default=None,
                  help="output L1X path (default: derived from --cam-l1b, "
                       "under %s)" % DEFAULT_OUT_DIR)
    p.add_argument("--plot", action="store_true",
                  help="show ONLY the pixels this run actually overwrote "
                       "(L1X_Proxy_Mask==1), not the rest of the frame -- "
                       "L2 won't use the rest either")
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
    src_lon, src_lat, src_rad = pr.tighten_source_crop(
        modis, lon_min, lon_max, lat_min, lat_max)

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())

    ds_out, cam_rad_by_frame, match, combined = build_l1x(
        args.cam_l1b, args.rad_l1b, lat0, lon0, args.window_s, args.scan_pad_s,
        (lon_min, lon_max, lat_min, lat_max), src_lon, src_lat, src_rad,
        components=components, table_dir=args.table_dir)
    ds_out.attrs["l1x_tile_pick_path"] = os.path.abspath(args.tile_pick)
    ds_out.attrs["l1x_modis_source_sza_mean_deg"] = (
        float(modis["sza_mean"]) if modis.get("sza_mean") is not None
        else float("nan"))

    out_path = args.out or l1x_path_for(args.cam_l1b)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print("\nWrote L1X: %s" % out_path)

    if args.plot:
        import matplotlib.pyplot as plt
        # Both panels share one color scale, in camera-native units, so the
        # comparison is meaningful on color as well as on shape: the MODIS
        # source crop is converted here for DISPLAY ONLY (the same factor
        # build_l1x() applies when writing the L1X file) -- src_rad itself,
        # as returned by tighten_source_crop(), stays in MODIS's native
        # W m-2 sr-1 um-1 everywhere else in this script.
        all_rad_nm = combined["rad"] * MODIS_RADIANCE_UM_TO_CAMERA_NM
        src_rad_nm = src_rad * MODIS_RADIANCE_UM_TO_CAMERA_NM
        vmin = min(float(np.nanmin(src_rad_nm)), float(np.nanmin(all_rad_nm)))
        vmax = max(float(np.nanmax(src_rad_nm)), float(np.nanmax(all_rad_nm)))

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        sc0 = axes[0].scatter(src_lon, src_lat, c=src_rad_nm, s=3,
                              cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].plot(lon0, lat0, "r+", markersize=14, markeredgewidth=2,
                    label="tile center")
        axes[0].set_xlabel("Longitude [deg]")
        axes[0].set_ylabel("Latitude [deg]")
        axes[0].set_title("MODIS source crop (tightened to tile box)")
        axes[0].legend(loc="upper right")
        plt.colorbar(sc0, ax=axes[0],
                    label="Radiance [W m-2 sr-1 nm-1]\n(converted from MODIS native um-1)")

        sc1 = axes[1].scatter(combined["lon"], combined["lat"], c=all_rad_nm,
                              s=4, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].plot(lon0, lat0, "r+", markersize=14, markeredgewidth=2,
                    label="tile center")
        axes[1].set_xlabel("Longitude [deg]")
        axes[1].set_ylabel("Latitude [deg]")
        axes[1].set_title("L1X overwritten pixels only (%s), %d frame(s)"
                          % (",".join(components), len(cam_rad_by_frame)))
        axes[1].legend(loc="upper right")
        plt.colorbar(sc1, ax=axes[1], label="Radiance [W m-2 sr-1 nm-1]")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
