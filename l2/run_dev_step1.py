#!/usr/bin/env python3
"""Dev-step-1 CF-CAM driver (Sebastian, 2026-10-22).

The first real end-to-end run of the plan agreed 2026-10-20: read the
three manifest files (camera L1B, radiometer L1B, ERA5), match the
hand-picked tile (l2/tile_pick.yml, from pick_tile_overview.py) to a
camera-frame window, run the existing validated chop()/tile()/rt_shdom()
chain (run_tile_rt.py) on ONE 100 km tile, and report the dev-step-1
`cam_radiance_footprint` (footprint-averaged clear-sky RT at 555 nm,
standing in for real camera-measured radiance until VIIRS-proxy imagery
is available -- dev-step-2) plus the other CF-CAM fields we can already
populate for real: n_camera_pixels_total/valid and the tile-mean
VZA/VAA/SZA/SAA geometry pass-through. `cam_cloud_fraction` is 0 by
design at this stage (per Sebastian's 2026-10-20 plan).

Deliberately does NOT yet write the final spec-compliant NetCDF or an
output manifest -- those need `cfg_L2_CF_CAM_dev.yml` (the static output
schema), which doesn't exist yet (see docs/L1B_TO_L2_ROADMAP.md). This
script writes a plain yml summary instead, specifically so the actual
numbers can be sanity-checked BEFORE committing to that schema -- building
the schema around a real computed example, not blind.

Uses the input manifest the way the project's plan actually calls for:
reads it, classifies each listed file by role (camera L1B / radiometer
L1B / ERA5) from its filename, and fails loudly if that classification is
ambiguous -- "specified and validated, not just opened and ignored."

Also exercises run_tile_rt.tile()'s new era5_path= override (added
2026-10-22 in this same session) so the ocean BRDF actually uses this
orbit's real ERA5 windspeed, not the old hardcoded MERRA2 file in l2_cam's
sample/input.yml.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import yaml

from libera_utils import Manifest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import run_tile_rt  # noqa: E402

# Default: this repo's own root (one level up from l2/), so a fresh
# clone works out of the box with no manual --l2-cam-root. Override it
# only if you've relocated cam_l2/software/er3t/software/shdom elsewhere.
DEFAULT_L2_CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST_DIR = "l2/data"
DEFAULT_TILE_PICK = "tile_pick.yml"
DEFAULT_OUT = "l2/data/dev_step1_result.yml"


def latest_manifest(manifest_dir):
    candidates = sorted(glob.glob(os.path.join(manifest_dir, "LIBERA_INPUT_MANIFEST_*.json")))
    if not candidates:
        raise FileNotFoundError(
            "No LIBERA_INPUT_MANIFEST_*.json found under %s -- run "
            "build_input_manifest.py first." % manifest_dir)
    return candidates[-1]


def classify_manifest_files(manifest):
    """Map the manifest's (unordered) file list to roles by filename --
    fails loudly on anything ambiguous or missing, rather than guessing by
    position. This IS the "specified and validated" input handling the
    2026-10-20 plan called for, as opposed to l2_cam/main_cam_l2.py's
    manifest, which is loaded and then never actually consulted."""
    roles = {}
    for record in manifest.files:
        name = os.path.basename(record.filename)
        if "_CAM_" in name:
            key = "cam_l1b"
        elif "_RAD-4CH_" in name:
            key = "rad_l1b"
        elif name.startswith("VJ143") or name.startswith("VNP43"):
            key = "brdf_land"
        elif "ERA5" in record.filename.upper() or (name.endswith(".nc") and "_L1B_" not in name):
            key = "era5"
        else:
            raise ValueError("Could not classify manifest file by role: %s "
                             "(expected a camera L1B, a RAD-4CH L1B, or an "
                             "ERA5 file)" % record.filename)
        if key in roles:
            raise ValueError("Manifest lists two files that both look like "
                             "the '%s' role: %s and %s -- ambiguous, fix the "
                             "manifest." % (key, roles[key], record.filename))
        roles[key] = record.filename
    missing = {"cam_l1b", "rad_l1b", "era5"} - set(roles)
    if missing:
        raise ValueError("Manifest is missing required role(s): %s (found: %s)"
                         % (sorted(missing), list(roles)))
    return roles


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-manifest", default=None,
                   help="defaults to the most recently written "
                        "LIBERA_INPUT_MANIFEST_*.json under --manifest-dir")
    p.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    p.add_argument("--tile-pick", default=DEFAULT_TILE_PICK,
                   help="yml written by pick_tile_overview.py (default: "
                        "tile_pick.yml at the repo root)")
    p.add_argument("--l2-cam-root", default=DEFAULT_L2_CAM_ROOT,
                   help="Hong's l2_cam repo -- contains software/shdom, "
                        "software/er3t, data/ (land/ocean mask, VIIRS BRDF)")
    p.add_argument("--tile-km", type=float, default=100.0,
                   help="production tile size, per Sebastian's 2026-10-22 "
                        "decision (prior diagnostic work used 500 km)")
    p.add_argument("--dx-km", type=float, default=5.5)
    p.add_argument("--dy-km", type=float, default=5.5)
    p.add_argument("--max-vza-deg", type=float, default=10.0,
                   help="tile()'s own default, tuned for a 100 km tile -- "
                        "the 500 km diagnostic runs used 60 deg instead")
    p.add_argument("--window-s", type=float, default=300.0)
    p.add_argument("--scan-pad-s", type=float, default=500.0)
    p.add_argument("--wavelength-nm", type=float, default=555.0,
                   help="CF-CAM's cam_radiance_footprint wavelength")
    p.add_argument("--altitude-km", type=float, default=831.16)
    p.add_argument("--niter", type=int, default=10)
    p.add_argument("--ncpu", type=int, default=4)
    p.add_argument("--solver", default="3D", choices=["3D", "IPA"],
                   help="rt_shdom()'s SHDOM solver mode, forwarded to "
                        "shdom_ng(): '3D' (default, IPFLAG=0) solves the "
                        "full 3D domain with horizontal photon transport "
                        "between columns; 'IPA' (independent pixel "
                        "approximation, IPFLAG=3) solves each column "
                        "independently, plane-parallel -- i.e. 1D RT, no "
                        "horizontal transport at all. Requested 2026-10-21 "
                        "(Sebastian) to test whether an apparent land-"
                        "radiance dip (full atm < near-vacuum atm over "
                        "bright land) is a 3D adjacency effect: under IPA "
                        "there is no horizontal transport to wrap around "
                        "periodic boundaries in the first place, so this "
                        "also serves as an IPA-mode control for the whole "
                        "periodic-BC/margin_cells issue above. Applies to "
                        "BOTH the main RT round and --thin-atm-diag's "
                        "extra round, so the two stay on the same solver "
                        "for an apples-to-apples comparison.")
    p.add_argument("--allow-ancillary-date-mismatch", action="store_true",
                   help="l2_cam's land BRDF (VJ143C1) and ocean windspeed "
                        "(MERRA2) are fixed sample files from one baked-in "
                        "date, used for every run regardless of the actual "
                        "overpass/proxy date, unless overridden with real "
                        "data (ERA5 for wind; there is no land-BRDF "
                        "override source yet) -- every run hits SOME "
                        "mismatch against that fixed sample, so most are "
                        "printed as a WARNING only and do not block the "
                        "run. tile() refuses to run (FATAL) only for the "
                        "categorically-impossible case -- an overpass/proxy "
                        "date before VIIRS/JPSS-1 even launched, so no real "
                        "match could ever exist -- pass this to proceed "
                        "anyway and accept untrustworthy land-surface "
                        "reflectance even then.")
    p.add_argument("--margin-cells", type=int, default=2,
                   help="pad the RT/BRDF domain this many extra VJ143C1-"
                        "native grid cells beyond --tile-km on every side, "
                        "then discard that margin from the result before "
                        "using/plotting/saving it (crop_tile_margin()). "
                        "Workaround for the SHDOM periodic-BC artifact that "
                        "shows up when running multi-process (see "
                        "docs/L1B_TO_L2_ROADMAP.md 'Hong input' section) -- "
                        "default changed to 2 on 2026-10-22 (Sebastian: "
                        "verified against real data, 'does away with the "
                        "strange wrapping/uniform-radiance artifact') so "
                        "nobody gets the un-padded/artifact-prone tile by "
                        "omission; pass --margin-cells 0 explicitly to go "
                        "back to the regular/un-padded tile.")
    p.add_argument("--thin-atm-diag", action="store_true",
                   help="run an EXTRA rt_shdom() round with a near-vacuum "
                        "atm_levels override (see --thin-atm-top-km/"
                        "--thin-atm-n-levels), to isolate what the real "
                        "atmosphere itself contributes vs. periodic-BC/"
                        "solver/geometry effects already present in "
                        "refl_app vs refl_noatm. Requested 2026-10-21 "
                        "(Sebastian, for the Hong email). Off by default "
                        "-- this is an extra slow RT solve, costing "
                        "roughly as much wall time again as the main run.")
    p.add_argument("--thin-atm-top-km", type=float, default=1.0,
                   help="top of the near-vacuum atm_levels range for "
                        "--thin-atm-diag (matches "
                        "rt_shdom_vacuum_test.py's own default)")
    p.add_argument("--thin-atm-n-levels", type=int, default=5,
                   help="number of atm_levels points for --thin-atm-diag. "
                        "Must stay >=2 (non-degenerate) -- a single near-"
                        "zero-depth level previously crashed SHDOM's MPI "
                        "domain decomposition; this default of 5 avoids "
                        "that crash while still driving Rayleigh/gas "
                        "optical depth toward ~0.")
    p.add_argument("--write-l2-output", action="store_true",
                   help="Also write a minimal CF-CAM L2 NetCDF + output "
                        "manifest (Sebastian, 2026-10-22: 'good enough for "
                        "SDC to test end to end... even if the single-tile "
                        "code doesn't go all the way'). Scope: 'inside' "
                        "radiometer footprints only (no multi-tile "
                        "stitching). cam_cloud_fraction is the real "
                        "e-fold-threshold per-footprint cloud fraction and "
                        "cam_radiance_footprint is the real L1X measured-"
                        "radiance footprint average by default (item C, "
                        "2026-10-23) -- see --cam-cloud-fraction to force "
                        "the old zero/random placeholder instead. See "
                        "l2/write_cf_cam_output.py.")
    p.add_argument("--l2-output-dir", default="output_l2",
                   help="Directory for --write-l2-output's NetCDF + output "
                        "manifest (default: %(default)s).")
    p.add_argument("--cam-cloud-fraction", default="real",
                   choices=["real", "zero", "random"],
                   help="--write-l2-output's cam_cloud_fraction source "
                        "(item C, 2026-10-23): 'real' (default) uses the "
                        "e-fold cloud-threshold fraction already computed "
                        "per footprint (run_tile_rt.radiance_and_counts_"
                        "for_footprints()'s cam_cloud_fraction_rt -- the "
                        "same number the QA plots' green-footprint labels "
                        "show), _FillValue where cloud detection didn't "
                        "run. 'zero' or 'random' (Sebastian, 2026-10-22) "
                        "are the original debug/testing placeholders -- a "
                        "hardcoded 0.0, or a uniform [0,1) draw per "
                        "footprint (a FRACTION per the CF-CAM spec's own "
                        "valid_range=[0,1], not 0-100) -- kept for "
                        "exercising downstream code paths without a real "
                        "cloud-flag array.")
    p.add_argument("--pad-to-orbit", action="store_true",
                   help="item D, 2026-10-23, optional (Sebastian's plan): "
                        "--write-l2-output's NetCDF is lengthened from "
                        "just this tile's ~37 'inside' footprints to the "
                        "FULL RAD-4CH orbit's record count (roles['rad_l1b'], "
                        "the same radiometer L1B already in the input "
                        "manifest) -- real Latitude/Longitude/Altitude/"
                        "SZA/VZA/RAA/VAA/SAA/radiometer_time passed "
                        "through for every footprint in the orbit, with "
                        "cam_radiance_footprint/cam_cloud_fraction/"
                        "psf_coverage_fraction real only at this tile's "
                        "own footprints and _FillValue elsewhere (no "
                        "multi-tile stitching yet -- see "
                        "write_cf_cam_output.build_cf_cam_arrays_padded()). "
                        "Default off: the original ~37-row-only file.")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--plot", action="store_true",
                   help="write a 4-panel QA PNG next to --out (footprint-"
                        "feeding map, RT radiance field, ERA5 windspeed "
                        "field, geometry sanity panel) -- added 2026-10-22 "
                        "per Sebastian's request to sanity-check runs visually")
    p.add_argument("--plot-out", default=None,
                   help="defaults to --out with .yml replaced by _qa.png")
    args = p.parse_args()

    manifest_path = args.input_manifest or latest_manifest(args.manifest_dir)
    print("Reading input manifest: %s" % manifest_path)
    manifest = Manifest.from_file(manifest_path)
    roles = classify_manifest_files(manifest)
    for role, path in roles.items():
        if not os.path.exists(path):
            raise FileNotFoundError("Manifest lists %s but it does not exist "
                                    "on disk: %s" % (role, path))
        print("  %-10s %s" % (role + ":", path))

    with open(args.tile_pick) as f:
        tile_pick = yaml.safe_load(f)["tile_pick"]
    seed_lat, seed_lon = tile_pick["lat"], tile_pick["lon"]
    print("\nTile pick (from %s): lat=%.4f lon=%.4f surface=%s"
          % (args.tile_pick, seed_lat, seed_lon, tile_pick.get("surface_type", "?")))

    t0 = time.time()
    print("\n=== chop(): matching the pick to a camera-frame window ===")
    match = run_tile_rt.chop(roles["rad_l1b"], roles["cam_l1b"], seed_lat, seed_lon,
                             args.window_s, scan_pad_s=args.scan_pad_s)

    print("\n=== footprints_feeding_tile(): which radiometer footprints does "
          "this tile populate? ===")
    footprint_info = run_tile_rt.footprints_feeding_tile(
        roles["rad_l1b"], seed_lat, seed_lon, args.tile_km, match["t_ns"],
        args.window_s, args.altitude_km)

    print("\n=== tile(): pooling camera pixels, cropping ancillary data "
          "(ERA5-overridden ocean windspeed) ===")
    data, pooled = run_tile_rt.tile(
        roles["cam_l1b"], match, args.tile_km, args.dx_km, args.dy_km,
        args.l2_cam_root, max_vza_deg=args.max_vza_deg,
        era5_path=roles["era5"], margin_cells=args.margin_cells,
        brdf_land_path=roles.get("brdf_land"),
        allow_ancillary_date_mismatch=args.allow_ancillary_date_mismatch)
    if args.margin_cells > 0:
        print("  margin_cells=%d: RT/BRDF domain padded %d extra "
              "VJ143C1-native grid cell(s) beyond --tile-km on every side "
              "(periodic-BC workaround -- see crop_tile_margin() below, "
              "called right after rt_shdom()/rt_noatm())"
              % (args.margin_cells, args.margin_cells))

    print("\n=== footprint_fill_fraction(): exposure-gap / fill-fraction "
          "diagnostic per footprint (cheap -- real-pixel nearest-neighbor "
          "geometry only, no PSF convolution) ===")
    footprint_info = run_tile_rt.footprint_fill_fraction(pooled, footprint_info)
    if footprint_info["fill_fraction_median"] is not None:
        print("  fill_fraction across %d footprint(s) with enough nearby "
              "data: median=%.3f, min=%.3f"
              % (sum(1 for r in footprint_info["records"]
                     if r["fill_fraction"] is not None),
                 footprint_info["fill_fraction_median"],
                 footprint_info["fill_fraction_min"]))
    else:
        print("  not enough nearby camera pixels to estimate fill_fraction "
              "for any footprint in this tile")

    print("\n=== rt_noatm(): no-atmosphere BRDF-only surface reflectance "
          "(cheap -- pure kernel algebra, for comparison against the SHDOM "
          "TOA output below) ===")
    refl_noatm, _refl_noatm_flag = run_tile_rt.rt_noatm(data)

    print("\n=== rt_shdom(): clear-sky RT, dev-step-1 camera-radiance "
          "proxy (this is the slow step) ===")
    rad, refl_app = run_tile_rt.rt_shdom(
        data, args.l2_cam_root, args.wavelength_nm, args.altitude_km,
        Niter=args.niter, Ncpu=args.ncpu, solver=args.solver)

    # --thin-atm-diag (2026-10-21, Sebastian: "an additional diagnostic
    # plot where the atmosphere is almost removed"): an EXTRA rt_shdom()
    # round on the same grid/geometry/BRDF/BC code path as the main run
    # above, but with a thin `atm_levels` override (0..thin_atm_top_km,
    # thin_atm_n_levels points) so Rayleigh scattering + gas absorption
    # optical depth both go toward ~0 -- a near-vacuum atmosphere. This is
    # NOT the same thing as refl_noatm (rt_noatm(), above): refl_noatm is
    # cheap pure BRDF-kernel algebra at one fixed geometry and never goes
    # through the SHDOM solve/periodic-BC machinery at all, whereas this
    # extra round goes through the exact same SHDOM solve as refl_app, so
    # comparing refl_app vs. this isolates what the REAL atmosphere itself
    # contributes (vs. periodic-BC/solver/geometry effects already present
    # with near-zero atmosphere).
    #
    # IMPORTANT: this reuses Ncpu=args.ncpu (same as the main run above),
    # NOT Ncpu=1. The pre-existing l2/rt_shdom_vacuum_test.py script
    # recommends Ncpu=1 for this kind of thin/degenerate-atmosphere test,
    # because a single near-zero-depth atm_levels point previously crashed
    # SHDOM's MPI domain decomposition under Ncpu=4. But this session's own
    # debugging (see docs/L1B_TO_L2_ROADMAP.md 'Hong input'/KNOWN ISSUE)
    # found that Ncpu=1 independently and completely breaks CAMID=3/
    # camera3 output (nan + -1.0 sentinel), for an unrelated reason in
    # SHDOM's serial computeRadRay path. So instead of Ncpu=1, this avoids
    # the MPI crash by using a NON-degenerate thin_atm_n_levels>=2 range
    # (rt_shdom_vacuum_test.py's own default: 5 levels spanning 0..1 km,
    # not one single point) together with Ncpu=args.ncpu.
    #
    # Runs into its own out_root so it never clobbers the main run's SHDOM
    # working files above.
    rad_thin = refl_thin = None
    if args.thin_atm_diag:
        print("\n=== rt_shdom() [--thin-atm-diag]: extra RT round, near-"
              "vacuum atmosphere (0..%.3g km, %d levels), Ncpu=%d (NOT 1 -- "
              "see comment above) ==="
              % (args.thin_atm_top_km, args.thin_atm_n_levels, args.ncpu))
        thin_atm_levels = np.linspace(0.0, args.thin_atm_top_km,
                                      args.thin_atm_n_levels)
        rad_thin, refl_thin = run_tile_rt.rt_shdom(
            data, args.l2_cam_root, args.wavelength_nm, args.altitude_km,
            Niter=args.niter, Ncpu=args.ncpu, solver=args.solver,
            out_root="tmp-data-thin-atm", atm_levels=thin_atm_levels)

    # crop_tile_margin() (2026-10-22): if margin_cells>0, `data`/`rad`/
    # `refl_app`/`refl_noatm`/(`rad_thin`/`refl_thin`) above were all
    # computed on the PADDED domain -- rt_shdom()/rt_noatm() themselves
    # know nothing about padding, they just solved whatever grid tile()
    # handed them. Crop back down to the real --tile-km tile now, before
    # anything below (footprints, diagnostics, YAML, QA plot) sees these
    # arrays, so the padding is fully invisible past this point. A no-op
    # when margin_cells=0 (the default).
    _extra = (rad, refl_app, refl_noatm)
    if args.thin_atm_diag:
        _extra = _extra + (rad_thin, refl_thin)
    data, _extra = run_tile_rt.crop_tile_margin(data, args.margin_cells,
                                                extra_arrays=_extra)
    if args.thin_atm_diag:
        rad, refl_app, refl_noatm, rad_thin, refl_thin = _extra
    else:
        rad, refl_app, refl_noatm = _extra

    print("\n=== bin_measured_radiance_to_grid(): binning L1X 'measured' "
          "camera radiance onto the same RT grid, for a cell-by-cell "
          "comparison against rad (dev-step-2, per Sebastian's "
          "2026-10-22 request to also show the measured radiances, a "
          "difference map, and an RT-vs-measured scatter) ===")
    rad_meas, n_px_per_cell = run_tile_rt.bin_measured_radiance_to_grid(data, pooled)
    sza_cam = float(data["tile_mean_angles"]["sza"])
    sza_src = None
    try:
        import netCDF4 as _netCDF4
        with _netCDF4.Dataset(roles["cam_l1b"], "r") as _src_ds:
            _sza_src_raw = float(getattr(
                _src_ds, "l1x_modis_source_sza_mean_deg", float("nan")))
        if np.isfinite(_sza_src_raw):
            sza_src = _sza_src_raw
    except Exception as _sza_exc:
        print("  (could not read l1x_modis_source_sza_mean_deg from %s: %r)"
             % (roles["cam_l1b"], _sza_exc))
    print("  SZA (tile-mean): camera=%.2f deg, MODIS source=%s"
         % (sza_cam, ("%.2f deg" % sza_src) if sza_src is not None
                    else "n/a (not an L1X file, or built before this "
                         "attribute existed -- rerun build_l1x.py)"))
    refl_meas = None
    if rad_meas is not None and sza_src is not None:
        refl_meas = np.pi * rad_meas / (
            run_tile_rt.F0_555NM * np.cos(np.radians(sza_src)))
    measured_vs_rt = None
    if rad_meas is not None:
        _both_finite = np.isfinite(rad) & np.isfinite(rad_meas)
        _n_both = int(_both_finite.sum())
        if _n_both >= 2:
            _rt_f, _meas_f = rad[_both_finite], rad_meas[_both_finite]
            _diff = _meas_f - _rt_f
            measured_vs_rt = {
                "note": "RT (rad, SHDOM clear-sky TOA) vs. 'measured' "
                        "(rad_meas, binned from the L1X file's real camera "
                        "Radiance field, nadir-stripe pixels the MODIS/"
                        "VIIRS proxy scene actually covers -- "
                        "L1X_Proxy_Mask==1) -- see "
                        "run_tile_rt.bin_measured_radiance_to_grid(). Both "
                        "in the SAME native units (W m^-2 sr^-1 nm^-1): "
                        "the camera-nm unit conversion already happened "
                        "once, at l2/build_l1x.py write time, so no "
                        "further conversion is needed here. A real-scene "
                        "comparison, not yet a validated algorithm metric "
                        "-- rad is clear-SKY only (cam_cloud_fraction=0 "
                        "placeholder), so any real cloud in the L1X scene "
                        "will show up here as RT-vs-measured disagreement "
                        "by design, not as a bug.",
                "n_rt_grid_cells_total": int(rad.size),
                "n_rt_grid_cells_with_measurement": int(np.isfinite(rad_meas).sum()),
                "n_rt_grid_cells_compared": _n_both,
                "sza_camera_deg": round(sza_cam, 4),
                "sza_modis_source_deg": (round(sza_src, 4)
                    if sza_src is not None else None),
                "cam_radiance_footprint_measured": float(np.nanmean(rad_meas)),
                "cam_radiance_footprint_measured_units":
                    "W m^-2 sr^-1 nm^-1 (same native units as "
                    "cam_radiance_footprint)",
                "mean_bias_measured_minus_rt": round(float(np.mean(_diff)), 6),
                "rmse_rt_vs_measured": round(float(np.sqrt(np.mean(_diff ** 2))), 6),
                "corr_rt_vs_measured": (
                    round(float(np.corrcoef(_rt_f, _meas_f)[0, 1]), 6)
                    if np.std(_rt_f) > 0 and np.std(_meas_f) > 0 else None),
            }
            print("  measured-vs-RT: %d/%d grid cells compared, bias(measured-"
                 "RT)=%.4g, RMSE=%.4g, corr=%s [W m^-2 sr^-1 nm^-1]"
                 % (_n_both, rad.size, measured_vs_rt["mean_bias_measured_minus_rt"],
                    measured_vs_rt["rmse_rt_vs_measured"],
                    measured_vs_rt["corr_rt_vs_measured"]))
        else:
            print("  measured-vs-RT: fewer than 2 grid cells have both RT "
                 "and measured values -- skipping comparison stats")
    # Adaptive e-fold cloud threshold -- Sebastian, 2026-10-23: "add the
    # Hong-approach directly to that histogram ... replicate something
    # similar (rather than simply importing)." Independently reimplemented
    # from the Libera ATBD Camera L2 CDR draft, Figure 6 (p.8) and Hong's
    # l2_cam/src/cam_l2/cld.py cal_threshold_efold (same recipe, confirmed
    # to match) -- see run_tile_rt.cloud_threshold_efold()'s own docstring
    # for the algorithm. Uses whichever quantity the QA histogram panel
    # itself uses: reflectance when both refl_app/refl_meas are available
    # (SZA-matched on both sides, so this is the ATBD's intended
    # quantity), falling back to radiance otherwise. cloud_info keeps the
    # full dict (incl. the bin_edges/hist arrays, so the QA plot can
    # redraw the exact histogram the threshold came from); only a
    # YAML-safe scalar subset goes into measured_vs_rt below.
    cloud_info = None
    cloud_flag_grid = None
    if rad_meas is not None:
        _have_refl_cmp = refl_app is not None and refl_meas is not None
        _cd_rt = refl_app if _have_refl_cmp else rad
        _cd_meas = refl_meas if _have_refl_cmp else rad_meas
        _cd_quantity = "reflectance" if _have_refl_cmp else "radiance"
        _cd_rt_f = np.asarray(_cd_rt, dtype=float).ravel()
        _cd_meas_f = np.asarray(_cd_meas, dtype=float).ravel()
        _cd_finite = np.isfinite(_cd_rt_f) & np.isfinite(_cd_meas_f)
        _cd_diff = _cd_meas_f[_cd_finite] - _cd_rt_f[_cd_finite]
        if _cd_diff.size >= 10:
            cloud_info = run_tile_rt.cloud_threshold_efold(_cd_diff)
            if cloud_info is not None:
                _cloud_flag = _cd_diff > cloud_info["threshold"]
                cloud_info["quantity_name"] = _cd_quantity
                cloud_info["cloud_fraction"] = float(np.mean(_cloud_flag))
                cloud_info["n_cells_cloud"] = int(np.sum(_cloud_flag))
                cloud_info["n_cells_compared"] = int(_cd_diff.size)
                print("  cloud detection (ATBD Fig. 6 e-fold threshold, on "
                     "%s diff): x_max=%.4g, f_max=%.4g, x_e=%.4g, "
                     "threshold=%.4g -- cloud_fraction=%.3f (%d/%d grid "
                     "cells flagged)"
                     % (cloud_info["quantity_name"], cloud_info["x_max"],
                        cloud_info["f_max"], cloud_info["x_e"],
                        cloud_info["threshold"], cloud_info["cloud_fraction"],
                        cloud_info["n_cells_cloud"],
                        cloud_info["n_cells_compared"]))
                # Full-grid (nx, ny) boolean cloud mask -- added 2026-10-23
                # for the QA-plot visualization follow-ups (Sebastian:
                # color the scatter, mark the diff map, highlight the QA
                # figure, label footprint CF numbers). _cloud_flag above
                # is only defined over the COMPRESSED _cd_finite subset;
                # this scatters it back to the full (nx, ny) grid
                # (False/clear everywhere a cell wasn't even compared --
                # same "absent, not clear, not cloudy" convention as
                # everywhere else this threshold is used, just encoded as
                # False here since the plotting/footprint code below only
                # asks "is this cell flagged cloudy", not "was it
                # compared at all").
                cloud_flag_grid = np.zeros(_cd_rt_f.shape, dtype=bool)
                cloud_flag_grid[_cd_finite] = _cloud_flag
                cloud_flag_grid = cloud_flag_grid.reshape(np.asarray(_cd_rt).shape)
        else:
            print("  cloud detection: fewer than 10 grid cells with both RT "
                 "and measured %s -- skipping threshold/cloud-fraction calc"
                 % _cd_quantity)
    if measured_vs_rt is not None and cloud_info is not None:
        measured_vs_rt["cloud_detection"] = {
            k: v for k, v in cloud_info.items()
            if k not in ("bin_edges", "hist")
        }

    # Populates each 'inside'/'edge' record in footprint_info with
    # cam_radiance_footprint_rt / cam_radiance_footprint_n_cells /
    # n_camera_pixels_total / n_camera_pixels_valid / cam_cloud_fraction_rt
    # (run_tile_rt.py, 2026-10-22, cam_cloud_fraction_rt added 2026-10-23)
    # from the final cropped data/rad -- the same arrays the tile-mean
    # cam_radiance_footprint below and the QA plot use. MOVED 2026-10-23
    # from inside `if args.write_l2_output:` to unconditional: the QA
    # plot's footprint-CF-number labels (Sebastian, 2026-10-23) need this
    # populated whenever --plot is used too, not only --write-l2-output.
    footprint_info = run_tile_rt.radiance_and_counts_for_footprints(
        data, rad, pooled, footprint_info, cloud_flag=cloud_flag_grid)

    tm = data["tile_mean_angles"]
    n_total = int(pooled["in_box"].sum())
    n_valid = int(pooled["in_tile"].sum())
    cam_radiance_footprint = float(np.nanmean(rad))

    # Which array index (0 or -1) is actually geographic south vs. north --
    # determined from the real lat_land values, NOT assumed from array
    # index order (2026-10-22, same orientation-assumption bug Sebastian's
    # "coastline runs slightly differently" question surfaced elsewhere --
    # see plot_dev_step1_qa.py's _panel_radiance orientation fix). Avoids
    # silently mislabeling "south"/"north" if Hong's VJ143C1 grid turns out
    # to be stored north-to-south internally.
    _lat_col0 = float(np.nanmean(data["lat_land"][:, 0]))
    _lat_colN = float(np.nanmean(data["lat_land"][:, -1]))
    _south_idx, _north_idx = (0, -1) if _lat_col0 <= _lat_colN else (-1, 0)

    def _edge_stats(field):
        south, north = field[:, _south_idx], field[:, _north_idx]
        if (south.size < 2 or not np.isfinite(south).all()
                or not np.isfinite(north).all() or np.std(south) == 0
                or np.std(north) == 0):
            return dict(corr=None, mean_abs_diff=None, max_abs_diff=None,
                        mean_ratio=None)
        diff = north - south
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(south != 0, north / south, np.nan)
        return dict(
            corr=round(float(np.corrcoef(south, north)[0, 1]), 6),
            mean_abs_diff=round(float(np.mean(np.abs(diff))), 6),
            max_abs_diff=round(float(np.max(np.abs(diff))), 6),
            mean_ratio=(round(float(np.nanmean(ratio)), 6)
                        if np.isfinite(ratio).any() else None),
        )

    # Added 2026-10-22, same day Sebastian flagged "the bottom row is
    # suspiciously resembling the top row": correlation alone understated
    # how close the TOA edges actually are in ABSOLUTE terms (corr=0.9997
    # was already striking, but mean_abs_diff/mean_ratio make it concrete
    # -- e.g. a ~1-2% mean difference between two edges 100+ km apart,
    # despite the real surface (refl_noatm) edges NOT correlating at all).
    # Also report per-row means so the full south-to-north profile (not
    # just the two edges) is visible without re-deriving it by hand.
    edge_stats_toa = _edge_stats(refl_app)
    edge_stats_sfc = _edge_stats(refl_noatm)
    # row_profile_* always goes south -> north regardless of which raw
    # array index that is (reversed via [::-1] if index 0 was actually
    # north).
    _row_toa = np.nanmean(refl_app, axis=0)
    _row_sfc = np.nanmean(refl_noatm, axis=0)
    if _south_idx != 0:
        _row_toa, _row_sfc = _row_toa[::-1], _row_sfc[::-1]
    row_profile_toa = [round(float(v), 6) for v in _row_toa]
    row_profile_sfc = [round(float(v), 6) for v in _row_sfc]
    edge_corr_toa = edge_stats_toa["corr"]
    edge_corr_sfc = edge_stats_sfc["corr"]

    # --thin-atm-diag summary stats (2026-10-21): refl_app - refl_thin
    # isolates what the real atmosphere contributes, since both go
    # through the identical SHDOM/BRDF/geometry/BC solve otherwise (see
    # the long comment above the extra rt_shdom() call). Per
    # rt_shdom_vacuum_test.py's own physical prediction, the real
    # atmosphere should make cells BRIGHTER almost everywhere (Rayleigh
    # adds path radiance) -- frac_cells_atm_darkens close to 0 is the
    # healthy case; a large fraction would be a smoking gun for an
    # atmosphere-application bug.
    thin_atm_result = None
    if args.thin_atm_diag:
        edge_stats_thin = _edge_stats(refl_thin)
        _row_thin = np.nanmean(refl_thin, axis=0)
        if _south_idx != 0:
            _row_thin = _row_thin[::-1]
        row_profile_thin = [round(float(v), 6) for v in _row_thin]
        with np.errstate(invalid="ignore"):
            atm_contribution = refl_app - refl_thin
        thin_atm_result = {
            "note": "extra rt_shdom() round with atm_levels overridden to "
                    "a near-vacuum profile (0..%.3g km, %d levels) -- same "
                    "SHDOM/BRDF/geometry/BC solve as refl_app, so refl_app "
                    "- refl_thin isolates what the real atmosphere itself "
                    "(Rayleigh scattering + gas absorption) contributes, "
                    "as opposed to periodic-BC/solver/geometry effects "
                    "already present in refl_app vs refl_noatm above. Ran "
                    "with Ncpu=%d (not Ncpu=1 -- see run_dev_step1.py "
                    "comment above the extra rt_shdom() call)."
                    % (args.thin_atm_top_km, args.thin_atm_n_levels, args.ncpu),
            "atm_top_km": args.thin_atm_top_km,
            "n_levels": args.thin_atm_n_levels,
            "mean_refl_thin": round(float(np.nanmean(refl_thin)), 6),
            "mean_atm_contribution": round(float(np.nanmean(atm_contribution)), 6),
            "min_atm_contribution": round(float(np.nanmin(atm_contribution)), 6),
            "max_atm_contribution": round(float(np.nanmax(atm_contribution)), 6),
            "frac_cells_atm_darkens": round(
                float(np.nanmean(atm_contribution < 0)), 4),
            "edge_thin_atm": edge_stats_thin,
            "row_profile_thin_atm": row_profile_thin,
        }

    result = {
        "dev_step1_result": {
            "manifest": manifest_path,
            "tile_pick_source": args.tile_pick,
            "tile_km": args.tile_km,
            "lat": seed_lat,
            "lon": seed_lon,
            "overpass_utc": str(data["tile_overpass_utc"]),
            "geometry_deg": {k: round(tm[k], 4) for k in tm},
            "n_camera_pixels_total": n_total,
            "n_camera_pixels_valid": n_valid,
            "n_radiometer_footprints_inside": footprint_info["n_inside"],
            "n_radiometer_footprints_edge_needs_stitching": footprint_info["n_edge"],
            "n_radiometer_footprints_vza_model_capped": footprint_info["n_vza_model_capped"],
            "footprint_fill_fraction": {
                "note": "per-footprint exposure-gap / fill-fraction diagnostic "
                        "(Sebastian, 2026-10-22): for the footprint's own "
                        "ellipse area, a regular grid at the LOCAL real-pixel "
                        "nearest-neighbor pitch gives the 'would-have' "
                        "denominator; actual populated cells (real pixel "
                        "within 0.75x that pitch) give the numerator. "
                        "gap_km_max is the single largest real nearest-"
                        "neighbor gap found near that footprint. None where "
                        "there wasn't enough nearby real-pixel data to "
                        "estimate a local pitch at all.",
                "median_across_footprints": footprint_info["fill_fraction_median"],
                "min_across_footprints": footprint_info["fill_fraction_min"],
                "per_footprint": [
                    {"lat": r["lat"], "lon": r["lon"],
                     "classification": r["classification"],
                     "fill_fraction": r["fill_fraction"],
                     "gap_km_median": r["gap_km_median"],
                     "gap_km_max": r["gap_km_max"]}
                    for r in footprint_info["records"]
                ],
            },
            "periodic_bc_diagnostic": {
                "note": "south-vs-north edge comparison of the (nx,ny) "
                        "field, for TOA (SHDOM, BCFLAG=0 periodic) vs. "
                        "surface-only no-atmosphere reflectance -- "
                        "investigates whether SHDOM's periodic boundary "
                        "condition is driving the edge-similarity Sebastian "
                        "flagged 2026-10-22, by comparing against a field "
                        "(refl_noatm) that never goes through the periodic "
                        "solve at all. row_profile_* is the along-row mean "
                        "from south (index 0) to north (index -1), so the "
                        "full south-to-north shape is visible, not just the "
                        "two edges.",
                "edge_toa_shdom": edge_stats_toa,
                "edge_surface_noatm": edge_stats_sfc,
                "row_profile_toa_shdom": row_profile_toa,
                "row_profile_surface_noatm": row_profile_sfc,
            },
            "thin_atm_diagnostic": thin_atm_result,
            "measured_vs_rt": measured_vs_rt,
            "cam_cloud_fraction": 0.0,
            "cam_radiance_footprint": cam_radiance_footprint,
            "cam_radiance_footprint_units": "W m^-2 sr^-1 nm^-1 (SHDOM native output units)",
            "wavelength_nm": args.wavelength_nm,
            "provenance": {
                "radiance_source": "RT_proxy_dev_step1",
                "solver": args.solver,
                "libera_psf": "proxy",
                "ocean_windspeed_source": "ERA5 (this orbit's manifest file, "
                                          "overriding tile()'s default MERRA2)",
                "atmosphere": "standard_atmosphere",
                "f0_555nm_note": "refl_app uses a placeholder F0 constant "
                                 "(run_tile_rt.F0_555NM) -- NOT used for "
                                 "cam_radiance_footprint, which is raw "
                                 "SHDOM radiance, unaffected by that TBD.",
            },
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(result, f, default_flow_style=False, sort_keys=False)

    l2_output_paths = None
    if args.write_l2_output:
        if footprint_info["n_inside"] == 0:
            print("\n--write-l2-output: 0 'inside' radiometer footprints "
                  "for this tile -- nothing to write (try a larger "
                  "--tile-km). Skipping CF-CAM output.")
        else:
            import write_cf_cam_output
            l2_output_paths = write_cf_cam_output.write_cf_cam_output(
                footprint_info, manifest_path, args.l2_output_dir,
                wavelength_nm=args.wavelength_nm,
                tile_provenance={
                    "lat": seed_lat, "lon": seed_lon,
                    "tile_km": args.tile_km, "solver": args.solver,
                    "margin_cells": args.margin_cells,
                },
                cam_cloud_fraction_mode=args.cam_cloud_fraction,
                pad_rad_l1b_path=(roles["rad_l1b"] if args.pad_to_orbit
                                 else None))

    if args.plot:
        import plot_dev_step1_qa
        plot_out = args.plot_out or (os.path.splitext(args.out)[0] + "_qa.png")
        plot_dev_step1_qa.make_qa_plots(data, pooled, rad, footprint_info,
                                        args.tile_km, plot_out,
                                        refl_noatm=refl_noatm, refl_app=refl_app,
                                        cloud_flag_grid=cloud_flag_grid)
        if args.thin_atm_diag:
            thin_atm_plot_out = os.path.splitext(plot_out)[0] + "_thin_atm.png"
            plot_dev_step1_qa.make_thin_atm_diag_plot(
                data, refl_app, refl_thin, refl_noatm,
                pooled["lat0"], pooled["lon0"], thin_atm_plot_out)
        if rad_meas is not None:
            measured_vs_rt_plot_out = (os.path.splitext(plot_out)[0]
                                       + "_measured_vs_rt.png")
            plot_dev_step1_qa.make_measured_vs_rt_plot(
                data, rad, rad_meas, n_px_per_cell, pooled["lat0"], pooled["lon0"],
                measured_vs_rt_plot_out, refl_app=refl_app, refl_meas=refl_meas,
                sza_camera_deg=sza_cam, sza_source_deg=sza_src,
                cloud_info=cloud_info, cloud_flag_grid=cloud_flag_grid)
        else:
            print("  (skipping measured-vs-RT PNG: no 'measured' radiance "
                 "available -- pass an L1X cam_l1b, not a plain L1B, via the "
                 "input manifest to get one)")

    print("\n=== done (%.0f s) ===" % (time.time() - t0))
    print("cam_radiance_footprint = %.6g W m^-2 sr^-1 nm^-1" % cam_radiance_footprint)
    print("n_camera_pixels: %d total / %d valid (within tile box, near-nadir)"
          % (n_total, n_valid))
    print("radiometer footprints fed by this tile: %d fully inside, %d "
          "straddling the edge (need an adjacent tile too, not yet stitched)"
          % (footprint_info["n_inside"], footprint_info["n_edge"]))
    print("tile-mean geometry: " + ", ".join("%s=%.2f deg" % (k, tm[k]) for k in tm))
    print("periodic-BC diagnostic -- south/north edge correlation: "
          "TOA(SHDOM)=%s, surface(no-atm)=%s"
          % (edge_corr_toa, edge_corr_sfc))
    if measured_vs_rt is not None:
        print("measured-vs-RT -- bias(measured-RT)=%.4g, RMSE=%.4g, corr=%s "
              "[W m^-2 sr^-1 nm^-1] (%d/%d grid cells)"
              % (measured_vs_rt["mean_bias_measured_minus_rt"],
                 measured_vs_rt["rmse_rt_vs_measured"],
                 measured_vs_rt["corr_rt_vs_measured"],
                 measured_vs_rt["n_rt_grid_cells_compared"],
                 measured_vs_rt["n_rt_grid_cells_total"]))
    if args.thin_atm_diag:
        print("thin-atm-diag -- mean atm contribution (refl_app - "
              "refl_thin): %.6g (frac cells where atm DARKENS instead of "
              "brightens: %.3f, expect ~0)"
              % (thin_atm_result["mean_atm_contribution"],
                 thin_atm_result["frac_cells_atm_darkens"]))

    if l2_output_paths is not None:
        print("Wrote CF-CAM output NetCDF: %s" % l2_output_paths[0])
        print("Wrote output manifest: %s" % l2_output_paths[1])

    print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
