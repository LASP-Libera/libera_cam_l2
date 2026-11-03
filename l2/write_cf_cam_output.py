"""
Minimal CF-CAM L2 NetCDF + output-manifest writer for the dev-step-1
single-tile RT-proxy pipeline (Sebastian, 2026-10-22): "I would actually
like to produce an output file and output manifest, even if the
single-tile code doesn't go all the way to the end and generates
placeholder data (0 for CF, RT-generated radiances for the measured and
averaged 555 nm radiances)... We need something that is good enough for
SDC to test end to end (i.e., input manifest to output manifest) on
their system."

Scope, explicitly NOT more than this (per the same message):
- Only radiometer footprints classified 'inside' this one tile go into
  the output -- no multi-tile stitching ("I don't want to get into any
  multi-tile work just yet"). 'edge' footprints are skipped entirely for
  now (same records footprints_feeding_tile() already flags as needing
  an adjacent tile).
- cam_cloud_fraction is a hardcoded 0.0 placeholder (dev-step-1: camera
  L1B is dark/pre-launch, no real cloud detection yet -- matches the
  roadmap's "Output, this week" decision).
- cam_radiance_footprint is the RT-proxy SHDOM radiance
  (run_tile_rt.radiance_and_counts_for_footprints()), NOT a real PSF
  convolution of measured camera pixels -- flagged via this variable's
  own `radiance_source`/`libera_psf` attributes, per the roadmap's
  explicit "flag it as such... not silently" decision, not left as a
  silent/unlabeled number.
- psf_coverage_fraction (roadmap: "dropped entirely for dev-step-1") and
  cam_cloud_fraction_uncertainty (spec: "methodology TBD") are both left
  at the CF-CAM spec's own _FillValue, not computed.
- Solar_Azimuth_Surface_WRT_North/Relative_Azimuth_Surface ARE passed
  through from RAD-4CH even though the 2026-06-24 spec sheet still flags
  the former "PENDING" -- confirmed present as real fields in the actual
  RAD-4CH V0-7-0 file this pipeline reads (checked directly via
  netCDF4's own variable list, not assumed from the spec note).

2026-10-23 update (item C, Sebastian): the two placeholders described
above are now wired to real per-footprint values by default --
cam_cloud_fraction_mode="real" (new default) uses
run_tile_rt.radiance_and_counts_for_footprints()'s cam_cloud_fraction_rt
(the e-fold cloud-threshold fraction over the SAME RT grid cells used
for the QA plots' cloud markers -- "the footprints in green could get
small cloud fraction numbers"), and cam_radiance_footprint now carries
cam_radiance_footprint_measured -- the REAL L1X measured-radiance
average over camera pixels inside each footprint's ellipse, NOT the RT
radiance (Sebastian, same day: "what should be sent to the output file
is not the RT radiance... the 'measured' (L1X average footprint by
footprint) goes into the footprint-level radiance output"). The old
"zero"/"random" cam_cloud_fraction_mode values are kept, now as
explicit debug/testing overrides rather than the default -- see
build_cf_cam_arrays()'s docstring. radiance_source/libera_psf/notes on
the cam_radiance_footprint NetCDF variable updated to describe this
(still an ellipse-membership approximation, not a true PSF
convolution).

Variable names/units/dtypes/valid_range/_FillValue below are transcribed
directly from the CF-CAM sheet of
docs/Libera_Data_Product_Specifications_PipelineDelivery_local_20260624_populated_CFCAM.xlsx
(the populated spec) -- NOT from L2a/l2_cam/data/cfg_L2_CF_CAM.yml, which
is still the generic, unfilled-in scaffold template (confirmed by
reading it: placeholder variable names like cloud_fraction/quality_flags
that don't match the real CF-CAM product at all).
"""
import datetime
import os
import sys

import numpy as np
import netCDF4 as nc

from libera_utils import Manifest
from libera_utils.io.filenaming import LiberaDataProductFilename

# Item D (2026-10-23, optional full-orbit padding): needs
# run_tile_rt.read_rad4ch_full_geometry(), same sibling-module convention
# run_dev_step1.py already uses.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import run_tile_rt  # noqa: E402


# CCSDS epoch used by RADIOMETER_TIME's own on-disk encoding (nanoseconds
# since 1958-01-01), vs. the 1970-01-01 (numpy/unix) epoch this codebase
# uses internally for all its own time-window arithmetic (chop()'s
# t_center_ns, footprints_feeding_tile()'s t_ns -- both come from
# xarray's CF-decoding, which always lands on the 1970 epoch regardless
# of a variable's own on-disk reference date; see run_tile_rt.py's own
# comments on this). Getting this conversion wrong here would be the
# SAME bug category already hit twice elsewhere in this codebase (see
# docs/L1B_TO_L2_ROADMAP.md) -- computed via datetime/timedelta, not
# hand-counted, to rule out a leap-year slip.
_EPOCH_1958 = datetime.datetime(1958, 1, 1, tzinfo=datetime.timezone.utc)
_EPOCH_1970 = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_NS_1970_MINUS_1958 = int((_EPOCH_1970 - _EPOCH_1958).total_seconds() * 1e9)

_FILL_F32 = -999.0
_RADIANCE_FILL = -999.0
_CAM_CLOUD_FRACTION_MODES = ("real", "zero", "random")


def _cam_cloud_fraction_array(n, mode, rng, records=None):
    """records required (and used) only for mode="real" -- the real,
    per-footprint cam_cloud_fraction_rt values from
    run_tile_rt.radiance_and_counts_for_footprints() (None, where cloud
    detection didn't run or had too few comparable cells, maps to this
    spec field's own _FillValue, same convention every other optional
    field here uses -- NOT 0.0, which would wrongly read as "confirmed
    clear")."""
    if mode == "real":
        if records is None:
            raise ValueError(
                "cam_cloud_fraction_mode='real' requires records "
                "(internal error -- build_cf_cam_arrays() should always "
                "pass this)")
        return np.array(
            [(r["cam_cloud_fraction_rt"]
              if r.get("cam_cloud_fraction_rt") is not None else _FILL_F32)
             for r in records], dtype=np.float32)
    if mode == "zero":
        return np.zeros(n, dtype=np.float32)
    if mode == "random":
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    raise ValueError("cam_cloud_fraction_mode must be one of %r, got %r"
                     % (_CAM_CLOUD_FRACTION_MODES, mode))


def _ns1970_to_ns1958(t_ns_1970):
    """Convert this codebase's internal 1970-epoch nanosecond timestamps
    (xarray-decoded) to the 1958 CCSDS-epoch nanoseconds RADIOMETER_TIME
    must be written in."""
    return int(t_ns_1970) + _NS_1970_MINUS_1958


def _add_var(ds, name, data_arr, dtype, long_name, units, valid_range,
            fill_value, notes=None):
    var = ds.createVariable(name, dtype, ("RADIOMETER_TIME",),
                            fill_value=fill_value)
    var.long_name = long_name
    var.units = units
    var.valid_range = np.array(valid_range, dtype=dtype)
    if notes:
        var.notes = notes
    var[:] = data_arr
    return var


def _inside_records_sorted(footprint_info):
    records = [r for r in footprint_info.get("records", [])
              if r["classification"] == "inside"]
    if not records:
        raise ValueError(
            "No 'inside' footprints to write -- nothing for this tile to "
            "contribute to a CF-CAM output file (check "
            "footprints_feeding_tile()'s own n_inside count; a larger "
            "--tile-km may be needed)")
    return sorted(records, key=lambda r: r["t_ns"])


def _per_record_arrays(records, cam_cloud_fraction_mode, rng):
    """The flat per-variable arrays for exactly these (already 'inside'-
    filtered, time-sorted) records -- factored out of build_cf_cam_arrays()
    2026-10-23 (item D) so build_cf_cam_arrays_padded() can build this
    same 37-ish-row 'patch' and then scatter it into a full-orbit-length
    array, instead of a second, possibly-inconsistent transcription of
    this per-field logic. See build_cf_cam_arrays()'s docstring for what
    cam_cloud_fraction_mode means; rng is passed straight through to
    _cam_cloud_fraction_array()."""
    n = len(records)

    # 2026-10-23, item C (Sebastian): the REAL L1X measured-radiance
    # footprint average, NOT the RT-calculated one -- "what should be
    # sent to the output file is not the RT radiance... the 'measured'
    # (L1X average footprint by footprint) goes into the footprint-level
    # radiance output." cam_radiance_footprint_rt (SHDOM) remains
    # available on each record for QA/diagnostic use, just no longer
    # what this NetCDF variable carries.
    rad_nm = np.array(
        [(r["cam_radiance_footprint_measured"]
          if r.get("cam_radiance_footprint_measured") is not None else np.nan)
         for r in records], dtype=np.float64)
    # Spec units: W m^-2 sr^-1 um^-1; our native RT/L1X units are per-nm.
    # 1 um = 1000 nm, so a per-um density is 1000x a per-nm density for
    # the same total power (L_um * 1um == L_nm * 1000nm => L_um=1000*L_nm).
    rad_um = np.where(np.isfinite(rad_nm), rad_nm * 1000.0,
                      _RADIANCE_FILL).astype(np.float32)

    return dict(
        t_ns=np.array([r["t_ns"] for r in records], dtype=np.int64),
        t_ns_1958=np.array([_ns1970_to_ns1958(r["t_ns"]) for r in records],
                           dtype=np.int64),
        lat=np.array([r["lat"] for r in records], dtype=np.float32),
        lon=np.array([r["lon"] for r in records], dtype=np.float32),
        alt=np.array([r["altitude_m"] for r in records], dtype=np.float32),
        sza=np.array([r["sza_deg"] for r in records], dtype=np.float32),
        vza=np.array([r["vza_deg"] for r in records], dtype=np.float32),
        raa=np.array([r["raa_deg"] for r in records], dtype=np.float32),
        vaa=np.array([r["vaa_deg"] for r in records], dtype=np.float32),
        saa=np.array([r["saa_deg"] for r in records], dtype=np.float32),
        n_total=np.array([r["n_camera_pixels_total"] for r in records],
                         dtype=np.uint16),
        n_valid=np.array([r["n_camera_pixels_valid"] for r in records],
                         dtype=np.uint16),
        cam_cf=_cam_cloud_fraction_array(n, cam_cloud_fraction_mode, rng,
                                         records=records),
        cam_cf_unc=np.full(n, _FILL_F32, dtype=np.float32),
        # run_tile_rt.footprint_fill_fraction()'s own fill_fraction --
        # designed earlier for exactly this (Sebastian: local real-pixel
        # nearest-neighbor pitch -> regular-grid "would-have" denominator
        # vs. actually-populated numerator over the footprint ellipse) --
        # NOT left uncomputed; wired in 2026-10-22 after Sebastian flagged
        # that it had been designed but not connected to this field. None
        # (not enough nearby real pixels to estimate a local pitch) maps
        # to the spec's own _FillValue, same as every other missing field
        # here, not to 0.0 (which would wrongly read as "zero coverage").
        psf_cov=np.array(
            [(r["fill_fraction"] if r.get("fill_fraction") is not None
              else _FILL_F32) for r in records], dtype=np.float32),
        rad_um=rad_um,
    )


def build_cf_cam_arrays(footprint_info, wavelength_nm=555.0,
                        cam_cloud_fraction_mode="real", rng=None):
    """Pure-numpy half of the writer: turns footprint_info['records']
    ('inside' ones only) into the flat per-variable arrays the NetCDF
    writer needs, plus the UTC start/end of the records. Split out from
    the NetCDF/manifest I/O so this part -- the part most likely to have
    a transcription bug -- can be unit-tested without needing netCDF4/
    libera_utils or a real tile run.

    cam_cloud_fraction_mode: "real" (default, added 2026-10-23, item C)
    uses each record's own cam_cloud_fraction_rt -- the e-fold cloud-
    threshold fraction run_tile_rt.radiance_and_counts_for_footprints()
    already computed per footprint ("the footprints in green could get
    small cloud fraction numbers"), _FillValue where that wasn't
    available (cloud detection skipped/too few comparable cells). The
    original 2026-10-22 placeholders are kept as explicit debug/testing
    overrides: "zero" (hardcoded 0.0) or "random" (a uniform [0, 1)
    draw per footprint) -- NOT 0-100: cam_cloud_fraction's own CF-CAM
    spec entry (units="1", valid_range=[0,1]) is a unitless fraction,
    and a 0-100 placeholder would just fail SDC's validation for no
    reason (confirmed with Sebastian 2026-10-22). rng lets a caller pass
    a seeded np.random.Generator for a reproducible "random"-mode test
    file; defaults to an unseeded one; unused by "real"/"zero".

    This is the ORIGINAL (2026-10-22) tile-only shape: the output has
    exactly one record per 'inside' footprint (today ~37), nothing else.
    See build_cf_cam_arrays_padded() (item D, 2026-10-23) for the
    optional full-orbit-length variant."""
    records = _inside_records_sorted(footprint_info)
    arrays = _per_record_arrays(records, cam_cloud_fraction_mode, rng)
    del arrays["t_ns"]  # internal to _per_record_arrays/padding; not a NetCDF var
    n = len(records)
    utc_start = datetime.datetime.fromtimestamp(
        records[0]["t_ns"] / 1e9, tz=datetime.timezone.utc)
    utc_end = datetime.datetime.fromtimestamp(
        records[-1]["t_ns"] / 1e9, tz=datetime.timezone.utc)
    return arrays, utc_start, utc_end, n


def build_cf_cam_arrays_padded(footprint_info, rad_l1b_path, wavelength_nm=555.0,
                               cam_cloud_fraction_mode="real", rng=None):
    """Item D (2026-10-23, Sebastian's plan, optional): same per-footprint
    quantities as build_cf_cam_arrays(), but the output is lengthened to
    the FULL RAD-4CH orbit (every record in rad_l1b_path, via
    run_tile_rt.read_rad4ch_full_geometry() -- same file run_dev_step1.py
    already reads as roles['rad_l1b']) instead of just the ~37 'inside'
    footprints. Geolocation/geometry (lat/lon/altitude/SZA/VZA/RAA/VAA/
    SAA/time) are passed through from the real RAD-4CH file for EVERY
    record in the orbit; the camera-derived fields this one tile can
    actually populate (cam_radiance_footprint, cam_cloud_fraction,
    cam_cloud_fraction_uncertainty, psf_coverage_fraction,
    n_camera_pixels_total/valid) are real only at this tile's own
    footprints and the spec's _FillValue everywhere else -- EXCEPT the
    two pixel-count fields, which are genuinely 0 outside the tile (the
    camera truly covered zero pixels there), not an unknown placeholder,
    so those stay their natural np.zeros() default rather than _FillValue.

    Each 'inside' footprint is placed in the full-orbit array at the
    index whose RAD-4CH radiometer_time EXACTLY matches the footprint's
    own t_ns -- both values trace back to the same original RAD-4CH
    samples (footprints_feeding_tile() reads rad_l1b_path itself to build
    footprint_info in the first place), so this is an exact lookup, not a
    nearest-neighbor search; a footprint whose t_ns isn't found at all
    would mean rad_l1b_path here doesn't match the one footprint_info was
    actually built from, which raises loudly rather than silently
    mis-placing a row.

    utc_start/utc_end (and therefore the output filename) span the WHOLE
    orbit's own first/last timestamp here, not just the tile -- the file
    now genuinely covers that full range."""
    records = _inside_records_sorted(footprint_info)
    patch = _per_record_arrays(records, cam_cloud_fraction_mode, rng)

    full = run_tile_rt.read_rad4ch_full_geometry(rad_l1b_path)
    t_ns_full = full["t_ns"]
    n_full = t_ns_full.size

    idx_of_t_ns = {int(t): i for i, t in enumerate(t_ns_full)}
    patch_idx = np.empty(len(records), dtype=np.int64)
    for k, r in enumerate(records):
        idx = idx_of_t_ns.get(int(r["t_ns"]))
        if idx is None:
            raise ValueError(
                "build_cf_cam_arrays_padded(): 'inside' footprint at "
                "t_ns=%d (lat=%.4f, lon=%.4f) not found anywhere in "
                "rad_l1b_path=%r's own radiometer_time -- this almost "
                "certainly means rad_l1b_path is a DIFFERENT RAD-4CH file "
                "than the one footprints_feeding_tile() actually used to "
                "build footprint_info (pass the SAME roles['rad_l1b'] "
                "the rest of this pipeline run used)."
                % (r["t_ns"], r["lat"], r["lon"], rad_l1b_path))
        patch_idx[k] = idx

    arrays = dict(
        t_ns_1958=np.array([_ns1970_to_ns1958(int(t)) for t in t_ns_full],
                           dtype=np.int64),
        lat=full["lat"].astype(np.float32),
        lon=full["lon"].astype(np.float32),
        alt=full["alt"].astype(np.float32),
        sza=full["sza"].astype(np.float32),
        vza=full["vza"].astype(np.float32),
        raa=full["raa"].astype(np.float32),
        vaa=full["vaa"].astype(np.float32),
        saa=full["saa"].astype(np.float32),
        n_total=np.zeros(n_full, dtype=np.uint16),
        n_valid=np.zeros(n_full, dtype=np.uint16),
        cam_cf=np.full(n_full, _FILL_F32, dtype=np.float32),
        cam_cf_unc=np.full(n_full, _FILL_F32, dtype=np.float32),
        psf_cov=np.full(n_full, _FILL_F32, dtype=np.float32),
        rad_um=np.full(n_full, _RADIANCE_FILL, dtype=np.float32),
    )
    for key in ("n_total", "n_valid", "cam_cf", "cam_cf_unc", "psf_cov", "rad_um"):
        arrays[key][patch_idx] = patch[key]

    utc_start = datetime.datetime.fromtimestamp(
        int(t_ns_full[0]) / 1e9, tz=datetime.timezone.utc)
    utc_end = datetime.datetime.fromtimestamp(
        int(t_ns_full[-1]) / 1e9, tz=datetime.timezone.utc)
    return arrays, utc_start, utc_end, n_full


def write_cf_cam_netcdf(arrays, n, nc_path, wavelength_nm=555.0,
                        version="0.1.0", tile_provenance=None,
                        cam_cloud_fraction_mode="real", padded=False):
    """Write the CF-CAM NetCDF4 file itself from build_cf_cam_arrays()'s
    output. Separate from write_cf_cam_output() so it can be exercised
    (and was, in a standalone smoke test) without libera_utils."""
    os.makedirs(os.path.dirname(nc_path) or ".", exist_ok=True)
    with nc.Dataset(nc_path, "w", format="NETCDF4") as ds:
        ds.createDimension("RADIOMETER_TIME", n)

        tvar = ds.createVariable("radiometer_time", "i8", ("RADIOMETER_TIME",))
        tvar.units = "nanoseconds since 1958-01-01"
        tvar.calendar = "standard"
        tvar.long_name = ("Time of each measurement (UTC) against a base "
                          "reference of the CCSDS Epoch (January 1, 1958). "
                          "Uniquely identifies each radiometer sample.")
        tvar[:] = arrays["t_ns_1958"]

        _add_var(ds, "Latitude", arrays["lat"], "f4",
                "Geodetic latitude of radiometer Earth point. CRS WGS84",
                "degrees_north", [-90, 90], -999.0,
                "Passed through from RAD-4CH")
        _add_var(ds, "Longitude", arrays["lon"], "f4",
                "Longitude of radiometer Earth point. CRS WGS84",
                "degrees_east", [-180, 180], -999.0,
                "Passed through from RAD-4CH")
        _add_var(ds, "Altitude", arrays["alt"], "f4",
                "Height of radiometer Earth point above WGS84 ellipsoid "
                "(EPSG:4979)", "meters", [-1000, 30000], -9999.0,
                "Passed through from RAD-4CH")
        _add_var(ds, "Solar_Zenith_Surface", arrays["sza"], "f4",
                "Geodetic zenith angle at the Earth point of the Sun",
                "degrees", [0, 180], -999.0, "Passed through from RAD-4CH")
        _add_var(ds, "Viewing_Zenith_Surface", arrays["vza"], "f4",
                "Geodetic angle at the Earth point to the satellite",
                "degrees", [0, 180], -999.0, "Passed through from RAD-4CH")
        _add_var(ds, "Relative_Azimuth_Surface", arrays["raa"], "f4",
                "Geodetic azimuth angle at the Earth point of the "
                "satellite relative to the solar plane (measured "
                "clockwise in the plane normal to the geodetic zenith)",
                "degrees", [0, 360], -999.0, "Passed through from RAD-4CH")
        _add_var(ds, "Viewing_Azimuth_Surface_WRT_North", arrays["vaa"], "f4",
                "Libera viewing azimuth angle at the surface with "
                "respect to North", "degrees", [0, 360], -999.0,
                "Passed through from RAD-4CH")
        _add_var(ds, "Solar_Azimuth_Surface_WRT_North", arrays["saa"], "f4",
                "Solar azimuth angle at the surface with respect to "
                "North (clockwise)", "degrees", [0, 360], -999.0,
                "Passed through from RAD-4CH (present in this pipeline's "
                "RAD-4CH file despite the 2026-06-24 spec sheet's "
                "'PENDING' note)")
        _add_var(ds, "n_camera_pixels_total", arrays["n_total"], "u2",
                "Total number of camera pixels within the radiometer "
                "PSF boundary", "count", [0, 65535], 0,
                "dev-step-1: ellipse-membership test against tile()'s "
                "pooled camera pixels, not a real PSF footprint")
        _add_var(ds, "n_camera_pixels_valid", arrays["n_valid"], "u2",
                "Number of valid (unmasked) camera pixels within the "
                "radiometer PSF boundary", "count", [0, 65535], 0,
                "Used as denominator for cam_cloud_fraction")
        _add_var(ds, "psf_coverage_fraction", arrays["psf_cov"], "f4",
                "Fraction of the radiometer PSF area covered by valid "
                "camera pixels", "1", [0, 1], -999.0,
                "dev-step-1 PROXY, not a real PSF areal overlap: "
                "run_tile_rt.footprint_fill_fraction()'s purely "
                "geometric/empirical estimate -- a regular grid at the "
                "LOCAL real-camera-pixel nearest-neighbor pitch over the "
                "footprint ellipse gives the 'would-have' denominator, "
                "actual populated cells (a real pixel within 0.75x that "
                "pitch) give the numerator. _FillValue where there "
                "weren't enough nearby real pixels to estimate a local "
                "pitch at all (not necessarily zero coverage).")
        if cam_cloud_fraction_mode == "real":
            _cf_notes = (
                "dev-step-1 RT-PROXY cloud detection (not yet a real "
                "camera-imagery algorithm): run_tile_rt.cloud_threshold_"
                "efold()'s e-fold adaptive threshold on the (measured-RT) "
                "difference, averaged over the same RT/BRDF grid cells "
                "cam_radiance_footprint's ellipse-membership test uses "
                "(run_tile_rt.radiance_and_counts_for_footprints()). "
                "_FillValue where cloud detection didn't run for this "
                "tile (e.g. too few grid cells with both a measured and "
                "an RT value to compare) -- NOT 0.0, which would wrongly "
                "read as a confirmed clear-sky footprint.")
        else:
            _cf_notes = (
                "PLACEHOLDER (cam_cloud_fraction_mode=%r, not the default "
                "'real') -- either a constant 0.0, or a uniform [0,1) "
                "draw per footprint, for exercising downstream code paths "
                "a constant 0.0 can't -- see build_cf_cam_arrays()."
                % cam_cloud_fraction_mode)
        _add_var(ds, "cam_cloud_fraction", arrays["cam_cf"], "f4",
                "Fraction of valid camera pixels within the radiometer "
                "footprint classified as cloudy", "1", [0, 1], -999.0,
                _cf_notes)
        _add_var(ds, "cam_cloud_fraction_uncertainty", arrays["cam_cf_unc"],
                "f4", "Estimated uncertainty in cam_cloud_fraction", "1",
                [0, 1], -999.0, "NOT COMPUTED -- methodology TBD per spec")

        rvar = ds.createVariable("cam_radiance_footprint", "f4",
                                 ("RADIOMETER_TIME",), fill_value=_RADIANCE_FILL)
        rvar.long_name = ("Camera narrowband radiance (555 nm) convolved "
                          "with the Libera radiometer point spread "
                          "function (PSF) and aggregated to each "
                          "radiometer footprint.")
        rvar.units = "W m-2 sr-1 um-1"
        rvar.valid_range = np.array([0, 1000], dtype="f4")
        rvar.radiance_source = "L1X_measured_proxy_dev_step1"
        rvar.libera_psf = "proxy"
        rvar.notes = (
            "dev-step-1 PROXY, not a real PSF convolution: nanmean of "
            "the camera L1X's own MEASURED %g nm radiance (the real "
            "MODIS/VIIRS-proxy-scene pixel values build_l1x.py wrote "
            "into the camera L1B, restricted to pixels the proxy crop "
            "actually overwrote) over every real camera pixel inside "
            "this footprint's ellipse "
            "(run_tile_rt.radiance_and_counts_for_footprints()'s "
            "cam_radiance_footprint_measured -- an ellipse-membership "
            "test against real pixel positions, not true PSF weighting). "
            "NOT the RT/SHDOM-calculated radiance (see "
            "cam_radiance_footprint_rt in the pipeline's own YAML/QA "
            "output for that, kept for diagnostic comparison only). "
            "Converted from native W m^-2 sr^-1 nm^-1 to this variable's "
            "W m^-2 sr^-1 um^-1 by x1000." % wavelength_nm)
        rvar[:] = arrays["rad_um"]

        ds.ProductID = "L2_CF-CAM"
        ds.algorithm_version = version
        if padded:
            ds.summary = (
                "DEV-STAGE PLACEHOLDER OUTPUT -- single-tile RT-proxy "
                "pipeline (l2/run_dev_step1.py), padded (item D, "
                "2026-10-23) to the full RAD-4CH orbit's record count: "
                "Latitude/Longitude/Altitude/Solar_Zenith_Surface/"
                "Viewing_Zenith_Surface/Relative_Azimuth_Surface/"
                "Viewing_Azimuth_Surface_WRT_North/"
                "Solar_Azimuth_Surface_WRT_North/radiometer_time are real "
                "for EVERY record in the orbit; cam_radiance_footprint/"
                "cam_cloud_fraction/cam_cloud_fraction_uncertainty/"
                "psf_coverage_fraction are real ONLY at this one tile's "
                "'inside' radiometer footprints and this product's own "
                "_FillValue everywhere else (no multi-tile stitching yet); "
                "n_camera_pixels_total/valid are genuinely 0 outside the "
                "tile, not a placeholder. NOT for science use. See "
                "docs/L1B_TO_L2_ROADMAP.md.")
        else:
            ds.summary = (
                "DEV-STAGE PLACEHOLDER OUTPUT -- single-tile RT-proxy "
                "pipeline (l2/run_dev_step1.py), 'inside' radiometer "
                "footprints only (no multi-tile stitching). NOT for science "
                "use. See docs/L1B_TO_L2_ROADMAP.md.")
        ds.cf_cam_full_orbit_padded = "true" if padded else "false"
        ds.history = ("Created %s by l2/write_cf_cam_output.py"
                      % datetime.datetime.now(datetime.timezone.utc).isoformat())
        if tile_provenance:
            for k, v in tile_provenance.items():
                setattr(ds, "tile_%s" % k, str(v))
    return nc_path


def write_cf_cam_output(footprint_info, input_manifest_path, out_dir,
                        wavelength_nm=555.0, version="0.1.0",
                        tile_provenance=None, cam_cloud_fraction_mode="real",
                        rng=None, verbose=True, pad_rad_l1b_path=None):
    """Write a minimal CF-CAM L2 NetCDF plus its output manifest, from a
    run_dev_step1.py run's own footprint_info (after
    run_tile_rt.radiance_and_counts_for_footprints() has populated it).

    cam_cloud_fraction_mode: see build_cf_cam_arrays()'s docstring.

    pad_rad_l1b_path (item D, 2026-10-23, optional): None (default) keeps
    the original one-record-per-'inside'-footprint output (today ~37
    rows). Given a RAD-4CH L1B path (normally the SAME file already read
    as roles['rad_l1b'] elsewhere in this pipeline run -- see
    build_cf_cam_arrays_padded()'s docstring for why it must be the same
    file), the output is instead lengthened to that file's full orbit,
    with real geometry passed through everywhere and the camera-derived
    fields fill-valued outside this one tile's own footprints.

    Returns (nc_path, manifest_path).
    """
    if pad_rad_l1b_path is not None:
        arrays, utc_start, utc_end, n = build_cf_cam_arrays_padded(
            footprint_info, pad_rad_l1b_path, wavelength_nm=wavelength_nm,
            cam_cloud_fraction_mode=cam_cloud_fraction_mode, rng=rng)
    else:
        arrays, utc_start, utc_end, n = build_cf_cam_arrays(
            footprint_info, wavelength_nm=wavelength_nm,
            cam_cloud_fraction_mode=cam_cloud_fraction_mode, rng=rng)

    # Manifest.add_files() (libera_utils/io/manifest.py:check_file_structure)
    # hard-requires an absolute path and raises ValueError otherwise --
    # hit for real 2026-10-22 (Sebastian) with the CLI's relative default
    # "output_l2". Resolve once here so every path handed to
    # LiberaDataProductFilename/add_files/write is absolute, regardless
    # of whether the caller passed a relative or absolute out_dir.
    out_dir_abs = os.path.abspath(out_dir)

    out_filename = LiberaDataProductFilename.from_filename_parts(
        product_name="CF-CAM", version=version,
        utc_start=utc_start, utc_end=utc_end, basepath=out_dir_abs)
    nc_path = str(out_filename.path)
    assert os.path.isabs(nc_path), (
        "LiberaDataProductFilename.from_filename_parts() returned a "
        "non-absolute path (%r) even with an absolute basepath (%r) -- "
        "Manifest.add_files() will reject this; check for a "
        "libera_utils API change." % (nc_path, out_dir_abs))

    write_cf_cam_netcdf(arrays, n, nc_path, wavelength_nm=wavelength_nm,
                        version=version, tile_provenance=tile_provenance,
                        cam_cloud_fraction_mode=cam_cloud_fraction_mode,
                        padded=(pad_rad_l1b_path is not None))

    input_manifest = Manifest.from_file(input_manifest_path)
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
    output_manifest.add_files(nc_path)
    manifest_path = output_manifest.write(out_path=out_dir_abs)

    if verbose:
        if pad_rad_l1b_path is not None:
            print("Wrote CF-CAM output NetCDF: %s (%d full-orbit records, "
                  "real data at this tile's 'inside' footprints only)"
                  % (nc_path, n))
        else:
            print("Wrote CF-CAM output NetCDF: %s (%d 'inside' footprint "
                  "records)" % (nc_path, n))
        print("Wrote output manifest: %s" % manifest_path)
    return nc_path, str(manifest_path)
