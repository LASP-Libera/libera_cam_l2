"""
l2/fetch_proxy_scene.py

Dev-step-2, stage 1 of the "L1X" proxy-radiance plan (see
docs/L1B_TO_L2_ROADMAP.md): dev-step-1 runs the real camera geometry
through SHDOM with no real scene (dark/near-zero camera radiances).
Dev-step-2 will populate camera pixels with a real reflectance field
borrowed from a co-located, co-timed MODIS/VIIRS overpass instead, via
the same ellipsoidal-Gaussian PSF apps/gcp/gcp_proxy.py already uses for
ground-control-point work (libcam.camera.Platform.mod2cam).

This script is step 1 of that: given a tile pick (lat/lon/overpass time
from tile_pick.yml), search the NASA LAADS DAAC archive for MODIS (or
VIIRS) granules close in time, download just their (small) geolocation
files, and compare each candidate's REAL viewing/solar geometry at the
tile location against the tile's own recorded vza_deg/sza_deg -- closest
in time is not necessarily best, since MODIS/VIIRS cross-track swaths
are ~2300/3000 km wide and a granule can be close in time but viewed
from near the swath edge (high VZA), which both distorts pixel footprint
size and changes the BRDF-driven reflectance away from the near-nadir
view our own camera geometry has (see tile_pick.yml's vza_deg, usually
small for a camera-centered site pick). Only once a candidate is picked
(via --download INDEX) does this script fetch the L1B radiance file
itself. The PSF projection into camera pixel space is a later, separate
step, once we have real imager radiance in hand.

Uses er3t.util.daac's get_satfile_tag() (date+lon/lat granule search
against NASA's geoMeta catalog, generic across MODIS/VIIRS) and
download_laads_https() (the actual pull, via EARTHDATA_TOKEN) -- both
vendored inside L2a/l2_cam/software/er3t, confirmed 2026-10-22 to be the
same machinery https://konradsebastian.github.io/er3t-edu's Example 06
(06_modis_data_download.py) uses. Geometry-at-tile is read directly with
this repo's own libcam.modis.ModisGeo (same reader apps/gcp/gcp_proxy.py
already uses), not a second/duplicate er3t reader.

Needs a valid Earthdata token: generate one at urs.earthdata.nasa.gov
(Generate Token tab), then in your shell rc file:
    setenv EARTHDATA_TOKEN "your-token-here"     (csh/tcsh, e.g. ~/.tcshrc)
    export EARTHDATA_TOKEN="your-token-here"     (bash/zsh)

Usage
-----
    # list candidate MODIS granules with time offset AND real viewing/
    # solar geometry at the tile -- downloads only the small geolocation
    # files, not any L1B radiance yet
    python l2/fetch_proxy_scene.py

    # same, for VIIRS (SNPP/NOAA-20/NOAA-21) -- download not wired up yet
    python l2/fetch_proxy_scene.py --instrument viirs

    # after looking at the printed table, download candidate index 0's
    # (closest-in-time, not necessarily best-geometry -- read the table)
    # geolocation + L1B radiance
    python l2/fetch_proxy_scene.py --download 0
"""
import argparse
import datetime
import glob
import os
import sys

import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Default: this repo's own root (one level up from l2/), so a fresh
# clone works out of the box with no manual --l2-cam-root. Override it
# only if you've relocated cam_l2/software/er3t/software/shdom elsewhere.
DEFAULT_L2_CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TILE_PICK = "tile_pick.yml"

MODIS_SATELLITES = ["aqua", "terra"]
VIIRS_SATELLITES = ["snpp", "noaa20", "noaa21"]

# MODIS band covering our ~555 nm camera channel, at the native (non-1km-
# aggregated) resolution -- libcam/modis.py's own band-group table shows
# band 4 (545-565 nm) lives in the 500 m "HKM" product (EV_500_RefSB), not
# the 250 m "QKM" one (bands 1-2 only). Collection 61 is current as of
# 2026-10 -- bump MODIS_COLLECTION if LAADS retires it.
MODIS_COLLECTION = "61"
MODIS_DATASET_L1B = MODIS_COLLECTION + "/{prefix}02HKM"
MODIS_DATASET_GEO = MODIS_COLLECTION + "/{prefix}03"
MODIS_BAND = 4


def _add_l2_cam_to_path(l2_cam_root):
    """Put L2a/l2_cam's vendored er3t on sys.path -- same convention as
    l2/run_tile_rt.py's rt_shdom(), so this script works whether or
    not er3t has also been `pip install -e`'d."""
    er3t_dir = os.path.join(l2_cam_root, "software", "er3t")
    if er3t_dir not in sys.path:
        sys.path.insert(0, er3t_dir)


def _parse_time_utc(s):
    """Parse tile_pick.yml's time_utc string. It is written with
    nanosecond-precision fractional seconds (e.g.
    '2014-07-12T03:28:22.611179000'), but Python's datetime only keeps
    microsecond precision and datetime.fromisoformat() does not accept
    9-digit fractions on every Python version -- truncate to 6 digits
    rather than relying on fromisoformat's version-specific leniency."""
    if "." in s:
        head, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        s = "%s.%s" % (head, frac)
    return datetime.datetime.fromisoformat(s)


def load_tile_pick(path):
    """Returns the tile_pick dict with lat/lon as float and time_utc
    replaced by a parsed datetime -- all other keys (sza_deg, vza_deg,
    surface_type, era5_windspeed_ms, ...) pass through unchanged, so
    callers can compare against the tile's own recorded geometry."""
    with open(path) as f:
        d = dict(yaml.safe_load(f)["tile_pick"])
    d["lat"] = float(d["lat"])
    d["lon"] = float(d["lon"])
    d["time_utc"] = _parse_time_utc(d["time_utc"])
    return d


def _tag_to_datetime(tag):
    """'A2026193.0325' (AYYYYDDD.HHMM, as returned by get_satfile_tag) ->
    datetime.datetime, or None if it doesn't parse."""
    try:
        doy_part, hhmm_part = tag.split(".")
        year = int(doy_part[1:5])
        doy = int(doy_part[5:8])
        hh = int(hhmm_part[:2])
        mm = int(hhmm_part[2:4])
        return (datetime.datetime(year, 1, 1)
                + datetime.timedelta(days=doy - 1, hours=hh, minutes=mm))
    except Exception:
        return None


def find_candidates(satellites, instrument, lat, lon, t_center, window_min,
                    verbose=True):
    """Search `satellites` (MODIS_SATELLITES or VIIRS_SATELLITES) for
    granules overlapping (lat, lon) within +/-window_min of t_center.

    Returns a list of dicts (satellite, instrument, filename_tag,
    granule_dt, offset_min), sorted by |offset_min| ascending (closest
    in time first). offset_min/granule_dt are None if the returned tag
    could not be parsed as a date (should not normally happen).
    """
    from er3t.util import get_satfile_tag

    date = datetime.datetime(t_center.year, t_center.month, t_center.day)
    win_start = t_center - datetime.timedelta(minutes=window_min)
    win_end = t_center + datetime.timedelta(minutes=window_min)

    results = []
    for sat in satellites:
        lon_arr = np.array([lon], dtype=np.float64)
        lat_arr = np.array([lat], dtype=np.float64)
        try:
            tags = get_satfile_tag(
                date, lon_arr, lat_arr,
                satellite=sat, instrument=instrument,
                start_dt_hhmm=win_start, end_dt_hhmm=win_end,
            )
        except Exception as exc:
            if verbose:
                print("  %s/%s: search failed (%r)" % (sat, instrument, exc))
            continue

        for tag in tags:
            granule_dt = _tag_to_datetime(tag)
            offset_min = (None if granule_dt is None else
                         (granule_dt - t_center).total_seconds() / 60.0)
            results.append(dict(satellite=sat, instrument=instrument,
                                filename_tag=tag, granule_dt=granule_dt,
                                offset_min=offset_min))

    results.sort(key=lambda r: (abs(r["offset_min"])
                                if r["offset_min"] is not None else 1.0e9))
    return results


def _download_or_find_cached(date, dataset_tag, filename_tag, file_prefix,
                             out_dir, verbose=True):
    """Wraps er3t.util.download_laads_https() (vendored in l2_cam, read-
    only -- not ours to change). Bug found 2026-10-23, Sebastian's first
    real runs: that function returns only the files it just freshly
    downloaded THIS call -- a file that already exists on disk and passes
    its own integrity check is silently skipped and left OUT of the
    returned list. That makes every caller that re-requests an
    already-cached file (which is every re-run of this pipeline against
    the same tile, once the files exist at all, AND the within-one-run
    case where the candidate-comparison loop below fetches a geo file
    that download_modis_candidate() then asks for again) see an empty
    list and wrongly conclude the download failed. Before giving up, glob
    out_dir directly for the expected filename pattern -- if it's there,
    it's a real cache hit, not a failure."""
    from er3t.util import download_laads_https

    files = download_laads_https(
        date, dataset_tag, filename_tag, fdir_out=out_dir, verbose=verbose)
    if files:
        return files

    pattern = os.path.join(out_dir, "%s.%s*.hdf" % (file_prefix, filename_tag))
    found = sorted(glob.glob(pattern))
    if found and verbose:
        print("Message [_download_or_find_cached]: download_laads_https() "
             "returned nothing new for %s, but found an already-cached "
             "match on disk: %s" % (dataset_tag, found))
    return found


def download_modis_geo(cand, out_dir, verbose=True):
    """Download just the geolocation file (MOD03/MYD03) for one MODIS
    candidate dict from find_candidates() -- small (a few MB), safe to
    fetch for every in-window candidate just to compare geometry."""
    prefix = "MYD" if cand["satellite"] == "aqua" else "MOD"
    geo_tag = MODIS_DATASET_GEO.format(prefix=prefix)
    return _download_or_find_cached(
        cand["granule_dt"], geo_tag, cand["filename_tag"],
        "%s03" % prefix, out_dir, verbose=verbose)


def download_modis_candidate(cand, out_dir, verbose=True):
    """Download geolocation (MOD03/MYD03) + 500 m L1B radiance
    (MOD02HKM/MYD02HKM) for one MODIS candidate dict from
    find_candidates(). Re-downloading the geolocation file is a no-op if
    download_modis_geo() already fetched it -- see
    _download_or_find_cached()'s docstring for why that used to look like
    a failure instead of a cache hit."""
    prefix = "MYD" if cand["satellite"] == "aqua" else "MOD"
    date = cand["granule_dt"]

    geo_tag = MODIS_DATASET_GEO.format(prefix=prefix)
    l1b_tag = MODIS_DATASET_L1B.format(prefix=prefix)

    geo_files = _download_or_find_cached(
        date, geo_tag, cand["filename_tag"], "%s03" % prefix,
        out_dir, verbose=verbose)
    l1b_files = _download_or_find_cached(
        date, l1b_tag, cand["filename_tag"], "%s02HKM" % prefix,
        out_dir, verbose=verbose)

    return geo_files, l1b_files


def geometry_at_tile(geo_fname, lat, lon):
    """Load a MOD03/MYD03 geolocation file and return the sensor/solar
    geometry at the pixel nearest (lat, lon): vza_deg, vaa_deg, sza_deg,
    dist_km. dist_km is a sanity check (flat-Earth approximation, fine
    at these scales) -- it should be small, since get_satfile_tag()
    already required the granule's footprint to contain this point; a
    large value here would mean something is wrong (e.g. a dateline/
    antimeridian wraparound) and the geometry below should not be
    trusted."""
    from libcam.modis import ModisGeo

    geo = ModisGeo(geo_fname)
    dlat = geo.lat.astype(np.float64) - lat
    dlon = (geo.lon.astype(np.float64) - lon) * np.cos(np.radians(lat))
    d2 = dlat * dlat + dlon * dlon
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)

    r_earth_km = 6371.0
    dist_km = r_earth_km * np.radians(float(np.sqrt(d2[iy, ix])))

    return dict(
        vza_deg=float(geo.vza[iy, ix]),
        vaa_deg=float(geo.vaa[iy, ix]),
        sza_deg=float(geo.sza[iy, ix]),
        dist_km=dist_km,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-pick", default=DEFAULT_TILE_PICK,
                  help="path to tile_pick.yml (default: %s, repo root)"
                       % DEFAULT_TILE_PICK)
    p.add_argument("--instrument", choices=["modis", "viirs"], default="modis",
                  help="which imager to search (default: modis)")
    p.add_argument("--window-min", type=float, default=45.0,
                  help="search window, +/- minutes around the tile's "
                       "time_utc (default: 45)")
    p.add_argument("--l2-cam-root", default=DEFAULT_L2_CAM_ROOT,
                  help="path to the L2a/l2_cam clone (for its vendored er3t)")
    p.add_argument("--download", type=int, default=None, metavar="INDEX",
                  help="download the candidate at this table index (0-based, "
                       "as printed) -- geolocation + L1B radiance (MODIS "
                       "only, for now). Omit to just print the comparison "
                       "table and pick later.")
    p.add_argument("--out-dir", default="l2/data/proxy_imagery",
                  help="where downloaded files go "
                       "(default: l2/data/proxy_imagery)")
    return p.parse_args()


def main():
    args = parse_args()
    _add_l2_cam_to_path(args.l2_cam_root)

    tile_pick = load_tile_pick(args.tile_pick)
    lat, lon, t_center = tile_pick["lat"], tile_pick["lon"], tile_pick["time_utc"]
    print("Tile pick (%s): lat=%.4f lon=%.4f time_utc=%s"
         % (args.tile_pick, lat, lon, t_center.isoformat()))
    tile_vza = tile_pick.get("vza_deg")
    tile_sza = tile_pick.get("sza_deg")
    if tile_vza is not None and tile_sza is not None:
        print("Tile's own recorded geometry: vza=%.2f deg  sza=%.2f deg "
             "(this is what we're trying to match)" % (tile_vza, tile_sza))

    satellites = MODIS_SATELLITES if args.instrument == "modis" else VIIRS_SATELLITES
    print("\nSearching %s (%s) within +/-%.0f min ..."
         % (args.instrument.upper(), ", ".join(satellites), args.window_min))

    candidates = find_candidates(satellites, args.instrument, lat, lon,
                                 t_center, args.window_min)

    if not candidates:
        print("No candidates found in this window -- try a larger "
             "--window-min, or --instrument viirs.")
        return

    if args.instrument == "modis":
        print("\nFetching geolocation (small) for each candidate to compare "
             "real viewing/solar geometry -- closer in time is not "
             "necessarily closer in viewing angle ...")
        os.makedirs(args.out_dir, exist_ok=True)
        for c in candidates:
            try:
                geo_files = download_modis_geo(c, args.out_dir, verbose=False)
                geo_file = geo_files[0] if geo_files else None
                if geo_file is None or not os.path.isfile(geo_file):
                    raise RuntimeError("no geolocation file returned/found")
                c.update(geometry_at_tile(geo_file, lat, lon))
                c["geo_file"] = geo_file
            except Exception as exc:
                print("  %s %s: could not fetch/read geolocation (%r)"
                     % (c["satellite"], c["filename_tag"], exc))
                c["vza_deg"] = c["sza_deg"] = c["vaa_deg"] = c["dist_km"] = None
                c["geo_file"] = None

    print("\n%d candidate granule(s):" % len(candidates))
    header = "  idx  sat     tag              overpass (UTC)        dt_min"
    if args.instrument == "modis":
        header += "   vza_deg  sza_deg  dist_km"
    print(header)
    for i, c in enumerate(candidates):
        off = c["offset_min"]
        off_str = "   n/a" if off is None else "%+6.1f" % off
        line = "  %-4d %-7s %-16s %-21s %s" % (
            i, c["satellite"], c["filename_tag"], c["granule_dt"], off_str)
        if args.instrument == "modis":
            if c.get("vza_deg") is not None:
                line += "   %6.2f   %6.2f   %6.1f" % (
                    c["vza_deg"], c["sza_deg"], c["dist_km"])
            else:
                line += "      n/a      n/a      n/a"
        print(line)

    if args.instrument == "modis":
        with_geom = [c for c in candidates if c.get("vza_deg") is not None]
        if with_geom and tile_vza is not None:
            best = min(with_geom, key=lambda c: abs(c["vza_deg"] - tile_vza))
            best_i = candidates.index(best)
            if best_i != 0:
                print("\nNote: index %d has the viewing angle closest to the "
                     "tile's own (vza=%.2f deg vs. target %.2f deg), even "
                     "though index 0 is closest in time -- your call which "
                     "matters more here." % (best_i, best["vza_deg"], tile_vza))

    if args.download is not None:
        if args.instrument != "modis":
            raise SystemExit(
                "--download is only wired up for --instrument modis so "
                "far -- VIIRS download is a follow-up.")
        if not (0 <= args.download < len(candidates)):
            raise SystemExit("--download %d out of range (0..%d)"
                             % (args.download, len(candidates) - 1))
        best = candidates[args.download]
        print("\nDownloading index %d: %s %s ..."
             % (args.download, best["satellite"], best["filename_tag"]))
        os.makedirs(args.out_dir, exist_ok=True)
        geo_files, l1b_files = download_modis_candidate(best, args.out_dir)
        print("  geolocation: %s" % geo_files)
        print("  L1B:         %s" % l1b_files)


if __name__ == "__main__":
    main()
