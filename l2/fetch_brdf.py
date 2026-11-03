"""
l2/fetch_brdf.py

Dev-step-2 BRDF-date-matching follow-up (Sebastian, 2026-10-22): the
land BRDF input every dev-step-1/2 run has used so far
(L2a/l2_cam/data/sample/brdf_parms/VJ143C1.A2021138.002....h5) is Hong's
own static demo file, dated May 2021 -- a different year AND season
than this tile's July MODIS proxy scene, and a likely major contributor
to the land-surface brightness mismatch seen in _measured_vs_rt.png.

Rather than chase an exact-date match (impossible anyway: NOAA-20/
VIIRS, which VJ143C1 comes from, didn't launch until Nov 2017, so no
real VJ143C1 file exists for the MODIS proxy scene's own 2014-07-12
date), the agreed plan is: re-fetch the MODIS proxy scene from a
DIFFERENT, more recent year (Sebastian's call: 2020) on the same
calendar day (07-12) -- see tile_pick_2020.yml -- then pull the VJ143C1
file from THAT SAME year/day, so the two real-world inputs (MODIS
proxy scene, BRDF) agree on year and season with each other, even
though neither matches the camera's own (synthetic, future) 2026 L1B
date -- that mismatch is unavoidable and accepted.

**2026-10-23 update (Sebastian): wired into run_full_pipeline.py by
default.** `fetch_brdf_for_date()` below is now called automatically
for every --proxy-date run whose date is on/after VIIRS/JPSS-1's
2017-11-18 launch (run_full_pipeline.py checks the date itself before
calling in -- this module doesn't need to know about that cutoff) --
no more need to manually find/pass --brdf-land by hand for the normal
case. It checks l2/data/brdf_test/ for an already-downloaded file
for that exact calendar date FIRST (a plain filename glob, no network
call at all) before ever querying CMR, so a repeat run for a date
already fetched costs nothing extra -- same "don't re-download what's
already on disk" pattern fetch_proxy_scene.py's
_download_or_find_cached() uses for the MODIS proxy imagery itself.
--brdf-land (or --no-brdf-auto-fetch) on run_full_pipeline.py still
overrides/disables this. The old pre-VIIRS-launch "different year,
same calendar day" substitution plan above is NOT automated by this --
that's still a deliberate, by-hand choice (pick which substitute year)
left to run_tile_rt.py's existing FATAL ancillary-date check and
--allow-ancillary-date-mismatch.

REVISION HISTORY / why this doesn't just call download_laads_https()
like fetch_proxy_scene.py does for MODIS: the first version of this
script guessed a classic ladsweb.modaps.eosdis.nasa.gov archive path
("002/VJ143C1") the same way fetch_proxy_scene.py's MODIS_DATASET_GEO
is built -- that failed (the "CSV listing" it got back was actually an
HTML error page). Sebastian then confirmed via a direct NASA CMR query
(`curl ".../search/granules.json?short_name=VJ143C1&version=002&..."`)
that VJ143C1 is served through LPCLOUD/Earthdata Cloud instead, at
URLs like
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/VJ143C1.002/<granule>/<granule>.h5
-- a completely different shape than the classic archive/allData tree.
So instead of guessing a URL shape, find_granule() below asks CMR
directly for the real, current URL (DAAC-agnostic, always correct) and
hands it to er3t's existing get_command_earthdata() -- which already
knows how to build a properly-authenticated curl/wget command for ANY
target URL via the same EARTHDATA_TOKEN env var fetch_proxy_scene.py's
download_laads_https() already uses (NASA's Earthdata Login token
system covers both the classic LAADS archive and the newer Earthdata
Cloud "protected" buckets the same way). No new credential setup
needed if EARTHDATA_TOKEN is already in your environment.

VJ143C1 (NOAA-20 VIIRS BRDF/Albedo model parameters) is global, daily,
0.05-deg CMG -- one file per day, no lat/lon search needed, unlike
MODIS L1B. Confirmed from the real CMR response: the day-of-year in the
filename IS the UTC day the file actually represents (not shifted) --
e.g. day 194 of 2020 (a leap year) is 2020-07-12.

Needs EARTHDATA_TOKEN set in your environment, same as
fetch_proxy_scene.py. Downloads are substantial (~340 MB, a global
grid) -- this is a single whole-Earth file, not a small regional crop.

Usage
-----
    python l2/fetch_brdf.py --date 2020-07-12
"""
import argparse
import datetime
import glob
import os
import sys

import requests

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Default: this repo's own root (one level up from l2/), so a fresh
# clone works out of the box with no manual --l2-cam-root. Override it
# only if you've relocated cam_l2/software/er3t/software/shdom elsewhere.
DEFAULT_L2_CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = "l2/data/brdf_test"
DEFAULT_SHORT_NAME = "VJ143C1"
DEFAULT_VERSION = "002"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# VIIRS/JPSS-1 (NOAA-20), which VJ143C1 comes from, launched 2017-11-18 --
# no real VJ143C1 granule can exist for any earlier date. Kept here (not
# just in run_tile_rt.py's own copy of this same cutoff) since this is
# the module that actually knows what VJ143C1 is and where it comes
# from; run_full_pipeline.py imports this one rather than reaching into
# run_tile_rt.py's private constant.
VIIRS_JPSS1_LAUNCH_DATE = datetime.date(2017, 11, 18)


def _add_l2_cam_to_path(l2_cam_root):
    """Put L2a/l2_cam's vendored er3t on sys.path -- same convention
    fetch_proxy_scene.py and run_tile_rt.py's rt_shdom() already use."""
    er3t_dir = os.path.join(l2_cam_root, "software", "er3t")
    if er3t_dir not in sys.path:
        sys.path.insert(0, er3t_dir)


def find_cached_for_date(date, short_name=DEFAULT_SHORT_NAME, out_dir=DEFAULT_OUT_DIR):
    """Look for an already-downloaded `short_name` granule for `date` (a
    datetime.date) under out_dir, WITHOUT querying CMR or touching the
    network. The granule's day-of-year is baked into its filename
    (<short_name>.A<year><doy>.<version>.<proc_time>.h5), so a plain
    glob is enough to recognize a file we already have -- this is what
    lets a repeated run for the same date cost nothing beyond a
    directory listing, same as fetch_proxy_scene.py's own
    _download_or_find_cached() glob fallback for MODIS imagery.

    Returns the local path of the first match, or None if nothing's
    there yet. Does not re-verify file integrity byte-for-byte (same as
    that MODIS-side glob fallback) -- if a previous download left a
    corrupt partial file behind, delete it by hand to force a refetch.
    """
    doy = date.timetuple().tm_yday
    pattern = os.path.join(out_dir, "%s.A%04d%03d.*.h5" % (short_name, date.year, doy))
    found = sorted(glob.glob(pattern))
    return found[0] if found else None


def find_granule(date, short_name=DEFAULT_SHORT_NAME, version=DEFAULT_VERSION,
                 verbose=True):
    """Query NASA's CMR (DAAC-agnostic metadata catalog) for the one
    global `short_name`/`version` granule covering `date` (a
    datetime.date), and return (download_url, producer_granule_id,
    granule_size_mb_str).

    CMR's own `temporal` filter is an OVERLAP test, not an exact-day
    one -- confirmed in practice: querying 2020-07-12 also returned
    2020-07-11's granule, because that granule's own time_end
    (2020-07-12T00:00:00Z) exactly touches the query window's start.
    So this filters the response down to entries whose OWN time_start
    actually falls on `date`, which for a true daily product should
    always leave exactly one."""
    day_start = datetime.datetime(date.year, date.month, date.day)
    day_end = day_start + datetime.timedelta(days=1)
    params = {
        "short_name": short_name,
        "version": version,
        "temporal": "%sZ,%sZ" % (day_start.isoformat(), day_end.isoformat()),
    }
    if verbose:
        print("Querying CMR for %s v%s on %s ..."
             % (short_name, version, date.isoformat()))
    resp = requests.get(CMR_GRANULES_URL, params=params, timeout=30)
    resp.raise_for_status()
    entries = resp.json()["feed"].get("entry", [])

    same_day = [e for e in entries
               if e.get("time_start", "").startswith(date.isoformat())]
    candidates = same_day or entries
    if not candidates:
        raise SystemExit("No %s v%s granule found for %s via CMR -- check "
                         "the date, short_name, and version."
                         % (short_name, version, date.isoformat()))
    if len(candidates) > 1:
        raise SystemExit(
            "CMR returned %d candidates for %s v%s on %s (expected "
            "exactly 1 -- it's a daily global product) -- inspect "
            "manually:\n  %s"
            % (len(candidates), short_name, version, date.isoformat(),
               "\n  ".join(e["producer_granule_id"] for e in candidates)))

    entry = candidates[0]
    data_links = [l["href"] for l in entry.get("links", [])
                 if l.get("rel", "").endswith("/data#")
                 and not l.get("inherited")]
    if not data_links:
        raise SystemExit("Found granule %s but no direct (non-inherited) "
                         "download link in its CMR record -- inspect "
                         "manually at https://cmr.earthdata.nasa.gov/"
                         "search/granules.json?short_name=%s&version=%s"
                         % (entry["producer_granule_id"], short_name, version))
    return data_links[0], entry["producer_granule_id"], entry.get("granule_size")


def fetch_brdf_for_date(date, short_name=DEFAULT_SHORT_NAME, version=DEFAULT_VERSION,
                        out_dir=DEFAULT_OUT_DIR, l2_cam_root=DEFAULT_L2_CAM_ROOT,
                        verbose=True):
    """Return a local path to a real `short_name` (default VJ143C1) BRDF
    file for `date` (a datetime.date) -- the function callers (added
    2026-10-23, run_full_pipeline.py's default BRDF-auto-fetch step)
    should use instead of hardcoding a --brdf-land path by hand.

    Checks find_cached_for_date() FIRST (no network at all) and only
    falls through to a CMR lookup + download when nothing's cached yet,
    so a repeated run for an already-fetched date is effectively free.

    Soft-fails: returns None (after printing why), rather than raising,
    on any lookup or download problem -- a date with no real BRDF
    available yet (CMR down, no EARTHDATA_TOKEN, no granule published)
    should fall back to l2_cam's own hardcoded sample BRDF file, not
    abort the whole pipeline run. Use the standalone `python
    fetch_brdf.py --date ...` CLI instead (main(), below) when you want
    a hard failure on error.
    """
    cached = find_cached_for_date(date, short_name, out_dir)
    if cached is not None:
        if verbose:
            print("fetch_brdf_for_date: found already-downloaded %s granule "
                 "for %s (no download needed): %s"
                 % (short_name, date.isoformat(), cached))
        return cached

    try:
        _add_l2_cam_to_path(l2_cam_root)
        from er3t.util import get_command_earthdata, final_file_check
        url, granule_id, size_mb = find_granule(date, short_name, version,
                                                verbose=verbose)
    except Exception as exc:
        if verbose:
            print("fetch_brdf_for_date: could not look up a %s granule for "
                 "%s (%r) -- falling back to l2_cam's own hardcoded sample "
                 "BRDF file" % (short_name, date.isoformat(), exc))
        return None

    if verbose:
        print("fetch_brdf_for_date: found granule %s (~%s MB) -- downloading "
             "(a whole-Earth grid -- may take a while) ..."
             % (granule_id, size_mb))
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(url)
    fname_local = os.path.join(out_dir, filename)
    try:
        primary_cmd, backup_cmd = get_command_earthdata(
            url, filename=filename, fdir_save=out_dir, verbose=verbose)
        os.system(primary_cmd)
        if not final_file_check(fname_local, "h5", verbose=verbose):
            if verbose:
                print("fetch_brdf_for_date: primary (curl) download missing/"
                     "bad -- trying backup (wget) ...")
            os.system(backup_cmd)
    except Exception as exc:
        if verbose:
            print("fetch_brdf_for_date: download attempt raised %r -- "
                 "falling back to l2_cam's own hardcoded sample BRDF file"
                 % exc)
        return None

    if not final_file_check(fname_local, "h5", verbose=verbose):
        if verbose:
            print("fetch_brdf_for_date: download failed for %s (checked "
                 "EARTHDATA_TOKEN is set and valid?) -- falling back to "
                 "l2_cam's own hardcoded sample BRDF file" % date.isoformat())
        return None

    if verbose:
        print("fetch_brdf_for_date: success: %s" % fname_local)
    return fname_local


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", required=True,
                  help="YYYY-MM-DD, e.g. 2020-07-12 (the actual UTC day "
                       "this VJ143C1 granule represents)")
    p.add_argument("--short-name", default=DEFAULT_SHORT_NAME,
                  help="CMR short_name (default: %s)" % DEFAULT_SHORT_NAME)
    p.add_argument("--version", default=DEFAULT_VERSION,
                  help="product version (default: %s)" % DEFAULT_VERSION)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                  help="where the downloaded file goes (default: %s)"
                       % DEFAULT_OUT_DIR)
    p.add_argument("--l2-cam-root", default=DEFAULT_L2_CAM_ROOT,
                  help="path to the L2a/l2_cam clone (for its vendored er3t)")
    return p.parse_args()


def main():
    """Standalone CLI -- unlike fetch_brdf_for_date(), fails loudly
    (SystemExit) rather than returning None, since a human ran this by
    hand specifically to get this file."""
    args = parse_args()
    date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    result = fetch_brdf_for_date(date, short_name=args.short_name,
                                 version=args.version, out_dir=args.out_dir,
                                 l2_cam_root=args.l2_cam_root, verbose=True)
    if result is None:
        raise SystemExit(
            "Could not get a %s v%s file for %s -- see the messages above "
            "(check EARTHDATA_TOKEN is set and valid, and that CMR/"
            "Earthdata Cloud are reachable)."
            % (args.short_name, args.version, date.isoformat()))


if __name__ == "__main__":
    main()
