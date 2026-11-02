"""
apps/l2/run_full_pipeline.py

Dev-step-1, "one call" wrapper (Sebastian, 2026-10-23, item B of the
2026-10-23 next-steps list: "streamline dev-step-1 into one call").

Today this needs four separate invocations, with a human reading each
one's printed output and typing a path or an index into the next:

    python apps/l2/fetch_proxy_scene.py                  # look at the table
    python apps/l2/fetch_proxy_scene.py --download N     # then download N
    python apps/l2/build_l1x.py --modis-geo ... --modis-l1b ...
    python apps/l2/build_input_manifest.py --cam-l1b <the L1X file>
    python apps/l2/run_dev_step1.py --input-manifest <the new manifest>

This script runs all four steps in one call, in-process (importing each
script as a module and calling its own functions directly -- not
subprocess, and not a reimplementation of any of their logic), so the
things a human would otherwise have to read off one script's stdout and
retype into the next (the chosen MODIS candidate's downloaded file
paths, the L1X path, the new manifest's path) are passed as plain
Python values instead:

  1. fetch_proxy_scene.py's find_candidates()/download_modis_geo() to
     search LAADS and compare each in-window candidate's REAL viewing
     geometry (not just overpass time) against the tile pick's own
     recorded vza_deg -- then download_modis_candidate() the chosen one
     (geolocation + 500 m L1B radiance).

     fetch_proxy_scene.py's own --download INDEX step is semi-
     interactive: it always prints the full ranked table, and only
     *recommends* an index (the geometry-closest one) rather than acting
     on it. This wrapper auto-picks that exact same recommended index by
     default -- same rule, lifted verbatim from fetch_proxy_scene.main()
     ("closest in viewing angle to the tile's own vza_deg") -- so the
     whole pipeline runs with no human in the loop. The table is still
     printed every time, so if the auto-pick isn't the one you want,
     rerun with --download-index to override it.

  2. build_l1x.build_l1x() -- splice that candidate's proxy radiance
     into a cropped copy of the REAL camera L1B (nadir stripe only, by
     default: see --components), producing an "L1X" file. Every other
     pixel keeps its original L1B value untouched (see build_l1x.py's
     own module docstring).

  3. build_input_manifest.py's own Manifest-building logic (its checksum
     caching included, not reimplemented) -- write a fresh input
     manifest whose "cam_l1b" role points at that L1X file, not the real
     camera L1B, while "rad_l1b" and "era5" stay the same real files
     every other manifest uses (--rad-l1b/--era5 below -- these are also
     exactly what build_l1x needs for its own camera/radiometer time
     matching, so this script takes them once and uses them for both).

  4. run_dev_step1.main() -- run the dev-step-1 tile pipeline against
     that fresh manifest, exactly as if --input-manifest had been typed
     by hand.

Any flag this script does not itself define (--write-l2-output, --plot,
--max-vza-deg, --dx-km, --dy-km, ... -- the whole of run_dev_step1.py's
own argument list) is passed straight through to run_dev_step1.main()
unexamined. --input-manifest and --manifest-dir are the one exception:
this script always builds and passes its own fresh manifest, so passing
either of those yourself is rejected with an explanation rather than
silently overridden.

Proxy date, independent of the camera L1B baseline date (Sebastian,
2026-10-23)
----------------------------------------------------------------------
The camera/radiometer L1B pair (--cam-l1b/--rad-l1b) is the fixed real
orbit baseline -- July 2026 at the time of writing, and that doesn't
change run to run. What SHOULD be freely configurable is which day's
MODIS scene gets spliced in as the proxy radiance -- e.g. to try an
actually cloudy scene from a different date over the same ground
footprint. --proxy-date does exactly that: it overrides only the time
center used for step 1's MODIS search (lat/lon and the vza/sza used for
candidate matching still come from the tile pick, i.e. from the real
camera geometry -- only the calendar day being searched changes).

Every run is tagged with a "proxy tag" (the proxy search time, or the
tile pick's own time_utc when --proxy-date isn't given, as
YYYYMMDDTHHMM) that is: (a) folded into the default L1X filename and the
default run_dev_step1 --out/--l2-output-dir, so different proxy-date
runs don't overwrite each other's outputs, and (b) appended as one JSON
line to apps/l2/data/proxy_run_log.jsonl -- a running ledger of every
proxy date processed, which MODIS candidate was chosen for it, and
where its L1X/manifest/outputs ended up, so multiple dates processed at
different times stay traceable.

Usage
-----
    # fully automatic: search, auto-pick by geometry, download, splice,
    # build a fresh manifest, run dev-step-1 with QA plots + L2 output
    python apps/l2/run_full_pipeline.py --write-l2-output --plot

    # same, but splice in a MODIS scene from a different day (the real
    # camera/radiometer L1B baseline is unchanged) -- note the wider
    # default search window when no time-of-day is given
    python apps/l2/run_full_pipeline.py --proxy-date 2026-09-03 --plot

    # a specific time too, and a tighter search window
    python apps/l2/run_full_pipeline.py \\
        --proxy-date 2026-09-03T18:40:00 --search-window-min 45 --plot

    # override the auto-picked MODIS candidate (see the printed table)
    python apps/l2/run_full_pipeline.py --download-index 2 --plot

    # anything run_dev_step1.py itself accepts also works here, e.g.
    python apps/l2/run_full_pipeline.py --plot --max-vza-deg 15 --niter 20
"""
import argparse
import datetime
import glob
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import fetch_proxy_scene as fps            # noqa: E402
import build_l1x as bl1x                   # noqa: E402
import build_input_manifest as bim         # noqa: E402
import inspect_proxy_scene as ips          # noqa: E402
import proxy_radiance as pr                # noqa: E402
import run_dev_step1                       # noqa: E402
import fetch_brdf                          # noqa: E402

REJECTED_PASSTHROUGH_FLAGS = ("--input-manifest", "--manifest-dir")
RUN_LOG_FILENAME = "proxy_run_log.jsonl"

# No time-of-day given in --proxy-date (a bare YYYY-MM-DD) -> we don't know
# the actual overpass time, so search a much wider window by default than
# the tile-pick-time case (where the camera's own exposure time is exact).
DEFAULT_SEARCH_WINDOW_MIN_WITH_TIME = 45.0
DEFAULT_SEARCH_WINDOW_MIN_DATE_ONLY = 720.0


def parse_proxy_date(s):
    """Parse --proxy-date: a bare 'YYYY-MM-DD' date, or a full ISO
    datetime (optionally with a trailing 'Z' or a UTC offset, which is
    converted to naive UTC to match tile_pick.yml's own convention).
    Returns (datetime, had_time) -- had_time is False for a bare date,
    telling the caller the overpass time-of-day is unknown."""
    s = s.strip()
    if len(s) == 10 and s.count("-") == 2:
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%d"), False
        except ValueError:
            pass
    s2 = s[:-1] + "+00:00" if s.endswith("Z") else s
    dt = datetime.datetime.fromisoformat(s2)
    if dt.tzinfo is not None:
        dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return dt, True


def pick_candidate(candidates, tile_vza, download_index=None):
    """Same selection fetch_proxy_scene.main() prints as a "Note:" and
    leaves for a human to act on via --download -- lifted verbatim so
    this wrapper can act on it automatically. Returns (index, reason)."""
    if download_index is not None:
        if not (0 <= download_index < len(candidates)):
            raise SystemExit("--download-index %d out of range (0..%d)"
                             % (download_index, len(candidates) - 1))
        return download_index, "explicit --download-index"

    with_geom = [c for c in candidates if c.get("vza_deg") is not None]
    if with_geom and tile_vza is not None:
        best = min(with_geom, key=lambda c: abs(c["vza_deg"] - tile_vza))
        return candidates.index(best), ("geometry-closest to the tile's own "
                                        "vza=%.2f deg (auto)" % tile_vza)
    return 0, "closest-in-time (no usable geometry -- auto fallback to index 0)"


def find_and_download_candidate(args):
    """Steps 1: resolve the proxy search time (tile pick's own time_utc,
    or --proxy-date if given), search, print the table (same as
    fetch_proxy_scene.py), pick (auto or --download-index), download.
    Returns (geo_file, l1b_file, tile_pick_dict, t_center, had_time,
    chosen_candidate_dict)."""
    fps._add_l2_cam_to_path(args.l2_cam_root)

    tile_pick = fps.load_tile_pick(args.tile_pick)
    lat, lon = tile_pick["lat"], tile_pick["lon"]
    print("Tile pick (%s): lat=%.4f lon=%.4f time_utc=%s (real camera "
         "baseline -- --cam-l1b/--rad-l1b, unaffected by --proxy-date)"
         % (args.tile_pick, lat, lon, tile_pick["time_utc"].isoformat()))
    tile_vza = tile_pick.get("vza_deg")
    tile_sza = tile_pick.get("sza_deg")
    if tile_vza is not None and tile_sza is not None:
        print("Tile's own recorded geometry: vza=%.2f deg  sza=%.2f deg "
             "(this is what we're trying to match, regardless of which day "
             "the proxy scene comes from)" % (tile_vza, tile_sza))

    if args.proxy_date:
        t_center, had_time = parse_proxy_date(args.proxy_date)
        print("--proxy-date override: searching for MODIS granules near %s "
             "(NOT the tile pick's own time_utc above) at the same lat/lon"
             % t_center.isoformat())
    else:
        t_center, had_time = tile_pick["time_utc"], True

    search_window_min = args.search_window_min
    if search_window_min is None:
        search_window_min = (DEFAULT_SEARCH_WINDOW_MIN_WITH_TIME if had_time
                             else DEFAULT_SEARCH_WINDOW_MIN_DATE_ONLY)
        print("(no --search-window-min given -- defaulting to +/-%.0f min "
             "since %s)" % (search_window_min,
                           "a time-of-day was given" if had_time
                           else "only a bare date was given, so the overpass "
                                "time-of-day is unknown"))

    if args.instrument != "modis":
        raise SystemExit(
            "--instrument viirs: download is only wired up for MODIS so far "
            "(same limitation as fetch_proxy_scene.py) -- VIIRS download is "
            "a follow-up.")

    satellites = fps.MODIS_SATELLITES
    print("\nSearching MODIS (%s) within +/-%.0f min of %s ..."
         % (", ".join(satellites), search_window_min, t_center.isoformat()))
    candidates = fps.find_candidates(satellites, "modis", lat, lon, t_center,
                                     search_window_min)
    if not candidates:
        raise SystemExit(
            "No candidates found in this window -- rerun with a larger "
            "--search-window-min (or double-check --proxy-date).")

    print("\nFetching geolocation (small) for each candidate to compare "
         "real viewing/solar geometry -- closer in time is not necessarily "
         "closer in viewing angle ...")
    os.makedirs(args.proxy_out_dir, exist_ok=True)
    for c in candidates:
        try:
            geo_files = fps.download_modis_geo(c, args.proxy_out_dir, verbose=False)
            geo_file = geo_files[0] if geo_files else None
            if geo_file is None or not os.path.isfile(geo_file):
                raise RuntimeError("no geolocation file returned/found")
            c.update(fps.geometry_at_tile(geo_file, lat, lon))
            c["geo_file"] = geo_file
        except Exception as exc:
            print("  %s %s: could not fetch/read geolocation (%r)"
                 % (c["satellite"], c["filename_tag"], exc))
            c["vza_deg"] = c["sza_deg"] = c["vaa_deg"] = c["dist_km"] = None
            c["geo_file"] = None

    print("\n%d candidate granule(s):" % len(candidates))
    print("  idx  sat     tag              overpass (UTC)        dt_min"
         "   vza_deg  sza_deg  dist_km")
    for i, c in enumerate(candidates):
        off = c["offset_min"]
        off_str = "   n/a" if off is None else "%+6.1f" % off
        line = "  %-4d %-7s %-16s %-21s %s" % (
            i, c["satellite"], c["filename_tag"], c["granule_dt"], off_str)
        if c.get("vza_deg") is not None:
            line += "   %6.2f   %6.2f   %6.1f" % (
                c["vza_deg"], c["sza_deg"], c["dist_km"])
        else:
            line += "      n/a      n/a      n/a"
        print(line)

    idx, reason = pick_candidate(candidates, tile_vza, args.download_index)
    best = candidates[idx]
    print("\nPicked index %d (%s): %s %s"
         % (idx, reason, best["satellite"], best["filename_tag"]))

    geo_files, l1b_files = fps.download_modis_candidate(best, args.proxy_out_dir)
    print("  geolocation: %s" % geo_files)
    print("  L1B:         %s" % l1b_files)
    if not geo_files or not l1b_files:
        raise SystemExit("download_modis_candidate() returned no file(s) -- "
                         "check EARTHDATA_TOKEN and the printed error above.")
    return geo_files[0], l1b_files[0], tile_pick, t_center, had_time, best


def build_l1x_file(args, modis_geo, modis_l1b, tile_pick, proxy_tag):
    """Step 2: same work as build_l1x.py's own main(), minus the argparse
    and the --plot display. The default output path (when --l1x-out isn't
    given) has proxy_tag folded in, so different --proxy-date runs don't
    overwrite each other's L1X file. Returns the written L1X path."""
    lat0, lon0 = float(tile_pick["lat"]), float(tile_pick["lon"])

    print("\nLoading MODIS crop (%s, %s) ..."
         % (os.path.basename(modis_geo), os.path.basename(modis_l1b)))
    modis = ips.inspect_crop(modis_geo, modis_l1b, lat0, lon0,
                             args.proxy_tile_km, band=args.band, verbose=True)

    lon_min, lon_max, lat_min, lat_max = ips.tile_bbox(lat0, lon0, args.proxy_tile_km)
    src_lon, src_lat, src_rad = pr.tighten_source_crop(
        modis, lon_min, lon_max, lat_min, lat_max)

    components = tuple(c.strip() for c in args.components.split(",") if c.strip())

    ds_out, cam_rad_by_frame, match, combined = bl1x.build_l1x(
        args.cam_l1b, args.rad_l1b, lat0, lon0, args.window_s, args.scan_pad_s,
        (lon_min, lon_max, lat_min, lat_max), src_lon, src_lat, src_rad,
        components=components, table_dir=args.table_dir)
    ds_out.attrs["l1x_tile_lat"] = lat0
    ds_out.attrs["l1x_tile_lon"] = lon0
    ds_out.attrs["l1x_tile_pick_path"] = os.path.abspath(args.tile_pick)
    ds_out.attrs["l1x_proxy_tag"] = proxy_tag
    ds_out.attrs["l1x_proxy_date_arg"] = args.proxy_date or ""
    ds_out.attrs["l1x_modis_source_sza_mean_deg"] = (
        float(modis["sza_mean"]) if modis.get("sza_mean") is not None
        else float("nan"))

    if args.l1x_out:
        out_path = args.l1x_out
    else:
        base = bl1x.l1x_path_for(args.cam_l1b)
        stem, ext = os.path.splitext(base)
        out_path = "%s_PROXY%s%s" % (stem, proxy_tag, ext)
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print("\nWrote L1X: %s" % out_path)
    return out_path


def build_manifest(args, l1x_path):
    """Step 3: same work as build_input_manifest.py's own main() (its
    checksum caching included), except cam_l1b is the just-built L1X
    file instead of the real camera L1B. Returns the written manifest
    path."""
    from libera_utils import Manifest
    from libera_utils.io.manifest import ManifestType

    files = [("camera L1X", l1x_path), ("radiometer L1B", args.rad_l1b),
             ("ERA5", args.era5)]
    if args.brdf_land:
        files.append(("BRDF land (VJ143C1)", args.brdf_land))
    for label, path in files:
        if not os.path.exists(path):
            raise FileNotFoundError("%s file not found: %s" % (label, path))
        print("  %-16s %s (%.2f GB)" % (label + ":", path, os.path.getsize(path) / 1e9))

    os.makedirs(args.manifest_out_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.manifest_out_dir,
                                        "LIBERA_INPUT_MANIFEST_*.json")))

    print("\nComputing checksums (chunked, cached -- see "
         "build_input_manifest.py's module docstring) ...")
    cache_path = os.path.join(args.manifest_out_dir, bim.CACHE_FILENAME)
    cache = {} if args.no_checksum_cache else bim.load_cache(cache_path)
    file_dicts = []
    for label, path in files:
        abspath = os.path.abspath(path)
        checksum = bim.cached_checksum(abspath, cache, label=label)
        file_dicts.append({"filename": abspath, "checksum": checksum})
    bim.save_cache(cache_path, cache)

    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=file_dicts)
    input_manifest.write(out_path=args.manifest_out_dir)

    after = set(glob.glob(os.path.join(args.manifest_out_dir,
                                       "LIBERA_INPUT_MANIFEST_*.json")))
    new_files = sorted(after - before)
    if not new_files:
        raise RuntimeError(
            "Manifest.write() didn't produce a new LIBERA_INPUT_MANIFEST_*.json "
            "under %s (couldn't isolate the new filename -- list the dir)"
            % args.manifest_out_dir)
    print("\nWrote input manifest: %s" % new_files[0])
    return new_files[0]


def append_run_log(args, proxy_tag, t_center, had_time, chosen, l1x_path,
                   manifest_path, dev_step1_out):
    """Append one JSON line to apps/l2/data/proxy_run_log.jsonl -- a
    running ledger of every proxy date processed (which MODIS candidate
    was picked, where its L1X/manifest/output ended up), so multiple
    dates processed at different times stay traceable. Best-effort: a
    logging failure should never fail the pipeline run itself."""
    log_path = os.path.join(bim.DEFAULT_OUT_DIR, RUN_LOG_FILENAME)
    record = dict(
        run_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        proxy_tag=proxy_tag,
        proxy_date_arg=args.proxy_date,
        proxy_t_center_utc=t_center.isoformat(),
        proxy_time_of_day_known=had_time,
        tile_pick=os.path.abspath(args.tile_pick),
        candidate=dict(
            satellite=chosen.get("satellite"),
            filename_tag=chosen.get("filename_tag"),
            granule_dt=str(chosen.get("granule_dt")),
            vza_deg=chosen.get("vza_deg"), sza_deg=chosen.get("sza_deg"),
            dist_km=chosen.get("dist_km")),
        cam_l1b=os.path.abspath(args.cam_l1b),
        rad_l1b=os.path.abspath(args.rad_l1b),
        era5=os.path.abspath(args.era5),
        components=args.components,
        brdf_land=os.path.abspath(args.brdf_land) if args.brdf_land else None,
        l1x_path=os.path.abspath(l1x_path),
        manifest_path=os.path.abspath(manifest_path),
        dev_step1_out=os.path.abspath(dev_step1_out) if dev_step1_out else None,
    )
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print("\nAppended run record: %s" % log_path)
    except Exception as exc:
        print("\n(could not append to %s: %r -- continuing anyway)"
             % (log_path, exc))
    return log_path


def _has_flag(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _flag_value(argv, flag):
    """Value of the LAST occurrence of --flag VALUE or --flag=VALUE in
    argv, or None if not present."""
    val = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(flag + "="):
            val = a[len(flag) + 1:]
    return val


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-pick", default=fps.DEFAULT_TILE_PICK,
                  help="used for all four steps (default: %(default)s)")
    p.add_argument("--l2-cam-root", default=fps.DEFAULT_L2_CAM_ROOT,
                  help="used for all four steps (default: %(default)s)")
    p.add_argument("--instrument", choices=["modis", "viirs"], default="modis")
    p.add_argument("--proxy-date", default=None, metavar="DATE",
                  help="step 1: override the date (and optionally time) "
                       "searched for the MODIS proxy scene -- 'YYYY-MM-DD' "
                       "or a full ISO datetime, e.g. "
                       "'2026-09-03T18:40:00'. Independent of the real "
                       "camera/radiometer L1B baseline (--cam-l1b/"
                       "--rad-l1b) -- lets you splice in a DIFFERENT day's "
                       "scene (e.g. an actually cloudy one) onto the same "
                       "real camera geometry. Default: use the tile pick's "
                       "own time_utc (today's behavior).")
    p.add_argument("--search-window-min", type=float, default=None,
                  help="step 1: +/- minutes around the proxy search time "
                       "to look for MODIS granules. Default: %.0f when the "
                       "search time has an explicit time-of-day (tile "
                       "pick's own time_utc, or a full --proxy-date "
                       "datetime), %.0f when --proxy-date is a bare date "
                       "(overpass time-of-day unknown)."
                       % (DEFAULT_SEARCH_WINDOW_MIN_WITH_TIME,
                          DEFAULT_SEARCH_WINDOW_MIN_DATE_ONLY))
    p.add_argument("--download-index", type=int, default=None, metavar="INDEX",
                  help="step 1: override the auto-picked (geometry-closest) "
                       "candidate -- see the printed table")
    p.add_argument("--proxy-out-dir", default="apps/l2/data/proxy_imagery",
                  help="step 1: where downloaded MODIS files go "
                       "(default: %(default)s)")
    p.add_argument("--cam-l1b", default=pr.DEFAULT_CAM_L1B,
                  help="step 2: the REAL camera L1B to crop/splice into -- "
                       "the fixed orbit baseline, NOT affected by "
                       "--proxy-date (default: %(default)s)")
    p.add_argument("--rad-l1b", default=pr.DEFAULT_RAD_L1B,
                  help="steps 2+3: the REAL radiometer L1B, used for time "
                       "matching in step 2 and written as-is into the new "
                       "manifest in step 3 -- fixed orbit baseline, NOT "
                       "affected by --proxy-date (default: %(default)s)")
    p.add_argument("--era5", default=bim.DEFAULT_ERA5,
                  help="step 3: written as-is into the new manifest "
                       "(default: %(default)s)")
    p.add_argument("--brdf-land", default=None,
                  help="step 3: explicit real, date-matched VJ143C1 BRDF "
                       "file to use -- optional, see build_input_manifest.py. "
                       "Normally not needed: by default (see "
                       "--no-brdf-auto-fetch) step 2 auto-fetches/caches the "
                       "correct VJ143C1 file for the proxy date itself. "
                       "Given explicitly, this disables that auto-fetch and "
                       "is used as-is, same as before 2026-10-23.")
    p.add_argument("--no-brdf-auto-fetch", action="store_true",
                  help="step 2: disable the default auto-fetch (added "
                       "2026-10-23) of a real, date-matched VJ143C1 land "
                       "BRDF file for the proxy date -- falls back to "
                       "l2_cam's own hardcoded (stale, 2021-05-18) sample "
                       "BRDF file instead, same as before this feature "
                       "existed. Ignored if --brdf-land is given (that "
                       "already disables auto-fetch on its own).")
    p.add_argument("--window-s", type=float, default=pr.DEFAULT_WINDOW_S)
    p.add_argument("--scan-pad-s", type=float, default=pr.DEFAULT_SCAN_PAD_S)
    p.add_argument("--proxy-tile-km", type=float, default=ips.DEFAULT_TILE_KM,
                  help="step 2: size of the MODIS proxy crop (default: "
                       "%(default)s) -- NOT the same knob as run_dev_step1's "
                       "own --tile-km, which sizes the camera tile in step 4")
    p.add_argument("--band", type=int, default=ips.MODIS_BAND)
    p.add_argument("--components", default="nadir",
                  help="step 2: comma-separated pixel-mask component(s) to "
                       "overwrite from the proxy scene (default: %(default)s)")
    p.add_argument("--table-dir", default=None)
    p.add_argument("--l1x-out", default=None,
                  help="step 2: output L1X path (default: derived from "
                       "--cam-l1b, tagged with the proxy date -- see module "
                       "docstring)")
    p.add_argument("--manifest-out-dir", default=bim.DEFAULT_OUT_DIR,
                  help="step 3: where the new manifest json goes (default: "
                       "%(default)s)")
    p.add_argument("--no-checksum-cache", action="store_true")
    p.add_argument("--no-run-log", action="store_true",
                  help="skip appending to apps/l2/data/%s" % RUN_LOG_FILENAME)
    return p.parse_known_args()


def main():
    args, passthrough = parse_args()

    for flag in REJECTED_PASSTHROUGH_FLAGS:
        if _has_flag(passthrough, flag):
            raise SystemExit(
                "run_full_pipeline.py always builds and uses its own fresh "
                "manifest (step 3) -- passing %s yourself would be silently "
                "overridden, so it's rejected instead. Run the four scripts "
                "by hand if you need an existing manifest." % flag)

    print("=== Step 1/4: find + download proxy scene ===")
    (modis_geo, modis_l1b, tile_pick, t_center, had_time,
     chosen) = find_and_download_candidate(args)
    proxy_tag = t_center.strftime("%Y%m%dT%H%M")

    print("\n=== Step 2/4: build L1X ===")
    l1x_path = build_l1x_file(args, modis_geo, modis_l1b, tile_pick, proxy_tag)

    if not args.brdf_land and not args.no_brdf_auto_fetch:
        # Added 2026-10-23 (Sebastian): match the land BRDF to the proxy
        # date BY DEFAULT instead of silently leaving l2_cam's fixed
        # 2021-05-18 sample file in use -- the root cause traced for the
        # "good" vs. --proxy-date 2024-07-12 brightness mismatch (see
        # docs/L1B_TO_L2_ROADMAP.md). t_center is the same overpass time
        # already used to pick the MODIS candidate in step 1, so this
        # needs no new input.
        if t_center.date() >= fetch_brdf.VIIRS_JPSS1_LAUNCH_DATE:
            print("\nBRDF auto-fetch: looking for a real, date-matched "
                  "VJ143C1 land BRDF file for %s ..." % t_center.date().isoformat())
            auto_brdf_path = fetch_brdf.fetch_brdf_for_date(
                t_center.date(), l2_cam_root=args.l2_cam_root, verbose=True)
            if auto_brdf_path is not None:
                args.brdf_land = auto_brdf_path
            else:
                print("BRDF auto-fetch: no real VJ143C1 file available for "
                      "%s -- falling back to l2_cam's own hardcoded "
                      "(stale, 2021-05-18) sample BRDF file."
                      % t_center.date().isoformat())
        else:
            print("\nBRDF auto-fetch: overpass date %s predates VIIRS/"
                  "JPSS-1's 2017-11-18 launch -- no real VJ143C1 file can "
                  "exist for this date; leaving l2_cam's own hardcoded "
                  "sample BRDF file in use (see run_tile_rt.py's "
                  "ancillary-date FATAL check)." % t_center.date().isoformat())

    print("\n=== Step 3/4: build input manifest ===")
    manifest_path = build_manifest(args, l1x_path)

    print("\n=== Step 4/4: run_dev_step1 ===")
    extra_defaults = []
    if not _has_flag(passthrough, "--out"):
        extra_defaults += ["--out",
                          "apps/l2/data/dev_step1_result_PROXY%s.yml" % proxy_tag]
    if not _has_flag(passthrough, "--l2-output-dir"):
        extra_defaults += ["--l2-output-dir", "output_l2/PROXY%s" % proxy_tag]
    run_dev_step1_argv = (
        ["run_dev_step1.py", "--input-manifest", manifest_path,
         "--tile-pick", args.tile_pick, "--l2-cam-root", args.l2_cam_root]
        + extra_defaults + passthrough)
    print("run_dev_step1.py argv: %s" % " ".join(run_dev_step1_argv[1:]))
    dev_step1_out = _flag_value(run_dev_step1_argv, "--out")
    old_argv = sys.argv
    try:
        sys.argv = run_dev_step1_argv
        run_dev_step1.main()
    finally:
        sys.argv = old_argv

    if not args.no_run_log:
        append_run_log(args, proxy_tag, t_center, had_time, chosen,
                       l1x_path, manifest_path, dev_step1_out)


if __name__ == "__main__":
    main()
