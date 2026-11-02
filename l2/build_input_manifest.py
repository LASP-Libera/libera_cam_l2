#!/usr/bin/env python3
"""Build the INPUT manifest (JSON) for the dev-step-1 one-tile run
(Sebastian, 2026-10-21/22).

Per the 2026-10-20 plan: our apps/l2 pipeline actually reads and
validates three files via a formal input manifest -- camera L1B,
radiometer L1B, ERA5 reanalysis -- unlike l2_cam/main_cam_l2.py, whose
manifest plumbing exists but is never acted on (it loads a manifest, then
ignores it and reads a hardcoded sample/input.yml instead). Everything
else (VIIRS BRDF climatology, the EASE2 land/ocean mask, sea ice) stays in
the per-run settings yml for now and migrates into the manifest gradually
-- see docs/L1B_TO_L2_ROADMAP.md.

**2026-10-22: works around a real bug in libera_utils 5.5.2.**
`Manifest.add_files()` calls `libera_utils.io.manifest.calculate_checksum()`,
which does `md5(fh.read())` -- the ENTIRE file read into memory in one
shot before hashing, no chunking. Our camera L1B for this orbit is 85 GB;
calling `add_files()` on it tries to allocate ~85 GB of RAM at once, which
is exactly what froze Sebastian's laptop (confirmed by reading
libera_utils' actual installed source, not guessed).

Workaround: compute each file's MD5 ourselves with a proper chunked/
streaming reader (`checksum_streamed()` below -- same algorithm,
`usedforsecurity=False` MD5, just bounded memory), then hand Manifest a
pre-computed checksum via the dict form of the `files` field
(`{"filename": ..., "checksum": ...}`). Manifest's own validation
(`transform_files` in libera_utils/io/manifest.py) only calls the naive
`calculate_checksum()` when a checksum is NOT already supplied, so this
is a supported path, not a hack around a private API -- we just avoid
ever calling `add_files()` on a bare path, which is the only call that
forces the naive read.

**2026-10-22: added a local checksum cache.** Hashing the 85 GB camera
L1B always requires reading all 85 GB off disk -- that's unavoidable
for a content checksum to mean anything, no way around it on a cold
run. But this script gets rerun during active development without the
underlying files changing, so `apps/l2/data/.checksum_cache.json` keys
each cached checksum on (path, size, mtime) and skips the re-read
entirely when those match -- a no-op rerun becomes near-instant instead
of paying the full read again. Delete that file (or pass
--no-checksum-cache) to force a fresh hash.
"""
import argparse
import glob
import hashlib
import json
import os
import time

from libera_utils import Manifest
from libera_utils.io.manifest import ManifestType

DEFAULT_CAM_L1B = "data/DITL2/L1B/LIBERA_L1B_CAM_V0-3-2_20260712T001717_20260712T040151_R26281034503.nc"
DEFAULT_RAD_L1B = "data/DITL2/L1B/LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_R26281034503.nc"
DEFAULT_ERA5 = "data/DITL2/ERA5/39cfa975852f21f218547a3706f8c128.nc"
DEFAULT_OUT_DIR = "apps/l2/data"
CACHE_FILENAME = ".checksum_cache.json"

CHUNK_BYTES = 64 * 1024 * 1024  # 64 MB per read -- bounded memory regardless of file size


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache_path, cache):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def cached_checksum(path, cache, label=""):
    """Checksum from the cache if (size, mtime) still match, otherwise a
    fresh streamed hash (which updates the cache in place)."""
    stat = os.stat(path)
    entry = cache.get(path)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime") == stat.st_mtime:
        print("    %s: using cached checksum (unchanged since last run)" % label)
        return entry["md5"]
    checksum = checksum_streamed(path, label=label)
    cache[path] = {"size": stat.st_size, "mtime": stat.st_mtime, "md5": checksum}
    return checksum


def checksum_streamed(path, label=""):
    """MD5 of a file, read in CHUNK_BYTES pieces instead of all at once.
    Same algorithm libera_utils' own calculate_checksum() uses
    (md5(..., usedforsecurity=False)), just without the 85 GB memory spike."""
    size = os.path.getsize(path)
    h = hashlib.md5(usedforsecurity=False)
    read = 0
    t0 = time.time()
    next_report_gb = 5
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if read / 1e9 >= next_report_gb:
                print("    %s: %.1f / %.1f GB hashed (%.0f s elapsed)"
                      % (label, read / 1e9, size / 1e9, time.time() - t0))
                next_report_gb += 5
    print("    %s: done (%.1f GB, %.0f s)" % (label, size / 1e9, time.time() - t0))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cam-l1b", default=DEFAULT_CAM_L1B)
    p.add_argument("--rad-l1b", default=DEFAULT_RAD_L1B)
    p.add_argument("--era5", default=DEFAULT_ERA5)
    p.add_argument("--brdf-land", default=None,
                  help="path to a real, date-matched VJ143C1 BRDF file "
                       "(e.g. from apps/l2/fetch_brdf.py) -- optional; "
                       "omit to keep using l2_cam's own hardcoded "
                       "(stale, 2021-05-18) sample BRDF file")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="where the LIBERA_INPUT_MANIFEST_*.json gets written "
                        "(default: apps/l2/data)")
    p.add_argument("--no-checksum-cache", action="store_true",
                   help="force a fresh checksum even if a cached one matches "
                        "this file's current size/mtime")
    args = p.parse_args()

    files = [("camera L1B", args.cam_l1b), ("radiometer L1B", args.rad_l1b),
             ("ERA5", args.era5)]
    if args.brdf_land:
        files.append(("BRDF land (VJ143C1)", args.brdf_land))
    for label, path in files:
        if not os.path.exists(path):
            raise FileNotFoundError("%s file not found: %s "
                                    "(run this from the libera-cam repo root, "
                                    "or pass an explicit path)" % (label, path))
        size_gb = os.path.getsize(path) / 1e9
        print("  %-16s %s (%.2f GB)" % (label + ":", path, size_gb))

    os.makedirs(args.out_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.out_dir, "LIBERA_INPUT_MANIFEST_*.json")))

    print("\nComputing checksums (chunked -- this avoids libera_utils' "
          "whole-file-in-memory checksum, which is what froze things on the "
          "85 GB camera L1B last time). The camera L1B will take a while on "
          "a cold run; progress prints every 5 GB. Unchanged files reuse a "
          "cached checksum from a prior run (see --no-checksum-cache).")
    cache_path = os.path.join(args.out_dir, CACHE_FILENAME)
    cache = {} if args.no_checksum_cache else load_cache(cache_path)
    file_dicts = []
    for label, path in files:
        abspath = os.path.abspath(path)
        checksum = cached_checksum(abspath, cache, label=label)
        file_dicts.append({"filename": abspath, "checksum": checksum})
    save_cache(cache_path, cache)

    # Passing pre-computed checksums this way means Manifest's own
    # validation (transform_files) uses ours instead of calling its naive
    # calculate_checksum() again -- see module docstring.
    input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=file_dicts)
    input_manifest.write(out_path=args.out_dir)

    after = set(glob.glob(os.path.join(args.out_dir, "LIBERA_INPUT_MANIFEST_*.json")))
    new_files = sorted(after - before)
    if new_files:
        print("\nWrote input manifest: %s" % new_files[0])
    else:
        print("\nWrote input manifest under %s (couldn't isolate the new "
              "filename -- list the dir to find it)" % args.out_dir)


if __name__ == "__main__":
    main()
