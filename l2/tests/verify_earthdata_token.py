#!/usr/bin/env python3
"""Verify EARTHDATA_TOKEN is set and (optionally) actually works end to
end, before handing this repo to a new user or debugging a confusing
download failure three layers down in build_input_manifest.py /
fetch_proxy_scene.py / fetch_brdf.py.

    python l2/tests/verify_earthdata_token.py              # no network
    python l2/tests/verify_earthdata_token.py --worldview  # ~30s, needs internet

Modeled on er3t-edu's own tests/verify_install.py --worldview check
(same underlying er3t.util.daac.download_worldview_image() call -- this
project vendors the same er3t package, just Hong's fork of it) rather
than invented from scratch.

The plain (no --worldview) run only checks that EARTHDATA_TOKEN is set
in the environment -- it does NOT confirm the token is valid, only that
you remembered to set *something*. --worldview actually exercises it:
download_worldview_image() internally looks up a LAADS geoMeta file
for the given date/region, which needs a working, non-expired token --
so a successful run is real end-to-end confirmation, not just a env-var
presence check.
"""
import datetime
import os
import sys
import argparse

parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--worldview", action="store_true",
                     help="also download a small test image from NASA "
                          "Worldview/GIBS to confirm the token actually "
                          "works, not just that it's set (~30s, needs "
                          "internet)")
args = parser.parse_args()

FAILURES = []


def check(label, cond, hint=""):
    status = "ok" if cond else "FAIL"
    print("  [%s] %s" % (status, label))
    if not cond:
        FAILURES.append(label)
        if hint:
            print("         -> %s" % hint)


print("1. EARTHDATA_TOKEN")
token = os.environ.get("EARTHDATA_TOKEN")
check("EARTHDATA_TOKEN is set in this shell", bool(token),
      "see docs/GETTING_STARTED.md section 4 -- get one at "
      "https://urs.earthdata.nasa.gov/users/<your-username>/user_tokens "
      "then setenv/export EARTHDATA_TOKEN in your shell rc file and "
      "reload it (source ~/.cshrc or ~/.bashrc)")

if not token:
    print("\nNo token set -- skipping --worldview check even if requested "
          "(nothing to test against).")
    sys.exit(1 if FAILURES else 0)

print("\n2. er3t import")
try:
    import er3t  # noqa: F401
    check("import er3t", True)
except Exception as e:
    check("import er3t", False, str(e))
    print("\nCannot continue without er3t -- exiting.")
    sys.exit(1)

if args.worldview:
    print("\n3. Worldview RGB download (needs internet, ~30s)")
    try:
        from er3t.util.daac import download_worldview_image
        date = datetime.datetime(2024, 7, 12)
        extent = [-92.0, -90.0, 33.0, 35.0]  # small ~200 km patch, same as er3t-edu's own check
        fname = download_worldview_image(
            date, extent,
            fdir_out="tmp-data/verify",
            instrument="modis",
            satellite="aqua",
            fmt="png",
        )
        check("download_worldview_image() -- saved %s" % fname,
              bool(fname) and os.path.exists(fname),
              "token may be expired/invalid, or LAADS/GIBS is unreachable "
              "from here right now")
    except Exception as e:
        check("download_worldview_image()", False, str(e))
else:
    print("\n3. Worldview RGB download -- skipped (pass --worldview to "
          "actually confirm the token works, not just that it's set)")

print()
if FAILURES:
    print("%d check(s) failed." % len(FAILURES))
    sys.exit(1)
else:
    print("All checks passed.")
