"""
libcam/env_check.py -- verify the running interpreter actually has what this
codebase's code paths need, and say so loudly rather than let an optional
import silently degrade into a misleading result.

Added 2026-10-13, after a day lost to "0 ADM blobs" on BOTH a known-good
granule and a new one. The cause: scipy had gone missing from the interpreter
actually in use, and measure_blob_centroids() in libcam/mask/retrieve_mask.py
catches ImportError and returns an empty blob list, by design (see its
docstring), rather than raising. Nothing said "environment problem" instead
of "data problem" until someone went looking by hand.

Usage
-----
Command line, before trusting any output from this environment:
    python3 -m libcam.env_check
    python3 -m libcam.env_check --extra full

From code, to fail loudly at the exact point something is actually needed
(not meant to replace every try/except ImportError in this codebase -- only
the specific spots where "missing" would otherwise look like "found nothing"):
    from libcam.env_check import require
    require("scipy", why="ADM blob detection (measure_blob_centroids)")
"""
from __future__ import annotations

import importlib
import sys

# Mirrors setup.py's install_requires / extras_require['full']. Not
# mechanically linked to setup.py (nothing here parses it) -- if one
# changes, change the other by hand.
CORE = ("numpy",)
EXTRAS = {
    "full": ("scipy", "matplotlib", "cartopy", "h5py", "pyhdf", "netCDF4"),
}


def _probe(mod_name):
    try:
        m = importlib.import_module(mod_name)
    except Exception as e:  # ImportError, plus the rare unrelated failure
        return False, "%s: %s" % (type(e).__name__, e)
    return True, getattr(m, "__version__", "(no __version__)")


def have(mod_name):
    """True if ``mod_name`` imports cleanly in THIS interpreter, right now."""
    return _probe(mod_name)[0]


def explain_missing(mod_name, why=""):
    ok, detail = _probe(mod_name)
    if ok:
        return None
    return ("%r is not importable in this interpreter (%s)%s. This is an "
            "ENVIRONMENT problem, not a data problem -- a 0-count or empty "
            "result that follows this is not a finding about the granule. "
            "Run `python3 -m libcam.env_check` for the full picture; "
            "setup.py lists %r under extras_require['full']."
            % (mod_name, sys.executable,
               (" (needed for %s)" % why) if why else "", mod_name))


def require(mod_name, why=""):
    """Import ``mod_name`` or raise a clear, actionable RuntimeError."""
    msg = explain_missing(mod_name, why=why)
    if msg is not None:
        raise RuntimeError(msg)


def report(extras=("full",)):
    """Returns (interpreter_info: dict, packages: dict[name -> (ok, detail)])."""
    interp = dict(
        executable=sys.executable, prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        in_venv=(sys.prefix != sys.base_prefix),
        version=sys.version.split()[0])
    names = list(CORE)
    for e in extras:
        names += list(EXTRAS.get(e, ()))
    return interp, {n: _probe(n) for n in names}


def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Verify this interpreter has what libera-cam needs.")
    p.add_argument("--extra", action="append", default=[],
                   help="extras_require group to check (default: full). Repeatable.")
    a = p.parse_args(argv)
    interp, packages = report(extras=tuple(a.extra) or ("full",))
    print("interpreter : %s" % interp["executable"])
    print("version     : %s" % interp["version"])
    print("in venv     : %s%s" % (interp["in_venv"], "" if interp["in_venv"] else
          "   <-- plain system/user interpreter, not an isolated env"))
    print()
    missing = []
    for name, (ok, detail) in packages.items():
        print("  %-7s %-10s %s" % ("OK " if ok else "MISSING", name, detail))
        if not ok:
            missing.append(name)
    print()
    if missing:
        print("MISSING: %s" % ", ".join(missing))
        print("Code that degrades gracefully on these (e.g. ADM blob "
              "detection) will still run -- it just silently skips real "
              "work, and its output can look like a data finding when it "
              "is actually this.")
        return 1
    print("All checked packages present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
