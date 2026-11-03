#!/usr/bin/env python3
"""Plain-assert tests for run_full_pipeline.py's pure selection/parsing
logic -- pick_candidate() (the MODIS/VIIRS candidate auto-picker) and
parse_proxy_date() (--proxy-date parsing). No network, no real data
files, no pytest dependency -- just run it directly:

    python l2/tests/test_run_full_pipeline.py

Written 2026-11-03 specifically to get confidence in pick_candidate()
(the "which MODIS/VIIRS index gets auto-selected" logic) before handing
the pipeline to a new user, since it's otherwise only ever exercised
indirectly, buried inside a live network-dependent full-pipeline run.
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import run_full_pipeline as rfp  # noqa: E402

FAILURES = []


def check(label, cond):
    status = "ok" if cond else "FAIL"
    print("  [%s] %s" % (status, label))
    if not cond:
        FAILURES.append(label)


def cand(tag, vza=None, offset_min=0.0):
    """A minimal fake candidate dict -- just enough for pick_candidate()."""
    return {"satellite": "test", "filename_tag": tag, "vza_deg": vza,
            "sza_deg": 30.0 if vza is not None else None,
            "dist_km": 10.0 if vza is not None else None,
            "offset_min": offset_min, "granule_dt": "2024-07-12T00:00:00"}


def test_pick_candidate_explicit_index():
    print("pick_candidate: explicit --download-index")
    candidates = [cand("a", 5.0), cand("b", 10.0), cand("c", 15.0)]
    idx, reason = rfp.pick_candidate(candidates, tile_vza=5.0, download_index=1)
    check("honors explicit index regardless of geometry", idx == 1)
    check("reason says explicit", reason == "explicit --download-index")


def test_pick_candidate_explicit_index_out_of_range():
    print("pick_candidate: explicit --download-index out of range")
    candidates = [cand("a", 5.0), cand("b", 10.0)]
    try:
        rfp.pick_candidate(candidates, tile_vza=5.0, download_index=5)
        check("raises SystemExit on out-of-range index", False)
    except SystemExit as exc:
        check("raises SystemExit on out-of-range index", True)
        check("message mentions the bad index", "5" in str(exc))


def test_pick_candidate_geometry_closest():
    print("pick_candidate: auto-pick by geometry closeness")
    # tile vza = 0.32 (this orbit's real tile_pick.yml value) -- candidate
    # "b" (0.5) should win over "a" (3.0) and "c" (-10.0).
    candidates = [cand("a", 3.0), cand("b", 0.5), cand("c", -10.0)]
    idx, reason = rfp.pick_candidate(candidates, tile_vza=0.32)
    check("picks the geometry-closest candidate (index 1, 'b')", idx == 1)
    check("reason cites geometry-closest", "geometry-closest" in reason)


def test_pick_candidate_ignores_missing_geometry():
    print("pick_candidate: candidates with no geometry are excluded, not just deprioritized")
    candidates = [cand("a", None), cand("b", 7.0), cand("c", None)]
    idx, reason = rfp.pick_candidate(candidates, tile_vza=7.0)
    check("skips candidates with vza_deg=None entirely", idx == 1)


def test_pick_candidate_no_geometry_at_all():
    print("pick_candidate: no candidate has usable geometry -> falls back to index 0")
    candidates = [cand("a", None), cand("b", None)]
    idx, reason = rfp.pick_candidate(candidates, tile_vza=7.0)
    check("falls back to index 0", idx == 0)
    check("reason cites the no-geometry fallback", "closest-in-time" in reason)


def test_pick_candidate_tile_vza_none():
    print("pick_candidate: tile_vza itself missing -> also falls back to index 0")
    candidates = [cand("a", 1.0), cand("b", 2.0)]
    idx, reason = rfp.pick_candidate(candidates, tile_vza=None)
    check("falls back to index 0 when tile_vza is None", idx == 0)
    check("reason cites the fallback", "closest-in-time" in reason)


def test_pick_candidate_tie_goes_to_first():
    print("pick_candidate: an exact tie in |vza - tile_vza| goes to the first candidate in list order")
    candidates = [cand("a", 4.0), cand("b", 6.0)]  # both |x-5| == 1
    idx, reason = rfp.pick_candidate(candidates, tile_vza=5.0)
    check("tie resolves to the first-listed candidate (index 0, 'a')", idx == 0)


def test_parse_proxy_date_bare_date():
    print("parse_proxy_date: bare 'YYYY-MM-DD'")
    dt, had_time = rfp.parse_proxy_date("2024-07-12")
    check("had_time is False for a bare date", had_time is False)
    check("parses the right calendar date",
          (dt.year, dt.month, dt.day) == (2024, 7, 12))
    check("time-of-day defaults to midnight",
          (dt.hour, dt.minute, dt.second) == (0, 0, 0))


def test_parse_proxy_date_full_datetime():
    print("parse_proxy_date: full ISO datetime, no timezone")
    dt, had_time = rfp.parse_proxy_date("2024-07-12T18:40:00")
    check("had_time is True", had_time is True)
    check("parses hour/minute exactly",
          (dt.hour, dt.minute, dt.second) == (18, 40, 0))


def test_parse_proxy_date_trailing_z():
    print("parse_proxy_date: trailing 'Z' (UTC) is accepted and stripped to naive UTC")
    dt, had_time = rfp.parse_proxy_date("2024-07-12T18:40:00Z")
    check("had_time is True", had_time is True)
    check("hour unchanged for Z (already UTC)", dt.hour == 18)
    check("result is naive (no tzinfo)", dt.tzinfo is None)


def test_parse_proxy_date_utc_offset_converted():
    print("parse_proxy_date: a non-UTC offset is converted to naive UTC, not left as local time")
    # 18:40 at UTC-06:00 == 00:40 the next day in UTC.
    dt, had_time = rfp.parse_proxy_date("2024-07-12T18:40:00-06:00")
    check("had_time is True", had_time is True)
    check("converted to the correct UTC instant",
          (dt.month, dt.day, dt.hour, dt.minute) == (7, 13, 0, 40))
    check("result is naive (no tzinfo) to match tile_pick.yml's convention",
          dt.tzinfo is None)


def test_parse_proxy_date_garbage_raises():
    print("parse_proxy_date: unparseable input raises rather than silently misparsing")
    try:
        rfp.parse_proxy_date("not-a-date")
        check("raises on garbage input", False)
    except ValueError:
        check("raises on garbage input", True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print()
    if FAILURES:
        print("%d check(s) FAILED:" % len(FAILURES))
        for f in FAILURES:
            print("  - %s" % f)
        sys.exit(1)
    print("All checks passed (%d test functions)." % len(tests))


if __name__ == "__main__":
    main()
