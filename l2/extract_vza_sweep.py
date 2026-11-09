"""One-time extraction of a representative radiometer cross-track sweep's
signed Viewing_Zenith_Surface (VZA) sequence, to be reused as the common
static pseudo-footprint sampling-location configuration for L2B
(CF-CAM-CAMTIME): CROSSTRACK_STRIPE and ALONGTRACK_STRIPE sample camera
radiances at exactly these same |VZA| positions (same list, different
stripe direction, azimuth-agnostic -- as if the radiometer swept exactly
through the stripe center); FORWARD_STRIPE instead reuses CROSSTRACK_
STRIPE's own x-pixel offsets, shifted in y (computed elsewhere, not by
this script). ADM_CIRCLES follows a separate, mask-retrieval-based route
and does not use this file at all.

Run once per representative RAD-4CH L1B granule (not at pipeline runtime)
per Sebastian's instruction, 2026-11-09 -- output is a static (but
re-generatable/configurable) YAML, read by the L2B code at runtime.

Source field for VZA (confirmed with Sebastian, 2026-11-09): the real
radiometer Viewing_Zenith_Surface, NOT the satellite-frame Elevation field
l2/elevation_sweep_timing.py falls back to. That script was built against
an earlier granule (2026-02-25) where Viewing_Zenith_Surface was
identically -999 (see docs/NEXT.md and Sebastian's own standing
instruction there to be asked before anyone touches that field) -- this
script's own default/tested granule
(LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_
R26281034503.nc) is a newer product version where Viewing_Zenith_Surface
is populated with real, physically sensible values (confirmed: a clean
V-shape around the nadir crossing, consistently signed the same as
Cross_Track_Angle). If a different/older granule is used with this
script and the field comes back entirely non-finite, the loader raises
rather than silently falling back to a different angle convention.

Sweep selection duplicates (does not import) elevation_sweep_timing.py's
own local_extrema/find_earth_view_sweeps -- pure-numpy, already tested
there -- rather than importing that module directly, both per this
repo's own established convention ("no precedent in this repo for one
apps/ script importing another", l2/elevation_sweep_timing.py's own
docstring) and because importing it drags in MaskRetriever/cartopy at
module level, which currently cannot resolve from this device_bash
Linux VM anyway (the repo's .venv has an absolute macOS interpreter path
and an absolute-path libcam.pth, both baked in on Sebastian's Mac and
unreachable from the mounted-folder view here -- unrelated to this
script, not something broken in the repo; native runs on his Mac are
unaffected).

No camera granule or camera-overlap window is needed here (unlike
elevation_sweep_timing.py's own pick_sweep, which exists to choose the
sweep that overlaps a specific camera active window): this script just
wants ANY one representative, fully-sampled sweep from the radiometer
file alone, so among all candidates it picks the one with the most
finite VZA samples (least edge-clipped).

Usage
-----
    python3 extract_vza_sweep.py <RAD-4CH.nc> [--rad-mode 128]
                                  [--sweep-index N] [--out PATH]
"""

import argparse
import datetime
import os

import numpy as np
import xarray as xr
import yaml

FILL_VAL = -999.0


# ─── duplicated from l2/elevation_sweep_timing.py (see module docstring) ──

def local_extrema(x):
    """Indices of local minima/maxima in a 1-D series (plateaus handled by
    forward-filling zero-diffs with the last nonzero direction). Returns
    (indices, kinds) with kinds[i] in {"min", "max"}, in index order."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=object)
    d = np.diff(x)
    sign = np.sign(d)
    filled = np.zeros_like(sign)
    last = 0.0
    for i, s in enumerate(sign):
        if s == 0:
            filled[i] = last
        else:
            filled[i] = s
            last = s
    idx, kinds = [], []
    for i in range(1, len(filled)):
        if filled[i - 1] > 0 and filled[i] < 0:
            idx.append(i)
            kinds.append("max")
        elif filled[i - 1] < 0 and filled[i] > 0:
            idx.append(i)
            kinds.append("min")
    return np.array(idx, dtype=int), np.array(kinds, dtype=object)


def find_earth_view_sweeps(elevation, time_s, geo_valid, max_gap_s=None):
    """The real "one full elevation scan sweep, horizon to horizon and
    back" -- groups geo_valid samples into sessions (bridging internal
    dropouts up to max_gap_s: None -> max(3x median geo_valid sample
    spacing, 1.0 s)), then splits each session into back-and-forth cycles
    via local_extrema. See elevation_sweep_timing.py's own docstring for
    the full "why" (the two-stage approach is not optional -- a
    single-stage version was confirmed wrong on real data there)."""
    elevation = np.asarray(elevation, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    geo_valid = np.asarray(geo_valid, dtype=bool)
    idxs = np.where(geo_valid)[0]
    if idxs.size == 0:
        return []
    if max_gap_s is None:
        dt = np.diff(time_s[idxs])
        max_gap_s = max(3.0 * float(np.median(dt)), 1.0) if dt.size else float("inf")

    breaks = np.where(np.diff(time_s[idxs]) > max_gap_s)[0]
    sessions = np.split(idxs, breaks + 1)

    sweeps = []
    for session in sessions:
        if session.size < 3:
            continue
        sub_elev = elevation[session]
        rel_idx, kinds = local_extrema(sub_elev)
        if rel_idx.size < 3:
            continue
        for i in range(len(rel_idx) - 2):
            if kinds[i] == "min" and kinds[i + 1] == "max" and kinds[i + 2] == "min":
                s = int(session[rel_idx[i]])
                p = int(session[rel_idx[i + 1]])
                e = int(session[rel_idx[i + 2]])
                sweeps.append(dict(
                    start_idx=s, peak_idx=p, end_idx=e,
                    start_t=float(time_s[s]), peak_t=float(time_s[p]), end_t=float(time_s[e]),
                    start_val=float(elevation[s]), peak_val=float(elevation[p]),
                    end_val=float(elevation[e])))
    return sweeps


# ─── this script's own loading/extraction logic ────────────────────────────

def load_radiometer_vza_fields(rad_path, mode=128):
    """Like elevation_sweep_timing.load_radiometer_elevation, but also
    pulls Cross_Track_Angle (for sign) and Viewing_Zenith_Surface (the
    actual VZA magnitude used downstream for footprint ellipse sizing,
    e.g. l2/run_tile_rt.py's footprint_axes_km). Same valid-sample masking
    convention (finite Elevation, above the -999 fill, in the requested
    Operational_Mode) so indices line up with what find_earth_view_sweeps
    expects -- Elevation (not Viewing_Zenith_Surface) still gates
    validity/mode-selection here, matching that convention exactly;
    Viewing_Zenith_Surface's own finiteness is checked separately, per
    sample, only when extracting the signed VZA sequence below (it is
    NaN at the sweep's own grazing-incidence edges even when Elevation
    and Operational_Mode are fine -- confirmed on the real granule)."""
    ds = xr.open_dataset(rad_path)

    def _get(name):
        return np.asarray(ds[name].values) if name in ds else None

    required = ["radiometer_time", "Elevation", "Cross_Track_Angle",
                "Viewing_Zenith_Surface"]
    missing = [n for n in required if n not in ds]
    if missing:
        raise SystemExit(
            "missing field(s) %s in %s -- available fields: %s"
            % (missing, rad_path, ", ".join(sorted(ds.data_vars))))

    time = np.asarray(ds["radiometer_time"].values, dtype=np.int64)
    elevation = _get("Elevation")
    cross_track = _get("Cross_Track_Angle")
    vza = _get("Viewing_Zenith_Surface")
    lat, lon = _get("Latitude"), _get("Longitude")
    mode_arr = _get("Operational_Mode")

    valid = np.isfinite(elevation) & (elevation > FILL_VAL)
    if mode_arr is not None:
        valid &= (mode_arr == mode)

    if not np.any(np.isfinite(vza[valid]) & (vza[valid] > FILL_VAL)):
        raise SystemExit(
            "Viewing_Zenith_Surface is entirely non-finite/fill in %s for "
            "Operational_Mode=%d -- this granule looks like the older "
            "product version where this field is an unpopulated -999 "
            "placeholder (see docs/NEXT.md). Confirm with Sebastian before "
            "falling back to a different angle convention for this "
            "granule; refusing to guess." % (rad_path, mode))

    n = int(valid.sum())
    out = dict(time=time[valid], elevation=elevation[valid],
               cross_track=cross_track[valid], vza=vza[valid])
    out["lat"] = lat[valid] if lat is not None else np.full(n, np.nan)
    out["lon"] = lon[valid] if lon is not None else np.full(n, np.nan)
    out["geo_valid"] = (np.isfinite(out["lat"]) & np.isfinite(out["lon"]) &
                        (out["lat"] > FILL_VAL) & (out["lon"] > FILL_VAL))
    return out


def pick_most_complete_sweep(sweeps, vza):
    """Among candidate sweeps (each a start/peak/end_idx dict from
    find_earth_view_sweeps), the one with the most finite VZA samples in
    its [start_idx, end_idx] span -- i.e. the most fully-sampled, least
    edge-clipped sweep in the file. No camera window to overlap with here
    (see module docstring), so completeness is the only criterion."""
    best_i, best_n = None, -1
    for i, sw in enumerate(sweeps):
        seg = vza[sw["start_idx"]:sw["end_idx"] + 1]
        n = int(np.isfinite(seg).sum())
        if n > best_n:
            best_i, best_n = i, n
    return best_i


def extract_signed_vza(rad, sweep):
    """Signed VZA sequence for one sweep: magnitude from the real
    Viewing_Zenith_Surface, sign from Cross_Track_Angle (same side of
    nadir -- confirmed on the real granule: sign(Cross_Track_Angle) flips
    exactly at the sample closest to the nadir crossing, where
    Viewing_Zenith_Surface itself dips to its sweep minimum). Non-finite
    VZA samples are dropped (near the sweep's own turning-point edges,
    where the surface-intersection VZA calculation is known to fail at
    near-grazing incidence -- confirmed numerically, not a bug). Returned
    sorted by signed VZA (already monotonic along the sweep on the real
    granule; sorting just guards against a noisy/non-monotonic edge
    case)."""
    s, e = sweep["start_idx"], sweep["end_idx"] + 1
    cross_track = rad["cross_track"][s:e]
    vza = rad["vza"][s:e]
    finite = np.isfinite(vza) & (vza > FILL_VAL) & np.isfinite(cross_track)
    signed_vza = np.sign(cross_track[finite]) * vza[finite]
    return np.sort(signed_vza)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rad_path", help="RAD-4CH radiometer L1B granule (.nc)")
    p.add_argument("--rad-mode", type=int, default=128,
                   help="Operational_Mode to select (default: 128, cross-track/CT)")
    p.add_argument("--sweep-index", type=int, default=None,
                   help="Override the automatic pick with a specific "
                        "candidate sweep index (see the printed list)")
    p.add_argument("--max-gap-s", type=float, default=None,
                   help="passed through to find_earth_view_sweeps (default: "
                        "max(3x median geo_valid sample spacing, 1.0 s))")
    p.add_argument("--out", default="vza_sweep_config.yml")
    args = p.parse_args()

    print("Loading %s (Operational_Mode=%d) ..." % (args.rad_path, args.rad_mode))
    rad = load_radiometer_vza_fields(args.rad_path, mode=args.rad_mode)
    order = np.argsort(rad["time"])
    for k in list(rad):
        rad[k] = rad[k][order]
    print("  %d samples with valid Elevation in this mode (%d also geolocated)"
          % (len(rad["time"]), int(rad["geo_valid"].sum())))

    t_ref = int(rad["time"].min())
    time_s = (rad["time"].astype(np.float64) - t_ref) / 1e9

    sweeps = find_earth_view_sweeps(rad["elevation"], time_s, rad["geo_valid"],
                                     max_gap_s=args.max_gap_s)
    if not sweeps:
        raise SystemExit("no full Earth-view sweeps found in this file/mode")
    print("  %d candidate sweep(s) found:" % len(sweeps))
    for i, sw in enumerate(sweeps):
        seg = rad["vza"][sw["start_idx"]:sw["end_idx"] + 1]
        n_finite = int(np.isfinite(seg).sum())
        print("    [%d] samples %d..%d (%.2f..%.2f s), %d finite VZA samples"
              % (i, sw["start_idx"], sw["end_idx"],
                 sw["start_t"], sw["end_t"], n_finite))

    chosen = args.sweep_index if args.sweep_index is not None \
        else pick_most_complete_sweep(sweeps, rad["vza"])
    sweep = sweeps[chosen]
    print("Using sweep [%d]." % chosen)

    signed_vza = extract_signed_vza(rad, sweep)
    print("  %d finite samples, signed VZA range [%.3f, %.3f] deg"
          % (len(signed_vza), signed_vza.min(), signed_vza.max()))

    cfg = dict(
        description=(
            "Static, one-time-extracted signed Viewing-Zenith-Angle (VZA) "
            "sweep -- the common pseudo-footprint sampling-location list "
            "for L2B (CF-CAM-CAMTIME)'s CROSSTRACK_STRIPE and "
            "ALONGTRACK_STRIPE (identical angles, different stripe "
            "direction), and, offset in x->y, for FORWARD_STRIPE. "
            "Azimuth-agnostic: as if the radiometer swept exactly through "
            "the stripe center. ADM_CIRCLES does not use this file "
            "(separate mask-retrieval-based route)."),
        source_granule=os.path.basename(args.rad_path),
        operational_mode=args.rad_mode,
        sweep_sample_index_range=[int(sweep["start_idx"]), int(sweep["end_idx"])],
        sweep_time_range_s=[sweep["start_t"], sweep["end_t"]],
        n_samples=int(len(signed_vza)),
        vza_min_deg=float(signed_vza.min()),
        vza_max_deg=float(signed_vza.max()),
        sign_convention=(
            "sign(Cross_Track_Angle) * Viewing_Zenith_Surface -- negative "
            "on one side of nadir, positive on the other; true azimuth is "
            "not stored (azimuth-agnostic by construction, per Sebastian's "
            "2026-11-09 instruction)."),
        generated=datetime.datetime.utcnow().isoformat() + "Z",
        method="l2/extract_vza_sweep.py",
        signed_vza_deg=[float(v) for v in signed_vza],
    )
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
