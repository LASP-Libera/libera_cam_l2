"""One-frame L2B (CF-CAM-CAMTIME) diagnostic: retrieve the pixel mask for a
single CT-mode camera exposure, compute pseudo-footprint anchor locations
for CROSSTRACK_STRIPE / ALONGTRACK_STRIPE / FORWARD_STRIPE (from the static
VZA-sweep config) and ADM_CIRCLES (from the retrieval's own cells), and
render three diagnostic PNGs:

  1. mask_components.png -- every selected pixel colored by its retrieved
     component (nadir/forward/cross/adm), UNEXPLAINED ("ghost") pixels and
     MISSING (predicted but not selected in the data) pixels each in their
     own distinct color.
  2. pseudofootprints.png -- the same frame with every computed
     pseudo-footprint anchor drawn and colored by tag.
  3. vza_sweep_check.png -- the static radiometer VZA-sweep config plotted
     vs. time within its own sweep, to confirm (not just assert) it is one
     clean, homogeneous elevation scan, not stitched from more than one.

Anchor placement (fixed 2026-11-09, per Sebastian: anchors must sit on
each stripe's own CENTERLINE, not wherever a raw per-pixel VZA value
happens to match most closely -- the earlier version searched the whole
stripe width for the closest VZA, which let it wander off-centerline in
the perpendicular/azimuthal direction when two pixels at different
widths had near-identical VZA):

  CROSSTRACK_STRIPE (nadir, runs along FPA columns): for every column
  present in the component, the centerline row is the midpoint between
  that column's own min/max member row (so a tilt in the stripe is
  followed, not averaged away). The real pixel nearest that midpoint
  gives a clean column -> VZA curve; targets are matched against THAT
  curve, so the result always sits on the centerline. Sign is left/right
  of the column with the smallest centerline VZA (the nadir crossing).

  ALONGTRACK_STRIPE (cross, runs along FPA rows): identical, with rows
  and columns swapped -- centerline column per row, sign up/down.

  FORWARD_STRIPE: still inherits CROSSTRACK_STRIPE's matched COLUMN
  directly (parallel stripe on the FPA), but now looks up forward's OWN
  centerline row at that column, rather than nearest-matching by pixel
  index within forward's raw member set.

Explicitly ONE exposure only, per Sebastian's 2026-11-09 instruction, as a
checkpoint before looping this over a real range of exposures. Duplicates
(does not import) haversine_bearing/footprint_axes_km/_ellipse_membership
from l2/run_tile_rt.py -- same repo convention as elsewhere (that module
pulls in elevation_sweep_timing + brdf_kernels at import time, more than
this diagnostic needs).
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath("libcam"))
from libcam.mask.retrieve_mask import MaskRetriever, MASK_BITS, BIT_UNEXPLAINED

FILL_VAL = -999.0
EARTH_R_KM = 6371.0
COMPONENTS_ORDER = ("nadir", "forward", "cross", "raps", "limb", "adm")


# ─── duplicated from l2/run_tile_rt.py (pure numpy, see module docstring) ──

def haversine_bearing(lat0, lon0, lat, lon):
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(lat), np.radians(lon)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    dist = 2 * EARTH_R_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    y = np.sin(dlon) * np.cos(latr)
    x = np.cos(lat0r) * np.sin(latr) - np.sin(lat0r) * np.cos(latr) * np.cos(dlon)
    return dist, np.arctan2(y, x)


def footprint_axes_km(vza_deg, altitude_km, half_fov_deg=np.degrees(np.arctan(10.0 / 831.16)),
                      length_vza_cap_deg=80.0):
    phi_grid = np.linspace(1e-6, np.radians(40), 200000)
    RE, Rs = EARTH_R_KM, EARTH_R_KM + altitude_km
    r_grid = np.sqrt(RE ** 2 + Rs ** 2 - 2 * RE * Rs * np.cos(phi_grid))
    vza_grid = np.degrees(np.arcsin(np.clip(Rs * np.sin(phi_grid) / r_grid, -1, 1)))
    order = np.argsort(vza_grid)
    vza_sorted, r_sorted = vza_grid[order], r_grid[order]
    R = np.interp(vza_deg, vza_sorted, r_sorted)
    theta0 = np.radians(half_fov_deg)
    w = 2.0 * R * np.tan(theta0)
    vza_for_length = min(vza_deg, length_vza_cap_deg)
    l = w / np.cos(np.radians(vza_for_length))
    return w, l


def _ellipse_membership(x_km, y_km, center_east_km, center_north_km,
                        length_km, width_km, angle_deg):
    a_km, b_km = length_km / 2.0, width_km / 2.0
    ang = np.radians(angle_deg)
    dx, dy = x_km - center_east_km, y_km - center_north_km
    xl = dx * np.cos(ang) + dy * np.sin(ang)
    yl = -dx * np.sin(ang) + dy * np.cos(ang)
    return (xl / b_km) ** 2 + (yl / a_km) ** 2 <= 1.0


def _fpa(a):
    """L1B is (CAMERA_TIME, X, Y) FPA-column-major; retrieval wants FPA
    row-first -- same transpose as apps/l1b/inspect_l1b_mask.py /
    apps/l2/collocate_adm_footprints.py."""
    return np.asarray(a).T


# ─── frame selection + retrieval ───────────────────────────────────────────

def load_frame(granule, frame_index=None):
    ds = xr.open_dataset(granule, decode_timedelta=True)
    obsid = ds["Camera_Observation_ID"].values
    az_all = ds["Azimuth"].values
    ct_idx = np.where(obsid == 128)[0]
    if frame_index is None:
        chosen = None
        for i in ct_idx:
            a = _fpa(ds["Latitude"].isel(CAMERA_TIME=int(i)).values)
            if np.isfinite(a).any() and np.nanmax(np.abs(a)) > 0:
                chosen = int(i)
                break
        if chosen is None:
            raise SystemExit("no geolocated CT frame found")
    else:
        chosen = frame_index

    fields = {}
    for name in ["valid_pixel_mask", "Radiance", "Latitude", "Longitude",
                 "Viewing_Zenith_Surface", "Viewing_Azimuth_Surface_WRT_North",
                 "Altitude"]:
        fields[name] = _fpa(ds[name].isel(CAMERA_TIME=chosen).values)
    header = dict(cam_obs_id=int(obsid[chosen]), pixel_mask_id=0)
    az = float(az_all[chosen])
    return chosen, fields, header, az


def retrieve(fields, header, az, index):
    r = MaskRetriever(table_dir=None, verbose=False)
    mask = fields["valid_pixel_mask"].astype(bool)
    res = r.process_frame(mask, header=header, index=index, az=az)
    return r, res


# ─── pseudo-footprint anchor selection (centerline-constrained) ───────────

def centerline_curve(rows, cols, vza):
    """For a stripe's member pixels, the running-axis position -> (centerline
    perpendicular-axis position, VZA there) mapping. The running axis is
    whichever of row/col has the larger index spread; the centerline at
    each running-axis position is the midpoint between the min and max
    perpendicular-axis member pixel there (follows a tilt, not just the
    stripe's single global mean row/col). VZA at that centerline position
    is taken from the real member pixel closest to the midpoint -- so the
    resulting curve only ever uses pixels that already belong to this
    component, never an interpolated/synthetic one.

    Returns (axis, run_values, center_perp, center_vza) where axis=0 means
    the stripe runs along rows (perpendicular axis = columns) and axis=1
    means it runs along columns (perpendicular axis = rows) -- i.e. axis
    is the PERPENDICULAR axis index (0=row, 1=col), matching stripe_axis's
    original convention (the axis along which rows/cols vary the LEAST)."""
    axis = 0 if np.std(rows) >= np.std(cols) else 1
    # axis here = the RUNNING axis per the old convention (bigger spread).
    # Rename for clarity below: running axis 0 => iterate over rows is
    # wrong; recompute explicitly instead of reusing the old helper.
    run_is_row = np.std(rows) >= np.std(cols)
    run_vals = rows if run_is_row else cols
    perp_vals = cols if run_is_row else rows

    uniq = np.unique(run_vals)
    center_perp = np.full(uniq.shape, np.nan)
    center_vza = np.full(uniq.shape, np.nan)
    center_row = np.full(uniq.shape, np.nan)
    center_col = np.full(uniq.shape, np.nan)
    for i, u in enumerate(uniq):
        m = run_vals == u
        p = perp_vals[m]
        mid = 0.5 * (p.min() + p.max())
        j = np.argmin(np.abs(p - mid))
        center_perp[i] = p[j]
        if run_is_row:
            r0, c0 = u, p[j]
        else:
            r0, c0 = p[j], u
        center_row[i], center_col[i] = r0, c0
        center_vza[i] = vza[r0, c0]
    return run_is_row, uniq, center_perp, center_row, center_col, center_vza


def crosstrack_or_alongtrack_anchors(rows, cols, vza, signed_vza_targets):
    """Match each target signed-VZA sample against the stripe's own
    CENTERLINE VZA curve (see centerline_curve) -- never the raw
    full-width pixel set -- so every anchor sits on the centerline by
    construction. Sign = which side of the centerline position with the
    smallest VZA (the nadir/along-track crossing)."""
    run_is_row, run_vals, center_perp, center_row, center_col, center_vza = \
        centerline_curve(rows, cols, vza)
    finite = np.isfinite(center_vza) & (center_vza > FILL_VAL)
    run_vals, center_row, center_col, center_vza = (
        run_vals[finite], center_row[finite], center_col[finite], center_vza[finite])

    center_i = np.argmin(center_vza)
    center_run = run_vals[center_i]
    signed = np.sign(run_vals.astype(float) - center_run) * center_vza

    out = []
    for target in signed_vza_targets:
        j = int(np.argmin(np.abs(signed - target)))
        out.append(dict(row=int(center_row[j]), col=int(center_col[j]),
                        run_is_row=run_is_row, run_val=float(run_vals[j]),
                        matched_vza=float(center_vza[j]), target_vza=float(target)))
    return out, run_is_row, run_vals, center_row, center_col


def forward_anchors(rows, cols, vza, crosstrack_anchors, crosstrack_run_is_row):
    """Inherit CROSSTRACK_STRIPE's own matched running-axis coordinate
    directly (parallel stripe on the FPA -- Sebastian's instruction,
    2026-11-09), then look up FORWARD_STRIPE's own centerline at that
    same coordinate (not nearest-pixel-index matching within forward's
    raw member set)."""
    run_is_row, run_vals, center_perp, center_row, center_col, center_vza = \
        centerline_curve(rows, cols, vza)
    assert run_is_row == crosstrack_run_is_row, \
        "forward stripe is not parallel to crosstrack stripe -- axis mismatch"

    out = []
    for a in crosstrack_anchors:
        j = int(np.argmin(np.abs(run_vals.astype(float) - a["run_val"])))
        out.append(dict(row=int(center_row[j]), col=int(center_col[j]),
                        run_is_row=run_is_row, run_val=float(run_vals[j]),
                        inherited_from_run_val=a["run_val"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence-stripe-vza", type=int, default=1,
                   help="Thin the STRIPE (crosstrack/alongtrack/forward) VZA "
                        "sampling: 1 = every sweep sample (default), 2 = "
                        "every other, etc. Speeds up early runs / makes "
                        "plots less crowded. Independent of "
                        "--cadence-adm-vza.")
    ap.add_argument("--cadence-adm-vza", type=int, default=1,
                   help="Thin the ADM_CIRCLES sampling by VZA RING: 1 = "
                        "every VZA ring the retrieval produced (default), "
                        "2 = every other ring, etc. All VAA (azimuth) "
                        "positions within a kept ring are still used -- "
                        "this thins rings, not azimuth. Independent of "
                        "--cadence-stripe-vza.")
    args = ap.parse_args()

    granule = "../Libera/camera/libera-cam/data/DITL2/L1B/LIBERA_L1B_CAM_V0-3-2_20260712T001717_20260712T040151_R26281034503.nc"
    out_dir = "l2/data"
    os.makedirs(out_dir, exist_ok=True)

    with open("l2/data/vza_sweep_config.yml") as f:
        vza_cfg = yaml.safe_load(f)
    signed_vza_targets = np.array(vza_cfg["signed_vza_deg"])
    if args.cadence_stripe_vza > 1:
        signed_vza_targets = signed_vza_targets[::args.cadence_stripe_vza]
    print("stripe VZA cadence=%d -> %d stripe sample locations (of %d in the sweep)"
         % (args.cadence_stripe_vza, len(signed_vza_targets), len(vza_cfg["signed_vza_deg"])))

    # ── plot 0: confirm the static VZA config is one clean sweep ──────────
    rad_path = ("../Libera/camera/libera-cam/data/DITL2/L1B/"
               "LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_R26281034503.nc")
    rds = xr.open_dataset(rad_path)
    s, e = vza_cfg["sweep_sample_index_range"]
    rt = rds["radiometer_time"].values[s:e + 1]
    rt_s = (rt.astype("datetime64[ns]").astype(np.int64)
            - rt.astype("datetime64[ns]").astype(np.int64)[0]) / 1e9
    r_cross = rds["Cross_Track_Angle"].values[s:e + 1]
    r_vza = rds["Viewing_Zenith_Surface"].values[s:e + 1]
    r_signed = np.sign(r_cross) * r_vza
    finite = np.isfinite(r_signed) & (r_vza > FILL_VAL)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
    axes[0].plot(rt_s[finite], r_signed[finite], ".", markersize=3)
    axes[0].set_xlabel("time within sweep [s]")
    axes[0].set_ylabel("signed VZA [deg]")
    axes[0].set_title("Static VZA-sweep config source: one radiometer sweep\n"
                      "(source_granule=%s, samples %d..%d)"
                      % (vza_cfg["source_granule"], s, e))
    axes[0].grid(alpha=0.3)

    stored = np.array(vza_cfg["signed_vza_deg"])
    axes[1].plot(np.arange(len(stored)), stored, ".", markersize=3)
    axes[1].set_xlabel("stored sample index (sorted)")
    axes[1].set_ylabel("signed VZA [deg]")
    axes[1].set_title("Stored config (what the camera code reads)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p0 = os.path.join(out_dir, "vza_sweep_check.png")
    fig.savefig(p0, dpi=150)
    plt.close(fig)
    print("wrote", p0, "- monotonic:", bool(np.all(np.diff(r_signed[finite]) > 0)))

    # ── retrieval ───────────────────────────────────────────────────────────
    index, fields, header, az = load_frame(granule)
    print("Frame %d, Camera_Observation_ID=%d, Azimuth=%.2f"
         % (index, header["cam_obs_id"], az))
    retriever, res = retrieve(fields, header, az, index)
    print(res.summary())
    if not res.status.ok:
        raise SystemExit("retrieval failed: %s" % res.status.code)

    # reconstruct the full predicted-pixel raster (same call process_frame
    # itself makes internally -- build() is already a public method, so no
    # library change was needed) to recover WHICH pixels are "missing"
    # (predicted but not selected in the data), not just the count.
    predicted_mask, _ = retriever.build(res.phase, res.az)
    predicted_mask = predicted_mask.astype(bool)
    obs_mask = fields["valid_pixel_mask"].astype(bool)
    missing_mask = predicted_mask & ~obs_mask
    print("missing pixel check: reconstructed=%d vs res.missing=%d"
         % (int(missing_mask.sum()), res.missing))

    membership = res.membership
    lat, lon = fields["Latitude"], fields["Longitude"]
    vza = fields["Viewing_Zenith_Surface"]
    vaa = fields["Viewing_Azimuth_Surface_WRT_North"]
    alt = fields["Altitude"]
    rad = fields["Radiance"]

    # built once, reused by plot 1, the missing-pixel zoom panels, and plot 2
    label_colors = {
        "nadir": "#1f77b4", "forward": "#2ca02c", "cross": "#d62728",
        "raps": "#9467bd", "limb": "#8c564b", "adm": "#ff7f0e",
        "unexplained": "#7f7f7f", "missing": "#e377c2", "none": "#eeeeee",
    }
    nx, ny = membership.shape
    rgb = np.zeros((nx, ny, 3))
    rgb[:] = mcolors.to_rgb(label_colors["none"])
    rgb[missing_mask] = mcolors.to_rgb(label_colors["missing"])
    for name in COMPONENTS_ORDER:
        bit = (membership & MASK_BITS[name]).astype(bool)
        rgb[bit] = mcolors.to_rgb(label_colors[name])
    unexp = (membership & BIT_UNEXPLAINED).astype(bool)
    rgb[unexp] = mcolors.to_rgb(label_colors["unexplained"])

    def comp_rc(name):
        rr, cc = np.where((membership & MASK_BITS[name]).astype(bool))
        return rr, cc

    nadir_rc = comp_rc("nadir")
    cross_rc = comp_rc("cross")
    forward_rc = comp_rc("forward")
    print("component pixel counts: nadir=%d forward=%d cross=%d"
         % (len(nadir_rc[0]), len(forward_rc[0]), len(cross_rc[0])))

    # which components does "missing" fall under? -- check overlap of
    # missing_mask against each component's own PREDICTED layer (not just
    # observed membership, since missing pixels are by definition not
    # observed/labelled at all).
    fams = {k: v for k, v in retriever.solution.enabled.items() if v}
    from libcam.mask.combine_masks import combine_masks
    _, counts = combine_masks(retriever._tbl(), fams, az=res.az, frame=res.phase,
                             nx=retriever.nx, ny=retriever.ny)
    missing_by_component = {}
    print("missing pixels by predicted component:")
    for name, layer in counts["layers"].items():
        comp_missing = layer.astype(bool) & missing_mask
        n = int(comp_missing.sum())
        if n:
            print("  %s: %d" % (name, n))
            missing_by_component[name] = comp_missing

    # ── zoom panels: missing pixels are invisible at full-FPA scale (a few
    # thousand single pixels against a 2048x2048 image) -- crop tightly
    # around where they actually are, one panel per component that has any.
    if missing_by_component:
        n_panels = len(missing_by_component)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]
        for ax, (name, comp_missing) in zip(axes, missing_by_component.items()):
            rr, cc = np.where(comp_missing)
            pad = 40
            r0, r1 = max(0, rr.min() - pad), min(nx, rr.max() + pad)
            c0, c1 = max(0, cc.min() - pad), min(ny, cc.max() + pad)
            ax.imshow(rgb[r0:r1, c0:c1], origin="upper",
                     extent=[c0, c1, r1, r0])
            # overplot missing pixels explicitly -- the base rgb image
            # already colors them, but a bold marker on top makes them
            # unambiguous at this zoom.
            ax.plot(cc, rr, "x", color="black", markersize=5, markeredgewidth=1.2,
                   label="missing (%d total)" % int(comp_missing.sum()))
            ax.set_title("'%s' component: %d missing pixels (zoom)"
                        % (name, int(comp_missing.sum())))
            ax.set_xlabel("FPA col (y)")
            ax.set_ylabel("FPA row (x)")
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        pz = os.path.join(out_dir, "l2b_frame%d_missing_zoom.png" % index)
        fig.savefig(pz, dpi=150)
        plt.close(fig)
        print("wrote", pz)

    nadir_anchors, nadir_run_is_row, _, _, _ = crosstrack_or_alongtrack_anchors(
        *nadir_rc, vza, signed_vza_targets)
    cross_anchors, cross_run_is_row, _, _, _ = crosstrack_or_alongtrack_anchors(
        *cross_rc, vza, signed_vza_targets)
    fwd_anchors = forward_anchors(*forward_rc, vza, nadir_anchors, nadir_run_is_row)

    print("nadir anchors: %d, cross anchors: %d, forward anchors: %d"
         % (len(nadir_anchors), len(cross_anchors), len(fwd_anchors)))

    adm_cells_all = list(res.cells())
    vza_ring_idxs = sorted(set(iv for iv, ia, cell in adm_cells_all))
    if args.cadence_adm_vza > 1:
        keep_vza_rings = set(vza_ring_idxs[::args.cadence_adm_vza])
    else:
        keep_vza_rings = set(vza_ring_idxs)
    adm_cells = [(iv, ia, cell) for iv, ia, cell in adm_cells_all if iv in keep_vza_rings]
    print("ADM VZA cadence=%d -> %d VZA rings kept (of %d), %d cells (of %d total)"
         % (args.cadence_adm_vza, len(keep_vza_rings), len(vza_ring_idxs),
            len(adm_cells), len(adm_cells_all)))

    # ── plot 1: mask components + missing ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="upper")
    ax.set_title("Frame %d -- retrieved pixel components (CT mode)\n"
                "obs=%d explained=%d unexplained=%d missing=%d"
                % (index, res.observed, res.explained, res.unexplained, res.missing))
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=c,
                      markersize=12, label=n)
              for n, c in label_colors.items() if n != "none"]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_xlabel("FPA col (y)")
    ax.set_ylabel("FPA row (x)")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "l2b_frame%d_mask_components.png" % index)
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("wrote", p1)

    # ── plot 2: pseudofootprints by object ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="upper", alpha=0.5)

    tag_colors = {"CROSSTRACK_STRIPE": "#1f77b4", "ALONGTRACK_STRIPE": "#d62728",
                 "FORWARD_STRIPE": "#2ca02c", "ADM_CIRCLES": "#ff7f0e"}

    ANCHOR_MARKER_COLOR = "black"  # deliberately NOT any tag_colors value --
    # the ADM anchor was previously plotted in the same orange as its own
    # scatter dots underneath and was invisible; every anchor (all 4
    # objects) now uses this one high-contrast color/style so it reads
    # against any background color.

    def draw_anchors(anchors):
        if not anchors:
            return
        rr = [a["row"] for a in anchors]
        cc = [a["col"] for a in anchors]
        ax.plot(cc, rr, "+", color=ANCHOR_MARKER_COLOR, markersize=7,
               markeredgewidth=1.3, zorder=5)

    draw_anchors(nadir_anchors)
    draw_anchors(cross_anchors)
    draw_anchors(fwd_anchors)

    adm_anchor_rows, adm_anchor_cols = [], []
    for iv, ia, cell in adm_cells:
        rr, cc = np.unravel_index(cell.pixels, membership.shape)
        ax.scatter(cc, rr, s=4, color=tag_colors["ADM_CIRCLES"], alpha=0.4, zorder=2)
        # one anchor marker per cell: its observed centroid (mean of the
        # pixels actually retrieved) if the retrieval fitted one, else the
        # generator's predicted centre, else just the pixel-mean fallback.
        if cell.centre_obs is not None and np.isfinite(cell.centre_obs).all():
            ar, ac = cell.centre_obs
        elif cell.centre_pred is not None and np.isfinite(cell.centre_pred).all():
            ar, ac = cell.centre_pred
        else:
            ar, ac = float(rr.mean()), float(cc.mean())
        adm_anchor_rows.append(ar)
        adm_anchor_cols.append(ac)
    ax.plot(adm_anchor_cols, adm_anchor_rows, "x", color=ANCHOR_MARKER_COLOR,
           markersize=7, markeredgewidth=1.3, zorder=5)

    handles2 = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=10, label=t) for t, c in tag_colors.items()]
    handles2.append(Line2D([0], [0], marker="+", color=ANCHOR_MARKER_COLOR,
                           linestyle="", markersize=10, label="anchor (any object)"))
    ax.legend(handles=handles2, loc="upper right", fontsize=8)
    ax.set_title("Frame %d -- pseudo-footprint anchors by object (centerline-fixed)\n"
                "cadence_stripe_vza=%d, cadence_adm_vza=%d: "
                "crosstrack=%d alongtrack=%d forward=%d adm_cells=%d (of %d)"
                % (index, args.cadence_stripe_vza, args.cadence_adm_vza,
                   len(nadir_anchors), len(cross_anchors),
                   len(fwd_anchors), len(adm_cells), len(adm_cells_all)))
    ax.set_xlabel("FPA col (y)")
    ax.set_ylabel("FPA row (x)")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "l2b_frame%d_pseudofootprints_stripe%d_admvza%d.png"
                      % (index, args.cadence_stripe_vza, args.cadence_adm_vza))
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("wrote", p2)


if __name__ == "__main__":
    main()
