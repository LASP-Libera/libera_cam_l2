"""
Standard QA figure for run_dev_step1.py (Sebastian, 2026-10-22).

One PNG per run, saved next to the YAML result, so a dev-step-1 run can
be "sanity-checked" by eye as Sebastian asked, not just by reading
numbers out of a YAML file. Five panels (2x3 grid, bottom-right blank):

  A. Footprint-feeding map: the tile box, every real RAD-4CH footprint
     ellipse that overlaps it (green = this tile alone can populate its
     cam_radiance_footprint, orange = it straddles the tile edge and
     also needs an adjacent tile -- the stitching case Sebastian
     flagged 2026-10-22, not yet implemented, just identified; dashed
     outline = VZA beyond the length-model cap, so the ellipse LENGTH
     is a bounded approximation, not literal -- these are still fully
     accounted for, just flagged), and the pooled camera pixels colored
     by whether they made it into the final near-nadir tile selection.
     Ellipse geometry/orientation matches plot_footprint_maps.py
     exactly (angle=-VAA).
  B. The computed RT radiance field itself (rad, from rt_shdom()) as a
     2D image -- the most direct check of whether real spatial
     structure exists across the tile (an argument for per-footprint
     output, not one tile-wide mean). East-west extent uses the
     cos(lat)-corrected physical spacing (2026-10-22 fix -- see
     run_tile_rt.py's rt_shdom() docstring for the bug this corrects:
     at this tile's latitude the uncorrected extent was ~2x too wide,
     which is what made this panel look like "two tiles side by side").
  C. Context map: the tile's location on a wider view of the globe (an
     Orthographic globe view with coastlines if cartopy is importable,
     otherwise a plain equirectangular band spanning the full longitude
     range and +/-40 deg of latitude around the tile) -- added
     2026-10-22 per Sebastian's request, purely for "where are we on
     the globe" orientation; carries no physics.
  D. The ocean windspeed field used for the surface BRDF
     (data['windspeed_ocean']) -- specifically to visually confirm the
     2026-10-22 ERA5 fix is spatially variable across the tile, not the
     constant field an earlier (buggy) version of that fix produced.
  E. Geometry sanity panel: SZA/VZA across pooled camera pixels in the
     tile, with the tile-mean values marked, so outliers are easy to
     spot.

This is read-only/diagnostic -- it doesn't change any of the computed
values, only visualizes them.
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

INK, MUT = "#1a1a1a", "#6b6b6b"
INSIDE_COLOR = "#2a9d55"
EDGE_COLOR = "#e07b1a"
EXCLUDED_COLOR = "#b4b4b4"
TILE_MARK_COLOR = "#c0392b"


def _style(ax):
    ax.tick_params(colors=MUT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#cfcfcf")
    ax.set_facecolor("white")


def _panel_footprint_map(ax, pooled, footprint_info, tile_km):
    half_km = tile_km / 2.0
    ax.add_patch(mpatches.Rectangle((-half_km, -half_km), tile_km, tile_km,
                                     fill=False, edgecolor=INK, linewidth=1.3,
                                     zorder=3))

    x_km, y_km = pooled["x_km"], pooled["y_km"]
    in_tile = pooled["in_tile"]
    in_box_only = pooled["in_box"] & ~in_tile
    ax.scatter(x_km[in_tile], y_km[in_tile], s=2.0, color="#4C78A8", alpha=0.5,
               linewidths=0, zorder=2, label="pooled, in tile (near-nadir)")
    if in_box_only.any():
        ax.scatter(x_km[in_box_only], y_km[in_box_only], s=2.0,
                   color=EXCLUDED_COLOR, alpha=0.4, linewidths=0, zorder=1,
                   label="in box, excluded (off-nadir)")

    n_inside = n_edge = n_capped = 0
    for rec in footprint_info.get("records", []):
        color = INSIDE_COLOR if rec["classification"] == "inside" else EDGE_COLOR
        n_inside += rec["classification"] == "inside"
        n_edge += rec["classification"] == "edge"
        capped = rec.get("vza_model_capped", False)
        n_capped += capped
        # Dashed = this footprint's VZA is beyond length_vza_cap_deg, so its
        # length uses a capped (not literal) limb-geometry approximation --
        # see footprint_axes_km()'s docstring (2026-10-22 revision, per
        # Sebastian: these must stay in the record, not be dropped, but
        # they're worth visually flagging as approximate).
        # Alpha encodes fill_fraction (added 2026-10-22, the exposure-gap /
        # fill-fraction diagnostic Sebastian requested): faint = badly
        # under-filled by real camera pixels relative to what a regular
        # grid at the local real-pixel pitch would give, opaque = fully
        # filled. Unknown (not enough nearby real pixels to estimate a
        # local pitch at all) stays at the old fixed alpha so it doesn't
        # read as "badly filled" when it's really just "no data to judge."
        ff = rec.get("fill_fraction")
        alpha = 0.85 if ff is None else float(np.clip(0.25 + 0.65 * ff, 0.25, 0.9))
        e = mpatches.Ellipse((rec["east_km"], rec["north_km"]),
                              width=rec["width_km"], height=rec["length_km"],
                              angle=rec["angle_deg"], fill=False,
                              edgecolor=color, linewidth=1.1, alpha=alpha,
                              linestyle=("--" if capped else "-"), zorder=4)
        ax.add_patch(e)

        # Cloud-fraction label (added 2026-10-23, Sebastian: "the
        # footprints in green could get small cloud fraction numbers for
        # those ones that HAVE clouds in them, no numbers where it was
        # clear sky") -- cam_cloud_fraction_rt comes from
        # run_tile_rt.radiance_and_counts_for_footprints()'s optional
        # cloud_flag argument (run_dev_step1.py's e-fold cloud
        # detection); None whenever that wasn't computed (e.g. not an
        # L1X input, or too few compared cells) or this is an 'edge'
        # record, in which case nothing is drawn, same as a clear-sky
        # (cf==0) footprint.
        cf = rec.get("cam_cloud_fraction_rt")
        if rec["classification"] == "inside" and cf is not None and cf > 0:
            ax.text(rec["east_km"], rec["north_km"], "%.2f" % cf,
                   fontsize=6.5, color=color, ha="center", va="center",
                   fontweight="bold", zorder=5)

    pad = half_km * 0.25
    ax.set_xlim(-half_km - pad, half_km + pad)
    ax.set_ylim(-half_km - pad, half_km + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("east-west [km]", fontsize=9, color=MUT)
    ax.set_ylabel("north-south [km]", fontsize=9, color=MUT)
    cap_note = (", %d dashed (VZA > cap, length approximate)" % n_capped
                if n_capped else "")
    ff_vals = [r["fill_fraction"] for r in footprint_info.get("records", [])
               if r.get("fill_fraction") is not None]
    ff_note = (", fill_fraction median %.2f (faint=sparse, opaque=full)"
               % float(np.median(ff_vals)) if ff_vals else "")
    ax.set_title("Footprints fed by this tile: %d inside, %d edge "
                  "(need adjacent tile, not yet stitched)%s%s"
                  % (n_inside, n_edge, cap_note, ff_note),
                  fontsize=8.8, color=INK)
    handles = [
        mpatches.Patch(edgecolor=INSIDE_COLOR, facecolor="none", label="footprint: inside"),
        mpatches.Patch(edgecolor=EDGE_COLOR, facecolor="none", label="footprint: edge (stitching needed)"),
    ]
    if n_capped:
        handles.append(Line2D([0], [0], color=MUT, linestyle="--",
                               label="dashed: VZA > cap, length approximate"))
    if ff_vals:
        handles.append(Line2D([0], [0], color=MUT, alpha=0.3, linewidth=2,
                               label="faint/opaque: low/high fill_fraction"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    _style(ax)


def _panel_radiance(ax, rad, lon_land, lat_land, lat0, lon0, sfc_flag=None):
    """ORIENTATION FIX (2026-10-22, Sebastian: "wondering whether the RT
    tile has the orientation correct (the coastline seems to run slightly
    differently)"): the previous version built its (east_km, north_km)
    extent from a naive linspace over nx/ny*dx/dy, which silently ASSUMED
    grid index 0 = west/south and increasing index = east/north -- never
    verified against Hong's actual VJ143C1 grid-index convention (many
    satellite raster products store rows north-to-south, the opposite of
    that assumption; a wrong assumption here would flip or mirror the
    plotted coastline with no error raised). Fixed by computing each grid
    cell's REAL local east/north km position directly from its own
    lon_land/lat_land value -- the exact same equirectangular approx
    rt_shdom() itself uses for the SHDOM sensor ray positions (xpos0/
    ypos0) -- and plotting with pcolormesh instead of imshow+extent, so
    this panel is correct regardless of the array's internal index
    ordering (same technique _panel_windspeed already used successfully
    for the ocean windspeed field, via lon_ocean/lat_ocean).

    `sfc_flag`, if given (data['brdf'][...,6]: 1=ocean, 3=land, 4=ice), is
    drawn as a thin land/ice contour on top of the radiance -- so "is this
    a real coastline, and does the RT follow it" is directly checkable by
    eye, not just inferred from brightness."""
    lat_ref = float(np.mean(lat_land))
    km_per_deg = 6378.1370 * np.pi / 180.0  # matches rt_shdom()'s own R*pi/180
    east_km = (lon_land - lon0) * km_per_deg * np.cos(np.radians(lat_ref))
    north_km = (lat_land - lat0) * km_per_deg
    im = ax.pcolormesh(east_km, north_km, rad, cmap="inferno", shading="auto")
    if sfc_flag is not None:
        land_or_ice = (sfc_flag == 3.0) | (sfc_flag == 4.0)
        if land_or_ice.any() and not land_or_ice.all():
            ax.contour(east_km, north_km, land_or_ice.astype(float),
                       levels=[0.5], colors="#4C78A8", linewidths=0.9,
                       zorder=5)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=MUT)
    cb.set_label("radiance [SHDOM native units]", fontsize=8, color=MUT)
    ax.set_xlabel("east-west [km]", fontsize=9, color=MUT)
    ax.set_ylabel("north-south [km]", fontsize=9, color=MUT)
    ax.set_aspect("equal")
    finite = rad[np.isfinite(rad)]
    spread = ("n/a" if finite.size == 0 else
              "%.3g .. %.3g (spread %.1f%% of mean)" %
              (finite.min(), finite.max(),
               100.0 * (finite.max() - finite.min()) / max(abs(finite.mean()), 1e-12)))
    ax.set_title("RT radiance across tile: %s" % spread, fontsize=9.5, color=INK)
    _style(ax)


def _panel_context_map(fig, gridspec_slot, lat0, lon0, tile_km=100.0,
                       context_km=1000.0):
    """Zoomed regional context map centered on the tile -- changed
    2026-10-22 from a full globe view to a ~1000 km regional view, per
    Sebastian: "What would help if we zoomed a little bit in on the map
    on the top right panel, let's say around 1000 km context or so. We
    can confirm what comes out of the run [against the radiance panel's
    coastline orientation]." Pure orientation/reference, no physics.

    Uses an Azimuthal Equidistant projection centered on the tile (NOT
    Orthographic, the previous global-view projection) specifically
    because distances from the center are exactly true in that
    projection -- so `context_km` is a literal, not approximate, zoom
    radius, and the tile-extent box drawn on top is geometrically
    consistent with the radiance panel's own local east/north km axes
    (both are equirectangular/locally-flat approximations centered on
    the same point, so the two panels should show the same coastline
    orientation if the radiance panel's new real-lon/lat-derived
    orientation fix is correct)."""
    half_km = context_km / 2.0
    tile_half_km = tile_km / 2.0
    if HAS_CARTOPY:
        proj = ccrs.AzimuthalEquidistant(central_longitude=lon0, central_latitude=lat0)
        ax = fig.add_subplot(gridspec_slot, projection=proj)
        ax.set_extent([-half_km * 1000.0, half_km * 1000.0,
                       -half_km * 1000.0, half_km * 1000.0], crs=proj)
        # Finer-resolution coastline for this zoomed-in view than the
        # 110m default (pick_tile_overview.py's cfeature.COASTLINE/.LAND)
        # -- falls back to that 110m default if the 50m Natural Earth
        # shapefiles aren't already cached and can't be downloaded (no
        # network from wherever this runs).
        try:
            land_feat = cfeature.NaturalEarthFeature(
                "physical", "land", "50m", edgecolor="none", facecolor="0.92")
            coast_feat = cfeature.NaturalEarthFeature(
                "physical", "coastline", "50m", edgecolor="0.3", facecolor="none")
            # cartopy features fetch/parse their shapefile lazily, on first
            # geometry access at DRAW time -- force that now, inside this
            # try, so a download failure (no network) is caught here and
            # falls back cleanly, instead of surfacing later as a crash
            # inside fig.savefig().
            list(land_feat.geometries())
            list(coast_feat.geometries())
            ax.add_feature(land_feat, zorder=0)
            ax.add_feature(coast_feat, linewidth=0.6, zorder=1)
        except Exception:
            ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="0.3", zorder=1)
        gl = ax.gridlines(linewidth=0.3, color="0.7", draw_labels=True)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = dict(fontsize=6.5, color=MUT)
        # tile-extent box, drawn directly in projection meters (exact,
        # since this projection preserves true distance from its center,
        # which IS the tile center).
        tb = tile_half_km * 1000.0
        ax.add_patch(mpatches.Rectangle(
            (-tb, -tb), 2 * tb, 2 * tb, transform=proj, fill=False,
            edgecolor=TILE_MARK_COLOR, linewidth=1.3, zorder=5))
        ax.plot(0.0, 0.0, marker="*", color=TILE_MARK_COLOR, markersize=10,
                markeredgecolor="white", markeredgewidth=0.5,
                transform=proj, zorder=6)
        ax.set_title("Tile location (+/-%g km): red box = this %g km tile"
                     % (half_km, tile_km), fontsize=9, color=INK)
        return ax
    # No cartopy: plain local equirectangular box, no coastlines.
    ax = fig.add_subplot(gridspec_slot)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(0.05, np.cos(np.radians(lat0)))
    ax.set_xlim(-half_km, half_km)
    ax.set_ylim(-half_km, half_km)
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Rectangle(
        (-tile_half_km, -tile_half_km), tile_km, tile_km, fill=False,
        edgecolor=TILE_MARK_COLOR, linewidth=1.3, zorder=5))
    ax.plot(0.0, 0.0, marker="*", color=TILE_MARK_COLOR, markersize=12,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.set_xlabel("east-west [km]", fontsize=9, color=MUT)
    ax.set_ylabel("north-south [km]", fontsize=9, color=MUT)
    ax.set_title("Tile location (lat=%.2f, lon=%.2f, +/-%g km) -- cartopy "
                 "not available, no coastlines" % (lat0, lon0, half_km),
                 fontsize=8.2, color=INK)
    _style(ax)
    return ax


def _panel_windspeed(ax, data):
    if "windspeed_ocean" not in data or data["windspeed_ocean"].size == 0:
        ax.text(0.5, 0.5, "no ocean-windspeed field in this tile",
                ha="center", va="center", fontsize=9, color=MUT,
                transform=ax.transAxes)
        _style(ax)
        return
    ws = np.array(data["windspeed_ocean"], dtype=float)
    ws_masked = np.ma.masked_where(ws <= -900, ws)
    lon_ocean = data.get("lon_ocean")
    lat_ocean = data.get("lat_ocean")
    if ws_masked.count() == 0:
        ax.text(0.5, 0.5, "no ocean cells in this tile\n(all land/ice)",
                ha="center", va="center", fontsize=9, color=MUT,
                transform=ax.transAxes)
        _style(ax)
        return
    if lon_ocean is not None and lat_ocean is not None:
        pm = ax.pcolormesh(lon_ocean, lat_ocean, ws_masked, cmap="viridis",
                            shading="nearest")
        ax.set_xlabel("longitude [deg]", fontsize=9, color=MUT)
        ax.set_ylabel("latitude [deg]", fontsize=9, color=MUT)
    else:
        pm = ax.imshow(ws_masked, origin="lower", cmap="viridis")
    cb = plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=MUT)
    cb.set_label("ERA5 windspeed [m/s]", fontsize=8, color=MUT)
    vmin, vmax = float(ws_masked.min()), float(ws_masked.max())
    ax.set_title("ERA5 ocean windspeed: %.2f .. %.2f m/s across tile "
                  "(spatially variable, not constant)" % (vmin, vmax),
                  fontsize=9, color=INK)
    _style(ax)


def _panel_geometry(ax, pooled, tile_mean_angles):
    angles = pooled["angles"]
    in_tile = pooled["in_tile"]
    sza, vza = angles["sza"][in_tile], angles["vza"][in_tile]
    ax.scatter(vza, sza, s=4.0, color="#4C78A8", alpha=0.4, linewidths=0)
    if "vza" in tile_mean_angles and "sza" in tile_mean_angles:
        ax.axvline(tile_mean_angles["vza"], color=INK, linewidth=1.0, linestyle="--",
                   alpha=0.7)
        ax.axhline(tile_mean_angles["sza"], color=INK, linewidth=1.0, linestyle="--",
                   alpha=0.7)
        ax.plot(tile_mean_angles["vza"], tile_mean_angles["sza"], "+", color="#c0392b",
                ms=12, mew=2.0, zorder=5, label="tile mean")
        ax.legend(loc="best", fontsize=7.5, frameon=False)
    ax.set_xlabel("VZA [deg]", fontsize=9, color=MUT)
    ax.set_ylabel("SZA [deg]", fontsize=9, color=MUT)
    ax.set_title("Geometry across pooled pixels (tile mean marked)",
                 fontsize=9.5, color=INK)
    _style(ax)


def _overlay_cloud_markers(ax, east_km, north_km, cloud_flag):
    """Draws a black 'x' at the center of every grid cell where
    cloud_flag is truthy -- shared by _panel_thin_atm_map() (the
    measured-vs-RT difference map) and _panel_surface_vs_toa() (the main
    QA figure's bottom-right panel), both already working in local
    (east_km, north_km) grid coordinates. Added 2026-10-23, Sebastian:
    "indicate cloud detections with a cross or something like that."
    cloud_flag may be None (nothing drawn, silently) or any boolean-
    castable array matching east_km/north_km's shape."""
    if cloud_flag is None:
        return
    flag = np.asarray(cloud_flag, dtype=bool)
    if flag.shape != np.asarray(east_km).shape or not flag.any():
        return
    ax.scatter(np.asarray(east_km)[flag], np.asarray(north_km)[flag],
              marker="x", s=30, color="black", linewidths=1.3, zorder=6,
              label="cloud candidate")


def _panel_surface_vs_toa(ax, refl_noatm, refl_app, lon_land, lat_land, lat0, lon0,
                          cloud_flag=None):
    """Surface-only (no-atmosphere, single-angle BRDF) reflectance vs.
    SHDOM's TOA apparent reflectance -- added 2026-10-22 per Sebastian's
    question: "the bottom row is suspiciously resembling the top row.
    Is that because of periodic boundary conditions? ... we should make
    that plot ... that actually shows the surface reflectance itself for
    comparison to the TOA reflectance that SHDOM spits out."

    BCFLAG=0 in the property file confirms SHDOM is run with periodic
    boundary conditions in both X and Y here. refl_noatm never goes
    through the periodic SHDOM solve at all (pure per-cell BRDF kernel
    algebra at the one tile-mean geometry) -- so if the north/south-edge
    similarity shows up in refl_noatm TOO, it's a property of the real
    surface/geometry, not a periodic-BC artifact; if it shows up ONLY in
    refl_app, that implicates periodic boundary conditions / multiple
    scattering in the SHDOM solve itself.

    Plotted in real local km via lon_land/lat_land (same orientation fix
    and reference origin as _panel_radiance, 2026-10-22) so this panel
    lines up visually with that one instead of risking a different
    (possibly flipped) index-based orientation of its own."""
    diff = refl_app - refl_noatm
    lat_ref = float(np.mean(lat_land))
    km_per_deg = 6378.1370 * np.pi / 180.0
    east_km = (lon_land - lon0) * km_per_deg * np.cos(np.radians(lat_ref))
    north_km = (lat_land - lat0) * km_per_deg
    vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
    im = ax.pcolormesh(east_km, north_km, diff, cmap="RdBu_r", shading="auto",
                       vmin=-vmax, vmax=vmax)
    _overlay_cloud_markers(ax, east_km, north_km, cloud_flag)
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=MUT)
    cb.set_label("refl_app - refl_noatm", fontsize=8, color=MUT)
    if cloud_flag is not None and np.asarray(cloud_flag, dtype=bool).any():
        ax.legend(loc="lower right", fontsize=7, frameon=False)

    def _edge_corr(field):
        south, north = field[:, 0], field[:, -1]
        if (south.size < 2 or not np.isfinite(south).all()
                or not np.isfinite(north).all() or np.std(south) == 0
                or np.std(north) == 0):
            return float("nan")
        return float(np.corrcoef(south, north)[0, 1])

    corr_toa = _edge_corr(refl_app)
    corr_sfc = _edge_corr(refl_noatm)
    ax.text(0.02, 0.98,
            "south-vs-north edge corr:\n  TOA (SHDOM): %.4f\n  surface "
            "(no-atm): %.4f" % (corr_toa, corr_sfc),
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=INK, bbox=dict(facecolor="white", alpha=0.85,
                                  edgecolor="#cfcfcf", linewidth=0.5))
    ax.set_xlabel("east-west [km]", fontsize=9, color=MUT)
    ax.set_ylabel("north-south [km]", fontsize=9, color=MUT)
    ax.set_title("TOA (SHDOM) vs. surface-only reflectance", fontsize=9.5,
                 color=INK)
    _style(ax)


# --thin-atm-diag (2026-10-21, Sebastian: "an additional diagnostic plot
# where the atmosphere is almost removed"): data['brdf'][...,6] land/ocean/
# ice codes (1=ocean, 3=land, 4=ice -- same convention _panel_radiance's
# `sfc_flag` already uses) get a small fixed categorical palette here, kept
# separate from INSIDE_COLOR/EDGE_COLOR above (those are footprint-
# classification semantics, not surface type, and reusing them would be
# confusing side by side).
_SFC_CODE_COLORS = {
    1.0: ("#4C78A8", "ocean"),
    3.0: ("#54A24B", "land"),
    4.0: ("#B279A2", "ice"),
}


def _panel_thin_atm_map(ax, refl, lon_land, lat_land, lat0, lon0, title,
                        vmin=None, vmax=None, cbar_label="apparent reflectance",
                        cmap="viridis", cloud_flag=None):
    """Same real-lon/lat-derived local-km orientation as _panel_radiance/
    _panel_surface_vs_toa, applied to an apparent-reflectance field (or,
    via `cbar_label`, any other per-cell field on the same lon_land/
    lat_land grid -- reused 2026-10-22 for the fvol/fgeo anisotropy
    context map, and again for the RT/measured/difference radiance maps
    in make_measured_vs_rt_plot() -- `cmap` made overridable there so the
    difference map can use a diverging colormap instead of the default
    sequential one)."""
    lat_ref = float(np.mean(lat_land))
    km_per_deg = 6378.1370 * np.pi / 180.0
    east_km = (lon_land - lon0) * km_per_deg * np.cos(np.radians(lat_ref))
    north_km = (lat_land - lat0) * km_per_deg
    im = ax.pcolormesh(east_km, north_km, refl, cmap=cmap,
                       shading="auto", vmin=vmin, vmax=vmax)
    _overlay_cloud_markers(ax, east_km, north_km, cloud_flag)
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=MUT)
    cb.set_label(cbar_label, fontsize=8, color=MUT)
    ax.set_xlabel("east-west [km]", fontsize=9, color=MUT)
    ax.set_ylabel("north-south [km]", fontsize=9, color=MUT)
    ax.set_title(title, fontsize=9.5, color=INK)
    if cloud_flag is not None and np.asarray(cloud_flag, dtype=bool).any():
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    _style(ax)


def _rayleigh_optical_depth_555(wavelength_nm=555.0):
    """Bodhaine et al. (1999) closed-form Rayleigh optical depth at sea-
    level standard pressure (1013.25 hPa), as a function of wavelength
    (nm). Illustrative only -- SHDOM's own molecular-scattering optical
    depth (from er3t's own absorption/Rayleigh tables, standard
    atmosphere) is the authoritative number used in the actual RT solve;
    this closed form is just a cheap, independent estimate for the
    analytic overlay curve below, not a replacement for it."""
    wl_um = wavelength_nm / 1000.0
    exponent = 3.916 + 0.074 * wl_um + 0.05 / wl_um
    return 0.00864 * wl_um ** (-exponent)


def _rayleigh_toa_reflectance_analytic(refl_surface, sza_deg, vza_deg,
                                       relative_azimuth_deg=0.0,
                                       tau_rayleigh=None,
                                       wavelength_nm=555.0):
    """Illustrative single-scattering, two-stream Rayleigh-atmosphere
    prediction for TOA reflectance vs. Lambertian surface reflectance --
    added 2026-10-22 per Sebastian, to overlay an analytic expectation on
    the thin-atm-diag scatter (full-atm vs. near-vacuum-atm reflectance),
    using the same "crossover/critical reflectance" argument discussed in
    chat: the atmosphere brightens a DARK surface (fixed path-radiance
    add-on dominates) but can DARKEN a BRIGHT one (two-way transmission
    loss on the surface term grows with refl_surface and eventually wins)
    -- standard result, e.g. Fraser & Kaufman (1985); Kaufman (1993)'s
    "critical surface reflectance" used in aerosol-over-land retrievals:

        refl_toa = refl_path + T_down * T_up * refl_surface / (1 - s * refl_surface)

    refl_path = tau * P(scattering_angle) / (4 * mu0 * muv)   -- single-
                scattering atmospheric path reflectance
    T_down/up = exp(-tau / mu0), exp(-tau / muv)              -- direct-
                beam two-way transmittance (ignores diffuse transmittance
                -- a known simplification at this tau)
    s         = 0.75 * tau                                    -- coarse
                closed-form estimate of the atmospheric spherical albedo
                (surface-atmosphere multiple-reflection term), not from a
                full adding/doubling calculation
    P(Theta)  = 0.75 * (1 + cos(Theta)**2)                    -- Rayleigh
                phase function

    NOT a substitute for SHDOM's own multiple-scattering/correlated-k
    calculation -- a cheap, independent check on the SHAPE and rough
    crossover point of the atmosphere's effect, not a precision match.
    `tau_rayleigh` defaults to `_rayleigh_optical_depth_555(wavelength_nm)`
    if not given explicitly (e.g. to compare against SHDOM's own number)."""
    tau = (tau_rayleigh if tau_rayleigh is not None
          else _rayleigh_optical_depth_555(wavelength_nm))
    mu0 = np.cos(np.radians(sza_deg))
    muv = np.cos(np.radians(vza_deg))
    cos_theta = (-mu0 * muv
                + np.sqrt(max(1.0 - mu0 ** 2, 0.0))
                * np.sqrt(max(1.0 - muv ** 2, 0.0))
                * np.cos(np.radians(relative_azimuth_deg)))
    phase = 0.75 * (1.0 + cos_theta ** 2)
    refl_path = tau * phase / (4.0 * mu0 * muv)
    t_down = np.exp(-tau / mu0)
    t_up = np.exp(-tau / muv)
    s = 0.75 * tau
    refl_surface = np.asarray(refl_surface, dtype=float)
    denom = 1.0 - s * refl_surface
    return refl_path + t_down * t_up * refl_surface / denom, tau


def _panel_thin_atm_scatter(ax, refl_app, refl_thin, refl_noatm, sfc_flag,
                            tile_mean_angles=None):
    """Per-cell full-atmosphere vs. near-vacuum-atmosphere reflectance,
    color-coded by land/ocean/ice (sfc_flag) and sized by the no-
    atmosphere BRDF-only surface reflectance (refl_noatm) -- Sebastian's
    request: "a scatter plot that is color-coded by surface reflectance
    and/or land/ocean." Both requested encodings are used together: color
    = surface category, marker size = refl_noatm magnitude (per-category
    5th-95th percentile stretch, so size is legible within each surface
    type rather than swamped by the ocean/land reflectance gap).

    A 1:1 reference line is drawn: per rt_shdom_vacuum_test.py's own
    physical prediction, real-atmosphere points should sit ABOVE this
    line almost everywhere (Rayleigh adds path radiance, so full-
    atmosphere should be brighter than near-vacuum) -- points below the
    line are the "smoking gun" case worth flagging to Hong."""
    refl_app_f = np.asarray(refl_app, dtype=float).ravel()
    refl_thin_f = np.asarray(refl_thin, dtype=float).ravel()
    sfc_f = (np.asarray(sfc_flag, dtype=float).ravel()
             if sfc_flag is not None else None)
    noatm_f = (np.asarray(refl_noatm, dtype=float).ravel()
               if refl_noatm is not None else None)
    finite = np.isfinite(refl_app_f) & np.isfinite(refl_thin_f)

    codes = np.unique(sfc_f[finite]) if sfc_f is not None else np.array([])
    if codes.size == 0:
        ax.scatter(refl_thin_f[finite], refl_app_f[finite], s=6.0,
                  color="#4C78A8", alpha=0.5, linewidths=0)
    else:
        for code in codes:
            color, label = _SFC_CODE_COLORS.get(code, (MUT, "code %g" % code))
            m = finite & (sfc_f == code)
            if not m.any():
                continue
            sizes = 6.0
            if noatm_f is not None:
                nf = noatm_f[m]
                nf_finite = np.isfinite(nf)
                if nf_finite.any():
                    lo, hi = np.nanpercentile(nf[nf_finite], [5, 95])
                    span = max(hi - lo, 1e-9)
                    sizes = 4.0 + 16.0 * np.clip((nf - lo) / span, 0.0, 1.0)
            ax.scatter(refl_thin_f[m], refl_app_f[m], s=sizes, color=color,
                      alpha=0.45, linewidths=0, label=label)

    lims = []
    for arr in (refl_thin_f[finite], refl_app_f[finite]):
        if arr.size:
            lims.extend([float(np.nanmin(arr)), float(np.nanmax(arr))])
    if lims:
        lo, hi = min(lims), max(lims)
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK,
               linewidth=1.0, linestyle="--", alpha=0.6, zorder=1,
               label="1:1 (no atm. effect)")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if tile_mean_angles and "sza" in tile_mean_angles and "vza" in tile_mean_angles:
            rel_az = (tile_mean_angles["vaa"] - tile_mean_angles["saa"]
                      if "vaa" in tile_mean_angles and "saa" in tile_mean_angles
                      else 0.0)
            rho_s_grid = np.linspace(max(lo - pad, 0.0), hi + pad, 100)
            rho_toa_pred, tau_used = _rayleigh_toa_reflectance_analytic(
                rho_s_grid, tile_mean_angles["sza"], tile_mean_angles["vza"],
                relative_azimuth_deg=rel_az)
            ax.plot(rho_s_grid, rho_toa_pred, color="#c0392b", linewidth=1.3,
                   linestyle="-.", zorder=2,
                   label="Rayleigh single-scatter prediction "
                         "(tau=%.3f, illustrative)" % tau_used)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_xlabel("near-vacuum-atm reflectance (thin-atm-diag)", fontsize=9,
                 color=MUT)
    ax.set_ylabel("full-atmosphere reflectance (refl_app)", fontsize=9,
                 color=MUT)
    ax.set_title("Per-cell: full atm vs. near-vacuum atm (color=surface "
                 "type, size~surface reflectance) -- above 1:1 = atm "
                 "brightens, as expected", fontsize=8.6, color=INK)
    _style(ax)


def _panel_thin_atm_scatter_by_anisotropy(ax, refl_app, refl_thin,
                                          fvol_fgeo_ratio, sfc_flag,
                                          tile_mean_angles=None):
    """Same full-atm-vs-near-vacuum scatter as _panel_thin_atm_scatter
    (same x/y, same 1:1 and Rayleigh-prediction overlay), but colored
    continuously by each LAND cell's own fvol/fgeo kernel-anisotropy
    ratio (data['brdf'][...,2]/data['brdf'][...,1]) instead of by
    surface category -- added 2026-10-22 per Sebastian's request, after
    his --solver 3D rerun tightened the main panel's scatter back onto
    the analytic prediction line (3D run: "did tighten the scatter
    considerably, now it looks completely in line with your analytical
    prediction line"). This panel is the direct test of the BRDF-
    anisotropy hypothesis for whatever scatter remains/reappears (e.g.
    under --solver IPA): if cells with a similar near-vacuum reflectance
    but a different fvol/fgeo ratio land at visibly different heights
    above/below the theoretical line -- i.e. a color GRADIENT across the
    scatter, not just noise -- that supports kernel SHAPE (not just
    magnitude) driving the spread. Sebastian's own prior (2026-10-22): he
    considers this -- and per-cell VZA/SZA variation across the tile --
    unlikely to be large enough to explain the scatter, favoring instead
    a solver-specific IPA limitation (Hong has mentioned SHDOM doesn't
    handle IPA well; see docs/L1B_TO_L2_ROADMAP.md "Hong input" for this
    added to the question list). Ocean/ice cells (no meaningful fvol/fgeo
    kernel split for Cox-Munk, or where fgeo<=0 makes the ratio
    undefined) are plotted small and gray for context, not colored."""
    refl_app_f = np.asarray(refl_app, dtype=float).ravel()
    refl_thin_f = np.asarray(refl_thin, dtype=float).ravel()
    ratio_f = np.asarray(fvol_fgeo_ratio, dtype=float).ravel()
    sfc_f = (np.asarray(sfc_flag, dtype=float).ravel()
             if sfc_flag is not None else None)
    finite = np.isfinite(refl_app_f) & np.isfinite(refl_thin_f)
    is_land = (sfc_f == 3.0) if sfc_f is not None else np.ones(finite.shape, dtype=bool)
    land_m = finite & is_land & np.isfinite(ratio_f)
    other_m = finite & ~land_m

    if other_m.any():
        ax.scatter(refl_thin_f[other_m], refl_app_f[other_m], s=5.0,
                  color="#cccccc", alpha=0.4, linewidths=0,
                  label="ocean/ice/n.a.", zorder=1)
    sc = None
    if land_m.any():
        lo, hi = np.nanpercentile(ratio_f[land_m], [5, 95])
        sc = ax.scatter(refl_thin_f[land_m], refl_app_f[land_m], s=10.0,
                       c=ratio_f[land_m], cmap="viridis", vmin=lo, vmax=hi,
                       alpha=0.8, linewidths=0, zorder=2)

    lims = []
    for arr in (refl_thin_f[finite], refl_app_f[finite]):
        if arr.size:
            lims.extend([float(np.nanmin(arr)), float(np.nanmax(arr))])
    if lims:
        lo_xy, hi_xy = min(lims), max(lims)
        pad = 0.05 * max(hi_xy - lo_xy, 1e-6)
        ax.plot([lo_xy - pad, hi_xy + pad], [lo_xy - pad, hi_xy + pad],
               color=INK, linewidth=1.0, linestyle="--", alpha=0.6,
               zorder=1, label="1:1 (no atm. effect)")
        ax.set_xlim(lo_xy - pad, hi_xy + pad)
        ax.set_ylim(lo_xy - pad, hi_xy + pad)
        if tile_mean_angles and "sza" in tile_mean_angles and "vza" in tile_mean_angles:
            rel_az = (tile_mean_angles["vaa"] - tile_mean_angles["saa"]
                      if "vaa" in tile_mean_angles and "saa" in tile_mean_angles
                      else 0.0)
            rho_s_grid = np.linspace(max(lo_xy - pad, 0.0), hi_xy + pad, 100)
            rho_toa_pred, tau_used = _rayleigh_toa_reflectance_analytic(
                rho_s_grid, tile_mean_angles["sza"], tile_mean_angles["vza"],
                relative_azimuth_deg=rel_az)
            ax.plot(rho_s_grid, rho_toa_pred, color="#c0392b", linewidth=1.3,
                   linestyle="-.", zorder=3,
                   label="Rayleigh single-scatter prediction "
                         "(tau=%.3f, illustrative)" % tau_used)
    if sc is not None:
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.5, colors=MUT)
        cbar.set_label("fvol/fgeo ratio (land BRDF anisotropy)", fontsize=8,
                      color=MUT)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_xlabel("near-vacuum-atm reflectance (thin-atm-diag)", fontsize=9,
                 color=MUT)
    ax.set_ylabel("full-atmosphere reflectance (refl_app)", fontsize=9,
                 color=MUT)
    ax.set_title("Same scatter, colored by BRDF anisotropy (land fvol/"
                 "fgeo) -- a color gradient across the cloud supports "
                 "kernel SHAPE driving the spread", fontsize=8.6, color=INK)
    _style(ax)


def _panel_noatm_vs_thin_scatter(ax, refl_noatm, refl_thin, sfc_flag):
    """Control panel added 2026-10-22, per Sebastian's --solver IPA run
    ("the atmosphere is making NO impact, relative to the level of noise
    added for this mode, it's crazy" / "this is unusable"): compares the
    cheap, no-RT-at-all BRDF-kernel reflectance (refl_noatm) against the
    near-vacuum-atmosphere RT reflectance (refl_thin) -- both are
    evaluated on the exact same surface-BRDF grid and should both be
    ~atmosphere-free, so this tells apart two different explanations for
    a noisy/scattered main panel (refl_app vs refl_thin, above):

    - if THIS panel is already loose/scattered (even with both sides
      near-zero-atmosphere), the scatter is intrinsic to the RT solve
      itself at this solver setting (e.g. a convergence/quadrature issue
      specific to --solver IPA) and has nothing to do with the real
      atmosphere.
    - if THIS panel is tight/near-1:1, the scatter in the main panel is a
      genuine EFFECT of adding the real atmosphere on top of per-cell
      surface anisotropy -- plausible hypothesis discussed in chat:
      non-Lambertian BRDF kernel SHAPE (the fvol/fgeo mix, not just
      magnitude) interacts with diffuse sky light differently cell by
      cell; 3D's horizontal photon transport partially smooths this over
      by blending light across neighboring columns, but IPA (no
      horizontal coupling at all) does not, so it would show up as
      exactly this kind of unmasked point-cloud scatter once the
      atmosphere is turned on -- real physics, not solver noise."""
    noatm_f = np.asarray(refl_noatm, dtype=float).ravel()
    thin_f = np.asarray(refl_thin, dtype=float).ravel()
    sfc_f = (np.asarray(sfc_flag, dtype=float).ravel()
             if sfc_flag is not None else None)
    finite = np.isfinite(noatm_f) & np.isfinite(thin_f)
    codes = np.unique(sfc_f[finite]) if sfc_f is not None else np.array([])
    if codes.size == 0:
        ax.scatter(noatm_f[finite], thin_f[finite], s=6.0, color="#4C78A8",
                  alpha=0.5, linewidths=0)
    else:
        for code in codes:
            color, label = _SFC_CODE_COLORS.get(code, (MUT, "code %g" % code))
            m = finite & (sfc_f == code)
            if m.any():
                ax.scatter(noatm_f[m], thin_f[m], s=6.0, color=color,
                          alpha=0.45, linewidths=0, label=label)
    lims = []
    for arr in (noatm_f[finite], thin_f[finite]):
        if arr.size:
            lims.extend([float(np.nanmin(arr)), float(np.nanmax(arr))])
    if lims:
        lo, hi = min(lims), max(lims)
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK,
               linewidth=1.0, linestyle="--", alpha=0.6, zorder=1,
               label="1:1")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_xlabel("no-atm BRDF-kernel reflectance (refl_noatm)", fontsize=9,
                 color=MUT)
    ax.set_ylabel("near-vacuum-atm RT reflectance (thin-atm-diag)",
                 fontsize=9, color=MUT)
    ax.set_title("Control: both ~atm-free -- tight here = main panel's "
                 "scatter is a real atmosphere effect, not solver noise",
                 fontsize=8.4, color=INK)
    _style(ax)


def _panel_measured_vs_rt_scatter(ax, rt_vals, meas_vals, n_px_per_cell=None,
                                  quantity_name="radiance",
                                  units="[W m-2 sr-1 nm-1]", cloud_flag=None):
    """Per-RT-grid-cell scatter: x=measured, y=RT -- Sebastian, 2026-10-22:
    "a scatter plot of 'RT' vs. 'measured'." Point size encodes how many
    real camera pixels landed in that cell (n_px_per_cell, when given) --
    a cell averaged over many camera pixels is a more trustworthy
    comparison point than one with just a couple, so this is visible at a
    glance rather than every point looking equally solid. A 1:1 reference
    line is drawn; points are a real-scene comparison, not yet a
    validated algorithm metric -- RT is clear-SKY only, so real cloud in
    the L1X scene shows up here as scatter off the 1:1 line BY DESIGN,
    not as evidence of a bug.

    quantity_name/units (added 2026-10-22, so this same panel can be
    reused for the reflectance-converted comparison too -- Sebastian:
    "we should actually convert the radiances into reflectances. That
    would make it fairly obvious."): purely cosmetic (axis labels/
    title) -- the comparison math doesn't care what units rt_vals/
    meas_vals are in.

    cloud_flag (added 2026-10-23, Sebastian: "color the cloudy points in
    another color"): optional boolean (or 0/1-castable) array, same
    shape as rt_vals/meas_vals, True/1 where run_dev_step1.py's e-fold
    threshold flagged that cell as a cloud candidate. When given (and
    shaped right), cloud-candidate points are drawn in a second color
    instead of the usual single blue -- None (default) keeps the
    original single-color behavior."""
    rt_f = np.asarray(rt_vals, dtype=float).ravel()
    meas_f = np.asarray(meas_vals, dtype=float).ravel()
    finite = np.isfinite(rt_f) & np.isfinite(meas_f)

    sizes = 10.0
    if n_px_per_cell is not None:
        n_f = np.asarray(n_px_per_cell, dtype=float).ravel()
        n_fin = n_f[finite]
        if n_fin.size and np.ptp(n_fin) > 0:
            lo, hi = np.nanpercentile(n_fin, [5, 95])
            span = max(hi - lo, 1e-9)
            sizes = 6.0 + 24.0 * np.clip((n_f - lo) / span, 0.0, 1.0)

    cloud_f = (np.asarray(cloud_flag, dtype=bool).ravel()
              if cloud_flag is not None else None)
    if cloud_f is not None and cloud_f.shape == finite.shape:
        clear_mask = finite & ~cloud_f
        cloud_mask = finite & cloud_f
        ax.scatter(meas_f[clear_mask], rt_f[clear_mask],
                  s=(sizes[clear_mask] if np.ndim(sizes) else sizes),
                  color="#4C78A8", alpha=0.55, linewidths=0,
                  label="clear-sky candidate")
        ax.scatter(meas_f[cloud_mask], rt_f[cloud_mask],
                  s=(sizes[cloud_mask] if np.ndim(sizes) else sizes),
                  color="#E45756", alpha=0.7, linewidths=0,
                  label="cloud candidate")
    else:
        ax.scatter(meas_f[finite], rt_f[finite],
                  s=(sizes[finite] if np.ndim(sizes) else sizes),
                  color="#4C78A8", alpha=0.55, linewidths=0)

    lims = []
    for arr in (meas_f[finite], rt_f[finite]):
        if arr.size:
            lims.extend([float(np.nanmin(arr)), float(np.nanmax(arr))])
    stats_txt = "n/a"
    if lims:
        lo, hi = min(lims), max(lims)
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK,
               linewidth=1.0, linestyle="--", alpha=0.6, zorder=1,
               label="1:1")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if finite.sum() >= 2:
            diff = meas_f[finite] - rt_f[finite]
            bias = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            corr = (float(np.corrcoef(rt_f[finite], meas_f[finite])[0, 1])
                   if np.std(rt_f[finite]) > 0 and np.std(meas_f[finite]) > 0
                   else float("nan"))
            stats_txt = ("bias(meas-RT)=%.3g\nRMSE=%.3g\ncorr=%.3f\nn=%d"
                        % (bias, rmse, corr, int(finite.sum())))
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=8,
           color=INK, va="top", ha="left",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.75,
                     edgecolor="#cccccc"))
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    xlabel = ("measured %s (L1X, binned to RT grid) %s"
             % (quantity_name, units)).strip()
    ylabel = ("RT (clear-sky SHDOM) %s %s" % (quantity_name, units)).strip()
    ax.set_xlabel(xlabel, fontsize=9, color=MUT)
    ax.set_ylabel(ylabel, fontsize=9, color=MUT)
    ax.set_title("Per-cell: RT vs. measured %s (point size ~ camera pixel "
                 "count in that cell)" % quantity_name, fontsize=8.8, color=INK)
    _style(ax)


def _panel_histogram_with_sza(ax, rt_vals, meas_vals, quantity_name="reflectance",
                              units="", sza_camera_deg=None,
                              sza_source_deg=None, cloud_info=None):
    """PDF of the per-cell (measured - RT) difference over the same RT
    grid cells, with the ATBD Figure 6 e-fold adaptive cloud threshold
    overlaid -- changed 2026-10-23 from overlaid separate RT/measured
    histograms to this difference distribution, per Sebastian: now that
    reflectances are SZA-matched (camera-side and MODIS-source-side each
    normalized by their OWN tile-mean SZA) and line up well in genuinely
    clear-sky cells, this difference IS the clear-sky/cloud test
    statistic itself -- same sign convention as Hong's l2_cam cld.py
    (rad_diff = data_run['rad_cam'] - data_run['rad_rtm'], positive =
    brighter than clear-sky RT predicts = cloud candidate), so this panel
    is directly comparable to his cld_msk = (rad_diff > threshold) logic
    (cal_threshold_efold/_std/_gauss, same file).

    Extended 2026-10-23 (Sebastian: "add the Hong-approach directly to
    that histogram ... replicate something similar rather than simply
    importing") with run_tile_rt.cloud_threshold_efold()'s result,
    `cloud_info` -- an independent reimplementation of Hong's
    cal_threshold_efold / the ATBD's own Figure 6 recipe (p.8): find the
    PDF's peak (x_max, f_max), then walk the PDF from the peak toward
    the brighter/more-cloud-like tail until it falls to f_max/e (or /2,
    /3 -- `scale`) -- that's (x_e, f_e), and threshold=x_e. Cells with
    diff > threshold are flagged cloud candidates. When `cloud_info` is
    given, this panel redraws the EXACT histogram (same bin_edges/hist,
    both already density-normalized) the threshold was computed from,
    rather than rebinning with different edges here, and colors the
    flagged-cloud bins separately -- see that function's own docstring.
    Falls back to a plain (no-threshold) density histogram, rebinned
    locally, when `cloud_info` is None (too few paired cells -- see
    run_dev_step1.py's >=10-cell guard).

    Also reports the camera's own tile-mean SZA against the real MODIS
    source scene's tile-mean SZA in a text box -- Sebastian, 2026-10-22:
    "plot radiance histograms and report the target and source SZA."""
    rt_f = np.asarray(rt_vals, dtype=float).ravel()
    meas_f = np.asarray(meas_vals, dtype=float).ravel()
    finite = np.isfinite(rt_f) & np.isfinite(meas_f)
    diff = meas_f[finite] - rt_f[finite]

    if cloud_info is not None:
        bin_edges = np.asarray(cloud_info["bin_edges"], dtype=float)
        hist = np.asarray(cloud_info["hist"], dtype=float)
        bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        threshold = cloud_info["threshold"]
        cloudy = bin_center > threshold
        widths = np.diff(bin_edges)
        ax.bar(bin_edges[:-1], np.where(cloudy, 0.0, hist), width=widths,
              align="edge", color="#4C78A8", alpha=0.75,
              label="clear-sky candidate")
        ax.bar(bin_edges[:-1], np.where(cloudy, hist, 0.0), width=widths,
              align="edge", color="#E45756", alpha=0.75,
              label="cloud candidate")
        ax.plot([cloud_info["x_max"]], [cloud_info["f_max"]], marker="o",
               color="0.35", markersize=6, linestyle="none",
               label="(x_max, f_max)")
        ax.axhline(cloud_info["f_max"], color="0.35", linewidth=0.9,
                  linestyle="--", alpha=0.7)
        ax.plot([cloud_info["x_e"]], [cloud_info["f_e"]], marker="D",
               color="#2CA02C", markersize=6, linestyle="none",
               label="(x_e, f_e) = peak/%.3g" % cloud_info["scale"])
        ax.axhline(cloud_info["f_e"], color="#2CA02C", linewidth=0.9,
                  linestyle=":", alpha=0.8)
        ax.axvline(threshold, color="#E45756", linewidth=1.3,
                  linestyle="-", alpha=0.85,
                  label="threshold (cloud if diff>this)")
        cloud_txt = ("e-fold threshold (ATBD Fig. 6 / Hong cal_threshold_efold)\n"
                    "x_max=%.3g  f_max=%.3g\nx_e=%.3g  threshold=%.3g\n"
                    "cloud fraction=%.3f (%d/%d cells)"
                    % (cloud_info["x_max"], cloud_info["f_max"],
                       cloud_info["x_e"], threshold,
                       cloud_info["cloud_fraction"],
                       cloud_info["n_cells_cloud"],
                       cloud_info["n_cells_compared"]))
        ax.text(0.03, 0.97, cloud_txt, transform=ax.transAxes, fontsize=7.3,
               color=INK, va="top", ha="left",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.82,
                         edgecolor="#cccccc"))
        ax.legend(loc="lower left", fontsize=6.6, frameon=False)
    elif diff.size:
        lo, hi = float(np.nanmin(diff)), float(np.nanmax(diff))
        bins = np.linspace(lo, hi, 31) if hi > lo else 31
        ax.hist(diff, bins=bins, density=True, color="#4C78A8", alpha=0.7)

    ax.axvline(0.0, color=INK, linewidth=1.0, linestyle="--", alpha=0.6)

    xlabel = ("measured - RT %s %s" % (quantity_name, units)).strip()
    ax.set_xlabel(xlabel, fontsize=9, color=MUT)
    ax.set_ylabel("probability density", fontsize=9, color=MUT)
    ax.set_title("Per-cell (measured - RT) %s PDF (positive = cloud "
                 "candidate)" % quantity_name, fontsize=8.8, color=INK)

    sza_cam_txt = ("%.2f deg" % sza_camera_deg
                  if sza_camera_deg is not None else "n/a")
    sza_src_txt = ("%.2f deg" % sza_source_deg
                  if sza_source_deg is not None else "n/a")
    sza_txt = ("SZA (tile-mean):\n  camera=%s\n  MODIS source=%s"
              % (sza_cam_txt, sza_src_txt))
    ax.text(0.97, 0.97, sza_txt, transform=ax.transAxes, fontsize=8,
           color=INK, va="top", ha="right",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                     edgecolor="#cccccc"))
    _style(ax)


def make_measured_vs_rt_plot(data, rad, rad_meas, n_px_per_cell, lat0, lon0,
                             out_path, refl_app=None, refl_meas=None,
                             sza_camera_deg=None, sza_source_deg=None,
                             cloud_info=None, cloud_flag_grid=None, verbose=True):
    """6-panel PNG (2x3, same house style as make_thin_atm_diag_plot())
    comparing RT-calculated and "measured" (L1X) radiance/reflectance
    over the tile -- added 2026-10-22, Sebastian: "in addition to the
    RT-calculated TOA radiances, we also see the 'measured' radiances,
    as well as a difference ... a scatter plot of 'RT' vs. 'measured'."
    Extended the same day (follow-up request) to also show radiance/
    reflectance histograms, the camera vs. MODIS-source SZA, and a
    reflectance-converted comparison ("we should actually convert the
    radiances into reflectances. That would make it fairly obvious [that
    the measured land looks like RT's bright clouds/snow]. we could
    still have a mismatch in SZA on top of that."). A SEPARATE PNG from
    make_qa_plots()'s own figure, as asked, so the existing QA figure
    (which has no measured-radiance panel at all, and is still exactly
    as useful for a plain, non-L1X L1B) is untouched.

    (A) RT radiance (same field as make_qa_plots()'s own radiance panel,
        just without that panel's coastline overlay, for a clean shared
        color scale against panel B).
    (B) "Measured" radiance -- rad_meas, binned from the L1X file's real
        camera Radiance field onto this same RT grid (see
        run_tile_rt.bin_measured_radiance_to_grid()) -- SAME color scale
        as panel A, so a visual match/mismatch in brightness is directly
        comparable, not an artifact of two different scales.
    (C) Difference (measured - RT), diverging colormap centered at 0 --
        where the real scene has cloud (this run's RT is clear-sky-only,
        cam_cloud_fraction=0 placeholder), expect RT to read LOWER than
        the real/measured value over a bright cloud.
    (D) Per-cell scatter, RT vs. measured RADIANCE, point size ~
        camera-pixel count backing that cell.
    (E) Per-cell scatter, RT vs. measured REFLECTANCE (refl_app/
        refl_meas) -- only drawn if both are given. refl_meas needs a
        real MODIS-source SZA (see apps/l2/build_l1x.py's
        l1x_modis_source_sza_mean_deg attribute, only present in an L1X
        file built after 2026-10-22); refl_app is run_tile_rt.tile()'s
        own existing RT-side conversion, already computed upstream for
        other purposes. A land-surface mismatch that looks like "no
        relationship" in radiance (e.g. because a stale-BRDF or SZA
        offset shifts one side's whole brightness scale) can look
        cleaner here, or conversely confirm it is a genuine scene/
        texture mismatch rather than a unit/geometry offset.
    (F) PDF of the per-cell (measured - RT) reflectance difference
        (falls back to radiance if refl_app/refl_meas aren't both
        available), with the ATBD Figure 6 / Hong cal_threshold_efold
        e-fold adaptive cloud threshold overlaid when `cloud_info` is
        given (run_tile_rt.cloud_threshold_efold(), computed by the
        caller in run_dev_step1.py) -- see _panel_histogram_with_sza()'s
        own docstring. Also a text box reporting the camera's own
        tile-mean SZA against the real MODIS source's tile-mean SZA.

    This is read-only/diagnostic, same as make_qa_plots() -- it doesn't
    change any computed value, only visualizes its inputs. Both
    refl_app and refl_meas use run_tile_rt.F0_555NM, a documented
    PLACEHOLDER solar-flux constant (see run_dev_step1.py's own
    provenance.f0_555nm_note) -- fine for an apples-to-apples RT-vs-
    measured comparison (the same constant cancels out of the
    comparison to first order), not yet an absolute reflectance
    validated against anything else."""
    both = np.concatenate([np.asarray(rad, dtype=float).ravel(),
                           np.asarray(rad_meas, dtype=float).ravel()])
    finite_both = both[np.isfinite(both)]
    vmin = float(finite_both.min()) if finite_both.size else None
    vmax = float(finite_both.max()) if finite_both.size else None

    diff = rad_meas - rad
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size:
        dmax = float(np.nanpercentile(np.abs(finite_diff), 98))
        dmax = dmax if dmax > 0 else 1e-9
    else:
        dmax = 1e-9

    have_refl = refl_app is not None and refl_meas is not None

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.4))
    _panel_thin_atm_map(axes[0, 0], rad, data["lon_land"], data["lat_land"],
                       lat0, lon0, "RT radiance (clear-sky SHDOM)",
                       vmin=vmin, vmax=vmax, cmap="inferno",
                       cbar_label="radiance [W m-2 sr-1 nm-1]")
    _panel_thin_atm_map(axes[0, 1], rad_meas, data["lon_land"], data["lat_land"],
                       lat0, lon0, "Measured radiance (L1X, binned to RT grid)",
                       vmin=vmin, vmax=vmax, cmap="inferno",
                       cbar_label="radiance [W m-2 sr-1 nm-1]")
    _panel_thin_atm_map(axes[0, 2], diff, data["lon_land"], data["lat_land"],
                       lat0, lon0, "Difference (measured - RT)",
                       vmin=-dmax, vmax=dmax, cmap="RdBu_r",
                       cbar_label="radiance diff [W m-2 sr-1 nm-1]",
                       cloud_flag=cloud_flag_grid)
    _panel_measured_vs_rt_scatter(axes[1, 0], rad, rad_meas, n_px_per_cell,
                                 quantity_name="radiance",
                                 units="[W m-2 sr-1 nm-1]",
                                 cloud_flag=cloud_flag_grid)

    if have_refl:
        _panel_measured_vs_rt_scatter(axes[1, 1], refl_app, refl_meas,
                                     n_px_per_cell, quantity_name="reflectance",
                                     units="", cloud_flag=cloud_flag_grid)
        _panel_histogram_with_sza(axes[1, 2], refl_app, refl_meas,
                                 quantity_name="reflectance", units="",
                                 sza_camera_deg=sza_camera_deg,
                                 sza_source_deg=sza_source_deg,
                                 cloud_info=cloud_info)
    else:
        axes[1, 1].text(
            0.5, 0.5,
            "reflectance comparison unavailable\n(no MODIS-source SZA -- "
            "rerun build_l1x.py to get\nl1x_modis_source_sza_mean_deg)",
            ha="center", va="center", fontsize=9, color=MUT,
            transform=axes[1, 1].transAxes)
        axes[1, 1].set_axis_off()
        _panel_histogram_with_sza(axes[1, 2], rad, rad_meas,
                                 quantity_name="radiance",
                                 units="[W m-2 sr-1 nm-1]",
                                 sza_camera_deg=sza_camera_deg,
                                 sza_source_deg=sza_source_deg,
                                 cloud_info=cloud_info)

    n_hit = int(np.isfinite(rad_meas).sum())
    fig.suptitle("measured-vs-RT: lat=%.4f, lon=%.4f -- %d/%d RT grid cell(s) "
                "have a real L1X measurement" % (lat0, lon0, n_hit, rad.size),
                fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print("Wrote measured-vs-RT figure: %s" % out_path)
    return out_path


def make_thin_atm_diag_plot(data, refl_app, refl_thin, refl_noatm, lat0, lon0,
                            out_path, verbose=True):
    """6-panel diagnostic PNG for --thin-atm-diag (2x3): (A) apparent
    reflectance with the real atmosphere (refl_app); (B) with it almost
    entirely removed (refl_thin, same SHDOM/BRDF/geometry/BC solve
    otherwise); (C) context map of each land cell's own fvol/fgeo BRDF-
    kernel anisotropy ratio, added 2026-10-22; (D) the per-cell full-atm-
    vs-near-vacuum scatter with an analytic Rayleigh crossover-
    reflectance overlay; (E) the no-atm-vs-near-vacuum CONTROL scatter
    added 2026-10-22 (Sebastian's --solver IPA run looked too noisy to
    use -- see _panel_noatm_vs_thin_scatter's docstring for what panel E
    tells apart); (F) panel D's same scatter re-colored by the fvol/fgeo
    ratio instead of surface category, added 2026-10-22 per Sebastian
    ("for the future, please do add coloring by the fvol/fgeo ratio") --
    see _panel_thin_atm_scatter_by_anisotropy's docstring. Requested
    2026-10-21 (Sebastian, for the Hong email) so "the atmosphere itself"
    can be visually separated from the periodic-BC/solver effects
    already visible in the main QA figure's surface-vs-TOA panel."""
    sfc_flag = data["brdf"][..., 6] if "brdf" in data else None
    # fvol/fgeo land-BRDF-kernel anisotropy ratio, on the same lon_land/
    # lat_land grid as everything else here (data["brdf"][...,1]=fgeo,
    # [...,2]=fvol -- see run_tile_rt.py's gen_brdf_shdom fixed-grid
    # writeback). NaN where fgeo<=0 (non-land, or a degenerate kernel)
    # so it never gets silently plotted as a bogus ratio.
    fgeo = data["brdf"][..., 1] if "brdf" in data else None
    fvol = data["brdf"][..., 2] if "brdf" in data else None
    if fgeo is not None and fvol is not None:
        fvol_fgeo_ratio = np.where(fgeo > 0, fvol / np.where(fgeo > 0, fgeo, 1.0),
                                   np.nan)
    else:
        fvol_fgeo_ratio = None
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.4))
    both = np.concatenate([np.asarray(refl_app, dtype=float).ravel(),
                           np.asarray(refl_thin, dtype=float).ravel()])
    finite_both = both[np.isfinite(both)]
    vmin = float(finite_both.min()) if finite_both.size else None
    vmax = float(finite_both.max()) if finite_both.size else None
    _panel_thin_atm_map(axes[0, 0], refl_app, data["lon_land"],
                       data["lat_land"], lat0, lon0,
                       "Full atmosphere (refl_app)", vmin=vmin, vmax=vmax)
    _panel_thin_atm_map(axes[0, 1], refl_thin, data["lon_land"],
                       data["lat_land"], lat0, lon0,
                       "Atmosphere almost removed (thin-atm-diag)",
                       vmin=vmin, vmax=vmax)
    if fvol_fgeo_ratio is not None and np.isfinite(fvol_fgeo_ratio).any():
        finite_ratio = fvol_fgeo_ratio[np.isfinite(fvol_fgeo_ratio)]
        r_lo, r_hi = np.nanpercentile(finite_ratio, [5, 95])
        _panel_thin_atm_map(axes[0, 2], fvol_fgeo_ratio, data["lon_land"],
                           data["lat_land"], lat0, lon0,
                           "Land BRDF anisotropy (fvol/fgeo)",
                           vmin=r_lo, vmax=r_hi,
                           cbar_label="fvol/fgeo ratio")
    else:
        axes[0, 2].axis("off")
    _panel_thin_atm_scatter(axes[1, 0], refl_app, refl_thin, refl_noatm,
                           sfc_flag,
                           tile_mean_angles=data.get("tile_mean_angles"))
    _panel_noatm_vs_thin_scatter(axes[1, 1], refl_noatm, refl_thin, sfc_flag)
    _panel_thin_atm_scatter_by_anisotropy(
        axes[1, 2], refl_app, refl_thin, fvol_fgeo_ratio, sfc_flag,
        tile_mean_angles=data.get("tile_mean_angles"))
    fig.suptitle("thin-atm-diag: lat=%.4f, lon=%.4f -- isolating the "
                "atmosphere's own contribution" % (lat0, lon0),
                fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print("Wrote thin-atm-diag figure: %s" % out_path)
    return out_path


def make_qa_plots(data, pooled, rad, footprint_info, tile_km, out_path,
                   refl_noatm=None, refl_app=None, cloud_flag_grid=None,
                   verbose=True):
    """Write the 5-panel QA figure to out_path (PNG). `data`/`pooled`/`rad`
    are exactly tile()'s and rt_shdom()'s own return values -- no
    reloading from disk, this is meant to be called in-process right
    after rt_shdom() returns, from run_dev_step1.py."""
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.32)

    _panel_footprint_map(fig.add_subplot(gs[0, 0]), pooled, footprint_info, tile_km)
    sfc_flag = data["brdf"][..., 6] if "brdf" in data else None
    _panel_radiance(fig.add_subplot(gs[0, 1]), rad, data["lon_land"],
                    data["lat_land"], pooled["lat0"], pooled["lon0"],
                    sfc_flag=sfc_flag)
    _panel_context_map(fig, gs[0, 2], pooled["lat0"], pooled["lon0"], tile_km=tile_km)
    _panel_windspeed(fig.add_subplot(gs[1, 0]), data)
    _panel_geometry(fig.add_subplot(gs[1, 1]), pooled, data["tile_mean_angles"])
    if refl_noatm is not None and refl_app is not None:
        _panel_surface_vs_toa(fig.add_subplot(gs[1, 2]), refl_noatm, refl_app,
                              data["lon_land"], data["lat_land"],
                              pooled["lat0"], pooled["lon0"],
                              cloud_flag=cloud_flag_grid)
    # else: gs[1, 2] left blank (room for the planned exposure-gap /
    # footprint-fill-fraction diagnostic panel, requested 2026-10-22, not
    # yet implemented).

    fig.suptitle("dev-step-1 QA: tile at lat=%.4f, lon=%.4f, %g km, %s UTC"
                 % (pooled["lat0"], pooled["lon0"], tile_km,
                    str(data["tile_overpass_utc"])),
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print("Wrote QA figure: %s" % out_path)
    return out_path
