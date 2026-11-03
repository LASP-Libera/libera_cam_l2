#!/usr/bin/env python3
"""Interactive orbit quicklook for hand-picking a dev-step-1 tile (Sebastian,
2026-10-20/21).

v1 plotted every (strided) radiometer footprint across the whole swath,
flooding the map. v2 reduced the swath per ~5.5 s scan cycle to three thin
traces (center/left/right). v3 fixed a NaN-driven cartopy crash and added
Latitude + land/ocean panels. v4 (this version), per feedback after v3:

- **Dropped the left/right swath envelope entirely.** Only the near-nadir
  ("center", min-VZA-per-scan-cycle) track is computed and plotted now --
  simpler maps, same SZA color scale as before.
- **Added click-to-pick on the three map panels themselves**, not just the
  bottom time-series panels. A click on any of the three Orthographic
  views is inverse-projected back to lon/lat (via that panel's own
  projection) and matched to the nearest near-nadir track point by
  great-circle distance. Both click paths (map or bottom panel) go
  through the same pick/save/close logic.
- **Added a color-coded link between the three maps and the bottom
  panels.** Each map gets its own accent color (title text + a colored
  globe border); the near-nadir track crosses in and out of that map's
  visible hemisphere (great-circle angle < 90 deg from the map's center
  point) one or more separate times as the orbit progresses -- each such
  pass is drawn as its own short horizontal bar, in that map's color, in
  a row stacked near the top of all three bottom panels (v7: v6 wrongly
  joined every visible pass into a single min-to-max span; now each
  contiguous in-view segment gets its own bar).

Per-~5.5s-scan-cycle reduction and the NaN-geo-validity filter (~22% of
this granule's RAD-4CH samples are NaN -- the mechanism sweeps past the
Earth's limb into space once per rotation; see elevation_sweep_timing.py
from earlier this project) are unchanged from v3. The land/ocean flag
(reusing l2_cam's own EASE2 mask / `cam_l2.cld.read_surface_mask`
convention: -1=ocean, 0=land, -2=coast, 101=unclassified/fill) is also
unchanged, shown as point color on the SZA and Latitude panels.

The figure closes itself automatically right after a pick (briefly
flashing the picked point first, on all three maps plus a vertical line on
each bottom panel).

Deliberately does NOT open the camera L1B (90+ GB for this orbit) or
attempt a surface-reflectance panel. The separate automated criteria-based
finder (find_site.py, find_glint_geometry.py) stays the tool for "find me
a specific geometric condition"; this is for eyeballing a scene by hand.

Run on Sebastian's own machine (interactive matplotlib window required; do
NOT matplotlib.use('Agg') here).

Ported into libera_cam_l2 2026-11-03 (was libera-cam-proxy's
apps/l2/pick_tile_overview.py; no logic changed). Two defaults updated
for this standalone repo's self-containment, per the apps/l2 -> l2
cleanup: --l2-cam-root now defaults to this repo's own root (where
the EASE2 surface mask already lives at data/sample/surface_mask/,
symlinked per docs/GETTING_STARTED.md section 5b) instead of the
sibling l2_cam checkout; --rad-l1b/--era5 now default to this repo's
own DITL2 convention (data/DITL2/...), same as build_input_manifest.py,
instead of being required with no default.
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Default location of the static EASE2 land/ocean mask (see module
# docstring -- this is real global geography, not orbit-date-specific
# sample data, so it's fine to reuse directly for any orbit/granule).
# Repo-root-relative (same convention as fetch_brdf.py/fetch_proxy_scene.py/
# run_dev_step1.py's own DEFAULT_L2_CAM_ROOT) rather than assuming a
# sibling l2_cam checkout -- data/sample/surface_mask/ is this repo's
# own symlinked copy, see docs/GETTING_STARTED.md section 5b.
DEFAULT_L2_CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFC_LAND, SFC_OCEAN, SFC_COAST, SFC_FILL = 0, -1, -2, 101


def decode_radiometer_time(ds):
    """`radiometer_time` carries real CF attrs (units='nanoseconds since
    1958-01-01', calendar='standard'), so xarray's default CF decoding
    already turns it into absolute UTC datetime64 on open -- correctly
    (checked directly against the raw ints + attrs: this granule really is
    2026-07-12, matching its filename). A bug in an earlier version of this
    script re-added the 1958 epoch on top of that already-decoded value,
    landing 12 years off (2014 instead of 2026) -- fixed here by using
    xarray's decoded value as-is, with a fallback only for the case where
    CF decoding didn't fire (see docs/L1B_TO_L2_ROADMAP.md)."""
    raw = ds["radiometer_time"].values
    if np.issubdtype(raw.dtype, np.datetime64):
        return raw
    return np.datetime64("1958-01-01T00:00:00") + raw.astype("timedelta64[ns]")


def load_radiometer_track(rad_l1b_path):
    """Full resolution -- RAD-4CH's own Latitude/Longitude/Solar_Zenith_Surface/
    Viewing_Zenith_Surface/Elevation/radiometer_time. Drops the ~20% of
    samples that are not geo-valid (NaN lat/lon/VZA -- mechanism pointed
    past the Earth's limb into space; see module docstring)."""
    ds = xr.open_dataset(rad_l1b_path)
    lat = ds["Latitude"].values
    lon = ds["Longitude"].values
    sza = ds["Solar_Zenith_Surface"].values
    vza = ds["Viewing_Zenith_Surface"].values
    elevation = ds["Elevation"].values
    time_utc = decode_radiometer_time(ds)
    ds.close()

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(vza) & np.isfinite(elevation)
    n_dropped = (~valid).sum()
    print("  dropping %d / %d samples (%.1f%%) that are not geo-valid "
          "(mechanism past the limb into space)"
          % (n_dropped, lat.size, 100.0 * n_dropped / lat.size))
    return (lat[valid], lon[valid], sza[valid], vza[valid], elevation[valid],
            time_utc[valid])


def reduce_to_center_track(lat, lon, sza, vza, time_utc, bin_seconds=5.5):
    """Per ~bin_seconds scan cycle, keep only the near-nadir (min VZA)
    footprint. (v4: left/right swath envelope dropped -- center only.)"""
    time_s = (time_utc - time_utc[0]) / np.timedelta64(1, "s")
    edges = np.arange(0.0, float(time_s[-1]) + bin_seconds, bin_seconds)
    starts = np.searchsorted(time_s, edges[:-1], side="left")
    ends = np.searchsorted(time_s, edges[1:], side="left")

    idx = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        sl = slice(s, e)
        idx.append(s + int(np.argmin(vza[sl])))
    idx = np.array(idx, dtype=int)

    return dict(idx=idx, lat=lat[idx], lon=lon[idx], sza=sza[idx], vza=vza[idx],
                time_utc=time_utc[idx])


def load_era5_windspeed(era5_path):
    ds = xr.open_dataset(era5_path)
    speed = np.hypot(ds["u10"].values, ds["v10"].values)  # (time, lat, lon)
    out = dict(speed=speed, lat=ds["latitude"].values, lon=ds["longitude"].values,
               time=ds["valid_time"].values)
    ds.close()
    return out


def sample_era5_at(era5, lat, lon, time_utc):
    """Nearest-neighbor ERA5 windspeed -- coarse (0.25 deg, hourly) by
    design, this is a quicklook, not RT input."""
    era5_lat, era5_lon = era5["lat"], era5["lon"]
    era5_lon_0_360 = np.mod(era5_lon, 360.0)
    lon_0_360 = np.mod(lon, 360.0)

    i_lat = np.abs(era5_lat[:, None] - lat[None, :]).argmin(axis=0)
    i_lon = np.abs(era5_lon_0_360[:, None] - lon_0_360[None, :]).argmin(axis=0)
    era5_time_s = era5["time"].astype("datetime64[s]").astype(np.int64)
    time_s = time_utc.astype("datetime64[s]").astype(np.int64)
    i_time = np.abs(era5_time_s[:, None] - time_s[None, :]).argmin(axis=0)

    return era5["speed"][i_time, i_lat, i_lon]


def load_land_ocean_mask(l2_cam_root):
    """l2_cam's own static EASE2 3 km land/ocean mask -- the same file
    cam_l2.cld.read_surface_mask / run_tile_rt.tile() use. Returns
    (lon_1d, lat_1d, sfc_2d) with sfc_2d indexed [lat_idx, lon_idx],
    matching cld.py's own reshape((4872, 11568)) convention."""
    sfc_dir = os.path.join(l2_cam_root, "data", "sample", "surface_mask")
    geo_fn = os.path.join(sfc_dir, "NSIDC0772_LatLon_EASE2_M03km_v1.1.nc")
    bin_fn = os.path.join(sfc_dir, "EASE2_M03km.LOCImask_land50_coast0km.11568x4872.bin")
    if not (os.path.exists(geo_fn) and os.path.exists(bin_fn)):
        print("  land/ocean mask not found under %s -- skipping land/ocean flag"
              % sfc_dir)
        return None
    import netCDF4
    f = netCDF4.Dataset(geo_fn, "r")
    lon_1d = f.variables["longitude"][0, :]
    lat_1d = f.variables["latitude"][:, 0]
    f.close()
    sfc_2d = np.fromfile(bin_fn, dtype="<i1").reshape((4872, 11568))
    return lon_1d, lat_1d, sfc_2d


def classify_land_ocean(mask, lat, lon):
    """Nearest-cell land/ocean classification. Returns an array of codes
    (SFC_LAND/SFC_OCEAN/SFC_COAST/SFC_FILL) or None if the mask couldn't
    be loaded."""
    if mask is None:
        return None
    lon_1d, lat_1d, sfc_2d = mask
    i_lat = np.abs(lat_1d[:, None] - lat[None, :]).argmin(axis=0)
    i_lon = np.abs(lon_1d[:, None] - lon[None, :]).argmin(axis=0)
    return sfc_2d[i_lat, i_lon]


def land_ocean_colors(codes):
    color_map = {SFC_LAND: "#54A24B", SFC_OCEAN: "#4C78A8",
                SFC_COAST: "#E45756", SFC_FILL: "0.6"}
    return [color_map.get(int(c), "0.6") for c in codes]


def save_pick(picked, out_yml):
    if HAS_YAML:
        with open(out_yml, "w") as f:
            yaml.safe_dump({"tile_pick": picked}, f, default_flow_style=False)
        print("Saved pick to %s" % out_yml)
    else:
        print("PyYAML not available -- pick NOT saved to file. Picked value:")
        print(picked)


def nearest_index_by_time(c, elapsed_min_click):
    return int(np.argmin(np.abs(c["elapsed_min"] - elapsed_min_click)))


def true_runs(mask):
    """Contiguous runs of True in a 1D boolean array (time-ordered),
    as a list of (start_idx, end_idx) with end_idx exclusive. Used so a
    map's visible-hemisphere coverage -- which the orbit can cross in
    and out of more than once -- becomes one bar per pass, not one bar
    spanning first-to-last (which would wrongly bridge separate passes).
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks] + 1, [idx[-1] + 1]))
    return list(zip(starts.tolist(), ends.tolist()))


def nearest_index_by_lonlat(c, lon_click, lat_click):
    """Great-circle nearest neighbor (via unit-vector dot product -- robust
    to longitude wraparound, unlike a naive lon/lat Euclidean distance)."""
    x0 = c["_xyz"][0]
    y0 = c["_xyz"][1]
    z0 = c["_xyz"][2]
    lat0 = np.radians(lat_click)
    lon0 = np.radians(lon_click)
    xc = np.cos(lat0) * np.cos(lon0)
    yc = np.cos(lat0) * np.sin(lon0)
    zc = np.sin(lat0)
    dot = x0 * xc + y0 * yc + z0 * zc
    return int(np.argmax(dot))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rad-l1b",
                   default="data/DITL2/L1B/LIBERA_L1B_RAD-4CH_V0-7-0_20260712T001238_20260712T040726_R26281034503.nc",
                   help="RAD-4CH L1B granule (default: this repo's DITL2 "
                        "convention, same file build_input_manifest.py uses)")
    p.add_argument("--era5",
                   default="data/DITL2/ERA5/39cfa975852f21f218547a3706f8c128.nc",
                   help="ERA5 reanalysis file (default: this repo's DITL2 "
                        "convention, same file build_input_manifest.py uses)")
    p.add_argument("--l2-cam-root", default=DEFAULT_L2_CAM_ROOT,
                   help="for the static EASE2 land/ocean mask; pass --no-land-ocean "
                        "to skip if you don't have this checked out")
    p.add_argument("--no-land-ocean", action="store_true")
    p.add_argument("--bin-seconds", type=float, default=5.5,
                   help="one radiometer scan cycle, per the elevation-sweep "
                        "timing measured earlier this project (~4.5-5.7 s)")
    p.add_argument("--out-yml", default="tile_pick.yml")
    args = p.parse_args()

    print("Loading RAD-4CH track from %s ..." % args.rad_l1b)
    lat, lon, sza, vza, elevation, time_utc = load_radiometer_track(args.rad_l1b)
    print("  %d geo-valid footprints, span %s .. %s" % (lat.size, time_utc[0], time_utc[-1]))

    c = reduce_to_center_track(lat, lon, sza, vza, time_utc, bin_seconds=args.bin_seconds)
    print("  reduced to %d near-nadir scan-cycle points" % c["lat"].size)

    # precompute unit vectors for great-circle nearest-neighbor map clicks
    lat_rad = np.radians(c["lat"])
    lon_rad = np.radians(c["lon"])
    c["_xyz"] = (np.cos(lat_rad) * np.cos(lon_rad),
                np.cos(lat_rad) * np.sin(lon_rad),
                np.sin(lat_rad))

    print("Loading ERA5 from %s ..." % args.era5)
    era5 = load_era5_windspeed(args.era5)
    c["windspeed"] = sample_era5_at(era5, c["lat"], c["lon"], c["time_utc"])
    print("  windspeed range (near-nadir track): %.2f .. %.2f m/s"
          % (c["windspeed"].min(), c["windspeed"].max()))

    land_ocean_mask = None if args.no_land_ocean else load_land_ocean_mask(args.l2_cam_root)
    c["sfc_code"] = classify_land_ocean(land_ocean_mask, c["lat"], c["lon"])
    if c["sfc_code"] is not None:
        c["sfc_color"] = land_ocean_colors(c["sfc_code"])
        n_ocean = int((c["sfc_code"] == SFC_OCEAN).sum())
        n_land = int((c["sfc_code"] == SFC_LAND).sum())
        print("  land/ocean (near-nadir track): %d ocean, %d land, %d other"
              % (n_ocean, n_land, c["sfc_code"].size - n_ocean - n_land))
    else:
        c["sfc_color"] = "#4C78A8"

    c["elapsed_min"] = (c["time_utc"] - time_utc[0]) / np.timedelta64(1, "m")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.32, wspace=0.28)

    # --- top row: 3 orthographic viewpoints, near-nadir track only ---
    # Each map gets its own accent color; used for the map's title/border
    # AND for shading that map's visible-hemisphere time range on the
    # bottom panels (map_info, built below, carries this link).
    MAP_COLORS = ["#9467BD", "#FF7F0E", "#17BECF"]
    n_c = c["lat"].size
    centers_idx = [int(n_c * f) for f in (0.15, 0.5, 0.85)]
    map_axes = []
    map_info = []
    xyz = c["_xyz"]
    for col, i_ctr in enumerate(centers_idx):
        color = MAP_COLORS[col % len(MAP_COLORS)]
        if HAS_CARTOPY:
            proj = ccrs.Orthographic(central_longitude=float(c["lon"][i_ctr]),
                                     central_latitude=float(c["lat"][i_ctr]))
            ax = fig.add_subplot(gs[0, col], projection=proj)
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.3")
            ax.add_feature(cfeature.LAND, facecolor="0.92")
            ax.add_feature(cfeature.OCEAN, facecolor="0.98")
            tfm = dict(transform=ccrs.PlateCarree())
            try:
                ax.spines["geo"].set_edgecolor(color)
                ax.spines["geo"].set_linewidth(3.5)
            except Exception:
                pass
        else:
            ax = fig.add_subplot(gs[0, col])
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")
            tfm = {}
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        sc = ax.scatter(c["lon"], c["lat"], c=c["sza"], s=12, cmap="viridis",
                        zorder=3, picker=False, **tfm)
        ax.set_title("View centered near t=%.0f min -- click to pick"
                     % c["elapsed_min"][i_ctr], fontsize=10, color=color,
                    fontweight="bold")
        map_axes.append(ax)

        # hemisphere-visible passes for this map (great-circle angle < 90
        # deg from the map's own center point) -- the orbit can cross in
        # and out of view more than once, so this is one (tmin, tmax) per
        # contiguous pass, not a single span over every visible sample.
        x0, y0, z0 = xyz[0][i_ctr], xyz[1][i_ctr], xyz[2][i_ctr]
        visible = (xyz[0] * x0 + xyz[1] * y0 + xyz[2] * z0) > 0
        segments = [(float(c["elapsed_min"][s]), float(c["elapsed_min"][e - 1]))
                   for s, e in true_runs(visible)]
        map_info.append(dict(color=color, segments=segments))
    fig.colorbar(sc, ax=map_axes, orientation="horizontal", fraction=0.04,
                pad=0.06, label="Solar Zenith Angle [deg] (near-nadir track)",
                shrink=0.6)
    if not HAS_CARTOPY:
        print("  NOTE: cartopy not importable -- plain lon/lat scatter used "
              "instead of a real globe projection, and map-click picking is "
              "disabled (needs a projection to invert).")

    # --- bottom row: SZA, Latitude, windspeed (near-nadir track); click to pick ---
    ax_sza = fig.add_subplot(gs[1, 0])
    ax_sza.scatter(c["elapsed_min"], c["sza"], c=c["sfc_color"], s=10)
    ax_sza.set_xlabel("Elapsed time [min]")
    ax_sza.set_ylabel("SZA [deg]")
    ax_sza.set_title("SZA (color = land/ocean) -- click to pick")

    ax_lat = fig.add_subplot(gs[1, 1])
    ax_lat.scatter(c["elapsed_min"], c["lat"], c=c["sfc_color"], s=10)
    ax_lat.set_xlabel("Elapsed time [min]")
    ax_lat.set_ylabel("Latitude [deg]")
    ax_lat.set_title("Latitude, near-nadir track (color = land/ocean)")

    ax_ws = fig.add_subplot(gs[1, 2])
    ax_ws.plot(c["elapsed_min"], c["windspeed"], ".-", color="#B279A2", markersize=3)
    ax_ws.set_xlabel("Elapsed time [min]")
    ax_ws.set_ylabel("ERA5 10 m windspeed [m/s]")
    ax_ws.set_title("Wind speed -- click to pick")

    if land_ocean_mask is not None:
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                              markersize=7, label=name)
                  for name, col in (("land", "#54A24B"), ("ocean", "#4C78A8"),
                                     ("coast", "#E45756"), ("other", "0.6"))]
        ax_lat.legend(handles=handles, loc="best", fontsize=7)

    pick_axes = (ax_sza, ax_lat, ax_ws)

    # draw each map's visible-hemisphere passes as horizontal bars -- one
    # bar per contiguous in-view segment (a map can come into view more
    # than once), all at the same stacked row per map/color, near the top
    # of all three bottom panels.
    for i, info in enumerate(map_info):
        y_level = 0.97 - 0.07 * i
        for ax in pick_axes:
            for tmin, tmax in info["segments"]:
                ax.hlines(y_level, tmin, tmax, colors=info["color"],
                         linewidth=4, alpha=0.9, zorder=5,
                         transform=ax.get_xaxis_transform(), clip_on=False)

    picked = {}

    def do_pick(i):
        sfc_name = {SFC_LAND: "land", SFC_OCEAN: "ocean", SFC_COAST: "coast"}.get(
            int(c["sfc_code"][i]) if c["sfc_code"] is not None else None, "unknown")
        picked.update(lat=float(c["lat"][i]), lon=float(c["lon"][i]),
                      time_utc=str(c["time_utc"][i]), sza_deg=float(c["sza"][i]),
                      vza_deg=float(c["vza"][i]),
                      era5_windspeed_ms=float(c["windspeed"][i]),
                      surface_type=sfc_name)
        print("\nPicked: lat=%.4f lon=%.4f time=%s SZA=%.2f deg VZA=%.2f deg "
              "windspeed=%.2f m/s surface=%s"
              % (picked["lat"], picked["lon"], picked["time_utc"],
                 picked["sza_deg"], picked["vza_deg"], picked["era5_windspeed_ms"],
                 picked["surface_type"]))
        save_pick(picked, args.out_yml)

        for ax in pick_axes:
            ax.axvline(c["elapsed_min"][i], color="k", linestyle="--", linewidth=1)
        for ax in map_axes:
            kw = dict(transform=ccrs.PlateCarree()) if HAS_CARTOPY else {}
            ax.scatter([c["lon"][i]], [c["lat"][i]], marker="*", s=250,
                      color="red", zorder=5, **kw)
        fig.canvas.draw()
        plt.pause(0.6)
        plt.close(fig)

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes in pick_axes:
            i = nearest_index_by_time(c, event.xdata)
            do_pick(i)
        elif event.inaxes in map_axes:
            ax = event.inaxes
            if not HAS_CARTOPY:
                return
            try:
                lon_click, lat_click = ccrs.PlateCarree().transform_point(
                    event.xdata, event.ydata, ax.projection)
            except Exception:
                return
            if not (np.isfinite(lon_click) and np.isfinite(lat_click)):
                print("  (click fell outside the visible globe -- try again)")
                return
            i = nearest_index_by_lonlat(c, lon_click, lat_click)
            do_pick(i)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.suptitle("Orbit quicklook -- click a map view OR the SZA/Latitude/"
                "Windspeed panels to pick a time/location (figure closes "
                "itself after the pick)", fontsize=12)

    plt.show()

    if not picked:
        print("\nNo pick made (closed without clicking) -- nothing saved.")


if __name__ == "__main__":
    main()
