"""
l2/inspect_proxy_scene.py

Dev-step-2, stage 1b of the "L1X" proxy-radiance plan: sanity-check a
MODIS scene fetch_proxy_scene.py downloaded -- confirm it actually opens,
crop it to the real tile (same --tile-km / dlat_per_km / dlon_per_km
convention l2/run_tile_rt.py uses for its own RT domain, so the
cropped extent matches what dev-step-1 already treats as "the tile"),
and report whether the band-4 (~555 nm) radiance in that crop looks like
real data (sane magnitude, not all-NaN, a sensible fraction of valid
pixels for a land scene) -- before spending effort on the PSF-projection
step that actually builds camera-pixel radiances from it.

Does not use er3t at all -- loads the MOD03/MYD03 geolocation and
MOD02HKM/MYD02HKM L1B files with this repo's own libcam.modis readers
(ModisGeo, ModisL1B), the same ones apps/gcp/gcp_proxy.py already uses.

Usage
-----
    # auto-finds the one geo/L1B file pair in --img-dir (errors with a
    # list if it finds more than one -- pass --geo/--l1b explicitly then)
    python l2/inspect_proxy_scene.py

    # make a quick plot too (tile box + band-4 radiance, land/ocean
    # context) -- opens an interactive matplotlib window, no --plot-only
    # Agg nonsense since you are the one running this
    python l2/inspect_proxy_scene.py --plot
"""
import argparse
import glob
import os
import sys

import numpy as np
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_TILE_PICK = "tile_pick.yml"
DEFAULT_IMG_DIR = "l2/data/proxy_imagery"
DEFAULT_TILE_KM = 100.0
MODIS_BAND = 4

# Same equirectangular approximation l2/run_tile_rt.py's tile() uses
# for its own RT-domain extent (dlat_per_km/dlon_per_km, with the 0.05
# floor on cos(lat) near the poles) -- kept identical here so "the tile"
# means the same ground box in both places.
KM_PER_DEG_LAT = 111.32


def tile_bbox(lat0, lon0, tile_km):
    half_km = tile_km / 2.0
    dlat_per_km = 1.0 / KM_PER_DEG_LAT
    dlon_per_km = 1.0 / (KM_PER_DEG_LAT * max(0.05, np.cos(np.radians(lat0))))
    half_deg_lat = half_km * dlat_per_km
    half_deg_lon = half_km * dlon_per_km
    return (lon0 - half_deg_lon, lon0 + half_deg_lon,
            lat0 - half_deg_lat, lat0 + half_deg_lat)


def _find_one(pattern, img_dir, what):
    matches = sorted(glob.glob(os.path.join(img_dir, pattern)))
    if len(matches) == 0:
        raise SystemExit("No %s file found in %s matching %r"
                         % (what, img_dir, pattern))
    if len(matches) > 1:
        raise SystemExit(
            "Multiple %s files found in %s -- pass --geo/--l1b explicitly "
            "to disambiguate:\n  %s" % (what, img_dir, "\n  ".join(matches)))
    return matches[0]


def load_tile_pick(path):
    with open(path) as f:
        d = dict(yaml.safe_load(f)["tile_pick"])
    return float(d["lat"]), float(d["lon"])


def inspect_crop(geo_fname, l1b_fname, lat0, lon0, tile_km, band=MODIS_BAND,
                 verbose=True):
    """Load geo+L1B, crop to the tile box, and return a stats dict. Raises
    SystemExit with a clear message if the crop is empty or the band
    wasn't loaded (e.g. wrong product for this band)."""
    from libcam.modis import ModisGeo, ModisL1B

    lon_min, lon_max, lat_min, lat_max = tile_bbox(lat0, lon0, tile_km)
    if verbose:
        print("Tile box (%.1f km around lat=%.4f lon=%.4f): "
             "lon [%.4f, %.4f]  lat [%.4f, %.4f]"
             % (tile_km, lat0, lon0, lon_min, lon_max, lat_min, lat_max))

    geo = ModisGeo(geo_fname)
    l1b = ModisL1B(l1b_fname, bands=[band])
    l1b.add_geo(geo)

    if band not in l1b.band_ids:
        raise SystemExit(
            "Band %d not found in %s (bands present: %s) -- wrong product "
            "for this band, or it genuinely has no data for it."
            % (band, os.path.basename(l1b_fname), l1b.band_ids))

    crop = l1b.crop(lon_min, lon_max, lat_min, lat_max)
    rad = crop["rad"][crop["band_ids"].index(band)]

    if rad.size == 0:
        raise SystemExit(
            "Crop is EMPTY -- the tile box doesn't overlap this granule "
            "at all. Check --tile-pick/lat/lon and that --geo/--l1b are "
            "really the granule fetch_proxy_scene.py found for this tile.")

    finite = np.isfinite(rad)
    n_total = rad.size
    n_finite = int(finite.sum())

    sza = crop.get("sza")
    sza_mean = (float(np.nanmean(sza[finite])) if sza is not None and n_finite
               else None)
    stats = dict(
        shape=rad.shape,
        n_total=n_total,
        n_finite=n_finite,
        frac_finite=n_finite / n_total if n_total else 0.0,
        min=float(np.nanmin(rad)) if n_finite else None,
        max=float(np.nanmax(rad)) if n_finite else None,
        mean=float(np.nanmean(rad)) if n_finite else None,
        lon=crop["lon"], lat=crop["lat"], rad=rad, sza=sza,
        sza_mean=sza_mean,
    )
    return stats


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-pick", default=DEFAULT_TILE_PICK)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR,
                  help="directory fetch_proxy_scene.py downloaded into "
                       "(default: %s)" % DEFAULT_IMG_DIR)
    p.add_argument("--geo", default=None,
                  help="MOD03/MYD03 geolocation file (default: auto-find "
                       "the one *D03.*.hdf in --img-dir)")
    p.add_argument("--l1b", default=None,
                  help="MOD02HKM/MYD02HKM L1B file (default: auto-find "
                       "the one *02HKM*.hdf in --img-dir)")
    p.add_argument("--tile-km", type=float, default=DEFAULT_TILE_KM)
    p.add_argument("--band", type=int, default=MODIS_BAND)
    p.add_argument("--plot", action="store_true",
                  help="show a quick interactive plot of the cropped "
                       "band radiance (no file is saved/no Agg backend)")
    return p.parse_args()


def main():
    args = parse_args()

    lat0, lon0 = load_tile_pick(args.tile_pick)

    geo_fname = args.geo or _find_one("*D03.*.hdf", args.img_dir, "geolocation")
    l1b_fname = args.l1b or _find_one("*02HKM*.hdf", args.img_dir, "L1B")
    print("geo: %s" % geo_fname)
    print("L1B: %s" % l1b_fname)

    stats = inspect_crop(geo_fname, l1b_fname, lat0, lon0, args.tile_km,
                         band=args.band)

    print("\nCrop shape: %s  (%d pixels total)" % (stats["shape"], stats["n_total"]))
    print("Valid (finite) pixels: %d / %d (%.1f%%)"
         % (stats["n_finite"], stats["n_total"], 100.0 * stats["frac_finite"]))
    if stats["n_finite"]:
        print("Band %d radiance: min=%.4f  mean=%.4f  max=%.4f  "
             "[W m-2 sr-1 um-1]"
             % (args.band, stats["min"], stats["mean"], stats["max"]))
    else:
        print("WARNING: no finite pixels in the crop at all -- something "
             "is wrong (wrong band? wrong files? bad crop box?).")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(stats["lon"], stats["lat"], c=stats["rad"], s=4,
                        cmap="viridis")
        ax.plot(lon0, lat0, "r+", markersize=14, markeredgewidth=2,
               label="tile center")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title("MODIS band %d radiance, %.0f km tile crop" % (args.band, args.tile_km))
        ax.legend(loc="upper right")
        plt.colorbar(sc, ax=ax, label="radiance [W m-2 sr-1 um-1]")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
