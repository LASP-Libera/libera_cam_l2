"""
libcam/util/geo.py — forward and inverse camera-to-ground projections

geolocate() : (VZA, VAA) + platform state → (lon, lat)  [forward]
geo2pix()   : (lon, lat) + platform state → (ix, iy)    [inverse]

Forward projection adapted from cam-hong (cam/util/geo.py).
Inverse projection ported from gl2 (cam.py::geo2pix), rewritten to use
cartopy instead of pyproj.
"""

import numpy as np
import cartopy.geodesic as cg

import libcam.common as C
from libcam.util.util import cal_geodesic_lonlat, vza_to_fpa_radius


__all__ = ['geolocate', 'geo2pix']


# ── Forward projection: (VZA, VAA) → (lon, lat) ───────────────────────────────

def geolocate(
        vza,
        vaa,
        lon0,
        lat0,
        heading,
        altitude=C.LIBERA_ALTITUDE,
        planet_radius=C.PLANET_RADIUS,
        mask=None,
        vza_limit=C.VZA_LIMIT,
):
    """
    Project every camera pixel (given by VZA and VAA in the satellite frame)
    to geographic coordinates on the Earth's surface.

    Parameters
    ----------
    vza, vaa : (nx, ny) arrays   — viewing zenith / azimuth [deg]
    lon0, lat0 : float           — sub-satellite point
    heading : float              — satellite heading [deg, N=0 clockwise]
    altitude : float             — satellite altitude [m]
    planet_radius : float        — Earth radius [m]
    mask : (nx, ny) int array    — pixels with mask!=0 are skipped (→ NaN)
    vza_limit : float            — pixels beyond this VZA → NaN

    Returns
    -------
    lon, lat : (nx, ny) float arrays, NaN where masked or beyond FOV
    """
    if mask is None:
        mask = np.zeros_like(vza, dtype=np.int32)

    # Ground range (m) from the sub-satellite point, via quadratic formula
    # that accounts for Earth curvature and satellite altitude:
    #   h = altitude above sea level of the ground point
    #   solves: (R+h)^2 = R^2 + (range)^2 (spherical Earth, simplification)
    # The cam-hong version:
    tan2 = np.tan(np.deg2rad(vza)) ** 2
    A0   = 1.0 + tan2
    B0   = 2.0 * tan2 * altitude - 2.0 * planet_radius
    C0   = tan2 * altitude ** 2
    discriminant = B0**2 - 4.0 * A0 * C0
    discriminant = np.maximum(discriminant, 0.0)   # guard numerical noise
    h    = (-B0 - np.sqrt(discriminant)) / (2.0 * A0)
    degrees = np.rad2deg(np.arccos(np.clip((planet_radius - h) / planet_radius, -1, 1)))
    ground_range = 2.0 * np.pi * planet_radius * degrees / 360.0  # m

    # Azimuth of the ground point w.r.t. north = satellite heading + VAA
    azimuth = vaa + heading

    lon = np.full_like(vza, np.nan, dtype=np.float64)
    lat = np.full_like(vza, np.nan, dtype=np.float64)

    valid = (vza <= vza_limit) & (mask == 0)
    ix, iy = np.where(valid)
    if ix.size == 0:
        return lon, lat

    lon_v, lat_v = cal_geodesic_lonlat(
        np.full(ix.size, lon0),
        np.full(ix.size, lat0),
        ground_range[ix, iy],
        azimuth[ix, iy],
    )
    lon[ix, iy] = lon_v
    lat[ix, iy] = lat_v

    return lon, lat


# ── Inverse projection: (lon, lat) → (ix, iy) ─────────────────────────────────

def geo2pix(
        lon,
        lat,
        vza_grid,
        vaa_grid,
        lon_nadir,
        lat_nadir,
        heading,
        altitude=C.LIBERA_ALTITUDE,
        planet_radius=C.PLANET_RADIUS,
        coef_ang2dist=C.COEF_ANG2DIST,
        fpa_dx=C.FPA_DX,
        fpa_dy=C.FPA_DY,
):
    """
    Map a geographic point (lon, lat) back to the nearest camera pixel.

    This implements the inverse of geolocate() and is the key function for
    efficient PSF-based source-pixel → camera-pixel accumulation:
    instead of iterating over every camera pixel and asking "what does it see?",
    we iterate over every source pixel (MODIS, Landsat …) and ask
    "which camera pixel observes this location?".

    Algorithm (ported from gl2/cam.py::geo2pix, rewritten with cartopy):
      1. Compute geodesic distance and azimuth from the sub-satellite nadir
         point to the target (lon, lat).
      2. Convert ground range → VZA using the spherical geometry inversion.
      3. VAA = azimuth − satellite_heading (put in sensor frame).
      4. Find the nearest pixel in the pre-computed (vza_grid, vaa_grid) arrays.

    Parameters
    ----------
    lon, lat       : float — target geographic coordinates
    vza_grid, vaa_grid : (nx, ny) arrays — pre-computed from cal_vza_vaa()
    lon_nadir, lat_nadir : float — sub-satellite nadir point
    heading        : float — satellite heading [deg, N=0 clockwise]
    altitude       : float — satellite altitude [m]
    planet_radius  : float — Earth radius [m]

    Returns
    -------
    (ix, iy) : tuple of ints — nearest camera pixel indices,
               or None if the target is outside the FOV.
    """
    geo = cg.Geodesic()

    # Step 1 — geodesic azimuth and distance from nadir to target
    pts_nadir  = np.array([[lon_nadir, lat_nadir]])
    pts_target = np.array([[lon, lat]])
    inv = np.asarray(geo.inverse(pts_nadir, pts_target)).ravel()
    az_geo = inv[1]   # azimuth nadir→target [deg, N=0 clockwise]
    dist   = inv[0]   # geodesic distance [m]

    # Step 2 — ground range → VZA  (spherical Earth inversion)
    beta = dist / planet_radius   # central angle [radians]

    # Special case: target is at (or very near) the nadir point → VZA ≈ 0
    # VAA is undefined at nadir, so skip the VAA term and just minimise VZA.
    if beta < 1e-9:
        idx_flat = int(np.nanargmin(vza_grid))
        return np.unravel_index(idx_flat, vza_grid.shape)
    else:
        # tan(VZA) = Re * sin(β) * tan(β) / ((Re+alt)*tan(β) − Re*sin(β))
        sin_b = np.sin(beta)
        tan_b = np.tan(beta)
        denom = (planet_radius + altitude) * tan_b - planet_radius * sin_b
        if abs(denom) < 1e-9:
            return None
        tan_vza = planet_radius * sin_b * tan_b / denom
        vza = np.rad2deg(np.arctan(abs(tan_vza)))

        if vza > C.VZA_LIMIT:
            return None

        # Step 3 — convert geographic azimuth to sensor VAA
        # az_geo from cartopy: 0=N, 90=E, in range (-180, 180]
        # Normalise to [0, 360) then subtract heading
        vaa = (az_geo % 360.0) - heading

    # Step 4 — find nearest pixel (VAA is periodic; wrap difference to [-180, 180])
    vaa_diff    = ((vaa_grid - vaa) + 180.0) % 360.0 - 180.0
    dist_metric = np.abs(vza_grid - vza) + np.abs(vaa_diff)
    idx_flat    = np.nanargmin(dist_metric)
    ix, iy = np.unravel_index(idx_flat, vza_grid.shape)

    return ix, iy
