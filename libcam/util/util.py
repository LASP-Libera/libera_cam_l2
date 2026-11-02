"""
libcam/util/util.py — general-purpose utilities

Ported and cleaned from cam-hong (cam/util/util.py) and gl2 (lib.py).
"""

import datetime
import numpy as np
import cartopy.geodesic as cg

import libcam.common as C


__all__ = [
    'cal_vza_vaa',
    'dtime_to_jday',
    'jday_to_dtime',
    'jpss_tai_to_utc',
    'OBS_ID_WFOV',
    'cal_geodesic_lonlat',
    'cal_geodesic_dist',
    'cal_heading',
    'load_h5',
    'save_h5',
]


# ── FPA → viewing angles ──────────────────────────────────────────────────────

def cal_vza_vaa(
        fpa_nx=C.FPA_NX,
        fpa_ny=C.FPA_NY,
        fpa_dx=C.FPA_DX,
        fpa_dy=C.FPA_DY,
        coef_dist2ang=C.COEF_DIST2ANG,
):
    """
    Compute VZA and VAA (in degrees) for every FPA pixel.

    The VAA is measured clockwise from the satellite's along-track (y) axis,
    so a pixel at (dx>0, dy=0) has VAA=90° (to the right of track) and a pixel
    at (dx=0, dy>0) has VAA=0° (forward along track).

    Returns a dict with keys:
        'vza'    : (fpa_nx, fpa_ny) float array — viewing zenith angle [deg]
        'vaa'    : (fpa_nx, fpa_ny) float array — viewing azimuth angle [deg]
        'fpa_cx' : int — x-index of optical axis
        'fpa_cy' : int — y-index of optical axis
    """
    fpa_cx = fpa_nx // 2
    fpa_cy = fpa_ny // 2

    fpa_x0 = (np.arange(fpa_nx) - fpa_cx + 0.5) * fpa_dx   # mm
    fpa_y0 = (np.arange(fpa_ny) - fpa_cy + 0.5) * fpa_dy   # mm

    fpa_x, fpa_y = np.meshgrid(fpa_x0, fpa_y0, indexing='ij')

    # VAA: arctan2(across-track, along-track)
    vaa = np.rad2deg(np.arctan2(fpa_x, fpa_y))

    # VZA: polynomial in radial distance from optical axis
    fpa_dist = np.sqrt(fpa_x**2 + fpa_y**2)
    vza = np.polyval(coef_dist2ang, fpa_dist)

    return {'vza': vza, 'vaa': vaa, 'fpa_cx': fpa_cx, 'fpa_cy': fpa_cy}


# ── VZA → FPA radius (inverse lens model) ────────────────────────────────────

def vza_to_fpa_radius(vza, coef_ang2dist=C.COEF_ANG2DIST):
    """Return FPA radial distance (mm) for a given VZA (deg)."""
    return np.polyval(coef_ang2dist, vza)


# ── JPSS / Libera time utilities ─────────────────────────────────────────────

# WFOV Camera obs-ID → plain-language name mapping.
# Source: Libera Confluence — "Modes, Obs IDs, States, and Transitions"
OBS_ID_WFOV: dict[int, str] = {
    0:   'Boot',
    1:   'Safe',
    2:   'Stowed',
    128: 'Cross-Track',
    132: 'RAP',
    133: 'Reserved-133',
    134: 'Reserved-134',
    135: 'Reserved-135',
    136: 'Along-Track',
    137: 'Earth-Target',
    138: 'Geo-Scan-PNG',
    139: 'Geo-Scan-Pacific',
    256: 'SWC-365nm',
    257: 'SWC-405nm',
    258: 'SWC-520nm',
    259: 'SWC-635nm',
    260: 'SWC-840nm',
    261: 'SWC-1550nm',
}


_VIIRS_EPOCH = datetime.datetime(1958, 1, 1)

# Grain (leap-second-aware TAI ↔ UTC) lives in camera/py/ alongside this repo.
# We locate it at import time so any app using this util doesn't need to know
# the path.
def _load_grain():
    import sys
    from pathlib import Path
    # Walk up from this file: util.py → util/ → libcam/ → libera-cam/ → camera/
    _camera_dir = Path(__file__).resolve().parents[3]
    _grain_dir = _camera_dir / 'py'
    if str(_grain_dir) not in sys.path:
        sys.path.insert(0, str(_grain_dir))
    from grain import Grain
    _leap = _grain_dir / 'leap-seconds.list'
    return Grain(open(_leap))


_grain_instance = None


def jpss_tai_to_utc(tai_s: int) -> datetime.datetime:
    """
    Convert a JPSS TAI timestamp (integer seconds since 1958-01-01) to UTC.

    This is the native time format used in Libera FITS housekeeping files and
    in the obs-ID log CSVs (time_jpss_start / time_jpss_end columns).

    Uses the Grain leap-second library (camera/py/grain.py + leap-seconds.list)
    for a correct TAI→UTC conversion.  Falls back to a UTC ≈ TAI−37 s
    approximation (valid from 2017 onward, when the last leap second was
    added) if Grain cannot be loaded.
    """
    global _grain_instance
    if _grain_instance is None:
        try:
            _grain_instance = _load_grain()
        except Exception:
            _grain_instance = False   # mark as unavailable

    if _grain_instance is not False:
        return _grain_instance.tai2utc(tai_s, epoch=_VIIRS_EPOCH)
    else:
        # Fallback: 37 leap seconds accumulated by 2017 (no new ones since)
        return _VIIRS_EPOCH + datetime.timedelta(seconds=tai_s - 37)


# ── Julian day utilities ──────────────────────────────────────────────────────

def dtime_to_jday(dtime):
    """datetime → Julian day (epoch = 0001-01-01)."""
    return (dtime - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0


def jday_to_dtime(jday):
    """Julian day → datetime."""
    return datetime.datetime(1, 1, 1) + datetime.timedelta(
        seconds=round((jday - 1) * 86400.0)
    )


# ── Geodesic utilities (cartopy) ──────────────────────────────────────────────

def cal_geodesic_lonlat(lon0, lat0, dist, azimuth):
    """
    Given a starting point (lon0, lat0), a distance (m) and azimuth (deg
    clockwise from north), return the end-point (lon1, lat1).

    All inputs may be scalars or arrays of matching shape.
    """
    lon0 = np.atleast_1d(np.asarray(lon0, dtype=float)).ravel()
    lat0 = np.atleast_1d(np.asarray(lat0, dtype=float)).ravel()
    dist = np.atleast_1d(np.asarray(dist, dtype=float)).ravel()
    azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float)).ravel()

    geo = cg.Geodesic()
    points = np.column_stack((lon0, lat0))
    out = np.asarray(geo.direct(points, azimuth, dist))
    if out.ndim == 1:
        out = out[np.newaxis, :]
    return out[:, 0], out[:, 1]


def cal_geodesic_dist(lon0, lat0, lon1, lat1):
    """
    Return geodesic distance (m) between (lon0,lat0) and (lon1,lat1).
    """
    lon0 = np.atleast_1d(np.asarray(lon0, dtype=float)).ravel()
    lat0 = np.atleast_1d(np.asarray(lat0, dtype=float)).ravel()
    lon1 = np.atleast_1d(np.asarray(lon1, dtype=float)).ravel()
    lat1 = np.atleast_1d(np.asarray(lat1, dtype=float)).ravel()

    geo = cg.Geodesic()
    pts0 = np.column_stack((lon0, lat0))
    pts1 = np.column_stack((lon1, lat1))
    out = np.asarray(geo.inverse(pts0, pts1))
    if out.ndim == 1:
        out = out[np.newaxis, :]
    return out[:, 0]


def cal_heading(lon, lat):
    """Compute heading (deg, N=0, E=90) along a lon/lat track."""
    dx = lat[1:] - lat[:-1]
    dy = lon[1:] - lon[:-1]
    heading = np.rad2deg(np.arctan2(dy, dx))
    heading = np.append(heading[0], heading) % 360.0
    return heading


# ── HDF5 convenience ──────────────────────────────────────────────────────────

def load_h5(fname):
    """Load all datasets from an HDF5 file into a flat dict."""
    import h5py

    def _walk(obj, prefix=''):
        for key in obj.keys():
            path = f'{prefix}/{key}'
            if isinstance(obj[key], h5py.Dataset):
                yield path
            elif isinstance(obj[key], h5py.Group):
                yield from _walk(obj[key], path)

    data = {}
    with h5py.File(fname, 'r') as f:
        for path in _walk(f):
            data[path[1:]] = f[path[1:]][...]
    return data


def save_h5(fname, data):
    """Save a flat dict of arrays to HDF5."""
    import h5py
    with h5py.File(fname, 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val, compression='gzip',
                             compression_opts=6, chunks=True)
    print(f'Saved: {fname}')
