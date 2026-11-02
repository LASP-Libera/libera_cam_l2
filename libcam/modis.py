"""
libcam/modis.py — MODIS L1B + geolocation reader

Supports both HDF4 (.hdf, via pyhdf) and HDF5/NetCDF4 (.nc, via h5py).
NASA Collection 6.1 files distributed from Earthdata use the .nc extension
but are stored in HDF5/HDF-EOS5 format; pyhdf cannot read them.

Supported L1B products
-----------------------
  MOD021KM / MYD021KM  — 1 km   (geolocation from MOD03 used as-is)
  MOD02HKM / MYD02HKM  — 500 m  (MOD03 1km geo bilinearly upsampled ×2)
  MOD02QKM / MYD02QKM  — 250 m  (MOD03 1km geo bilinearly upsampled ×4)

Band mapping by product
-----------------------
  1KM : EV_250_Aggr1km_RefSB → bands 1, 2
        EV_500_Aggr1km_RefSB → bands 3–7
        EV_1KM_RefSB         → bands 8–19, 26
  HKM : EV_250_Aggr500m_RefSB → bands 1, 2
        EV_500_RefSB           → bands 3–7
  QKM : EV_250_RefSB           → bands 1, 2
"""

import os
import glob
import datetime
import numpy as np

__all__ = ['ModisGeo', 'ModisL1B', 'find_modis_granules']


# ── Format detection ──────────────────────────────────────────────────────────

def _is_hdf5(fname):
    """Return True if file starts with the HDF5 magic bytes."""
    with open(fname, 'rb') as f:
        return f.read(4) == b'\x89HDF'


# ── HDF5 path constants ───────────────────────────────────────────────────────

_GEO_ROOT  = 'HDFEOS/SWATHS/MODIS_Swath_Type_GEO'
_L1B_ROOT  = 'HDFEOS/SWATHS/MODIS_SWATH_Type_L1B'

_GEO_PATHS = {
    'Latitude':      f'{_GEO_ROOT}/Geolocation Fields/Latitude',
    'Longitude':     f'{_GEO_ROOT}/Geolocation Fields/Longitude',
    'SensorZenith':  f'{_GEO_ROOT}/Data Fields/SensorZenith',
    'SensorAzimuth': f'{_GEO_ROOT}/Data Fields/SensorAzimuth',
    'SolarZenith':   f'{_GEO_ROOT}/Data Fields/SolarZenith',
}

# EV group name → (HDF5 path, band list), keyed by resolution
_L1B_GROUPS = {
    '1km': {
        'EV_250_Aggr1km_RefSB': (f'{_L1B_ROOT}/Data Fields/EV_250_Aggr1km_RefSB', [1, 2]),
        'EV_500_Aggr1km_RefSB': (f'{_L1B_ROOT}/Data Fields/EV_500_Aggr1km_RefSB', [3, 4, 5, 6, 7]),
        'EV_1KM_RefSB':         (f'{_L1B_ROOT}/Data Fields/EV_1KM_RefSB',
                                 [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]),
    },
    'hkm': {
        'EV_250_Aggr500m_RefSB': (f'{_L1B_ROOT}/Data Fields/EV_250_Aggr500m_RefSB', [1, 2]),
        'EV_500_RefSB':          (f'{_L1B_ROOT}/Data Fields/EV_500_RefSB',          [3, 4, 5, 6, 7]),
    },
    'qkm': {
        'EV_250_RefSB': (f'{_L1B_ROOT}/Data Fields/EV_250_RefSB', [1, 2]),
    },
}

# HDF4 dataset names (same structure, no path prefix needed)
_L1B_GROUPS_HDF4 = {
    '1km': {
        'EV_250_Aggr1km_RefSB': [1, 2],
        'EV_500_Aggr1km_RefSB': [3, 4, 5, 6, 7],
    },
    'hkm': {
        'EV_250_Aggr500m_RefSB': [1, 2],
        'EV_500_RefSB':          [3, 4, 5, 6, 7],
    },
    'qkm': {
        'EV_250_RefSB': [1, 2],
    },
}


def _detect_l1b_resolution(fname):
    """Detect 1km/hkm/qkm from the L1B filename."""
    base = os.path.basename(fname).upper()
    if 'HKM' in base:
        return 'hkm'
    if 'QKM' in base:
        return 'qkm'
    return '1km'


# ── ModisGeo ─────────────────────────────────────────────────────────────────

class ModisGeo:
    """
    Read MOD03 / MYD03 geolocation product.

    Attributes
    ----------
    lon  : (nscans*10, 1354)   float32  — longitude [deg]
    lat  : (nscans*10, 1354)   float32  — latitude [deg]
    vza  : same shape          float32  — sensor zenith [deg]
    vaa  : same shape          float32  — sensor azimuth [deg]
    sza  : same shape          float32  — solar zenith [deg]
    utc  : (nscans*10,)        float64  — UTC hour of day
    """

    def __init__(self, fname):
        self.fname = fname
        if _is_hdf5(fname):
            self._read_hdf5(fname)
        else:
            self._read_hdf4(fname)

    def _read_hdf4(self, fname):
        from pyhdf.SD import SD, SDC
        h = SD(fname, SDC.READ)
        self.lat = h.select('Latitude')[:]
        self.lon = h.select('Longitude')[:]
        self.vza = h.select('SensorZenith')[:] * 0.01
        self.vaa = h.select('SensorAzimuth')[:] * 0.01
        self.sza = h.select('SolarZenith')[:] * 0.01
        ct = h.select('EV center time')[:]
        h.end()
        self._set_utc(ct)

    def _read_hdf5(self, fname):
        import h5py
        with h5py.File(fname, 'r') as f:
            self.lat = f[_GEO_PATHS['Latitude']][:]
            self.lon = f[_GEO_PATHS['Longitude']][:]
            self.vza = f[_GEO_PATHS['SensorZenith']][:] * 0.01
            self.vaa = f[_GEO_PATHS['SensorAzimuth']][:] * 0.01
            self.sza = f[_GEO_PATHS['SolarZenith']][:] * 0.01
            ct = f['EV center time'][:]
        self._set_utc(ct)

    def _set_utc(self, ct):
        """Convert TAI-1993 seconds → UTC hour-of-day array."""
        tai_start = float(ct[0])
        epoch = datetime.datetime(1993, 1, 1)
        t0 = epoch + datetime.timedelta(seconds=tai_start)
        utc0 = t0.hour + t0.minute / 60.0 + t0.second / 3600.0
        self.utc = (ct - ct[0]) / 3600.0 + utc0

    @property
    def nadir_col(self):
        return 677

    @property
    def nadir_lon(self):
        return self.lon[:, self.nadir_col]

    @property
    def nadir_lat(self):
        return self.lat[:, self.nadir_col]


# ── ModisL1B ─────────────────────────────────────────────────────────────────

class ModisL1B:
    """
    Read MOD021KM / MYD021KM 1km calibrated radiance product.

    After construction call add_geo() to attach geolocation.

    Attributes (after add_geo)
    --------------------------
    lon, lat  : (nrows, 1354) — from ModisGeo
    rad       : (n_bands, nrows, 1354)  float32 — radiance [W m-2 sr-1 µm-1]
    band_ids  : list of MODIS band numbers corresponding to rad[i]
    """

    def __init__(self, fname, bands=None):
        """
        Parameters
        ----------
        fname : str — path to MOD021KM/HKM/QKM or MYD equivalent (.hdf or .nc)
        bands : list of int, optional — MODIS band numbers to load (default: [1])
        """
        self.fname = fname
        if bands is None:
            bands = [1]
        self._requested  = list(bands)
        self._resolution = _detect_l1b_resolution(fname)

        if _is_hdf5(fname):
            self._read_hdf5(fname)
        else:
            self._read_hdf4(fname)

        self.lon = None
        self.lat = None
        self.sza = None

    def _read_hdf4(self, fname):
        from pyhdf.SD import SD, SDC
        h = SD(fname, SDC.READ)
        rad_list, loaded_bands = [], []

        for group_name, group_bands in _L1B_GROUPS_HDF4[self._resolution].items():
            wanted = [b for b in self._requested if b in group_bands]
            if not wanted:
                continue
            try:
                sds = h.select(group_name)
            except Exception:
                continue
            attrs   = sds.attributes()
            scales  = np.array(attrs.get('radiance_scales',  [1.0] * len(group_bands)))
            offsets = np.array(attrs.get('radiance_offsets', [0.0] * len(group_bands)))
            fill    = attrs.get('_FillValue', 65535)
            raw = sds.get().astype(np.float32)
            for local_idx, band_num in enumerate(group_bands):
                if band_num not in wanted:
                    continue
                row = raw[local_idx].copy()
                row[row == fill] = np.nan
                rad = (row - offsets[local_idx]) * scales[local_idx]
                rad[rad < 0] = np.nan
                rad_list.append(rad)
                loaded_bands.append(band_num)
        h.end()
        self.band_ids = loaded_bands
        self.rad = np.stack(rad_list, axis=0) if rad_list else None

    def _read_hdf5(self, fname):
        import h5py
        rad_list, loaded_bands = [], []

        with h5py.File(fname, 'r') as f:
            for group_name, (h5_path, group_bands) in _L1B_GROUPS[self._resolution].items():
                wanted = [b for b in self._requested if b in group_bands]
                if not wanted or h5_path not in f:
                    continue
                ds      = f[h5_path]
                attrs   = dict(ds.attrs)
                scales  = np.array(attrs.get('radiance_scales',  [1.0] * len(group_bands)), dtype=np.float32)
                offsets = np.array(attrs.get('radiance_offsets', [0.0] * len(group_bands)), dtype=np.float32)
                fill    = int(attrs.get('_FillValue', [65535])[0])
                raw     = ds[:].astype(np.float32)

                for local_idx, band_num in enumerate(group_bands):
                    if band_num not in wanted:
                        continue
                    row = raw[local_idx].copy()
                    row[row == fill] = np.nan
                    rad = (row - offsets[local_idx]) * scales[local_idx]
                    rad[rad < 0] = np.nan
                    rad_list.append(rad)
                    loaded_bands.append(band_num)

        self.band_ids = loaded_bands
        self.rad = np.stack(rad_list, axis=0) if rad_list else None

    def add_geo(self, geo: ModisGeo):
        """Attach a ModisGeo object, upsampling if L1B is HKM (×2) or QKM (×4)."""
        if self.rad is None:
            raise RuntimeError(
                f'No radiance loaded from {os.path.basename(self.fname)} '
                f'(resolution={self._resolution}, bands={self._requested}). '
                f'Check that the requested band is available in this product.')

        l1b_rows, l1b_cols = self.rad.shape[-2:]
        geo_rows, geo_cols = geo.lon.shape

        if (l1b_rows, l1b_cols) == (geo_rows, geo_cols):
            lon, lat, sza = geo.lon, geo.lat, geo.sza
        else:
            # Bilinear upsample 1km MOD03 geo to match HKM (×2) or QKM (×4)
            from scipy.ndimage import zoom
            row_f = l1b_rows / geo_rows
            col_f = l1b_cols / geo_cols
            print(f'  Upsampling MOD03 geo {geo.lon.shape} → {(l1b_rows, l1b_cols)} '
                  f'(×{col_f:.1f} for {self._resolution})')
            lon = zoom(geo.lon.astype(np.float64), (row_f, col_f), order=1).astype(np.float32)
            lat = zoom(geo.lat.astype(np.float64), (row_f, col_f), order=1).astype(np.float32)
            sza = zoom(geo.sza.astype(np.float64), (row_f, col_f), order=1).astype(np.float32)

        self.lon = lon
        self.lat = lat
        self.sza = sza
        self.vza = geo.vza
        self.utc = geo.utc
        self.nadir_lon = geo.nadir_lon
        self.nadir_lat = geo.nadir_lat

    def crop(self, lon_min, lon_max, lat_min, lat_max):
        """
        Return a dict with lon/lat/rad cropped to a bounding box.
        Rows that have *any* pixel in the box are kept (full scanlines).
        """
        if self.lon is None:
            raise RuntimeError('Call add_geo() first')
        in_box = (
            (self.lon >= lon_min) & (self.lon <= lon_max) &
            (self.lat >= lat_min) & (self.lat <= lat_max)
        )
        row_mask = in_box.any(axis=1)
        return {
            'lon':      self.lon[row_mask],
            'lat':      self.lat[row_mask],
            'rad':      self.rad[:, row_mask, :],
            'band_ids': self.band_ids,
            'sza':      (self.sza[row_mask] if self.sza is not None
                        else None),
        }


# ── Helper: find MODIS granules overlapping a bounding box ───────────────────

def find_modis_granules(fdir, lon_min, lon_max, lat_min, lat_max,
                        product='MOD03', verbose=False):
    """
    Scan fdir for MODIS geo files (*.hdf or *.nc) matching the product prefix
    whose geolocation overlaps the bounding box.

    Returns a sorted list of matching geo file paths.
    """
    candidates = sorted(
        glob.glob(os.path.join(fdir, f'{product}*.hdf')) +
        glob.glob(os.path.join(fdir, f'{product}*.nc'))
    )
    matches = []
    for fpath in candidates:
        try:
            if _is_hdf5(fpath):
                import h5py
                with h5py.File(fpath, 'r') as f:
                    lat = f[_GEO_PATHS['Latitude']][:]
                    lon = f[_GEO_PATHS['Longitude']][:]
            else:
                from pyhdf.SD import SD, SDC
                h = SD(fpath, SDC.READ)
                lat = h.select('Latitude')[:]
                lon = h.select('Longitude')[:]
                h.end()

            if (lon.max() >= lon_min and lon.min() <= lon_max and
                    lat.max() >= lat_min and lat.min() <= lat_max):
                matches.append(fpath)
                if verbose:
                    print(f'  match: {os.path.basename(fpath)}')
        except Exception as e:
            if verbose:
                print(f'  skip {os.path.basename(fpath)}: {e}')
    return matches
