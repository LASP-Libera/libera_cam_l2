"""
libcam/l2/domains.py
--------------------
RT domain decomposition for the Libera L2b camera science pipeline.

A **Domain** is a rectangular tile of camera pixels for which a single
RT calculation will be performed.  Tiles are uniform in pixel space
(first pass); within each tile the angular geometry (SZA, VZA, RAZ)
should vary by less than a threshold so that one representative geometry
describes the whole tile.

Typical call sequence
---------------------
>>> import netCDF4 as nc
>>> import numpy as np
>>> from libcam.l2.domains import chop_frame, summarise_domains
>>>
>>> ds = nc.Dataset('LIBERA_L1B_CAM_...nc', 'r')
>>> frame = 0
>>> sza = np.asarray(ds.variables['Solar_Zenith_Surface'][frame])
>>> vza = np.asarray(ds.variables['Viewing_Zenith_Surface'][frame])
>>> raz = np.asarray(ds.variables['Relative_Azimuth_Surface'][frame])
>>> vm  = np.asarray(ds.variables['valid_pixel_mask'][frame])
>>> lat = np.asarray(ds.variables['Latitude'][frame])
>>> lon = np.asarray(ds.variables['Longitude'][frame])
>>>
>>> domains = chop_frame(sza, vza, raz,
...                      valid_mask=vm, lat_2d=lat, lon_2d=lon,
...                      tile_size=64)
>>> summarise_domains(domains)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── Domain dataclass ──────────────────────────────────────────────────────────

@dataclass
class Domain:
    """One RT domain — a rectangular sub-tile of a camera frame.

    Pixel coordinates are in **full-resolution FPA space** (0…2047 in both
    row and col).  Slices are half-open: [row_start, row_end).

    Attributes
    ----------
    row_start, row_end : int   — FPA row slice [row_start, row_end)
    col_start, col_end : int   — FPA col slice [col_start, col_end)
    tile_size          : int   — nominal tile side length [px]

    mean_sza, mean_vza, mean_raz : float — tile-mean angular geometry [deg]
    range_sza, range_vza, range_raz : float — peak-to-peak variation [deg]

    flag_geometry : bool  — True if any geometry range exceeds its threshold
    geo_populated : bool  — False when geometry arrays were all-zero placeholders

    lat_min, lat_max : float — bounding latitudes [deg];  NaN when unavailable
    lon_min, lon_max : float — bounding longitudes [deg]; NaN when unavailable

    n_valid  : int — number of valid camera pixels (from valid_pixel_mask)
    n_pixels : int — total pixels in this tile (= n_rows × n_cols)
    """
    row_start: int
    row_end:   int
    col_start: int
    col_end:   int
    tile_size: int

    mean_sza:  float = np.nan
    mean_vza:  float = np.nan
    mean_raz:  float = np.nan
    range_sza: float = np.nan
    range_vza: float = np.nan
    range_raz: float = np.nan

    flag_geometry: bool = False
    geo_populated: bool = False

    lat_min: float = np.nan
    lat_max: float = np.nan
    lon_min: float = np.nan
    lon_max: float = np.nan

    n_valid:  int = 0
    n_pixels: int = 0

    # ── convenience properties ─────────────────────────────────────────────────

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def n_cols(self) -> int:
        return self.col_end - self.col_start

    @property
    def centre_row(self) -> int:
        return (self.row_start + self.row_end) // 2

    @property
    def centre_col(self) -> int:
        return (self.col_start + self.col_end) // 2

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_pixels if self.n_pixels > 0 else 0.0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def __repr__(self) -> str:
        geo_str = (f'SZA={self.mean_sza:.1f}°±{self.range_sza:.1f}° '
                   f'VZA={self.mean_vza:.1f}°±{self.range_vza:.1f}°'
                   if self.geo_populated else 'geo=placeholder')
        flag_str = ' [FLAGGED]' if self.flag_geometry else ''
        return (f'Domain(rows={self.row_start}:{self.row_end}, '
                f'cols={self.col_start}:{self.col_end}, '
                f'{geo_str}, valid={self.n_valid}/{self.n_pixels}){flag_str}')


# ── main decomposition function ───────────────────────────────────────────────

def chop_frame(
    sza_2d,
    vza_2d,
    raz_2d,
    *,
    valid_mask=None,
    lat_2d=None,
    lon_2d=None,
    tile_size: int = 64,
    max_sza_range: float = 5.0,
    max_vza_range: float = 5.0,
    max_raz_range: float = 10.0,
    fill_value: float = -999.0,
) -> List[Domain]:
    """Decompose a 2-D camera frame into uniform RT domain tiles.

    Parameters
    ----------
    sza_2d, vza_2d, raz_2d : (NX, NY) array-like — per-pixel angular
        geometry from L1B variables Solar_Zenith_Surface,
        Viewing_Zenith_Surface, Relative_Azimuth_Surface [deg].
        May be all-zeros when L1B geometry is not yet populated.
    valid_mask : (NX, NY) array-like, optional
        Camera valid_pixel_mask.  Pixels > 0 are valid.
        If None, all pixels are counted as valid.
    lat_2d, lon_2d : (NX, NY) array-like, optional
        Per-pixel Latitude / Longitude [deg].
        Pass None (or omit) to skip the bounding-box computation.
    tile_size : int
        Side length of each uniform tile [pixels].  Default 64.
    max_sza_range, max_vza_range, max_raz_range : float
        Flag a tile if the peak-to-peak variation of that angle exceeds
        this threshold [deg].
    fill_value : float
        Sentinel fill value in lat/lon arrays (ignored in bounding-box
        computation).

    Returns
    -------
    domains : list[Domain]
        One Domain per tile, row-major order (left→right, top→bottom).
        Edge tiles at the right / bottom boundary are smaller than
        tile_size when the FPA dimensions are not multiples of tile_size.

    Notes
    -----
    When sza_2d / vza_2d / raz_2d are all-zero placeholders the function
    sets ``geo_populated=False`` and ``flag_geometry=False`` for every tile.
    No tile will be erroneously flagged due to geometry that has not yet
    been computed by the L1B pipeline.
    """
    sza = np.asarray(sza_2d, dtype=np.float32)
    vza = np.asarray(vza_2d, dtype=np.float32)
    raz = np.asarray(raz_2d, dtype=np.float32)

    NX, NY = sza.shape

    # Detect all-zero placeholder geometry (applies to the entire frame)
    # A frame where all three variables are identically zero has not been
    # populated by the L1B geometry pipeline yet.
    frame_geo_populated = not (
        sza.max() == 0.0 and sza.min() == 0.0
        and vza.max() == 0.0 and vza.min() == 0.0
    )

    has_mask = valid_mask is not None
    has_geo  = (lat_2d is not None and lon_2d is not None)
    if has_geo:
        lat = np.asarray(lat_2d, dtype=np.float32)
        lon = np.asarray(lon_2d, dtype=np.float32)

    domains: List[Domain] = []

    for rs in range(0, NX, tile_size):
        re = min(rs + tile_size, NX)
        for cs in range(0, NY, tile_size):
            ce = min(cs + tile_size, NY)

            n_pix = (re - rs) * (ce - cs)

            # ── valid pixel count ──────────────────────────────────────────────
            if has_mask:
                tile_mask = np.asarray(valid_mask[rs:re, cs:ce])
                n_valid   = int((tile_mask > 0).sum())
            else:
                n_valid = n_pix

            # ── angular geometry stats ─────────────────────────────────────────
            if frame_geo_populated:
                s_sza = sza[rs:re, cs:ce]
                s_vza = vza[rs:re, cs:ce]
                s_raz = raz[rs:re, cs:ce]
                mean_sza  = float(s_sza.mean())
                mean_vza  = float(s_vza.mean())
                mean_raz  = float(s_raz.mean())
                range_sza = float(s_sza.max() - s_sza.min())
                range_vza = float(s_vza.max() - s_vza.min())
                range_raz = float(s_raz.max() - s_raz.min())
                flag = (range_sza > max_sza_range or
                        range_vza > max_vza_range or
                        range_raz > max_raz_range)
            else:
                mean_sza = mean_vza = mean_raz = 0.0
                range_sza = range_vza = range_raz = 0.0
                flag = False

            # ── lat/lon bounding box ───────────────────────────────────────────
            if has_geo:
                s_lat = lat[rs:re, cs:ce].ravel()
                s_lon = lon[rs:re, cs:ce].ravel()
                good  = (s_lat != fill_value) & (s_lon != fill_value)
                if good.any():
                    lat_min = float(s_lat[good].min())
                    lat_max = float(s_lat[good].max())
                    lon_min = float(s_lon[good].min())
                    lon_max = float(s_lon[good].max())
                else:
                    lat_min = lat_max = lon_min = lon_max = np.nan
            else:
                lat_min = lat_max = lon_min = lon_max = np.nan

            domains.append(Domain(
                row_start=rs, row_end=re,
                col_start=cs, col_end=ce,
                tile_size=tile_size,
                mean_sza=mean_sza,   mean_vza=mean_vza,   mean_raz=mean_raz,
                range_sza=range_sza, range_vza=range_vza, range_raz=range_raz,
                flag_geometry=flag,
                geo_populated=frame_geo_populated,
                lat_min=lat_min, lat_max=lat_max,
                lon_min=lon_min, lon_max=lon_max,
                n_valid=n_valid,
                n_pixels=n_pix,
            ))

    return domains


# ── summary / reporting ───────────────────────────────────────────────────────

def summarise_domains(domains: List[Domain], verbose: bool = True) -> dict:
    """Print and return summary statistics for a domain list.

    Parameters
    ----------
    domains : list[Domain] — output of chop_frame()
    verbose : bool — print to stdout (default True)

    Returns
    -------
    dict with keys:
        n_tiles, n_flagged, frac_flagged,
        n_valid_total, n_pixels_total,
        sza_mean, sza_range_mean, sza_range_max,
        vza_mean, vza_range_mean, vza_range_max,
        geo_populated
    """
    n_tiles   = len(domains)
    n_flagged = sum(1 for d in domains if d.flag_geometry)
    n_valid   = sum(d.n_valid   for d in domains)
    n_pixels  = sum(d.n_pixels  for d in domains)
    geo_pop   = any(d.geo_populated for d in domains)

    sza_means  = [d.mean_sza  for d in domains if geo_pop]
    vza_means  = [d.mean_vza  for d in domains if geo_pop]
    sza_ranges = [d.range_sza for d in domains if geo_pop]
    vza_ranges = [d.range_vza for d in domains if geo_pop]

    sza_mean = float(np.nanmean(sza_means)) if sza_means else np.nan
    vza_mean = float(np.nanmean(vza_means)) if vza_means else np.nan
    sza_range_mean = float(np.nanmean(sza_ranges)) if sza_ranges else np.nan
    sza_range_max  = float(np.nanmax(sza_ranges))  if sza_ranges else np.nan
    vza_range_mean = float(np.nanmean(vza_ranges)) if vza_ranges else np.nan
    vza_range_max  = float(np.nanmax(vza_ranges))  if vza_ranges else np.nan

    if verbose:
        W = 58
        print('  ' + '-' * W)
        print(f'  Domain summary  ({n_tiles} tiles, tile_size='
              f'{domains[0].tile_size if domains else "?"}px)')
        print('  ' + '-' * W)
        print(f'  Flagged tiles : {n_flagged} / {n_tiles} '
              f'({100 * n_flagged / n_tiles:.1f}%)' if n_tiles else '  No tiles')
        print(f'  Valid pixels  : {n_valid:,} / {n_pixels:,} '
              f'({100 * n_valid / n_pixels:.1f}%)' if n_pixels else '')
        if geo_pop:
            print(f'  Mean SZA      : {sza_mean:.1f}°')
            print(f'  Mean VZA      : {vza_mean:.1f}°')
            print(f'  SZA range     : mean={sza_range_mean:.2f}°  '
                  f'max={sza_range_max:.2f}°')
            print(f'  VZA range     : mean={vza_range_mean:.2f}°  '
                  f'max={vza_range_max:.2f}°')
        else:
            print('  Geometry      : not populated (all-zero placeholder)')
        print('  ' + '-' * W)

    return dict(
        n_tiles=n_tiles,
        n_flagged=n_flagged,
        frac_flagged=n_flagged / n_tiles if n_tiles else 0.0,
        n_valid_total=n_valid,
        n_pixels_total=n_pixels,
        sza_mean=sza_mean,
        sza_range_mean=sza_range_mean,
        sza_range_max=sza_range_max,
        vza_mean=vza_mean,
        vza_range_mean=vza_range_mean,
        vza_range_max=vza_range_max,
        geo_populated=geo_pop,
    )


# ── domain overlay helpers ────────────────────────────────────────────────────

def domain_overlay_patches(domains: List[Domain], cmap_name: str = 'RdYlGn_r'):
    """Return matplotlib patch data for overlaying domains on an FPA image.

    The transposed FPA display (x = FPA_row, y = FPA_col) is assumed.

    Parameters
    ----------
    domains   : list[Domain]
    cmap_name : str — colormap for the valid-fraction colouring

    Returns
    -------
    list of dicts, each with keys:
        x, y    : lower-left corner of patch in display coords (x=row, y=col)
        width   : patch width  (= n_rows in display coords)
        height  : patch height (= n_cols in display coords)
        color   : RGBA tuple — red if flag_geometry, else cmap mapped to vf
        label   : short string for tooltip / legend
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap  = cm.get_cmap(cmap_name)
    patches = []
    for d in domains:
        if d.flag_geometry:
            rgba = (1.0, 0.15, 0.15, 0.45)   # semi-transparent red
        else:
            # colour by valid-pixel fraction (green = fully valid)
            vf   = d.valid_fraction
            rgba = cmap(vf)[:3] + (0.25,)     # lower alpha so mask shows through

        # Transposed display: x=FPA_row, y=FPA_col
        patches.append(dict(
            x=d.row_start, y=d.col_start,
            width=d.n_rows, height=d.n_cols,
            color=rgba,
            edge_color='white' if d.flag_geometry else 'gray',
            edge_alpha=0.6,
            label=(f'SZA={d.mean_sza:.1f}±{d.range_sza:.1f}° '
                   f'VZA={d.mean_vza:.1f}±{d.range_vza:.1f}°'
                   if d.geo_populated else 'geo placeholder'),
        ))
    return patches


# ── unit-test / smoke test ────────────────────────────────────────────────────

def _smoke_test():
    """Quick sanity check — run with  python -m libcam.l2.domains"""
    print('=== domains.py smoke test ===')

    NX, NY = 2048, 2048
    rng    = np.random.default_rng(42)

    # Case 1: placeholder geometry (all zeros)
    print('\n[Case 1] All-zero geometry (placeholder)')
    sza = np.zeros((NX, NY), dtype=np.float32)
    vza = np.zeros((NX, NY), dtype=np.float32)
    raz = np.zeros((NX, NY), dtype=np.float32)
    vm  = (rng.random((NX, NY)) > 0.15).astype(np.uint8)

    d1 = chop_frame(sza, vza, raz, valid_mask=vm, tile_size=64)
    s1 = summarise_domains(d1)
    assert s1['n_flagged'] == 0,    'no tiles should be flagged for zero geometry'
    assert not s1['geo_populated'], 'geo_populated should be False for all-zero'
    n_expected = (NX // 64) * (NY // 64)
    assert s1['n_tiles'] == n_expected, f'expected {n_expected} tiles, got {s1["n_tiles"]}'
    print(f'  {s1["n_tiles"]} tiles — OK')

    # Case 2: realistic synthetic geometry (slowly varying SZA/VZA).
    # With a 40° SZA gradient over 2048 px the variation per 64-px tile is
    # ~1.25°, well below the 5° threshold.  Use tile_size=512 (8 tiles wide)
    # so each tile sees ~10° variation → all tiles on the gradient boundary
    # should be flagged.
    print('\n[Case 2] Synthetic geometry (smoothly varying SZA 30–70°, VZA 0–60°)')
    row_idx = np.linspace(0, 1, NX, dtype=np.float32)[:, np.newaxis]
    col_idx = np.linspace(0, 1, NY, dtype=np.float32)[np.newaxis, :]
    sza2    = (30.0 + 40.0 * row_idx * np.ones_like(col_idx)).astype(np.float32)
    vza2    = (0.0  + 60.0 * col_idx * np.ones_like(row_idx)).astype(np.float32)
    raz2    = (np.ones((NX, NY), dtype=np.float32) * 90.0)

    # With tile_size=512: ~10° SZA per tile and ~15° VZA per tile → flagged
    d2 = chop_frame(sza2, vza2, raz2, valid_mask=vm, tile_size=512,
                    max_sza_range=5.0, max_vza_range=5.0)
    s2 = summarise_domains(d2)
    assert s2['geo_populated'], 'geo_populated should be True for non-zero geometry'
    assert s2['n_flagged'] > 0, 'tiles should be flagged for 10° per-tile SZA gradient'
    print(f'  {s2["n_flagged"]}/{s2["n_tiles"]} flagged — OK')
    print(f'  SZA mean={s2["sza_mean"]:.1f}°  range_max={s2["sza_range_max"]:.2f}°')
    # Also verify tile_size=64 produces fine tiles that mostly pass the threshold
    d2b = chop_frame(sza2, vza2, raz2, valid_mask=vm, tile_size=64,
                     max_sza_range=5.0, max_vza_range=5.0)
    s2b = summarise_domains(d2b, verbose=False)
    assert s2b['n_tiles'] == 1024
    print(f'  tile_size=64: {s2b["n_flagged"]}/{s2b["n_tiles"]} flagged '
          f'(SZA range_max={s2b["sza_range_max"]:.2f}°) — expected ~0 — OK')

    # Case 3: edge tiles (non-multiple FPA size)
    print('\n[Case 3] Non-multiple FPA (100×100 with tile_size=64)')
    sza3 = np.full((100, 100), 45.0, dtype=np.float32)
    vza3 = np.full((100, 100), 20.0, dtype=np.float32)
    raz3 = np.full((100, 100), 90.0, dtype=np.float32)
    d3   = chop_frame(sza3, vza3, raz3, tile_size=64)
    tile_sizes = set((d.n_rows, d.n_cols) for d in d3)
    print(f'  Tile sizes present: {tile_sizes}')
    total_px = sum(d.n_pixels for d in d3)
    assert total_px == 100 * 100, f'total pixels mismatch: {total_px}'
    print(f'  Total pixels covered: {total_px} — OK')

    print('\n=== smoke test passed ===')


if __name__ == '__main__':
    _smoke_test()
