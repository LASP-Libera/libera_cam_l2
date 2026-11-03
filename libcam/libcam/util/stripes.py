"""
libcam/util/stripes.py
-----------------------
Detect bright stripe patterns and their intersections in WFOV camera images.

A typical cross-track observation produces a characteristic pattern:
  - One vertical stripe (along-track nadir stripe)
  - One or more horizontal stripes (cross-track sweep lines)
  - Two intersection points when both full and partial horizontal stripes are
    present (indicator of cross-track operational mode)

Public API
----------
detect_stripe_intersections(img, dn_thresh, min_lit_frac)
    → dict with vertical_cols, horizontal_rows, intersections, cross_track,
           stripe_lengths

_find_stripe_centers(proj, min_lit_px)
    Internal helper: cluster a 1D projection into stripe center positions.

Usage
-----
  from libcam.util.stripes import detect_stripe_intersections

  result = detect_stripe_intersections(img)
  if result['cross_track']:
      for col, row in result['intersections']:
          print(f'  intersection at col={col}, row={row}')
"""

import numpy as np

__all__ = ['detect_stripe_intersections']


def _find_stripe_centers(proj, min_lit_px, max_gap=3):
    """Cluster a 1D lit-pixel projection into stripe center positions.

    Parameters
    ----------
    proj        : 1D int array  — number of lit pixels at each row or column
    min_lit_px  : int           — minimum value for a position to be part of a stripe
    max_gap     : int           — maximum index gap to bridge within a single stripe

    Returns
    -------
    centers : list of int   — centroid index of each detected stripe
    lengths : list of int   — peak lit-pixel count for each stripe
                              (proxy for stripe length along its axis)
    """
    hot = np.where(proj >= min_lit_px)[0]
    if len(hot) == 0:
        return [], []

    # Group consecutive (or near-consecutive) indices into clusters
    groups = []
    group  = [int(hot[0])]
    for idx in hot[1:]:
        if int(idx) - group[-1] <= max_gap:
            group.append(int(idx))
        else:
            groups.append(group)
            group = [int(idx)]
    groups.append(group)

    centers = []
    lengths = []
    for g in groups:
        weights  = proj[g].astype(float)
        center   = int(round(np.average(g, weights=weights)))
        peak_len = int(proj[g].max())
        centers.append(center)
        lengths.append(peak_len)

    return centers, lengths


def detect_stripe_intersections(img, dn_thresh=50, min_lit_frac=0.05):
    """Detect horizontal and vertical bright stripes and their intersections.

    Uses 1D row/column projections (count of pixels above *dn_thresh*) to
    locate stripe centre positions, then returns every (col, row) crossing.

    Cross-track mode indicator: when >= 2 intersections are found the camera
    was likely in a cross-track observation mode (full nadir stripe + sweep
    line(s) produce >= 2 crossings with the vertical nadir stripe).

    Parameters
    ----------
    img          : (NY, NX) array-like  — 2D camera image (uint16 or float)
    dn_thresh    : int    — DN threshold for classifying a pixel as lit
                            (default 50, well above dark-frame noise floor)
    min_lit_frac : float  — minimum fraction of dimension that must be lit
                            for a row/column to count as a stripe
                            (default 0.05 = 5 % of NY or NX)

    Returns
    -------
    dict with keys:
        vertical_cols    : list of int       — column centre(s) of vertical stripes
        horizontal_rows  : list of int       — row centre(s) of horizontal stripes
        intersections    : list of (col, row) — all (vertical×horizontal) crossings
        cross_track      : bool              — True when >= 2 intersections found
        stripe_lengths   : dict              — lit-pixel count for each stripe,
                                               keyed 'v_<col>' and 'h_<row>'
        row_proj         : ndarray           — raw row projection (lit px per row)
        col_proj         : ndarray           — raw col projection (lit px per col)
    """
    img = np.asarray(img)
    ny, nx = img.shape

    mask     = img > dn_thresh
    row_proj = mask.sum(axis=1).astype(int)   # lit pixels per row  → horizontal stripes
    col_proj = mask.sum(axis=0).astype(int)   # lit pixels per col  → vertical stripes

    min_row_px = int(min_lit_frac * nx)       # e.g. 5 % of 2048 = ~102 px
    min_col_px = int(min_lit_frac * ny)

    v_centers, v_lengths = _find_stripe_centers(col_proj, min_col_px)
    h_centers, h_lengths = _find_stripe_centers(row_proj, min_row_px)

    intersections = [(col, row) for col in v_centers for row in h_centers]

    stripe_lengths = {}
    for col, lng in zip(v_centers, v_lengths):
        stripe_lengths[f'v_{col}'] = lng
    for row, lng in zip(h_centers, h_lengths):
        stripe_lengths[f'h_{row}'] = lng

    return {
        'vertical_cols':   v_centers,
        'horizontal_rows': h_centers,
        'intersections':   intersections,
        'cross_track':     len(intersections) >= 2,
        'stripe_lengths':  stripe_lengths,
        'row_proj':        row_proj,
        'col_proj':        col_proj,
    }
