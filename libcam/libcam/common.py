"""
libcam/common.py — Libera WFOVC camera constants

Lens model polynomial coefficients from pre-flight calibration.
FPA: 2048×2048 pixels, 5.5 µm pitch.

COEF_DIST2ANG : r (mm) → VZA (deg)   — forward projection (pixel → ground)
COEF_ANG2DIST : VZA (deg) → r (mm)   — inverse lens model (ground → pixel)
"""

import os
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
PLANET_RADIUS = 6370997.0        # m  (sphere approximation used throughout)
LIBERA_ALTITUDE = 824000.0       # m  (nominal CLARREO Pathfinder orbit)

# ── Focal Plane Array ─────────────────────────────────────────────────────────
FPA_NX = 2048                    # pixels along x (across-track direction)
FPA_NY = 2048                    # pixels along y (along-track direction)
FPA_DX = 0.0055                  # mm  pixel pitch x
FPA_DY = 0.0055                  # mm  pixel pitch y

# ── Lens model ────────────────────────────────────────────────────────────────
# r = sqrt((fpa_x)**2 + (fpa_y)**2)  in mm, where fpa_x/y are measured from
# the optical axis (centre of the FPA).

# VZA = polyval(COEF_DIST2ANG, r)   units: degrees
COEF_DIST2ANG = np.array([
     1.397428e-02,
    -1.500364e-01,
     6.352646e-01,
    -9.551312e-01,
     9.042359,
     0.0,
])

# r = polyval(COEF_ANG2DIST, VZA)   units: mm
COEF_ANG2DIST = np.array([
     1.21367e-10,
     2.60014e-09,
    -7.50181e-06,
    -2.84788e-06,
     1.17022e-01,
     0.0,
])

# ── FOV limit ─────────────────────────────────────────────────────────────────
VZA_LIMIT = 63.0                 # degrees — pixels beyond this are masked

# ── Digitisation ─────────────────────────────────────────────────────────────
NBIT = 12                        # ADC resolution

