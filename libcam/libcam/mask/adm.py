"""ADM lock-on pixel mask for the Libera WFOV camera.

Selects, for each exposure, the FPA pixels that sample a chosen set of
(VZA, VAA) angular bins -- the CERES/TRMM angular bin convention -- and grows a
circular patch around each selected pixel to cover the Libera PSF.

Provenance
----------
Algorithm by Jake Gristey (NOAA), 2021-2024; the original flat script is kept
verbatim at ``reference/pmask_adm_original.py``. A C translation by Hong Chen
(LASP) is at ``reference/pmask_adm.c`` and is the version that matches flight
software structure. This module is a refactor of the Python that fixes one
divergence from the C -- see ``clip`` below and ``libcam/mask/BOUNDS_BUG.md``.

Two deliberate departures from the ancestral Python, both to match what flies:
the sample centres are TRUNCATED rather than rounded (2026-10-06; the C
reference was corrected to match, since its own header claims to follow flight
structure and on that line it did not), and ``asin_mode`` can reproduce the
flight-software arcsine defect. ``tests/test_c_python_equivalence.py`` asserts
C and Python agree pixel for pixel, so the two cannot drift apart again -- but
note it SKIPS silently without a C compiler or without the reference file
present, which is how the truncation change reached a commit candidate with
that test quietly not running.
"""

import numpy as np

__all__ = ["adm_mask", "adm_sample_centres", "vza_surface_to_sat",
           "ASIN_MODES", "DEFAULTS"]

# Angular bin boundaries in the SURFACE reference frame, matching CERES-TRMM.
VZA_BOUNDARIES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
VAA_BOUNDARIES = np.arange(0, 361, 20, dtype=float)

DEFAULTS = dict(
    percent_samp=100,        # percentage of angular bins sampled per exposure
    max_radius_of_pix=20,    # circular radius [pixels] of the patch per ADM sample
    vza_shift=0.0,           # shift applied to the VZA bin centre [deg]
    vaa_shift=0.0,           # shift applied to the VAA bin centre [deg]
    x_center_pix=1024,       # optical axis on the FPA
    y_center_pix=1024,
)

# Fixed instrument / orbit constants.
PIX_SPACING_MM = 0.0055
COEF_ANG2DIST = [0.0, -2.07789e-6, -1.16828e-5, 9.92181e-2, 0.0]  # from cam code
EARTH_RADIUS_KM = 6371.0
ALTITUDE_KM = 824.0
FPA_NX = 2048
FPA_NY = 2048

# The original code hard-codes pi=3.14159 rather than using math.pi. That 8.5e-7
# relative difference is retained deliberately so this module reproduces the C
# and the flight code bit-for-bit. Do not "fix" it without re-running
# compare_c_python.py.
PI = 3.14159


# ─── surface -> satellite frame: the one place this conversion lives ─────────
#
# "correct" is the intended geometry. "as_coded" reproduces a defect in the
# flight software that is live as of 2026-10-06:
#
#     WFOVPixelMask_ADM.cpp :: getSelectedADMBins()
#         vza_sat_ref = asinf((sinf((vza / 180.0f) * M_PI)
#                              * WFOV_EARTH_SAT_RATIO) / M_PI) * 180.0f;
#
# The / M_PI sits INSIDE the asinf argument instead of being applied to the
# result. With u = sin(vza) * ratio the intended form expands as
# (180/pi)(u + u^3/6 + 3u^5/40 + ...) and the coded one as
# (180/pi)(u + u^3/6pi^2 + 3u^5/40pi^4 + ...): identical linear terms, every
# higher term suppressed by a power of pi^2 = 9.87. The line linearises the
# arcsine. Error is exactly zero at nadir and -10.7 deg of commanded VZA at the
# rim, which is why it hid for months and why it looked like a scale factor on
# the polynomial.
#
# Keep BOTH. The retrieval decides which one the imagery is consistent with, so
# that the day the fix flies, the answer changes by itself. Do not delete
# "as_coded" when that happens -- old granules still need it to be read back.
ASIN_MODES = ("correct", "as_coded")
FSW_ASIN_MODE = "as_coded"   # what flight is believed to run, as of 2026-10-06
EARTH_SAT_RATIO = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + ALTITUDE_KM)


def vza_surface_to_sat(vza_deg, asin_mode="correct"):
    """Surface VZA [deg] -> satellite-frame VZA [deg], spherical Earth.

    ``asin_mode="as_coded"`` reproduces the flight-software defect described
    above. See ``docs/FSW_KNOWN_BEHAVIOUR.md``.
    """
    if asin_mode not in ASIN_MODES:
        raise ValueError("asin_mode must be one of %r, got %r"
                         % (ASIN_MODES,asin_mode))
    u = np.sin(np.asarray(vza_deg, dtype=float) / 180 * PI) * EARTH_SAT_RATIO
    if asin_mode == "as_coded":
        return np.arcsin(u / PI) * 180            # the misplaced / M_PI
    return np.arcsin(u) / PI * 180


def _select_bins(percent_samp):
    """Evenly-spaced subset of the (VZA x VAA) bin grid.

    ``int()`` truncation is deliberate -- it rounds the sample count down, which
    is the conservative direction for the data rate.
    """
    n_vza = len(VZA_BOUNDARIES) - 1
    n_vaa = len(VAA_BOUNDARIES) - 1
    n_bins = n_vza * n_vaa
    n_samples = int(percent_samp / 100.0 * n_bins)
    if n_samples < 1:
        return [], n_vaa
    return [int(n_bins / n_samples * i) for i in range(n_samples)], n_vaa


def adm_sample_centres(percent_samp=100, vza_shift=0.0, vaa_shift=0.0,
                       x_center_pix=1024, y_center_pix=1024, coef_ang2dist=None,
                       asin_mode="correct"):
    """FPA pixel coordinates of the ADM sample centres.

    Returns ``(x, y)`` integer arrays. Coordinates may fall outside the FPA for
    the highest VZA bins; that is expected and is handled by ``adm_mask``.

    ``coef_ang2dist`` overrides the field-angle polynomial. It defaults to the
    ground value, which is NOT what flight software uses -- the flown ring radii
    disagree (8 rings, 145-855 px, versus 9 at 85-1015 px). The flight value is
    not a WFOV table parameter and is believed hard-coded; see
    ``wfov_csv2cfg.py``.

    As of 2026-10-06 that disagreement is understood and is NOT a coefficient
    problem: flight runs these same v0 coefficients through a defective
    surface-to-satellite conversion. Pass ``asin_mode="as_coded"`` to reproduce
    the flown geometry; the coefficients do not need changing.
    """
    coef = list(COEF_ANG2DIST if coef_ang2dist is None else coef_ang2dist)
    bins, n_vaa = _select_bins(percent_samp)
    xs, ys = [], []
    for bin_num in bins:
        vza_i, vaa_i = divmod(bin_num, n_vaa)

        vza = VZA_BOUNDARIES[vza_i] + 0.5 * (
            VZA_BOUNDARIES[vza_i + 1] - VZA_BOUNDARIES[vza_i]) + vza_shift
        vaa = VAA_BOUNDARIES[vaa_i] + 0.5 * (
            VAA_BOUNDARIES[vaa_i + 1] - VAA_BOUNDARIES[vaa_i]) + vaa_shift

        # A shift can push the angle out of range; wrap it back.
        vza = vza % 90
        vaa = vaa % 360

        # Surface -> satellite reference frame (spherical Earth).
        vza_sat = vza_surface_to_sat(vza, asin_mode)

        # Radial distance on the FPA, via the camera distortion polynomial.
        dist = 0.0
        for i, c in enumerate(coef):
            dist += c * vza_sat ** (len(coef) - i - 1)

        # TRUNCATE, do not round. Flight software writes
        #     (int)(xcenter + x_dist_from_center / WFOV_MASK_ADM_PIX_SPACING)
        # in WFOVPixelMask_ADM.cpp::getSelectedADMBins(), and a C cast to int
        # truncates toward zero. These coordinates are ~1000, so it always
        # floors: every sample centre sits up to one pixel closer to the origin
        # than rounding would put it. Matched here deliberately (S. Schmidt,
        # 2026-10-06) rather than asking FSW to change -- the point of this
        # module is to reproduce what flies, not what is tidiest.
        xs.append(int(x_center_pix + dist * np.sin(vaa / 180 * PI) / PIX_SPACING_MM))
        ys.append(int(y_center_pix + dist * np.cos(vaa / 180 * PI) / PIX_SPACING_MM))

    return np.array(xs, dtype=int), np.array(ys, dtype=int)


def adm_mask(percent_samp=100, max_radius_of_pix=20, vza_shift=0.0, vaa_shift=0.0,
             x_center_pix=1024, y_center_pix=1024, clip=True, nx=FPA_NX, ny=FPA_NY,
             coef_ang2dist=None, asin_mode="correct"):
    """Build the ADM lock-on pixel mask.

    Returns a ``(nx, ny)`` uint8 array indexed ``mask[x, y]`` -- note this is
    x-major, i.e. transposed relative to image display convention. Use
    ``mask.T`` when plotting with ``imshow``, as the reference scripts do.

    Parameters
    ----------
    clip : bool
        If True (default, and matching ``reference/pmask_adm.c``), patches are
        clipped at all four FPA edges. If False, the lower bound is not enforced,
        reproducing the original Python, in which a patch overhanging the x=0 or
        y=0 edge wraps around to the opposite edge via negative indexing. That
        wrap silently places ADM samples on the wrong side of the detector
        whenever the optical axis is not exactly at the array centre -- with the
        measured centre (1026, 1000) it misplaces ~1500 pixels. Only set
        ``clip=False`` to reproduce historical masks; see ``libcam/mask/BOUNDS_BUG.md``.
    """
    xs, ys = adm_sample_centres(percent_samp, vza_shift, vaa_shift,
                                x_center_pix, y_center_pix, coef_ang2dist,
                                asin_mode)
    mask = np.zeros((nx, ny), dtype=np.uint8)
    r = max_radius_of_pix
    r2 = (r + 0.5) ** 2  # +0.5 buffer, matches the cam result

    for xc, yc in zip(xs, ys):
        if clip:
            x0, x1 = max(xc - r, 0), min(xc + r + 1, nx)
            y0, y1 = max(yc - r, 0), min(yc + r + 1, ny)
            if x1 <= x0 or y1 <= y0:
                continue
            gx = np.arange(x0, x1)
            gy = np.arange(y0, y1)
            inside = ((gx[:, None] - xc) ** 2 + (gy[None, :] - yc) ** 2) <= r2
            mask[x0:x1, y0:y1][inside] = 1
        else:
            # Faithful reproduction of the original scalar loop, including the
            # negative-index wraparound. Slicing cannot express this -- a negative
            # slice start does not wrap the way a negative element index does --
            # so the element-wise loop is kept deliberately.
            for x in range(xc - r, xc + r + 1):
                if x >= nx:
                    continue
                for y in range(yc - r, yc + r + 1):
                    if y >= ny:
                        continue
                    if (x - xc) ** 2 + (y - yc) ** 2 <= r2:
                        mask[x][y] = 1

    return mask


def _main(from_table=False, table_dir=None, frame=0, plot=None, table_overrides=None, **overrides):
    """Tester: build the ADM mask, optionally from the WFOV table, and plot it.

    The keyword defaults are what the editable block at the bottom of this file
    sets. Any command-line flag overrides them.

    ``**overrides`` takes ``adm_mask`` keywords that the WFOV table does **not**
    define:

        _main(from_table=True, percent_samp=50, max_radius_of_pix=12)

    ``table_overrides`` replaces individual WFOV table entries, keyed by the
    actual table parameter name (the string ``cfg.report()`` prints). Those names
    are unique, so this is how you target one thing rather than all of them --
    essential for the stripe families, which share generator keyword names:

        _main(from_table=True, table_overrides=dict(wfov_tbl_maxPixR=12, wfov_tbl_persamp=50))

    Table parameters (``percent_samp``, ``max_radius_of_pix``, ``x_center_pix``,
    ``y_center_pix``, ``vza_shift``, ``vaa_shift``, ``*_init``, ``coef_ang2dist``)
    override the table value, so the frame precession is then computed from the
    overridden shift. ``clip``, ``nx`` and ``ny`` are not table parameters and go
    straight to the generator.
    """
    import argparse, os, sys
    if __package__ in (None, ""):
        # Running this file as a script (`python libcam/mask/adm.py`) puts
        # libcam/mask/ on sys.path, not the repo root -- so `import libcam...`
        # would fail. Walk up three levels to the repo root and add that:
        #     adm.py  ->  libcam/mask/  ->  libcam/  ->  <repo root>
        _root = os.path.abspath(__file__)
        for _ in range(3):
            _root = os.path.dirname(_root)
        sys.path.insert(0, _root)

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--from_table", action="store_true", default=from_table,
                   help="take parameters from the WFOV tables instead of defaults")
    p.add_argument("--table_dir", default=table_dir)
    p.add_argument("--frame", type=int, default=frame,
                   help="frame index; applies the per-mask precession")
    p.add_argument("--plot", nargs="?", const="adm_mask.png", default=plot,
                   help="write a PNG of the mask")
    a = p.parse_args()

    kw, note = {}, "built-in defaults"
    if a.from_table:
        from libcam.mask.wfov_csv2cfg import (load_tables, apply_table_overrides,
                                              adm_cfg, AdmCfg,
                                              split_overrides)
        cfg_over, gen_over = split_overrides(
            AdmCfg, overrides, extra_ok=("clip", "nx", "ny"))
        cfg = adm_cfg(apply_table_overrides(load_tables(a.table_dir), table_overrides), **cfg_over)
        cfg.report()
        vza_s, vaa_s = cfg.shift_at(a.frame)
        # shift_num is carried but not consumed -- see AdmCfg docstring
        kw = cfg.as_kwargs("vaa_shift", "vza_shift",
                           "vaa_shift_init", "vza_shift_init", "shift_num")
        kw["vza_shift"], kw["vaa_shift"] = vza_s, vaa_s
        kw.update(gen_over)
        note = "WFOV table, frame %d (vza_shift=%.2f, vaa_shift=%.2f)" % (
            a.frame, vza_s, vaa_s)
        if cfg_over or gen_over:
            note += " + overrides"
        print()
    else:
        kw = dict(overrides)
        if kw:
            note += " + overrides"

    m = adm_mask(**kw)
    n = int(m.sum())
    print("ADM mask [%s]: %d populated pixels (%.3f %% of the FPA)"
          % (note, n, 100.0 * n / (FPA_NX * FPA_NY)))

    if a.plot:
        # No backend is forced: the figure opens interactively where a display
        # is available, and is saved to output/ either way.
        import matplotlib.pyplot as plt
        from libcam.mask.io import figure_path
        fig, ax = plt.subplots(figsize=(7.4, 7.6))
        ax.imshow(np.where(m.T, 0, 1), cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ax.set_title("ADM mask - %s" % note, fontsize=11, loc="left")
        ax.set_xlabel("%d populated  (%.3f %% of FPA)"
                      % (n, 100.0 * n / (FPA_NX * FPA_NY)), fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = figure_path(a.plot)
        fig.savefig(out, dpi=130)
        print("wrote %s" % out)
        plt.show()
    return m


if __name__ == "__main__":
    # ─── settings ────────────────────────────────────────────────────────────
    # Edit these to run from an editor. Each is also a command-line flag, and
    # the flag wins when given:  python libcam/mask/adm.py --frame 40 --plot
    FROM_TABLE = True          # False -> use the built-in defaults instead
    TABLE_DIR  = None          # None  -> docs/wfov_tables
    FRAME      = 0             # frame index; applies vza/vaa precession
    PLOT       = "adm_mask.png"   # None -> no figure
    #
    # Generator keywords that are NOT WFOV table parameters. Anything the table
    # does define must go in TABLE_OVERRIDES below instead -- passing it here is
    # an error naming the table parameter to use. e.g.
    #     OVERRIDES = dict(coef_ang2dist=[...], clip=False)
    OVERRIDES  = dict()
    #
    # The route for every WFOV table parameter. Keyed by the real table name,
    # which is unique -- so this is what targets ONE component. Copy the names
    # straight out of the report above the run, e.g.
    #     TABLE_OVERRIDES = dict(wfov_tbl_maxPixR=12, wfov_tbl_persamp=50)
    TABLE_OVERRIDES = dict()
    # ─────────────────────────────────────────────────────────────────────────
    _main(from_table=FROM_TABLE, table_dir=TABLE_DIR, frame=FRAME, plot=PLOT,
          table_overrides=TABLE_OVERRIDES, **OVERRIDES)
