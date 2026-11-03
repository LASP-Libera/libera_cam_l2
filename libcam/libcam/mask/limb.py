"""Limb pixel mask for the Libera WFOV camera.

An annulus: every pixel whose distance from ``(xc, yc)`` lies strictly between
``r1`` and ``r2``. Used for limb viewing, where the scene of interest is a ring
near the edge of the field rather than the disc.

Provenance: S. Schmidt (originally written for Beth/Nicolette); refactored here
from ``camera/pixelmasks/pixelmask_limb_simple.py``, kept verbatim at
``reference/pixelmask_limb_simple.py``.

Table parameters live in ``docs/wfov_tables/icie_tbl_wfov_limb.csv``
(``xcenter_limb``, ``ycenter_limb``, ``r1``, ``r2``) and reach this module
through :mod:`libcam.mask.wfov_csv2cfg`.
"""

import numpy as np

__all__ = ["limb_mask"]

FPA_NX = 2048
FPA_NY = 2048


def limb_mask(xp=FPA_NX, yp=FPA_NY, xc=1024, yc=1024, r1=895, r2=1024):
    """Build an annular limb pixel mask.

    Returns an ``(xp, yp)`` uint8 array indexed ``mask[x, y]`` — x-major,
    transposed relative to image display convention, matching the other
    generators in this package. Transpose for plotting: ``imshow(mask.T)``.

    Parameters
    ----------
    xc, yc : int    centre pixel in FPA coordinates.
    r1, r2 : float  inner and outer radius [pixels]. Both bounds are **strict**
                    (``r1 < r < r2``), matching the original, so pixels sitting
                    exactly on either radius are excluded.

    Notes
    -----
    The original built its coordinate grid with ``np.meshgrid`` (default
    ``indexing='xy'``) and then indexed a ``(xp, yp)`` array with those row/column
    indices, which silently mixes the two conventions. For the default centred
    annulus the result is symmetric under transpose so the two agree exactly; for
    ``xc != yc`` the original produces a *transposed* annulus. This version is
    explicitly ``mask[x, y]`` throughout and so differs from the original off
    centre — deliberately. See ``tests/test_masks.py``.
    """
    if r1 > r2:
        r1, r2 = r2, r1
    x = np.arange(xp) - xc
    y = np.arange(yp) - yc
    rr2 = (x[:, None] ** 2) + (y[None, :] ** 2)      # [x, y]
    mask = np.zeros((xp, yp), dtype=np.uint8)
    mask[(rr2 > r1 ** 2) & (rr2 < r2 ** 2)] = 1
    return mask


def _main(from_table=False, table_dir=None, plot=None, table_overrides=None, **overrides):
    """Tester: build the limb mask, optionally from the WFOV table, and plot it.

    The keyword defaults are what the editable block at the bottom of this file
    sets. Any command-line flag overrides them.

    ``**overrides`` takes ``limb_mask`` keywords that the WFOV table does **not**
    define:

        _main(from_table=True, r1=800, r2=1000)

    ``table_overrides`` replaces individual WFOV table entries, keyed by the
    actual table parameter name (the string ``cfg.report()`` prints). Those names
    are unique, so this is how you target one thing rather than all of them --
    essential for the stripe families, which share generator keyword names:

        _main(from_table=True, table_overrides=dict(wfov_tbl_r1=800, wfov_tbl_r2=1000))

    Table parameters are ``xc``, ``yc``, ``r1``, ``r2``; ``xp`` and ``yp`` go
    straight to the generator.
    """
    import argparse, os, sys
    if __package__ in (None, ""):
        # Running this file as a script (`python libcam/libcam/mask/limb.py`)
        # puts libcam/libcam/mask/ on sys.path, not the repo root -- so
        # `import libcam...` would fail. Walk up four levels to the repo
        # root and add that (one more than before the 2026-11-03 uv_build
        # re-nesting -- see docs/PATCHES.md):
        #     limb.py  ->  libcam/libcam/mask/  ->  libcam/libcam/  ->  libcam/  ->  <repo root>
        _root = os.path.abspath(__file__)
        for _ in range(4):
            _root = os.path.dirname(_root)
        sys.path.insert(0, _root)

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--from_table", action="store_true", default=from_table,
                   help="take parameters from the WFOV tables instead of defaults")
    p.add_argument("--table_dir", default=table_dir)
    p.add_argument("--plot", nargs="?", const="limb_mask.png", default=plot,
                   help="write a PNG of the mask")
    a = p.parse_args()

    kw, note = {}, "built-in defaults"
    if a.from_table:
        from libcam.mask.wfov_csv2cfg import (load_tables, apply_table_overrides,
                                              limb_cfg, LimbCfg,
                                              split_overrides)
        cfg_over, gen_over = split_overrides(
            LimbCfg, overrides, extra_ok=("xp", "yp"))
        cfg = limb_cfg(apply_table_overrides(load_tables(a.table_dir), table_overrides), **cfg_over)
        cfg.report()
        kw = cfg.as_kwargs()
        kw.update(gen_over)
        note = "WFOV table (r1=%s, r2=%s)" % (cfg.r1, cfg.r2)
        if cfg_over or gen_over:
            note += " + overrides"
        print()
    else:
        kw = dict(overrides)
        if kw:
            note += " + overrides"

    m = limb_mask(**kw)
    n = int(m.sum())
    print("limb mask [%s]: %d populated pixels (%.3f %% of the FPA)"
          % (note, n, 100.0 * n / (FPA_NX * FPA_NY)))

    if a.plot:
        # No backend is forced: the figure opens interactively where a display
        # is available, and is saved to output/ either way.
        import matplotlib.pyplot as plt
        from libcam.mask.io import figure_path
        fig, ax = plt.subplots(figsize=(7.4, 7.6))
        ax.imshow(np.where(m.T, 0, 1), cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")
        ax.set_title("limb mask - %s" % note, fontsize=11, loc="left")
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
    # the flag wins when given:  python libcam/mask/limb.py --plot
    FROM_TABLE = True            # False -> use the built-in defaults instead
    TABLE_DIR  = None            # None  -> docs/wfov_tables
    PLOT       = "limb_mask.png"    # None -> no figure
    #
    # Generator keywords that are NOT WFOV table parameters. Anything the table
    # does define must go in TABLE_OVERRIDES below instead -- passing it here is
    # an error naming the table parameter to use. e.g.
    #     OVERRIDES = dict(xp=1024, yp=1024)
    OVERRIDES  = dict()
    #
    # The route for every WFOV table parameter. Keyed by the real table name,
    # which is unique -- so this is what targets ONE component. Copy the names
    # straight out of the report above the run, e.g.
    #     TABLE_OVERRIDES = dict(wfov_tbl_r1=800, wfov_tbl_r2=1000)
    TABLE_OVERRIDES = dict()
    # ─────────────────────────────────────────────────────────────────────────
    _main(from_table=FROM_TABLE, table_dir=TABLE_DIR, plot=PLOT,
          table_overrides=TABLE_OVERRIDES, **OVERRIDES)
