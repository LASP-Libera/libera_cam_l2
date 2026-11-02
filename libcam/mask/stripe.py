"""Stripe pixel mask for the Libera WFOV camera.

A straight band of pixels across the FPA, of configurable offset, rotation and
width, clipped to a circular field-of-view radius. Used for the nadir /
along-track / cross-track stripe products.

Provenance: S. Schmidt, Oct 2021 (originally written for Beth/Magnus);
refactored here from ``camera/py/pixelmasks_funcs.py::pix_stripe``.

All parameters except ``xp``/``yp`` are uplinkable from the ground.
"""

import numpy as np

__all__ = ["stripe_mask"]

FPA_NX = 2048
FPA_NY = 2048


def stripe_mask(xp=FPA_NX, yp=FPA_NY, xc=1024, yc=1024, fr=1000,
                sy=0, sr=0.0, su=20, sd=20, clip_fov=False):
    """Build a stripe pixel mask.

    Returns a ``(xp, yp)`` uint8 array indexed ``mask[x, y]`` -- x-major, i.e.
    transposed relative to image display convention. Use ``mask.T`` for
    ``imshow``.

    Parameters
    ----------
    xc, yc : int    centre pixel in FPA coordinates (the optical axis).
    fr : int        radius of the field of view to populate [pixels].
    sy : int        offset of the stripe from the centre, along y [pixels].
    sr : float      stripe rotation angle [degrees].
    su, sd : int    stripe half-widths above and below the generator line
                    [pixels]; total width is ``1 + su + sd``.
    clip_fov : bool Clip the stripe to the circle of radius ``fr`` about
                    ``(xc, yc)``. **Default False, which is what flight software
                    does** -- FSW performs no circular clipping and emits a plain
                    rectangle. Note ``fr`` still sets the stripe *length* either
                    way (the generator line runs ``xc-fr`` to ``xc+fr``); the flag
                    only controls whether the corners are trimmed.

                    With clipping on, every row except the centre one loses its
                    two end pixels -- 80 px on the 2025_294 nadir stripe. Set True
                    only to reproduce the original ground code, which did clip.

    Notes
    -----
    The stripe is generated axis-aligned, then rotated about ``(xc, yc)`` and
    clipped to the circle of radius ``fr``. Rotation is applied to pixel centres
    and the result is rounded, so for non-multiples of 90 degrees the rendered
    stripe edge is slightly ragged and the populated count varies by a few
    pixels with angle. That matches the flight behaviour and is not corrected
    here.
    """
    width = 1 + su + sd

    # Generator line plus its parallel offsets, in unrotated coordinates.
    xg = np.arange(xc - fr, xc + fr + 1, 1)
    xg = xg[(xg >= 0) & (xg < xp)]

    offsets = [0]
    offsets += [i + 1 for i in range(su)]
    offsets += [-(i + 1) for i in range(sd)]
    assert len(offsets) == width

    xi = np.repeat(xg[:, None], width, axis=1).astype(float)
    yi = np.tile(np.array(offsets, dtype=float) + yc + sy, (len(xg), 1))

    # Rotate the whole object about (xc, yc).
    th = np.deg2rad(sr)
    ct, st = np.cos(th), np.sin(th)
    dx = xi - xc
    dy = yi - yc
    xr = ct * dx - st * dy + xc
    yr = st * dx + ct * dy + yc

    # Optionally clip to the circular field of view, then always to the array.
    # Truncation (not rounding) is deliberate: the original used int(), and at
    # rotations near 90 degrees the two differ by ~100 pixels at the stripe ends.
    if clip_fov:
        keep = ((xr - xc) ** 2 + (yr - yc) ** 2) <= fr ** 2
        xr, yr = xr[keep], yr[keep]
    else:
        xr, yr = xr.ravel(), yr.ravel()
    xr = xr.astype(int)
    yr = yr.astype(int)
    # The original guarded xr >= 0 but not yr >= 0, so a stripe offset far enough
    # in -y wrapped to the opposite edge. Both bounds are enforced here.
    keep = (xr >= 0) & (xr < xp) & (yr >= 0) & (yr < yp)

    mask = np.zeros((xp, yp), dtype=np.uint8)
    mask[xr[keep], yr[keep]] = 1
    return mask


def _main(from_table=False, table_dir=None, family="all", az=0.0, plot=None,
          table_overrides=None, **overrides):
    """Tester: build stripe masks, optionally from the WFOV table, and plot them.

    The keyword defaults are what the editable block at the bottom of this file
    sets. Any command-line flag overrides them.

    ``**overrides`` takes ``stripe_mask`` keywords that the WFOV table does **not**
    define for every family requested:

        _main(from_table=True, family='raps', xc=1015, yc=949, fr=1000)

    ``table_overrides`` replaces individual WFOV table entries, keyed by the
    actual table parameter name (the string ``cfg.report()`` prints). Those names
    are unique, so this is how you target one thing rather than all of them --
    essential for the stripe families, which share generator keyword names:

        _main(from_table=True, table_overrides=dict(wfov_tbl_su_forward=15, wfov_tbl_su_cross=30))

    Table parameters are ``xc``, ``yc``, ``fr``, ``sy``, ``sr``, ``su``, ``sd``,
    ``az_rot``; ``clip_fov``, ``xp`` and ``yp`` go straight to the generator.
    """
    import argparse, os, sys
    if __package__ in (None, ""):
        # Running this file as a script (`python libcam/mask/stripe.py`) puts
        # libcam/mask/ on sys.path, not the repo root -- so `import libcam...`
        # would fail. Walk up three levels to the repo root and add that:
        #     stripe.py  ->  libcam/mask/  ->  libcam/  ->  <repo root>
        _root = os.path.abspath(__file__)
        for _ in range(3):
            _root = os.path.dirname(_root)
        sys.path.insert(0, _root)

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--from_table", action="store_true", default=from_table,
                   help="take parameters from the WFOV tables instead of defaults")
    p.add_argument("--table_dir", default=table_dir)
    p.add_argument("--family", default=family,
                   help="nadir | forward | cross | raps | all")
    p.add_argument("--az", type=float, default=az,
                   help="mechanism azimuth [deg]; added to sr when az_rot is set")
    p.add_argument("--plot", nargs="?", const="stripe_mask.png", default=plot,
                   help="write a PNG of the mask(s)")
    a = p.parse_args()

    if a.from_table:
        from libcam.mask.wfov_csv2cfg import (load_tables, apply_table_overrides,
                                              stripe_cfg, StripeCfg,
                                              STRIPE_FAMILIES, split_overrides)
        cfg_over, gen_over = split_overrides(
            StripeCfg, overrides, extra_ok=("clip_fov", "xp", "yp"))
        fams = STRIPE_FAMILIES if a.family == "all" else (a.family,)
        tbl = apply_table_overrides(load_tables(a.table_dir), table_overrides)
        items = []
        for fam in fams:
            cfg = stripe_cfg(tbl, fam, **cfg_over)
            cfg.report()
            kw = cfg.as_kwargs("az_rot")
            kw.update(gen_over)
            # az_rot != 0 means this stripe tracks the mechanism azimuth
            if cfg.az_rot:
                kw["sr"] = (kw.get("sr") or 0) + a.az
                lbl = "%s (az_rot=%s, AZ=%.2f -> sr=%.2f)" % (
                    fam, cfg.az_rot, a.az, kw["sr"])
            else:
                lbl = "%s (fixed, sr=%s)" % (fam, kw.get("sr"))
            items.append((lbl, stripe_mask(**kw)))
            print()
    else:
        items = [(lbl, stripe_mask(**dict(kw, **overrides))) for lbl, kw in (
            ("nadir", dict(fr=900)),
            ("forward", dict(sy=180, fr=850, su=16, sd=16)),
            ("along-track", dict(sr=90)))]

    for lbl, m in items:
        n = int(m.sum())
        print("%-46s %8d populated (%.3f %% of FPA)"
              % (lbl, n, 100.0 * n / (FPA_NX * FPA_NY)))

    if a.plot:
        # No backend is forced: the figure opens interactively where a display
        # is available, and is saved to output/ either way.
        import matplotlib.pyplot as plt
        from libcam.mask.io import figure_path
        k = len(items)
        fig, axes = plt.subplots(1, k, figsize=(5.2 * k, 5.7), squeeze=False)
        for ax, (lbl, m) in zip(axes[0], items):
            ax.imshow(np.where(m.T, 0, 1), cmap="gray", vmin=0, vmax=1,
                      origin="lower", interpolation="nearest")
            ax.set_title(lbl, fontsize=10, loc="left")
            ax.set_xlabel("%d populated" % int(m.sum()), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out = figure_path(a.plot)
        fig.savefig(out, dpi=125)
        print("wrote %s" % out)
        plt.show()
    return items


if __name__ == "__main__":
    # ─── settings ────────────────────────────────────────────────────────────
    # Edit these to run from an editor. Each is also a command-line flag, and
    # the flag wins when given:  python libcam/mask/stripe.py --family raps --az 30
    FROM_TABLE = True             # False -> use the built-in demo configurations
    TABLE_DIR  = None             # None  -> docs/wfov_tables
    FAMILY     = "all"            # nadir | forward | cross | raps | all
    AZ         = 0.0              # mechanism azimuth [deg]; only affects az_rot stripes
    PLOT       = "stripe_mask.png"   # None -> no figure
    #
    # Generator keywords that are NOT WFOV table parameters. Anything the table
    # does define must go in TABLE_OVERRIDES below instead -- passing it here is
    # an error naming the table parameter to use. e.g.
    #     OVERRIDES = dict(clip_fov=True)
    OVERRIDES  = dict()
    #
    # The route for every WFOV table parameter. Keyed by the real table name,
    # which is unique -- so this is what targets ONE component. Copy the names
    # straight out of the report above the run, e.g.
    #     TABLE_OVERRIDES = dict(wfov_tbl_su_forward=15, wfov_tbl_su_cross=30)
    TABLE_OVERRIDES = dict()
    # ─────────────────────────────────────────────────────────────────────────
    _main(from_table=FROM_TABLE, table_dir=TABLE_DIR, family=FAMILY,
          az=AZ, plot=PLOT, table_overrides=TABLE_OVERRIDES, **OVERRIDES)
