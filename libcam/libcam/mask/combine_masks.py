"""
libcam/mask/combine_masks.py
----------------------------
Compose one frame's complete WFOV pixel mask from the enabled components.

Every parameter comes from the master CSVs in ``docs/wfov_tables/`` via
:mod:`libcam.mask.wfov_csv2cfg`; nothing mask-shaped is hard-coded here. Which
components are switched on is a user choice — the six booleans just below.

    from libcam.mask.combine_masks import combine_masks, MODES
    mask, counts = combine_masks(tbl, MODES["RAPS"], az=7.12, frame=0)

Two frame-dependent inputs, deliberately kept separate because they are
different mechanisms:

``az``      the mechanism azimuth, in degrees. Added to ``sr`` for stripes whose
            table ``az_rot`` flag is NORMAL(1) or INVERTED(2) — added for NORMAL,
        subtracted for INVERTED (0/NONE leaves ``sr`` unchanged). In practice
        the raps stripe only.
            It is viewing geometry, not a table parameter, so it is always
            supplied by the caller.
``frame``   the frame index, which drives the **ADM precession**: the angular bins
            step by ``vza_shift`` / ``vaa_shift`` degrees per mask, so frame *n*
            uses ``init + n * shift``. Nothing to do with where the mechanism is
            pointing.

One call renders one frame. Sweeping frames is ``apps/mask/mask_series.py``.

Counting
--------
The per-component counts sum to **more** than the mask total wherever components
overlap — nadir and cross cross at the centre, and the ADM discs sit on top of
both. ``combine_masks`` therefore reports the sum of the parts, the union total,
and the difference, rather than leaving you to reconcile them.
"""

import os
import sys

import numpy as np

if __package__ in (None, ""):
    # Unlike the single-family modules, this one imports its siblings at module
    # level -- so the repo root has to be on sys.path *before* those imports,
    # not inside _main(). Walk up four levels when run as a script (one more
    # than before the 2026-11-03 uv_build re-nesting -- see docs/PATCHES.md):
    #     combine_masks.py  ->  libcam/libcam/mask/  ->  libcam/libcam/  ->  libcam/  ->  <repo root>
    _root = os.path.abspath(__file__)
    for _ in range(4):
        _root = os.path.dirname(_root)
    sys.path.insert(0, _root)

from libcam.mask.adm import adm_mask
from libcam.mask.stripe import stripe_mask
from libcam.mask.limb import limb_mask
from libcam.mask.wfov_csv2cfg import (load_tables, apply_table_overrides,
                                      stripe_cfg, adm_cfg, limb_cfg,
                                      StripeCfg, AdmCfg, LimbCfg,
                                      split_overrides, field_table_names,
                                      STRIPE_FAMILIES)

__all__ = ["combine_masks", "COMPONENTS", "MODES", "NADIR", "FORWARD", "CROSS",
           "RAPS", "LIMB", "ADM", "enabled_from_switches", "plot_combined",
           "mask_rgb", "summarise"]

FPA_NX = 2048
FPA_NY = 2048

# ─── component switches ──────────────────────────────────────────────────────
# Which mask families contribute. Edit these; they are intentionally NOT
# command-line arguments. `--mode CT` / `--mode RAPS` overrides them wholesale
# (see MODES below).
NADIR   = True     # nadir stripe, fixed
FORWARD = False    # forward-offset stripe, fixed
CROSS   = False    # cross-track stripe, fixed (vertical, sr=90)
RAPS    = True     # the stripe that rotates with AZ (az_rot != 0)
LIMB    = False    # limb annulus
ADM     = True     # ADM lock-on samples, precessing frame to frame
# ─────────────────────────────────────────────────────────────────────────────

# Fixed order. Also the plot's colour order, so a component keeps its colour
# regardless of which others are enabled.
COMPONENTS = ("nadir", "forward", "cross", "raps", "limb", "adm")

# Operational modes. Note "RAPS" the mode versus "raps" the stripe that is
# activated during it.
MODES = {
    "CT":   dict(nadir=True, forward=True,  cross=True,  raps=False,
                 limb=False, adm=True),
    "RAPS": dict(nadir=True, forward=False, cross=False, raps=True,
                 limb=False, adm=True),
}


# Generator keywords each component family accepts that are NOT table parameters.
_EXTRA_OK = {"stripe": ("clip_fov", "xp", "yp"),
             "adm":    ("clip", "nx", "ny", "asin_mode"),
             "limb":   ("xp", "yp")}
_CFG_CLS = {"stripe": StripeCfg, "adm": AdmCfg, "limb": LimbCfg}


def _kind(name):
    return "stripe" if name in STRIPE_FAMILIES else name


def _accepted(kind):
    """Override names this component kind can take."""
    cls = _CFG_CLS[kind]
    free = {f for f in cls._fields if not field_table_names(f)}
    return free | set(_EXTRA_OK[kind])


def _route_overrides(overrides, enabled):
    """Validate the whole override set once, then split it per component.

    Overrides are supplied globally but only some apply to each component --
    ``coef_ang2dist`` is ADM-only, ``clip_fov`` is stripe-only. Validating
    per-component would reject a keyword simply because the *other* component was
    built first, so the check happens once against every enabled component and
    each then receives only the keys it accepts.
    """
    overrides = dict(overrides or {})
    kinds = {_kind(n) for n in COMPONENTS if enabled.get(n)}

    table_backed = [(k, field_table_names(k)) for k in overrides
                    if field_table_names(k)]
    if table_backed:
        lines = ["  %s -> set it in TABLE_OVERRIDES as one of:\n      %s"
                 % (k, "\n      ".join("%s=..." % m for m in mapped))
                 for k, mapped in sorted(table_backed)]
        raise TypeError(
            "these are WFOV table parameters and must go in TABLE_OVERRIDES,\n"
            "not in the plain overrides -- a bare field name cannot say which\n"
            "component is meant:\n%s" % "\n".join(lines))

    usable = set().union(*(_accepted(k) for k in kinds)) if kinds else set()
    unknown = [k for k in overrides if k not in usable]
    if unknown:
        detail = ", ".join("%s: %s" % (k, ", ".join(sorted(_accepted(k))))
                           for k in sorted(kinds))
        raise TypeError(
            "override(s) %s not accepted by any enabled component.\n"
            "  enabled kinds and what each takes -- %s\n"
            "  WFOV table parameters go in TABLE_OVERRIDES instead"
            % (", ".join(sorted(unknown)), detail))

    return {k: {kk: vv for kk, vv in overrides.items() if kk in _accepted(k)}
            for k in kinds}


def enabled_from_switches():
    """The six module-level switches as a dict."""
    return dict(nadir=NADIR, forward=FORWARD, cross=CROSS,
                raps=RAPS, limb=LIMB, adm=ADM)


def combine_masks(tbl, enabled, az=0.0, frame=0, overrides=None,
                  nx=FPA_NX, ny=FPA_NY, report=False):
    """Build the combined mask for one frame.

    Parameters
    ----------
    tbl : dict
        Loaded WFOV tables (``load_tables()``), optionally already passed through
        ``apply_table_overrides``.
    enabled : dict
        ``{component: bool}`` over :data:`COMPONENTS`. Missing keys are False.
    az : float
        Mechanism azimuth [deg], added to ``sr`` where ``az_rot`` is set.
    frame : int
        Frame index, driving the ADM precession.
    overrides : dict, optional
        Generator keywords with no table backing (``coef_ang2dist``, ``clip``,
        ``clip_fov``, ``xp``, ``yp``). Table parameters belong in ``tbl`` via
        ``apply_table_overrides``; passing one here raises TypeError naming the
        table parameter to use instead.
    report : bool
        Print each component's config provenance as it is built.

    Returns
    -------
    mask : (nx, ny) uint8
        The union, indexed ``mask[x, y]`` like every generator here.
    counts : dict
        ``parts``    {component: populated pixels}, in COMPONENTS order
        ``layers``   {component: the component's own raster}
        ``sum_parts``  sum of the per-component counts
        ``total``      populated pixels in the union
        ``overlap``    ``sum_parts - total``
        ``az``, ``frame``, and the applied ADM shifts, for labelling
    """
    routed = _route_overrides(overrides, enabled)
    layers, parts = {}, {}
    adm_shift = (0.0, 0.0)

    for name in COMPONENTS:
        if not enabled.get(name):
            continue

        if name in STRIPE_FAMILIES:
            cfg_over, gen_over = split_overrides(
                StripeCfg, routed["stripe"], extra_ok=("clip_fov", "xp", "yp"))
            cfg = stripe_cfg(tbl, name, **cfg_over)
            if report:
                cfg.report()
            kw = cfg.as_kwargs("az_rot")
            kw.update(gen_over)
            # az_rot: 0=NONE (fixed), 1=NORMAL (+= az), 2=INVERTED (-= az).
            # Fixed 2026-11-03: previously treated az_rot as a bool and always
            # added az, which gave RAPS (az_rot=2, INVERTED) the wrong sign.
            if cfg.az_rot == 1:
                kw["sr"] = (kw.get("sr") or 0) + az
            elif cfg.az_rot == 2:
                kw["sr"] = (kw.get("sr") or 0) - az
            layer = stripe_mask(nx if "xp" not in kw else kw.pop("xp"),
                                ny if "yp" not in kw else kw.pop("yp"), **kw)

        elif name == "adm":
            cfg_over, gen_over = split_overrides(
                AdmCfg, routed["adm"], extra_ok=_EXTRA_OK["adm"])
            cfg = adm_cfg(tbl, **cfg_over)
            if report:
                cfg.report()
            vza_s, vaa_s = cfg.shift_at(frame)
            adm_shift = (vza_s, vaa_s)
            kw = cfg.as_kwargs("vaa_shift", "vza_shift", "vaa_shift_init",
                               "vza_shift_init", "shift_num")
            kw["vza_shift"], kw["vaa_shift"] = vza_s, vaa_s
            kw.update(gen_over)
            kw.setdefault("nx", nx)
            kw.setdefault("ny", ny)
            layer = adm_mask(**kw)

        elif name == "limb":
            cfg_over, gen_over = split_overrides(
                LimbCfg, routed["limb"], extra_ok=("xp", "yp"))
            cfg = limb_cfg(tbl, **cfg_over)
            if report:
                cfg.report()
            kw = cfg.as_kwargs()
            kw.update(gen_over)
            layer = limb_mask(nx if "xp" not in kw else kw.pop("xp"),
                              ny if "yp" not in kw else kw.pop("yp"), **kw)
        else:
            raise ValueError("unknown component %r (known: %s)"
                             % (name, ", ".join(COMPONENTS)))

        layer = layer.astype(bool)
        layers[name] = layer
        parts[name] = int(layer.sum())

    mask = np.zeros((nx, ny), dtype=bool)
    for layer in layers.values():
        mask |= layer

    sum_parts = sum(parts.values())
    total = int(mask.sum())
    counts = dict(parts=parts, layers=layers, sum_parts=sum_parts, total=total,
                  overlap=sum_parts - total, az=az, frame=frame,
                  adm_vza_shift=adm_shift[0], adm_vaa_shift=adm_shift[1],
                  fill_pct=100.0 * total / (nx * ny))
    return mask.astype(np.uint8), counts


def summarise(counts, prefix="  "):
    """Print the per-component counts, the total, and the overlap."""
    print("%scomponent          pixels     %% of FPA" % prefix)
    print("%s%s" % (prefix, "-" * 40))
    npix = counts["total"] / (counts["fill_pct"] / 100.0) if counts["fill_pct"] else 1
    for name, n in counts["parts"].items():
        print("%s%-16s %9d %10.3f" % (prefix, name, n, 100.0 * n / npix))
    print("%s%s" % (prefix, "-" * 40))
    print("%s%-16s %9d" % (prefix, "sum of parts", counts["sum_parts"]))
    print("%s%-16s %9d %10.3f   <- union, overlap counted once"
          % (prefix, "TOTAL", counts["total"], counts["fill_pct"]))
    print("%s%-16s %9d            (%.1f %% of the sum)"
          % (prefix, "overlap", counts["overlap"],
             100.0 * counts["overlap"] / counts["sum_parts"]
             if counts["sum_parts"] else 0.0))


# ─── plotting ────────────────────────────────────────────────────────────────

# Fixed categorical colours, validated for colour-vision deficiency. Index by
# COMPONENTS order so a component's colour never depends on what else is on.
_COLOURS = {
    "nadir":   "#2a78d6",
    "forward": "#eb6834",
    "cross":   "#1baf7a",
    "raps":    "#eda100",
    "limb":    "#e87ba4",
    "adm":     "#008300",
}
_OVERLAP_COLOUR = "#0b0b0b"
_SURFACE = "#fcfcfb"


def _pool(m, k=2):
    """Max-pool so single-pixel-wide structures survive downsampling."""
    n = m.shape[0] // k
    return m[:n * k, :n * k].reshape(n, k, n, k).max(axis=(1, 3))


def _hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])


def mask_rgb(counts, pool=2):
    """Composite the per-component layers into one RGB image.

    Components are painted in :data:`COMPONENTS` order so a component keeps its
    colour regardless of which others are on, then pixels belonging to two or more
    are painted over in the overlap colour -- otherwise the last-drawn component
    would hide the overlap the counts are telling you about.

    Shared by ``plot_combined`` and ``mask_series.write_gif``.
    """
    layers = counts["layers"]
    if not layers:
        raise SystemExit("no components enabled - nothing to draw")

    stack = np.zeros(next(iter(layers.values())).shape, dtype=np.uint8)
    for layer in layers.values():
        stack += layer.astype(np.uint8)
    multi = _pool((stack > 1).T, pool)

    shown = [n for n in COMPONENTS if n in layers]
    rgb = np.ones(_pool(layers[shown[0]].T, pool).shape + (3,), dtype=float)
    for name in shown:
        rgb[_pool(layers[name].T, pool)] = _hex2rgb(_COLOURS[name])
    rgb[multi] = _hex2rgb(_OVERLAP_COLOUR)
    return rgb, shown


def plot_combined(counts, title=None, path=None, pool=2):
    """One figure: components in fixed colours, overlap picked out, counts in the legend."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from libcam.mask.io import figure_path

    rgb, shown = mask_rgb(counts, pool)
    hex2rgb = _hex2rgb

    fig, ax = plt.subplots(figsize=(8.4, 8.8), facecolor=_SURFACE)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(_SURFACE)
    for sp in ax.spines.values():
        sp.set_color("#d8d7d2")

    handles = [Patch(facecolor=_COLOURS[n], label="%-8s %8d" % (n, counts["parts"][n]))
               for n in shown]
    if counts["overlap"]:
        handles.append(Patch(facecolor=_OVERLAP_COLOUR,
                             label="%-8s %8d" % ("overlap", counts["overlap"])))
    ax.legend(handles=handles, loc="upper right", frameon=True,
              facecolor=_SURFACE, edgecolor="#d8d7d2", fontsize=9,
              prop={"family": "monospace"})

    ax.set_title(title or "combined pixel mask", fontsize=12,
                 fontweight="bold", loc="left", color="#0b0b0b")
    ax.set_xlabel(
        "AZ = %.2f°   frame %d (ADM shift vza %.2f°, vaa %.2f°)\n"
        "total %d px = %.3f %% of FPA   ·   sum of parts %d   ·   overlap %d"
        % (counts["az"], counts["frame"], counts["adm_vza_shift"],
           counts["adm_vaa_shift"], counts["total"], counts["fill_pct"],
           counts["sum_parts"], counts["overlap"]),
        fontsize=9.5, color="#52514e")
    fig.tight_layout()

    if path:
        out = figure_path(path)
        fig.savefig(out, dpi=130, facecolor=_SURFACE)
        print("wrote %s" % out)
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def _main(from_table=True, table_dir=None, mode=None, az=0.0, frame=0,
          plot=None, report=True, table_overrides=None, **overrides):
    """Tester: compose one frame's mask and plot it.

    ``mode`` (``"CT"`` or ``"RAPS"``) replaces the module-level switches
    wholesale. Left as None, the switches at the top of this file decide.
    """
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    p.add_argument("--table_dir", default=table_dir)
    p.add_argument("--mode", default=mode, choices=sorted(MODES),
                   help="CT or RAPS; replaces the switches at the top of the file")
    p.add_argument("--az", type=float, default=az,
                   help="mechanism azimuth [deg]; affects az_rot stripes only")
    p.add_argument("--frame", type=int, default=frame,
                   help="frame index; drives the ADM precession")
    p.add_argument("--plot", nargs="?", const="combined_mask.png", default=plot,
                   help="write a PNG of the combined mask")
    p.add_argument("--quiet", action="store_true", help="skip the config reports")
    a = p.parse_args()

    tbl = apply_table_overrides(load_tables(a.table_dir), table_overrides)

    if a.mode:
        enabled = dict(MODES[a.mode])
        source = "--mode %s" % a.mode
    else:
        enabled = enabled_from_switches()
        source = "switches at the top of combine_masks.py"
    on = [n for n in COMPONENTS if enabled.get(n)]
    print("components: %s   (%s)\n" % (", ".join(on) or "NONE", source))

    mask, counts = combine_masks(tbl, enabled, az=a.az, frame=a.frame,
                                 overrides=overrides,
                                 report=report and not a.quiet)
    print()
    summarise(counts)

    if a.plot:
        title = "combined mask - %s" % (("mode %s" % a.mode) if a.mode
                                        else ", ".join(on))
        plot_combined(counts, title=title, path=a.plot)
        import matplotlib.pyplot as plt
        plt.show()
    return mask, counts


if __name__ == "__main__":
    # ─── settings ────────────────────────────────────────────────────────────
    # Component switches are at the TOP of this file. Everything else here.
    # Each is also a command-line flag, and the flag wins when given:
    #     python libcam/mask/combine_masks.py --mode CT --az 30 --plot
    TABLE_DIR = None                  # None -> docs/wfov_tables
    MODE      = "CT"                  # "CT" | "RAPS" | None
                                      #   None means: use the NADIR / FORWARD / CROSS /
                                      #   RAPS / LIMB / ADM switches at the top of this
                                      #   file, in the "component switches" block.
    AZ        = 45.0                  # mechanism azimuth [deg]. Ignored in CT -- no CT
                                      #   stripe has az_rot set. For RAPS keep it away
                                      #   from 0: at AZ=0 the raps stripe lands exactly
                                      #   on nadir, so the two almost entirely overlap
                                      #   and the total collapses (267k vs 336k px).
    FRAME     = 0                     # frame index; drives the ADM precession
    PLOT      = "combined_mask.png"   # None -> no figure
    REPORT    = True                  # print each component's config provenance
    #
    # Generator keywords that are NOT WFOV table parameters, e.g.
    #     OVERRIDES = dict(coef_ang2dist=[...], clip_fov=True)
    OVERRIDES = dict()
    #
    # The route for every WFOV table parameter, keyed by the real table name, e.g.
    #     TABLE_OVERRIDES = dict(wfov_tbl_su_forward=15, wfov_tbl_maxPixR=12)
    TABLE_OVERRIDES = dict()
    # ─────────────────────────────────────────────────────────────────────────
    _main(table_dir=TABLE_DIR, mode=MODE, az=AZ, frame=FRAME, plot=PLOT,
          report=REPORT, table_overrides=TABLE_OVERRIDES, **OVERRIDES)
