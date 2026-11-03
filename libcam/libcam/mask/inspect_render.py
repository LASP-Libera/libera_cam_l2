"""Shared rendering for the mask-retrieval inspectors.

``apps/l1a/inspect_l1a.py`` and ``apps/l1b/inspect_l1b_mask.py`` produce the same
pictures from the same :class:`~libcam.mask.retrieve_mask.FrameResult` objects --
only the reader and the extra diagnostic panels differ. Everything that draws
lives here so the two cannot drift apart, which is exactly what happened between
``inspect_l1b.py`` and ``inspect_l1b_cam.py``.

The colour scheme is the contract: one colour per mask component, magenta for
anything unexplained, and shared pixels drawn as the lightened mean of their
members. See ``_blend``.
"""

import os

import numpy as np

from libcam.mask.retrieve_mask import (LABELS, LABEL_NAMES, UNEXPLAINED,
                                       MASK_BITS, BIT_UNEXPLAINED)
from libcam.mask.combine_masks import COMPONENTS

__all__ = ["COLOURS", "write_gif", "az_panel", "diagnostic_axes", "finish_figure",
           "pool_membership", "blend", "font"]

COLOURS = {
    0:                 (14, 16, 20),      # not selected
    LABELS["nadir"]:   (56, 134, 230),    # blue
    LABELS["forward"]: (138, 92, 220),    # violet
    LABELS["cross"]:   (32, 176, 128),    # green
    LABELS["raps"]:    (232, 140, 44),    # orange
    LABELS["limb"]:    (64, 200, 214),    # cyan
    LABELS["adm"]:     (226, 220, 92),    # pale yellow
    UNEXPLAINED:       (228, 44, 98),     # magenta -- deliberately loud
}


def pool_membership(nb, k=4):
    """Downsample a membership raster by OR-ing the bits in each block.

    A plain stride would drop most of the mask -- the stripes are 41 px wide but
    the ADM patches are isolated, and at k=4 a nearest-neighbour pick loses
    them. OR-ing also preserves overlaps: a block containing both a stripe and
    an ADM pixel comes out carrying both bits, which is the whole point.
    """
    h, w = nb.shape
    blk = nb[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k)
    return blk.max(axis=(1, 3)) | np.bitwise_or.reduce(
        np.bitwise_or.reduce(blk, axis=3), axis=1)


def blend(bits):
    """Colour for one membership value: the mean of its members, lightened.

    Overlaps are real -- a pixel under the cross-track stripe that also sits in
    an ADM patch belongs to both -- so they get a blended colour rather than
    whichever component happened to be drawn last. The lightening is what keeps
    a blend from being mistaken for some third pure component: anything paler
    than the legend swatches is a shared pixel.
    """
    cols = [COLOURS[LABELS[c]] for c in COMPONENTS
            if int(bits) & int(MASK_BITS[c])]
    if int(bits) & int(BIT_UNEXPLAINED):
        cols.append(COLOURS[UNEXPLAINED])
    if not cols:
        return COLOURS[0]
    rgb = np.mean(cols, axis=0)
    if len(cols) > 1:
        rgb = rgb + (255 - rgb) * 0.34
    return tuple(int(round(v)) for v in rgb)


def font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _az_line(draw, az, centre, sr, nx, ny, k, colour, dash=None):
    """Draw the generator line of a stripe at mechanism azimuth ``az``.

    ``k`` maps FPA pixels to image pixels (scale / pool). The line is the stripe
    centre line, not the band, so it can be laid over the rendered mask without
    hiding it.
    """
    th = np.deg2rad((sr or 0.0) + az)
    ct, st = np.cos(th), np.sin(th)
    t = np.array([-1.0, 1.0]) * max(nx, ny)
    # generator runs along x in the unrotated frame; (row, col) -> (x=col, y=row)
    xs = (st * t + centre[1]) * k
    ys = (ct * t + centre[0]) * k
    if dash:
        n = 60
        for j in range(0, n, 2):
            a, b = j / float(n), (j + 1) / float(n)
            draw.line([(xs[0] + (xs[1] - xs[0]) * a, ys[0] + (ys[1] - ys[0]) * a),
                       (xs[0] + (xs[1] - xs[0]) * b, ys[0] + (ys[1] - ys[0]) * b)],
                      fill=colour, width=2)
    else:
        draw.line([(xs[0], ys[0]), (xs[1], ys[1])], fill=colour, width=2)


def write_gif(results, path, pool=4, fps=4, tag_cells=True, min_tag_px=120,
              az_overlay=None):
    """Animated GIF of the labelled frames, ADM samples tagged by angular bin.

    Each ADM blob carries its ``cell_tag`` -- a ring letter and an azimuth
    number, ``A1`` through ``I18``. The tag is derived from the bin indices, not
    handed out in order of appearance, so the same tag means the same angular
    bin on every frame and in every granule: ``C7`` does not become a different
    sample when the pattern precesses or when a blob drops off the FPA.

    ``az_overlay`` is ``(centre_row, centre_col, sr)`` for the rotating stripe.
    When given, two lines are drawn on every frame: the stripe angle the granule
    *reports* in its ``Azimuth`` field, and the angle *measured* from the stripe
    actually in the data. On the granules we hold these do not agree -- the field
    is identically zero while the stripe sweeps -- and the point of drawing both
    is that the reader can judge that for themselves rather than take it from a
    number in a report.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  (PIL not available -- skipping the GIF)")
        return None
    scale = 2
    f_hdr, f_tag = font(13), font(9)
    imgs = []
    for r in results:
        if r.membership is None:
            continue
        small = pool_membership(r.membership, pool)
        rgb = np.zeros(small.shape + (3,), dtype=np.uint8)
        for v in np.unique(small):
            if v:
                rgb[small == v] = blend(v)
        im = Image.fromarray(rgb).resize(
            (small.shape[1] * scale, small.shape[0] * scale), Image.NEAREST)
        d = ImageDraw.Draw(im)

        if tag_cells and r.adm_cells is not None:
            k = scale / float(pool)
            for _iv, _ia, cell in r.cells():
                if cell.n < min_tag_px or cell.centre_obs is None:
                    continue
                x = cell.centre_obs[1] * k
                y = cell.centre_obs[0] * k
                d.text((x, y), cell.tag, fill=(24, 24, 24), font=f_tag,
                       anchor="mm")

        if az_overlay is not None:
            cr, cc, sr = az_overlay
            k = scale / float(pool)
            hdr = r.az_header if r.az_header is not None else 0.0
            _az_line(d, hdr, (cr, cc), sr, 2048, 2048, k, (255, 90, 90),
                     dash=True)
            if r.az_fitted is not None:
                _az_line(d, r.az_fitted, (cr, cc), sr, 2048, 2048, k,
                         (90, 220, 255))
            # Keyed in the corner, not at the centre: the two lines cross
            # exactly where the mask is busiest, and labels there sat on top of
            # the ADM tags.
            w = im.size[0]
            d.text((w - 8, 6), "AZ reported  %6.1f deg" % hdr,
                   fill=(255, 90, 90), font=f_hdr, anchor="ra")
            d.text((w - 8, 22),
                   "AZ measured %6.1f deg" % r.az_fitted
                   if r.az_fitted is not None else
                   "AZ measured      n/a (stripes parallel)",
                   fill=(90, 220, 255), font=f_hdr, anchor="ra")
            if r.az_fitted is not None:
                dd = (r.az_fitted - hdr + 90.0) % 180.0 - 90.0
                d.text((w - 8, 38), "difference  %+6.1f deg" % dd,
                       fill=(255, 220, 120), font=f_hdr, anchor="ra")

        d.text((8, 6), "frame %03d   phase %s   %s"
               % (r.index, r.phase, r.status.code), fill=(255, 255, 255),
               font=f_hdr)
        d.text((8, 22), "coverage %.3f   unexplained %d   ADM cells %d   "
               "shared px %d"
               % (r.status.coverage, r.unexplained,
                  sum(1 for _ in r.cells()), r.overlap),
               fill=(255, 180, 200) if r.unexplained > 0.05 * max(r.observed, 1)
               else (190, 200, 210), font=f_hdr)

        # Ring letters are bin indices, so they do NOT run in radial order: the
        # VZA bin centres are taken mod 90, and once the precession carries the
        # low bins past 90 deg they reappear as the OUTERMOST rings. At frame 0
        # of the 2026-07-12 granule the outward sequence is C D E F G H I A B.
        # Spell it out per frame so the ordering is never a puzzle.
        rings = {}
        for _iv, _ia, cell in r.cells():
            rings.setdefault(cell.i_vza, (cell.tag[0], cell.radius_pred))
        if rings:
            seq = " ".join(rings[k][0] for k in
                           sorted(rings, key=lambda k: rings[k][1]))
            d.text((8, 38), "rings outward: %s" % seq, fill=(150, 158, 166),
                   font=f_hdr)
        y = im.size[1] - 16 * (len(COLOURS) - 1) - 24
        for code, col in COLOURS.items():
            if code == 0:
                continue
            d.rectangle([8, y, 20, y + 11], fill=col)
            d.text((26, y - 1), LABEL_NAMES[code], fill=(210, 215, 220),
                   font=f_hdr)
            y += 16
        d.text((8, y + 2), "ADM tags: ring letter + azimuth bin   |   paler = "
               "shared by two components", fill=(150, 158, 166), font=f_hdr)
        imgs.append(im)
    if not imgs:
        return None
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    return path


# ─── diagnostics: the four panels both inspectors share ──────────────────────

INK, MUT, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e3"
NOTE_BOX = dict(fc="white", ec="#e3e3e3", lw=0.8, pad=3.5)


def core_panels(axs, results, retriever):
    """Draw panels A-D into a 2x2 array of axes.

    A  mask precession phase, with discontinuities marked
    B  recall and coverage against their floors
    C  observed pixels per component, overlaps counted in each
    D  ADM ring residuals

    These are the panels that say whether the retrieval worked, so they are
    identical for L1A and L1B. Instrument-specific panels go in extra rows.
    """
    from libcam.mask.retrieve_mask import _predicted_radii, measure_blob_centroids
    from libcam.mask.wfov_csv2cfg import adm_cfg

    idx = np.array([r.index for r in results])
    rec = np.array([r.status.recall for r in results])
    cov = np.array([r.status.coverage for r in results])
    une = np.array([r.unexplained for r in results], dtype=float)
    obs = np.array([r.observed for r in results], dtype=float)
    ph = np.array([-1 if r.phase is None else r.phase for r in results])

    # Panel A used to plot the mask counter. That was the right thing to show
    # while the phase was the retrieved quantity, but it no longer is: the
    # shifts themselves are measured on every frame and the counter is just the
    # frame index. What matters now is whether the two shifts walk at a constant
    # rate -- a clean saw-tooth means the precession is being tracked, and a
    # break in it is the signature of a frame where the search jumped a bin.
    ax = axs[0, 0]
    sh = [r.measured_shifts for r in results]
    if any(s is not None for s in sh):
        vz = np.array([s[0] if s else np.nan for s in sh], dtype=float)
        va = np.array([s[1] if s else np.nan for s in sh], dtype=float)
        ax.plot(idx, vz, "o-", ms=3.5, lw=1.2, color="#2a78d6",
                label="VZA shift")
        ax.plot(idx, va, "s-", ms=3.5, lw=1.2, color="#8e44ad",
                label="VAA shift")
        ax.set_ylabel("retrieved shift [deg]")
        ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        sol = retriever.solution
        if sol is not None and sol.step_vza is not None:
            ax.text(0.02, 0.06, "precession %+.2f deg/mask VZA, "
                                "%+.2f deg/mask VAA\n(measured, not from the table)"
                    % (sol.step_vza, sol.step_vaa), transform=ax.transAxes,
                    fontsize=9.5, color=MUT, va="bottom")
    else:
        ax.plot(idx, ph, "o-", ms=3.5, lw=1.2, color="#2a78d6")
        ax.set_ylabel("retrieved mask counter")
    ax.set_xlabel("frame")
    ax.set_title("A.  Retrieved precession", loc="left", fontsize=12, color=INK)

    ax = axs[0, 1]
    ax.plot(idx, rec, "o-", ms=3.5, lw=1.2, color="#1baf7a", label="recall")
    ax.plot(idx, cov, "s-", ms=3.5, lw=1.2, color="#e8590c", label="coverage")
    ax.axhline(retriever.good_recall, color=MUT, ls="--", lw=1.0)
    ax.axhline(retriever.min_coverage, color="#e8590c", ls=":", lw=1.0)
    ax.set_ylim(0, 1.05); ax.set_xlabel("frame")
    ax.set_title("B.  Retrieval quality", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.text(0.02, 0.08, "recall: predicted px that are selected\n"
                        "coverage: selected px that are explained",
            transform=ax.transAxes, fontsize=9.5, color=MUT, va="bottom")

    ax = axs[1, 0]
    for name in COMPONENTS:
        v = np.array([r.observed_parts.get(name, 0) for r in results], dtype=float)
        if v.max() > 0:
            ax.plot(idx, v / 1000.0, lw=1.4, label=name)
    ax.plot(idx, une / 1000.0, lw=1.6, color="#e42c62", ls="--",
            label="unexplained")
    ax.set_xlabel("frame"); ax.set_ylabel("thousand pixels")
    ax.set_title("C.  Observed pixels by component (overlaps counted in each)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9.5, ncol=2)

    ax = axs[1, 1]
    ok = [r for r in results if r.status.ok and r.membership is not None]
    if ok and retriever.solution:
        cfg = adm_cfg(retriever._tbl())
        xc, yc = cfg.x_center_pix, cfg.y_center_pix
        pr, pm = [], []
        for r in ok[:: max(1, len(ok) // 30)]:
            rows, cols = measure_blob_centroids(
                (r.membership & MASK_BITS["adm"]).astype(bool))
            if not len(rows):
                continue
            meas = np.hypot(rows - xc, cols - yc)
            vz = r.vza_shift
            pred = _predicted_radii(vz, retriever.solution.coef, cfg.percent_samp,
                                    retriever.solution.asin_mode)
            pred = pred[pred > 1]
            for m in meas:
                j = int(np.argmin(np.abs(pred - m)))
                pr.append(pred[j]); pm.append(m)
        if pr:
            pr, pm = np.array(pr), np.array(pm)
            ax.plot(pr, pm - pr, ".", ms=4, color="#2a78d6", alpha=0.6)
            ax.axhline(0, color=MUT, lw=1.0)
            ax.set_xlabel("predicted ring radius [px]")
            ax.set_ylabel("measured - predicted [px]")
            ax.set_title("D.  ADM ring residuals (%d blobs, rms %.2f px)"
                         % (len(pr), float(np.sqrt(np.mean((pm - pr) ** 2)))),
                         loc="left", fontsize=12, color=INK)
    if not ax.has_data():
        ax.text(0.5, 0.5, "no ADM rings measured", ha="center", va="center",
                transform=ax.transAxes, color=MUT)
        ax.set_title("D.  ADM ring residuals", loc="left", fontsize=12, color=INK)


def az_panel(ax, results, retriever):
    """Mechanism azimuth: what the granule reports against what the stripe shows.

    This panel exists because the two disagree. Every granule we hold carries an
    ``Azimuth`` field that is identically zero, including the RAPS granule whose
    rotating stripe sweeps through 45 deg over the frames shown here. Plotting
    them together is the fastest way to see whether a new data version has fixed
    it: if the field is populated the two traces should lie on top of each other,
    and any offset between them is a pointing error worth chasing.
    """
    def fold(a):
        # A stripe with no offset from the centre is unchanged by a half turn,
        # so the angle only means anything modulo 180. Folding to (-90, 90]
        # turns the sweep into the smooth ramp it physically is instead of a
        # 180 deg jump between the first and second frame.
        #
        # NaN-safe: in CT there is no rotating stripe, so az_fitted is None on
        # every frame and the array is all-NaN. numpy's remainder warns on NaN,
        # which surfaced as a RuntimeWarning on every cross-track run.
        a = np.asarray(a, dtype=float)
        out = np.full(a.shape, np.nan)
        ok = np.isfinite(a)
        if ok.any():
            out[ok] = (a[ok] + 90.0) % 180.0 - 90.0
        return out

    idx = [r.index for r in results]
    hdr = fold([r.az_header if r.az_header is not None else np.nan
                for r in results])
    fit = fold([r.az_fitted if r.az_fitted is not None else np.nan
                for r in results])
    ax.plot(idx, hdr, "o-", ms=4, lw=1.6, color="#c0392b",
            label="reported (granule Azimuth)")
    if np.isfinite(fit).any():
        ax.plot(idx, fit, "s-", ms=4, lw=1.6, color="#1f77b4",
                label="measured from the rotating stripe")
    ax.set_xlabel("frame", color=MUT, fontsize=10)
    ax.set_ylabel("stripe angle, folded to ±90 [deg]", color=MUT, fontsize=10)
    ax.set_title("E.  mechanism azimuth, reported vs retrieved",
                 loc="left", fontsize=12, color=INK)
    if not np.isfinite(fit).any():
        ax.text(0.5, 0.42, "no rotating stripe in this mode --\n"
                           "nothing to measure the azimuth against",
                ha="center", va="center", transform=ax.transAxes, color=MUT)
    else:
        # mod 180: a centred stripe is unchanged by a half turn, so 178.8 and
        # -1.2 are the same picture. fit and hdr are already folded.
        d = np.nanmax(np.abs(np.asarray(fit, dtype=float)
                             - np.asarray(hdr, dtype=float)))
        ax.text(0.02, 0.06, "they differ by up to %.0f deg" % d,
                transform=ax.transAxes, color=INK, fontsize=10, bbox=NOTE_BOX)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def style_axes(axs):
    for a in np.asarray(axs).ravel():
        a.grid(True, color="#f0f0f0", lw=0.8)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=MUT, labelsize=10)


def finish_figure(fig, axs, retriever, path, level="L1A", nrow=2):
    """Common styling, title and save."""
    style_axes(axs)
    sol = retriever.solution
    fig.suptitle("%s pixel-mask retrieval  \u2014  %s, polynomial %s"
                 % (level, sol.mode if sol else "?",
                    ("x %.5f" % sol.coef_scale) if sol else "?"),
                 x=0.012, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.035 * 2 / max(nrow, 1)])
    fig.savefig(path, dpi=140, facecolor="white")
    return path
