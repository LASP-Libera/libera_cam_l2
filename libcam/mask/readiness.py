"""Can this granule be used for L2? A checklist, run before anything else.

Every L1A and L1B inspection starts here. The retrieval that follows is the
expensive part and it is worth nothing if the granule is missing the fields L2
needs, or if the mask parameters that flew are not the ones in the table. This
module answers that in one screen, in a fixed order, so the same question gets
the same answer from one data version to the next.

Each check returns one of four verdicts:

    PASS   the thing is there and usable
    WARN   present but not what we expected -- usable with care, say why
    FAIL   absent or unusable; L2 cannot proceed on this item
    INFO   not pass/fail, just something that has to be on the record

The point of separating WARN from FAIL is that most of what is wrong with these
granules is WARN: the geolocation is right but only on 11 frames of 248, the
azimuth field exists but is zero, the mask parameters are retrievable but are not
the ones the table says. A checklist that collapsed those into "bad" would be
ignored within a week.

Two of the checks are retrieval results rather than file contents -- the mask
parameters and the centres -- so they are filled in after the retrieval has run.
Call :func:`file_checks` before, :func:`retrieval_checks` after, and
:func:`render` on the union.

Provenance: written 2026-10-05 after the 2025-11-18 and 2026-03-27 granules each
failed for a different reason that a checklist would have surfaced in seconds.
"""

import numpy as np

__all__ = ["Check", "PASS", "WARN", "FAIL", "INFO",
           "file_checks", "retrieval_checks", "render", "verdict"]

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
_ORDER = {FAIL: 0, WARN: 1, INFO: 2, PASS: 3}


class Check(object):
    """One line of the checklist.

    ``scope`` narrows what a FAIL means. Without it, a FAIL says the granule is
    not usable: something required is absent. With it, the granule IS usable --
    everything outside the named scope is fine -- and the failure is a hard
    restriction on that scope alone.

    The distinction was added 2026-10-06 after the FSW geometry check landed.
    That check fails because the commanded angles are WRONG, not because
    anything is missing, and the summary printed

        VERDICT: NOT READY for L2 -- at least one required item is absent.

    directly above "12 of 12 frames usable for L2". Both lines were true and
    together they were nonsense. A verdict nobody can act on is worse than no
    verdict: it trains people to scroll past the banner.
    """

    __slots__ = ("key", "status", "summary", "detail", "scope")

    def __init__(self, key, status, summary, detail="", scope=None):
        self.key = key
        self.status = status
        self.summary = summary
        self.detail = detail
        self.scope = scope

    def __repr__(self):
        return "Check(%r, %r, %r)" % (self.key, self.status, self.summary)


def verdict(checks):
    """Overall verdict: FAIL if anything failed, else WARN if anything warned."""
    worst = min((_ORDER[c.status] for c in checks), default=_ORDER[PASS])
    for k, v in _ORDER.items():
        if v == worst:
            return k
    return PASS


# ─── the checks ──────────────────────────────────────────────────────────────

ANGLE_FIELDS = {
    "SZA": "Solar_Zenith_Surface",
    "SAA": "Solar_Azimuth_Surface_WRT_North",
    "VZA": "Viewing_Zenith_Surface",
    "VAA": "Viewing_Azimuth_Surface_WRT_North",
}

# obs_id -> human name. Deliberately not a fallback: an id that is not here means
# we do not know which components to expect, and that is a FAIL, not a guess.
MODE_NAMES = {128: "CT (cross-track)", 132: "RAPS", 137: "ET (Earth-Target)"}


def _finite_fraction(da, frames=None, dim="CAMERA_TIME"):
    """Fraction of finite values, read from a SAMPLE of frames.

    Never ``da.values`` on a whole granule: a per-pixel angle field is one float
    per FPA pixel per frame, which is 1.1 GB for a 68-frame granule and 4 GB for
    all four angles together. Reading them whole is what ran the container out
    of memory, and it answers a question a handful of frames answers just as
    well.
    """
    try:
        if frames is not None and dim in getattr(da, "dims", ()):
            take = list(frames)
            if len(take) > 4:
                take = [take[0], take[len(take) // 3],
                        take[2 * len(take) // 3], take[-1]]
            da = da.isel({dim: take})
        a = np.asarray(da.values, dtype=float)
    except Exception:
        return float("nan")
    return float(np.isfinite(a).mean()) if a.size else 0.0


def file_checks(ds, mask_field=None, n_frames=None, level="L1B",
                obs_ids=None, az=None, fill_pct=None, frames=None,
                frame_dim="CAMERA_TIME", fill_ref=None):
    """Checks that read only the granule -- run before the retrieval.

    ``obs_ids`` is the per-frame operational mode, ``az`` the per-frame
    mechanism azimuth, ``fill_pct`` the per-frame selected-pixel percentage, and
    ``fill_ref`` a single representative fill percentage used to judge whether
    the per-pixel angle fields are complete (see the angle check).
    They are passed in rather than re-read because the caller has already
    decoded them and the field names differ between L1A and L1B.
    """
    out = []

    # ── frames and the mask itself ──────────────────────────────────────────
    if mask_field is None:
        out.append(Check("mask_field", FAIL,
                         "no pixel-mask field in the granule",
                         "L2 cannot know which pixels were selected."))
    else:
        out.append(Check("mask_field", PASS,
                         "pixel mask present as %r" % mask_field,
                         "%s frames" % (n_frames if n_frames is not None else "?")))

    # ── mode census ─────────────────────────────────────────────────────────
    if obs_ids is None or not len(obs_ids):
        out.append(Check("modes", FAIL, "no operational-mode field",
                         "Without it the component set is unknown and the mask "
                         "cannot be reproduced."))
    else:
        ids = np.asarray(obs_ids)
        ids = ids[np.isfinite(ids.astype(float))] if ids.dtype.kind == "f" else ids
        uniq, cnt = np.unique(ids, return_counts=True)
        parts = ["%s x%d" % (MODE_NAMES.get(int(u), "UNKNOWN id %d" % int(u)), c)
                 for u, c in zip(uniq, cnt)]
        unknown = [int(u) for u in uniq if int(u) not in MODE_NAMES]
        out.append(Check(
            "modes", WARN if unknown else PASS,
            "modes in this granule: " + ", ".join(parts),
            ("No component set is defined for %s, so those frames cannot be "
             "retrieved and must be excluded from L2."
             % ", ".join(str(u) for u in unknown)) if unknown else
            "Every mode present has a defined component set."))
        if len(uniq) > 1:
            out.append(Check(
                "mode_changes", INFO,
                "the granule spans %d different modes" % len(uniq),
                "The mask solution is re-derived at each change; frames either "
                "side of a change are not interchangeable."))

    # ── obs_id proper ───────────────────────────────────────────────────────
    if "obs_id" in getattr(ds, "variables", {}):
        out.append(Check("obs_id", PASS, "obs_id field present"))
    else:
        out.append(Check(
            "obs_id", WARN, "no field named obs_id",
            "Camera_Operational_Mode is being used in its place. It tracks mode "
            "changes within a granule, which is the evidence for the "
            "substitution, but it has not been confirmed by the SDC."))

    # ── geometry ────────────────────────────────────────────────────────────
    missing, empty, ok = [], [], []
    for short, name in sorted(ANGLE_FIELDS.items()):
        if name not in getattr(ds, "variables", {}):
            missing.append(short)
            continue
        f = _finite_fraction(ds[name], frames=frames, dim=frame_dim)
        (ok if f > 0 else empty).append("%s %.0f%%" % (short, 100 * f))
    if missing:
        out.append(Check(
            "angles_missing", FAIL,
            "viewing/solar angle field(s) absent: " + ", ".join(missing),
            "L2 needs all four. Absent is worse than empty -- it means the "
            "product does not define the field at all."))
    if empty:
        out.append(Check("angles_empty", FAIL,
                         "angle field(s) entirely NaN: " + ", ".join(empty)))
    # The finite fraction has to be read against the FILL fraction, not against
    # 100 %. These are per-pixel fields and only selected pixels have an angle,
    # so a cross-track granule at 7 % fill should show about 7 % finite -- and
    # that means COMPLETE, not 93 % missing. Reporting the bare percentage made
    # a healthy granule look broken.
    if ok and not missing and not empty:
        detail = ("Per-pixel fields: only selected pixels carry an angle, so "
                  "the finite fraction should match the fill fraction, not "
                  "100 %.")
        verdict = PASS
        head = "all four angles present and populated: " + ", ".join(ok)
        if fill_ref is not None and fill_ref > 0:
            got = np.nanmean([float(x.split()[1].rstrip("%")) for x in ok])
            ratio = got / fill_ref
            if ratio > 0.9:
                head = ("all four angles populated on essentially every "
                        "selected pixel (%.1f %% finite against %.1f %% fill)"
                        % (got, fill_ref))
            else:
                verdict = WARN
                head = ("angles populated on only %.0f %% of the selected "
                        "pixels (%.1f %% finite against %.1f %% fill)"
                        % (100 * ratio, got, fill_ref))
        out.append(Check("angles", verdict, head, detail))
    elif ok:
        out.append(Check("angles_partial", WARN,
                         "angles present on only part of the granule: "
                         + ", ".join(ok)))

    # ── mechanism azimuth ───────────────────────────────────────────────────
    if az is None:
        out.append(Check("az", WARN, "no mechanism-azimuth field",
                         "Needed to place the rotating stripe in RAPS."))
    else:
        a = np.asarray(az, dtype=float)
        if not np.isfinite(a).any():
            out.append(Check("az", FAIL, "mechanism azimuth is entirely NaN"))
        elif len(np.unique(a[np.isfinite(a)])) == 1:
            out.append(Check(
                "az", WARN,
                "mechanism azimuth is constant at %.1f deg over the whole granule"
                % float(a[np.isfinite(a)][0]),
                "If any frame is in RAPS this is wrong -- the stripe rotates. "
                "The retrieval measures the angle from the stripe itself and "
                "reports both; see the azimuth panel of the diagnostics."))
        else:
            out.append(Check("az", PASS,
                             "mechanism azimuth varies over %.1f to %.1f deg"
                             % (np.nanmin(a), np.nanmax(a))))

    # ── fill ────────────────────────────────────────────────────────────────
    if fill_pct is not None and len(np.asarray(fill_pct).ravel()):
        f = np.asarray(fill_pct, dtype=float)
        out.append(Check(
            "fill", PASS if np.nanmedian(f) > 0.5 else WARN,
            "selected pixels: median %.2f %% of the FPA (range %.2f to %.2f)"
            % (np.nanmedian(f), np.nanmin(f), np.nanmax(f)),
            "Cross-track runs about 7 %, RAPS about 4 %. A median far below "
            "that means frames are empty or the mask is not being applied."))

    return out


def retrieval_checks(retriever, results):
    """Checks that depend on the retrieval having run."""
    out = []

    # Fill lives here rather than in file_checks because the frame loop has
    # already counted the selected pixels. Re-reading the mask array to count
    # them again costs a full pass over the granule -- on the 900 MB and 1.6 GB
    # granules that was enough to run the container out of memory.
    nx = getattr(retriever, "nx", 2048)
    ny = getattr(retriever, "ny", 2048)
    f = np.array([100.0 * r.observed / float(nx * ny) for r in results])
    if f.size:
        out.append(Check(
            "fill", PASS if np.nanmedian(f) > 0.5 else WARN,
            "selected pixels: median %.2f %% of the FPA (range %.2f to %.2f)"
            % (np.nanmedian(f), np.nanmin(f), np.nanmax(f)),
            "Cross-track runs about 7 %, RAPS about 4 %. A median far below "
            "that means frames are empty or the mask is not being applied."))

    sol = getattr(retriever, "solution", None)
    if sol is None or sol.mode is None:
        out.append(Check("retrieval", FAIL, "no mask solution was established",
                         "Nothing downstream can use this granule."))
        return out

    ok = [r for r in results if r.status.ok]
    out.append(Check(
        "retrieval", PASS if len(ok) == len(results) else
        (WARN if ok else FAIL),
        "%d of %d frames reproduce from the retrieved parameters"
        % (len(ok), len(results)),
        "Only frames marked OK may be used for L2, and within those only the "
        "pixels that carry a component label."))

    # ── centres ─────────────────────────────────────────────────────────────
    if sol.adm_centre is not None:
        out.append(Check(
            "centres", WARN,
            "the ADM pattern and the stripes have DIFFERENT centres: ADM at "
            "(row %d, col %d), stripes elsewhere" % sol.adm_centre,
            "The WFOV table cannot express this -- ADM_MAP points the ADM "
            "generator at the nadir stripe's centre. In an earlier flight "
            "software build the ADM centre was hard-coded to the physical "
            "centre of the array while everything else stayed configurable, so "
            "this is the signature of that build. Worth raising immediately: it "
            "may mean the flight software has reverted."))
    else:
        out.append(Check("centres", PASS,
                         "one centre for all components, as the table assumes"))
    if sol.centre_pending:
        out.append(Check(
            "centre_pinned", WARN,
            "the centre was never pinned by two crossing stripes",
            "In RAPS the only stripes drawn are nadir and the rotating one, and "
            "they stayed near parallel for every frame processed. One axis of "
            "the centre is held at the table value. Process a frame where the "
            "mechanism has turned at least %.0f deg to resolve it."
            % getattr(retriever, "MIN_STRIPE_SEPARATION_DEG", 20.0)))

    # ── mask parameters against the table ───────────────────────────────────
    cfg_persamp = sol.table_overrides.get("wfov_tbl_persamp")
    if cfg_persamp is not None:
        out.append(Check(
            "persamp", INFO, "percent_samp retrieved as %s" % cfg_persamp,
            "This drives the data rate. It has differed between granules: 50 on "
            "2026-03-27 and 100 on 2026-07-12 and 2025-11-18."))
    if sol.step_vza is not None:
        out.append(Check(
            "precession", INFO,
            "precession retrieved as %+.2f deg/mask VZA, %+.2f deg/mask VAA"
            % (sol.step_vza, sol.step_vaa),
            "Measured from the frame-to-frame change of the shifts, not taken "
            "from the table."))
    # ── the FSW geometry defect, as a standing per-granule verdict ─────────
    #
    # Until 2026-10-06 this was reported as a polynomial scale factor, because a
    # scale on the v1 shape was the only way the retrieval could express the
    # flown ring radii. It was never a coefficient problem: flight runs the v0
    # coefficients through an arcsine with a misplaced / M_PI. The retrieval now
    # searches both conversions, so this check reports which one the imagery
    # actually shows -- and will say so by itself the day the fix flies.
    if getattr(sol, "asin_mode", "correct") == "as_coded":
        out.append(Check(
            "fsw_geometry", FAIL,
            "FSW GEOMETRY DEFECT ACTIVE: commanded ADM angles in this granule "
            "are WRONG",
            "The ADM sample positions match the misplaced / M_PI in "
            "WFOVPixelMask_ADM.cpp::getSelectedADMBins(), which linearises the "
            "arcsine. The error is zero at nadir and about -10.7 deg at the "
            "rim, and roughly 28 % of the intended field is never sampled. L2 "
            "must take viewing angles from the L1B GEOLOCATION -- which does "
            "NOT inherit this and is believed correct -- and not from the ADM "
            "bin labels. See docs/FSW_KNOWN_BEHAVIOUR.md. This is a FAIL for "
            "angle-dependent science, not for the mask retrieval itself.",
            scope="angle-dependent science"))
    else:
        out.append(Check(
            "fsw_geometry", PASS,
            "FSW geometry defect NOT present: ADM positions match the correct "
            "arcsine",
            "If this granule post-dates the FSW fix, that is the fix confirmed "
            "from imagery -- add the date to docs/TABLE_HISTORY.md and tell "
            "Beth. If it pre-dates the fix, something else changed and needs "
            "explaining before the granule is used."))

    # A genuine coefficient mismatch is a separate, unexplained finding. Keep it
    # distinct from the arcsine defect, which now has its own check above.
    if sol.coef_scale == sol.coef_scale and abs(sol.coef_scale - 1.0) > 0.025:
        out.append(Check(
            "polynomial", WARN,
            "the field-angle COEFFICIENTS do not match the table either: best "
            "fit is scale %.4f" % sol.coef_scale,
            "This is separate from the arcsine defect and is not explained. "
            "Check that the ADM centre and percent_samp retrieved sensibly "
            "before treating it as real -- a wrong centre shows up here first."))

    # ── azimuth, measured against reported ──────────────────────────────────
    pairs = [(r.az_header, r.az_fitted) for r in results
             if r.az_fitted is not None and r.az_header is not None]
    if pairs:
        # Modulo 180: a stripe with no offset from the centre maps onto itself
        # under a half turn, so 178.8 and -1.2 describe the same picture and
        # reporting the difference as 179 deg would overstate it.
        d = max(abs((f - h + 90.0) % 180.0 - 90.0) for h, f in pairs)
        out.append(Check(
            "az_agreement", PASS if d < 2.0 else WARN,
            "measured stripe azimuth differs from the reported Azimuth by up to "
            "%.0f deg" % d,
            "" if d < 2.0 else
            "The model follows the measured angle. If a new data version "
            "populates Azimuth correctly these should agree to a degree."))

    return out


# ─── rendering ───────────────────────────────────────────────────────────────

_GLYPH = {PASS: "ok  ", WARN: "WARN", FAIL: "FAIL", INFO: "note"}


def render(checks, width=78, title="DATA READINESS"):
    """The checklist as text, worst first within its original grouping."""
    lines = [title, "-" * width]
    v = verdict(checks)
    if v == FAIL:
        failed = [c for c in checks if c.status == FAIL]
        scopes = [c.scope for c in failed if c.scope]
        if len(scopes) == len(failed):
            # Everything that failed is scoped: the granule is usable, with a
            # named restriction. Say exactly what is off limits.
            uniq = []
            for sc in scopes:
                if sc not in uniq:
                    uniq.append(sc)
            lines.append("VERDICT: usable for L2, but NOT for %s -- see the "
                         "FAIL below." % " or ".join(uniq))
        else:
            lines.append("VERDICT: NOT READY for L2 -- at least one required "
                         "item is absent.")
    else:
        lines.append({
            WARN: "VERDICT: usable for L2 with the caveats below.",
            INFO: "VERDICT: usable for L2.",
            PASS: "VERDICT: usable for L2.",
        }[v])
    lines.append("")
    for c in checks:
        lines.append("  [%s]  %s" % (_GLYPH[c.status], c.summary))
        if c.detail:
            for chunk in _wrap(c.detail, width - 10):
                lines.append("          " + chunk)
    lines.append("")
    return "\n".join(lines)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out
