"""Retrieve WFOV pixel-mask table parameters and the field-angle polynomial
from the selected-pixel pattern of a science frame.

Why this exists
---------------
``inspect_l1b_cam.py`` detected stripes with a Radon transform and ADM samples
with blob labelling, knowing nothing about the pixel-mask generator. That is a
hard problem: the objects are sparse, the ADM pattern precesses frame to frame,
and a selected pixel whose radiance happens to be zero is indistinguishable
from an unselected one.

This module inverts it. We have a generator (``combine_masks``) that reproduces
a real frame to Jaccard 0.997, so the question is not "what shapes are in this
image" but "which parameters make my generator output this image". Detection
becomes a residual check rather than the primary method, and every selected
pixel comes out of it with a *provenance* -- nadir stripe, ADM ring 4 bin 11 --
rather than a bounding box.

The three tiers
---------------
The per-frame unknown is the **mask precession phase**. ``vzashift`` and
``vaashift`` advance the ADM bin centres by ~0.9 deg per mask, so the pattern is
a function of the mask counter -- which we cannot read, because
``WFOV_FSW_HEADER_IMAGE_COUNT`` is 0 on every frame of the 2026-07-12 DITL2
snippet (question with Matt/Caitlin as of 2026-10-05). Everything else is
slowly varying. So:

  1. **Per frame, always.** Solve the phase alone -- a 1-D search over the
     precession period. Build the forward model, score it, and if it passes,
     label every selected pixel. Cheap: a few hundred array lookups.
  2. **On a score drop.** Refit the table parameters: mask centre, ``fovr``,
     ``maxPixR``, ``persamp``.
  3. **Only if that still fails.** Refit the field-angle polynomial. Gated --
     see below -- and cached, so it does not run per frame.

The polynomial gate
-------------------
A single frame gives nine ring radii at nine VZA bin centres. That is enough to
fit *one scale factor* onto a known coefficient shape, which is what the
2026-10-03 analysis did (fitted 0.84943, rms 1.85 px). It is NOT enough to fit
six coefficients -- the sample points are clustered at nine values and the fit
is wildly under-determined between them.

The VZA bin spacing is 10 deg and the precession is ~0.9 deg per mask, so the
ring centres sweep a complete bin spacing every ~11.2 frames. Frames beyond
that add redundancy, not new angular coverage. The gate is therefore on
*accumulated phase span*, not frame count:

    scale-only refit :  >= 3 frames          (enough to reject one bad frame)
    full-coefficient :  >= 10 deg phase span  (~12 frames at 0.9 deg/mask)
                        and >= MIN_FRAMES_POLY_FULL frames

When the gate is not met the retrieval does not guess. It returns a status of
``FAIL_INSUFFICIENT_FRAMES`` and the frame is marked unusable.

Failure is a result, not an exception
-------------------------------------
Every frame comes back with a :class:`RetrievalStatus`. ``status.ok`` is False
whenever the forward model could not be made to reproduce the frame, and the
reason is in ``status.code``. Downstream processing must gate on it: **only
pixels carrying a component label may go to L2.** Pixels that are selected in
the data but that no component explains are labelled ``UNEXPLAINED`` and are
excluded by construction.

Orientation
-----------
Every generator in ``libcam.mask`` returns ``mask[x, y]`` with x-major
indexing, and -- verified against the 2026-07-12 frame -- that first axis is the
**FPA row**. So ``x_center_pix`` / ``xc`` is the row of the optical axis (1015)
and ``y_center_pix`` / ``yc`` is the column (949), and a generator raster can be
compared with a decoded image array element-for-element with no transpose. Use
``.T`` only when handing one to ``imshow``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from libcam.mask.adm import (VZA_BOUNDARIES, VAA_BOUNDARIES, PI,
                             PIX_SPACING_MM, EARTH_RADIUS_KM, ALTITUDE_KM,
                             COEF_ANG2DIST, ASIN_MODES, FSW_ASIN_MODE,
                             vza_surface_to_sat, _select_bins)
from libcam.mask.combine_masks import combine_masks, COMPONENTS, MODES
from libcam.mask.wfov_csv2cfg import (load_tables, apply_table_overrides,
                                      adm_cfg, stripe_cfg)

__all__ = [
    "MaskRetriever", "MaskSolution", "FrameResult", "RetrievalStatus",
    "LABELS", "LABEL_NAMES", "UNEXPLAINED", "OBS_ID_MODE", "AdmCell", "cell_tag",
    "MASK_BITS", "BIT_UNEXPLAINED", "BIT_ANY_COMPONENT", "primary_label", "members",
    "ring_radii_for_phase", "measure_blob_centroids",
    "measured_ring_radii", "measured_blob_radii",
]

FPA_NX = 2048
FPA_NY = 2048

# ─── pixel labels ────────────────────────────────────────────────────────────
# 0 is "not selected". Components follow COMPONENTS order so a component keeps
# its code regardless of which others are enabled. UNEXPLAINED is the one that
# matters for data quality: selected in the data, attributed to nothing.
LABELS = {name: i + 1 for i, name in enumerate(COMPONENTS)}
UNEXPLAINED = len(COMPONENTS) + 1
LABELS["unexplained"] = UNEXPLAINED
LABEL_NAMES = {v: k for k, v in LABELS.items()}
LABEL_NAMES[0] = "none"

# ─── membership bits ─────────────────────────────────────────────────────────
# A pixel can belong to more than one component, and it genuinely does: the
# stripes cross the ADM field, so a pixel inside an ADM patch that also lies
# under the cross-track stripe is BOTH samples. Nothing "wins" -- forcing a
# single label there silently deletes one of the two measurements and biases
# whichever component lost, which showed up as ADM cells of 14 pixels sitting
# under a stripe.
#
# So membership is a bitmask, one bit per component plus one for unexplained,
# and ``FrameResult.labels`` is only a convenience view of it (the lowest set
# bit) for code that wants one number per pixel.
MASK_BITS = {name: np.uint16(1 << i) for i, name in enumerate(COMPONENTS)}
MASK_BITS["unexplained"] = np.uint16(1 << len(COMPONENTS))
BIT_UNEXPLAINED = MASK_BITS["unexplained"]
BIT_ANY_COMPONENT = np.uint16(sum(int(MASK_BITS[c]) for c in COMPONENTS))


def primary_label(membership):
    """Lowest set component bit of a membership raster, as a LABELS code.

    A lossy view, kept only for callers that need one value per pixel. Anything
    counting pixels per component should use the bitmask, or it will undercount
    every component that shares pixels with another.
    """
    out = np.zeros(membership.shape, dtype=np.uint8)
    for name in COMPONENTS:                       # COMPONENTS order = bit order
        hit = (membership & MASK_BITS[name]).astype(bool) & (out == 0)
        out[hit] = LABELS[name]
    out[(membership & BIT_UNEXPLAINED).astype(bool) & (out == 0)] = UNEXPLAINED
    return out


def members(membership_value):
    """Component names set in one membership value."""
    return [c for c in COMPONENTS
            if int(membership_value) & int(MASK_BITS[c])] + \
           (["unexplained"] if int(membership_value) & int(BIT_UNEXPLAINED) else [])

# ─── obs_id -> which components to expect ────────────────────────────────────
# obs_id numbering is libcam.util.util.OBS_ID_WFOV; the component sets are
# combine_masks.MODES. Only CT and RAPS have defined component sets -- the other
# modes have never been seen in data and deliberately map to None, which makes
# the retriever say so rather than assume CT.
OBS_ID_MODE = {
    128: "CT",      # Cross-Track
    132: "RAPS",    # Rotating Azimuth Plane Scan
    # Earth-Target draws the SAME component set as RAPS -- nadir stripe, the
    # rotating stripe, and the ADM samples. Per S. Schmidt, 2026-10-05. Before
    # that it had no entry here and its frames returned FAIL_UNKNOWN_MODE, which
    # was the right default while we did not know: 7 of the 44 frames of
    # 2026-03-27 are mode 137 and guessing at them would have produced a
    # confident wrong answer rather than an honest refusal.
    137: "RAPS",    # Earth-Target
}

# What each obs_id is called in the product, for reports. Separate from the
# component set above because two ids can share a component set and still be
# different observations -- which is exactly the case for 132 and 137.
OBS_ID_NAME = {128: "CT", 132: "RAPS", 137: "ET"}

# ─── gates and thresholds ────────────────────────────────────────────────────
MIN_RECALL = 0.80           # predicted pixels that must be non-zero in the data
GOOD_RECALL = 0.90          # above this, no escalation is even considered
MIN_COVERAGE = 0.85         # observed pixels a component must account for
REUNITE_PX = 3.0            # how far beyond a patch an orphan may be adopted
MIN_FRAMES_POLY_SCALE = 3
MIN_FRAMES_POLY_FULL = 12
MIN_PHASE_SPAN_DEG = 10.0   # one VZA bin spacing; see module docstring
RING_TOL_PX = 25.0          # blob-to-predicted-ring association tolerance

# How close to the best stripe recall still counts as "the same answer" when
# measuring how far the fitted centre can slide. This has to be loose, not tight:
# sliding the centre ALONG a stripe barely changes the rendered band, so the
# score falls off by a fraction of a percent over a 100 px slide -- but it does
# fall off, so an exact-tie test sees a clean peak and reports a centre 98 px out
# (2026-03-27 frame 7, which landed on col 851 against a true 949). Anything
# within this of the best is treated as not distinguishable.
CENTRE_PLATEAU_TOL = 0.01
CENTRE_PLATEAU_PX = 24      # a plateau wider than this means that axis is free
BLOB_AREA = (100, 6000)     # see blob_area(): a floor of 600 silently deleted
                            # every clipped blob and fabricated a ring structure

# ─── candidate field-angle polynomials ───────────────────────────────────────
# The retrieval does not assume which coefficient set flight uses; it tries each
# shape, fits a scale onto it, and reports which won and with what residual.
# That is how the 2026-10-03 analysis identified the defect, and keeping it a
# discovery rather than a hard-coded constant means the code will notice if the
# answer ever changes.
#
#   v0  the set in libcam (and in the C reference, pmask_adm.c:27)
#   v1  the newer set from the same source, which flight reproduces the SHAPE
#       of while applying v0's scale -- fitted factor 0.84943 against a
#       v0_linear/v1_linear ratio of 0.84786, 0.2 % apart.
COEF_V0 = list(COEF_ANG2DIST)
COEF_V1 = [1.21367e-10, 2.60014e-09, -7.50181e-06, -2.84788e-06, 1.17022e-01, 0.0]
COEF_CANDIDATES = {"v0": COEF_V0, "v1": COEF_V1}

# ─── candidate surface->satellite conversions ────────────────────────────────
# The second axis of the search, added 2026-10-06. Beth Cervelli's source shows
# flight runs the v0 coefficients through an arcsine with a misplaced / M_PI --
# see vza_surface_to_sat() in adm.py. Before that was known, the retrieval could
# only express the resulting radii as "the other coefficient set, scaled", and
# it reported exactly that: v1 x 0.846. The two curves agree to ~2 px rms over
# the measured range, which is why the proxy was convincing and wrong.
#
# Both conversions are now searched explicitly, so the retrieval reports the
# physical configuration instead of a numerical stand-in -- and so that the day
# the FSW fix flies, the reported answer changes by itself.
ASIN_CANDIDATES = tuple(ASIN_MODES)

# A free scale factor has no physical referent; "as_coded" is a known code
# defect. So when the defect explains the radii at scale 1 to within this
# tolerance, prefer it over a fitted scale that happens to fit slightly better.
# Without this, the 2-parameter (shape, scale) option can out-fit the
# 0-parameter physical one and we would re-derive the retracted conclusion.
# In px of ring-radius rms. See docs/FSW_KNOWN_BEHAVIOUR.md.
UNFITTED_PREFERENCE_PX = 8.0
UNFITTED_SCALE_TOL = 0.025  # a scale this close to 1 counts as "not fitted"


# ═══════════════════════════════════════════════════════════════════════════
# small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _vza_sat(vza_surface, asin_mode="correct"):
    """Surface -> satellite viewing zenith angle [deg], spherical Earth.

    Delegates to the generator so there is exactly one copy of this arithmetic,
    including the hard-coded PI and the ``as_coded`` flight-software variant.
    """
    return vza_surface_to_sat(vza_surface, asin_mode)


def _poly_radius_px(vza_surface, coef, asin_mode="correct"):
    """Ring radius on the FPA [pixels] for a surface VZA, given a polynomial."""
    t = _vza_sat(vza_surface, asin_mode)
    n = len(coef)
    dist = np.zeros_like(np.atleast_1d(t), dtype=float)
    for i, c in enumerate(coef):
        dist = dist + c * t ** (n - i - 1)
    return dist / PIX_SPACING_MM


def _bin_indices(percent_samp):
    """(vza_index, vaa_index) for each ADM sample, in generator order."""
    bins, n_vaa = _select_bins(percent_samp)
    vza_i = np.array([b // n_vaa for b in bins], dtype=int)
    vaa_i = np.array([b % n_vaa for b in bins], dtype=int)
    return vza_i, vaa_i


def ring_radii_for_phase(vza_shift, percent_samp=100, coef=None,
                         asin_mode="correct"):
    """Predicted (vza_surface, radius_px) for each distinct ADM ring at a phase.

    The VZA bin centres are ``5, 15, ... 85`` plus the accumulated shift, wrapped
    at 90 -- the wrap is what produces a new innermost ring every ~100 frames.
    """
    coef = COEF_ANG2DIST if coef is None else coef
    vza_i, _ = _bin_indices(percent_samp)
    centres = (VZA_BOUNDARIES[:-1] + 0.5 * np.diff(VZA_BOUNDARIES))
    vza = (centres[np.unique(vza_i)] + vza_shift) % 90
    return vza, _poly_radius_px(vza, coef, asin_mode)


def blob_area(max_radius_of_pix=20):
    """Area window for an ADM patch of radius r.

    A full patch is pi*r^2 (1257 px at r=20), but a patch clipped by the FPA
    edge or cut by a stripe is a fraction of that and is still a real sample.
    The floor used to be a flat 600 px, which on the 2026-03-27 L1B granule
    dropped 13 of 72 blobs -- and because the losses were concentrated in the
    outer and stripe-crossed rings, it turned 8 rings of 9 into an apparent
    "7 rings of 7-9" and sent the whole retrieval down the wrong path. Do not
    raise this floor to make the blob list look tidier.
    """
    full = np.pi * float(max_radius_of_pix) ** 2
    return (max(60.0, 0.08 * full), 4.0 * full)


def measure_blob_centroids(obs, area=BLOB_AREA, exclude=None):
    """Centroids of compact blobs in a boolean frame -- the ADM samples.

    ``exclude`` is a boolean raster (typically the predicted stripes) removed
    before labelling, so a stripe crossing the ADM field does not weld several
    samples into one blob.

    Returns ``(rows, cols)`` float arrays. Empty arrays if scipy is unavailable.
    """
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy is in the [full] extra
        return np.array([]), np.array([])

    m = np.asarray(obs, dtype=bool).copy()
    if exclude is not None:
        m &= ~np.asarray(exclude, dtype=bool)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.array([]), np.array([])
    size = ndimage.sum(m, lab, range(1, n + 1))
    keep = [k for k in range(1, n + 1) if area[0] < size[k - 1] < area[1]]
    if not keep:
        return np.array([]), np.array([])
    cen = ndimage.center_of_mass(m, lab, keep)
    return (np.array([c[0] for c in cen], dtype=float),
            np.array([c[1] for c in cen], dtype=float))


# ═══════════════════════════════════════════════════════════════════════════
# result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalStatus:
    """Outcome of one frame's retrieval.

    ``ok`` is the gate for downstream processing. ``code`` is machine-readable
    and stable; ``message`` is for humans and the log file.
    """
    ok: bool
    code: str
    message: str = ""
    recall: float = float("nan")
    jaccard: float = float("nan")
    coverage: float = float("nan")   # explained / observed
    tier: int = 1               # highest tier reached: 1 phase, 2 table, 3 poly

    def __str__(self):
        return "%-28s recall=%.3f jaccard=%.3f coverage=%.3f  %s" % (
            self.code, self.recall, self.jaccard, self.coverage, self.message)


@dataclass
class MaskSolution:
    """The slowly-varying part of the retrieval -- cached across frames.

    Keyed by ``(obs_id, pixel_mask_id, table_fingerprint)``. A change in any of
    those invalidates the cache and forces a fresh solve, so a mode change can
    never silently reuse the previous mode's answer.
    """
    obs_id: int | None = None
    pixel_mask_id: int | None = None
    table_fingerprint: str = ""
    mode: str | None = None
    enabled: dict = field(default_factory=dict)
    table_overrides: dict = field(default_factory=dict)
    coef: list = field(default_factory=lambda: list(COEF_ANG2DIST))
    coef_scale: float = 1.0
    asin_mode: str = "correct"  # which surface->satellite conversion the
                                # imagery is consistent with; "as_coded" means
                                # the FSW M_PI defect is still flying
    adm_centre: tuple = None    # (row, col) if the ADM pattern is not on the stripe centre
    obs_name: str = None        # what the product calls this obs_id (CT / RAPS / ET)
    step_vza: float = None      # retrieved precession, deg per mask -- NOT the table value
    step_vaa: float = None
    centre_pending: bool = False  # the stripes could not pin the centre on the bootstrap frame
    provenance: dict = field(default_factory=dict)
    n_frames_fitted: int = 0
    phase_span_deg: float = 0.0

    def key(self):
        return (self.obs_id, self.pixel_mask_id, self.table_fingerprint)

    def fsw_defect_active(self):
        """True while the imagery says the misplaced-M_PI defect is flying.

        This is the single predicate everything else should ask. When the FSW
        fix reaches the instrument, a granule taken after it will retrieve
        ``asin_mode == "correct"`` and this goes False by itself -- no code
        change, no constant to remember to edit. That is the whole point.
        """
        return self.asin_mode == FSW_ASIN_MODE

    def fsw_geometry_banner(self):
        """Lines announcing the state of the FSW geometry defect. Never empty."""
        if self.fsw_defect_active():
            return [
                "",
                "  " + "=" * 68,
                "  ==%s==" % "FSW GEOMETRY DEFECT IS ACTIVE IN THIS GRANULE".center(64),
                "  " + "=" * 68,
                "  The ADM sample positions are consistent with the misplaced",
                "  / M_PI in WFOVPixelMask_ADM.cpp::getSelectedADMBins(), which",
                "  linearises the arcsine. COMMANDED ADM ANGLES ARE WRONG: the",
                "  error is zero at nadir and about -10.7 deg at the rim, and",
                "  roughly 28 % of the intended field is never sampled.",
                "",
                "  L2 must take viewing angles from the L1B GEOLOCATION, not",
                "  from the ADM bin labels, for this granule.",
                "  See docs/FSW_KNOWN_BEHAVIOUR.md.",
                "  " + "=" * 68,
            ]
        return [
            "",
            "  ++ FSW geometry defect NOT detected: the ADM positions are",
            "     consistent with the CORRECT arcsine. If this granule post-",
            "     dates the FSW fix, that is the fix confirmed from imagery --",
            "     record the date in docs/TABLE_HISTORY.md. If it pre-dates the",
            "     fix, something else changed and it needs explaining.",
        ]

    def describe(self):
        lines = ["mask solution",
                 "  obs_id          = %s (%s%s)" % (
                     self.obs_id, self.obs_name or self.mode or "unknown mode",
                     "" if (self.obs_name or self.mode) == self.mode
                     else ", same component set as %s" % self.mode),
                 "  pixel_mask_id   = %s" % self.pixel_mask_id,
                 "  components      = %s" % ", ".join(
                     k for k, v in self.enabled.items() if v),
                 "  coef_scale      = %.5f  [%s]" % (
                     self.coef_scale, self.provenance.get("coef", "table/default")),
                 "  coefficients    = [%s]" % ", ".join("%.6g" % c for c in self.coef),
                 "  asin_mode       = %s  [%s]" % (
                     self.asin_mode,
                     self.provenance.get("asin_mode", "assumed"))]
        lines += self.fsw_geometry_banner()
        if self.step_vza is not None:
            lines.append("  precession     = %+.3f deg/mask VZA, %+.3f deg/mask VAA"
                         "   [%s]" % (self.step_vza, self.step_vaa,
                                      self.provenance.get("precession", "measured")))
        if self.adm_centre is not None:
            lines += [
                "  adm_centre      = (row %d, col %d)  <- NOT the stripe centre"
                % self.adm_centre,
                "      %s" % self.provenance.get("adm_centre", ""),
                "      There is no table parameter for this. The ADM generator "
                "takes its centre",
                "      from wfov_tbl_x/ycenter_stripe_nadir, so a granule whose "
                "two patterns are",
                "      drawn about different centres cannot be described by the "
                "table at all.",
            ]
        for k, v in sorted(self.table_overrides.items()):
            # Round: these come out of a 0.02 deg grid search, so printing them
            # to 15 significant figures implies a precision that is not there.
            if isinstance(v, float):
                v = round(v, 3)
            lines.append("  %-15s = %-12s [%s]" % (k, v, self.provenance.get(k, "fitted")))
        return "\n".join(lines)


@dataclass
class AdmCell:
    """One ADM angular sample, as a distinct object.

    ``pixels`` holds flat indices into the ``(nx, ny)`` frame -- flat rather
    than a boolean raster because a frame carries up to 162 of these and a
    2048x2048 bool each would be 680 MB. Recover the 2-D form with
    ``np.unravel_index(cell.pixels, (nx, ny))``.
    """
    i_vza: int
    i_vaa: int
    vza: float                  # surface viewing zenith angle of the bin centre [deg]
    vaa: float                  # viewing azimuth angle of the bin centre [deg]
    tag: str = ""               # "C7": ring letter + azimuth number
    pixels: np.ndarray = None   # flat indices
    centre_pred: tuple = None   # (row, col) where the generator puts it
    centre_obs: tuple = None    # (row, col) centroid of the pixels actually there
    radius_pred: float = float("nan")
    n_reunited: int = 0         # of ``pixels``, how many were orphans

    @property
    def n(self):
        return 0 if self.pixels is None else int(len(self.pixels))

    @property
    def offset(self):
        """Observed minus predicted centre, in pixels -- the per-sample residual."""
        if self.centre_obs is None or self.centre_pred is None:
            return (float("nan"), float("nan"))
        return (self.centre_obs[0] - self.centre_pred[0],
                self.centre_obs[1] - self.centre_pred[1])


@dataclass
class FrameResult:
    """Per-frame retrieval output."""
    index: int
    phase: int | None = None
    vza_shift: float = 0.0
    vaa_shift: float = 0.0
    az: float = 0.0
    status: RetrievalStatus = None
    labels: np.ndarray = None           # (nx, ny) uint8, see LABELS
    parts: dict = field(default_factory=dict)      # component -> predicted px
    observed: int = 0
    explained: int = 0
    unexplained: int = 0
    missing: int = 0                    # predicted but not selected in the data
    membership: np.ndarray = None       # (nx, ny) uint16 bitmask, see MASK_BITS
    observed_parts: dict = field(default_factory=dict)  # per component, px SEEN
    overlap: int = 0                    # pixels belonging to >1 component
    adm_cells: np.ndarray = None        # (n_vza, n_vaa) object array of AdmCell
    reunited: int = 0                   # orphan pixels folded back into a cell
    measured_shifts: tuple = None       # (vza_shift, vaa_shift, blob hit rate)
    az_header: float = None             # mechanism azimuth as the granule states it
    az_fitted: float = None             # ... and as the rotating stripe actually shows

    def cells(self):
        """Iterate the populated ADM cells as ``(i_vza, i_vaa, cell)``."""
        if self.adm_cells is None:
            return
        for iv in range(self.adm_cells.shape[0]):
            for ia in range(self.adm_cells.shape[1]):
                cell = self.adm_cells[iv, ia]
                if cell is not None:
                    yield iv, ia, cell

    def summary(self):
        return ("frame %3d  phase %-5s  %s  obs=%d explained=%d "
                "unexplained=%d missing=%d"
                % (self.index, self.phase, self.status.code, self.observed,
                   self.explained, self.unexplained, self.missing))


# ═══════════════════════════════════════════════════════════════════════════
# the retriever
# ═══════════════════════════════════════════════════════════════════════════

class MaskRetriever:
    """Forward-model retrieval of mask parameters, frame by frame.

    Usage::

        r = MaskRetriever(table_dir="docs/wfov_tables")
        for i, frame in enumerate(frames):
            res = r.process_frame(frame > 0, header=hdr[i], index=i)
        r.write_report("retrieved_mask_parameters.txt")

    The retriever holds the cached :class:`MaskSolution` and a rolling buffer of
    accepted frames used by the tier-2 and tier-3 refits. It never raises on a
    bad frame -- check ``res.status.ok``.
    """

    def __init__(self, table_dir=None, min_recall=MIN_RECALL,
                 good_recall=GOOD_RECALL, min_coverage=MIN_COVERAGE,
                 reunite_orphans=True, reunite_px=REUNITE_PX,
                 measure_shifts=True, measure_az=True,
                 period=None, verbose=True,
                 allow_poly_fit=True, coef_candidates=None,
                 asin_candidates=None,
                 nx=FPA_NX, ny=FPA_NY):
        self.tbl0 = load_tables(table_dir)
        self.table_dir = table_dir
        self.min_recall = min_recall
        self.good_recall = good_recall
        self.min_coverage = min_coverage
        self.reunite_orphans = reunite_orphans
        self.reunite_px = reunite_px
        self.measure_shifts = measure_shifts
        self.measure_az = measure_az
        self._persamp_fixed = False
        self.verbose = verbose
        self.allow_poly_fit = allow_poly_fit
        self.coef_candidates = dict(coef_candidates or COEF_CANDIDATES)
        self.asin_candidates = tuple(asin_candidates or ASIN_CANDIDATES)
        self.nx, self.ny = nx, ny

        self.solution = None
        self.log = []
        self._buffer = []        # accepted (phase, rows, cols) for refits
        self._last_phase = None
        self._anchor = None      # (frame index, phase) once established
        self._table_fitted = False
        self._bootstrap_phase = None
        self._cfg_cache = (None, None)   # (overrides repr, AdmCfg)
        self._shift_hist = []    # (frame index, vza_shift, vaa_shift)
        self._az_hist = []       # (frame index, header az, measured az, score)

        # Precession period in frames: the shift that returns the VZA bin grid
        # to its starting configuration. 90 deg / 0.9 deg-per-mask = 100.
        # Precession period in frames -- the mask counter at which BOTH bin
        # grids return to their starting configuration.
        #
        # Getting this wrong is subtle and expensive. The VZA grid repeats after
        # 90 deg of shift (100 frames at 0.9 deg/mask), so 100 looks like the
        # answer -- but the VAA grid only repeats after 360 deg, which is 400
        # frames. Treating the period as 100 makes frame 22 of the 2026-07-12
        # granule "wrap" from phase 99 to phase 0 when the true next phase is
        # 100, and the azimuth is then a quarter turn out: coverage drops from
        # 0.98 to 0.60 while recall stays near 0.92, because recall alone cannot
        # see pixels the model has stopped predicting.
        import math
        cfg = adm_cfg(self.tbl0)
        step = float(cfg.vza_shift or 0.9)
        astep = float(cfg.vaa_shift or step)
        self.step_deg = step
        if period:
            self.period = period
        else:
            n_vza = int(round(90.0 / step)) if step else 1
            n_vaa = int(round(360.0 / astep)) if astep else 1
            self.period = max(1, math.lcm(max(n_vza, 1), max(n_vaa, 1)))

    # ── logging ──────────────────────────────────────────────────────────────

    def _say(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)

    # ── table assembly ───────────────────────────────────────────────────────

    def _fingerprint(self, tbl):
        items = sorted((k, v) for k, v in tbl.items() if not k.startswith("_"))
        return "%08x" % (abs(hash(repr(items))) & 0xFFFFFFFF)

    def _tbl(self):
        """The working table: file values plus whatever has been refitted."""
        if self.solution and self.solution.table_overrides:
            return apply_table_overrides(self.tbl0, self.solution.table_overrides)
        return self.tbl0

    def _ensure_solution(self, header):
        """Create or invalidate the cached solution for this frame's header."""
        obs_id = header.get("cam_obs_id")
        pm_id = header.get("pixel_mask_id")
        fp = self._fingerprint(self.tbl0)
        want = (obs_id, pm_id, fp)

        if self.solution is not None and self.solution.key() == want:
            if self.solution.mode is not None:
                return None  # cache valid
            # A cached FAIL is not a usable solution. Without this the second and
            # every later frame of an unknown mode matched the cache, skipped the
            # check below and ran the whole retrieval with enabled={} -- no
            # stripes to exclude, no ADM component -- and "succeeded" on noise.
            # Seen on 2026-03-27 frames 0-6 (mode 137, Earth-Target).
            return RetrievalStatus(
                False, "FAIL_UNKNOWN_MODE",
                "obs_id %s has no component set in OBS_ID_MODE; known: %s"
                % (obs_id, ", ".join(str(k) for k in OBS_ID_MODE)))

        mode = OBS_ID_MODE.get(obs_id)
        if mode is None:
            # 2026-03-27 carries Camera_Operational_Mode 132 AND 137 in one
            # granule -- RAPS and Earth-Target. 137 has no component set, so
            # those frames are reported rather than guessed at.
            # Deliberately not falling back to CT. An unknown mode means we do
            # not know which components to expect, and guessing would produce a
            # confident wrong answer.
            self.solution = MaskSolution(obs_id=obs_id, pixel_mask_id=pm_id,
                                         table_fingerprint=fp)
            return RetrievalStatus(
                False, "FAIL_UNKNOWN_MODE",
                "obs_id %s has no component set in OBS_ID_MODE; known: %s"
                % (obs_id, ", ".join(str(k) for k in OBS_ID_MODE)))

        if self.solution is not None:
            self._say("  solution cache invalidated: %s -> %s"
                      % (self.solution.key(), want))
            self._buffer.clear()
            self._last_phase = None
            self._anchor = None
            self._table_fitted = False

        self.solution = MaskSolution(
            obs_id=obs_id, pixel_mask_id=pm_id, table_fingerprint=fp,
            mode=mode, obs_name=OBS_ID_NAME.get(obs_id, str(obs_id)),
            enabled=dict(MODES[mode]),
            provenance={"enabled": "obs_id %s -> MODES[%r]" % (obs_id, mode),
                        "coef": "table default (NOT the flown value until fitted)"})
        self._say("  new solution for obs_id=%s pixel_mask_id=%s -> mode %s (%s)"
                  % (obs_id, pm_id, mode,
                     ", ".join(k for k, v in MODES[mode].items() if v)))
        return None

    # ── forward model ────────────────────────────────────────────────────────

    def solve_adm_centre(self, obs, stripes=None):
        """Find the centre the ADM pattern is drawn about.

        It is NOT always the stripe centre. On 2025-11-18 the stripes sit about
        (949, 949) while the ADM rings are concentric about (1024, 1024) -- the
        array centre. ``ADM_MAP`` points the ADM generator at the nadir stripe's
        centre, so with a single centre the rings are 75 px out and nothing
        fits; the retrieval then blames the polynomial and fails.

        Scored on how tightly the blob radii cluster: a wrong centre smears
        every ring into a broad band, the right one makes them narrow. That is
        independent of the polynomial, the phase and percent_samp, so it can be
        solved before any of them.
        """
        cfg = self._acfg()
        rows, cols = measure_blob_centroids(
            obs, area=blob_area(cfg.max_radius_of_pix), exclude=stripes)
        if len(rows) < 8:
            return None, 0.0
        tightness = self._ring_tightness(rows, cols)

        cands = [(int(cfg.x_center_pix), int(cfg.y_center_pix)),
                 (self.nx // 2, self.ny // 2)]
        best = None
        for r0, c0 in cands:
            # Same width as the stripe-centre search. A narrow window only works
            # when one of the two seeds is already right; on the RAPS frames the
            # stripe seed is itself degenerate in one axis, so the true ADM
            # centre is tens of pixels from either seed.
            for step, span in ((8, 88), (2, 10)):
                local = None
                for dr in range(-span, span + 1, step):
                    for dc in range(-span, span + 1, step):
                        t = tightness(r0 + dr, c0 + dc)
                        if local is None or t < local[0]:
                            local = (t, r0 + dr, c0 + dc)
                r0, c0 = local[1], local[2]
            if best is None or local[0] < best[0]:
                best = (local[0], r0, c0)
        return (best[1], best[2]), best[0]

    @staticmethod
    def _ring_tightness(rows, cols):
        """Closure scoring a candidate centre by how tightly the rings cluster.

        Rings are cut at radial gaps wider than ``RING_TOL_PX``; the score is the
        mean within-ring standard deviation. A wrong centre turns every ring into
        a broad band, so this is minimised at the true one -- with no reference
        to the polynomial, the phase or percent_samp.
        """
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)

        def tightness(r0, c0):
            rad = np.sort(np.hypot(rows - r0, cols - c0))
            out, start = [], 0
            for i in range(1, len(rad) + 1):
                if i == len(rad) or rad[i] - rad[i - 1] > RING_TOL_PX:
                    if i - start >= 3:
                        out.append(float(np.std(rad[start:i])))
                    start = i
            return float(np.mean(out)) if out else 1e9
        return tightness

    def _adm_centre_score(self, obs, stripes, centre):
        """Ring tightness at one given centre -- the baseline to beat."""
        cfg = self._acfg()
        rows, cols = measure_blob_centroids(
            obs, area=blob_area(cfg.max_radius_of_pix), exclude=stripes)
        if len(rows) < 8:
            return centre, 1e9
        return centre, self._ring_tightness(rows, cols)(centre[0], centre[1])

    def build(self, phase, az=0.0):
        """The forward model at one phase. Returns ``(mask, counts)``.

        ``coef_ang2dist`` is only passed when the ADM component is enabled --
        ``combine_masks`` rejects an override no enabled component accepts, and
        a stripe-only mode (or an unrecognised one, where ``enabled`` is empty)
        would otherwise raise instead of simply building the stripes.

        When ``solution.adm_centre`` is set the two patterns do not share a
        centre, and that cannot be expressed in one ``combine_masks`` call:
        ``ADM_MAP`` routes ``x_center_pix`` to ``wfov_tbl_xcenter_stripe_nadir``,
        so a single table carries one centre for both. The mask is then built in
        two passes -- stripes on the stripe table, ADM on a copy whose nadir
        centre is the ADM centre -- and the layers merged. See
        :meth:`solve_adm_centre` for where that case comes from.
        """
        en = self.solution.enabled
        split = en.get("adm") and self.solution.adm_centre is not None
        if not split:
            over = {"coef_ang2dist": list(self.solution.coef),
                    "asin_mode": self.solution.asin_mode} \
                if en.get("adm") else None
            return combine_masks(self._tbl(), en, az=az, frame=phase,
                                 nx=self.nx, ny=self.ny, overrides=over)

        # pass 1: everything but the ADM, on the stripe centre
        others = {k: (v and k != "adm") for k, v in en.items()}
        if any(others.values()):
            mask, counts = combine_masks(self._tbl(), others, az=az, frame=phase,
                                         nx=self.nx, ny=self.ny)
        else:
            mask = np.zeros((self.nx, self.ny), dtype=np.uint8)
            counts = dict(parts={}, layers={}, sum_parts=0, total=0, overlap=0,
                          az=az, frame=phase, adm_vza_shift=0.0,
                          adm_vaa_shift=0.0, fill_pct=0.0)

        # pass 2: the ADM alone, on a table whose nadir centre IS the ADM centre
        ar, ac = self.solution.adm_centre
        tbl2 = apply_table_overrides(self.tbl0, dict(
            self.solution.table_overrides,
            wfov_tbl_xcenter_stripe_nadir=int(ar),
            wfov_tbl_ycenter_stripe_nadir=int(ac)))
        amask, acounts = combine_masks(
            tbl2, {"adm": True}, az=az, frame=phase, nx=self.nx, ny=self.ny,
            overrides={"coef_ang2dist": list(self.solution.coef),
                       "asin_mode": self.solution.asin_mode})

        mask = (mask.astype(bool) | amask.astype(bool))
        counts["layers"]["adm"] = acounts["layers"]["adm"]
        counts["parts"]["adm"] = acounts["parts"]["adm"]
        # keep COMPONENTS order, which the renderer's colour map relies on
        counts["layers"] = {k: counts["layers"][k] for k in COMPONENTS
                            if k in counts["layers"]}
        counts["parts"] = {k: counts["parts"][k] for k in COMPONENTS
                           if k in counts["parts"]}
        counts["sum_parts"] = sum(counts["parts"].values())
        counts["total"] = int(mask.sum())
        counts["overlap"] = counts["sum_parts"] - counts["total"]
        counts["fill_pct"] = 100.0 * counts["total"] / (self.nx * self.ny)
        counts["adm_vza_shift"] = acounts["adm_vza_shift"]
        counts["adm_vaa_shift"] = acounts["adm_vaa_shift"]
        counts["adm_centre"] = (int(ar), int(ac))
        return mask.astype(np.uint8), counts

    def _acfg(self):
        """Cached AdmCfg for the current override state.

        ``adm_cfg(apply_table_overrides(...))`` rebuilds the whole table dict, so
        calling it inside a 100-phase search dominated the runtime. The override
        dict changes rarely; key the cache on it.
        """
        key = repr(sorted(self.solution.table_overrides.items())) \
            if self.solution else ""
        if self._cfg_cache[0] != key:
            self._cfg_cache = (key, adm_cfg(self._tbl()))
        return self._cfg_cache[1]

    def _adm_xy(self):
        """Centre the ADM pattern is drawn about -- its own, or the stripes'."""
        if self.solution is not None and self.solution.adm_centre is not None:
            return self.solution.adm_centre
        c = self._acfg()
        return c.x_center_pix, c.y_center_pix

    def _adm_centres(self, phase):
        """Predicted ADM sample centres (rows, cols) at a phase -- the fast path.

        Phase search scores 162 points rather than ~200 000 pixels, which makes
        a full-period search cost about as much as building one mask.
        """
        from libcam.mask.adm import adm_sample_centres
        cfg = self._acfg()
        vza_s, vaa_s = cfg.shift_at(phase)
        _xc, _yc = self._adm_xy()
        r, c = adm_sample_centres(
            percent_samp=cfg.percent_samp,
            vza_shift=vza_s, vaa_shift=vaa_s,
            x_center_pix=_xc, y_center_pix=_yc,
            coef_ang2dist=list(self.solution.coef),
            asin_mode=self.solution.asin_mode)
        return np.asarray(r), np.asarray(c), vza_s, vaa_s

    # ── bootstrap ────────────────────────────────────────────────────────────

    def bootstrap(self, obs, az=0.0):
        """Break the circular dependency, once, on the first usable frame.

        The phase cannot be solved without the polynomial (wrong radii put every
        predicted sample in the wrong place) and the polynomial cannot be fitted
        without the phase (the VZA bin centres depend on it). Three steps, each
        of which depends only on things already known:

          1. **Centre, from the stripes alone.** The stripe geometry is purely
             table-driven: no polynomial, no phase. A local search on stripe
             recall gives the mask centre outright.
          2. **Ring radii, from the blobs about that centre.** Still no
             polynomial and no phase -- a radius is a radius.
          3. **Phase and polynomial jointly.** For every candidate coefficient
             *shape* and every phase in the period, the best scale factor is a
             closed-form least-squares solve against the measured ring radii.
             The winner is the (shape, phase, scale) with the lowest residual.

        Step 3 is the step that identified the flight-software defect by hand on
        2026-10-03; it is written here as a search so the code reaches the same
        conclusion from the data rather than being told.
        """
        sol = self.solution

        # ── 1. centre from the stripes ──────────────────────────────────────
        fams = {k: v for k, v in sol.enabled.items()
                if v and k in ("nadir", "forward", "cross", "raps")}
        self.bootstrap_centre(obs, az)

        # ── 1b. the ADM centre, which need not be the stripe centre ─────────
        #
        # ADM_MAP points the ADM generator at the nadir stripe's centre, so the
        # code above would force one centre on both. On 2025-11-18 that is wrong:
        # the stripes sit at (949, 949) and the ADM rings are concentric about
        # the array centre (1024, 1024). Detected, not assumed -- see
        # solve_adm_centre; it only overrides when the rings are materially
        # tighter about the other centre, so the usual single-centre granules
        # (2026-03-27, 2026-07-12) are untouched.
        stripes = np.zeros((self.nx, self.ny), dtype=bool)
        if fams:
            m, _ = combine_masks(self._tbl(), fams, az=az, frame=0,
                                 nx=self.nx, ny=self.ny)
            stripes = m.astype(bool)
        self.fit_adm_centre(obs, stripes)
        return self._bootstrap_rings(obs, az, fams, stripes)

    def bootstrap_centre(self, obs, az=0.0):
        """Fit the mask centre from the stripes alone. Returns a provenance line.

        Separate from :meth:`bootstrap` because it may have to be run again on a
        later frame -- see :meth:`retry_centre`.
        """
        sol = self.solution
        fams = {k: v for k, v in sol.enabled.items()
                if v and k in ("nadir", "forward", "cross", "raps")}
        best = (None, -1.0)
        if fams:
            # Coarse-then-fine, and WIDE. The centre is not a small correction
            # to the table value: 2025-11-18 ran (949, 949) where 2026-03-27 and
            # 2026-07-12 ran (1015, 949) -- a 66 px move. A +/-20 px search
            # missed it entirely and the whole retrieval failed on that granule.
            tbl_r = base_r = int(self._acfg().x_center_pix)
            tbl_c = base_c = int(self._acfg().y_center_pix)
            flat_r = flat_c = 0.0
            for step, span in ((8, 88), (2, 10)):
              grid = []
              for dr in range(-span, span + 1, step):
                for dc in range(-span, span + 1, step):
                    trial = dict(sol.table_overrides)
                    for fam in ("nadir", "forward", "cross", "raps"):
                        trial["wfov_tbl_xcenter_stripe_%s" % fam] = base_r + dr
                        trial["wfov_tbl_ycenter_stripe_%s" % fam] = base_c + dc
                    trial["wfov_tbl_xcenter_limb"] = base_r + dr
                    trial["wfov_tbl_ycenter_limb"] = base_c + dc
                    sol.table_overrides = trial
                    self._cfg_cache = (None, None)
                    m, _ = combine_masks(self._tbl(), fams, az=az, frame=0,
                                         nx=self.nx, ny=self.ny)
                    rec = _recall(m.astype(bool), obs)
                    grid.append((rec, base_r + dr, base_c + dc))
                    if rec > best[1]:
                        best = (trial, rec)
              # How far can the centre move along each axis without the score
              # changing? That is the width of the plateau, and it is how the
              # degenerate case announces itself.
              # Take the WIDEST plateau seen over the passes, not the last one.
              # The fine pass only spans +/-10 px, so a plateau measured there
              # can never exceed 20 and a genuinely free axis reads as pinned.
              top = [g for g in grid if g[0] >= best[1] - CENTRE_PLATEAU_TOL]
              flat_r = max(flat_r, max(g[1] for g in top) - min(g[1] for g in top))
              flat_c = max(flat_c, max(g[2] for g in top) - min(g[2] for g in top))
              if best[0] is not None:     # refine around the coarse winner
                  base_r = best[0]["wfov_tbl_xcenter_stripe_nadir"]
                  base_c = best[0]["wfov_tbl_ycenter_stripe_nadir"]

            # ── identifiability ─────────────────────────────────────────────
            #
            # Stripe recall only pins the centre ACROSS each stripe, never along
            # it. In CT three stripes cross and both axes are pinned. In RAPS the
            # component set is nadir + the rotating raps stripe, and when the
            # mechanism sits near az 0 the two lie on top of each other: the
            # frame carries one horizontal band, the row is pinned to a pixel and
            # the column is free. The search then reports recall 1.0000 at an
            # arbitrary column -- on 2026-03-27 frame 7 it produced col 861
            # against a true 949, and every later step inherited the error.
            # So when an axis is flat, keep the table value for it and say so.
            PLATEAU_PX = CENTRE_PLATEAU_PX
            fitted = dict(best[0])
            degen = []
            if flat_r > PLATEAU_PX:
                degen.append(("x", tbl_r))
            if flat_c > PLATEAU_PX:
                degen.append(("y", tbl_c))
            for axis, val in degen:
                for fam in ("nadir", "forward", "cross", "raps"):
                    fitted["wfov_tbl_%scenter_stripe_%s" % (axis, fam)] = val
                fitted["wfov_tbl_%scenter_limb" % axis] = val
            # Leave a flag, not a conclusion. In CT the cross-track stripe
            # crosses the nadir stripe and their intersection pins the centre
            # outright. RAPS has no cross-track stripe -- only nadir plus the
            # rotating one -- so until the mechanism has carried that stripe away
            # from parallel there is nothing to intersect, and the centre is
            # genuinely not in this frame. ``retry_centre`` picks it up again on
            # the first frame where the two stripes are far enough apart.
            sol.centre_pending = bool(degen)
            sol.table_overrides = fitted
            self._cfg_cache = (None, None)
            sol.provenance["centre"] = (
                "fitted from the stripes alone (no polynomial, no phase), "
                "stripe recall %.4f; plateau width %d px in row, %d px in col%s"
                % (best[1], flat_r, flat_c,
                   "" if not degen else
                   " -- %s unidentifiable from this component set, left at the "
                   "table value" % " and ".join(a for a, _ in degen)))
            self._say("  bootstrap 1/3  centre -> (row %s, col %s), stripe recall "
                      "%.4f  [plateau %d x %d px]"
                      % (self._acfg().x_center_pix, self._acfg().y_center_pix,
                         best[1], flat_r, flat_c))
            if degen:
                self._say("                 %s unidentifiable from the stripes "
                          "alone (the enabled stripes are collinear at az %.1f); "
                          "held at the table value" % (
                              " and ".join("%s-centre" % a for a, _ in degen), az))
            return sol.provenance["centre"]
        return None

    def fit_adm_centre(self, obs, stripes):
        """Decide whether the ADM pattern has a centre of its own, and record it."""
        sol = self.solution
        if sol.enabled.get("adm"):
            cand, tight = self.solve_adm_centre(obs, stripes)
            if cand is not None:
                scfg = self._acfg()
                s_xy = (int(scfg.x_center_pix), int(scfg.y_center_pix))
                _, tight_stripe = self._adm_centre_score(obs, stripes, s_xy)
                moved = max(abs(cand[0] - s_xy[0]), abs(cand[1] - s_xy[1]))
                # Accept only a clear win: 25 % tighter rings AND a real move.
                if moved >= 4 and tight < 0.75 * tight_stripe:
                    sol.adm_centre = (int(cand[0]), int(cand[1]))
                    sol.provenance["adm_centre"] = (
                        "ADM rings are concentric about (%d, %d), NOT the stripe "
                        "centre (%d, %d): within-ring radial spread %.1f px vs "
                        "%.1f px. The two patterns are drawn about different "
                        "centres in this granule."
                        % (cand[0], cand[1], s_xy[0], s_xy[1], tight, tight_stripe))
                    self._say("  bootstrap 1b   ADM centre -> (row %d, col %d), "
                              "distinct from the stripe centre (row %d, col %d); "
                              "ring spread %.1f px vs %.1f px"
                              % (cand[0], cand[1], s_xy[0], s_xy[1],
                                 tight, tight_stripe))
                else:
                    self._say("  bootstrap 1b   ADM centre agrees with the stripes "
                              "(best alternative (%d, %d) moves %d px, spread "
                              "%.1f px vs %.1f px -- not accepted)"
                              % (cand[0], cand[1], moved, tight, tight_stripe))

    def _bootstrap_rings(self, obs, az, fams, stripes):
        """Steps 2 and 3 of the bootstrap: ring radii, then phase and polynomial."""
        sol = self.solution
        # ── 2. observed ring radii ──────────────────────────────────────────
        cfg = self._acfg()
        rows, cols = measure_blob_centroids(
            obs, area=blob_area(cfg.max_radius_of_pix), exclude=stripes)
        if len(rows) < 9:
            # Distinguish "this granule genuinely has too few blobs" from
            # "scipy isn't importable in this interpreter, so
            # measure_blob_centroids() returned an empty list by design (see
            # its docstring)". Added 2026-10-13 after exactly that silent
            # case read as a DITL2 data finding for most of a day, on an
            # interpreter where scipy had quietly gone missing -- it failed
            # identically on a known-good granule, which is what gave it away.
            from libcam.env_check import explain_missing
            env_msg = explain_missing("scipy", why="ADM blob detection "
                                      "(measure_blob_centroids)")
            if env_msg is not None:
                return RetrievalStatus(False, "FAIL_ENV_SCIPY_MISSING",
                                       env_msg, tier=3)
            return RetrievalStatus(
                False, "FAIL_NO_RINGS",
                "bootstrap found only %d ADM blobs; cannot establish the ring "
                "radii" % len(rows), tier=3)
        ps = cfg.percent_samp
        xc, yc = self._adm_xy()
        radii = np.hypot(rows - xc, cols - yc)
        azim = np.degrees(np.arctan2(rows - xc, cols - yc)) % 360
        self._say("  bootstrap 2/3  %d ADM blobs about (row %s, col %s), "
                  "radii %.0f-%.0f px"
                  % (len(rows), xc, yc, radii.min(), radii.max()))

        # ── 3a. (shape, scale, phase) from the radii of ALL blobs ───────────
        #
        # Scoring the handful of clustered ring medians is under-determined:
        # rings drop off the FPA at high VZA and the innermost appears and
        # disappears with the precession, so six medians can be matched to a
        # wrong phase at a plausible residual. Every blob radius is a separate
        # constraint, so score on all of them: the fraction lying within
        # tolerance of SOME predicted ring. Radii are independent of azimuth,
        # which is why this can be solved before the azimuth offset.
        scales = np.arange(0.70, 1.1001, 0.005)
        best, all_fits = None, []
        for am in self.asin_candidates:
            for name, shape in self.coef_candidates.items():
                for p in range(self.period):
                    vza_s, _ = cfg.shift_at(p)
                    base = _predicted_radii(vza_s, shape, ps, am)
                    base = base[base > 1.0]
                    if len(base) < 4:
                        continue
                    for s in scales:
                        d = np.abs(radii[:, None] - s * base[None, :]).min(axis=1)
                        hit = float((d < RING_TOL_PX).mean())
                        cost = float(np.mean(np.minimum(d, RING_TOL_PX)))
                        all_fits.append((hit, cost, name, p, float(s), am))
                        if best is None or (hit, -cost) > (best[0], -best[1]):
                            best = (hit, cost, name, p, float(s), am)
        if best is None:
            return RetrievalStatus(False, "FAIL_NO_RINGS",
                                   "no candidate polynomial could be matched to "
                                   "the measured radii", tier=3)
        # Keep the best phases per shape, not just the single best. The VZA bins
        # are 10 deg apart and the precession is 0.9 deg/mask, so phases ~11
        # apart produce near-identical ring radii -- the radial fit genuinely
        # cannot separate them. The azimuth step below can.
        ranked = sorted(all_fits, key=lambda z: (-z[0], z[1]))
        hit, cost, name, phase, scale, asin_mode = ranked[0]

        # Parsimony: prefer the explanation with no free parameter.
        #
        # A scale factor is a fitted number with no physical referent; the
        # as-coded arcsine is a defect we can point at in flight software
        # source. The two produce ring radii that agree to ~2 px, so on residual
        # alone the fitted one can win -- which is exactly how the retracted
        # "v1 at an effective 6.47 um" conclusion was reached in the first
        # place. So if the unfitted configuration (table shape, scale 1) is
        # within UNFITTED_PREFERENCE_PX of the best fit, take it.
        def _is_unfitted(z):
            """Table shape at scale 1 -- nothing adjusted to make it fit."""
            return z[2] == "v0" and abs(z[4] - 1.0) <= UNFITTED_SCALE_TOL

        if not _is_unfitted(ranked[0]):
            unfitted = [z for z in ranked if _is_unfitted(z)]
            if unfitted and unfitted[0][0] >= hit - 0.08 \
                    and unfitted[0][1] <= cost + UNFITTED_PREFERENCE_PX:
                uh, uc, un, up, us, uam = unfitted[0]
                self._say("  bootstrap 3a  PARSIMONY: %r x %.3f / %s fits "
                          "marginally better (%.0f%% of blobs on a ring, cost "
                          "%.2f px), but %r x %.3f / %s explains the radii with "
                          "NO fitted parameter (%.0f%%, %.2f px). Taking the "
                          "unfitted one -- a scale factor has no physical "
                          "referent and a code defect does."
                          % (name, scale, asin_mode, 100 * hit, cost,
                             un, us, uam, 100 * uh, uc))
                hit, cost, name, phase, scale, asin_mode = unfitted[0]

        shortlist, seen = [(name, phase, scale, asin_mode)], {(name, phase, asin_mode)}
        for h, c_, nm, pp, sc_, am in ranked:
            if h < hit - 0.08:
                break
            if (nm, pp, am) in seen:
                continue
            seen.add((nm, pp, am))
            shortlist.append((nm, pp, sc_, am))
            if len(shortlist) >= 18:
                break
        coef = [c * scale for c in self.coef_candidates[name]]

        # ── 3b. azimuth offset, jointly with the shortlisted phases ─────────
        #
        # The offset is only identifiable MODULO the VAA bin spacing: with every
        # bin sampled, shifting by exactly one spacing maps the sample set onto
        # itself. So search one bin width, centred on zero. A result near half a
        # bin is the open "ADM azimuth is half a bin off" question -- measured
        # here rather than assumed either way.
        #
        # Scored on mean distance from each predicted sample centre to the
        # nearest observed blob centroid. A "does the centre land on a selected
        # pixel" test saturates at 1.0 -- the patches are 41 px across, so many
        # wrong configurations score perfectly -- and then ties break
        # arbitrarily. Distance does not saturate.
        # Distance is measured FROM each observed blob TO the nearest prediction,
        # not the other way round. Every detected blob is real and must be
        # explained; a predicted sample with no blob is usually just one that
        # fell off the FPA or was welded to a stripe, and averaging those in
        # swamps the statistic. Median, not mean, for the same reason.
        #
        # The range runs slightly past half a bin so the true optimum cannot sit
        # on the boundary -- an offset of +10.2 deg would otherwise be clipped to
        # the edge of a [-10, +10) window and read as something else.
        bin_w = float(np.diff(VAA_BOUNDARIES)[0])
        offsets = np.arange(-bin_w / 2 - 2.0, bin_w / 2 + 2.0, 0.2)
        best_off, best_phase, best_name, best_scale, best_d = 0.0, phase, name, scale, np.inf
        best_am = asin_mode
        for nm, p, s, am in shortlist:
            c_try = [c * s for c in self.coef_candidates[nm]]
            vz, va = cfg.shift_at(p)
            for off in offsets:
                pr, pc, _, _ = _centres_fast(vz, va + off, xc, yc, c_try, ps, am)
                inside = (pr >= 0) & (pr < self.nx) & (pc >= 0) & (pc < self.ny)
                if inside.sum() < 10:
                    continue
                dd = np.hypot(rows[:, None] - pr[inside][None, :],
                              cols[:, None] - pc[inside][None, :]).min(axis=1)
                md = float(np.median(dd))
                if md < best_d:
                    best_d, best_off = md, float(off)
                    best_phase, best_name, best_scale, best_am = p, nm, s, am
        if best_d < np.inf and (best_phase, best_name, best_am) != (phase, name, asin_mode):
            self._say("  bootstrap 3b  azimuth step moved the solution: phase "
                      "%d -> %d, shape %r -> %r, asin %r -> %r"
                      % (phase, best_phase, name, best_name, asin_mode, best_am))
        phase, name, scale, asin_mode = best_phase, best_name, best_scale, best_am
        vza_s, _ = cfg.shift_at(phase)

        # ── 3c. refine the scale by least squares, at the settled phase ─────
        base = _predicted_radii(vza_s, self.coef_candidates[name], ps, asin_mode)
        base = base[base > 1.0]
        idx = np.abs(radii[:, None] - scale * base[None, :]).argmin(axis=1)
        keep = np.abs(radii - scale * base[idx]) < RING_TOL_PX
        if keep.sum() >= 6:
            b = base[idx][keep]
            scale = float((radii[keep] * b).sum() / (b ** 2).sum())
        coef = [c * scale for c in self.coef_candidates[name]]
        rms = float(np.sqrt(np.mean((radii[keep] - scale * base[idx][keep]) ** 2))) \
            if keep.any() else float("nan")

        # ── 3d. polish on the real objective ────────────────────────────────
        #
        # Steps 3a-3c optimise cheap proxies -- blob radii and centroid
        # distances -- which get close but not close enough. The objective is
        # sharply peaked: on the 2026-07-12 frame, 0.3 deg of azimuth is worth
        # 2.6 % of recall and 0.5 % of scale is worth 1.3 %. So finish with a
        # short coordinate descent on recall itself, over the ADM layer alone
        # (the only layer these two parameters touch, and much cheaper to
        # rebuild than the full mask).
        def adm_recall(sc, off):
            from libcam.mask.adm import adm_mask
            vz, va = cfg.shift_at(phase)
            m = adm_mask(percent_samp=ps, max_radius_of_pix=cfg.max_radius_of_pix,
                         vza_shift=vz, vaa_shift=va + off,
                         x_center_pix=xc, y_center_pix=yc, nx=self.nx, ny=self.ny,
                         coef_ang2dist=[c * sc for c in self.coef_candidates[name]],
                         asin_mode=asin_mode
                         ).astype(bool)
            return _recall(m, obs)

        for width, step in ((1.6, 0.2), (0.4, 0.05)):
            offs = np.arange(best_off - width, best_off + width + 1e-9, step)
            best_off = max(offs, key=lambda o: adm_recall(scale, o))
            lo, hi = scale * (1 - width / 100), scale * (1 + width / 100)
            scs = np.linspace(lo, hi, 17)
            scale = float(max(scs, key=lambda s: adm_recall(s, best_off)))
        coef = [c * scale for c in self.coef_candidates[name]]
        polished = adm_recall(scale, best_off)
        self._say("  bootstrap 3d  polished to scale %.5f, azimuth %+.2f deg "
                  "(ADM-layer recall %.4f)" % (scale, best_off, polished))

        sol.coef = coef
        sol.coef_scale = scale
        sol.asin_mode = asin_mode
        sol.provenance["asin_mode"] = (
            "retrieved from the ring radii: %r beat the alternative over "
            "%d (shape, phase, scale) combinations" % (asin_mode, len(all_fits)))
        sol.n_frames_fitted = 1
        sol.phase_span_deg = 0.0
        if abs(best_off) > 0.05:
            sol.table_overrides["wfov_tbl_vaashift_init"] = round(
                (cfg.vaa_shift_init or 0.0) + best_off, 3)
            self._cfg_cache = (None, None)
            sol.provenance["wfov_tbl_vaashift_init"] = (
                "fitted: %+.1f deg, which brings the predicted sample centres to "
                "a mean %.1f px of the observed blob centroids. Identifiable only "
                "modulo the %.0f deg bin spacing." % (best_off, best_d, bin_w))
        sol.provenance["coef"] = (
            "bootstrap: shape %r x %.5f through the %r arcsine, rms %.2f px over "
            "%d matched blobs (shapes tried: %s; conversions tried: %s)"
            % (name, scale, asin_mode, rms, int(keep.sum()),
               ", ".join(self.coef_candidates), ", ".join(self.asin_candidates)))
        self._last_phase = phase - 1
        self._bootstrap_phase = phase
        self._say("  bootstrap 3/3  shape %r x %.5f through the %r arcsine at "
                  "phase %d, rms %.2f px; azimuth offset %+.1f deg (mean "
                  "centre-to-blob %.1f px)"
                  % (name, scale, asin_mode, phase, rms, best_off, best_d))

        # What each rejected configuration would have cost, so the choice is
        # auditable rather than asserted.
        for am in self.asin_candidates:
            for other, shape in self.coef_candidates.items():
                if (other, am) == (name, asin_mode):
                    continue
                b2 = _predicted_radii(vza_s, shape, ps, am)
                b2 = b2[b2 > 1.0]
                if len(b2) < 4:
                    continue
                hits = [float((np.abs(radii[:, None] - sc_ * b2[None, :]).min(axis=1)
                               < RING_TOL_PX).mean()) for sc_ in scales]
                j = int(np.argmax(hits))
                self._say("        rejected: %r / %s, best at x %.5f with %.0f%% "
                          "of blobs on a ring"
                          % (other, am, scales[j], 100 * hits[j]))

        # The two things that must never pass silently. Keep them separate: the
        # geometry defect is FSW's, a genuine coefficient mismatch would be ours.
        for line in sol.fsw_geometry_banner():
            self._say(line)
        if name != "v0" or abs(scale - 1.0) > UNFITTED_SCALE_TOL:
            self._say("  NOTE: the COEFFICIENTS do not match our table either. "
                      "Best fit is shape %r scaled by %.5f (table shape is 'v0' "
                      "at scale 1). This is separate from the arcsine defect "
                      "and, unlike it, is not yet explained." % (name, scale))
        return RetrievalStatus(True, "BOOTSTRAP_OK",
                               "shape %r x %.5f / %s arcsine, rms %.2f px"
                               % (name, scale, asin_mode, rms), tier=3)

    def _choose_persamp(self, obs, az=0.0):
        """Pick percent_samp by Jaccard, with the precession already solved.

        Jaccard, never recall: recall is maximised by predicting fewer samples,
        so it will always walk persamp downward. On the 2026-03-27 L1B granule
        the truth is 50 -- 8 rings of 9 blobs at exactly 40 deg azimuth spacing,
        i.e. every other 20 deg bin -- and recall-scoring chose 25.
        """
        n_bins = (len(VZA_BOUNDARIES) - 1) * (len(VAA_BOUNDARIES) - 1)
        base = dict(self.solution.table_overrides)
        cands = sorted({100, 75, 50, 40, 33, 25, 20,
                        int(self._acfg().percent_samp or 100)})
        # The shifts have to be re-solved for every candidate. percent_samp
        # selects a SUBSET of the bins -- at 50 the generator takes every other
        # azimuth -- so which half it takes depends on the shift, and a shift
        # fitted at persamp 100 (where every bin is sampled, so any shift
        # "works") does not transfer. Solving them jointly is what makes 50 win
        # on the 2026-03-27 granule; solving them in sequence picked 100.
        fams = {k: v for k, v in self.solution.enabled.items()
                if v and k in ("nadir", "forward", "cross", "raps")}
        stripes = None
        if fams:
            sm, _ = combine_masks(self._tbl(), fams, az=az, frame=0,
                                  nx=self.nx, ny=self.ny)
            stripes = sm.astype(bool)

        best = (None, -1.0, -1.0, None)
        for cand in cands:
            trial = dict(base)
            trial["wfov_tbl_persamp"] = cand
            self.solution.table_overrides = trial
            self._cfg_cache = (None, None)
            vz, va, _sc = self.solve_shifts(obs, stripes=stripes)
            if vz is not None:
                trial["wfov_tbl_vzashift_init"] = float(vz)
                trial["wfov_tbl_vaashift_init"] = float(va)
                self.solution.table_overrides = trial
                self._cfg_cache = (None, None)
            m, _ = self.build(0, az)
            mb = m.astype(bool)
            jac = _jaccard(mb, obs)
            if jac > best[1]:
                best = (cand, jac, _recall(mb, obs), (vz, va))
        base["wfov_tbl_persamp"] = best[0]
        if best[3] and best[3][0] is not None:
            base["wfov_tbl_vzashift_init"] = float(best[3][0])
            base["wfov_tbl_vaashift_init"] = float(best[3][1])
        self.solution.table_overrides = base
        self._cfg_cache = (None, None)
        self.solution.provenance["wfov_tbl_persamp"] = (
            "chosen by Jaccard over %s with the measured precession -> %d "
            "(jaccard %.3f, recall %.3f) of %d angular bins"
            % (cands, best[0], best[1], best[2], n_bins))
        self._say("  percent_samp retrieved: %d (jaccard %.3f)" % (best[0], best[1]))
        return best[0]

    # ── the centre, once a later frame can actually pin it ──────────────────

    MIN_STRIPE_SEPARATION_DEG = 20.0

    def retry_centre(self, obs, az):
        """Re-fit the stripe centre on a frame where the stripes are not parallel.

        Only called when the bootstrap frame could not pin an axis. The centre of
        the mask is the intersection of two stripes, so it needs two stripes that
        actually cross. In cross-track that is free -- nadir and cross-track are
        90 deg apart by construction. In RAPS the only two drawn are nadir and
        the rotating stripe, which start on top of each other; the intersection
        becomes well conditioned only once the mechanism has turned, and the
        angle between them is what decides when.

        Returns True when the centre was re-fitted and frozen.
        """
        sol = self.solution
        if not sol.centre_pending:
            return False
        sep = abs((az + 90.0) % 180.0 - 90.0)   # angle to parallel, 0..90
        if sep < self.MIN_STRIPE_SEPARATION_DEG:
            return False
        st = self.bootstrap_centre(obs, az)
        if st is None:
            return False
        sol.centre_pending = False
        sol.provenance["centre"] = (
            "re-fitted on frame where the rotating stripe is %.0f deg from the "
            "nadir stripe -- the bootstrap frame had them parallel, so the "
            "centre was not identifiable there. %s" % (sep, st))
        self._say("  centre re-fitted with the stripes %.0f deg apart -> "
                  "(row %s, col %s)"
                  % (sep, self._acfg().x_center_pix, self._acfg().y_center_pix))
        return True

    # ── mechanism azimuth retrieved from the rotating stripe ────────────────

    def solve_az(self, obs, fallback=0.0):
        """Retrieve the mechanism azimuth from the rotating stripe itself.

        The RAPS stripe carries ``az_rot``, so its rotation angle *is* the
        mechanism azimuth and the frame shows it directly. The granule is
        supposed to say so in its ``Azimuth`` field -- and in
        ``CAM 20260327`` that field is **identically zero on all 44 frames**
        while the stripe visibly sweeps. Holding the model at az 0 left two
        unexplained slabs of ~38 000 px each by frame 11 and dropped coverage
        from 0.94 to 0.67 while recall stayed at 0.96, which is exactly the
        signature of a component the model has stopped predicting at all.

        So it is measured. The score is the recall of the rotating stripe's own
        layer -- what fraction of the pixels it predicts are actually selected.
        That is near 1 at the true angle and falls away immediately, because a
        stripe at the wrong angle runs across unselected FPA. Jaccard against
        the whole frame would be diluted by the ADM samples and the fixed
        stripes, so only this one layer is scored.

        Returns ``(az, score)``, or ``(fallback, nan)`` if there is no rotating
        stripe to measure.
        """
        rot = [k for k in ("nadir", "forward", "cross", "raps")
               if self.solution.enabled.get(k)
               and stripe_cfg(self._tbl(), k).az_rot]
        if not rot:
            return float(fallback), float("nan")

        obs = np.asarray(obs, dtype=bool)
        tbl = self._tbl()

        # Score only against what the FIXED stripes do not already explain.
        # Scoring against the raw frame picks az 0 every time: there the RAPS
        # stripe lies exactly on top of the nadir stripe, so it scores a perfect
        # recall without predicting a single new pixel, and the real rotated
        # slab is left unexplained. Against the residual, az 0 contributes
        # nothing and has to win on its merits -- which, when the two stripes
        # really are coincident, it cannot, so that case is detected below and
        # handed back to the granule's own value rather than guessed at.
        fixed = {k: True for k in ("nadir", "forward", "cross", "raps")
                 if self.solution.enabled.get(k) and k not in rot}
        fx = np.zeros((self.nx, self.ny), dtype=bool)
        if fixed:
            m, _ = combine_masks(tbl, fixed, az=0.0, frame=0,
                                 nx=self.nx, ny=self.ny)
            fx = m.astype(bool)
        resid = obs & ~fx

        # Coarse pass on the generator LINE rather than the rendered band.
        # Rendering a 2048x2048 stripe at every candidate angle costs ~30 ms, so
        # a 1-degree sweep of the full turn would be 11 s per frame. Sampling the
        # centre line is a few hundred array lookups and is just as sharp: the
        # band is 41 px wide, so at the far end of a 1000 px stripe an error of
        # 1 degree already walks the line off its own band.
        samples = []
        for fam in rot:
            c = stripe_cfg(tbl, fam)
            t = np.linspace(-float(c.fr), float(c.fr), 600)
            samples.append((float(c.xc), float(c.yc), float(c.sy),
                            float(c.sr or 0.0), t))

        def line_score(a):
            hit = tot = 0
            for xc, yc, sy, sr, t in samples:
                th = np.deg2rad(sr + a)
                ct, st_ = np.cos(th), np.sin(th)
                xr = np.rint(ct * t - st_ * sy + xc).astype(int)
                yr = np.rint(st_ * t + ct * sy + yc).astype(int)
                ok = (xr >= 0) & (xr < self.nx) & (yr >= 0) & (yr < self.ny)
                ok &= ~fx[np.clip(xr, 0, self.nx - 1), np.clip(yr, 0, self.ny - 1)]
                if ok.sum() < 100:          # the line is under a fixed stripe
                    continue
                hit += int(resid[xr[ok], yr[ok]].sum())
                tot += int(ok.sum())
            return hit / tot if tot else 0.0

        sweep = np.arange(0.0, 360.0, 0.5)
        curve = np.array([line_score(a) for a in sweep])
        best_a = float(sweep[int(np.argmax(curve))])
        peak = float(curve.max())
        floor = float(np.median(curve))

        # Accept on PROMINENCE, not on an absolute score. The absolute score
        # falls away as the stripe rotates -- 0.58 at frame 12, 0.40 by frame 14
        # -- because the stripe drawn in the data is clipped where the table's
        # rectangle is not, so a growing part of what the model predicts was
        # never going to be selected. (Worth chasing separately: it means the
        # flown stripe length or its FOV clipping does not match the table.) The
        # peak stays obvious throughout, so what is tested is that there IS a
        # peak: a flat curve means either no rotating stripe is being drawn or
        # it is coincident with a fixed one, and in both cases the angle is
        # unmeasurable AND does not change the mask, so the granule's own value
        # stands.
        if peak < 0.20 or peak < 2.5 * max(floor, 1e-6):
            return float(fallback), float("nan")

        # Fine pass on the real rendered layer -- the thing actually being fitted.
        enabled = {k: True for k in rot}

        def score(a):
            m, _ = combine_masks(tbl, enabled, az=float(a), frame=0,
                                 nx=self.nx, ny=self.ny)
            lay = m.astype(bool) & ~fx
            n = int(lay.sum())
            if n < 2000:
                return 0.0
            return float((lay & resid).sum()) / n

        best_a = float(max(np.arange(best_a - 1.0, best_a + 1.0 + 1e-9, 0.2),
                           key=score))
        return best_a % 360.0, score(best_a)

    # ── shifts retrieved from the frame, not taken from the table ───────────

    def solve_shifts(self, obs, stripes=None, near=None):
        """Retrieve (vza_shift, vaa_shift) for this frame directly.

        ``near`` is an optional ``(vza, vaa)`` the precession predicts for this
        frame; when given, the search runs in a narrow window about it instead
        of over the whole range. See :meth:`_predict_shifts`.

        The phase-plus-step approach assumed the table's ``vzashift`` and
        ``vaashift`` were the ones in force. They were not. The 2026-03-27 L1B
        granule precesses at **9 deg/mask in VZA and 7 deg/mask in VAA** -- the
        ``adm_1d1`` values -- while our CSV and the 2026-07-12 L1A data use
        0.9/0.9 from ``adm_1d0``. Assuming 0.9 put the ADM Jaccard at 0.34; the
        right shifts give 0.82 on the same frame.

        So the shifts are measured, not assumed, and the per-mask step becomes a
        *result* (the frame-to-frame difference) rather than an input. Two cheap
        one-dimensional searches, because the two angles separate cleanly:

          vza_shift  from the ring RADII    -- independent of azimuth
          vaa_shift  from the blob AZIMUTHS -- at the vza_shift just found

        Returns ``(vza_shift, vaa_shift, score)`` with score the fraction of
        observed blobs lying within RING_TOL_PX of a predicted sample centre.
        """
        cfg = self._acfg()
        ps = cfg.percent_samp
        xc, yc = self._adm_xy()
        rows, cols = measure_blob_centroids(
            obs, area=blob_area(cfg.max_radius_of_pix), exclude=stripes)
        if len(rows) < 6:
            return None, None, 0.0
        radii = np.hypot(rows - xc, cols - yc)

        centres = VZA_BOUNDARIES[:-1] + 0.5 * np.diff(VZA_BOUNDARIES)
        vza_i, _ = _bin_indices(ps)
        centres = centres[np.unique(vza_i)]

        vz_grid = np.arange(0.0, 90.0, 0.25)
        va_grid = np.arange(0.0, 360.0, 0.5)
        if near is not None:
            # +/-2 deg is wide enough to absorb a frame the precession fit does
            # not quite describe, and narrow enough that a bin-width jump (10 deg
            # in VZA, 20 in VAA) cannot be reached.
            vz_grid = (np.arange(near[0] - 2.0, near[0] + 2.0001, 0.1)) % 90.0
            va_grid = (np.arange(near[1] - 2.0, near[1] + 2.0001, 0.1)) % 360.0

        best_vz, best_s = float(vz_grid[0]), -1.0
        for vz in vz_grid:
            pred = _poly_radius_px((centres + vz) % 90, self.solution.coef,
                                   self.solution.asin_mode)
            pred = pred[pred > 1.0]
            if not len(pred):
                continue
            d = np.abs(radii[:, None] - pred[None, :]).min(axis=1)
            sc = float((d < RING_TOL_PX).mean()) - 0.001 * float(np.mean(d))
            if sc > best_s:
                best_s, best_vz = sc, float(vz)

        best_va, best_d = float(va_grid[0]), np.inf
        for va in va_grid:
            pr, pc, _, _ = _centres_fast(best_vz, va, xc, yc,
                                         self.solution.coef, ps,
                                         self.solution.asin_mode)
            inside = (pr >= 0) & (pr < self.nx) & (pc >= 0) & (pc < self.ny)
            if inside.sum() < 5:
                continue
            dd = np.hypot(rows[:, None] - pr[inside][None, :],
                          cols[:, None] - pc[inside][None, :]).min(axis=1)
            md = float(np.median(dd))
            if md < best_d:
                best_d, best_va = md, float(va)

        # ── polish on the blob-to-centre distance, jointly ──────────────────
        #
        # The two one-dimensional searches above are run on a coarse grid and
        # each is blind to the other, so the pair they return can sit a bin
        # boundary away from the true one. On 2025-11-18 frame 10 that put every
        # predicted sample ~25 px from its blob -- just past the 20 px patch plus
        # the 3 px reunite tolerance -- so 58 whole samples came out UNEXPLAINED
        # and the frame failed the coverage gate, with nothing visibly wrong in
        # the picture except that the labels had slipped off their blobs. A short
        # joint refinement on the real objective fixes it.
        def centre_dist(vz, va):
            pr, pc, _, _ = _centres_fast(vz, va, xc, yc, self.solution.coef, ps,
                                         self.solution.asin_mode)
            ok = (pr >= 0) & (pr < self.nx) & (pc >= 0) & (pc < self.ny)
            if ok.sum() < 5:
                return np.inf
            return float(np.median(np.hypot(rows[:, None] - pr[ok][None, :],
                                            cols[:, None] - pc[ok][None, :]
                                            ).min(axis=1)))

        for span, step in ((1.0, 0.1), (0.2, 0.02)):
            best_vz = float(min(np.arange(best_vz - span, best_vz + span + 1e-9, step),
                                key=lambda v: centre_dist(v, best_va))) % 90.0
            best_va = float(min(np.arange(best_va - span, best_va + span + 1e-9, step),
                                key=lambda a: centre_dist(best_vz, a))) % 360.0

        pr, pc, _, _ = _centres_fast(best_vz, best_va, xc, yc,
                                     self.solution.coef, ps,
                                     self.solution.asin_mode)
        inside = (pr >= 0) & (pr < self.nx) & (pc >= 0) & (pc < self.ny)
        dd = np.hypot(rows[:, None] - pr[inside][None, :],
                      cols[:, None] - pc[inside][None, :]).min(axis=1)
        return best_vz, best_va, float((dd < RING_TOL_PX).mean())

    # ── the precession step, as a retrieved quantity ────────────────────────

    def _record_shifts(self, index, vz, va):
        """Remember this frame's shifts and re-estimate the per-mask step.

        The step is not an input here -- the table's ``vzashift``/``vaashift``
        were wrong on every granule we have looked at. It is the frame-to-frame
        difference of the measured shifts, taken modulo the grid period because
        the shift wraps (VZA at 90, VAA at 360) long before the granule ends.

        Median of the differences, not a least-squares line: one slipped frame
        should not drag the step, and early on there are only a handful of
        samples.
        """
        self._shift_hist.append((int(index), float(vz), float(va)))
        self._shift_hist = self._shift_hist[-40:]
        if len(self._shift_hist) < 4:
            return
        idx = np.array([h[0] for h in self._shift_hist], dtype=float)
        out = []
        for col, period in ((1, 90.0), (2, 360.0)):
            val = np.array([h[col] for h in self._shift_hist], dtype=float)
            d_i = np.diff(idx)
            # Shortest way round, so a step that runs DOWN reads as negative
            # rather than as (period - step). On 2026-03-27 the VZA shift goes
            # 9, 8, 7, 6 ... -- that is -1 deg/mask, not +89, and -1 is also what
            # makes it recognisable as the same family as the table's +9 (the
            # two differ by exactly one 10 deg VZA bin).
            d_v = (np.diff(val) + period / 2.0) % period - period / 2.0
            keep = d_i > 0
            out.append(float(np.median(d_v[keep] / d_i[keep])) if keep.any() else 0.0)
        self.solution.step_vza, self.solution.step_vaa = out[0], out[1]
        self.solution.provenance["precession"] = (
            "measured from the frame-to-frame change of the retrieved shifts "
            "over %d frames: %.3f deg/mask in VZA, %.3f deg/mask in VAA "
            "(the table says %.3f / %.3f)"
            % (len(self._shift_hist), out[0], out[1],
               adm_cfg(self.tbl0).vza_shift or 0.0,
               adm_cfg(self.tbl0).vaa_shift or 0.0))

    def _predict_shifts(self, index):
        """Where the precession says this frame's shifts should be, or None.

        Used to keep the per-frame search near the track instead of re-solving
        globally every time. A global search is free to jump a bin -- which is
        exactly what it did on 2025-11-18 frames 10 and 11 -- and a jump is
        indistinguishable from a real wrap unless continuity is consulted.
        """
        if self.solution is None or self.solution.step_vza is None:
            return None
        if len(self._shift_hist) < 4:
            return None
        i0, vz0, va0 = self._shift_hist[-1]
        n = int(index) - i0
        if n <= 0 or n > 8:          # too far ahead to extrapolate safely
            return None
        return ((vz0 + n * self.solution.step_vza) % 90.0,
                (va0 + n * self.solution.step_vaa) % 360.0)

    # ── tier 1: phase ────────────────────────────────────────────────────────

    def solve_phase(self, obs, index=None, window=None):
        """Find the mask counter index that best explains this frame.

        Scores candidate phases by the fraction of predicted ADM sample centres
        that land on a selected pixel.

        A caution about what is identifiable. The VZA bins are 10 deg apart and
        the precession is ~0.9 deg per mask, so the ring configuration very
        nearly repeats every ~11 frames; the VAA bins are 20 deg apart, so the
        azimuth pattern nearly repeats every ~22. A single frame therefore
        pins the phase only up to those near-degeneracies -- the only thing
        breaking them is the wrap at 90 deg, where a new innermost ring appears.

        So the phase is carried by **continuity**, not re-derived per frame:
        once anchored, it advances one per frame and is verified. A full search
        runs only at the anchor, or when verification fails.
        """
        def score(p):
            r, c, _, _ = self._adm_centres(p)
            inside = (r >= 0) & (r < self.nx) & (c >= 0) & (c < self.ny)
            if not inside.any():
                return 0.0, 0
            rr, cc = r[inside].astype(int), c[inside].astype(int)
            return float(obs[rr, cc].mean()), int(inside.sum())

        if window is not None:
            cands = [w % self.period for w in window]
        elif self._anchor is not None and index is not None:
            a_idx, a_ph = self._anchor
            expect = (a_ph + (index - a_idx)) % self.period
            cands = [(expect + d) % self.period for d in (0, 1, -1, 2, -2)]
        else:
            cands = list(range(self.period))

        best, best_s, best_n = None, -1.0, 0
        for p in cands:
            s, n = score(p)
            if s > best_s:
                best, best_s, best_n = p, s, n

        # The continuity guess has to be convincing; otherwise search everything.
        if best_s < 0.7 and len(cands) < self.period:
            self._say("  frame %s: continuity guess scored %.2f -- full search"
                      % (index, best_s))
            for p in range(self.period):
                s, n = score(p)
                if s > best_s:
                    best, best_s, best_n = p, s, n
        return best, best_s, best_n

    # ── tier 2: table parameters ─────────────────────────────────────────────

    def refit_table(self, obs, phase, az=0.0):
        """Refit mask centre, maxPixR and persamp from one frame.

        Centre is refined by a small local search on recall; ``maxPixR`` comes
        from the median blob area and ``persamp`` from the blob count, both of
        which are direct measurements rather than fits.
        """
        tbl = self._tbl()
        over = dict(self.solution.table_overrides)
        prov = {}

        # persamp and maxPixR, measured from the blobs.
        pred_stripes = np.zeros((self.nx, self.ny), dtype=bool)
        for name, on in self.solution.enabled.items():
            if on and name in ("nadir", "forward", "cross", "raps"):
                m, cnt = combine_masks(tbl, {name: True}, az=az, frame=phase,
                                       nx=self.nx, ny=self.ny)
                pred_stripes |= m.astype(bool)
        rows, cols = measure_blob_centroids(obs, exclude=pred_stripes)
        n_blob = len(rows)
        if n_blob:
            try:
                from scipy import ndimage
                m = np.asarray(obs, bool) & ~pred_stripes
                lab, n = ndimage.label(m)
                size = ndimage.sum(m, lab, range(1, n + 1))
                size = size[(size > BLOB_AREA[0]) & (size < BLOB_AREA[1])]
                if len(size):
                    r_fit = float(np.sqrt(np.median(size) / np.pi))
                    over["wfov_tbl_maxPixR"] = int(round(r_fit))
                    prov["wfov_tbl_maxPixR"] = (
                        "fitted: median blob %d px -> r=%.2f" % (np.median(size), r_fit))
            except ImportError:
                pass
        # persamp is NOT fitted from the blob count. The count is a lower bound --
        # samples fall off the FPA at high VZA, and any blob welded to a stripe
        # is lost to the exclusion -- so dividing it by 162 reads 64 % where the
        # truth is 100 %, and the model then under-predicts by a third. Score a
        # few candidate values on recall instead, which cannot be fooled that way.
        n_bins = (len(VZA_BOUNDARIES) - 1) * (len(VAA_BOUNDARIES) - 1)
        # Candidates are derived from the blob count, not a fixed shortlist.
        # The 2026-03-27 L1B granule runs at roughly 36 % -- 59 blobs in 7 rings
        # of 7-9 rather than 9 rings of 18 -- which {100, 50, 25} simply cannot
        # express, and the retrieval then fails on every frame. The blob count is
        # a lower bound (samples fall off the FPA, and any blob welded to a
        # stripe is lost to the exclusion), so search from it upward.
        cands = {100, 50, 25, int(self._acfg().percent_samp or 100)}
        if n_blob:
            lo = max(1, int(round(100.0 * n_blob / n_bins)))
            cands.update(range(lo, min(lo + 14, 101), 2))
        cands = sorted(c for c in cands if 1 <= c <= 100)
        if len(cands) > 1:
            # Scored by JACCARD, not recall. Recall is maximised by predicting
            # less -- dropping persamp from 100 to 25 removes three quarters of
            # the ADM samples and every one that remains still lands on selected
            # pixels, so recall goes UP while coverage collapses. On the
            # 2026-03-27 L1B granule that chose persamp 25 at recall 0.88 and
            # left 27 % of the mask unexplained. Jaccard cannot be gamed that
            # way because it counts the pixels the model stopped predicting.
            best_ps = (None, -1.0, -1.0)
            for cand in cands:
                trial = dict(over)
                trial["wfov_tbl_persamp"] = cand
                self.solution.table_overrides = trial
                self._cfg_cache = (None, None)
                m, _ = self.build(phase, az)
                mb = m.astype(bool)
                jac = _jaccard(mb, obs)
                if jac > best_ps[1]:
                    best_ps = (cand, jac, _recall(mb, obs))
            over["wfov_tbl_persamp"] = best_ps[0]
            prov["wfov_tbl_persamp"] = (
                "chosen by Jaccard over %s -> %d (jaccard %.3f, recall %.3f); "
                "%d blobs detected, a lower bound on the %d bins"
                % (cands, best_ps[0], best_ps[1], best_ps[2], n_blob, n_bins))

        self.solution.table_overrides = over
        self.solution.provenance.update(prov)

        # Centre: local search, recall as the objective.
        #
        # The centre lives in the stripe table, once per family, and the ADM
        # table has none of its own -- ADM_MAP points it at the nadir stripe's.
        # So moving "the mask centre" means writing every family's pair plus the
        # limb's, or the components come apart from each other.
        best = (None, -1.0)
        base_r = int(adm_cfg(self._tbl()).x_center_pix)
        base_c = int(adm_cfg(self._tbl()).y_center_pix)
        for dr in range(-12, 13, 4):
            for dc in range(-12, 13, 4):
                trial = dict(over)
                for fam in ("nadir", "forward", "cross", "raps"):
                    trial["wfov_tbl_xcenter_stripe_%s" % fam] = base_r + dr
                    trial["wfov_tbl_ycenter_stripe_%s" % fam] = base_c + dc
                trial["wfov_tbl_xcenter_limb"] = base_r + dr
                trial["wfov_tbl_ycenter_limb"] = base_c + dc
                self.solution.table_overrides = trial
                mask, _ = self.build(phase, az)
                rec = _recall(mask.astype(bool), obs)
                if rec > best[1]:
                    best = (trial, rec)
        if best[0] is not None:
            self.solution.table_overrides = best[0]
            self.solution.provenance["centre"] = (
                "fitted by local search, recall %.3f" % best[1])
        return best[1]

    # ── tier 3: polynomial ───────────────────────────────────────────────────

    def _phase_span(self):
        if len(self._buffer) < 2:
            return 0.0
        ph = np.array([b[0] for b in self._buffer], dtype=float)
        return float((ph.max() - ph.min()) * self.step_deg)

    def refit_polynomial(self, full=False):
        """Refit the field-angle polynomial from the buffered frames.

        ``full=False`` fits a single scale factor onto the existing coefficient
        shape -- this is what recovered the flown geometry on 2026-10-03.
        ``full=True`` fits all coefficients and is gated hard; see the module
        docstring for why one frame can never be enough.

        Returns a :class:`RetrievalStatus`. Nothing is changed on failure.
        """
        n = len(self._buffer)
        span = self._phase_span()
        if full:
            if n < MIN_FRAMES_POLY_FULL or span < MIN_PHASE_SPAN_DEG:
                return RetrievalStatus(
                    False, "FAIL_INSUFFICIENT_FRAMES",
                    "full polynomial fit needs >= %d frames spanning >= %.0f deg "
                    "of phase; have %d frames spanning %.1f deg"
                    % (MIN_FRAMES_POLY_FULL, MIN_PHASE_SPAN_DEG, n, span), tier=3)
        elif n < MIN_FRAMES_POLY_SCALE:
            return RetrievalStatus(
                False, "FAIL_INSUFFICIENT_FRAMES",
                "scale fit needs >= %d frames, have %d" % (MIN_FRAMES_POLY_SCALE, n),
                tier=3)

        cfg = adm_cfg(self._tbl())
        xc, yc = self._adm_xy()
        vza_all, rad_all = [], []
        for phase, rows, cols in self._buffer:
            if not len(rows):
                continue
            meas = np.hypot(rows - xc, cols - yc)
            vza_s, _ = adm_cfg(self._tbl()).shift_at(phase)
            vza_pred, rad_pred = ring_radii_for_phase(
                vza_s, cfg.percent_samp, self.solution.coef,
                self.solution.asin_mode)
            for v, rp in zip(vza_pred, rad_pred):
                sel = np.abs(meas - rp) < RING_TOL_PX
                if sel.sum() >= 3:          # a ring, not a stray blob
                    vza_all.append(v)
                    rad_all.append(float(np.median(meas[sel])))
        if len(vza_all) < 6:
            return RetrievalStatus(
                False, "FAIL_NO_RINGS",
                "only %d ring measurements associated; cannot fit" % len(vza_all),
                tier=3)

        vza_all = np.array(vza_all)
        rad_all = np.array(rad_all)

        if not full:
            am = self.solution.asin_mode
            base = _poly_radius_px(vza_all, COEF_ANG2DIST, am)
            good = base > 1.0
            s = float((rad_all[good] * base[good]).sum() / (base[good] ** 2).sum())
            coef = [c * s for c in COEF_ANG2DIST]
            rms = float(np.sqrt(np.mean(
                (rad_all - _poly_radius_px(vza_all, coef, am)) ** 2)))
            self.solution.coef = coef
            self.solution.coef_scale = s
            self.solution.n_frames_fitted = n
            self.solution.phase_span_deg = span
            self.solution.provenance["coef"] = (
                "fitted scale %.5f on %d ring measurements from %d frames "
                "spanning %.1f deg through the %r arcsine, rms %.2f px"
                % (s, len(vza_all), n, span, am, rms))
            return RetrievalStatus(True, "REFIT_POLY_SCALE",
                                   "scale %.5f, rms %.2f px" % (s, rms), tier=3)

        am = self.solution.asin_mode
        t = _vza_sat(vza_all, am)
        order = len(COEF_ANG2DIST)
        A = np.vstack([t ** (order - i - 1) for i in range(order)]).T
        coef, *_ = np.linalg.lstsq(A * (1.0 / PIX_SPACING_MM), rad_all, rcond=None)
        coef = list(coef)
        rms = float(np.sqrt(np.mean(
            (rad_all - _poly_radius_px(vza_all, coef, am)) ** 2)))
        self.solution.coef = coef
        self.solution.coef_scale = float("nan")
        self.solution.n_frames_fitted = n
        self.solution.phase_span_deg = span
        self.solution.provenance["coef"] = (
            "fitted all %d coefficients on %d ring measurements from %d frames "
            "spanning %.1f deg, rms %.2f px" % (order, len(vza_all), n, span, rms))
        return RetrievalStatus(True, "REFIT_POLY_FULL", "rms %.2f px" % rms, tier=3)

    # ── the per-frame entry point ────────────────────────────────────────────

    def process_frame(self, obs, header=None, index=0, az=0.0):
        """Retrieve the mask for one frame and label every selected pixel.

        ``obs`` is a boolean raster of selected pixels -- for L1A, the decoded
        image masked to non-zero; for L1B, the pixel-mask field. ``header``
        carries ``cam_obs_id`` and ``pixel_mask_id``.
        """
        obs = np.asarray(obs, dtype=bool)
        header = header or {}
        res = FrameResult(index=index, az=az, az_header=az,
                          observed=int(obs.sum()))

        bad = self._ensure_solution(header)
        if bad is not None:
            res.status = bad
            res.labels = np.where(obs, UNEXPLAINED, 0).astype(np.uint8)
            res.membership = np.where(obs, BIT_UNEXPLAINED, 0).astype(np.uint16)
            res.unexplained = res.observed
            return res

        # mechanism azimuth, per frame ----------------------------------------
        #
        # Before anything else that uses it: the bootstrap fits the centre on
        # stripe recall, and every later step subtracts the stripes before
        # looking for ADM blobs. A rotating stripe drawn at the wrong angle both
        # scores badly and fails to be subtracted, so its pixels are mistaken
        # for ADM samples.
        if self.measure_az:
            a_fit, a_sc = self.solve_az(obs, fallback=az)
            if a_sc == a_sc:                # not NaN -> there is one to measure
                res.az_fitted = a_fit
                if abs((a_fit - az + 180.0) % 360.0 - 180.0) > 0.5:
                    self._say("  frame %d: mechanism azimuth measured at %.1f "
                              "deg from the rotating stripe (its recall %.4f); "
                              "the granule's Azimuth field says %.1f"
                              % (index, a_fit, a_sc, az))
                    self.solution.provenance["az"] = (
                        "measured from the rotating stripe, not taken from the "
                        "granule's Azimuth field, which disagreed (frame %d: "
                        "%.1f stated vs %.1f measured)" % (index, az, a_fit))
                az = a_fit
                res.az = az
            self._az_hist.append((int(index), float(res.az_header or 0.0),
                                  a_fit if a_sc == a_sc else None,
                                  a_sc if a_sc == a_sc else float("nan")))

        # bootstrap, once per solution ---------------------------------------
        if "coef" not in self.solution.provenance or \
                self.solution.provenance["coef"].startswith("table default"):
            self._say("  frame %d: bootstrapping the solution" % index)
            st = self.bootstrap(obs, az)
            if not st.ok:
                res.status = st
                res.labels = np.where(obs, UNEXPLAINED, 0).astype(np.uint8)
                res.unexplained = res.observed
                return res
            # Anchor on the bootstrap's phase. It was found by a global search
            # over every shape, scale and phase and then polished; re-deriving it
            # from the cheap per-frame proxy would throw that away, and the
            # phases are near-degenerate enough (see solve_phase) that the proxy
            # can prefer a slightly worse one.
            if self._bootstrap_phase is not None:
                self._anchor = (index, self._bootstrap_phase)

        # centre, if the bootstrap frame could not pin it ---------------------
        if self.solution.centre_pending:
            if self.retry_centre(obs, az):
                fams = {k: v for k, v in self.solution.enabled.items()
                        if v and k in ("nadir", "forward", "cross", "raps")}
                sm, _ = combine_masks(self._tbl(), fams, az=az, frame=0,
                                      nx=self.nx, ny=self.ny)
                # The ADM centre was judged against the old stripe centre, so it
                # has to be judged again against the new one.
                self.solution.adm_centre = None
                self.fit_adm_centre(obs, sm.astype(bool))
                self._shift_hist = []      # the track was fitted on a wrong centre

        # tier 1 ------------------------------------------------------------
        if self.measure_shifts:
            # Measure the precession directly rather than assuming the table's
            # per-mask step. ``phase`` then just records the frame index; the
            # shifts themselves are the retrieved quantity, and their
            # frame-to-frame difference IS the per-mask step.
            fams = {k: v for k, v in self.solution.enabled.items()
                    if v and k in ("nadir", "forward", "cross", "raps")}
            stripes = None
            if fams:
                sm, _ = combine_masks(self._tbl(), fams, az=az, frame=0,
                                      nx=self.nx, ny=self.ny)
                stripes = sm.astype(bool)
            near = self._predict_shifts(index)
            vz, va, sc = self.solve_shifts(obs, stripes=stripes, near=near)
            if near is not None and (vz is None or sc < 0.75):
                # The narrow search failed. Fall back to the global one rather
                # than carry a bad answer forward -- a real mode or table change
                # breaks continuity legitimately.
                vz, va, sc = self.solve_shifts(obs, stripes=stripes)
                self._say("  frame %d: shifts off the predicted track, re-solved "
                          "globally" % index)
            if vz is None:
                phase = 0
            else:
                self.solution.table_overrides["wfov_tbl_vzashift_init"] = float(vz)
                self.solution.table_overrides["wfov_tbl_vaashift_init"] = float(va)
                self.solution.table_overrides["wfov_tbl_vzashift"] = 0.0
                self.solution.table_overrides["wfov_tbl_vaashift"] = 0.0
                self._cfg_cache = (None, None)
                phase = 0
                res.measured_shifts = (float(vz), float(va), float(sc))
                self._record_shifts(index, vz, va)
                # With the shifts right, persamp can finally be chosen honestly.
                # It has to be done here rather than in the tier-2 refit,
                # because the measured shifts short-circuit that path -- and
                # scoring persamp against a wrong precession is what made it
                # pick 25 on this granule in the first place.
                if not self._persamp_fixed:
                    self._choose_persamp(obs, az)
                    self._persamp_fixed = True
        else:
            phase, centre_hit, n_in = self.solve_phase(obs, index=index)
        res.phase = phase
        mask, counts = self.build(phase, az)
        mask = mask.astype(bool)
        rec = _recall(mask, obs)
        jac = _jaccard(mask, obs)
        res.vza_shift = counts["adm_vza_shift"]
        res.vaa_shift = counts["adm_vaa_shift"]
        tier = 1

        # tier 2 ------------------------------------------------------------
        good_shift = (res.measured_shifts is not None
                      and res.measured_shifts[2] > 0.9)
        if rec < self.good_recall and not good_shift and (
                not self._table_fitted or rec < self.min_recall):
            self._say("  frame %d: recall %.3f below %.2f -- refitting table"
                      % (index, rec, self.good_recall))
            self._table_fitted = True
            self.refit_table(obs, phase, az)
            phase, centre_hit, n_in = self.solve_phase(obs, index=index,
                                                       window=range(self.period))
            res.phase = phase
            mask, counts = self.build(phase, az)
            mask = mask.astype(bool)
            rec, jac = _recall(mask, obs), _jaccard(mask, obs)
            tier = 2

        # tier 3 ------------------------------------------------------------
        #
        # Guard: never refit the polynomial while the STRIPES still reproduce.
        # Stripe geometry is purely table-driven -- no polynomial, no phase -- so
        # a high stripe recall is independent evidence that the centre and the
        # field-angle mapping are right and the problem lies in the ADM
        # sampling. Without this guard the 2026-03-27 L1B granule, whose ADM
        # pattern simply does not fit the 9x18 bin grid, dragged the scale from
        # 0.846 to 0.760 and destroyed the stripe fit as well: explained fell
        # from 209k pixels to 90k.
        stripe_rec = float("nan")
        fams = {k: v for k, v in self.solution.enabled.items()
                if v and k in ("nadir", "forward", "cross", "raps")}
        if fams:
            sm, _ = combine_masks(self._tbl(), fams, az=az, frame=phase,
                                  nx=self.nx, ny=self.ny)
            stripe_rec = _recall(sm.astype(bool), obs)
        if good_shift and rec < self.min_recall:
            self._say("  frame %d: %.0f %% of blobs sit on a predicted sample "
                      "centre, so the geometry is right -- not refitting"
                      % (index, 100 * res.measured_shifts[2]))
        elif rec < self.min_recall and self.allow_poly_fit and stripe_rec > 0.95:
            self._say("  frame %d: recall %.3f is low but the stripes reproduce "
                      "at %.4f -- the geometry is right, so NOT refitting the "
                      "polynomial. The ADM sampling is what does not match."
                      % (index, rec, stripe_rec))
        elif rec < self.min_recall and self.allow_poly_fit:
            rows, cols = measure_blob_centroids(obs)
            self._buffer.append((phase, rows, cols))
            st = self.refit_polynomial(full=False)
            self._say("  frame %d: %s" % (index, st))
            tier = 3
            if st.ok:
                mask, counts = self.build(phase, az)
                mask = mask.astype(bool)
                rec, jac = _recall(mask, obs), _jaccard(mask, obs)
            else:
                res.status = RetrievalStatus(False, st.code, st.message, rec, jac,
                                            float("nan"), 3)
                res.labels = np.where(obs, UNEXPLAINED, 0).astype(np.uint8)
                res.unexplained = res.observed
                return res

        # verdict -----------------------------------------------------------
        #
        # Recall alone is not enough. It measures only the predicted pixels, so
        # a model that has stopped predicting most of the mask can still score
        # 0.92 while explaining 60 % of what is actually there. Coverage --
        # observed pixels that a component accounts for -- is the other half,
        # and both have to pass.
        cov = float((mask & obs).sum()) / res.observed if res.observed else float("nan")
        low_cov = (cov == cov) and cov < self.min_coverage

        # A low-coverage frame is NOT wiped. On the 2026-03-27 L1B granule the
        # stripes reproduce at recall 0.9998 while the ADM pattern does not match
        # the 9x18 bin grid at any percent_samp -- marking every pixel
        # unexplained would throw away three quarters of the mask that is
        # perfectly well understood. The frame is flagged, the components that
        # did match keep their labels, and only the genuinely unattributable
        # pixels carry UNEXPLAINED. Downstream, gate ADM-dependent work on the
        # frame status and everything else on the per-pixel label.
        if rec >= self.good_recall:
            code, msg, ok = "OK", "", True
        elif rec >= self.min_recall:
            code, ok = "OK_DEGRADED", True
            msg = "recall %.3f below %.2f but above the floor" % (rec, self.good_recall)
        else:
            code, ok = "FAIL_NO_MATCH", False
            msg = ("forward model does not reproduce this frame (recall %.3f < %.2f); "
                   "pixels left unexplained" % (rec, self.min_recall))

        if low_cov:
            ok = False
            code = "FAIL_LOW_COVERAGE"
            msg = ("only %.1f %% of the selected pixels are explained (floor "
                   "%.0f %%). Components that matched keep their labels; check "
                   "the per-component counts to see which did not."
                   % (100 * cov, 100 * self.min_coverage))
        res.status = RetrievalStatus(ok, code, msg, rec, jac, cov, tier)

        # membership ---------------------------------------------------------
        # Every component a pixel belongs to, not just the first one found.
        nb = np.zeros((self.nx, self.ny), dtype=np.uint16)
        n_mem = np.zeros((self.nx, self.ny), dtype=np.uint8)
        for name in COMPONENTS:
            layer = counts["layers"].get(name)
            if layer is None:
                continue
            hit = layer.astype(bool) & obs
            nb[hit] |= MASK_BITS[name]
            n_mem[hit] += 1
            res.parts[name] = int(layer.sum())          # predicted
            res.observed_parts[name] = int(hit.sum())   # seen in the data
        nb[obs & (nb == 0)] |= BIT_UNEXPLAINED

        # Split the ADM component into individual angular samples.
        if self.solution.enabled.get("adm"):
            self.segment_adm(res, nb, obs, phase)

        res.membership = nb
        res.labels = primary_label(nb)
        has_comp = (nb & BIT_ANY_COMPONENT).astype(bool)
        res.explained = int(has_comp.sum())
        res.unexplained = int((nb == BIT_UNEXPLAINED).sum())
        res.missing = int((mask & ~obs).sum())
        # Recount membership after segmentation, which can add or clear bits.
        n_mem = np.zeros((self.nx, self.ny), dtype=np.uint8)
        for name in COMPONENTS:
            n_mem += (nb & MASK_BITS[name]).astype(bool)
            res.observed_parts[name] = int((nb & MASK_BITS[name]).astype(bool).sum())
        res.overlap = int((n_mem > 1).sum())

        if ok:
            if self._anchor is None:
                self._anchor = (index, phase)
                self._say("  phase anchored: frame %d is mask counter %d"
                          % (index, phase))
            self._last_phase = phase
            if len(self._buffer) < 400:
                rows, cols = measure_blob_centroids(obs)
                self._buffer.append((phase, rows, cols))

        return res

    # ── ADM segmentation ─────────────────────────────────────────────────────

    def segment_adm(self, res, nb, obs, phase):
        """Split the ADM label into one object per angular sample.

        The ADM component is not one class -- it is up to 162 separate
        measurements, each at a known (VZA, VAA) bin. This assigns every
        ADM-labelled pixel to its nearest predicted sample centre and files the
        result in a ``(n_vza, n_vaa)`` grid, so a downstream user can ask for
        ``cells[2, 7]`` and get exactly the pixels that sampled that angular
        bin. ``labels`` is modified in place: ADM pixels that fall near no
        predicted centre are demoted to UNEXPLAINED, since "ADM" with no bin is
        not a usable label.

        Orphan reunification (``reunite_orphans``, default on) additionally
        sweeps UNEXPLAINED pixels lying just outside a patch back into the
        nearest cell. On the 2026-07-12 frames almost all of the residual is a
        sub-pixel rim around each blob -- the generator rounds patch edges to
        integers -- so those pixels plainly belong to the sample they ring.
        The tolerance is deliberately tight (``reunite_px`` beyond the patch
        radius); it is there to recover rounding, not to adopt strays.
        """
        cfg = self._acfg()
        ps = cfg.percent_samp
        xc, yc = self._adm_xy()
        vza_s, vaa_s = cfg.shift_at(phase)
        pr, pc, vza, vaa = _centres_fast(vza_s, vaa_s, xc, yc,
                                         self.solution.coef, ps,
                                         self.solution.asin_mode)
        iv, ia = _bin_indices(ps)
        n_vza = len(VZA_BOUNDARIES) - 1
        n_vaa = len(VAA_BOUNDARIES) - 1
        r_pix = float(cfg.max_radius_of_pix or 20)

        # Every pixel carrying the ADM bit, including those that also carry a
        # stripe bit -- a pixel under the cross-track stripe is still an ADM
        # sample, and dropping it would bias the cells the stripes happen to
        # cross.
        rows, cols = np.nonzero((nb & MASK_BITS["adm"]).astype(bool))
        n_core = len(rows)
        n_reunited = 0
        if self.reunite_orphans:
            orow, ocol = np.nonzero(nb == BIT_UNEXPLAINED)
            if len(orow):
                j = _assign_to_centres(orow, ocol, pr, pc,
                                       r_pix + self.reunite_px)
                take = j >= 0
                n_reunited = int(take.sum())
                if n_reunited:
                    nb[orow[take], ocol[take]] = MASK_BITS["adm"]
                    rows = np.concatenate([rows, orow[take]])
                    cols = np.concatenate([cols, ocol[take]])

        j = _assign_to_centres(rows, cols, pr, pc, r_pix + self.reunite_px + 2)
        # An ADM pixel with no nearby centre is not attributable. Clear its ADM
        # bit; if that leaves it with no component at all, it is unexplained.
        lost = j < 0
        if lost.any():
            lr, lc = rows[lost], cols[lost]
            nb[lr, lc] &= ~MASK_BITS["adm"]
            none_left = (nb[lr, lc] & BIT_ANY_COMPONENT) == 0
            nb[lr[none_left], lc[none_left]] |= BIT_UNEXPLAINED

        cells = np.empty((n_vza, n_vaa), dtype=object)
        cells[:] = None
        flat = rows.astype(np.int64) * self.ny + cols.astype(np.int64)
        order = np.argsort(j)
        j_s, flat_s = j[order], flat[order]
        rows_s, cols_s = rows[order], cols[order]
        # was this pixel an orphan? the reunited ones were appended last
        was_orphan = np.zeros(len(rows), dtype=bool)
        was_orphan[n_core:] = True
        was_orphan = was_orphan[order]

        start = 0
        while start < len(j_s):
            k = j_s[start]
            end = start + int(np.searchsorted(j_s[start:], k, side="right"))
            if k >= 0:
                sel = slice(start, end)
                cell = AdmCell(
                    i_vza=int(iv[k]), i_vaa=int(ia[k]),
                    vza=float(vza[k]), vaa=float(vaa[k]),
                    tag=cell_tag(iv[k], ia[k]),
                    pixels=flat_s[sel].astype(np.int64),
                    centre_pred=(float(pr[k]), float(pc[k])),
                    centre_obs=(float(rows_s[sel].mean()),
                                float(cols_s[sel].mean())),
                    radius_pred=float(np.hypot(pr[k] - xc, pc[k] - yc)),
                    n_reunited=int(was_orphan[sel].sum()),
                )
                cells[cell.i_vza, cell.i_vaa] = cell
            start = end

        res.adm_cells = cells
        res.reunited = n_reunited
        return cells

    # ── reporting ────────────────────────────────────────────────────────────

    def write_report(self, path, results=None, source=None, extra=None):
        """Write the retrieved parameters and the per-frame status table."""
        sol = self.solution
        lines = [
            "Libera WFOV pixel-mask retrieval",
            "written %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
            "source  %s" % (source or "(unspecified)"),
            "tables  %s" % (self.table_dir or "docs/wfov_tables"),
            "",
            "RETRIEVED FROM THE DATA",
            "-" * 72,
        ]
        if sol is None:
            lines.append("no solution -- every frame failed")
        else:
            lines.append(sol.describe())
            lines += [
                "",
                "  precession step = %.4f deg/mask  (period %d frames)"
                % (self.step_deg, self.period),
                "  frames in fit   = %d, phase span %.1f deg"
                % (sol.n_frames_fitted, sol.phase_span_deg),
            ]
        if results:
            ok = sum(1 for r in results if r.status.ok)
            lines += [
                "", "PER-FRAME STATUS", "-" * 72,
                "%d of %d frames usable" % (ok, len(results)),
                "",
                "%5s %6s %16s %8s %8s %9s %9s %9s %8s"
                % ("frame", "phase", "status", "recall", "jaccard", "coverage",
                   "observed", "unexpl.", "missing"),
            ]
            for r in results:
                lines.append("%5d %6s %16s %8.3f %8.3f %9.3f %9d %9d %8d"
                             % (r.index, r.phase, r.status.code, r.status.recall,
                                r.status.jaccard, r.status.coverage, r.observed,
                                r.unexplained, r.missing))
            bad = [r for r in results if not r.status.ok]
            if bad:
                lines += ["", "FAILURES", "-" * 72]
                for r in bad:
                    lines.append("  frame %3d  %s" % (r.index, r.status))
                lines += ["",
                          "Frames above are NOT safe for L2. Only pixels carrying a",
                          "component label may be used downstream."]
        if extra:
            lines += list(extra)
        if self.log:
            lines += ["", "LOG", "-" * 72] + ["  " + m for m in self.log]
        text = "\n".join(lines) + "\n"
        with open(path, "w") as fh:
            fh.write(text)
        return text

    def solution_json(self):
        """The cached solution as JSON -- for carrying between runs."""
        return json.dumps(asdict(self.solution), indent=2, default=str) \
            if self.solution else "{}"


# ═══════════════════════════════════════════════════════════════════════════
# scoring
# ═══════════════════════════════════════════════════════════════════════════

def _centres_fast(vza_shift, vaa_shift, xc, yc, coef, percent_samp=100,
                  asin_mode="correct"):
    """Vectorised ``adm_sample_centres``.

    Identical arithmetic to ``libcam.mask.adm.adm_sample_centres`` -- same
    hard-coded PI, same rounding -- but computed over all bins at once. The
    bootstrap evaluates this a few thousand times, which the scalar loop in
    ``adm.py`` is too slow for. Kept here rather than replacing the generator's
    version, because that one is asserted against bit-for-bit by the frozen
    reference tests.
    """
    vza_i, vaa_i = _bin_indices(percent_samp)
    vza = (VZA_BOUNDARIES[vza_i] + 0.5 * np.diff(VZA_BOUNDARIES)[vza_i]
           + vza_shift) % 90
    vaa = (VAA_BOUNDARIES[vaa_i] + 0.5 * np.diff(VAA_BOUNDARIES)[vaa_i]
           + vaa_shift) % 360
    t = _vza_sat(vza, asin_mode)
    n = len(coef)
    dist = np.zeros_like(t)
    for i, c in enumerate(coef):
        dist = dist + c * t ** (n - i - 1)
    r = np.round(xc + dist * np.sin(vaa / 180 * PI) / PIX_SPACING_MM)
    c_ = np.round(yc + dist * np.cos(vaa / 180 * PI) / PIX_SPACING_MM)
    return r, c_, vza, vaa


def _predicted_radii(vza_shift, coef, percent_samp=100, asin_mode="correct"):
    """Distinct predicted ring radii [px] at a phase, ascending."""
    vza_i, _ = _bin_indices(percent_samp)
    centres = VZA_BOUNDARIES[:-1] + 0.5 * np.diff(VZA_BOUNDARIES)
    vza = (centres[np.unique(vza_i)] + vza_shift) % 90
    return np.sort(_poly_radius_px(vza, coef, asin_mode))


def cell_tag(i_vza, i_vaa):
    """Short human tag for one ADM cell: ring letter + azimuth number.

    ``A1`` .. ``I18`` for the nominal 9 VZA x 18 VAA grid. The tag is derived
    from the indices rather than handed out sequentially, so it means the same
    thing on every frame and in every granule -- ``C7`` is the same angular bin
    tomorrow as it is today, which a running counter could not promise. Letters
    wrap past Z if a future bin grid is ever larger.
    """
    letters = ""
    n = int(i_vza)
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return "%s%d" % (letters, int(i_vaa) + 1)


def _assign_to_centres(rows, cols, cr, cc, max_dist):
    """Nearest predicted centre for each pixel, or -1 beyond ``max_dist``."""
    if not len(rows) or not len(cr):
        return np.full(len(rows), -1, dtype=int)
    try:
        from scipy.spatial import cKDTree
        d, j = cKDTree(np.column_stack([cr, cc])).query(
            np.column_stack([rows, cols]))
    except ImportError:
        j = np.empty(len(rows), dtype=int)
        d = np.empty(len(rows), dtype=float)
        for s in range(0, len(rows), 20000):      # chunked: the full outer
            e = s + 20000                          # product is ~260 MB at 2048^2
            d2 = ((rows[s:e, None] - cr[None, :]) ** 2 +
                  (cols[s:e, None] - cc[None, :]) ** 2)
            j[s:e] = d2.argmin(axis=1)
            d[s:e] = np.sqrt(d2.min(axis=1))
    j = np.asarray(j, dtype=int).copy()
    j[np.asarray(d) > max_dist] = -1
    return j


def measured_blob_radii(retriever):
    """Per-frame MEASURED blob radii [px] -- the irreducible observation.

    Returns ``[(n, phase, [(radius_px, clipped), ...]), ...]``: one entry per
    detected ADM blob centroid, ascending by radius, for each buffered frame.
    ``clipped`` marks a blob whose patch reaches the FPA edge.

    **These radii have a measurement floor of roughly 4 px. Know it before
    fitting anything to them.** On a synthetic frame built from v0 at scale
    1.000 through the as-coded arcsine -- where the answer is known exactly --
    fitting these radii returns scale 1.0025 at **3.97 px rms**, not 1.000 at 0.

    The cause is **blob merging**, measured 2026-10-06: at ``percent_samp`` 100
    only ~126 blobs are resolved from 162 samples, because neighbours in the
    inner rings touch and label as one, putting that centroid between two true
    centres. It is not edge clipping (zero clipped blobs in that test) and not
    stripe cutting (the floor is the same with stripes removed entirely).

    The consequence is blunt: **model differences of 2-7 px rms are at or below
    this floor**, so ring radii cannot by themselves settle which field-angle
    configuration is flying. They corroborate; the flight-software source is
    what decides. See ``apps/l1b/check_asin_claim.py`` and
    ``docs/FSW_KNOWN_BEHAVIOUR.md``.

    ``clipped`` is flagged separately and is a real effect at phases where the
    outer ring reaches the array edge -- an edge-cut patch loses area on its
    outer side and its centroid is pulled inward. It just is not the dominant
    term at the phases tested so far.

    **This, not the ring medians, is the record that matters.** Nothing here
    uses the polynomial, the arcsine or the retrieved scale -- only centroids
    and a centre -- so it can test what the retrieval concluded instead of
    echoing it. Clustering into rings is a *processing step with parameters*
    (``_cluster_radii``'s 18 px gap and 3-member floor), and ``_bootstrap_rings``
    deliberately does not rely on it: "scoring the handful of clustered ring
    medians is under-determined ... every blob radius is a separate constraint".
    A recorder that kept only the medians would bake in a choice the retrieval
    itself refuses to make, and would be unrecoverable afterwards.

    Note also that radii must be recorded in PIXELS. An angle is not a
    measurement here -- converting a radius to a VZA runs it back through the
    very polynomial and arcsine under test, so a "measured VZA" is a prediction
    wearing a measurement's clothes. Pixels are the invariant.
    """
    if retriever.solution is None or not retriever._buffer:
        return []
    xc, yc = retriever._adm_xy()
    r_pix = float(retriever._acfg().max_radius_of_pix or 20)
    out = []
    for n, (phase, rows, cols) in enumerate(retriever._buffer):
        if not len(rows):
            continue
        rr = np.asarray(rows, dtype=float)
        cc = np.asarray(cols, dtype=float)
        rad = np.hypot(rr - xc, cc - yc)
        # A patch cut by the FPA edge loses area on its outer side, so its
        # centroid is pulled INWARD -- by several px on the outermost ring. That
        # bias mimics a polynomial scale slightly below 1, which is exactly the
        # kind of artefact that has cost us a retraction before. Flag it rather
        # than averaging it in; the consumer decides.
        edge = np.minimum.reduce([rr, cc,
                                  retriever.nx - 1 - rr,
                                  retriever.ny - 1 - cc])
        clipped = edge < r_pix
        order = np.argsort(rad)
        out.append((n, int(phase),
                    [(round(float(rad[i]), 2), bool(clipped[i])) for i in order]))
    return out


def measured_ring_radii(retriever, min_rings=4):
    """Per-frame ring radii [px]: the blob radii of :func:`measured_blob_radii`
    clustered into rings and reduced to medians.

    Returns ``[(n, phase, [r1, r2, ...]), ...]`` -- ``n`` is the ordinal of the
    buffered frame. The ordinal is carried because the retrieved phase is not
    always distinct between consecutive frames, and a table keyed on phase alone
    would silently collapse rows.

    A convenience summary, **derived and lossy**: it depends on the clustering
    parameters, and a merged or split ring is invisible once the medians are all
    that survive. Record it alongside the blob radii, never instead of them.
    """
    out = []
    for n, phase, blobs in measured_blob_radii(retriever):
        rings = _cluster_radii(np.array([b[0] for b in blobs], dtype=float))
        if len(rings) >= min_rings:
            out.append((n, int(phase), [round(float(v), 1) for v in rings]))
    return out


def _cluster_radii(radii, gap=18.0, min_members=3):
    """Group blob radii into rings. Returns the sorted per-ring medians.

    A ring is a run of radii with no gap wider than ``gap``. Singletons are
    dropped -- a lone blob at an odd radius is more likely a stripe fragment
    that survived the exclusion than a real ring.
    """
    r = np.sort(np.asarray(radii, dtype=float))
    if not len(r):
        return np.array([])
    out, start = [], 0
    for i in range(1, len(r) + 1):
        if i == len(r) or r[i] - r[i - 1] > gap:
            if i - start >= min_members:
                out.append(float(np.median(r[start:i])))
            start = i
    return np.array(out)


def _best_scale(obs_rings, pred_rings):
    """Least-squares scale taking ``pred_rings`` onto ``obs_rings``.

    The two lists need not be the same length -- rings fall off the FPA at high
    VZA and the innermost one appears and disappears with the precession -- so
    each observed ring is matched to its nearest scaled prediction, twice, which
    is enough for the matching to settle.

    Returns ``(scale, rms)`` or ``(None, inf)`` if nothing could be matched.
    """
    obs_rings = np.asarray(obs_rings, dtype=float)
    pred_rings = np.asarray(pred_rings, dtype=float)
    if not len(obs_rings) or not len(pred_rings):
        return None, float("inf")

    s = float(np.median(obs_rings) / np.median(pred_rings))
    for _ in range(2):
        idx = np.abs(obs_rings[:, None] - s * pred_rings[None, :]).argmin(axis=1)
        p = pred_rings[idx]
        denom = float((p ** 2).sum())
        if denom <= 0:
            return None, float("inf")
        s = float((obs_rings * p).sum() / denom)
    rms = float(np.sqrt(np.mean((obs_rings - s * p) ** 2)))
    return s, rms


def _recall(pred, obs):
    """Fraction of predicted pixels that are selected in the data.

    Recall rather than Jaccard, deliberately: a selected pixel whose radiance is
    zero is indistinguishable from an unselected one, so the data's selected set
    is a *subset* of the true mask. Recall is insensitive to that; Jaccard is
    not, and would penalise a correct retrieval over a dark scene.
    """
    p = int(pred.sum())
    return float((pred & obs).sum()) / p if p else float("nan")


def _jaccard(pred, obs):
    u = int((pred | obs).sum())
    return float((pred & obs).sum()) / u if u else float("nan")
