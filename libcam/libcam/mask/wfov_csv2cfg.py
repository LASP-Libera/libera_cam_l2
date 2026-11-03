"""
libcam/mask/wfov_csv2cfg.py
---------------------------
Map WFOV table parameters onto pixel-mask generator arguments.

The uplinked WFOV tables are the authority on how the flight software builds each
pixel mask. This module reads them and turns them into config objects whose fields
are exactly the keyword arguments of ``stripe_mask``, ``adm_mask`` and
``limb_mask``.

**The CSVs in ``docs/wfov_tables/`` are the master copy.** They are what this
module reads and what git tracks -- edit them to change a parameter. No Excel file
is read at runtime. When a new official ``WFOV_Tables.xlsx`` arrives, import it
with ``apps/mask/wfov_xlsx2csv.py``, which refuses to overwrite the master CSVs
without ``--force`` and shows a diff first.

    from libcam.mask.wfov_csv2cfg import load_tables, stripe_cfg, adm_cfg, limb_cfg
    tbl = load_tables()
    cfg = stripe_cfg(tbl, 'nadir')
    mask = stripe_mask(**cfg.as_kwargs())

Table values are a **starting point, not gospel.** They are the current/default set
and are not necessarily what was uplinked for any given test — the 2025_294 RAPS
data fits xc=1015 and a stripe centre row of 949, where the table says 1024/1024.
So every constructor takes overrides:

    cfg = stripe_cfg(tbl, 'raps', xc=1015, yc=949, fr=1000)

Each config records the provenance of every field, so you can always ask what came
from the table and what did not:

    cfg.report()      # prints field, value, source

Three things the tables do NOT contain
--------------------------------------
1. ``COEF_ANG2DIST`` — the field-angle polynomial that converts viewing zenith
   angle to a radial distance on the FPA. It is not a table parameter, so the
   flight software must hold it internally (probably hard-coded). The ground value
   below reproduces neither the flown ring radii nor their count: flown rings sit
   at 145–855 px in 8 rings, the ground value predicts 85–1015 px in 9. Measuring
   about the alternative FPA centres does not reconcile them. **Open issue — raise
   with Jake Gristey.** Consequence: ring radii cannot be predicted a priori, so
   the inspect code should fit the mapping from the data and assign discs to bins
   by rank rather than by absolute radius.
2. ``VZA_BOUNDARIES`` / ``VAA_BOUNDARIES`` — the angular bin edges, for the same
   reason. A different bin *count* in flight would explain 8 rings versus 9.
3. **An ADM mask centre.** ``icie_tbl_wfov_adm`` has no xcenter/ycenter of its own.
   Per Sebastian: the ADM mask uses the **nadir stripe centre**, so
   ``wfov_tbl_xcenter_stripe_nadir`` / ``_ycenter_stripe_nadir`` are mapped onto
   ``x_center_pix`` / ``y_center_pix`` and reported as such.

Open questions for Jake Gristey
-------------------------------
* Why do the flown ADM rings sit at 145-855 px in **8** rings when the ground
  polynomial predicts 85-1015 px in **9**? Right now we are not sampling out to
  the edge of the angular space. Suspected cause: a different ``COEF_ANG2DIST``
  inside flight software.
* What does ``wfov_tbl_shift_num`` = 162 actually mean? Described as "number of
  samples to shift per mask", but 162 is also exactly the bin count (9 x 18).

The lock-on table is parsed for completeness but is **not used** — that mask family
was dropped from the science plan.
"""

import csv
import os

__all__ = [
    "load_tables", "stripe_cfg", "adm_cfg", "limb_cfg", "wfov_cfg",
    "StripeCfg", "AdmCfg", "LimbCfg", "STRIPE_FAMILIES", "TABLE_DIR",
    "split_overrides", "apply_table_overrides", "table_params",
]

# docs/wfov_tables/ relative to the repository root. NOTE: this file now
# lives at libcam/libcam/mask/ (one level deeper than before the 2026-11-03
# uv_build re-nesting -- see docs/PATCHES.md), so it takes one more ".." to
# reach the repo root than it used to.
TABLE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "..", "docs", "wfov_tables"))

STRIPE_FAMILIES = ("nadir", "forward", "cross", "raps")

# Ground value of the field-angle polynomial. NOT a table parameter -- see the
# module docstring. Overridable so there is one place to change it when the
# flight value is known.
COEF_ANG2DIST = [0.0, -2.07789e-6, -1.16828e-5, 9.92181e-2, 0.0]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _num(v):
    """Table values arrive as strings; keep ints as ints."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f == int(f) and abs(f) < 2**53 else f


def load_tables(table_dir=None):
    """Read every ``icie_tbl_wfov*.csv`` into one dict: param name -> value.

    Repeated parameter names (the lock-on coefficient sets) accumulate into a
    list, in file order.
    """
    table_dir = table_dir or TABLE_DIR
    if not os.path.isdir(table_dir):
        raise SystemExit("WFOV table directory not found: %s" % table_dir)

    out, desc = {}, {}
    for name in sorted(os.listdir(table_dir)):
        if not name.startswith("icie_tbl_wfov") or not name.endswith(".csv"):
            continue
        with open(os.path.join(table_dir, name)) as fh:
            for row in csv.DictReader(fh):
                key = (row.get("param") or "").strip()
                if not key:
                    continue
                val = _num(row.get("value"))
                desc[key] = (row.get("description") or "").strip()
                if key in out:
                    if not isinstance(out[key], list):
                        out[key] = [out[key]]
                    out[key].append(val)
                else:
                    out[key] = val
    out["_descriptions"] = desc
    return out


# ---------------------------------------------------------------------------
# config objects
# ---------------------------------------------------------------------------

class _Cfg:
    """Base: holds values plus where each came from."""

    _fields = ()

    def __init__(self, values, sources):
        self._sources = dict(sources)
        for f in self._fields:
            setattr(self, f, values.get(f))

    def as_kwargs(self, *skip):
        """Keyword arguments for the matching generator, minus `skip`."""
        return {f: getattr(self, f) for f in self._fields
                if f not in skip and getattr(self, f) is not None}

    def source(self, field):
        return self._sources.get(field, "unknown")

    def report(self):
        label = getattr(self, "family", None)
        print("%s%s" % (self.__class__.__name__,
                        " [%s]" % label if label else ""))
        w = max(len(f) for f in self._fields)
        for f in self._fields:
            print("   %-*s = %-22s  [%s]"
                  % (w, f, repr(getattr(self, f)), self.source(f)))

    def __repr__(self):
        label = getattr(self, "family", None)
        inner = ", ".join("%s=%r" % (f, getattr(self, f)) for f in self._fields)
        return "%s(%s%s)" % (self.__class__.__name__,
                             "%r, " % label if label else "", inner)


class StripeCfg(_Cfg):
    """Arguments for ``stripe_mask``, plus the az_rot flag.

    ``az_rot`` is NOT a ``stripe_mask`` argument. It is the table's AZ-rotation
    flag (0 = fixed, 2 = rotate with the mechanism azimuth on the RAPS stripe);
    ``combine_masks`` consumes it and adds the azimuth to ``sr``. Use
    ``as_kwargs('az_rot')`` to get a dict safe to splat into ``stripe_mask``.
    """
    _fields = ("xc", "yc", "fr", "sy", "sr", "su", "sd", "az_rot")


class AdmCfg(_Cfg):
    """Arguments for ``adm_mask``, plus the precession terms.

    ``vaa_shift``/``vza_shift`` here are the **per-mask increments** from the
    table, not the shift to apply to one frame. For frame *n*:

        applied_shift = shift_init + n * shift

    ``combine_masks`` does that arithmetic and passes the result to ``adm_mask``
    as its ``vaa_shift``/``vza_shift``. ``as_kwargs('vaa_shift', 'vza_shift',
    'vaa_shift_init', 'vza_shift_init', 'shift_num', 'coef_ang2dist')`` gives the
    frame-independent part.

    ``shift_num`` (table: "Number of samples to shift per mask" = 162) is
    ambiguous: 162 is also exactly the number of angular bins (9 VZA x 18 VAA), so
    it may be a count rather than a shift. Carried through unused pending
    clarification from Matt or Jake.
    """
    _fields = ("percent_samp", "max_radius_of_pix", "x_center_pix", "y_center_pix",
               "vaa_shift", "vza_shift", "vaa_shift_init", "vza_shift_init",
               "shift_num", "coef_ang2dist")

    def shift_at(self, frame=0):
        """(vza_shift, vaa_shift) to apply at this frame index."""
        vza = (self.vza_shift_init or 0) + frame * (self.vza_shift or 0)
        vaa = (self.vaa_shift_init or 0) + frame * (self.vaa_shift or 0)
        return vza, vaa


class LimbCfg(_Cfg):
    """Arguments for ``limb_mask`` — an annulus between r1 and r2."""
    _fields = ("xc", "yc", "r1", "r2")


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------

def _build(cls, mapping, tbl, defaults, overrides):
    """Assemble a config: table value, else default, else override wins."""
    values, sources = {}, {}
    for field, key in mapping.items():
        if key is not None and tbl.get(key) is not None:
            values[field] = tbl[key]
            forced = key in tbl.get("_overridden", ())
            sources[field] = "table:%s%s" % (key, " -> OVERRIDDEN" if forced else "")
        else:
            values[field] = defaults.get(field)
            sources[field] = "default" if key is None else "MISSING from table"
    for field, val in overrides.items():
        if field not in cls._fields:
            raise TypeError("%s has no field %r (fields: %s)"
                            % (cls.__name__, field, ", ".join(cls._fields)))
        values[field] = val
        sources[field] = "override"
    return cls(values, sources)


def stripe_cfg(tbl, family, **overrides):
    """Config for one stripe family: nadir, forward, cross or raps."""
    if family not in STRIPE_FAMILIES:
        raise ValueError("family must be one of %s, got %r"
                         % (", ".join(STRIPE_FAMILIES), family))
    mapping = _stripe_map(family)
    defaults = dict(xc=1024, yc=1024, fr=1021, sy=0, sr=0, su=20, sd=20, az_rot=0)
    cfg = _build(StripeCfg, mapping, tbl, defaults, overrides)
    cfg.family = family

    # Rotation-sign convention. The WFOV table's `rot` runs opposite to
    # stripe_mask's `sr`: the table is written in FPA (row, column) order while
    # stripe_mask works in (x, y), which flips the handedness. Verified against
    # flight -- DITL2 2026-07-12 frame 0 reproduces to Jaccard 0.997 with this
    # negation and lands the forward stripe 368 px away without it.
    # Applied here, in the table adapter, so stripe_mask itself keeps the
    # signature the frozen reference tests assert against.
    if "sr" not in overrides and cfg.sr is not None:
        cfg.sr = -cfg.sr
        cfg._sources["sr"] += " (sign flipped: table rot -> sr)"
    return cfg


def adm_cfg(tbl, **overrides):
    """Config for the ADM lock-on sampling mask."""
    mapping = ADM_MAP
    defaults = dict(percent_samp=100, max_radius_of_pix=20,
                    x_center_pix=1024, y_center_pix=1024,
                    vaa_shift=0.0, vza_shift=0.0,
                    vaa_shift_init=0.0, vza_shift_init=0.0,
                    shift_num=None, coef_ang2dist=list(COEF_ANG2DIST))
    return _build(AdmCfg, mapping, tbl, defaults, overrides)


def limb_cfg(tbl, **overrides):
    """Config for the limb annulus mask."""
    mapping = LIMB_MAP
    defaults = dict(xc=1024, yc=1024, r1=895, r2=1024)
    return _build(LimbCfg, mapping, tbl, defaults, overrides)


def _stripe_map(family):
    """Field name -> table parameter name, for one stripe family."""
    return {
        "xc":     "wfov_tbl_xcenter_stripe_%s" % family,
        "yc":     "wfov_tbl_ycenter_stripe_%s" % family,
        "fr":     "wfov_tbl_fovr_%s" % family,
        "sy":     "wfov_tbl_ydist_%s" % family,
        "sr":     "wfov_tbl_rot_%s" % family,
        "su":     "wfov_tbl_su_%s" % family,
        "sd":     "wfov_tbl_sd_%s" % family,
        "az_rot": "wfov_tbl_az_rot_%s" % family,
    }


ADM_MAP = {
    "percent_samp":      "wfov_tbl_persamp",
    "max_radius_of_pix": "wfov_tbl_maxPixR",
    "vaa_shift":         "wfov_tbl_vaashift",
    "vza_shift":         "wfov_tbl_vzashift",
    "vaa_shift_init":    "wfov_tbl_vaashift_init",
    "vza_shift_init":    "wfov_tbl_vzashift_init",
    "shift_num":         "wfov_tbl_shift_num",
    # The ADM table carries no centre of its own; it uses the nadir stripe's.
    "x_center_pix":      "wfov_tbl_xcenter_stripe_nadir",
    "y_center_pix":      "wfov_tbl_ycenter_stripe_nadir",
    # not in any table -- see module docstring. Reachable only via a direct
    # generator override, which is how a fitted polynomial gets in.
    "coef_ang2dist":     None,
}

LIMB_MAP = {
    "xc": "wfov_tbl_xcenter_limb",
    "yc": "wfov_tbl_ycenter_limb",
    "r1": "wfov_tbl_r1",
    "r2": "wfov_tbl_r2",
}


def field_table_names(field):
    """Every table parameter name that any config maps `field` onto.

    Empty when the field has no table backing (``coef_ang2dist``), which is what
    makes it legal as a direct generator override. Non-empty means the value must
    be set through ``TABLE_OVERRIDES`` instead, because a bare field name cannot
    say *which* one is meant -- all four stripe families have an ``su``.
    """
    names = []
    for fam in STRIPE_FAMILIES:
        key = _stripe_map(fam).get(field)
        if key and key not in names:
            names.append(key)
    for mapping in (ADM_MAP, LIMB_MAP):
        key = mapping.get(field)
        if key and key not in names:
            names.append(key)
    return names


def table_params(tbl):
    """The set of real WFOV table parameter names in a loaded table dict."""
    return {k for k in tbl if not k.startswith("_")}


def apply_table_overrides(tbl, overrides):
    """Return a copy of `tbl` with specific table parameters replaced.

    Keyed by the **actual table parameter name**, which is unique by
    construction -- which is the point. The stripe families share generator
    keyword names (all four have an ``su``), so a generator-level override
    cannot single one out:

        apply_table_overrides(tbl, dict(wfov_tbl_su_forward=15,
                                        wfov_tbl_su_cross=30))

    The names are exactly what ``cfg.report()`` prints, so they can be copied
    straight from its output.

    An unrecognised name raises KeyError rather than being ignored: a silent
    no-op on ``wfov_tbl_su_forwards`` is the failure mode that costs an
    afternoon. Close matches are suggested.
    """
    if not overrides:
        return tbl
    import difflib
    known = table_params(tbl)
    bad = [k for k in overrides if k not in known]
    if bad:
        lines = []
        for k in sorted(bad):
            # A generator field name is the likeliest mistake -- name its table
            # parameter(s) explicitly rather than leaving the user to guess.
            mapped = field_table_names(k)
            if mapped:
                # The quantity IS in the table -- the user just used the
                # generator keyword instead of the table parameter name, which is
                # what this dict is keyed by.
                lines.append("  %s is the generator keyword. This dict is keyed by"
                             " table parameter\n      name%s -- use: %s"
                             % (k, "s" if len(mapped) > 1 else "", ", ".join(mapped)))
                continue
            near = difflib.get_close_matches(k, sorted(known), n=3, cutoff=0.6)
            lines.append("  %s%s" % (k, "   did you mean: %s ?" % ", ".join(near)
                                     if near else ""))
        raise KeyError("not valid WFOV table parameter name(s):\n%s\n"
                       "(%d parameters loaded; run "
                       "`python libcam/mask/wfov_csv2cfg.py` to list them)"
                       % ("\n".join(lines), len(known)))
    out = dict(tbl)
    out.update(overrides)
    # remember which ones were forced, so cfg.report() can say so
    out["_overridden"] = set(tbl.get("_overridden", set())) | set(overrides)
    return out


def split_overrides(cfg_cls, overrides, extra_ok=()):
    """Route user overrides to the config object or straight to the generator.

    The mask testers accept arbitrary generator keywords so that a value can be
    forced regardless of what the table says. Two destinations exist:

    * a name that is a field of ``cfg_cls`` (``percent_samp``, ``xc``, ``r1`` ...)
      overrides the **table value**, and any downstream arithmetic — notably the
      ADM frame precession — then runs on the overridden value;
    * a name in ``extra_ok`` (``clip``, ``clip_fov``, ``nx``, ``ny`` ...) is not a
      table parameter at all and is passed directly to the generator.

    Anything else raises TypeError naming the valid options, so a typo fails loudly
    rather than being silently ignored.
    """
    cfg_kw, gen_kw, bad, table_backed = {}, {}, [], []
    for key, val in overrides.items():
        mapped = field_table_names(key)
        if mapped:
            # Backed by the table, so a bare field name cannot say which one --
            # all four stripe families have an `su`. Route it to TABLE_OVERRIDES.
            table_backed.append((key, mapped))
        elif key in cfg_cls._fields:
            cfg_kw[key] = val          # e.g. coef_ang2dist: no table backing
        elif key in extra_ok:
            gen_kw[key] = val
        else:
            bad.append(key)

    if table_backed:
        lines = []
        for key, mapped in sorted(table_backed):
            lines.append("  %s -> set it in TABLE_OVERRIDES as one of:\n      %s"
                         % (key, "\n      ".join("%s=..." % m for m in mapped)))
        raise TypeError(
            "these are WFOV table parameters and must go in TABLE_OVERRIDES,\n"
            "not in the plain overrides -- a bare field name cannot say which\n"
            "component is meant:\n%s" % "\n".join(lines))

    if bad:
        free = [f for f in cfg_cls._fields if not field_table_names(f)]
        raise TypeError(
            "unknown override(s): %s\n"
            "  allowed here (no table backing): %s\n"
            "  generator-only: %s\n"
            "  everything else goes in TABLE_OVERRIDES"
            % (", ".join(sorted(bad)), ", ".join(free) or "(none)",
               ", ".join(extra_ok) or "(none)"))
    return cfg_kw, gen_kw


def wfov_cfg(tbl):
    """Camera-level settings from ``icie_tbl_wfov`` (not mask geometry).

    Includes ``pix_res`` = 12, the ADC resolution. Note that the **downlink word
    is 13 bits**: with dual integration times, the 13th bit encodes which exposure
    a pixel used (selected on board against ``pix_sel``). Use 13, not ``pix_res``,
    as the uncompressed basis for compression ratios.
    """
    keys = {
        "exp1": "wfov_tbl_exp1", "exp2": "wfov_tbl_exp2",
        "inter_img": "wfov_tbl_inter_img", "delta": "wfov_tbl_delta",
        "white_lim": "wfov_tbl_white_lim", "black_lim": "wfov_tbl_black_lim",
        "offset": "wfov_tbl_offset", "pix_res": "wfov_tbl_pix_res",
        "pix_sel": "wfov_tbl_pix_sel", "adc_gain": "wfov_tbl_adc_gain",
        "freqdiv": "wfov_tbl_freqdiv",
    }
    return {k: tbl.get(v) for k, v in keys.items()}


# ---------------------------------------------------------------------------

def _main():
    import argparse
    p = argparse.ArgumentParser(
        description="Show the WFOV table -> mask-argument mapping.")
    p.add_argument("--table_dir", default=None, help="directory of table CSVs")
    a = p.parse_args()

    tbl = load_tables(a.table_dir)
    n = len([k for k in tbl if not k.startswith("_")])
    print("loaded %d parameters from %s\n" % (n, a.table_dir or TABLE_DIR))

    for fam in STRIPE_FAMILIES:
        stripe_cfg(tbl, fam).report()
        print()
    adm_cfg(tbl).report()
    print()
    limb_cfg(tbl).report()
    print()
    print("camera settings (icie_tbl_wfov)")
    for k, v in wfov_cfg(tbl).items():
        print("   %-12s = %s" % (k, v))
    print()
    print("NOTE  pix_res = %s is the ADC resolution. The downlink word is 13 bits"
          % wfov_cfg(tbl).get("pix_res"))
    print("      (the 13th bit flags which integration time the pixel used).")
    print("NOTE  COEF_ANG2DIST and the VZA/VAA bin edges are NOT in the tables;")
    print("      the ground values do not reproduce the flown ring radii. Open issue.")


if __name__ == "__main__":
    _main()
