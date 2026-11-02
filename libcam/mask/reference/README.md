# `libcam/mask/reference/` — frozen originals

**Do not edit anything in this directory.** These are the original pixel-mask
implementations, kept byte-for-byte as they were written. They exist so that the
refactored generators in `libcam/mask/` can be proved to reproduce them, and so
the provenance of the algorithms is visible next to the code.

| File | Author | What it is |
|---|---|---|
| `pmask_adm.c` | Hong Chen (LASP) | C translation of the ADM generator. The version whose structure matches flight software, and the reference `tests/test_c_python_equivalence.py` builds and diffs against. |
| `pmask_adm_original.py` | Jake Gristey (NOAA), 2024 | ADM generator, surface-frame angular bins, with the `test-py.txt` dump the C comparison needs. |
| `pixelmask_adm_barebones_2021.py` | Jake Gristey (NOAA), 2021 | Earlier ADM generator. Uses bin boundaries **already in the satellite frame** with no surface→satellite conversion — a deliberate change of convention between versions, not a discrepancy. |
| `pixelmasks_funcs_original.py` | Sebastian Schmidt (LASP), 2021 | `pix_stripe` and `pix_adm` as functions; the direct ancestor of `stripe.py` and `adm.py`. |
| `pixelmask_limb_simple.py` | Sebastian Schmidt (LASP) | The limb annulus; ancestor of `limb.py`. |

## How these are used

`tests/test_masks.py` carries verbatim transcriptions of the Python originals
inside the test file itself and asserts the refactored generators reproduce them
**bit for bit**. That is the point of the whole arrangement: the refactor is only
trustworthy if it provably changed nothing.

Two behaviours in the current code look like bugs and are not — they are
preserved so this bit-identity holds:

* `pi = 3.14159` rather than `math.pi`, matching the C and the flight code.
* Truncation, not rounding, of rotated stripe coordinates. At rotations near 90°
  the two differ by about 100 pixels at the stripe ends.

Do not "fix" either; the tests will fail.

## Where the current code deliberately differs

Three divergences are intentional and documented where they occur:

1. **ADM edge clipping.** The original Python guarded the upper array bound but
   neither lower bound, so a patch overhanging `x=0` or `y=0` wrapped to the
   opposite edge of the detector via negative indexing. `pmask_adm.c` clips
   correctly. `adm_mask()` now clips by default; `clip=False` reproduces the old
   behaviour. See `../BOUNDS_BUG.md`.
2. **Stripe FOV clipping.** The originals clipped the stripe to a circle of
   radius `fr`. Flight software does not — it emits a plain rectangle, verified
   against the transmitted pixels of the 2025_294 test set. `stripe_mask()`
   therefore defaults to `clip_fov=False`; pass `True` to reproduce a
   ground-generated historical mask.
3. **Limb indexing.** The original built its grid with `np.meshgrid` (row/column)
   and indexed a mask declared `(x, y)`. For a centred annulus the two agree
   exactly, since the shape is symmetric under transpose; off centre the original
   produces a *transposed* annulus. `limb_mask()` is `mask[x, y]` throughout.

## Still open

The flown ADM ring radii do not match what these implementations predict — 8
rings spanning 145–855 px in the data, against 9 spanning 85–1015 px from the
polynomial. `COEF_ANG2DIST` is not a WFOV table parameter, so flight software
must hold its own copy. Raised with Jake Gristey; until it is resolved we are not
sampling out to the edge of the angular space.


## Known defect in `pmask_adm.c` (NOT in the Python)

`pmask_adm.c:116-117` wrap the shifted bin centres as

```c
vza = vza - floor(vza/90.0);
vaa = vaa - floor(vaa/360.0);
```

which should be `vza - 90.0*floor(vza/90.0)` and `vaa - 360.0*floor(vaa/360.0)`. As written,
an angle of 90.4 deg becomes 89.4 instead of 0.4. **The Python has always used `vza % 90`
and `vaa % 360` and is correct** -- this defect is confined to the C translation.

It is inert at zero shift (all bin centres are below 90/360, so `floor` is 0), which is why
`tests/test_c_python_equivalence.py` still reports IDENTICAL: that comparison runs at the
default shifts. It bites only once `vza_shift`/`vaa_shift` push an angle past the wrap point.

**Flight software does NOT carry this defect.** Verified against DITL2 L1A data of
2026-07-12: at 0.9 deg/mask the outermost bin crosses 90 deg at frame 6, and in the data the
ring wraps correctly to the centre (a new ring appears at 38 px at frame 7 and marches
outward). With the C's version it would have stayed pinned at ~868 px. See
`../../../../ADM_GEOMETRY_FINDINGS.md` section 7 and `output/wrap_evidence.png`.

Since the C is the only version with this defect and flight does not have it, **flight is not
a verbatim port of this C** -- relevant to the open question of what FSW was derived from.

The file is left unmodified here, as all reference copies are.
