# The missing lower-bound guard in the Python ADM mask

**Status:** present in the original Python, absent from the C. Fixed here by default
(`clip=True`); the old behaviour is still reachable with `clip=False` so historical
masks can be reproduced.

## What it is

The original Python grows a square box around each ADM sample centre and keeps the
pixels inside a circular radius:

```python
x_pix_box = range(int(x_pix_selected[i] - max_radius_of_pix),
                  int(x_pix_selected[i] + max_radius_of_pix + 1))
...
        if radius_from_central_pix <= max_radius_of_pix + 0.5:
            if x < x_center_pix*2 and y < y_center_pix*2:   # <-- upper bound only
                mask[x][y] = 1
```

The guard checks the *upper* bound on both axes and neither *lower* bound. When a
sample sits near the x=0 or y=0 edge, the box extends to negative indices, and in
Python `mask[-15][y]` does not raise — it writes to `mask[2033][y]`, on the **opposite
edge of the detector**.

The C reference does not have this problem. It clamps both ends before the loop:

```c
if (x_pix_box_s < 0)  { x_pix_box_s = 0;  }
if (x_pix_box_e > xp) { x_pix_box_e = xp; }
if (y_pix_box_s < 0)  { y_pix_box_s = 0;  }
if (y_pix_box_e > yp) { y_pix_box_e = yp; }
```

At the default 100 % sampling with `max_radius_of_pix = 20`, the VZA 85° / VAA 270°
sample lands at x = 5, so its patch reaches x = −15. 905 pixel writes fall outside
the array.

## Why nobody noticed

**Because the default mask is point-symmetric.** The VAA bins run 0–360° in even 20°
steps about the array centre, so every patch that overhangs one edge wraps onto the
position of its own 180° mirror — which is already set. The wrapped writes land on
pixels that were going to be 1 anyway, and the populated count is unchanged.

This is why the C and Python agree exactly today, and why `compare_c_python.py`
reports IDENTICAL:

```
C      :   206879 populated
Python :   206879 populated   (clip=True)
IDENTICAL -- the two implementations agree pixel for pixel.
```

The symmetry is doing the work, not the code.

## When it stops being benign

The moment the optical axis is not exactly at the array centre, the mirror symmetry
breaks and the wrapped pixels no longer land on already-set ones. The measured FPA
centre is **not** (1024, 1024) — the stripe-centre work puts it at (1026, 1000) or
(1026, 989). Rebuilding the default mask about those centres:

| configuration | populated (clipped) | populated (wrapped) | pixels on the wrong side |
|---|---|---|---|
| centre (1024, 1024) — array centre | 206 879 | 206 879 | 0 |
| centre (1026, 1000) — measured | 205 357 | 206 875 | **1 518** |
| centre (1026, 989) — alternative | 204 543 | 206 875 | **2 332** |
| centre (1000, 1024) | 206 495 | 206 879 | 384 |
| centre (1026, 1000), `vaa_shift=10` | 206 447 | 206 831 | 384 |
| centre (1026, 1000), `max_radius=30` | 435 627 | 437 691 | 2 064 |

With the measured centre, ~1 500 pixels are selected at y = 2025–2047 that should
have been dropped at y < 0. Those pixels get downlinked and then attributed to the
viewing geometry of the bin they appear to belong to — which is the wrong one, at the
opposite edge of the field.

Reproduce with:

```
python -m pytest tests/test_masks.py::test_clip_only_matters_off_centre
```

## Relevance to the FSW question

`libera-cam/TODO.md:68` records that the camera pixel masks may still carry the old
FSW bug. This is a candidate mechanism and it is worth checking which side the flight
software was generated from:

- If flight software follows the **C**, it clips correctly and this is a
  ground-tools-only issue.
- If it follows the **Python**, then any mask built about a non-centred optical axis
  has misplaced edge pixels, and the effect is largest exactly where it hurts ADM work
  most — the highest-VZA bins, which are the ones that reach the array edge.

Either way the two should not be allowed to drift again, which is what
`compare_c_python.py` in CI is for.

## A second, smaller one in the stripe mask

`pix_stripe` filters rotated coordinates with `np.where(xr >= 0)` — it guards x but
not y, so a stripe offset far enough in −y wraps the same way. It is not reachable at
the offsets in use (`sy = 0` or `180`), but it is the same defect and both bounds are
enforced in `stripe.py`.

Separately, the original rotates its `-1` placeholder rows along with the real ones.
At `sr = 0` they stay negative and are filtered out; at other angles they can rotate
into the valid range. The refactor never creates placeholders, so the question does
not arise — but it is why `stripe_mask` builds its coordinate arrays from `xg`
directly rather than from a full-width array.
