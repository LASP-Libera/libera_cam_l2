"""
libcam/mask — pixel mask generation for the Libera WFOV camera.

The camera cannot downlink every frame in full. Flight software transmits only
the pixels selected by an uplinked pixel mask, so this module is the ground
witness of that selection: it generates the same masks the flight software does.

Two uses:

  * interpreting real downlinked data — which pixels were transmitted, and why
  * driving the proxy simulation — a mask passed to ``Platform.geomap(mask=...)``
    restricts the forward model to the pixels the camera would actually send,
    which is what ``apps/adm/adm_pattern.py`` and ``apps/raps/raps_scan.py`` need

Two mask families:

    adm_mask     lock-on sampling of CERES/TRMM (VZA, VAA) angular bins, for
                 angular distribution model work
    stripe_mask  a rotatable band across the FPA, for the nadir / along-track /
                 cross-track stripe products
    limb_mask    an annulus near the edge of the field, for limb viewing

Both return ``(nx, ny)`` uint8 arrays indexed ``mask[x, y]`` — x-major,
transposed relative to image display convention, matching the flight code and
the C reference. Transpose for plotting: ``imshow(mask.T, origin='lower')``.

Requires numpy only; nothing else in this module pulls in the heavier optional
dependencies of the rest of the package.

Provenance
----------
ADM algorithm by Jake Gristey (NOAA), 2021–2024. C translation by Hong Chen
(LASP), kept in ``reference/pmask_adm.c`` and used by
``apps/mask/compare_c_python.py`` to verify the Python against it. Stripe
generator by Sebastian Schmidt (LASP), 2021. Originals unmodified in
``reference/``.

One behavioural change from the originals is documented in ``BOUNDS_BUG.md``.
"""

from libcam.mask.adm import adm_mask, adm_sample_centres, DEFAULTS
from libcam.mask.stripe import stripe_mask
from libcam.mask.limb import limb_mask
from libcam.mask.combine_masks import combine_masks, MODES, COMPONENTS
from libcam.mask.io import (write_csv, read_csv, write_pbm, read_pbm, summary)

__all__ = [
    "adm_mask", "adm_sample_centres", "stripe_mask", "limb_mask",
    "combine_masks", "MODES", "COMPONENTS", "DEFAULTS",
    "write_csv", "read_csv", "write_pbm", "read_pbm", "summary",
]
