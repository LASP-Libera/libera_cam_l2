"""Readers and writers for Libera camera pixel masks.

Formats
-------
csv   Row-per-x, comma-separated integers -- the format written by
      ``reference/pmask_adm.c``, so ``apps/mask/compare_c_python.py`` can diff the two
      implementations byte for byte.
pbm   Binary PBM (P4), the raster format used on the camera side.
npy   numpy array, for use as ``--mask_dir`` input to the compression analysis.
"""

import numpy as np

__all__ = ["write_csv", "read_csv", "write_pbm", "read_pbm", "summary",
           "output_dir", "figure_path"]


def output_dir():
    """Repository ``output/`` directory, created if absent.

    Figures and generated masks land here rather than the current working
    directory, so results do not depend on where you happened to run from.
    ``output/`` is gitignored.
    """
    import os
    d = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "output"))
    os.makedirs(d, exist_ok=True)
    return d


def figure_path(name):
    """Absolute path for a figure: ``<repo>/output/<name>``.

    An absolute or explicitly relative ``name`` is passed through untouched, so
    ``--plot /tmp/foo.png`` and ``--plot ./foo.png`` still go where you say.
    """
    import os
    if os.path.isabs(name) or name.startswith("." + os.sep) or name.startswith("./"):
        return name
    return os.path.join(output_dir(), name)


def write_csv(mask, path):
    """Write in the same layout as pmask_adm.c: one line per x, '%5d' fields."""
    with open(path, "w") as fh:
        for row in mask:
            fh.write(",".join("%5d" % v for v in row))
            fh.write("\n")


def read_csv(path):
    return np.loadtxt(path, delimiter=",").astype(np.uint8)


def write_pbm(mask, path):
    """Write a binary PBM (P4). A set mask pixel is written as 1 (black)."""
    nx, ny = mask.shape
    packed = np.packbits(mask.astype(np.uint8), axis=1)
    with open(path, "wb") as fh:
        fh.write(b"P4\n%d %d\n" % (ny, nx))
        fh.write(packed.tobytes())


def read_pbm(path):
    """Read a binary PBM (P4) into a uint8 array.

    Note: the ``.pbm`` files distributed alongside the 2025_294 camera test data
    are NOT pixel masks -- they are the dual-exposure map (which pixels use EXP1
    versus EXP2). Do not feed those to mask-population analysis.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    fields, i = [], 0
    while len(fields) < 3:
        while data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":
            while data[i:i + 1] != b"\n":
                i += 1
            continue
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    i += 1
    if fields[0] != b"P4":
        raise ValueError("not a binary PBM (P4): %s" % path)
    w, h = int(fields[1]), int(fields[2])
    raw = np.frombuffer(data[i:i + h * ((w + 7) // 8)], dtype=np.uint8)
    return np.unpackbits(raw).reshape(h, -1)[:, :w]


def summary(mask, label=""):
    n = int(mask.sum())
    total = mask.size
    return ("%-14s %8d populated  %6.3f %% of FPA  "
            "(uncompressed at 13 bit: %.1f kB)"
            % (label, n, 100.0 * n / total, n * 13 / 8 / 1000.0))
