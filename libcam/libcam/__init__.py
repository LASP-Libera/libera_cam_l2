# Camera geometry needs cartopy (via libcam.util.geo). Mask generation needs
# nothing but numpy, so keep the heavy import optional — otherwise
# `from libcam.mask import adm_mask` would require a full cartopy install.
# Same pattern already used for modis/pyhdf below.
try:
    from libcam.camera import Platform
    from libcam import util, common
except ImportError:  # cartopy not installed — geometry unavailable, mask still works
    pass

# modis is imported lazily — pyhdf (HDF4) required only when actually reading MODIS files
try:
    from libcam import modis
except ImportError:
    pass
