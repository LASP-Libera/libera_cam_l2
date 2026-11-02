"""
libcam/camera.py — Libera WFOVC Platform class

Central object for all camera geometry and proxy-image generation.

Designed after cam-hong/cam/camera.py, extended with:
  - geo2pix()  : inverse projection (from gl2)
  - mod2cam()  : MODIS-to-camera radiance sampling (PSF or nearest-neighbour)
  - local_grid(): sub-pixel coordinate grid for PSF convolution (from gl2)

Typical workflow
----------------
    platform = Platform(lon=lon0, lat=lat0, alt=alt0,
                        dtime=dtime0, heading=hed0)
    platform.geomap()          # compute lon/lat for every camera pixel
    cam_rad = platform.mod2cam(modis_lon, modis_lat, modis_rad,
                               sampling='nn')
"""

import datetime
import numpy as np

import libcam.common as C
import libcam.util as util


__all__ = ['Platform']


class Platform:
    """
    Represents the Libera satellite at a single instant in time.

    Parameters
    ----------
    lon, lat   : float — sub-satellite nadir longitude/latitude [deg]
    alt        : float — altitude [m]  (default: nominal CLARREO orbit)
    dtime      : datetime.datetime
    heading    : float — satellite heading [deg, N=0 clockwise]
    ang_pit, ang_rol : float — pitch and roll offsets [deg]  (default: 0)
    """

    def __init__(
            self,
            lon=0.0,
            lat=0.0,
            alt=C.LIBERA_ALTITUDE,
            dtime=None,
            heading=0.0,
            ang_pit=0.0,
            ang_rol=0.0,
            planet_radius=C.PLANET_RADIUS,
    ):
        self.lon = float(lon)
        self.lat = float(lat)
        self.alt = float(alt)
        self.heading = float(heading)
        self.ang_pit = float(ang_pit)
        self.ang_rol = float(ang_rol)
        self.planet_radius = float(planet_radius)
        self.dtime = dtime if dtime is not None else datetime.datetime.utcnow()
        self.jday  = util.dtime_to_jday(self.dtime)

        # Build FPA geometry (VZA/VAA for every pixel)
        fpa = util.cal_vza_vaa()
        self.vza    = fpa['vza']     # (FPA_NX, FPA_NY)
        self.vaa    = fpa['vaa']     # (FPA_NX, FPA_NY)
        self.fpa_cx = fpa['fpa_cx']
        self.fpa_cy = fpa['fpa_cy']

        # Geolocated arrays — populated by geomap()
        self.cam_lon = None
        self.cam_lat = None

        # Pixel spacing — cached by pixel_spacing()
        self._spacing_r = None
        self._spacing_l = None

    # ── Forward projection ────────────────────────────────────────────────────

    def geomap(self, mask=None):
        """
        Project every camera pixel to geographic coordinates.

        Result stored in self.cam_lon / self.cam_lat (NaN where masked / OOF).

        Parameters
        ----------
        mask : (FPA_NX, FPA_NY) int array, optional
               Pixels with mask!=0 are skipped (→ NaN in output).
        """
        self.cam_lon, self.cam_lat = util.geolocate(
            self.vza, self.vaa,
            lon0=self.lon, lat0=self.lat,
            heading=self.heading,
            altitude=self.alt,
            planet_radius=self.planet_radius,
            mask=mask,
            vza_limit=C.VZA_LIMIT,
        )

    # ── Inverse projection ────────────────────────────────────────────────────

    def geo2pix(self, lon, lat):
        """
        Map a ground point (lon, lat) to the nearest camera pixel (ix, iy).

        Returns None if the point falls outside the camera FOV.

        Parameters
        ----------
        lon, lat : float — geographic coordinates [deg]

        Returns
        -------
        (ix, iy) : tuple of ints, or None
        """
        return util.geo2pix(
            lon, lat,
            vza_grid=self.vza, vaa_grid=self.vaa,
            lon_nadir=self.lon, lat_nadir=self.lat,
            heading=self.heading,
            altitude=self.alt,
            planet_radius=self.planet_radius,
        )

    # ── MODIS → camera radiance sampling ──────────────────────────────────────

    def mod2cam(self, src_lon, src_lat, src_rad, sampling='nn_push',
                max_dist_km=None):
        """
        Sample source (MODIS or any gridded) radiance onto the camera FPA.

        Modes
        -----
        'nn_push' (default) :
            Source-driven push — for each source pixel, find the nearest valid
            camera pixel and accumulate (mean).  Equivalent to the legacy 'nn'.
        'nn'  : alias for 'nn_push' (backward compatibility).
        'nn_pull' :
            Camera-driven pull — for each valid camera pixel, find the nearest
            source pixel.  Multiple camera pixels may pull from the same source
            pixel (duplicates are expected and correct).  If max_dist_km is set,
            camera pixels whose nearest source pixel is farther than that
            threshold are blacked out (NaN).
        'psf' :
            Per-pixel ellipsoidal Gaussian PSF — for each source pixel, find
            all camera pixels within 3σ and accumulate weighted contributions.
            σ_r and σ_l come from pixel_spacing(); PSF ellipse oriented along
            the local radial/azimuthal axes.

        Parameters
        ----------
        src_lon : (M,) or (M, N) float array — longitude of source pixels [deg]
        src_lat : (M,) or (M, N) float array — latitude  of source pixels [deg]
        src_rad : same shape — radiance or reflectance values
        sampling : 'nn_push', 'nn', 'nn_pull', or 'psf'
        max_dist_km : float or None
            Only used with 'nn_pull': camera pixels farther than this from
            their nearest source pixel are set to NaN.  None = no threshold.

        Returns
        -------
        cam_rad : (FPA_NX, FPA_NY) float32 — radiance on camera FPA (NaN = no data)
        """
        if self.cam_lon is None:
            self.geomap()

        src_lon = np.asarray(src_lon, dtype=np.float64).ravel()
        src_lat = np.asarray(src_lat, dtype=np.float64).ravel()
        src_rad = np.asarray(src_rad, dtype=np.float32).ravel()

        valid_src = np.isfinite(src_lon) & np.isfinite(src_lat) & np.isfinite(src_rad)
        src_lon = src_lon[valid_src]
        src_lat = src_lat[valid_src]
        src_rad = src_rad[valid_src]

        cam_rad = np.full((C.FPA_NX, C.FPA_NY), np.nan, dtype=np.float32)

        if sampling in ('nn', 'nn_push'):
            return self._mod2cam_nn_push(src_lon, src_lat, src_rad, cam_rad)
        elif sampling == 'nn_pull':
            return self._mod2cam_nn_pull(src_lon, src_lat, src_rad, cam_rad,
                                         max_dist_km=max_dist_km)
        elif sampling == 'psf':
            return self._mod2cam_psf(src_lon, src_lat, src_rad, cam_rad)
        else:
            raise ValueError(
                f"sampling must be 'nn', 'nn_push', 'nn_pull', or 'psf', "
                f"got '{sampling}'")

    def _mod2cam_nn_push(self, src_lon, src_lat, src_rad, cam_rad):
        """
        Push nearest-neighbour: source-driven accumulation into camera pixels.

        For each source pixel, find the nearest valid camera pixel and
        accumulate (mean) all source values that map to the same camera pixel.
        Uses scipy KDTree for O(N log N) performance.

        This is the classic "push" direction: each MODIS pixel votes for the
        camera pixel it is geometrically closest to.
        """
        from scipy.spatial import KDTree

        valid = np.isfinite(self.cam_lon) & np.isfinite(self.cam_lat)
        valid_ix, valid_iy = np.where(valid)
        if valid_ix.size == 0:
            return cam_rad

        # Build tree on flattened valid camera pixel (lon, lat) coordinates
        cam_pts = np.column_stack([
            self.cam_lon[valid_ix, valid_iy],
            self.cam_lat[valid_ix, valid_iy],
        ])
        tree = KDTree(cam_pts)

        # For each source pixel find nearest camera pixel (vectorised)
        src_pts = np.column_stack([src_lon, src_lat])
        _, nn_idx = tree.query(src_pts, workers=-1)

        # Accumulate: mean of all source pixels that map to the same camera pixel
        n_cam = valid_ix.size
        count   = np.zeros(n_cam, dtype=np.int32)
        rad_sum = np.zeros(n_cam, dtype=np.float64)
        np.add.at(count,   nn_idx, 1)
        np.add.at(rad_sum, nn_idx, src_rad.astype(np.float64))

        has_data = count > 0
        values = np.full(n_cam, np.nan, dtype=np.float32)
        values[has_data] = (rad_sum[has_data] / count[has_data]).astype(np.float32)
        cam_rad[valid_ix, valid_iy] = values

        return cam_rad

    def _mod2cam_nn_pull(self, src_lon, src_lat, src_rad, cam_rad,
                         max_dist_km=None):
        """
        Pull nearest-neighbour: camera-driven lookup into source pixels.

        For each valid camera pixel, find the nearest source pixel and assign
        its radiance directly.  Unlike nn_push, multiple camera pixels may pull
        from the same source pixel — duplicates are expected when the source
        grid is coarser than the camera pixel spacing.

        Parameters
        ----------
        max_dist_km : float or None
            If set, camera pixels whose nearest source pixel is farther than
            this distance are blacked out (set to NaN).  A typical value is
            1–2 × the source pixel spacing (e.g. 1.5 km for HKM, 3 km for 1KM).
        """
        from scipy.spatial import KDTree

        valid = np.isfinite(self.cam_lon) & np.isfinite(self.cam_lat)
        valid_ix, valid_iy = np.where(valid)
        if valid_ix.size == 0:
            return cam_rad

        # Build KDTree on source pixels
        src_pts = np.column_stack([src_lon, src_lat])
        tree = KDTree(src_pts)

        # For each valid camera pixel, query nearest source pixel
        cam_pts = np.column_stack([
            self.cam_lon[valid_ix, valid_iy],
            self.cam_lat[valid_ix, valid_iy],
        ])
        dist_deg, nn_idx = tree.query(cam_pts, workers=-1)

        values = src_rad[nn_idx].astype(np.float32)

        # Optional blackout: NaN camera pixels too far from any source data
        if max_dist_km is not None:
            max_dist_deg = max_dist_km / 111.32
            values[dist_deg > max_dist_deg] = np.nan

        cam_rad[valid_ix, valid_iy] = values
        return cam_rad

    def _mod2cam_psf(self, src_lon, src_lat, src_rad, cam_rad,
                     spacing_r=None, spacing_l=None):
        """
        PSF-weighted accumulation with per-pixel ellipsoidal Gaussian.

        For each source pixel, find all camera pixels within 3σ and accumulate
        an ellipsoidal-Gaussian-weighted contribution.  The PSF shape at each
        camera pixel (i, j) is:

            weight = exp(−0.5 · (Δr² / σ_r² + Δl² / σ_l²))

        where σ_r and σ_l are the radial and azimuthal pixel spacings from
        pixel_spacing(), and (Δr, Δl) are the source-to-camera displacement
        rotated into the camera pixel's own radial/azimuthal frame:

            φ[i,j] = radians(VAA[i,j] + heading)   ← radial axis on the ground
            Δr =  Δx·cos(φ) + Δy·sin(φ)
            Δl = −Δx·sin(φ) + Δy·cos(φ)

        with (Δx, Δy) in km using a local flat-Earth approximation.

        Search radius: 3 × 90th-percentile of max(σ_r, σ_l) [km] → degrees.
        This covers the footprint up to ~55° VZA without the extreme-limb
        pixels blowing out the search radius.

        Uses scipy KDTree for O(N log N) performance.

        NOTE — actual Libera camera PSF (TODO, future development stage)
        ----------------------------------------------------------------
        The Gaussian PSF used here is a placeholder based purely on pixel
        spacing (σ_r, σ_l from the geometric pixel_spacing() Jacobian).
        The real camera PSF also includes:
          • detector pixel responsivity profile (top-hat convolved with
            diffraction / optical blur)
          • smear along the scan direction from finite integration time
          • stray-light wings from the optical design
        Sharpness metric to implement: when running 'psf' mode, compare
        the effective MTF of the simulated image against the 'nn' output
        by computing the edge-spread function (ESF) across sharp scene
        edges (e.g. coast lines or desert/cloud boundaries visible in the
        window), then differentiating to get the line-spread function (LSF)
        and its Fourier transform (MTF).  Target: MTF50 ≥ 1 cycle / (2×GSD).
        Earmarked for implementation after the L1B flat-fielding milestone.
        """
        from scipy.spatial import KDTree

        if spacing_r is None or spacing_l is None:
            spacing_r, spacing_l = self.pixel_spacing()

        # ── Pre-filter camera pixels to the source bounding box ───────────────
        # The source pixels are already cropped to the GCP region, so we only
        # need camera pixels near that region.  Pre-filter with a generous fixed
        # padding (1 deg ≈ 111 km) before computing the precise search radius.
        pad_pre = 1.0   # degrees — initial spatial pre-filter
        lon_lo = src_lon.min() - pad_pre;  lon_hi = src_lon.max() + pad_pre
        lat_lo = src_lat.min() - pad_pre;  lat_hi = src_lat.max() + pad_pre
        in_bbox = (
            (self.cam_lon >= lon_lo) & (self.cam_lon <= lon_hi) &
            (self.cam_lat >= lat_lo) & (self.cam_lat <= lat_hi)
        )
        valid = np.isfinite(self.cam_lon) & np.isfinite(self.cam_lat) & in_bbox
        valid_ix, valid_iy = np.where(valid)
        n_cam = valid_ix.size
        if n_cam == 0:
            return cam_rad

        # Per-camera-pixel parameters (flattened to valid pixels)
        sr_flat  = spacing_r[valid_ix, valid_iy].astype(np.float64)  # σ_r [km]
        sl_flat  = spacing_l[valid_ix, valid_iy].astype(np.float64)  # σ_l [km]
        cam_lons = self.cam_lon[valid_ix, valid_iy]                   # [deg]
        cam_lats = self.cam_lat[valid_ix, valid_iy]                   # [deg]

        # Radial-axis angle φ = VAA + heading, in the radial/azimuthal frame
        phi_flat = np.radians(
            self.vaa[valid_ix, valid_iy].astype(np.float64) + self.heading
        )
        cos_phi = np.cos(phi_flat)
        sin_phi = np.sin(phi_flat)

        # Search radius: 3σ at the 90th-percentile PSF size within the sub-domain.
        # The 90th percentile covers all valid pixels up to ~55° VZA.
        max_sigma_km = max(np.nanpercentile(sr_flat, 90),
                           np.nanpercentile(sl_flat, 90))
        radius_deg = 3.0 * max_sigma_km / 111.32

        # Report nadir PSF size and cache for external use (e.g. plot titles)
        cx, cy = C.FPA_NX // 2, C.FPA_NY // 2
        sr_nadir = float(spacing_r[cx, cy])
        sl_nadir = float(spacing_l[cx, cy])
        self.psf_sigma_r_nadir = sr_nadir
        self.psf_sigma_l_nadir = sl_nadir
        print(f'  PSF nadir: sigma_r={sr_nadir:.3f} km  sigma_l={sl_nadir:.3f} km  '
              f'search_radius={radius_deg:.3f} deg ({3*max_sigma_km:.1f} km)')

        # KDTree on valid camera pixels
        cam_pts = np.column_stack([cam_lons, cam_lats])
        tree = KDTree(cam_pts)

        weight_sum = np.zeros(n_cam, dtype=np.float64)
        rad_sum    = np.zeros(n_cam, dtype=np.float64)

        src_pts = np.column_stack([src_lon, src_lat])
        indices_list = tree.query_ball_point(src_pts, r=radius_deg)

        km_per_deg = 111.32

        for k, cam_indices in enumerate(indices_list):
            if not cam_indices:
                continue
            cam_idx = np.asarray(cam_indices, dtype=np.intp)

            # Displacement in km — flat-Earth over the PSF footprint
            cos_lat = np.cos(np.radians(
                0.5 * (cam_lats[cam_idx] + src_lat[k])
            ))
            d_lon_km = (cam_lons[cam_idx] - src_lon[k]) * km_per_deg * cos_lat
            d_lat_km = (cam_lats[cam_idx] - src_lat[k]) * km_per_deg

            # Rotate into each camera pixel's radial/azimuthal frame
            cp = cos_phi[cam_idx]
            sp = sin_phi[cam_idx]
            d_r =  d_lon_km * cp + d_lat_km * sp   # radial displacement [km]
            d_l = -d_lon_km * sp + d_lat_km * cp   # azimuthal displacement [km]

            # Ellipsoidal distance squared (units of σ²)
            ell_sq = d_r**2 / sr_flat[cam_idx]**2 + d_l**2 / sl_flat[cam_idx]**2

            # Hard 3σ cutoff in ellipsoidal space — prevents distant source
            # pixels (inside the conservative global search ball but outside
            # this camera pixel's own PSF footprint) from contributing.
            inside = ell_sq <= 9.0    # 3² = 9
            if not inside.any():
                continue
            cam_idx_in = cam_idx[inside]

            w = np.exp(-0.5 * ell_sq[inside])
            np.add.at(weight_sum, cam_idx_in, w)
            np.add.at(rad_sum,    cam_idx_in, w * float(src_rad[k]))

        with np.errstate(invalid='ignore', divide='ignore'):
            has_data = weight_sum > 0
            values = np.full(n_cam, np.nan, dtype=np.float32)
            values[has_data] = (rad_sum[has_data] / weight_sum[has_data]).astype(np.float32)

        cam_rad[valid_ix, valid_iy] = values
        return cam_rad

    # ── Pixel spacing (radial / lateral) ─────────────────────────────────────

    def pixel_spacing(self):
        """
        Compute ground pixel spacing in the radial and lateral/azimuthal
        directions for every FPA pixel.

        Method
        ------
        1. Estimate the Jacobian  d(lon,lat)/d(i,j)  via central finite
           differences on cam_lon / cam_lat (np.gradient handles boundaries).
        2. Convert to physical km/pixel using a local flat-Earth approximation
           (accurate to < 0.01% for single-pixel steps of ~750 m).
        3. Rotate to the per-pixel radial/lateral frame:
              θ[i,j] = atan2(j − cy, i − cx)
           Radial unit vector: (cos θ, sin θ)   [away from optical axis]
           Lateral unit vector: (−sin θ, cos θ) [perpendicular]

        Returns
        -------
        spacing_r : (FPA_NX, FPA_NY) float32 — radial spacing [km]
        spacing_l : (FPA_NX, FPA_NY) float32 — azimuthal spacing [km]

        Notes
        -----
        Pixels where cam_lon/lat are NaN (masked or beyond VZA limit) and
        their immediate neighbours will also be NaN in the output.
        """
        if self._spacing_r is not None and self._spacing_l is not None:
            return self._spacing_r, self._spacing_l

        if self.cam_lon is None:
            self.geomap()

        lon = self.cam_lon.astype(np.float64)   # (NX, NY)
        lat = self.cam_lat.astype(np.float64)

        # ── Jacobian via central differences ─────────────────────────────────
        # np.gradient along axis=0 → d/di;  axis=1 → d/dj
        dlon_di, dlon_dj = np.gradient(lon)   # deg / pixel
        dlat_di, dlat_dj = np.gradient(lat)

        # ── Convert to km / pixel ─────────────────────────────────────────────
        coslat = np.cos(np.radians(lat))
        km_per_deg = 111.32
        ex_ew = dlon_di * km_per_deg * coslat   # E-W component of 1-step in i
        ex_ns = dlat_di * km_per_deg             # N-S component of 1-step in i
        ey_ew = dlon_dj * km_per_deg * coslat   # E-W component of 1-step in j
        ey_ns = dlat_dj * km_per_deg             # N-S component of 1-step in j

        # ── Per-pixel radial angle in FPA ─────────────────────────────────────
        cx, cy = C.FPA_NX // 2, C.FPA_NY // 2
        ii = np.arange(C.FPA_NX)
        jj = np.arange(C.FPA_NY)
        I, J = np.meshgrid(ii, jj, indexing='ij')   # (NX, NY)
        theta = np.arctan2(J - cy, I - cx)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # ── Radial ground displacement ────────────────────────────────────────
        gr_ew = cos_t * ex_ew + sin_t * ey_ew
        gr_ns = cos_t * ex_ns + sin_t * ey_ns
        spacing_r = np.sqrt(gr_ew**2 + gr_ns**2)

        # ── Lateral ground displacement ───────────────────────────────────────
        gl_ew = -sin_t * ex_ew + cos_t * ey_ew
        gl_ns = -sin_t * ex_ns + cos_t * ey_ns
        spacing_l = np.sqrt(gl_ew**2 + gl_ns**2)

        # ── Mask invalid pixels ───────────────────────────────────────────────
        bad = ~(np.isfinite(lon) & np.isfinite(lat))
        spacing_r[bad] = np.nan
        spacing_l[bad] = np.nan

        self._spacing_r = spacing_r.astype(np.float32)
        self._spacing_l = spacing_l.astype(np.float32)
        return self._spacing_r, self._spacing_l

    def plot_pixel_spacing(self, spacing_r=None, spacing_l=None, vmax_km=10.0,
                           vza_circles=(30, 55, 62)):
        """
        Two-panel FPA image of radial and azimuthal pixel spacing, with optional
        VZA contour circles overlaid.

        Parameters
        ----------
        spacing_r, spacing_l : arrays from pixel_spacing() — computed if not supplied
        vmax_km : colour-scale upper limit [km]
        vza_circles : tuple of VZA values [deg] to overlay as contours.
                      Set to () or None to suppress.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        if spacing_r is None or spacing_l is None:
            spacing_r, spacing_l = self.pixel_spacing()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Camera pixel spacing  [km per pixel]')

        kw = dict(origin='lower', cmap='viridis', vmin=0, vmax=vmax_km,
                  aspect='auto')

        for ax, data, title in zip(
            axes,
            [spacing_r, spacing_l],
            ['Radial spacing', 'Azimuthal spacing'],
        ):
            im = ax.imshow(data.T, **kw)
            plt.colorbar(im, ax=ax, label='km / pixel', shrink=0.8)
            ax.set_title(title)
            ax.set_xlabel('FPA x (across-track) [pixels]')
            ax.set_ylabel('FPA y (along-track) [pixels]')

            if vza_circles:
                self._overlay_vza_circles(ax, vza_circles)

        fig.tight_layout()
        return fig

    def plot_fpa_vza_circles(self, vza_levels=(30, 55, 62)):
        """
        Standalone FPA diagram showing VZA contour circles in mm coordinates,
        with the FPA boundary and along-track reference lines.
        Mirrors the cam-hong cam.py panel[1] diagnostic (line 119).

        Parameters
        ----------
        vza_levels : VZA contour values to draw [deg]

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        # FPA mm grid (same construction as cal_vza_vaa)
        cx = C.FPA_NX // 2
        cy = C.FPA_NY // 2
        xg = (np.arange(C.FPA_NX) - cx + 0.5) * C.FPA_DX   # mm
        yg = (np.arange(C.FPA_NY) - cy + 0.5) * C.FPA_DY
        XG, YG = np.meshgrid(xg, yg, indexing='ij')

        # Unmasked VZA (self.vza from cal_vza_vaa, no NaN masking applied)
        vza_all = self.vza   # already the full unmasked array in libcam

        # Colours cycling through levels
        colours = ['blue', 'black', 'red', 'purple', 'orange'][:len(vza_levels)]

        fig, ax = plt.subplots(figsize=(6, 6))
        cs = ax.contour(XG, YG, vza_all, list(vza_levels), colors=colours)
        fmt = {v: f'${v}^\\circ$' for v in vza_levels}
        ax.clabel(cs, fmt=fmt, fontsize=9)

        # FPA boundary box  (±NX/2 × DX mm)
        hw = C.FPA_NX / 2 * C.FPA_DX
        hh = C.FPA_NY / 2 * C.FPA_DY
        for x in [-hw, hw]:
            ax.plot([x, x], [-hh, hh], 'k--', lw=0.8)
        for y in [-hh, hh]:
            ax.plot([-hw, hw], [y, y], 'k--', lw=0.8)

        # Along-track reference lines (centre + ±ii pixels ≈ 30° fw/bw)
        ii = 529   # pixel offset matching original cam-hong code
        ax.axhline(yg[cy],      color='green',  ls='--', lw=1, label='centre')
        ax.axhline(yg[cy + ii], color='blue',   ls='--', lw=1,
                   label=f'${self.vza[cx, cy + ii]:.1f}^\\circ$ fw')
        ax.axhline(yg[cy - ii], color='orange', ls='--', lw=1,
                   label=f'${self.vza[cx, cy - ii]:.1f}^\\circ$ bw')

        ax.set_aspect('equal')
        ax.set_xlabel('Distance from FPA centre [mm]')
        ax.set_ylabel('Distance from FPA centre [mm]')
        ax.set_title(', '.join(str(v) for v in vza_levels) +
                     r' deg VZA circles on FPA')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def pixel_spacing_profile(self, spacing_r=None, spacing_l=None):
        """
        Extract a 1D radial / azimuthal spacing profile along the FPA horizontal
        centreline (j = cy), as a function of signed ground distance from nadir.

        Also computes angular VZA/VAA change per pixel along the centreline.

        Returns
        -------
        dist_km    : (FPA_NX,) float64 — signed ground distance from nadir [km]
                     negative = left of nadir (i < cx), positive = right (i >= cx)
        sr_profile : (FPA_NX,) float32 — radial spacing along centreline [km]
        sl_profile : (FPA_NX,) float32 — azimuthal spacing along centreline [km]
        vz_prof    : (FPA_NX,) float64 — |dVZA/di| along centreline [deg / pixel]
        va_prof    : (FPA_NX,) float64 — |dVAA/dj| at centreline [deg / pixel]
        vza_1d     : (FPA_NX,) float64 — VZA along centreline [deg]
        """
        from libcam.util import cal_geodesic_dist

        if spacing_r is None or spacing_l is None:
            spacing_r, spacing_l = self.pixel_spacing()
        if self.cam_lon is None:
            self.geomap()

        cx, cy = C.FPA_NX // 2, C.FPA_NY // 2
        lons = self.cam_lon[:, cy]
        lats = self.cam_lat[:, cy]
        valid = np.isfinite(lons) & np.isfinite(lats)

        dist_km = np.full(C.FPA_NX, np.nan)
        if valid.any():
            idx = np.where(valid)[0]
            d = cal_geodesic_dist(
                np.full(idx.size, self.cam_lon[cx, cy]),
                np.full(idx.size, self.cam_lat[cx, cy]),
                lons[idx], lats[idx],
            ) / 1000.0                              # m → km
            dist_km[idx] = np.where(idx >= cx, 1, -1) * d

        # Angular VZA change per pixel — radial direction (along i at j=cy)
        vz_prof = np.abs(np.gradient(self.vza[:, cy].astype(np.float64)))

        # Angular VAA change per pixel — azimuthal direction (along j at each i)
        va_col = self.vaa[:, cy].astype(np.float64)   # not used directly
        # Use cross-track gradient: dVAA/dj evaluated at j=cy
        if cy > 0 and cy < C.FPA_NY - 1:
            va_prof = np.abs(self.vaa[:, cy + 1].astype(np.float64)
                             - self.vaa[:, cy - 1].astype(np.float64)) / 2.0
        else:
            va_prof = np.abs(np.gradient(self.vaa[:, cy].astype(np.float64)))

        vza_1d = self.vza[:, cy].astype(np.float64)

        return dist_km, spacing_r[:, cy], spacing_l[:, cy], vz_prof, va_prof, vza_1d

    def _overlay_vza_circles(self, ax, vza_levels):
        """
        Overlay VZA contour circles on an FPA pixel-index plot (imshow axes).
        Converts mm contour coordinates to pixel indices.
        """
        cx = C.FPA_NX // 2
        cy = C.FPA_NY // 2
        xg = (np.arange(C.FPA_NX) - cx + 0.5) * C.FPA_DX
        yg = (np.arange(C.FPA_NY) - cy + 0.5) * C.FPA_DY
        XG, YG = np.meshgrid(xg, yg, indexing='ij')

        XI = XG / C.FPA_DX + cx - 0.5
        YI = YG / C.FPA_DY + cy - 0.5

        colours = ['white'] * len(vza_levels)
        cs = ax.contour(XI, YI, self.vza, list(vza_levels),
                        colors=colours, linewidths=0.8, linestyles='--')
        fmt = {v: f'${v}^\\circ$' for v in vza_levels}
        ax.clabel(cs, fmt=fmt, fontsize=7, colors='white')

    # ── Sub-pixel local coordinate grid (for PSF convolution analysis) ────────

    def local_grid(self):
        """
        Compute the lon/lat coordinates at the *edges* between camera pixels
        (i.e., the grid half-way between pixel centres), along both FPA axes.

        This is used to define the spatial extent of each pixel for PSF
        convolution (analogous to gl2's local_grid function).

        Returns
        -------
        grid_lon_x : (FPA_NX+1, FPA_NY) — lon of pixel boundaries along x
        grid_lat_x : same shape
        grid_lon_y : (FPA_NX, FPA_NY+1) — lon of pixel boundaries along y
        grid_lat_y : same shape
        """
        if self.cam_lon is None:
            self.geomap()

        nx, ny = C.FPA_NX, C.FPA_NY

        def interp_edge(arr, axis):
            """Linear interpolation at half-pixel offset."""
            sl0 = [slice(None)] * 2
            sl1 = [slice(None)] * 2
            sl0[axis] = slice(None, -1)
            sl1[axis] = slice(1, None)
            midpoints = 0.5 * (arr[tuple(sl0)] + arr[tuple(sl1)])
            # Prepend/append edge by extrapolation
            first = 2 * arr[(slice(None), slice(None, 1)) if axis == 1
                            else (slice(None, 1), slice(None))] - midpoints[
                (slice(None), slice(0, 1)) if axis == 1
                else (slice(0, 1), slice(None))]
            last  = 2 * arr[(slice(None), slice(-1, None)) if axis == 1
                            else (slice(-1, None), slice(None))] - midpoints[
                (slice(None), slice(-1, None)) if axis == 1
                else (slice(-1, None), slice(None))]
            return np.concatenate([first, midpoints, last], axis=axis)

        grid_lon_x = interp_edge(self.cam_lon, axis=0)   # (nx+1, ny)
        grid_lat_x = interp_edge(self.cam_lat, axis=0)
        grid_lon_y = interp_edge(self.cam_lon, axis=1)   # (nx, ny+1)
        grid_lat_y = interp_edge(self.cam_lat, axis=1)

        return grid_lon_x, grid_lat_x, grid_lon_y, grid_lat_y

    # ── Mask helpers ──────────────────────────────────────────────────────────

    def make_imagery_stripe_mask(self, nline=21, axis='h'):
        """
        Return a boolean mask (True = *outside* the imagery stripe).

        axis='h' : central horizontal stripe (constant column index range)
        axis='v' : central vertical stripe   (constant row index range)

        The WFOVC imagery stripe is ~21 pixels wide around the FPA centre.
        """
        mask = np.ones((C.FPA_NX, C.FPA_NY), dtype=bool)
        cx, cy = self.fpa_cx, self.fpa_cy
        if axis == 'h':
            mask[:, cy - nline: cy + nline] = False
        elif axis == 'v':
            mask[cx - nline: cx + nline, :] = False
        else:
            raise ValueError("axis must be 'h' or 'v'")
        mask |= (self.vza > C.VZA_LIMIT)
        return mask

    # ── RAPS heading offset ───────────────────────────────────────────────────

    @staticmethod
    def raps_heading_offset(frame_index, ang_raa=None):
        """
        Return the RAPS relative azimuth angle [deg] for a given frame index.

        The default pattern cycles from 0→90 → 90→−90 → −90→0 in 2.5° steps,
        matching the cam-hong implementation.
        """
        if ang_raa is None:
            ang_raa = np.concatenate((
                np.arange(0.0,   90.0,  2.5),
                np.arange(90.0, -90.0, -2.5),
                np.arange(-90.0,  0.0,  2.5),
            ))
        return ang_raa[int(frame_index % ang_raa.size)]


if __name__ == '__main__':
    import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = Platform(lon=-117.0, lat=33.0, dtime=datetime.datetime(2022, 6, 1, 20, 0, 0),
                 heading=0.0)
    p.geomap()
    print('Centre pixel lon/lat:',
          p.cam_lon[p.fpa_cx, p.fpa_cy],
          p.cam_lat[p.fpa_cx, p.fpa_cy])
    print('Expected ~lon=-117, lat=33')

    # ── Pixel spacing test ────────────────────────────────────────────────────
    sr, sl = p.pixel_spacing()
    cx, cy = p.fpa_cx, p.fpa_cy
    print(f'Nadir pixel spacing — radial: {sr[cx, cy]:.3f} km  '
          f'lateral: {sl[cx, cy]:.3f} km  (expected ~0.75 km)')
    print(f'Spacing range — radial: {np.nanmin(sr):.2f}–{np.nanmax(sr):.2f} km  '
          f'lateral: {np.nanmin(sl):.2f}–{np.nanmax(sl):.2f} km')

    import os
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'output')
    os.makedirs(outdir, exist_ok=True)

    # Pixel spacing 2D map with VZA circle overlays (equator)
    fig = p.plot_pixel_spacing(sr, sl, vmax_km=10.0, vza_circles=(30, 55, 62))
    out = os.path.join(outdir, 'pixel_spacing_test.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

    # Standalone VZA circles on FPA (mm coordinates)
    fig2 = p.plot_fpa_vza_circles(vza_levels=(30, 55, 62))
    out2 = os.path.join(outdir, 'fpa_vza_circles.png')
    fig2.savefig(out2, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')

    # ── 1D profile comparison: equator vs 60°N ───────────────────────────────
    print('\nComputing spacing profiles at lat=0 and lat=60...')
    configs = [
        (60.0, '$60^\\circ$ N', 'blue'),
    ]

    # Collect profile data for both figures
    profile_data = []
    cx0 = C.FPA_NX // 2
    cy0 = C.FPA_NY // 2
    for lat_val, label, col in configs:
        p_lat = Platform(lon=0.0, lat=lat_val,
                         dtime=datetime.datetime(2022, 6, 1, 20, 0, 0),
                         heading=0.0)
        p_lat.geomap()
        sr_lat, sl_lat = p_lat.pixel_spacing()
        dist, sr_prof, sl_prof, vz_prof, va_prof, vza_1d = \
            p_lat.pixel_spacing_profile(sr_lat, sl_lat)
        profile_data.append((label, col, dist, sr_prof, sl_prof,
                              vz_prof, va_prof, vza_1d))
        print(f'  lat={lat_val:4.0f} deg  nadir: radial={sr_lat[cx0,cy0]:.3f} km  '
              f'azimuthal={sl_lat[cx0,cy0]:.3f} km')

    def _make_profile_fig(xkey, xlabel, outname):
        """Helper: build profile figure with twin angular y-axis.

        xkey = 'dist'  → x = signed ground distance from nadir [km]
        xkey = 'vza'   → x = VZA [deg], only positive side shown
        """
        fig, ax_km = plt.subplots(figsize=(10, 5))
        fig.suptitle('Pixel spacing profile along FPA centreline')
        ax_ang = ax_km.twinx()

        for label, col, dist, sr_prof, sl_prof, vz_prof, va_prof, vza_1d in profile_data:
            if xkey == 'dist':
                xarr = dist
                valid = np.isfinite(dist) & np.isfinite(sr_prof)
            else:
                # VZA axis: keep only i >= cx0 (positive / right side)
                valid = np.isfinite(vza_1d) & np.isfinite(sr_prof) & (
                    np.arange(C.FPA_NX) >= cx0)
                xarr = vza_1d

            ax_km.plot(xarr[valid], sr_prof[valid], '-',  color=col, lw=1.8,
                       label=f'Radial - {label}')
            ax_km.plot(xarr[valid], sl_prof[valid], '--', color=col, lw=1.8,
                       label=f'Azimuthal - {label}')
            # Angular profiles — only first config (latitude-independent)
            if label == configs[0][1]:
                ax_ang.plot(xarr[valid], vz_prof[valid], '-',  color='gold', lw=1.4,
                            label='VZA change / pixel')
                ax_ang.plot(xarr[valid], va_prof[valid], '--', color='steelblue', lw=1.4,
                            label='VAA change / pixel')

        ax_km.set_xlabel(xlabel)
        ax_km.set_ylabel('Spacing [km / pixel]')
        ax_km.set_ylim(0, 20)
        ax_km.axhline(0.75, color='k', ls=':', lw=0.8, label='0.75 km reference')

        if xkey == 'vza':
            vza_horizon = np.degrees(np.arcsin(
                C.PLANET_RADIUS / (C.PLANET_RADIUS + C.LIBERA_ALTITUDE)))
            ax_km.axvline(vza_horizon, color='gray', ls='--', lw=1.2,
                          label=f'Horizon ({vza_horizon:.1f}$^\\circ$)')
            ax_km.axvline(55.0, color='dimgray', ls=':', lw=1.2,
                          label=r'$55^\circ$ VZA')

        ax_ang.set_ylabel('deg VZA/VAA change / pixel')
        ax_ang.set_ylim(0, 0.2)

        # Combined legend (left axis entries + right axis entries)
        lines_km, labs_km = ax_km.get_legend_handles_labels()
        lines_ang, labs_ang = ax_ang.get_legend_handles_labels()
        ax_km.legend(lines_km + lines_ang, labs_km + labs_ang,
                     fontsize=7, ncol=2, loc='upper left')

        fig.tight_layout()
        out_path = os.path.join(outdir, outname)
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_path}')

    _make_profile_fig('dist',
                      'Ground distance from nadir [km]',
                      'pixel_spacing_profile.png')
    _make_profile_fig('vza',
                      r'VZA [$^\circ$]',
                      'pixel_spacing_profile_vza.png')
