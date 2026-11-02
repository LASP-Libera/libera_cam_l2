"""Standalone RossThick-LiSparse-Reciprocal (RTLSR) land BRDF kernel, and a
Cox-Munk sun-glint ocean BRDF, used for the "no atmosphere" reflectance panel
in run_tile_rt.py.

Why this file exists (for Sebastian's discussion with Hong): Hong's own
pipeline (cam_l2.cld.gen_brdf_shdom / run_sim_rad) builds the same land
fiso/fvol/fgeo (RTLSR) and ocean windspeed BRDF arrays, but only ever
*feeds* them to SHDOM's Fortran surface routine -- the actual kernel
evaluation (land) and Cox-Munk glint model (ocean) happen inside SHDOM/
Fortran and are never exposed as a standalone Python reflectance. For the
"BRDF-only, no atmosphere" panel we want here, no RT solve is needed at
all -- it's pure algebra, evaluated directly in Python from the same
fiso/fvol/fgeo (land) and windspeed (ocean) inputs Hong already reads.

References:
  - Land kernel (RossThick volumetric + LiSparse-Reciprocal geometric),
    the standard MODIS/VIIRS BRDF/Albedo (MOD43 / VJ143) formulation:
      Wanner, Li & Strahler (1995), JGR 100(D10):21077-21089
      Lucht, Schaaf & Strahler (2000), IEEE TGRS 38(2):977-998
    Parameterization used here (the MODIS/VIIRS-standard "AMBRALS" one):
    h/b = 2.0, b/r = 1.0.
  - Ocean sun-glint (isotropic Cox-Munk slope distribution + Fresnel
    reflection off a seawater-like interface):
      Cox, C. & Munk, W. (1954), J. Opt. Soc. Am. 44(11):838-850.
    Mean-square-slope vs. wind-speed relation (commonly used form, e.g.
    Mobley, "Light and Water in the Sea", 2015):
      sigma^2 = 0.003 + 0.00512 * U10   (U10 in m/s)
    This implementation is the standard literature closed form, NOT a
    transcription of SHDOM/propgen's own Fortran glint routine -- flagged
    explicitly (see WARNING below) since Hong's Fortran surface code may
    use a different slope-variance fit, whitecap correction, or refractive
    index, and a side-by-side check against it has not been done. Good
    enough for the qualitative "a little brighter with atmosphere" check
    this first build is for; revisit before trusting absolute glint
    reflectance values.
  - Ice/snow kernel: SHDOM's Fortran surface dispatch does NOT treat
    ice-flagged cells with plain 3-parameter RTLSR. cam_l2.cld.gen_brdf_shdom
    hardcodes a 5-parameter RTLSR-Jiao snow BRDF for them (fiso, fgeo, fvol,
    f_jiao, alpha -- brdf[...,0:5]; see Jiao et al. 2019 below), and SHDOM's
    shd_sfc__brdfSurface dispatches SFC_ID==4 to shd_sfc__brdfRTLSRJ
    (software/shdom/src/shdSfc.f90), not shd_sfc__brdfRTLSR. This file's
    ice branch used to call the plain 3-parameter rtlsr_brdf() -- a
    mismatch against what SHDOM itself evaluates for those cells, found
    during the BRDF diagnostic pass (list item #2). Fixed below:
    rtlsr_jiao_brdf() is a term-for-term Python port of
    shd_sfc__brdfRTLSRJ + shd_sfc__brdfJiao, and tile_noatm_reflectance()'s
    ice branch now calls it with the full 5-parameter set.
      Jiao, Y. et al. (2019), Remote Sensing of Environment, 221:198-209
      https://doi.org/10.1016/j.rse.2018.11.001

Kept dependency-free (numpy only) so it can be unit-tested and used
without the full er3t/SHDOM stack being importable.
"""

import numpy as np

__all__ = [
    "rtlsr_brdf", "cox_munk_glint_brdf", "rtlsr_jiao_brdf",
    "tile_noatm_reflectance",
]


def _safe_tan(theta_rad):
    return np.tan(theta_rad)


def rtlsr_brdf(sza_deg, vza_deg, raz_deg, fiso, fvol, fgeo,
               h_over_b=2.0, b_over_r=1.0):
    """RossThick-LiSparse-Reciprocal BRDF, MODIS/VIIRS AMBRALS convention.

    Parameters
    ----------
    sza_deg, vza_deg : solar / view zenith angle, degrees (scalar or array,
        broadcastable against fiso/fvol/fgeo).
    raz_deg : relative azimuth (view - solar), degrees. Any convention is
        fine (only cos(raz) enters), as long as it is the angle BETWEEN
        the solar and view azimuths.
    fiso, fvol, fgeo : VJ143C1-style kernel-weight arrays (isotropic,
        RossThick-volumetric, LiSparse-R-geometric), same shape, same
        convention the BRDF_Albedo_Parameter{1,2,3}_M4 fields use.
    h_over_b, b_over_r : LiSparse-Reciprocal shape parameters; 2.0/1.0 is
        the fixed MODIS/VIIRS "AMBRALS" standard used operationally for
        VJ143C1, not meant to be tuned per-scene.

    Returns
    -------
    BRDF (dimensionless bidirectional reflectance factor), same shape as
    fiso/fvol/fgeo.
    """
    sza = np.radians(np.asarray(sza_deg, dtype=np.float64))
    vza = np.radians(np.asarray(vza_deg, dtype=np.float64))
    raz = np.radians(np.asarray(raz_deg, dtype=np.float64))
    cos_raz = np.cos(raz)

    cos_sza = np.cos(sza)
    cos_vza = np.cos(vza)
    sin_sza = np.sin(sza)
    sin_vza = np.sin(vza)

    # --- RossThick volumetric kernel -----------------------------------
    cos_xi = cos_sza * cos_vza + sin_sza * sin_vza * cos_raz
    cos_xi = np.clip(cos_xi, -1.0, 1.0)
    xi = np.arccos(cos_xi)
    sin_xi = np.sin(xi)

    denom_rt = cos_sza + cos_vza
    # guard the (rare, grazing-angle) zero-denominator case
    denom_rt = np.where(np.abs(denom_rt) < 1e-6, 1e-6, denom_rt)
    k_vol = ((np.pi / 2.0 - xi) * cos_xi + sin_xi) / denom_rt - np.pi / 4.0

    # --- LiSparse-Reciprocal geometric kernel --------------------------
    tan_sza_p = b_over_r * _safe_tan(sza)
    tan_vza_p = b_over_r * _safe_tan(vza)
    sza_p = np.arctan(tan_sza_p)
    vza_p = np.arctan(tan_vza_p)

    cos_xi_p = np.cos(sza_p) * np.cos(vza_p) + np.sin(sza_p) * np.sin(vza_p) * cos_raz
    cos_xi_p = np.clip(cos_xi_p, -1.0, 1.0)

    sec_sza_p = 1.0 / np.where(np.abs(np.cos(sza_p)) < 1e-9, 1e-9, np.cos(sza_p))
    sec_vza_p = 1.0 / np.where(np.abs(np.cos(vza_p)) < 1e-9, 1e-9, np.cos(vza_p))

    D = np.sqrt(np.clip(
        tan_sza_p**2 + tan_vza_p**2 - 2.0 * tan_sza_p * tan_vza_p * cos_raz,
        0.0, None))
    cos_t = (h_over_b) * np.sqrt(D**2 + (tan_sza_p * tan_vza_p * np.sin(raz))**2) \
        / (sec_sza_p + sec_vza_p)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    t = np.arccos(cos_t)
    sin_t = np.sin(t)

    O = (1.0 / np.pi) * (t - sin_t * cos_t) * (sec_sza_p + sec_vza_p)

    k_geo = O - sec_sza_p - sec_vza_p + 0.5 * (1.0 + cos_xi_p) * sec_sza_p * sec_vza_p

    return fiso + fvol * k_vol + fgeo * k_geo


def cox_munk_glint_brdf(sza_deg, vza_deg, raz_deg, wind_speed_ms,
                          n_water=1.34):
    """Isotropic Cox-Munk sun-glint reflectance (see module docstring for
    the literature form and its caveats relative to SHDOM's own Fortran
    glint routine -- NOT yet verified to match it bit-for-bit).

    Parameters
    ----------
    sza_deg, vza_deg : solar / view zenith angle, degrees.
    raz_deg : relative azimuth (view - solar), degrees.
    wind_speed_ms : 10 m wind speed, m/s (same quantity Hong reads from
        MERRA-2 U10M/V10M).
    n_water : refractive index of seawater used for the Fresnel term
        (1.34 is the usual visible-wavelength value).

    Returns
    -------
    BRDF (dimensionless), same shape as the broadcast inputs. Zero wind
    speed is floored to a small value to avoid a singular (zero-variance)
    specular spike.
    """
    sza = np.radians(np.asarray(sza_deg, dtype=np.float64))
    vza = np.radians(np.asarray(vza_deg, dtype=np.float64))
    raz = np.radians(np.asarray(raz_deg, dtype=np.float64))
    U = np.maximum(np.asarray(wind_speed_ms, dtype=np.float64), 0.5)

    mu_s, mu_v = np.cos(sza), np.cos(vza)
    sin_s, sin_v = np.sin(sza), np.sin(vza)

    # specular (mirror) geometry: angle of the facet normal that would
    # reflect sun -> sensor, and the local incidence angle on that facet
    cos_g = mu_s * mu_v + sin_s * sin_v * np.cos(raz)   # sun-view scattering angle
    cos_g = np.clip(cos_g, -1.0, 1.0)
    # half-angle (incidence angle on the specular facet)
    cos_beta = np.sqrt(np.clip((1.0 + cos_g) / 2.0, 1e-12, 1.0))
    beta = np.arccos(cos_beta)

    # facet-normal zenith angle (tilt needed to specularly reflect sun->view)
    mu_n = (mu_s + mu_v) / (2.0 * cos_beta)
    mu_n = np.clip(mu_n, -1.0, 1.0)
    tan2_tilt = np.clip(1.0 / mu_n**2 - 1.0, 0.0, None)

    sigma2 = 0.003 + 0.00512 * U  # isotropic Cox-Munk mean-square slope

    p_slope = np.exp(-tan2_tilt / sigma2) / (np.pi * sigma2)

    # Fresnel reflectance of the facet at incidence angle beta
    sin_beta = np.sin(beta)
    sin_t = sin_beta / n_water
    sin_t = np.clip(sin_t, -1.0, 1.0)
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, 1.0))
    cos_b = np.cos(beta)
    rs = ((cos_b - n_water * cos_t) / (cos_b + n_water * cos_t))**2
    rp = ((n_water * cos_b - cos_t) / (n_water * cos_b + cos_t))**2
    fresnel = 0.5 * (rs + rp)

    brdf = (np.pi * fresnel * p_slope) / (4.0 * mu_s * mu_v * mu_n**4)
    return np.clip(brdf, 0.0, None)


def rtlsr_jiao_brdf(sza_deg, vza_deg, raz_deg, fiso, fgeo, fvol, f_jiao, alpha,
                     h_over_b=2.0, b_over_r=1.0, cos_min=0.03):
    """RossThick-LiSparse-Reciprocal + Jiao snow kernel (RTLSR-Jiao) BRDF.

    This is a term-for-term Python port of SHDOM's Fortran
    shd_sfc__brdfRTLSRJ() + shd_sfc__brdfJiao() (software/shdom/src/
    shdSfc.f90), which is what SHDOM itself evaluates for ice/snow-flagged
    cells (SFC_ID==4) -- NOT the plain 3-parameter rtlsr_brdf() above. Use
    this (not rtlsr_brdf) for any cell cam_l2.cld.gen_brdf_shdom() flagged
    ice (brdf[...,6]==4.0), so the no-atmosphere panel matches what SHDOM
    actually computes for those cells.

    Parameters
    ----------
    sza_deg, vza_deg : solar / view zenith angle, degrees.
    raz_deg : relative azimuth, degrees, SAME convention already used by
        rtlsr_brdf()/run_tile_rt.py (raz_deg = vaa - saa). The Jiao kernel
        only ever uses cos(relative azimuth) (sin enters only squared);
        SHDOM's own dPhi for this surface type is dPhi = phiInc - phiOut
        = -(raz_deg), and cosine is even, so cos(dPhi) == cos(raz_deg)
        exactly -- no extra sign flip or pi-offset needed here, unlike
        plain RTLSR's 'T' surface type (whose dPhi subtracts pi).
    fiso, fgeo, fvol : RTLSR kernel weights, same convention as rtlsr_brdf().
    f_jiao, alpha : Jiao snow-kernel weight and forward-scattering shape
        parameter (Jiao et al. 2019 suggest alpha in [0.2, 0.3]).
    h_over_b, b_over_r : LiSparse-Reciprocal shape parameters, same
        AMBRALS-standard defaults as rtlsr_brdf().
    cos_min : SHDOM's own grazing-angle floor on cos(SZA)/cos(VZA) (0.03
        in the Fortran); applied here for bit-for-bit consistency.

    Returns
    -------
    BRDF (dimensionless), same broadcast shape as the inputs.
    """
    sza = np.radians(np.asarray(sza_deg, dtype=np.float64))
    vza = np.radians(np.asarray(vza_deg, dtype=np.float64))
    raz = np.radians(np.asarray(raz_deg, dtype=np.float64))
    alpha = np.asarray(alpha, dtype=np.float64)

    cos_i = np.maximum(np.cos(sza), cos_min)
    cos_o = np.maximum(np.cos(vza), cos_min)
    sin_i = np.sqrt(1.0 - cos_i * cos_i)
    sin_o = np.sqrt(1.0 - cos_o * cos_o)
    tan_i = sin_i / cos_i
    tan_o = sin_o / cos_o
    cos_phi = np.cos(raz)
    sin_phi = np.sin(raz)

    # --- K_vol (RossThick) -- identical to shd_sfc__brdfRTLSR's K_vol ---
    cos_xi = cos_i * cos_o + sin_i * sin_o * cos_phi
    cos_xi = np.clip(cos_xi, -1.0, 1.0)
    sin_xi = np.sqrt(1.0 - cos_xi * cos_xi)
    xi = np.arccos(cos_xi)
    k_vol = ((np.pi / 2.0 - xi) * cos_xi + sin_xi) / (cos_i + cos_o) - np.pi / 4.0

    # --- K_geo (LiSparse-Reciprocal) -- identical to brdfRTLSR's K_geo --
    tan_i_p = b_over_r * tan_i
    cos_i_p = 1.0 / np.sqrt(1.0 + tan_i_p * tan_i_p)
    sin_i_p = tan_i_p * cos_i_p
    sec_i_p = 1.0 / cos_i_p

    tan_o_p = b_over_r * tan_o
    cos_o_p = 1.0 / np.sqrt(1.0 + tan_o_p * tan_o_p)
    sin_o_p = tan_o_p * cos_o_p
    sec_o_p = 1.0 / cos_o_p

    cos_xi_p = cos_i_p * cos_o_p + sin_i_p * sin_o_p * cos_phi

    d2 = tan_i_p**2 + tan_o_p**2 - 2.0 * tan_i_p * tan_o_p * cos_phi
    d2 = np.clip(d2, 0.0, None)

    cos_t = h_over_b * np.sqrt(d2 + (tan_i_p * tan_o_p * sin_phi) ** 2) / (sec_i_p + sec_o_p)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_t = np.sqrt(1.0 - cos_t * cos_t)
    t = np.arccos(cos_t)

    omicron = (1.0 / np.pi) * (t - sin_t * cos_t) * (sec_i_p + sec_o_p)
    k_geo = omicron - sec_i_p - sec_o_p + 0.5 * (1.0 + cos_xi_p) * sec_i_p * sec_o_p

    # --- K_j (Jiao snow kernel), shd_sfc__brdfJiao with NORMALIZE=1 -----
    # (shd_sfc__brdfRTLSRJ always calls brdfJiao with NORMALIZE=1.)
    cos_xi_j = sin_i * sin_o * cos_phi + cos_i * cos_o
    cos_xi_j = np.clip(cos_xi_j, -1.0, 1.0)
    xi_j_deg = np.degrees(np.arccos(cos_xi_j))

    k1, k2, k3 = 1.247, 1.186, 5.157
    p_pha = (11.1 * np.exp(-0.087 * (180.0 - xi_j_deg))
             + 1.1 * np.exp(-0.014 * (180.0 - xi_j_deg)))
    r0 = (k1 + k2 * (cos_i + cos_o) + k3 * cos_i * cos_o + p_pha) / (4.0 * (cos_i + cos_o))
    f_g = cos_xi_j * np.exp(-cos_xi_j)
    k0 = 0.4076 * alpha - 1.1081
    k_j = r0 * (1.0 - alpha * f_g) + k0

    brdf = fiso + fvol * k_vol + fgeo * k_geo + f_jiao * k_j
    return np.clip(brdf, 0.0, None)


def tile_noatm_reflectance(sza_deg, vza_deg, raz_deg, brdf_array):
    """Evaluate the no-atmosphere (surface-only) reflectance for a tile,
    given Hong's gen_brdf_shdom()-style brdf[Nx,Ny,7] array (flag in
    slot 6: 1=ocean, 3=land, 4=ice; see cam_l2.cld.gen_brdf_shdom) and a
    SINGLE tile-mean sza/vza/raz (this build's deliberate one-angle-per-
    tile choice -- see run_tile_rt.py docstring). Ice cells (flag==4) use
    the 5-parameter RTLSR-Jiao kernel (rtlsr_jiao_brdf), matching what
    SHDOM's own Fortran surface dispatch evaluates for them -- NOT the
    plain 3-parameter RTLSR used for land (flag==3).

    Returns a (Nx, Ny) reflectance map, and a same-shape surface-type
    code array (1/3/4, matching the brdf flag) for optional diagnostics.
    """
    brdf_array = np.asarray(brdf_array)
    flag = brdf_array[..., 6]
    out = np.full(flag.shape, np.nan, dtype=np.float64)

    logic_land = (flag == 3.0)
    if logic_land.any():
        out[logic_land] = rtlsr_brdf(
            sza_deg, vza_deg, raz_deg,
            fiso=brdf_array[..., 0][logic_land],
            fvol=brdf_array[..., 2][logic_land],
            fgeo=brdf_array[..., 1][logic_land],
        )

    logic_ocean = (flag == 1.0)
    if logic_ocean.any():
        out[logic_ocean] = cox_munk_glint_brdf(
            sza_deg, vza_deg, raz_deg,
            wind_speed_ms=brdf_array[..., 0][logic_ocean],
        )

    logic_ice = (flag == 4.0)
    if logic_ice.any():
        # ice: the 5-parameter RTLSR-Jiao snow kernel, matching what
        # SHDOM's Fortran surface dispatch (SFC_ID==4) actually evaluates
        # for these cells -- NOT plain 3-parameter RTLSR (see module
        # docstring; this was list item #2 of the BRDF diagnostic pass).
        # gen_brdf_shdom hardcodes fiso/fgeo/fvol/f_jiao/alpha in slots
        # 0,1,2,3,4 respectively for ice.
        out[logic_ice] = rtlsr_jiao_brdf(
            sza_deg, vza_deg, raz_deg,
            fiso=brdf_array[..., 0][logic_ice],
            fgeo=brdf_array[..., 1][logic_ice],
            fvol=brdf_array[..., 2][logic_ice],
            f_jiao=brdf_array[..., 3][logic_ice],
            alpha=brdf_array[..., 4][logic_ice],
        )

    return out, flag
