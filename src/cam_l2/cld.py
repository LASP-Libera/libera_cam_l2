#╭────────────────────────────────────────────────────────────────────────────╮!
# @LICENSE_HEADER_START@
#
#   This file is part of L2_CAM.
#
#   --
#   L2_CAM: Code for Processing Libera Camera L2 Products
#   --
#
#   Copyright (C) 2025-
#   EaR³T Community: https://discord.gg/ntqsguwaWv
#
#   L2_CAM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   L2_CAM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with L2_CAM. If not, see <http://www.gnu.org/licenses/>.
#
# @LICENSE_HEADER_END@
#
# History:
#   Created by: Hong Chen
#               hong.chen@hchen.com
#╰────────────────────────────────────────────────────────────────────────────╯!

import os
import sys
import argparse
from collections import OrderedDict
import glob
import datetime
import yaml
import h5py
from netCDF4 import Dataset
import numpy as np

from cam_l2.common import *
from cam_l2.util import *

import er3t


__all__ = ['main_L2_CF_CAM', 'main_L2_CF_RAD']

def cal_threshold_std(data_diff, bins=None, scale=1.0):

    if bins is None:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=100, density=True)
    else:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=bins, density=True)
    bin_center = (bin_edges[1:]+bin_edges[:-1])/2.0

    # standard deviation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    N_max = np.argmax(hist_diff)
    hist_target = np.std(hist_diff) * scale

    hist_diff_left = hist_diff[N_max:]
    N_threshold = np.argmin(np.abs(hist_diff_left-hist_target))
    threshold = bin_center[N_max:][N_threshold]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return threshold


def gaussian(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def cal_threshold_gauss(data_diff, bins=None, scale=1.0):

    if bins is None:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=100, density=True)
    else:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=bins, density=True)
    bin_center = (bin_edges[1:]+bin_edges[:-1])/2.0

    # gaussian
    #╭────────────────────────────────────────────────────────────────────────────╮#
    logic = (hist_diff > 1.0)
    x_ = bin_center[logic]
    y_ = hist_diff[logic]

    p0 = [20.0, 0.0, 1.0]

    coef, tmp = curve_fit(gaussian, x_, y_, p0, maxfev=1000000)

    threshold = np.abs(coef[1]) + np.abs(coef[2] * scale)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return threshold


def cal_threshold_efold(data_diff, bins=None, scale=np.e):

    if bins is None:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=100, density=True)
    else:
        hist_diff, bin_edges = np.histogram(data_diff.ravel(), bins=bins, density=True)
    bin_center = (bin_edges[1:]+bin_edges[:-1])/2.0

    # e-folding
    #╭────────────────────────────────────────────────────────────────────────────╮#
    N_max = np.argmax(hist_diff)
    hist_target = hist_diff[N_max] / scale

    hist_diff_left = hist_diff[N_max:]
    N_threshold = np.argmin(np.abs(hist_diff_left-hist_target))
    threshold = bin_center[N_max:][N_threshold]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return threshold







def read_cam_l1b(
        cfg,
        logger,
        data={},
        coarsenX=1,
        coarsenY=1,
        ):

    logger.info(f"Start <read_cam_l1b> ...")

    fname = f"{fdir_data}/{cfg['input']['cam']['f_rad']['fname']}"
    if os.path.exists(fname):
        with h5py.File(fname, 'r') as f_rad:
            rad = f_rad[cfg['input']['cam']['f_rad']['var_rad']][...][::coarsenY, ::coarsenX]/1000.0
        logger.info(f"Read in CAM-L1B from {fname}")
    else:
        logger.error(f"Missing {fname}")

    fname = f"{fdir_data}/{cfg['input']['cam']['f_geo']['fname']}"
    if os.path.exists(fname):
        with h5py.File(fname, 'r') as f_geo:
            utc = f_geo[cfg['input']['cam']['f_geo']['var_utc']][...]
            lon = f_geo[cfg['input']['cam']['f_geo']['var_lon']][...][::coarsenY, ::coarsenX]
            lat = f_geo[cfg['input']['cam']['f_geo']['var_lat']][...][::coarsenY, ::coarsenX]
            sza = f_geo[cfg['input']['cam']['f_geo']['var_sza']][...][::coarsenY, ::coarsenX]
            saa = f_geo[cfg['input']['cam']['f_geo']['var_saa']][...][::coarsenY, ::coarsenX]
            vza = f_geo[cfg['input']['cam']['f_geo']['var_vza']][...][::coarsenY, ::coarsenX]
            vaa = f_geo[cfg['input']['cam']['f_geo']['var_vaa']][...][::coarsenY, ::coarsenX]
        logger.info(f"Read in CAM-03 from {fname}")
    else:
        logger.error(f"Missing {fname}")

    logic_nan = (lon>180.0) | (lon<-180.0) | (lat>90.0) | (lat<-90.0) | (rad<0.0)
    rad[logic_nan] = np.nan

    jday0 = er3t.util.dtime_to_jday(datetime.datetime(1958, 1, 1))
    jday = jday0 + utc.mean()/86400.0/1.0e6
    data['utc_cam'] = (jday-int(jday)) * 24.0 * 60.0 # UTC minute
    data['lon_cam'] = lon
    data['lat_cam'] = lat
    data['sza_cam'] = sza
    data['saa_cam'] = saa
    data['vza_cam'] = vza
    data['vaa_cam'] = vaa
    data['rad_cam'] = rad
    data['extent'] = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]

    logger.info(f"Ended <read_cam_l1b>")

    return data


def read_surface_mask(
        cfg,
        logger,
        data={},
        ):

    logger.info(f"Start <read_surface_mask> ...")

    # land/ocean mask
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = f"{fdir_data}/{cfg['input']['sfc']['f_land']['fname_geo']}"
    if os.path.exists(fname):
        f = Dataset(fname, 'r')
        lon = f.variables[cfg['input']['sfc']['f_land']['var_lon']][...][0, :]
        lat = f.variables[cfg['input']['sfc']['f_land']['var_lat']][...][:, 0]
        f.close()

        logger.info(f"Read in land-ocean mask Lon/Lat from {fname}")
    else:
        logger.error(f"Missing {fname}")

    lon_indices = np.where((lon>=data['extent'][0])&(lon<=data['extent'][1]))[0]
    lat_indices = np.where((lat>=data['extent'][2])&(lat<=data['extent'][3]))[0]
    index_xs = lon_indices[0]  - 1
    index_xe = lon_indices[-1] + 1
    index_ys = lat_indices[0]  - 1
    index_ye = lat_indices[-1] + 1

    data['lon_sfc'], data['lat_sfc'] = np.meshgrid(lon[index_xs:index_xe], lat[index_ys:index_ye][::-1], indexing='ij')


    fname = f"{fdir_data}/{cfg['input']['sfc']['f_land']['fname_bin']}"
    if os.path.exists(fname):
        data['sfc'] = np.fromfile(fname, dtype='<i1').reshape((4872, 11568))[index_ys:index_ye, index_xs:index_xe][::-1, :].T
        logger.info(f"Read in land-ocean mask from {fname}")
    else:
        logger.error(f"Missing {fname}")

    data['sfc'] = data['sfc'].astype(np.int16)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    logger.info(f"Ended <read_surface_mask>")

    return data


def read_brdf_parms(
        cfg,
        data={},
        ):

    """
    Ocean 10-m wind speed from MERRA2_400.inst1_2d_asm_Nx
        - https://cmr.earthdata.nasa.gov/search/concepts/C1276812820-GES_DISC.html

    Land BRDF from VJ143C1:
        - https://www.earthdata.nasa.gov/data/catalog/lpcloud-vj143c1-002

        - Schaaf, C., Wang, Z., Erb, A., and Paynter, I.: VIIRS/JPSS1 BRDF/Albedo Model
            Parameters Daily L3 Global 0.05° CMG V002, NASA Land Processes Distributed
            Active Archive Center, https://doi.org/10.5067/VIIRS/VJ143C1.002,
            accessed on 15 Jan 2026, 2025.
    """

    # read ocean wind speed
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = Dataset(f"{fdir_data}/{cfg['input']['brdf']['f_ocean']['fname']}", 'r')
    f.set_auto_mask(False)

    utc = f.variables[cfg['input']['brdf']['f_ocean']['var_utc']][...]
    lon = f.variables[cfg['input']['brdf']['f_ocean']['var_lon']][...]
    lat = f.variables[cfg['input']['brdf']['f_ocean']['var_lat']][...]

    index_t = np.argmin(np.abs(utc - np.nanmean(data['utc_cam'])))

    lon_indices = np.where((lon>=data['extent'][0]-0.5)&(lon<=data['extent'][1]+0.5))[0]
    lat_indices = np.where((lat>=data['extent'][2]-0.2)&(lat<=data['extent'][3]+0.2))[0]
    index_xs = lon_indices[0]  - 1
    index_xe = lon_indices[-1] + 1
    index_ys = lat_indices[0]  - 1
    index_ye = lat_indices[-1] + 1

    u10 = f.variables[cfg['input']['brdf']['f_ocean']['var_u10']][...][index_t, index_ys:index_ye, index_xs:index_xe].T
    v10 = f.variables[cfg['input']['brdf']['f_ocean']['var_v10']][...][index_t, index_ys:index_ye, index_xs:index_xe].T

    f.close()

    logic_nan = (u10<-100.0) | (u10>100.0) | (v10<-100.0) | (v10>100.0)

    data['lon_ocean'], data['lat_ocean'] = np.meshgrid(lon[index_xs:index_xe], lat[index_ys:index_ye], indexing='ij')

    data['windspeed_ocean'] = np.sqrt(u10**2.0 + v10**2.0)
    data['windspeed_ocean'][logic_nan] = -999.0
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read land brdf
    #╭────────────────────────────────────────────────────────────────────────────╮#
    f = h5py.File(f"{fdir_data}/{cfg['input']['brdf']['f_land']['fname']}", 'r')

    lon = f[cfg['input']['brdf']['f_land']['var_lon']][...]
    lat = f[cfg['input']['brdf']['f_land']['var_lat']][...]

    lon_indices = np.where((lon>=data['extent'][0])&(lon<=data['extent'][1]))[0]
    lat_indices = np.where((lat>=data['extent'][2])&(lat<=data['extent'][3]))[0]
    index_xs = lon_indices[0]  - 1
    index_xe = lon_indices[-1] + 1
    index_ys = lat_indices[0]  - 1
    index_ye = lat_indices[-1] + 1

    fiso = f[cfg['input']['brdf']['f_land']['var_fiso']][...][index_ys:index_ye, index_xs:index_xe][::-1, :].T
    fvol = f[cfg['input']['brdf']['f_land']['var_fvol']][...][index_ys:index_ye, index_xs:index_xe][::-1, :].T
    fgeo = f[cfg['input']['brdf']['f_land']['var_fgeo']][...][index_ys:index_ye, index_xs:index_xe][::-1, :].T

    f.close()

    logic_nan = (fiso==cfg['input']['brdf']['f_land']['fill_value']) | \
                (fvol==cfg['input']['brdf']['f_land']['fill_value']) | \
                (fgeo==cfg['input']['brdf']['f_land']['fill_value'])
    fiso = fiso * 0.001
    fiso[logic_nan] = -999.0
    fvol = fvol * 0.001
    fvol[logic_nan] = -999.0
    fgeo = fgeo * 0.001
    fgeo[logic_nan] = -999.0

    data['lon_land'], data['lat_land'] = np.meshgrid(lon[index_xs:index_xe], lat[index_ys:index_ye][::-1], indexing='ij')

    data['fiso_land'] = fiso
    data['fvol_land'] = fvol
    data['fgeo_land'] = fgeo
    #╰────────────────────────────────────────────────────────────────────────────╯#

    return data





def gen_brdf_shdom(data={}):

    if 'brdf' not in data.keys():

        lon = data['lon_land']
        lat = data['lat_land']

        Nx, Ny = lon.shape

        mask = er3t.util.find_nearest(data['lon_sfc'], data['lat_sfc'], data['sfc'], lon, lat, Ngrid_limit=5, fill_value=-999)
        mask = np.int_(mask)

        logic_land  = (mask==0) | ((data['fiso_land']>=0.0) & (data['fgeo_land']>=0.0) & (data['fvol_land']>=0.0))
        logic_land_valid = (mask==0) & ((data['fiso_land']>=0.0) & (data['fgeo_land']>=0.0) & (data['fvol_land']>=0.0))

        logic_ocean = (mask==-1)
        logic_ice   = (~logic_ocean) & (~logic_land)

        brdf = np.zeros((Nx, Ny, 7), dtype=np.float32)
        brdf[...] = -999.0

        # Guard added 2026-10 (Sebastian): find_nearest crashes with
        # "IndexError: index 1 is out of bounds for axis 0 with size 0"
        # when logic_land_valid is empty -- i.e. a domain (e.g. a small,
        # purely-oceanic tile) with zero valid VJ143C1 land retrievals to
        # interpolate FROM. Previously unguarded because every domain run
        # here so far has included at least some land.
        if logic_land_valid.any():
            data_tmp = er3t.util.find_nearest(lon[logic_land_valid], lat[logic_land_valid], data['fiso_land'][logic_land_valid], lon[logic_land], lat[logic_land], Ngrid_limit=50, fill_value=0.0)
            brdf[logic_land, 0] = data_tmp
            data_tmp = er3t.util.find_nearest(lon[logic_land_valid], lat[logic_land_valid], data['fgeo_land'][logic_land_valid], lon[logic_land], lat[logic_land], Ngrid_limit=50, fill_value=0.0)
            brdf[logic_land, 1] = data_tmp
            data_tmp = er3t.util.find_nearest(lon[logic_land_valid], lat[logic_land_valid], data['fvol_land'][logic_land_valid], lon[logic_land], lat[logic_land], Ngrid_limit=50, fill_value=0.0)
            brdf[logic_land, 2] = data_tmp
            brdf[logic_land, 6] = 3.0
        elif logic_land.any():
            # cell(s) flagged/targeted as land exist, but none in this
            # domain have a valid VJ143C1 retrieval anywhere in-domain to
            # interpolate from -- can't be filled from local data; leave
            # at the -999 fill value rather than crashing on an empty
            # find_nearest source.
            print("Warning [gen_brdf_shdom]: %d land-flagged cell(s) have no "
                  "valid VJ143C1 retrieval anywhere in this domain to "
                  "interpolate from -- left at fill value (-999)." % int(logic_land.sum()))

        brdf[logic_ice, 0] = 0.986
        brdf[logic_ice, 1] = 0.033
        brdf[logic_ice, 2] = 0.000
        brdf[logic_ice, 3] = 0.447
        brdf[logic_ice, 4] = 0.300
        brdf[logic_ice, 6] = 4.0

        tmp, tmp, data_tmp = er3t.util.grid_by_lonlat(data['lon_ocean'], data['lat_ocean'], data['windspeed_ocean'], \
                lon_1d=lon[:, 0], lat_1d=lat[0, :], method='cubic', fill_value=0.0, Ngrid_limit=5)
        brdf[logic_ocean, 0] = data_tmp[logic_ocean]
        brdf[logic_ocean, 1] = 0.01
        brdf[logic_ocean, 6] = 1.0

        data['brdf'] = brdf

    return data

def lonlat_to_xy(lon, lat, vza, vaa, lon0, lat0, alt0):

    xpos0 = (lon-lon0) * (6378.1370 * np.pi) / 180.0
    ypos0 = (lat-lat0) * (6378.1370 * np.pi) / 180.0

    xpos = -alt0*np.tan(np.deg2rad(vza)) * np.sin(np.deg2rad(vaa)) + xpos0
    ypos = -alt0*np.tan(np.deg2rad(vza)) * np.cos(np.deg2rad(vaa)) + ypos0

    return xpos, ypos

def run_sim_rad(
        data,
        wavelength=555.0,
        solver='3D',
        Niter=10,
        overwrite=True,
        plot=False
        ):

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir = f"tmp-data/run_sim_rad/{wavelength:04.0f}"

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    levels = np.concatenate((np.arange(0.0, 10.0, 2.0), np.arange(10.0, 60.1, 10.0)))
    fname_atm = f'{fdir}/atm.pk'
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_abs = f'{fdir}/abs.pk'
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='coarse', atm_obj=atm0, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': 5.5,
            'dy': 5.5,
            'mixed_brdf': data['brdf'],
            }

    fname_sfc = f'{fdir}/sfc.pk'
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = f"{fdir}/cld.pk"

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([0.0]).reshape((1, 1)), # clear-sky, COT=0.0
            cer=np.array([1.0]).reshape((1, 1)),
            cth=np.array([4.0]).reshape((1, 1)),
            cgt=np.array([2.0]).reshape((1, 1)),
            dz=2.0,
            extent_xy=[0.0, sfc0.data['dx']['data']*sfc0.data['nx']['data'], 0.0, sfc0.data['dy']['data']*sfc0.data['ny']['data']],
            atm_obj=atm0,
            overwrite=overwrite
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname=f"{fdir}/shdom-sfc_{wavelength:04.0f}.txt", overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname=f"{fdir}/shdom-ckd.txt", overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname=f'{fdir}/shdom-prp.txt', fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # sensor geometries
    #╭────────────────────────────────────────────────────────────────────────────╮#
    lon = data['lon_cam']
    lat = data['lat_cam']
    vza = data['vza_cam']

    sza0 = np.nanmean(data['sza_cam'])
    saa0 = np.nanmean(data['saa_cam'])
    vaa = data['vaa_cam']
    saa = data['saa_cam']
    # vaa = vaa - (saa-saa0)

    lon0 = data['lon_land'][0, 0]
    lat0 = data['lat_land'][0, 0]
    alt0 = 705.0
    xpos, ypos = lonlat_to_xy(lon, lat, vza, vaa, lon0, lat0, alt0)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2021, 5, 8),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Ng=abs0.Ng,
            Nmu=8,
            Nphi=16,
            Niter=Niter,
            sol_acc=1.0e-4,
            split_acc=1.0e-4,
            target='radiance',
            solar_zenith_angle=sza0,
            solar_azimuth_angle=saa0,
            sensor_type='camera3',
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_xpos=xpos,
            sensor_ypos=ypos,
            sensor_altitude=alt0,
            sensor_dx=sfc0.data['dx']['data'],
            sensor_dy=sfc0.data['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=4,
            mp_mode='mpi',
            overwrite=True,
            force=False,
            )
    #╰────────────────────────────────────────────────────────────────────────────╯#

    data['rad_rtm'] = np.fromfile(f"{shd0.fnames_out[0]}.sHdOmNG-out", dtype='<f4').reshape(vza.shape, order='C')

    return data


def main_L2_CF_CAM(logger, fname_cfg):

    logger.info(f"Start main_L2_CF_CAM ...")

    cfg = read_cfg(fname_cfg)
    logger.info(f"Loaded input configuration file <{fname_cfg}> ...")

    data_run = {}

    # read camera l1b data (measured radiance, longitude, latitude, and utc)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_run = read_cam_l1b(cfg, logger, data=data_run)
    # key outputs:
    #   data_run['utc_cam']
    #   data_run['lon_cam']
    #   data_run['lat_cam']
    #   data_run['sza_cam']
    #   data_run['saa_cam']
    #   data_run['vza_cam']
    #   data_run['vaa_cam']
    #   data_run['rad_cam']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # read surface mask (land, ocean, snow/ice)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_run = read_surface_mask(cfg, logger, data=data_run)
    # key outputs:
    #   data_run['lon_sfc']
    #   data_run['lat_sfc']
    #   data_run['sfc']
    #╰────────────────────────────────────────────────────────────────────────────╯#
    logger.info(f"Processed surface mask")

    # read surface brdf parameters from reanalysis and/or satellite product files
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_run = read_brdf_parms(cfg, data=data_run)
    # key outputs:
    #   data_run['lon_ocean']
    #   data_run['lat_ocean']
    #   data_run['windspeed_ocean']
    #   data_run['lon_land']
    #   data_run['lat_land']
    #   data_run['fgeo_land']
    #   data_run['fvol_land']
    #   data_run['fiso_land']
    #╰────────────────────────────────────────────────────────────────────────────╯#
    logger.info(f"Processed surface BRDF")

    # generate surface input parameters for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_run = gen_brdf_shdom(data=data_run)
    # key output:
    #   data_run['brdf']
    #╰────────────────────────────────────────────────────────────────────────────╯#
    logger.info(f"Generated surface file for SHDOM")

    # run RT to simulation radiance under clear-sky
    #╭────────────────────────────────────────────────────────────────────────────╮#
    data_run = run_sim_rad(data_run)
    # key output:
    #   data_run['rad_rtm']
    #╰────────────────────────────────────────────────────────────────────────────╯#
    logger.info(f"Completed SHDOM run")

    # retrieve cloud fraction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    rad_diff = data_run['rad_cam'] - data_run['rad_rtm']
    threshold = cal_threshold_efold(rad_diff, bins=np.linspace(-0.5, 1.5, 201))
    data_run['cld_msk'] = (rad_diff>threshold)
    # key output:
    #   data_run['cld_mask']
    #╰────────────────────────────────────────────────────────────────────────────╯#
    logger.info(f"Generated cloud mask")


    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # h5f = h5py.File(cfg['output']['f_cld_msk']['fname'], 'w')
    # h5d_extent  = h5f.create_dataset('extent',  data=data_run['extent'],  compression='gzip', compression_opts=9, chunks=True)
    # h5d_lon_cam = h5f.create_dataset('lon_cam', data=data_run['lon_cam'], compression='gzip', compression_opts=9, chunks=True)
    # h5d_lat_cam = h5f.create_dataset('lat_cam', data=data_run['lat_cam'], compression='gzip', compression_opts=9, chunks=True)
    # h5d_rad_cam = h5f.create_dataset('rad_cam', data=data_run['rad_cam'], compression='gzip', compression_opts=9, chunks=True)
    # h5d_rad_rtm = h5f.create_dataset('rad_rtm', data=data_run['rad_rtm'], compression='gzip', compression_opts=9, chunks=True)
    # h5d_cld_msk = h5f.create_dataset('cld_msk', data=data_run['cld_msk'], compression='gzip', compression_opts=9, chunks=True)
    # h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # output
    #╭────────────────────────────────────────────────────────────────────────────╮#
    Nsamp = data_run['lon_cam'].size

    data_out = {}
    # data_out['time'] = data_run['utc_cam'].ravel().astype(np.float64)
    data_out['time'] = np.full((Nsamp,), datetime.datetime(2027, 1, 1))
    data_out['longitude'] = data_run['lon_cam'].ravel().astype(np.float64)
    data_out['latitude']  = data_run['lat_cam'].ravel().astype(np.float64)
    data_out['cloud_fraction'] = data_run['cld_msk'].ravel().astype(np.float64)

    data_out['cloud_fraction_uncertainty'] = \
            np.ones(Nsamp).ravel().astype(np.float64)
    data_out['quality_flags'] = \
            np.ones(Nsamp).ravel().astype(np.int32)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    logger.info(f"Ended main_L2_CF_CAM")

    return data_out



def main_L2_CF_RAD(logger, Nsamp=10):

    logger.info(f"Creating output arrays of size {Nsamp}")

    data = {}

    data['time'] = np.full((Nsamp,), datetime.datetime(2027, 1, 1))
    data['longitude'] = np.linspace(0.0, 1.0, Nsamp, dtype=np.float64)
    data['latitude']  = np.linspace(0.0, 1.0, Nsamp, dtype=np.float64)
    data['cloud_fraction'] = np.ones(Nsamp, dtype=np.float64)
    data['cloud_fraction_uncertainty'] = np.ones(Nsamp, dtype=np.float64)
    data['quality_flags']  = np.ones(Nsamp, dtype=np.int32)

    return data


if __name__ == '__main__':

    logger = get_logger()
    fname_cfg = f"{fdir_data}/sample/input.yml"
    data = main_L2_CF_CAM(logger, fname_cfg)

    pass
