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
import warnings
import h5py
import time
import numpy as np
import datetime
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
import matplotlib.image as mpl_img
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams, ticker
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import cartopy.crs as ccrs
# mpl.use('Agg')


import er3t




# global variables
#╭────────────────────────────────────────────────────────────────────────────╮#
name_tag = '00_er3t_shd'
fdir0 = '.'
Ncpu = 4
rcParams['font.size'] = 14
#╰────────────────────────────────────────────────────────────────────────────╯#



def sim_rad_atm1d_clear_over_forest(
        wavelength=550.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s/%4.4d' % (fdir0, name_tag, _metadata['Function'], wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % (fdir)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([0.0]).reshape((1, 1)),
            cer=np.array([1.0]).reshape((1, 1)),
            cth=np.array([1.0]).reshape((1, 1)),
            cgt=np.array([0.5]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vegetated land
    index_xs = 170; index_xe = 190; index_ys = 110; index_ye = 130

    # deserted land
    # index_xs = 30; index_xe = 50; index_ys = 10; index_ye = 30

    wvl_band = np.array([470.0, 555.0, 659.0], dtype=np.float32)
    index_band = np.argmin(np.abs(wvl_band-wavelength))
    wvl_band0 = wvl_band[index_band]
    print(wavelength, wvl_band0)

    f = h5py.File('data/surface_albedo_land.h5', 'r')
    fiso = f[f'fiso_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    fvol = f[f'fvol_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    fgeo = f[f'fgeo_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([fiso]).reshape((1, 1)),
            'fvol': np.array([fvol]).reshape((1, 1)),
            'fgeo': np.array([fgeo]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % (fdir)
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    sza = 30.0

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Niter=1000,
            Nmu=32,
            Nphi=64,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def sim_rad_atm1d_clear_over_desert(
        wavelength=550.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s/%4.4d' % (fdir0, name_tag, _metadata['Function'], wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % (fdir)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([0.0]).reshape((1, 1)),
            cer=np.array([1.0]).reshape((1, 1)),
            cth=np.array([1.0]).reshape((1, 1)),
            cgt=np.array([0.5]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vegetated land
    # index_xs = 170; index_xe = 190; index_ys = 110; index_ye = 130

    # deserted land
    index_xs = 30; index_xe = 50; index_ys = 10; index_ye = 30

    wvl_band = np.array([470.0, 555.0, 659.0], dtype=np.float32)
    index_band = np.argmin(np.abs(wvl_band-wavelength))
    wvl_band0 = wvl_band[index_band]
    print(wavelength, wvl_band0)

    f = h5py.File('data/surface_albedo_land.h5', 'r')
    fiso = f[f'fiso_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    fvol = f[f'fvol_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    fgeo = f[f'fgeo_{int(wvl_band0)}'][...][index_xs:index_xe, index_ys:index_ye].mean()
    f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([fiso]).reshape((1, 1)),
            'fvol': np.array([fvol]).reshape((1, 1)),
            'fgeo': np.array([fgeo]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % (fdir)
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    sza = 30.0

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Niter=1000,
            Nmu=32,
            Nphi=64,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def sim_rad_atm1d_clear_over_snow(
        wavelength=555.0,
        solver='IPA',
        overwrite=True,
        plot=True
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs_%4.4d.pk' % (fdir, wavelength)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([0.0]).reshape((1, 1)),
            cer=np.array([1.0]).reshape((1, 1)),
            cth=np.array([1.5]).reshape((1, 1)),
            cgt=np.array([1.0]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'fiso': np.array([0.986]).reshape((1, 1)),
            'fgeo': np.array([0.033]).reshape((1, 1)),
            'fvol': np.array([0.000]).reshape((1, 1)),
            'fj'  : np.array([0.447]).reshape((1, 1)),
            'alpha': np.array([0.3]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # vaa = np.arange(0.0, 360.1, 2.0)
    # vza = np.repeat(30.0, vaa.size)

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(1.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    sza = 63.0
    saa = 0.0
    raa = er3t.util.util.calculate_raa(saa=saa, vaa=vaa, forward_scattering='positive')

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Niter=1000,
            Nmu=32,
            Nphi=64,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=saa,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/%4.4d/rad_%s' % (fdir, wavelength, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s.png' % (name_tag, _metadata['Function'], solver.lower())

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape)
        cs = ax1.pcolormesh(np.deg2rad(raa).reshape(vaa_2d.shape), vza_2d, data, cmap='RdBu', vmin=0.15, vmax=0.5)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')
        cbar.ax.set_title('Radiance')

        ax1.set_title('Radiance at %.1f nm (SZA=%5.1f$^\\circ$, %s Mode)' % (wavelength, sza, solver))
        fig.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def sim_rad_atm1d_cloud_over_ocean(
        wavelength=555.0,
        cot=50.0,
        cer=12.0,
        sza=30.0,
        windspeed=1.0,
        pigment=0.01,
        solver='IPA',
        overwrite=True,
        plot=False,
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    fdir='%s/tmp-data/%s/%s/%04.1f_%04.1f_%04.1f/%4.4d' % (fdir0, name_tag, _metadata['Function'], cot, cer, sza, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % (fdir)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([cot]).reshape((1, 1)),
            cer=np.array([cer]).reshape((1, 1)),
            cth=np.array([1.5]).reshape((1, 1)),
            cgt=np.array([1.0]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([windspeed]).reshape((1, 1)),
            'pigment': np.array([pigment]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % (fdir), overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % (fdir), overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % (fdir), fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Ng=abs0.Ng,
            Niter=1000,
            Nmu=16,
            Nphi=32,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s_%4.4d.png' % (name_tag, _metadata['Function'], solver.lower(), wavelength)

        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        # data = out0.data['rad']['data'][:].reshape(vaa_2d.shape) * np.pi / (out0.data['toa']['data']*np.cos(np.deg2rad(sza)))
        data = out0.data['rad']['data'][:].reshape((vaa_1d.size, vza_1d.size))
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')

        ax1.set_title(f'Radiance at {wavelength:.1f} nm (SZA={sza:5.1f}$^\\circ$, {solver} Mode)')

        # plot2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(122)
        cs = ax2.plot(vaa_1d, data[:, np.argmin(np.abs(vza_1d-30.0))]) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax2.set_xlim((0, 1))
        # ax2.set_ylim((0, 1))
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # ax2.set_title('Plot2')
        # ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#

        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def sim_rad_atm1d_clear_over_ocean(
        wavelength=550.0,
        windspeed=1.0,
        pigment=0.01,
        sza=30.0,
        solver='IPA',
        overwrite=True,
        plot=False,
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s/%04.1f_%04.2f_%04.1f/%4.4d' % (fdir0, name_tag, _metadata['Function'], windspeed, pigment, sza, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.append(np.arange(0.0, 2.0, 0.1), np.arange(2.0, 40.1, 2.0))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % (fdir)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([0.0]).reshape((1, 1)),
            cer=np.array([1.0]).reshape((1, 1)),
            cth=np.array([1.5]).reshape((1, 1)),
            cgt=np.array([1.0]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([windspeed]).reshape((1, 1)),
            'pigment': np.array([pigment]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % fdir, overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % fdir, overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % fdir, fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Niter=1000,
            Nmu=32,
            Nphi=64,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s_%4.4d.png' % (name_tag, _metadata['Function'], solver.lower(), wavelength)

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        data = out0.data['rad']['data'][:].reshape(vaa_2d.shape) * np.pi / (out0.data['toa']['data']*np.cos(np.deg2rad(sza)))
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')

        ax1.set_title(f'Reflectance at {wavelength:.1f} nm (SZA={sza:5.1f}$^\\circ$, {solver} Mode)')
        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def sim_rad_atm1d_high_cloud_over_ocean(
        wavelength=555.0,
        cot=50.0,
        cer=12.0,
        sza=30.0,
        windspeed=1.0,
        pigment=0.01,
        solver='IPA',
        overwrite=True,
        plot=False,
        ):

    """
    Similar to test_02 but for calculating radiance fields using LES data (nadir radiance at
    the satellite altitude of 705km)

    Additionally, Mie phase function is used instead of HG that was used in test_01 - test_04

    To run this test, we will need data/00_er3t_mca/aux/les.nc
    """

    _metadata   = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    fdir='%s/tmp-data/%s/%s' % (fdir0, name_tag, _metadata['Function'])
    fdir='%s/tmp-data/%s/%s/%04.1f_%04.1f_%04.1f/%4.4d' % (fdir0, name_tag, _metadata['Function'], cot, cer, sza, wavelength)

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # define an atmosphere object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # levels: altitude of the layer interface in km, here, levels will be 0.0, 0.2, ..., 4.0, 6.0, ...., 20.0
    levels = np.concatenate((np.arange(0.0, 8.0, 2.0), np.arange(8.0, 10.0, 0.1), np.arange(10.0, 40.1, 2.0)))

    # file name of the pickle file for atmosphere
    fname_atm = '%s/atm.pk' % fdir

    # atmosphere object
    atm0      = er3t.pre.atm.atm_atmmod(levels=levels, fname=fname_atm, overwrite=overwrite)

    # data can be accessed at
    #     atm0.lev['altitude']['data']
    #     atm0.lev['pressure']['data']
    #     atm0.lev['temperature']['data']
    #     atm0.lev['h2o']['data']
    #     atm0.lev['o3']['data']
    #     atm0.lev['o2']['data']
    #     atm0.lev['co2']['data']
    #     atm0.lev['ch4']['data']
    #
    #     atm0.lay['altitude']['data']
    #     atm0.lay['pressure']['data']
    #     atm0.lay['temperature']['data']
    #     atm0.lay['h2o']['data']
    #     atm0.lay['o3']['data']
    #     atm0.lay['o2']['data']
    #     atm0.lay['co2']['data']
    #     atm0.lay['ch4']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an absorption object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # file name of the pickle file for absorption
    fname_abs = '%s/abs.pk' % (fdir)

    # absorption object
    abs0 = er3t.pre.abs.abs_rep(wavelength=wavelength, fname=fname_abs, target='fine', atm_obj=atm0, overwrite=overwrite)

    # data can be accessed at
    #     abs0.coef['wavelength']['data']
    #     abs0.coef['abso_coef']['data']
    #     abs0.coef['slit_func']['data']
    #     abs0.coef['solar']['data']
    #     abs0.coef['weight']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define an cloud object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname_cld = '%s/cld.pk' % fdir

    cld0 = er3t.pre.cld.cld_gen_cop(
            fname=fname_cld,
            cot=np.array([cot]).reshape((1, 1)),
            cer=np.array([cer]).reshape((1, 1)),
            cth=np.array([10.0]).reshape((1, 1)),
            cgt=np.array([1.0]).reshape((1, 1)),
            dz=0.1,
            extent_xy=[0.0, 1.0, 0.0, 1.0],
            atm_obj=atm0,
            overwrite=overwrite
            )

    # data can be accessed at
    #     cld0.lay['x']['data']
    #     cld0.lay['y']['data']
    #     cld0.lay['nx']['data']
    #     cld0.lay['ny']['data']
    #     cld0.lay['dx']['data']
    #     cld0.lay['dy']['data']
    #     cld0.lay['altitude']['data']
    #     cld0.lay['extinction']['data']
    #     cld0.lay['temperature']['data']
    #
    #     cld0.lev['altitude']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # sfc object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_dict = {
            'dx': cld0.lay['dx']['data']*cld0.lay['nx']['data'],
            'dy': cld0.lay['dy']['data']*cld0.lay['ny']['data'],
            'windspeed': np.array([windspeed]).reshape((1, 1)),
            'pigment': np.array([pigment]).reshape((1, 1)),
            }

    fname_sfc = '%s/sfc.pk' % fdir
    sfc0 = er3t.pre.sfc.sfc_2d_gen(sfc_dict=sfc_dict, fname=fname_sfc, overwrite=overwrite)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # generate surface, property files for SHDOM
    #╭────────────────────────────────────────────────────────────────────────────╮#
    sfc_2d = er3t.rtm.shd.shd_sfc_2d(atm_obj=atm0, sfc_obj=sfc0, fname='%s/shdom-sfc.txt' % (fdir), overwrite=overwrite)

    atm1d0  = er3t.rtm.shd.shd_atm_1d(atm_obj=atm0, abs_obj=abs0, fname='%s/shdom-ckd.txt' % (fdir), overwrite=overwrite)
    atm_1ds = [atm1d0]

    atm3d0  = er3t.rtm.shd.shd_atm_3d(atm_obj=atm0, abs_obj=abs0, cld_obj=cld0, fname='%s/shdom-prp.txt' % (fdir), fname_atm_1d=atm1d0.fname, overwrite=overwrite)
    atm_3ds = [atm3d0]
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # define shdom object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)
    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')
    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()

    # run shdom
    shd0 = er3t.rtm.shd.shdom_ng(
            date=datetime.datetime(2024, 5, 18),
            atm_1ds=atm_1ds,
            atm_3ds=atm_3ds,
            surface=sfc_2d,
            Ng=abs0.Ng,
            Niter=1000,
            Nmu=16,
            Nphi=32,
            sol_acc=1.0e-6,
            target='radiance',
            solar_zenith_angle=sza,
            solar_azimuth_angle=0.0,
            sensor_zenith_angles=vza,
            sensor_azimuth_angles=vaa,
            sensor_altitude=705.0,
            sensor_dx=cld0.lay['dx']['data'],
            sensor_dy=cld0.lay['dy']['data'],
            fdir='%s/rad_%s' % (fdir, solver.lower()),
            solver=solver,
            Ncpu=1,
            mp_mode='mpi',
            overwrite=overwrite,
            force=True,
            )

    # data can be accessed at
    #     shd0.Ng
    #     shd0.nml         (Ng), e.g., shd0.nml[0], namelist for the first g of the first run
    #     shd0.fnames_inp  (Ng), e.g., shd0.fnames_inp[0], input file name for the first g of the first run
    #     shd0.fnames_out  (Ng), e.g., shd0.fnames_out[0], output file name for the first g of the first run
    #     shd0.fnames_sav  (Ng), e.g., shd0.fnames_sav[0], state-sav file name for the first g of the first run
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # define shdom output object
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # read shdom output files (binary) and save the data into h5 file
    # The mode can be specified as 'all', 'mean', 'std', if 'all' is specified, the data will have last
    # dimension of number of runs
    # e.g.,
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='std' , squeeze=True, verbose=True, overwrite=True)
    # out0 = shd_out_ng(fname='shd-out-rad-3d_les.h5', shd_obj=shd0, abs_obj=abs0, mode='all' , squeeze=True, verbose=True, overwrite=True)

    fname_h5 = '%s/shd-out-rad-%s_%s.h5' % (fdir, solver.lower(), _metadata['Function'])
    out0 = er3t.rtm.shd.shd_out_ng(fname=fname_h5, shd_obj=shd0, abs_obj=abs0, mode='mean', squeeze=True, verbose=True, overwrite=overwrite)

    # data can be accessed at
    #     out0.data['rad']['data']
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if plot:
        fname_png = '%s-%s_%s_%4.4d.png' % (name_tag, _metadata['Function'], solver.lower(), wavelength)

        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(121, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        # data = out0.data['rad']['data'][:].reshape(vaa_2d.shape) * np.pi / (out0.data['toa']['data']*np.cos(np.deg2rad(sza)))
        data = out0.data['rad']['data'][:].reshape((vaa_1d.size, vza_1d.size))
        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, data, cmap='jet')

        cbar = fig.colorbar(cs, ax=ax1, shrink=0.5, aspect=30, pad=0.1, location='bottom')

        ax1.set_title(f'Radiance at {wavelength:.1f} nm (SZA={sza:5.1f}$^\\circ$, {solver} Mode)')

        # plot2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(122)
        cs = ax2.plot(vaa_1d, data[:, np.argmin(np.abs(vza_1d-30.0))]) #, extent=extent, vmin=0.0, vmax=0.5)
        # ax2.set_xlim((0, 1))
        # ax2.set_ylim((0, 1))
        # ax2.set_xlabel('X')
        # ax2.set_ylabel('Y')
        # ax2.set_title('Plot2')
        # ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax2.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#

        plt.savefig(fname_png, bbox_inches='tight')
        plt.close(fig)
    #╰────────────────────────────────────────────────────────────────────────────╯#


    # References
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # er3t.util.print_reference()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def cal_vis_clear_over_forest(
        ):

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()
    sza0 = 30.0

    wvl = np.arange(350.0, 701.0, 5.0)

    toa = np.zeros(wvl.size, dtype=np.float32)
    rad = np.zeros((vaa.size, wvl.size), dtype=np.float32)

    for i, wvl0 in enumerate(wvl):

        fname = f'tmp-data/00_er3t_shd/sim_rad_atm1d_clear_over_forest/{int(wvl0):04d}/shd-out-rad-ipa_sim_rad_atm1d_clear_over_forest.h5'
        f = h5py.File(fname, 'r')
        toa_ = f['mean/toa'][...]
        rad_ = f['mean/rad'][...]
        toa[i] = toa_
        rad[:, i] = rad_

        if (wvl0 == 555.0):
            ref_555 = np.pi * rad_ / (toa_*np.cos(np.deg2rad(sza0)))

        f.close()

    toa_vis = np.trapz(toa, x=wvl)
    ref_vis = np.pi * np.trapz(rad, x=wvl, axis=1) / (toa_vis*np.cos(np.deg2rad(sza0)))

    ref_555 = ref_555.reshape(vaa_2d.shape)
    ref_vis = ref_vis.reshape(vaa_2d.shape)

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname = f'{_metadata_['Function']}.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        fig = plt.figure(figsize=(24, 7))

        fig.suptitle(f'Clear over Forest', fontsize=24, y=1.02)

        # plot 1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555, cmap='jet', vmin=0.0, vmax=0.2)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax1.set_title('$Ref_{555}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(133, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rgrids(np.arange(20, 81, 20))
        ax2.set_rlim((0.0, 89.0))

        cs = ax2.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_vis, cmap='jet', vmin=0.0, vmax=0.2)
        cbar = fig.colorbar(cs, ax=ax2, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax2.set_title('$Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(132, projection='polar')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_rgrids(np.arange(20, 81, 20))
        ax3.set_rlim((0.0, 89.0))

        cs = ax3.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555/ref_vis, cmap='viridis', vmin=0.4, vmax=0.8)
        cbar = fig.colorbar(cs, ax=ax3, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax3.set_title('$Ref_{555} / Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        fname_fig = f'{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cal_vis_clear_over_desert(
        ):

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()
    sza0 = 30.0

    wvl = np.arange(350.0, 701.0, 5.0)

    toa = np.zeros(wvl.size, dtype=np.float32)
    rad = np.zeros((vaa.size, wvl.size), dtype=np.float32)

    for i, wvl0 in enumerate(wvl):

        fname = f'tmp-data/00_er3t_shd/sim_rad_atm1d_clear_over_desert/{int(wvl0):04d}/shd-out-rad-ipa_sim_rad_atm1d_clear_over_desert.h5'
        f = h5py.File(fname, 'r')
        toa_ = f['mean/toa'][...]
        rad_ = f['mean/rad'][...]
        toa[i] = toa_
        rad[:, i] = rad_

        if (wvl0 == 555.0):
            ref_555 = np.pi * rad_ / (toa_*np.cos(np.deg2rad(sza0)))

        f.close()

    toa_vis = np.trapz(toa, x=wvl)
    ref_vis = np.pi * np.trapz(rad, x=wvl, axis=1) / (toa_vis*np.cos(np.deg2rad(sza0)))

    ref_555 = ref_555.reshape(vaa_2d.shape)
    ref_vis = ref_vis.reshape(vaa_2d.shape)

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname = f'{_metadata_['Function']}.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        fig = plt.figure(figsize=(24, 7))

        fig.suptitle(f'Clear over Desert', fontsize=24, y=1.02)

        # plot 1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555, cmap='jet', vmin=0.1, vmax=0.4)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax1.set_title('$Ref_{555}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(133, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rgrids(np.arange(20, 81, 20))
        ax2.set_rlim((0.0, 89.0))

        cs = ax2.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_vis, cmap='jet', vmin=0.1, vmax=0.4)
        cbar = fig.colorbar(cs, ax=ax2, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax2.set_title('$Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(132, projection='polar')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_rgrids(np.arange(20, 81, 20))
        ax3.set_rlim((0.0, 89.0))

        cs = ax3.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555/ref_vis, cmap='viridis', vmin=0.4, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax3, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax3.set_title('$Ref_{555} / Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        fname_fig = f'{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cal_vis_clear_over_ocean(
        windspeed
        ):

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()
    sza0 = 30.0

    wvl = np.arange(350.0, 701.0, 5.0)

    toa = np.zeros(wvl.size, dtype=np.float32)
    rad = np.zeros((vaa.size, wvl.size), dtype=np.float32)

    for i, wvl0 in enumerate(wvl):

        fname = f'tmp-data/00_er3t_shd/sim_rad_atm1d_clear_over_ocean/{windspeed:04.1f}_0.01_30.0/{int(wvl0):04d}/shd-out-rad-ipa_sim_rad_atm1d_clear_over_ocean.h5'
        f = h5py.File(fname, 'r')
        toa_ = f['mean/toa'][...]
        rad_ = f['mean/rad'][...]
        toa[i] = toa_
        rad[:, i] = rad_

        if (wvl0 == 555.0):
            ref_555 = np.pi * rad_ / (toa_*np.cos(np.deg2rad(sza0)))

        f.close()

    toa_vis = np.trapz(toa, x=wvl)
    ref_vis = np.pi * np.trapz(rad, x=wvl, axis=1) / (toa_vis*np.cos(np.deg2rad(sza0)))

    ref_555 = ref_555.reshape(vaa_2d.shape)
    ref_vis = ref_vis.reshape(vaa_2d.shape)

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname = f'{_metadata_['Function']}_{windspeed:04.1f}.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        fig = plt.figure(figsize=(24, 7))

        fig.suptitle(f'Clear over Ocean ({windspeed: .1f} m/s)', fontsize=24, y=1.02)

        # plot 1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555, cmap='jet', vmin=0.0, vmax=0.6)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax1.set_title('$Ref_{555}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(133, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rgrids(np.arange(20, 81, 20))
        ax2.set_rlim((0.0, 89.0))

        cs = ax2.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_vis, cmap='jet', vmin=0.0, vmax=0.6)
        cbar = fig.colorbar(cs, ax=ax2, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax2.set_title('$Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(132, projection='polar')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_rgrids(np.arange(20, 81, 20))
        ax3.set_rlim((0.0, 89.0))

        cs = ax3.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555/ref_vis, cmap='viridis', vmin=0.6, vmax=1.2)
        cbar = fig.colorbar(cs, ax=ax3, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax3.set_title('$Ref_{555} / Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        fname_fig = f'{_metadata_['Function']}_{windspeed:04.1f}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cal_vis_cloud_over_ocean(
        cot,
        cer,
        ):

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()
    sza0 = 30.0

    wvl = np.arange(350.0, 701.0, 5.0)

    toa = np.zeros(wvl.size, dtype=np.float32)
    rad = np.zeros((vaa.size, wvl.size), dtype=np.float32)

    for i, wvl0 in enumerate(wvl):

        fname = f'tmp-data/00_er3t_shd/sim_rad_atm1d_cloud_over_ocean/{cot:04.1f}_{cer:04.1f}_30.0/{int(wvl0):04d}/shd-out-rad-ipa_sim_rad_atm1d_cloud_over_ocean.h5'
        f = h5py.File(fname, 'r')
        toa_ = f['mean/toa'][...]
        rad_ = f['mean/rad'][...]
        toa[i] = toa_
        rad[:, i] = rad_

        if (wvl0 == 555.0):
            rad_555 = rad_.reshape(vaa_2d.shape)
            ref_555 = np.pi * rad_ / (toa_*np.cos(np.deg2rad(sza0)))

        f.close()

    toa_vis = np.trapz(toa, x=wvl)
    ref_vis = np.pi * np.trapz(rad, x=wvl, axis=1) / (toa_vis*np.cos(np.deg2rad(sza0)))

    ref_555 = ref_555.reshape(vaa_2d.shape)
    ref_vis = ref_vis.reshape(vaa_2d.shape)

    # return ref_555, ref_vis

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname = f'{_metadata_['Function']}_{cot:04.1f}_{cer:04.1f}.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if False:
        fig = plt.figure(figsize=(24, 7))

        fig.suptitle(f'Cloud over Ocean (COT={cot:.1f}, CER={cer:.1f}$\\mu m$)', fontsize=24, y=1.02)

        # plot 1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555, cmap='jet', vmin=0.2, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax1.set_title('$Ref_{555}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(133, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rgrids(np.arange(20, 81, 20))
        ax2.set_rlim((0.0, 89.0))

        cs = ax2.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_vis, cmap='jet', vmin=0.2, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax2, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax2.set_title('$Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(132, projection='polar')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_rgrids(np.arange(20, 81, 20))
        ax3.set_rlim((0.0, 89.0))

        cs = ax3.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555/ref_vis, cmap='viridis', vmin=0.8, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax3, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax3.set_title('$Ref_{555} / Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        fname_fig = f'{_metadata_['Function']}_{cot:04.1f}_{cer:04.1f}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def cal_vis_high_cloud_over_ocean(
        cot,
        cer,
        ):

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    vaa = vaa_2d.ravel()
    vza = vza_2d.ravel()
    sza0 = 30.0

    wvl = np.arange(350.0, 701.0, 5.0)

    toa = np.zeros(wvl.size, dtype=np.float32)
    rad = np.zeros((vaa.size, wvl.size), dtype=np.float32)

    for i, wvl0 in enumerate(wvl):

        fname = f'tmp-data/00_er3t_shd/sim_rad_atm1d_high_cloud_over_ocean/{cot:04.1f}_{cer:04.1f}_30.0/{int(wvl0):04d}/shd-out-rad-ipa_sim_rad_atm1d_high_cloud_over_ocean.h5'
        f = h5py.File(fname, 'r')
        toa_ = f['mean/toa'][...]
        rad_ = f['mean/rad'][...]
        toa[i] = toa_
        rad[:, i] = rad_

        if (wvl0 == 555.0):
            rad_555 = rad_.reshape(vaa_2d.shape)
            ref_555 = np.pi * rad_ / (toa_*np.cos(np.deg2rad(sza0)))

        f.close()

    toa_vis = np.trapz(toa, x=wvl)
    ref_vis = np.pi * np.trapz(rad, x=wvl, axis=1) / (toa_vis*np.cos(np.deg2rad(sza0)))

    ref_555 = ref_555.reshape(vaa_2d.shape)
    ref_vis = ref_vis.reshape(vaa_2d.shape)

    # return ref_555, ref_vis

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
    fname = f'{_metadata_['Function']}_{cot:04.1f}_{cer:04.1f}.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # plot
    #╭────────────────────────────────────────────────────────────────────────────╮#
    if True:
        fig = plt.figure(figsize=(24, 7))

        fig.suptitle(f'Cloud over Ocean (COT={cot:.1f}, CER={cer:.1f}$\\mu m$)', fontsize=24, y=1.02)

        # plot 1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(131, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_rgrids(np.arange(20, 81, 20))
        ax1.set_rlim((0.0, 89.0))

        cs = ax1.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555, cmap='jet', vmin=0.2, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax1, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax1.set_title('$Ref_{555}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 2
        #╭──────────────────────────────────────────────────────────────╮#
        ax2 = fig.add_subplot(133, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_rgrids(np.arange(20, 81, 20))
        ax2.set_rlim((0.0, 89.0))

        cs = ax2.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_vis, cmap='jet', vmin=0.2, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax2, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax2.set_title('$Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        # plot 3
        #╭──────────────────────────────────────────────────────────────╮#
        ax3 = fig.add_subplot(132, projection='polar')
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_rgrids(np.arange(20, 81, 20))
        ax3.set_rlim((0.0, 89.0))

        cs = ax3.pcolormesh(np.deg2rad(vaa_2d), vza_2d, ref_555/ref_vis, cmap='viridis', vmin=0.8, vmax=1.0)
        cbar = fig.colorbar(cs, ax=ax3, shrink=0.6, aspect=30, pad=0.1, location='bottom')

        ax3.set_title('$Ref_{555} / Ref_{VIS}$')
        #╰──────────────────────────────────────────────────────────────╯#

        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        fname_fig = f'{_metadata_['Function']}_{cot:04.1f}_{cer:04.1f}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#



def cdata_cloud():

    vaa_1d = np.arange(0.0, 360.1, 1.0)
    vza_1d = np.arange(0.0, 89.1, 1.0)

    vaa_2d, vza_2d = np.meshgrid(vaa_1d, vza_1d, indexing='ij')

    cot = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 50.0])
    cer = np.array([2.0, 4.0, 6.0, 12.0, 18.0, 24.0])

    ref_555 = np.zeros((361, 90, cot.size, cer.size), dtype=np.float32)
    ref_vis = np.zeros((361, 90, cot.size, cer.size), dtype=np.float32)

    for i, cot0 in enumerate(cot):
        for j, cer0 in enumerate(cer):
            fname = f"data/cal_vis_high_cloud_over_ocean_{cot0:04.1f}_{cer0:04.1f}.h5"
            f = h5py.File(fname, 'r')
            ref_555[:, :, i, j] = f['ref_555'][...]
            ref_vis[:, :, i, j] = f['ref_vis'][...]
            f.close()

    # save h5 file
    #╭────────────────────────────────────────────────────────────────────────────╮#
    fname = 'data_cloud.h5'
    h5f = h5py.File(fname, 'w')
    h5d_vaa = h5f.create_dataset('vaa', data=vaa_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_vza = h5f.create_dataset('vza', data=vza_2d, compression='gzip', compression_opts=9, chunks=True)
    h5d_cot = h5f.create_dataset('cot', data=cot, compression='gzip', compression_opts=9, chunks=True)
    h5d_cer = h5f.create_dataset('cer', data=cer, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_555 = h5f.create_dataset('ref_555', data=ref_555, compression='gzip', compression_opts=9, chunks=True)
    h5d_ref_vis = h5f.create_dataset('ref_vis', data=ref_vis, compression='gzip', compression_opts=9, chunks=True)
    h5f.close()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def plot_cloud():
    fname = "data_cloud.h5"
    data = er3t.util.load_h5(fname)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(6, 6))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        for i, cot0 in enumerate(data['cot']):
            ax1.scatter(data['ref_555'][:, :, i, 4], data['ref_vis'][:, :, i, 4], s=1, lw=0.0)
        ax1.set_xlim((0, 1))
        ax1.set_ylim((0, 1))
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f'{_metadata_['Function']}.png'
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#
    pass

if __name__ == '__main__':

    warnings.warn('\nUnder active development ...')

    # radiance simulation
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # example_03_rad_atm1d_clear_over_snow()

    # for wvl0 in np.arange(350.0, 701.0, 5.0):
    #     sim_rad_atm1d_clear_over_forest(wavelength=wvl0)

    # for wvl0 in np.arange(350.0, 701.0, 5.0):
    #     sim_rad_atm1d_clear_over_desert(wavelength=wvl0)

    # for windspeed in np.array([2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]):
    #     for wvl0 in np.arange(350.0, 701.0, 5.0):
    #         sim_rad_atm1d_clear_over_ocean(wavelength=wvl0, windspeed=windspeed)

    # for cot in np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 50.0]):
    #     for cer in np.array([2.0, 4.0, 6.0, 12.0, 18.0, 24.0]):
    #         for wvl0 in np.arange(350.0, 701.0, 5.0):
    #             sim_rad_atm1d_cloud_over_ocean(wavelength=wvl0, cot=cot, cer=cer)
    #             sim_rad_atm1d_high_cloud_over_ocean(wavelength=wvl0, cot=cot, cer=cer)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # cal_vis_clear_over_forest()
    # cal_vis_clear_over_desert()

    # for windspeed in np.array([2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]):
    #     cal_vis_clear_over_ocean(windspeed)

    # for cot in np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 50.0]):
    #     for cer in np.array([2.0, 4.0, 6.0, 12.0, 18.0, 24.0]):
    #         cal_vis_cloud_over_ocean(cot, cer)
            # cal_vis_high_cloud_over_ocean(cot, cer)

    # cdata_cloud()
    plot_cloud()

    pass
