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
import glob
import datetime
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import h5py
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
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

def plot(data, iset, ivar, vmin=0.0, vmax=1.0):

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    extent = [0, 1859.00000000, 0, 819.50000000]
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(6, 6))
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111)
        cs = ax1.imshow(data.T, origin='lower', cmap='viridis', zorder=0, vmin=vmin, vmax=vmax, extent=extent)
        ax1.axis('off')
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = '%s_iset-%2.2d_ivar-%2.2d.png' % (os.path.basename(fname).replace('.txt', ''), iset, ivar)
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False, pad_inches=0.0)
        #╰──────────────────────────────────────────────────────────────╯#
        # plt.show()
        # sys.exit()
        plt.close(fig)
        plt.clf()
        #╰────────────────────────────────────────────────────────────────────────────╯#


def figure_les(fname):

    data = er3t.rtm.shd.get_shd_data_out(fname)

    for ivar in range(data.shape[-1]):
        for iset in range(data.shape[-2]):

            data_shd = np.squeeze(data[..., iset, ivar])

            if data_shd.ndim > 2:
                for iset_ in range(data_shd.shape[-1]):
                    plot(data_shd[..., iset_], iset_, ivar, vmin=0.0, vmax=1.0)
            else:
                plot(data_shd, iset, ivar, vmin=0.0, vmax=0.3)



def plot_all(data):

    # get aux data
    #╭────────────────────────────────────────────────────────────────────────────╮#
    run = False
    date = datetime.datetime(2021, 5, 18)
    data['fname_rgb'] = er3t.util.download_worldview_image(date, data['extent'], fdir_out='data', satellite='NOAA20', instrument='VIIRS', coastline=True, run=run)
    #╰────────────────────────────────────────────────────────────────────────────╯#

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj = ccrs.PlateCarree()
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(24, 6))
        # fig.suptitle('Figure')

        # rgb
        #╭──────────────────────────────────────────────────────────────╮#
        ax11 = fig.add_subplot(241, projection=proj)
        cs = ax11.imshow(mpl_img.imread(data['fname_rgb']), zorder=0, transform=proj, extent=data['extent']) #, extent=extent, vmin=0.0, vmax=0.5)
        ax11.coastlines(color='gray', lw=0.5)
        ax11.set_title('RGB')
        g11 = ax11.gridlines(draw_labels=True, ls='-')
        g11.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g11.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g11.top_labels = False
        g11.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # rad
        #╭──────────────────────────────────────────────────────────────╮#
        ax12 = fig.add_subplot(242, projection=proj)
        cs = ax12.pcolormesh(data['lon_cam'], data['lat_cam'], data['rad_cam'], cmap='jet', zorder=0, transform=proj) #, extent=extent, vmin=0.0, vmax=0.5)
        ax12.coastlines(color='gray', lw=0.5)
        ax12.set_title('Radiance')
        g12 = ax12.gridlines(draw_labels=True, ls='-')
        g12.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g12.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g12.top_labels = False
        g12.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # land surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax13 = fig.add_subplot(243, projection=proj)
        cs = ax13.pcolormesh(data['lon_land'], data['lat_land'], data['fiso_land'], cmap='jet', zorder=0, transform=proj, vmin=0.0, vmax=0.5) #, extent=extent, vmin=0.0, vmax=0.5)
        ax13.coastlines(color='gray', lw=0.5)
        ax13.set_title('BRDF [$f_{iso}$]')
        ax13.set_xlim(ax12.get_xlim())
        ax13.set_ylim(ax12.get_ylim())
        g13 = ax13.gridlines(draw_labels=True, ls='-')
        g13.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g13.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g13.top_labels = False
        g13.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # ocean surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax14 = fig.add_subplot(244, projection=proj)
        cs = ax14.pcolormesh(data['lon_ocean'], data['lat_ocean'], data['windspeed_ocean'], cmap='jet', zorder=0, transform=proj, vmin=0.0, vmax=20.0) #, extent=extent, vmin=0.0, vmax=0.5)
        ax14.coastlines(color='gray', lw=0.5)
        ax14.set_title('Wind Speed (10m)')
        ax14.set_xlim(ax12.get_xlim())
        ax14.set_ylim(ax12.get_ylim())
        g14 = ax14.gridlines(draw_labels=True, ls='-')
        g14.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g14.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g14.top_labels = False
        g14.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # sza
        #╭──────────────────────────────────────────────────────────────╮#
        ax21 = fig.add_subplot(245, projection=proj)
        cs = ax21.pcolormesh(data['lon_cam'], data['lat_cam'], data['sza_cam'], cmap='jet', zorder=0, transform=proj) #, extent=extent, vmin=0.0, vmax=0.5)
        ax21.coastlines(color='gray', lw=0.5)
        ax21.set_title('Solar Zenith Angle')
        g21 = ax21.gridlines(draw_labels=True, ls='-')
        g21.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g21.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g21.top_labels = False
        g21.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # saa
        #╭──────────────────────────────────────────────────────────────╮#
        ax22 = fig.add_subplot(246, projection=proj)
        cs = ax22.pcolormesh(data['lon_cam'], data['lat_cam'], data['saa_cam'], cmap='jet', zorder=0, transform=proj) #, extent=extent, vmin=0.0, vmax=0.5)
        ax22.coastlines(color='gray', lw=0.5)
        ax22.set_title('Solar Azimuth Angle')
        g22 = ax22.gridlines(draw_labels=True, ls='-')
        g22.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g22.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g22.top_labels = False
        g22.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # vza
        #╭──────────────────────────────────────────────────────────────╮#
        ax23 = fig.add_subplot(247, projection=proj)
        cs = ax23.pcolormesh(data['lon_cam'], data['lat_cam'], data['vza_cam'], cmap='jet', zorder=0, transform=proj) #, extent=extent, vmin=0.0, vmax=0.5)
        ax23.coastlines(color='gray', lw=0.5)
        ax23.set_title('Sensor Zenith Angle')
        g23 = ax23.gridlines(draw_labels=True, ls='-')
        g23.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g23.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g23.top_labels = False
        g23.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # vaa
        #╭──────────────────────────────────────────────────────────────╮#
        ax24 = fig.add_subplot(248, projection=proj)
        cs = ax24.pcolormesh(data['lon_cam'], data['lat_cam'], data['vaa_cam'], cmap='jet', zorder=0, transform=proj) #, extent=extent, vmin=0.0, vmax=0.5)
        ax24.coastlines(color='gray', lw=0.5)
        ax24.set_title('Sensor Azimuth Angle')
        g24 = ax24.gridlines(draw_labels=True, ls='-')
        g24.xlocator = FixedLocator(np.arange(-180, 181, 6))
        g24.ylocator = FixedLocator(np.arange(-80, 81, 2))
        g24.top_labels = False
        g24.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def plot_mask_imerg(
        data={},
        ):

    f = Dataset('/Users/hchen/Downloads/IMERG_land_sea_mask.nc', 'r')
    lon = f.variables['lon'][...]
    lat = f.variables['lat'][...]
    data['lon_sfc'], data['lat_sfc'] = np.meshgrid(lon, lat, indexing='ij')
    data['sfc'] = f.variables['landseamask'][...].T
    f.close()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj = ccrs.PlateCarree()
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')

        # ocean surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax14 = fig.add_subplot(111, projection=proj)
        cs = ax14.pcolormesh(data['lon_sfc'], data['lat_sfc'], data['sfc'], cmap='jet', zorder=0, transform=proj, vmin=-2.0, vmax=2.0)
        ax14.coastlines(color='gray', lw=0.5)
        ax14.set_title('IMERG Land/Ocean Mask')
        ax14.set_xlim((-180.0, 180.0))
        ax14.set_ylim((-90.0, 90.0))
        g14 = ax14.gridlines(draw_labels=True, ls='-')
        g14.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g14.ylocator = FixedLocator(np.arange(-90, 91, 30))
        g14.top_labels = False
        g14.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def plot_mask(
        data={},
        ):

    f = Dataset('data/mask-ease2_raw/NSIDC0772_LatLon_EASE2_M03km_v1.1.nc', 'r')
    data['lon_sfc'] = f.variables['longitude'][...]
    data['lat_sfc'] = f.variables['latitude'][...]
    f.close()

    data['sfc'] = np.fromfile('data/mask-ease2_raw/EASE2_M03km.LOCImask_land50_coast0km.11568x4872.bin', dtype='<i1').reshape((4872, 11568))[:, :]

    data['sfc'] = np.float_(data['sfc'])
    logic = (data['sfc']==-2.0)
    print(logic.sum())
    print(data['lon_sfc'][logic])
    print(data['lat_sfc'][logic])
    data['sfc'][logic] = np.nan

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj = ccrs.PlateCarree()
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')

        # ocean surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax14 = fig.add_subplot(111, projection=proj)
        cs = ax14.pcolormesh(data['lon_sfc'], data['lat_sfc'], data['sfc'], cmap='jet', zorder=0, transform=proj, vmin=-2.0, vmax=2.0)
        ax14.coastlines(color='gray', lw=0.5)
        ax14.set_title('EASE 2.0 Land/Ocean/Ice Mask')
        ax14.set_xlim((-180.0, 180.0))
        ax14.set_ylim((-90.0, 90.0))
        g14 = ax14.gridlines(draw_labels=True, ls='-')
        g14.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g14.ylocator = FixedLocator(np.arange(-90, 91, 30))
        g14.top_labels = False
        g14.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def plot_land(
        data={},
        ):

    f = h5py.File('data/brdf-viirs_raw/VJ143C1.A2021138.002.2024158071341.h5', 'r')

    lon = f['HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lon'][...]
    lat = f['HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lat'][...]

    fiso = f['HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/BRDF_Albedo_Parameter1_M4'][...].T
    fvol = f['HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/BRDF_Albedo_Parameter2_M4'][...].T
    fgeo = f['HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/BRDF_Albedo_Parameter3_M4'][...].T

    f.close()

    logic_nan = (fiso==32767) | (fvol==32767) | (fgeo==32767)
    fiso = fiso * 0.001
    fiso[logic_nan] = np.nan
    fvol = fvol * 0.001
    fvol[logic_nan] = np.nan
    fgeo = fgeo * 0.001
    fgeo[logic_nan] = np.nan

    data['lon_land'], data['lat_land'] = np.meshgrid(lon, lat, indexing='ij')

    data['fiso_land'] = fiso
    data['fvol_land'] = fvol
    data['fgeo_land'] = fgeo


    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj = ccrs.PlateCarree()
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')

        # ocean surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax14 = fig.add_subplot(111, projection=proj)
        cs = ax14.pcolormesh(data['lon_land'], data['lat_land'], data['fvol_land'], cmap='jet', zorder=0, transform=proj, vmin=0.0, vmax=0.5)
        ax14.coastlines(color='gray', lw=0.5)
        ax14.set_title('NOAA-20 VIIRS Land Product (VJ143C1, $f_{vol}$)')
        ax14.set_xlim((-180.0, 180.0))
        ax14.set_ylim((-90.0, 90.0))
        g14 = ax14.gridlines(draw_labels=True, ls='-')
        g14.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g14.ylocator = FixedLocator(np.arange(-90, 91, 30))
        g14.top_labels = False
        g14.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def plot_ocean(
        data={},
        ):

    f = Dataset('data/wind-merra2_raw/MERRA2_400.inst1_2d_asm_Nx.20210518.nc4', 'r')

    lon = f.variables['lon'][...]
    lat = f.variables['lat'][...]

    index_t = 5

    u10 = f.variables['U10M'][...][index_t, :, :].T
    v10 = f.variables['V10M'][...][index_t, :, :].T

    f.close()

    logic_nan = (u10<-100.0) | (u10>100.0) | (v10<-100.0) | (v10>100.0)
    u10[logic_nan] = np.nan
    v10[logic_nan] = np.nan

    data['lon_ocean'], data['lat_ocean'] = np.meshgrid(lon, lat, indexing='ij')

    data['windspeed_ocean'] = np.sqrt(u10**2.0 + v10**2.0)

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    proj = ccrs.PlateCarree()
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(12, 6))
        # fig.suptitle('Figure')

        # ocean surface
        #╭──────────────────────────────────────────────────────────────╮#
        ax14 = fig.add_subplot(111, projection=proj)
        cs = ax14.pcolormesh(data['lon_ocean'], data['lat_ocean'], data['windspeed_ocean'], cmap='jet', zorder=0, transform=proj, vmin=0.0, vmax=20.0)
        ax14.coastlines(color='gray', lw=0.5)
        ax14.set_title('MERRA-2 10m Wind Speed (M2I1NXASM)')
        # ax14.set_xlim(ax12.get_xlim())
        # ax14.set_ylim(ax12.get_ylim())
        ax14.set_xlim((-180.0, 180.0))
        ax14.set_ylim((-90.0, 90.0))
        g14 = ax14.gridlines(draw_labels=True, ls='-')
        g14.xlocator = FixedLocator(np.arange(-180, 181, 60))
        g14.ylocator = FixedLocator(np.arange(-90, 91, 30))
        g14.top_labels = False
        g14.right_labels = False
        #╰──────────────────────────────────────────────────────────────╯#

        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#

def plot_cloud_mask_all():

    wavelength = 555.0

    f = h5py.File('output.h5', 'r')
    extent = f['extent'][...]
    lon = f['lon_cam'][...]
    lat = f['lat_cam'][...]
    rad_obs = f['rad_cam'][...]
    rad_rtm = f['rad_rtm'][...]
    logic_cld = f['cld_msk'][...]
    f.close()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(20, 12))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(231, aspect='equal')
        cs = ax1.pcolormesh(lon, lat, rad_obs, cmap='jet', zorder=0, vmin=0.0, vmax=0.4)
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Latitude [$^\\circ$]')
        ax1.set_title('Observation (All-Sky)', color='red')

        ax2 = fig.add_subplot(232, aspect='equal')
        cs = ax2.pcolormesh(lon, lat, rad_rtm, cmap='jet', zorder=0, vmin=0.0, vmax=0.1)
        ax2.set_xlabel('Longitude [$^\\circ$]')
        ax2.set_ylabel('Latitude [$^\\circ$]')
        ax2.set_title('Simulation (Clear-Sky)', color='blue')

        rad_diff = rad_obs - rad_rtm
        ax3 = fig.add_subplot(233, aspect='equal')
        cs = ax3.pcolormesh(lon, lat, rad_diff, cmap='seismic', zorder=0, vmin=0.0, vmax=0.4)
        ax3.set_xlabel('Longitude [$^\\circ$]')
        ax3.set_ylabel('Latitude [$^\\circ$]')
        ax3.set_title('Difference')

        bins = np.linspace(-0.5, 1.0, 201)
        ax4 = fig.add_subplot(234)
        ax4.hist(rad_obs.ravel(), bins=bins, color='red', histtype='step', density=True)
        ax4.hist(rad_rtm.ravel(), bins=bins, color='blue', histtype='step', density=True)
        ax4.hist(rad_diff.ravel(), bins=bins, color='black', histtype='stepfilled', density=True)
        # ax4.set_yscale('log')
        patches_legend = [
                          mpatches.Patch(color='black' , label='Difference'), \
                          mpatches.Patch(color='red'   , label='Observation'), \
                          mpatches.Patch(color='blue'  , label='Simulation'), \
                         ]
        ax4.legend(handles=patches_legend, loc='upper right', fontsize=16)


        # threshold = cal_threshold_efold(rad_diff, bins=bins)
        # ax4.axvline(threshold, color='green', ls='--')

        # get aux data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        run = False
        date = datetime.datetime(2021, 5, 18)
        fname_img = er3t.util.download_worldview_image(date, extent, fdir_out='data', satellite='NOAA20', instrument='VIIRS', coastline=True, run=run)
        rad_img = mpl_img.imread(fname_img)
        #╰────────────────────────────────────────────────────────────────────────────╯#


        ax5 = fig.add_subplot(235, aspect='equal')
        ax5.imshow(rad_img, extent=extent)
        ax5.scatter(lon[logic_cld], lat[logic_cld], s=0.1, c='red', alpha=0.2)
        ax5.set_xlabel('Longitude [$^\\circ$]')
        ax5.set_ylabel('Latitude [$^\\circ$]')

        # ax1.scatter(_x, _y, s=2, c='k', lw=0.0)
        # ax1.set_xlim((0, 1))
        # ax1.set_ylim((0, 1))
        # ax1.set_title('Plot1')
        # ax1.xaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        # ax1.yaxis.set_major_locator(FixedLocator(np.arange(0, 100, 5)))
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.show()
        sys.exit()
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def plot_cloud_ori():

    wavelength = 555.0

    extent = [-98.2635, -80.0462, 18.8358, 26.4245]

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(16, 12))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111, aspect='equal')

        # get aux data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        fname_img = '../../data/sample/VIIRS-NOAA20_TrueColor_2021-05-18T19:06:44Z_(-98.2635,-80.0462,18.8358,26.4245).png'
        rad_img = mpl_img.imread(fname_img)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.imshow(rad_img, extent=extent)

        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Latitude [$^\\circ$]')

        ax1.set_xlim(extent[:2])
        ax1.set_ylim(extent[2:])
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#


def plot_cloud_mask():

    wavelength = 555.0

    extent = [-98.2635, -80.0462, 18.8358, 26.4245]

    fname = sorted(glob.glob('../../tmp-data/LIBERA_L2_CF-RAD_*.nc'))[-1]
    f = h5py.File(fname, 'r')
    lon       = f['longitude'][...]
    lat       = f['latitude'][...]
    logic_cld = (f['cloud_fraction'][...]>0.0)
    f.close()

    # figure
    #╭────────────────────────────────────────────────────────────────────────────╮#
    plot = True
    if plot:
        plt.close('all')
        fig = plt.figure(figsize=(16, 12))
        # fig.suptitle('Figure')
        # plot1
        #╭──────────────────────────────────────────────────────────────╮#
        ax1 = fig.add_subplot(111, aspect='equal')

        # get aux data
        #╭────────────────────────────────────────────────────────────────────────────╮#
        # run = False
        # date = datetime.datetime(2021, 5, 18)
        # fname_img = er3t.util.download_worldview_image(date, extent, fdir_out='data', satellite='NOAA20', instrument='VIIRS', coastline=True, run=run)
        fname_img = '../../data/sample/VIIRS-NOAA20_TrueColor_2021-05-18T19:06:44Z_(-98.2635,-80.0462,18.8358,26.4245).png'
        rad_img = mpl_img.imread(fname_img)
        #╰────────────────────────────────────────────────────────────────────────────╯#

        ax1.imshow(rad_img, extent=extent)

        ax1.scatter(lon[logic_cld], lat[logic_cld], s=0.1, lw=0.0, c='red', alpha=1.0, marker='s')
        ax1.set_xlabel('Longitude [$^\\circ$]')
        ax1.set_ylabel('Latitude [$^\\circ$]')

        ax1.set_xlim(extent[:2])
        ax1.set_ylim(extent[2:])
        #╰──────────────────────────────────────────────────────────────╯#
        # save figure
        #╭──────────────────────────────────────────────────────────────╮#
        fig.subplots_adjust(hspace=0.35, wspace=0.35)
        _metadata_ = {'Computer': os.uname()[1], 'Script': os.path.abspath(__file__), 'Function':sys._getframe().f_code.co_name, 'Date':datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}
        fname_fig = f"{_metadata_['Function']}.png"
        plt.savefig(fname_fig, bbox_inches='tight', metadata=_metadata_, transparent=False)
        #╰──────────────────────────────────────────────────────────────╯#
        plt.close(fig)
        plt.clf()
    #╰────────────────────────────────────────────────────────────────────────────╯#



if __name__ == '__main__':

    plot_cloud_ori()
    plot_cloud_mask()

    pass
