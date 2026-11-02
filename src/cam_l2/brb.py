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
import copy
import multiprocessing as mp
from collections import OrderedDict
# from tqdm import tqdm
import h5py
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

from cam_l2.util import *

__all__ = ['main_L2_BR_CAM', 'main_L2_BR_RAD']

def cdata_l2_ssw_rad():

    # assume we got camera l1b radiance here
    #╭────────────────────────────────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # ref_cam
    # lon_cam
    # lat_cam
    # jday_cam



    # pixel surface type
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # over ocean/snow or over land
    #╭──────────────────────────────────────────────────────────────╮#
    #╰──────────────────────────────────────────────────────────────╯#

    # over ocean/snow
    #╭──────────────────────────────────────────────────────────────╮#
    # over snow
    #╭────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────╯#

    # over ocean
    #╭────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────╯#
    #╰──────────────────────────────────────────────────────────────╯#

    # over land
    #╭──────────────────────────────────────────────────────────────╮#
    # download VIIRS surface product (MCD43A1)
    #╭────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────╯#
    # fgeo
    # fvol
    # fiso
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # cloud treatment
    #╭────────────────────────────────────────────────────────────────────────────╮#
    # if clear-sky
    #╭──────────────────────────────────────────────────────────────╮#
    #╰──────────────────────────────────────────────────────────────╯#

    # if cloud present
    #╭──────────────────────────────────────────────────────────────╮#
    #╰──────────────────────────────────────────────────────────────╯#
    #╰────────────────────────────────────────────────────────────────────────────╯#



    # run 3D-RT to get TOA reflectance under clear-sky (accounting for Rayleigh scattering)
    #╭────────────────────────────────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # ref_clear


    # retrieve cloud fraction
    #╭────────────────────────────────────────────────────────────────────────────╮#
    #╰────────────────────────────────────────────────────────────────────────────╯#
    # cld_frac
    pass

def get_arguments(args):

    fname = os.path.abspath(args.fname[0])
    if not os.path.exists(fname):
        msg = '\nError [cam_l2_bbr]: cannot locate file <{fname}>.'.format(fname=fname)
        raise OSError(msg)

    return fname

def main_L2_BR_CAM(fname):

    data = {}

    data['extent'] = np.array([0.0, 1.0, 0.0, 1.0])
    data['lon_cam'] = np.linspace(0.0, 1.0, 100)
    data['lat_cam'] = np.linspace(0.0, 1.0, 100)
    data['rad_cam'] = np.ones(100)
    data['rad_rtm'] = np.ones(100)
    data['brb_refl'] = np.ones(100)

    return data

def main_L2_BR_RAD(fname):

    data = {}

    data['extent'] = np.array([0.0, 1.0, 0.0, 1.0])
    data['lon_rad'] = np.linspace(0.0, 1.0, 10)
    data['lat_rad'] = np.linspace(0.0, 1.0, 10)
    data['rad_rad'] = np.ones(10)
    data['brb_refl'] = np.ones(10)

    return data


if __name__ == '__main__':

    pass
