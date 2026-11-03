"""
Created: 2 Feb 2024
Latest update: 9 Feb 2024
Pixel mask generator for Libera camera lock-on ADM samples, in barebones format
@author: Jake Gristey
"""

import sys
import numpy as np # only used for trig

# user-configurable parameters
percent_samp      = 100 # percentage of angular bins to sample at each exposure
max_radius_of_pix = 20  # circular radius of pixels for each ADM sample
vza_shift         = 0   # shift of viewing zenith angle bin [deg]
vaa_shift         = 0   # shift of viewing azimuth angle bin [deg]

# constants
x_center_pix=1024
y_center_pix=1024
pix_spacing=0.0055  # [mm]
coef_ang2dist=[0,-2.07789E-6,-1.16828E-5,9.92181E-2,0] # coefficients from cam code
re=6371 # Earth radius
alt=824 # satellite altitude
pi=3.14159

# define ADM bin boundaries in surface reference frame
vza_boundaries=[0,10,20,30,40,50,60,70,80,90] # equivalent to CERES-TRMM angular bins
vaa_boundaries=[0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360] # equivalent to CERES-TRMM angular bins

# create array of angular bin numbers [VZA x VAA]
n_vza_bins=9
n_vaa_bins=18
n_bins=n_vza_bins*n_vaa_bins # total number of bins
bin_nums_all=[[row * n_vaa_bins + col for col in range(n_vaa_bins)] for row in range(n_vza_bins)] # just an array [VZA x VAA] where each element is assigned a number

# select the angular bins to sample, determined by "percent_samp"
n_samples=int(percent_samp/100.*n_bins) # int always rounds down to be conservative for data rate
selected_bin_nums=[int(n_bins/n_samples*i) for i in range(n_samples)] # evenly sample angular bins based on n_samples

# loop over selected angular bins to identify corresponding camera pixels
x_pix_selected=[]
y_pix_selected=[]
for bin_num in selected_bin_nums:

    # select a VZA and VAA within each bin
    vza_i,vaa_i=[[i,bin_nums_all[i].index(bin_num)] for i in range(len(bin_nums_all)) if bin_num in bin_nums_all[i]][0] # get indicies of selected bin
    # vza_i_ = bin_num//n_vaa_bins
    # vaa_i_ = bin_num%n_vaa_bins

    vza=vza_boundaries[vza_i]+0.5*(vza_boundaries[vza_i+1]-vza_boundaries[vza_i])+vza_shift
    vaa=vaa_boundaries[vaa_i]+0.5*(vaa_boundaries[vaa_i+1]-vaa_boundaries[vaa_i])+vaa_shift

    # if the shift moved the angle out of possible range, put back in range
    vza=vza%90
    vaa=vaa%360

    # convert VZA to satellite reference frame (spherical Earth assumption)
    vza_sat_ref=np.arcsin(np.sin(vza/180*pi)*(re/(re+alt)))/pi*180

    # find the closest corresponding pixel
    dist_from_center=0
    for i,c in enumerate(coef_ang2dist):
        dist_from_center+=c*vza_sat_ref**(len(coef_ang2dist)-i-1)
    x_dist_from_center=dist_from_center*np.sin(vaa/180*pi)
    y_dist_from_center=dist_from_center*np.cos(vaa/180*pi)

    x_pix_selected.append(round(x_center_pix+x_dist_from_center/pix_spacing))
    y_pix_selected.append(round(y_center_pix+y_dist_from_center/pix_spacing))


# select a group of pixels sorrounding each ADM central pixel that encompasses the Libera PSF
mask=[[0 for col in range(x_center_pix*2)] for row in range(y_center_pix*2)] # initialize a 2048x2048 array of zeros
for i in range(len(x_pix_selected)):
    # start by selecting a box around each central pixel

    x_pix_box=range(int(x_pix_selected[i]-max_radius_of_pix),int(x_pix_selected[i]+max_radius_of_pix+1))
    y_pix_box=range(int(y_pix_selected[i]-max_radius_of_pix),int(y_pix_selected[i]+max_radius_of_pix+1))

    # only keep pixels from the box that are within a circular radius of the central pixel
    for x in x_pix_box:
        for y in y_pix_box:
            radius_from_central_pix=((x-x_pix_selected[i])**2+(y-y_pix_selected[i])**2)**0.5
            if radius_from_central_pix<=max_radius_of_pix+0.5: # +0.5 buffer (closer to cam result)
                if x<x_center_pix*2 and y<y_center_pix*2:
                    mask[x][y]=1

# save data
#/--------------------------------------------------------------\#
fmt  = '%5d'
data = mask.copy()
np.savetxt('test-py.txt', data, fmt=fmt, delimiter=',', newline='\n', header='', footer='', encoding=None)
sys.exit()
#\--------------------------------------------------------------/#

# just for visualization; ignore in flight software
import matplotlib.pyplot as plt
plt.imshow(mask)

