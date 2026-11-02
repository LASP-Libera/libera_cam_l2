"""
Created: 29 Oct 2021
Latest update: 12 Nov 2021
Pixel mask generator for Libera "ADM" samples
@author: Jake Gristey
"""

import numpy as np
import timeit
starttime = timeit.default_timer()
# parameters that are user-configurable (can be uplinked from the ground)
seed=42 # random seed
percent_samp=40 # percentage of angular bins to randomly sample at each exposure
max_radius_of_pix=12 # circular radius of pixels for each ADM sample (assume 12 from cam output)

# parameters that are fixed
x_center_pix=1024
y_center_pix=1024
pix_spacing=0.0055  # [mm]
coef_ang2dist=[0,-2.07789E-6,-1.16828E-5,9.92181E-2,0] # coefficients from cam code

# define ADM bin boundaries in satellite reference frame
vza_boundaries=np.array([0.,8.84496268,17.62889971,26.27870803,34.69257787,42.71203223,50.07138791,56.31251405,60.69465961,62.31015813])
vaa_boundaries=np.arange(-180,181,20)

# create array of angular bin numbers [VZA x VAA]
n_vza_bins=len(vza_boundaries)-1
n_vaa_bins=len(vaa_boundaries)-1
n_bins=n_vza_bins*n_vaa_bins
bin_nums_all=np.arange(n_bins).reshape((n_vza_bins,n_vaa_bins))

# randomly select the allowed number of bins, determined by "percent_samp"
n_samples=int(percent_samp/100.*n_bins) # int always rounds down to be conservative for data rate
np.random.seed(seed) # ensures predictable results
selected_bin_nums=sorted(np.random.choice(np.arange(n_bins),n_samples,replace=False))

# loop over selected angular bins to find selected camera pixels
x_pix_selected=[]
y_pix_selected=[]
for bin_num in selected_bin_nums:

    # randomly select a VZA and VAA within each bin
    vza_i,vaa_i=np.array(np.where(bin_nums_all==bin_num)).flatten()
    vza=vza_boundaries[vza_i]+np.random.random()*(vza_boundaries[vza_i+1]-vza_boundaries[vza_i])
    vaa=vaa_boundaries[vaa_i]+np.random.random()*(vaa_boundaries[vaa_i+1]-vaa_boundaries[vaa_i])

    # find the closest corresponding pixel
    dist_from_center=0 
    for i,c in enumerate(coef_ang2dist):
        dist_from_center+=c*np.power(vza,(len(coef_ang2dist)-i-1))
    x_dist_from_center=dist_from_center*np.sin(np.deg2rad(vaa))
    y_dist_from_center=dist_from_center*np.cos(np.deg2rad(vaa))
    x_pix_selected.append(round(x_center_pix+x_dist_from_center/pix_spacing))
    y_pix_selected.append(round(y_center_pix+y_dist_from_center/pix_spacing))

# select a group of pixels sorrounding each ADM central pixel that encompasses the Libera PSF
mask=np.zeros((x_center_pix*2,y_center_pix*2))
for i in range(len(x_pix_selected)):
    # start by selecting a box around each central pixel
    x_pix_box=np.arange(x_pix_selected[i]-max_radius_of_pix,x_pix_selected[i]+max_radius_of_pix+1)
    y_pix_box=np.arange(y_pix_selected[i]-max_radius_of_pix,y_pix_selected[i]+max_radius_of_pix+1)

    # only keep pixels from the box that are within a circular radius of the central pixel
    for x in x_pix_box:
        for y in y_pix_box:
            radius_from_central_pix=((x-x_pix_selected[i])**2+(y-y_pix_selected[i])**2)**0.5
            if radius_from_central_pix<=max_radius_of_pix+0.5: # +0.5 buffer (closer to cam result)
                if x<mask.shape[0] and y<mask.shape[0]:
                    mask[int(x),int(y)]=1
endtime = timeit.default_timer()
print(endtime-starttime)
