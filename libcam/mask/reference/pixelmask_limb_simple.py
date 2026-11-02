"""
Sample pixel mask for Libera, for Beth/Nicolette
@author: schmidt
"""

import numpy as np
plot=True
fill=True

# parameters that are fixed
xp   = 2048 # width of the FPA
yp   = 2048 #

# parameters that are user-configurable (can be uplinked from the ground)
xc = 1024 # center pixel with respect to FPA coordinate system
yc = 1024
r1 =  895
r2 = 1024

xi,yi = np.meshgrid(np.arange(xp)-xc,np.arange(yp)-yc)
rr2   = xi**2+yi**2
flt   = np.where((rr2>r1**2) & (rr2<r2**2))
mask  = np.zeros((xp,yp))
mask[flt[0],flt[1]]=1


# for visualization only, ignore for flight software
if plot:

    flt=np.where(mask==1)
    nmask=len(flt[0])
    import matplotlib.pyplot as plt
    plt.imshow(mask.T,origin='lower')
    plt.title('pixels in mask:'+str(nmask))
