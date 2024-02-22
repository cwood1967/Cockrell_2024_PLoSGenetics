# %%
import autoreload
%load_ext autoreload
%autoreload 2

import glob

import numpy as np
import tifffile
import scipy.ndimage as ndi
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import eig

import scipy.ndimage as ndi
from skimage.measure import inertia_tensor, inertia_tensor_eigvals

import cell_length

# %%    
tiffs = glob.glob("Data/Recon/*.tif")
len(tiffs)

# %%
x = tifffile.imread(tiffs[15])
x.shape

# %%
plt.imshow(x)

# %%
mrx = np.where(x > .95, 1, 0)
labels, nlabels = ndi.label(mrx)
nlabels

# %%
ncx = 30
mx = np.where(labels == ncx)
ds  = cell_length.calc_cell_length(mx)

print("***", ds['length'], ds.keys())

fig, ax = plt.subplots(2, 1, figsize=(6,12))
ax[0].scatter(mx[1], mx[0], s=1)
ax[0].axis('equal')
ax[1].scatter(ds['Xp'], ds['yp'], s=1, alpha=.5)
ax[1].plot(ds['xline'], ds['yline'])
#ax[1].scatter([p[-1]], segline([p[-1]], *p), s=14)
_ = ax[1].axis('equal')
print(ds['eigenvec'])
# %%

fig, ax = plt.subplots(4, 2, figsize=(8,10), sharex=True)

rc =np.random.randint(1, nlabels, 8)
irc = iter(rc)
for i in range(4):
    for j in range(2):
        ncx = next(irc)
        mx = np.where(labels == ncx)
        try:
            ds, Xp, yp, ux, upy  = cell_length.calc_cell_length(mx)
        except:
            print("######", len(mx[0]))
            continue
        ax[i, j].scatter(Xp, yp, s=1)
        ax[i, j].plot(ux, upy, color='orange')
        ax[i, j].set_aspect('equal')
        ax[i, j].set_title('{:8.2f}'.format(ds))
        ax[i, j].set_ylim((-30,30))       
# %%
p, a1, a0, m0
# %%
plt.imshow(mcell, origin='bottom')

# %%


# %%
