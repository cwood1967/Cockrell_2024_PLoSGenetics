import nd2
import tifffile
import numpy as np

from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, square, binary_closing
from skimage.morphology import binary_dilation, dilation, disk
from skimage.measure import label
from scipy import ndimage as ndi
from skimage.segmentation import flood, flood_fill
from skimage.feature import blob_dog, blob_log, peak_local_max


def region_grow(k, a, ndilation=1):

    res = np.zeros_like(k, dtype=np.uint16)
    vlist = list()
    
    for i, p in enumerate(a):
        v = k[p[0], p[1], p[2]]
        r = flood(k, tuple(p), tolerance=.5*v)
        res += r
        vlist.append(v)
        if res.sum() > 900_000:
           break 

    dk = disk(ndilation*1.5)
    res = np.where(res > 0, 1, 0)
    res = np.array([binary_dilation(j, dk) for j in res])
    #res = label(res)
    return res.astype(np.uint8)

def segment(image, ndilation=1):
    
    _image = gaussian(image, sigma=(0, 1, 1), preserve_range=True)
    
    nc  = image.shape[-1]

    b = gaussian(image, sigma=(0,1.5,1.5))

    pct = 100 - 100*5000/b.size
    brel = np.percentile(b, pct)
    Ab = peak_local_max(b, threshold_abs=brel,
                    min_distance=10, exclude_border=False)

    res = region_grow(b, Ab, ndilation=ndilation)
    return res
    

    
    