import os

import tifffile
import numpy as np


def groupfiles(filelist):
    #filelist = sorted(filelist)
    filenames = [j.split('/')[-1] for j in filelist]
    fields = dict()
    
    for i, fname in enumerate(filenames):
        f = fname[:-10]
        if f not in fields:
            fields[f] = list()
        
        fields[f].append(filelist[i])
    return fields
    
def fieldstack(field, bin=1):
    """
    Create an image stack from the individual tif files for this field
    
    Parameters
    ----------
    field : list
            List of files (paths) for this field
            
    Returns
    -------
    stack : Image stack for this field
    """
    
    image_list = list()
    zs = set([f[-10:-7] for f in field])
    cs = set([f[-7:-4] for f in field])
    nz = len(zs)
    nc = len(cs)
    
    qc = {1:1, 2:0, 3:2}
    stack = np.zeros(10)
    for f in field:
        q = tifffile.imread(f)
        if bin == 2:
            s = (q.shape[0]//2, q.shape[1]//2)
            qb = q.reshape(s[0], 2, s[1], 2)
            q = qb.sum(-1).sum(1)
            
        if stack.shape == (10,):
            stack = np.zeros((nz, nc) +q.shape, dtype=np.float32)
        z = int(f[-10:-7]) - 1
        c = qc[int(f[-7:-4])]
         
        stack[z, c, :, :] = q
        
    return stack
