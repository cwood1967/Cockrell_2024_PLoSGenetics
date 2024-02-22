import glob
import os

from joblib import Parallel, delayed
import tifffile
import numpy as np
import pandas as pd

from . import process
'''
r04c01f04p01-ch1sk1fk1fl1.tiff'
0123456789012345
          111111
          
r - row
c - column
f - field
p - z slice
'''

def get_image_files(screen_dir, image_glob='.tiff'):
    image_files = sorted(os.listdir(screen_dir))
    image_files = [x for x in image_files if x.endswith(image_glob)]
    prefixes = set([x[0:9] for x in image_files])
    fdict = {k:[] for k in prefixes} 

    for f in image_files:
        k = f[0:9]
        row = int(f[1:3])
        col = int(f[4:6])
        field = int(f[7:9])
        channel = int(f[15])
        pz = int(f[10:12])
        fdict[k].append(f)

    return fdict

def get_image(fdir, prefix, imagefiles, bin=1):

    stack = None
    for f in imagefiles:
        nz = len(imagefiles)//3
        row = int(f[1:3])
        col = int(f[4:6])
        field = int(f[7:9])
        channel = int(f[15])
        if channel == 1:
            ch = 1
        if channel == 2:
            ch = 0
        if channel == 3:
            ch = 2
        pz = int(f[10:12])
        data = tifffile.imread(fdir + f)
        if bin == 2:
            _dr = data.reshape(
                (data.shape[0]//2, 2, data.shape[1]//2, 2))
            data = _dr.sum(axis=-1).sum(axis=1)
            
        sy, sx = data.shape
        if stack is None:
            stack = np.zeros((nz, 3, sy, sx))
        stack[pz - 1, ch, :, : ] = data

    stack = stack.sum(axis=0, keepdims=True)
    #stack = np.expand_dims(stack, 0)
    resdict = {'files':imagefiles,
               'row':row,
               'col':col,
               'field':field,
               'data':stack}

    return resdict 

def image_generator(fdir, image_glob='.tiff', bin=1):
    fdict = get_image_files(fdir)
    for k, v in fdict.items():
        yield get_image(fdir, k, v, bin=bin)

def pfunc(d, **kwargs):
    s = d['files'][1][:9]
    print(s)
    just_green_min_size = kwargs['just_green_min_size'] 
    green_in_red_min_size = kwargs['green_in_red_min_size']
    cnn = kwargs['cnn']
    folder = kwargs['folder']
    _, sspdf = process.process(s, d['data'], cnn,
                        just_green_min_size=just_green_min_size,
                        green_in_red_min_size=green_in_red_min_size,
    )
    sspdf['row'] = d['row']
    sspdf['col'] = d['col']
    sspdf['field'] = d['field']
    sspdf['path'] = folder
    return sspdf

def run_screen(folders, pkl_name, modelfile, size=400,
               probability=0.95, just_green_min_size=200,
               green_in_red_min_size=64, bin=1,
               globpattern='*.tiff', channels=[1,0,2],
               zstart=None, zstop=None, njobs=1,
               mtype='screen'):

    if not os.path.exists('Data/Masks'):
        os.makedirs('Data/Masks')
    if not os.path.exists('Data/NoRDNA'):
        os.makedirs('Data/NoRDNA')

    if modelfile is None:
        cnn = None
    else:
        cnn = process.cnn_setup(modelfile, size, probability)

    if mtype != 'screen':
        print("input not set to 'screen'")
        return

    #ssdflist = list()
    kwargs = {'cnn':cnn,
              'just_green_min_size':just_green_min_size,
              'green_in_red_min_size':green_in_red_min_size }

    print(folders)
    for folder in folders:
        print(folder)
        kwargs['folder'] = folder
        ssdflist = Parallel(n_jobs=njobs)\
            (delayed(pfunc)(d, **kwargs) for d in image_generator(folder, bin=bin))

    ssdf = pd.concat(ssdflist)
    ssdf.to_pickle(pkl_name)
