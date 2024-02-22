import os
import glob
from os import mkdir
import sys 
import time
import pickle
import functools

from joblib import Parallel, delayed
from scipy.ndimage.measurements import label
import tifffile
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import pandas as pd
from nd2reader import ND2Reader
import cv2
from skimage.filters import threshold_otsu
from skimage.filters import edges, gaussian
import skimage as ski
import torch
import torchvision

import cellpose
from cellpose import models

from cellfinder import infer, train
from pombelength import cell_length
from . import region_grow

def getframe(nd2, index, channels=None):
    dims = nd2.sizes
    if not 't' in dims:
        dims['t'] = 1
    if not 'z' in dims:
        dims['z'] = 1
    if not 'c' in dims:
        dims['c'] = 1

    if channels is None:
        channels = range(dims['c'])
    frame = np.zeros((dims['z'], len(channels), dims['y'], dims['x']))
    for zi in range(dims['z']):
        for icx, ci in enumerate(channels):
            frame[zi,icx,:,:] = nd2.get_frame_2D(t=index, z=zi, c=ci)

    return frame.astype(np.float32)

def get_slice(nd2, t, z, c):
    zslice = nd2.get_frame_2D(t=t, z=z, c=c)
    return zslice

def get_nd2_image(filename, channels=[0,1]):
    try:
        nd2 = ND2Reader(filename)
        stack = getframe(nd2, 0, channels=channels)
    except Exception as e:
        print(f"!### Problem reading {filename}")
        return None

    return stack.astype(np.float32)

def getbestz(stack):
    if stack.shape[0] == 1:
        return stack[0, :, :]
    
    if stack[0].max() == 0:
        #print("zzzzz: only 1 z slice")
        return stack.max(axis=(0))

    nz, ny, nx = stack.shape
    fedges = np.zeros((nz, ny, nx))
    for i in range(nz):
        g0 = gaussian(stack[i,...], 1)
        e0 = edges.sobel(g0)
        fedges[i] = e0

    edge_sum = fedges.sum(axis=(1,2))
    g1 = np.gradient(edge_sum)
    g2 = np.gradient(g1)
    gbest = g2.argmax()
    #print(f"zzzzz: best z is {gbest}")
    return stack[gbest, :,:]

def bgs(gz, size):
    '''
    Background one slice at a time subtract using opencv

    Parameters
    ----------
    gz: The input image stack/movie
    size: the size of the structuring element

    Returns
    -------
    The background subtracted image stack/movie
    '''
    nz = gz.shape[0]
    nc = gz.shape[-1]
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    bgi = np.zeros_like(gz)
    for zi in range(nz):
        for ci in range(nc):
            x = gz[zi, :,:, ci]
            w_ = cv2.morphologyEx(x, cv2.MORPH_TOPHAT, s)
            bgi[zi, :,:, ci] = w_
    return bgi

def calc_lengths(labels):
    ncells = labels.max()
    ds_dict = dict() 
    ds_dict[0] = -1
    for i in range(ncells):
        mx = np.where(labels == (i + 1))
        if len(mx[0])*len(mx[1]) < 500:
            print("too small", i, i + 1)
            continue
        try:
            ds = cell_length.calc_cell_length(mx)
        except:
            print("length err", i)
            continue
        if ds is None:
            length = -1
        else:
            length = ds['length']
        ds_dict[i + 1] = length

    return ds_dict

def distances(a1, a2):
    z1, y1, x1 = np.where(a1 > 0)
    z2, y2, x2 = np.where(a2 > 0)

    z1m = z1.mean()
    y1m = y1.mean()
    x1m = x1.mean()
    z2m = z2.mean()
    y2m = y2.mean()
    x2m = x2.mean()

    dz = z2m - z1m
    dy = y2m - y1m
    dx = x2m - x1m
    return np.sqrt(dz*dz + dy*dy + dx* dx)

    
    
def process(filename, image, cnn, is_cellpose=True,
            frame=0,
            just_green_min_size=200, green_in_red_min_size=64):
    '''
    Process one image using threholds and 3d labeling

    Parameters
    ----------
    filename: str - the filename of the image
    image: numpy array - the image data
    red_threshold: float - threshold for channel 0
    gree_threshold: float - threshold for channel 1

    Returns
    -------
    pdf: Dataframe
    '''
    basename = os.path.basename(filename)
    dir0 = os.path.basename(os.path.dirname(filename))
    datadir = os.path.join(dir0, basename[:-4])
   
    mdir = f"Data/Masks/{dir0}" 
    nodir = f"Data/NoRDNA/{dir0}" 
    if not os.path.exists(mdir):
        os.makedirs(mdir)
        print("Created ", mdir)

    if not os.path.exists(nodir):
        os.makedirs(nodir)
        print("Created ", nodir)

    maskname = f'Data/Masks/{datadir}_{frame:03d}_mask.tif'

    infername = f'Data/Masks/{datadir}_{frame:03d}_inferred.tif'
    objname = 'Data/NoRDNA/' + datadir +  '.pkl'
    
    bf = image[:,3,:,:]
    ### this is assuming the transmitted light is in only one slice
    bf = getbestz(bf)
    
    if cnn is not None:
        if is_cellpose:
            cells_labels, _, _ = cnn(bf)
            ds_dict = calc_lengths(cells_labels)
        else:
            with torch.no_grad():
                prob_cells, _ = cnn(bf)
            cells_mask = np.where(prob_cells > .9, 1, 0)
            cells_labels, ncells = label(cells_mask)
            ds_dict = calc_lengths(cells_labels)
            del prob_cells
            torch.cuda.empty_cache()
    else:
        cells_labels = -1 + bf*0
        ds_dict = {}
        
    ### this get the fluorescent images
    f = image[:,0:3,:,:]
    no_rdna_list = list()
    f = np.moveaxis(f, 1, -1) # move the channel axis to last
## background subtract
    ti0 = time.time()
    f = bgs(f, 15) #ndi.white_tophat(f, (0, 25, 25, 0))
    ti1 = time.time()
    #g = ndi.gaussian_filter(f, (0,1,1,0))
    #ti2 = time.time()

    g = ski.filters.gaussian(f, (0, 1, 1, 0))
    g0 = g[:,:,:,0]  ## channel 0
    g1 = g[:,:,:,1] ## channel 1
    g2 = g[:,:,:,2] ## channel 2
    
    tc0 = 3*ski.filters.threshold_mean(g0)
    tc1 = 3*ski.filters.threshold_mean(g1)
    tc2 = 3*ski.filters.threshold_mean(g2)

    ''' create masks from thresholds'''
    tx0 = np.where(g0 > tc0, 1, 0) #channel 0 3d mask
    tx1 = np.where(g1 > tc1, 1, 0) #channel 2 3d mask
    tx2 = np.where(g2 > tc2, 1, 0) #channel 2 3d mask

    ball = ski.morphology.ball(3)
    tx1 = ski.morphology.binary_erosion(tx1)
    tx2 = ski.morphology.binary_erosion(tx2)
    tx2 = ski.morphology.binary_erosion(tx2)
    # use peak finder and region growing to segment
    #tx0 = region_grow.segment(g0, 2)
    #tx1 = region_grow.segment(g1, 1)
    #tx2 = region_grow.segment(g2, 1)
    #stx0 = (tx0.max(), tx1.max(), tx0.sum(), tx1.sum()) 
    
    if tx0.shape[0] < 3:
        selmz = 1
    else:
        selmz = 3

    stx0 = (tx0.max(), tx1.max(), tx0.sum(), tx1.sum())
    
    no_red = np.where((tx1 - tx0) > 0, 1, 0) 
    combined_mask = tx0 + tx1 + tx2
    combined_mask = np.where(combined_mask > 0, 1, 0)
    ball = ski.morphology.ball(3)
    combined_mask = ski.morphology.binary_dilation(combined_mask, ball)

    rdna_mask = np.where((tx0 + tx1) > 1, 1, 0)
    blue_rdna_mask = np.where((tx0 + tx2) > 1, 1, 0)
    fmask = np.stack((tx0, tx1, tx2))
    fmask = 255*(fmask.astype(np.uint8))

    try:
        tifffile.imwrite(maskname, fmask, imagej=True)
        if cnn is not None:
            tifffile.imwrite(infername,
                             cells_labels.astype(np.uint16), imagej=True)
    except:
        pass
    combined_labels, ncombinded = ndi.label(combined_mask)

    '''combined objects is a list slice objects'''
    combined_objects = ndi.find_objects(combined_labels)
    athresh = 150
    dlist = list()
    dsslist = list()
    for index, obj in enumerate(combined_objects):
        #print(obj)
        xcm = int((obj[2].stop + obj[2].start)/2)
        ycm = int((obj[1].stop + obj[1].start)/2)
        cell_key = cells_labels[ycm, xcm]
        if cell_key in ds_dict:
            this_length = ds_dict[cell_key]
        else:
            this_length = -1
            #print("####",  xcm, ycm)
        p = f[obj].copy()  # copy of the slice
        v = p.shape[0]*p.shape[1]*p.shape[2]
        a = p.shape[1]*p.shape[2]

        '''
        get the bgs image of channel 1, only inside the combined
        mask of the nucleus
        '''
        p1 = p[..., 1]*combined_mask[obj]
        just_green = p[:,:,:,1]*tx1[obj]
        just_blue = p[..., 2]*tx2[obj]

        projected_just_green = just_green.max(axis=0)
        projected_just_blue = just_blue.max(axis=0)
        #green_area = projected_p1[projected_p1 > 0].shape[0]
        green_area = projected_just_green[projected_just_green > 0].shape[0]
        blue_area = projected_just_blue[projected_just_blue > 0].shape[0]
        p1 = p1[p1 > 0]  ## doing this makes p1 just a 1d array

        green_obj = just_green.copy()
        just_green = just_green[just_green > 0]
        blue_obj = just_blue.copy()
        just_blue = just_blue[just_blue > 0]
        #if p1.shape[0] > 200:

        if just_green.shape[0] > just_green_min_size:
            green_std = just_green.std()
            green_mean = just_green.mean()
        else:
            green_std = 0 
            green_mean = 0

        if just_blue.shape[0] > just_green_min_size:
            blue_std = just_blue.std()
            blue_mean = just_blue.mean()
        else:
            blue_std = 0 
            blue_mean = 0 

        p2 = p[:, :, :, 2]*(no_red[obj])

        projected_p2 = p2.max(axis=(0,))
        p2_area = projected_p2[projected_p2 > 0].shape[0]

        p2 = p2[p2 > 0]
        p2_std = p2.std()
        p2_mean = p2.mean()

        rdna = p[:,:,:,1]*rdna_mask[obj]
        projected_rdna = rdna.max(axis=(0,))
        rdna_area = projected_rdna[projected_rdna > 0].shape[0]

        rdna_obj = rdna.copy()
        rdna = rdna[rdna > 0]
        if len(rdna) > 0: 
            rdna_std = rdna.std()
            rdna_mean = rdna.mean()
            rdna_vol = rdna.shape[0]
        else:
            #print("no rdna volume", index, rdna.sum(), just_green.shape[0])
            no_rdna_list.append(obj)
            rdna_std = 0
            rdna_mean = 0
            rdna_vol = 0

        blue_rdna = p[:,:,:,2]*blue_rdna_mask[obj]
        projected_blue_rdna = blue_rdna.max(axis=(0,))
        blue_rdna_area = projected_blue_rdna[projected_blue_rdna > 0].shape[0]

        blue_rdna_obj = blue_rdna.copy()
        blue_rdna = blue_rdna[blue_rdna > 0]
        if len(blue_rdna) > 0: 
            blue_rdna_std = blue_rdna.std()
            blue_rdna_mean = blue_rdna.mean()
            blue_rdna_vol = blue_rdna.shape[0]
        else:
            #print("no rdna volume", index, rdna.sum(), just_green.shape[0])
            no_rdna_list.append(obj)
            blue_rdna_std = 0
            blue_rdna_mean = 0
            blue_rdna_vol = 0

        green_in_red = p[:,:,:,1]*tx0[obj]
        blue_in_red = p[..., 2]*tx0[obj]

        projected_red = tx0[obj].max(axis=(0,))
        red_area = projected_red[projected_red > 0].shape[0]

        green_in_red = green_in_red[green_in_red > 0]
        blue_in_red = blue_in_red[blue_in_red > 0]

        if green_in_red.shape[0] > green_in_red_min_size:
            green_in_red_std = green_in_red.std()
            green_in_red_mean = green_in_red.mean()
        else:
            green_in_red_std = 0 
            green_in_red_mean = 0 

        if blue_in_red.shape[0] > green_in_red_min_size:
            blue_in_red_std = blue_in_red.std()
            blue_in_red_mean = blue_in_red.mean()
        else:
            blue_in_red_std = 0 
            blue_in_red_mean = 0 

        red_in_red = p[:,:,:,0]*tx0[obj]
        red_obj = red_in_red.copy()
        red_in_red = red_in_red[red_in_red > 0]
        red_in_red_std = red_in_red.std()
        red_in_red_mean = red_in_red.mean()

        green_rdna_dist = distances(green_obj, rdna_obj)
        green_red_dist = distances(green_obj, red_obj)
        rdna_red_dist = distances(rdna_obj, red_obj)
        blue_red_dist = distances(blue_obj, red_obj)
        blue_green_dist = distances(blue_obj, green_obj)
        gr = p[:,:,:,0]*tx0[obj]
#         p0h, p0hx = np.histogram(p0, bins=64)
#         p0h = p0h[p0h > 0]

        d = {'i':index, 'vol':just_green.shape[0], 'green_std':green_std, 'green_mean':green_mean,
             'p2_std':p2_std, 'p2_mean':p2_mean,
             'green_in_red_std':green_in_red_std, 'green_in_red_mean':green_in_red_mean,
             'red_in_red_std':red_in_red_std, 'red_in_red_mean':red_in_red_mean,
             'file':filename,
             'red_voxels':red_in_red.shape[0], 'no_red_voxels':p2.shape[0],
              'green_area':green_area, 'no_red_area':p2_area, 'red_area':red_area,
             'rdna_mean':rdna_mean, 'rdna_std':rdna_std, 'rdna_area':rdna_area, 'rdna_vol':rdna_vol,
            }
        shd = {'i':index, 'file':filename,
               'cell_length':this_length,
               'red_voxels':red_in_red.shape[0],
               'red_mean':red_in_red_mean,
               'vol':just_green.shape[0],
               'rdna_vol':rdna_vol,
               'no_red_voxels':p2.shape[0],
               'green_mean':green_mean,
               'rdna_mean':rdna_mean,
               'p2_mean':p2_mean,
               'xcm':xcm, 'ycm':ycm,
               'green_rdna_dist':green_rdna_dist,
               'green_red_dist':green_red_dist,
               'rdna_red_dist':rdna_red_dist,
               'blue_vol':just_blue.shape[0],
               'blue_mean':blue_mean,
               'blue_rdna_mean':blue_rdna_mean,
               'blue_rdna_vol':blue_rdna_vol,
               'blue_red_dist':blue_red_dist, 
               'blue_green_dist':blue_green_dist, 
              }

        dlist.append(d)
        dsslist.append(shd)

    with open(objname, 'wb') as pf:
        pickle.dump(no_rdna_list, pf)

    pdf = pd.DataFrame(dlist)
    sspdf = pd.DataFrame(dsslist)
    return pdf, sspdf
    



def cnn_setup(modelfile, size=400, probability=0.95):
    '''
    Setup the gpu or cpu as the device and create the neural network
    with size and probability
    
    Parameters
    ----------
    modelfile : str - The location of the trained model
    size : int - The size of image patches
    probability: float - Probability of found objects to be considered
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    #device = torch.device('cpu')
    tm = train.get_model(2)
    tm.load_state_dict(torch.load(modelfile))#,map_location=torch.device('cpu')))
    cnn = infer.predict(tm, device=device, size=(size, size),
                        probability=probability)
    
    return cnn

def cellpose_setup(modelfile, diameter, cpt=-3, flow_threshold=.8):
    """Create a cellpose model object from the modelfile

    Parameters
    ----------
    modelfile : str 
        The saved weights in the path modelfile
    diameter : int 
        The diameter to use for infering the model

    Returns
    -------
    _type_
        _description_
    """    
    channels = [0,0]
    model = models.CellposeModel(gpu=True,pretrained_model=modelfile)
    runcp = functools.partial(model.eval, diameter=diameter, channels=channels,
                resample=True, cellprob_threshold=cpt,
                flow_threshold=flow_threshold)
    return runcp
    
    
def pfunc(d, **kwargs):
    print(f"Working on {d['filename']}")
    if d['image'] is None:
        print(f"**### {d['filename']} in None")
        return None
        
    if (len(d['image'].shape) < 4):
        print(f"**### {d['filename']} has shape {d['image'].shape}")
        return None

    t1 = time.time()
    just_green_min_size = kwargs['just_green_min_size']
    green_in_red_min_size = kwargs['green_in_red_min_size']
    cnn = kwargs['cnn']
    is_cellpose = kwargs['is_cellpose']
  
    try:
        _, sspdf = process(d['filename'], d['image'], cnn, is_cellpose,
                        just_green_min_size=just_green_min_size,
                        green_in_red_min_size=green_in_red_min_size)
    except Exception as err:
        print(f"!### {d['filename']}, {err}")
        sspdf = None

    t2 = time.time()
    print(f"{d['filename']} ran in {t2 - t1} seconds")
    return sspdf

    
def nd2_generator(filenames, channels, zstart, zstop):
    for filename in filenames:
        image = get_nd2_image(filename, channels=channels)
        if image is not None:
            if len(image.shape) >= 4:
                image = image[zstart:zstop] 
        yield {'image':image, 'filename':filename}

    
def run(folder, pkl_name, modelfile, size=400, probability=0.95,
        just_green_min_size=200, green_in_red_min_size=64,
        globpattern='*.nd2', channels=[0,1,2, 3],
        zstart=None, zstop=None, bin=1,
        mtype='nikon', n_jobs=1):

    if not folder.endswith('/'):
        folder += '/'

    if not os.path.exists('Data/Masks'):
        os.makedirs('Data/Masks')
    if not os.path.exists('Data/NoRDNA'):
        os.makedirs('Data/NoRDNA')
    
    is_cellpose = False
    if modelfile is None:
        cnn = None
    elif "cellpose" in modelfile:
        print("Setting up Cellpose Model")
        is_cellpose = True 
        cnn = cellpose_setup(modelfile, 50)
    else:
        print("Setting up Mask R-CNN Model")
        cnn = cnn_setup(modelfile, size, probability)

    if mtype =='phenix':
        #phenix stuff
        filenames = phenix.groupfiles(sorted(glob.glob(folder + globpattern)))
    else:
        filenames = glob.glob(folder + globpattern)

    dflist = list()
    ssdflist = list()
    nfile = len(filenames)

    kwargs = {'just_green_min_size':just_green_min_size,
              'green_in_red_min_size':green_in_red_min_size,
              'cnn':cnn,
              'is_cellpose':is_cellpose}

    if mtype=='nikon':
        if n_jobs > 1:
            print('Running parallel')
            ssdflist = Parallel(n_jobs=n_jobs)\
                (delayed(pfunc)(d, **kwargs)
                    for d in nd2_generator(filenames, channels, zstart, zstop))
        else:
            ssdflist = list()
            for d in nd2_generator(filenames, channels, zstart, zstop):
                ssdflist.append(pfunc(d, **kwargs))
                
    else:
        for ni, filename in enumerate(filenames):
            base = os.path.basename(filename)
            if mtype == 'phenix':
                #do phenix stuff
                image = phenix.fieldstack(filenames[filename])
            else:
                print("Working on ", base)
                if base.endswith('nd2'):
                    image = get_nd2_image(filename, channels=channels)
                    image = image[zstart:zstop]
                elif base.endswith('tif'):
                    image = tifffile.imread(filename)
                    image = image[channels,:,:]
                    if len(image.shape) == 3:
                        image = np.expand_dims(image, 0)
                else:
                    continue

                if image[:,1,:,:].max() > 66000:
                    print(filename)
                    print("bright spot found,  image[:,1,:,:].max(), continuing")
                    continue

                pdf,sspdf = process(filename, image, cnn, just_green_min_size,
                                    green_in_red_min_size)
                #dflist.append(pdf)
                #ssdflist.append(sspdf)

            print(f'{ni} out of {nfile}')

    ssdflist = [s for s in ssdflist if s is not None]
    ssdf = pd.concat(ssdflist)
    ssdf.to_pickle(pkl_name)


if __name__ == '__main__':
    print(sys.argv)
    folder = sys.argv[1]
    pkl_name = sys.argv[2]
    modelfile = sys.argv[3]

    run(folder, pkl_name)
