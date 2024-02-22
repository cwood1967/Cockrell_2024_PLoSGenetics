from re import I
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
from skimage.transform import resize


def resize_to_patch(image, patchsize):
    ''' Resize the image to the size of the patch

    Parameters
    ----------
    image : array
        the image to resize
    patchsize : tuple (int, int)  
        the size of the patch in (y, x)
        
    Returns
    -------
    pad : array
        the resized image
    '''
    py, px = patchsize[0], patchsize[1]
    sy, sx = image.shape 
    ry = sy/py
    rx = sx/px

    sc = max(ry, rx)
    
    if sc > 1:
        _image = resize(image, (int(sy/ry), int(sx/rx)))
    else:
        _image = image.copy()
        
    _sy, _sx = _image.shape
    dy = (py - _sy)//2 
    dx = (px - _sx)//2
    
    pad = np.zeros(patchsize, dtype=np.float32)
    pad[dy:dy + _sy, dx:dx + _sx] = _image
    return pad


def resize_to_original(pad, patchsize, original_size):
    '''Resize an image (predicted usually) back to the original size
    
    Parameters
    ----------
    pad : array
        the image to resize
    patchsize : tuple
        the size the pad was resized to
    original_size : tuple
        the size before pad was resize, and the output size
        
    Returns
    -------
    image : array
        image resized bask to the original size:w
    ''' 

    py, px = patchsize
    sy, sx = original_size
    
    ry = sy/py
    rx = sx/px
    
    sc = max(ry, rx)
    if sc > 1:
        _sy = int(sy/ry)
        _sx = int(sx/rx) 
    else:
        _sx = sx
        _sy = sy
    
    dy = (py - _sy)//2
    dx = (px - _sx)//2
    
    _image = pad[dy:dy + _sy, dx:dx + _sx]
    image = resize(_image, (sy, sx))
    return image
    
    
def image_to_patches(image : np.array, size : Tuple =(256, 256), shift=False):
    ndim = len(image.shape)
    #print("input image shape", image.shape)
    if ndim==3 and image.shape[-1] in [3,4]:
        yc = 0
        xc = 1
    else:
        xc = -1
        yc = -2
        
    sizeX = image.shape[xc]
    sizeY = image.shape[yc]
    
    amin = image.min(axis=(yc, xc), keepdims=True)
    amax = image.max(axis=(yc, xc), keepdims=True)
    image = (image - amin)/(amax - amin)
    
    patch_x = size[1]
    patch_y = size[0]
    ## start off assuming one t and z, possibly multiple channels
    ## and will put everything in a patch into network
    numx = sizeX//patch_x
    numy = sizeY/patch_y

    dx = patch_x//2
    dy = patch_y//2
    patch_dict = dict()
    for jy in np.arange(0, sizeY, dy):
        jyf = jy + patch_y
        if jyf > sizeY:
            jyf = sizeY
            jy = sizeY - patch_y
        for jx in np.arange(0, sizeX, dx):
            jxf = jx + patch_x
            if jxf > sizeX:
                jxf = sizeX
                jx = sizeX - patch_x
            if yc == 0:
                patch_dict[(jy, jx)] = (image[jy:jyf, jx:jxf, :])
            elif ndim == 2:
                patch_dict[(jy, jx)] = (image[jy:jyf, jx:jxf])
            else:
                patch_dict[(jy, jx)] = (..., image[jy:jyf, jx:jxf])
            if jxf >= sizeX:
                break
        if jyf >= sizeY:
            break
    
    if shift:
        for jy in np.arange(patch_y//2, sizeY, dy):
            jyf = jy + patch_y
            if jyf > sizeY:
                jyf = sizeY
                jy = sizeY - patch_y
            for jx in np.arange(patch_x//2, sizeX, dx):
                jxf = jx + patch_x
                if jxf > sizeX:
                    jxf = sizeX
                    jx = sizeX - patch_x
                patch_dict[(jy, jx)] = (image[..., jy:jyf, jx:jxf])
    
    return patch_dict

def reconstruct_from_dict(patch_dict, shape, patchsize, shift=False,
                          edges=.01):
    if len(shape) == 3 and shape[-1] in [3, 4]:
        rshape = shape[:2]
    else:
        rshape = shape
    recon = np.zeros(rshape, dtype=np.float32)

    dx = patchsize[1]
    dy = patchsize[0]
    pad = np.zeros(patchsize, dtype=np.float32)
    if len(patch_dict) == 1:
        vs = list(patch_dict.values())[0]
        all_boxes = vs[1]
        recon = np.squeeze(vs[0])
    else:
        pds = int(edges*dx) #max(12, int(edges*dx))
        pdy = dy - 2*pds
        pdx = dx - 2*pds
        pad[pds:dy-pds, pds:dx-pds] = 1
        all_boxes = list()
        for k, vs in patch_dict.items():
            v = np.squeeze(vs[0])
            if len(v.shape) > 2:
                v = v.sum(axis=0)
            boxes = vs[1]
            ky, kx = k
            # ky = ky + pds
            # kx = kx + pds
            # pv = v[pds:dy - pds, pds: dx - pds]
            pv = pad*v

            # rv = recon[..., ky:ky + pdy, kx:kx + pdx]
            rv = recon[..., ky:ky + dy, kx:kx + dx]
            recon[..., ky:ky + dy, kx:kx + dx] = np.where(pv >= rv, pv, rv)
            
            if boxes:
                for box in boxes:
                    box = box.copy()  
                    try:
                        box[0] += kx
                        box[2] += kx
                        box[1] += ky
                        box[3] += ky
                    except Exception as e:
                        pass
                    all_boxes.append(box)
    return recon, all_boxes
    

def getHighProb(predictions, prob=0.9):
    masks = predictions['masks']
    if len(masks) == 0:
        return [0], [0]
    scores = predictions['scores']
    boxes = predictions['boxes']
    res_list= list()
    box_list = list()
    score_list = list()
    res_list.append(0*masks[0])
    for i, s in enumerate(scores):
        if s >= prob:
            if i==0:
                res_list.pop()
            res_list.append(masks[i])
            box_list.append(boxes[i])
            score_list.append(scores[i])

    return np.stack(res_list), (box_list, score_list)


def res_to_cpu(predictions):
    pcpu = list()
    for p in predictions:
        masks = p['masks'].cpu().detach().numpy()
        scores = p['scores'].cpu().detach().numpy()
        boxes = p['boxes'].cpu().detach().numpy()
        pcpu.append({'masks':masks, 'scores':scores, 'boxes':boxes})
        
    return pcpu
    
def tensors_to_cpu(tensor_list):
    tcpu = [t.cpu().detach().numpy() for t in tensor_list]
    return tcpu


def res_image(x, shape, max_project=True, prob=.75):
    """Return maximum projection of probabilities

    Parameters
    ----------
    x : dict
        predictions dictionary 
    shape : shape of the patch
        [description]
    prob : float, optional
        [description], by default .75

    Returns
    -------
    numpy array
        max projection of probabilites
    """
    r, boxes = getHighProb(x, prob=prob)
    #print("res_image", r.shape)
    if len(r) > 1:
        if max_project:
            return r.max(axis=0), boxes 
        else:
            return r, boxes
    else:
        return np.zeros(shape, dtype=np.float32), None
    
    
def plot_image(recon, image, boxes, filename):
    
    fig, ax = plt.subplots(2,1, figsize=(8,16))
    ax[0].imshow(recon)
    print(len(boxes))
    try:
        for b in boxes:
            rx, ry, rxf, ryf = b
            w = rxf - rx
            h = ryf - ry
            rp = patches.Rectangle((rx, ry), w, h, edgecolor='r', facecolor='none')
            _ = ax[0].add_patch(rp)
    except:
        print("no boxes to show")
        
    ax[1].imshow(image.squeeze())
    plt.show()
    if filename is not None:
        fname = filename[:-3] + ".pdf"
        plt.savefig('Data/Results/' + fname)
        print('Saved', 'Data/Results/' + fname)