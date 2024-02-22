import glob
import os
import pickle
from re import I 

import numpy as np
import tifffile
import torch

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn

from  . import utils
from . import transforms
from . import train

class predict():
    
    def __init__(self, model,
                 device='cuda', size=(400,400), axis=(-2, -1),
                 max_project=True, probability=0.9, edges=.01):
        """[summary]

        Parameters
        ----------
        model : torch model
            pytorch maskrcnn model
        device : str
            cuda or cpu, by default 'cuda'
        size : tuple, optional
            size of image patches, by default (400,400)
        axis : tuple of length 2, by default (-2, -1) 
            what axis the image composes
            
        """
        
        self.model = model
        self.patchsize = size
        _ = self.model.eval()
        self.device = torch.device(device)
        self.model.to(device)
        self.xt = transforms.get_transforms(train=False)
        self.probability = probability
        self.max_project = max_project
        self.edges = edges
    
    
    @torch.no_grad()
    def __call__(self, image, step=10, norm_axis=(-2, -1), return_patches=False):
        #print(return_patches) 
        image = self.normalize(image, axis=norm_axis)
        torch.cuda.empty_cache()
            
        
        tensor_list_gpu = list()
        key_list = list()
        dummy_mask = np.zeros((self.patchsize[0], self.patchsize[1]), dtype=np.float32)
        patches = utils.image_to_patches(image, size=self.patchsize)
        for k, v in patches.items():
            vtrans, _ = self.xt(v, dummy_mask)
            #print(vtrans.shape)
            tensor_list_gpu.append(vtrans.to(self.device))
            key_list.append(k)

        res = list()
        #step = 40
        for i in range(0, len(tensor_list_gpu), step):
            rg = self.model(tensor_list_gpu[i:i+step])
            r = utils.res_to_cpu(rg)
            del rg
            res.extend(r)

        try:
            tensor_list = utils.tensors_to_cpu(tensor_list_gpu)
            del tensor_list_gpu
        except:
            pass
        torch.cuda.empty_cache()

        patch_dict = {key_list[i]:
            utils.res_image(res[i],tensor_list[0].shape,
                            max_project=self.max_project,
                            prob=self.probability)
            for i in range(len(tensor_list))}

        recon, boxes = utils.reconstruct_from_dict(patch_dict,
                                            image.shape,
                                            self.patchsize, shift=True,
                                            edges=self.edges)
    
        self.prediction = recon
        self.boxes = boxes
        if return_patches:
            return recon, boxes, patch_dict
        else:
            return recon, boxes

    @torch.no_grad()
    def predict_batch(self, batch, norm_axis=(1,2)):
        batch = self.normalize(batch, axis=norm_axis)
        # torch.cuda.empty_cache()
        dummy_mask = np.zeros((self.patchsize[0], self.patchsize[1]),
                              dtype=np.float32)

        tensor_list_gpu = list()
        for v in batch:
            vtrans, _ = self.xt(v, dummy_mask)
            tensor_list_gpu.append(vtrans.to(self.device))
        
        res = self.model(tensor_list_gpu)
        for _r in res:
            _r['masks'] = _r['masks'].detach().cpu().numpy()    
            _r['boxes'] = _r['boxes'].detach().cpu().numpy()    
            _r['scores'] = _r['scores'].detach().cpu().numpy()    

        torch.cuda.empty_cache()
        return res
        
    def normalize(self, a, axis=(-2,-1)):
        amin = a.min(axis=axis, keepdims=True)
        amax = a.max(axis=axis, keepdims=True)
        return  (a - amin)/(amax - amin)
        
        
def inferdir(dirpath, modelfile, savedir, globpattern='*.tif',
             device='cuda', size=(400,400), axis=(-2, -1),
             probability=0.9):
    
    if not dirpath.endswith('/'):
        dirpath += '/'
    if not savedir.endswith('/'):
        savedir += '/'
    tm = train.get_model(2)
    tm.load_state_dict(torch.load(modelfile))
    files = sorted(glob.glob(dirpath + globpattern))

    cnn = predict(tm, size=size, probability=probability)
    for f in files:
        x = tifffile.imread(f)
        r, bx = cnn(x)
        bn = os.path.basename(f)
        bn = bn[:-3]
        rfile = savedir + "inferred_" + bn + "tif"
        tifffile.imwrite(rfile, r.astype(np.float32))
        bxname = savedir + "inferred_" + bn + "pkl"
        with open(bxname, 'wb') as p:
            pickle.dump(bx, p)
        