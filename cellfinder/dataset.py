import os

import torch
import torchvision
import numpy as np
#from PIL import Image
from skimage.io import imread
from scipy.ndimage import measurements


class PombeDataset(object):

    def __init__(self, rootdir, image_dir, mask_dir,
                 transforms, ext=None, multiplier=1, labels=False):
        self.rootdir = rootdir
        self.transforms = transforms
        self.islabels = labels
        image_path = os.path.join(rootdir, image_dir)
        mask_path = os.path.join(rootdir, mask_dir)
        images = list(sorted(os.listdir(image_path)))
        self.remove_nonimage(images, ext=ext)
        self.images = [os.path.join(rootdir, image_dir, j) for j in images]
        self.images = multiplier*self.images
        print("Images in training set:", len(self.images))
        masks = list(sorted(os.listdir(mask_path)))
        self.remove_nonimage(masks)
        self.masks = [os.path.join(rootdir, mask_dir, j) for j in masks]
        self.masks = multiplier*self.masks

    def __getitem__(self, idx):
        #print("idx", idx)
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        a_image = imread(image_path).astype(np.float32)
        if len(a_image.shape) == 2:
            a_image = np.expand_dims(a_image, -1)

        amin = a_image.min(axis=(0, 1), keepdims=True)
        amax = a_image.max(axis=(0, 1), keepdims=True)
        a_image = (a_image - amin)/(amax - amin)
        #a_image = np.stack(3*[a_image.squeeze()], axis=-1)
        
        a_mask = imread(mask_path)
        boxes = []
        while len(boxes) == 0:
            if self.transforms is not None:
                image, mask = self.transforms(a_image, a_mask)

            _mask = mask.numpy()

            mask_labels, nobjects = self.label_mask(_mask, minsize=50) 
            oid = np.unique(mask_labels)[1:]
            sep_masks = (mask_labels == oid[:, None, None])
            sep_masks = torch.as_tensor(sep_masks, dtype=torch.uint8)
            boxes = self.get_boxes(mask_labels, nobjects)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((nobjects,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:,1])*(boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((nobjects,), dtype=torch.int64)
            
        target = dict()
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = sep_masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)
        
        return image, target

    def label_mask(self, mask, minsize = None):
        if self.islabels:
            labels = mask
            nlabels = int(mask.max())
        else:
            labels, nlabels = measurements.label(mask)
        #print("Nlabels",nlabels)
        if minsize is not None:
            for i in range(1, nlabels):
                nx = (labels == i).sum()
                if nx < minsize:
                    labels[labels == i] = 0
                    #print(nx, i, "getting rid of small", (labels == i).sum())
        
            labels, nlabels = measurements.label(labels > 0)
        return labels, nlabels
    
    def remove_nonimage(self, pathlist, ext=None):
        if ext is None:
            ew = ('png', 'tif', 'tiff', 'jpg', 'jpeg')
        else:
            ex = ext
            
        for j in pathlist:
            if not j.endswith(ew):
                pathlist.remove(j)

    def get_boxes(self, mask_labels, nobjects):

        boxes = list()
        for i in range(1, nobjects + 1):
            pos = np.where(mask_labels == i)
            xmin = pos[2].min()
            xmax = pos[2].max()
            
            if xmax == xmin:
                mask_labels[mask_labels == i] = 0
            ymax = pos[1].max()
            ymin = pos[1].min()
            if ymax == ymin:
                mask_labels[mask_labels == i] = 0
            
            if (ymax > ymin) and (xmax > xmin):    
                boxes.append([xmin, ymin, xmax, ymax])

        return boxes

    def __len__(self):
        return len(self.images)
    
    def napari_boxes(self, b):
        nbox = np.array(
                        [[b[1], b[0]],
                        [b[3], b[0]],
                        [b[3], b[2]],
                        [b[1], b[2]]]
        )
        return nbox