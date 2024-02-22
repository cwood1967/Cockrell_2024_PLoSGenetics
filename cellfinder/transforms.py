import random
import torch
import numpy as np

from torchvision.transforms import RandomRotation as torchRotation
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class RandomRotation(object):
    def __call__(self, image, mask):
        
        pil = ToPILImage(mode='F')
        image = pil(image)
        mask = pil(mask)
        nr = 90*np.random.rand()
        image = image.rotate(nr)
        mask = mask.rotate(nr)
        image, mask = ToTensor()(image, mask)
        return image, mask

class RandomHFlip(object):
    """
    Class to flip an image horizontally
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask):
        """ used to perfrom the transform
        
        Parameters
        ----------
        image: a torch (tensor channel, y, x)
        mask: a torch (y, x)

        Returns
        -------
        Transformed image and target
        """

        if random.random() > self.prob:
            image = image.flip(-1)
            mask = mask.flip(-1)

        return image, mask

class RandomCrop(object):
    """
    Class to randomly crop to a size 
    """
    def __init__(self, size):
        '''
        Parameters
        ----------
        size: tuple(y, x) - size of the cropped image
        '''
        self.size = size

    def __call__(self, image, mask):
        """ Performs the  crop transform

        Parameters
        ----------
        image: torch tensor (channel, y, x)
        mask: torch tensor (channel, y, x)

        Returns
        -------
        Transformed image and mask
        """
        h, w = image.shape[-2:]
        # print("image shape in crop", image.shape)
        ypad = h 
        xpad = w 
        if h < self.size[0]:
            ypad = self.size[0] + 100
        if w < self.size[1]:
            xpad = self.size[1] + 100
        
        pad = np.zeros((ypad, xpad, image.shape[0]), dtype=np.float32) 
        mask_pad = np.zeros((ypad, xpad, 1), dtype=np.float32) 
        # mask_pad = pad.copy()
        pad = F.to_tensor(pad).type(torch.FloatTensor)
        mask_pad = F.to_tensor(mask_pad).type(torch.FloatTensor)
        yoffset = (ypad - h)//2 
        xoffset = (xpad - w)//2
        
        pad[:, yoffset:yoffset + h, xoffset:xoffset + w] = image 
        mask_pad[:, yoffset:yoffset + h, xoffset:xoffset + w] = mask 
        #mask_pad[yoffset:yoffset + h, xoffset:xoffset + w] = mask 
        image = pad
        mask = mask_pad 

        hc = self.size[0]
        wc = self.size[1]

        ymax = h - hc
        xmax = w - wc
        if hc >= h:
            ry = 0
        else:        
            ry = random.randint(0, ymax)
        if wc >=w:
            rx = 0
        else:
            rx = random.randint(0, xmax)

        image = image[:, ry:ry + hc, rx:rx + wc]
        mask = mask[:, ry:ry + hc, rx:rx + wc]
        #mask = mask[ry:ry + hc, rx:rx + wc]
        return image, mask


class RandomVFlip(object):
    """
    Class to flip an image vertically
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, mask):
        """ used to perfrom the transform
        
        Parameters
        ----------
        image: a torch (tensor channel, y, x)
        mask: a torch (tensor channel, y, x)

            Returns
            -------
            Transformed image and mask
            """

        if random.random() < self.prob:
            image = image.flip(-2)
            mask = mask.flip(-2)

        return image, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __xcall__(self, image, mask):
        nc = image.shape[0]
        #print(image.mean(), image.std(), image.max(), image.min())
        imean = image.mean()
        istd = image.std()
        #new_image = F.normalize(image, self.mean, self.std)
        new_image = self.std[0]*image/istd
        new_image = new_image - new_image.mean() + self.mean[0]
        #print(new_image.mean(), new_image.std(), new_image.max(), new_image.min())
        #mask = F.normalize(mask, self.mean, self.std)
        return new_image, mask
        
class ToTensor(object):
    def __call__(self, image, mask):
        """
        Convert numpy array to tensor. Will move channel dimension to in front
        of y,x. For example, [512, 512, 3] -> [3, 512, 512]
        Two dimensional images will be converted: [512, 512] -> [1, 512, 512]
        """
        image = F.to_tensor(image).type(torch.FloatTensor)
        mask = F.to_tensor(mask).type(torch.FloatTensor)
        return image, mask

def get_transforms(cropsize=(400,400), prob=0.5, train=True):
    transforms = list()
    if train:
        transforms.extend([ToTensor(),
                           RandomCrop(cropsize),
                           RandomHFlip(prob),
                           RandomVFlip(prob),
                           #ToTensor()
        ])
    else:
        transforms.extend([ToTensor()]) #, RandomCrop(cropsize)])
    # if train:
    #     transforms.extend([ToTensor(), RandomCrop(cropsize)])
        
    # else:
    #     transforms = ([ToTensor()])
        
    return Compose(transforms)
    