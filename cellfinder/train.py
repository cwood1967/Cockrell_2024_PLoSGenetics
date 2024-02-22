import os
from re import I
import time

import numpy as np
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from matplotlib import pyplot as plt
from skimage.io import imsave
from . import dataset
from . import transforms


def get_model(num_classes, pretrained=True):
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    print("IFM", in_features_mask)
    hidden_layer = 256
    
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    
    return model

    
def collate(batch):
    return tuple(zip(*batch))

def test_eval(model, test_dir):
    pass

def main(root='Data', image_dir='Images', mask_dir='Masks',
         epochs=50, cropsize=(400, 400),
         model=None, batch_size=8,
         test_ratio=0, test_interval=10, display=False,
         pretrained=True, png_progress=False, mask_labels=True):
    '''
    Train the model and save network snapshots.
    
    root : str
       The path to the data directory. Default is 'Data'
    
    image_dir : str
        Directory inside to root where traing images are located. Default 'Images'
    
    mask_dir : str
        Directory for training masks. Default is 'Masks'
    
    epochs : int
        Number of epochs to train. Default 50
    
    cropsize : tuple (int, int)
        Size of patches to use in training. Default (400, 400)
    
    batch_size : int
        Size of image batches. Default 8
    
    Returns : MaskRCNN
        The trained pytorch Mask RCNN model
    '''

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    num_classes = 2
    
    xforms = transforms.get_transforms(cropsize=cropsize)
    testforms = transforms.get_transforms(cropsize=cropsize, train=False)
    data_all = dataset.PombeDataset(root, image_dir, mask_dir, xforms,
                                    labels=mask_labels)
    #data_test = dataset.PombeDataset(root, image_dir, mask_dir, xforms)
     
    indices = torch.randperm(len(data_all)).tolist()
    if test_ratio > 0:
        ntest = max(int(test_ratio*len(data_all)), 1)
    else:
         ntest = 0

    print("####", ntest, test_ratio)
    if ntest == 0:
        data = torch.utils.data.Subset(data_all, indices)
    else:
        data = torch.utils.data.Subset(data_all, indices[:-ntest])
        data_test = torch.utils.data.Subset(data_all, indices[-ntest:])
        data_test.dataset.transforms = testforms
        data_test_loader = torch.utils.data.DataLoader(
            data_test, batch_size=ntest, shuffle=False, num_workers=2,
            collate_fn=collate)

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate)
    
    if model is None:
        model = get_model(2, pretrained=pretrained)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9,
                                weight_decay=0.0005)
    
    print("lr", 0.002)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.1)
   
     
    num_epochs = epochs
    model.train()
    print(f'Training on {device}...')
    min_mask_loss = 10.0
    timefmt = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    modeldir = f"model_{timefmt}"
    os.mkdir(modeldir)
    for i in range(num_epochs):
        elosses = 0 
        for images, targets in data_loader:
            #print('LEN', len(images))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
           
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            elosses += losses 
            #if i % (num_epochs//100) == 0: 
            #print("**", i, losses)#['loss_mask'])
            # print("***", i, loss_dict['loss_box_reg'].cpu().detach().numpy(),
            #       loss_dict['loss_mask'].cpu().detach().numpy())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(i, elosses)
        if (ntest > 0) and (i % test_interval) == 0:
            _ = model.eval()
            for _ti, _tt in data_test_loader:
                _ti = list(i.to(device) for i in _ti)
                #_tt = list(_tt.do(device) for t in _tt)
                _tr = model(_ti)
                _tr = [_r['masks'].cpu().detach().numpy().max(axis=0).squeeze()
                        for _r in _tr] 
                _tt = [_t['masks'].cpu().detach().numpy().max(axis=0)
                        for _t in _tt]

                if png_progress:
                    if not os.path.exists("png_progress"):
                        os.mkdir("png_progress")
                    pg = 255*_tr[0]
                    pg = pg.astype(np.uint8)
                    cputi = _ti[0].cpu().detach().numpy()
                    cputi = 255*(cputi - cputi.min())/(cputi.max() - cputi.min())
                    cputi = cputi.astype(np.uint8)
                    cputi = np.moveaxis(cputi, 0, -1)
                    imsave(f"png_progress/{i:04d}_image.png", cputi)
                    imsave(f"png_progress/{i:04d}_mask.png", pg)

                print(len(_tr), len(_tt)) 
                _tr = np.where(np.stack(_tr) > .8, 1, 0)#.max(axis=0)
                _tt = np.stack(_tt)#.max(axis=0)
                #dice = 2*((_tr*_tt).sum())/(_tr.sum() + _tt.sum() + 1.)
                dice = np.array([2*(_r*_t).sum()/(_t.sum() + _r.sum() + 1.) 
                        for _r, _t in zip(_tr, _tt)])

                print("*** Dice", dice.sum()/len(dice))
            if display:
                dmin = dice.argmin()
                print("Min dice", dmin, dice[dmin])
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                idisp = _ti[dmin].cpu().detach().numpy()
                idisp = np.moveaxis(idisp, 0, -1)
                ax[0].imshow(idisp)
                ax[1].imshow(_tr[dmin])
                plt.show()

            _ = model.train()

        if elosses < min_mask_loss:
            min_mask_loss = elosses        
        # if loss_dict['loss_mask'].cpu().detach().numpy() < min_mask_loss:
        #     min_mask_loss = loss_dict['loss_mask'].cpu().detach().numpy() 
            torch.save(model.state_dict(), f"{modeldir}/trained_min_mask_model_{i:04d}.pt")
            print("i, saved min mask loss")
            #lr_sched.step()
    print(loss_dict)
    torch.save(model.state_dict(), f"{modeldir}/trained_last_{i:04d}.pt")
    return model
