from RANSAC_Flow.quick_start.coarseAlignFeatMatch import CoarseAlign
import sys
sys.path.append('RANSAC_Flow/utils/')
import outil

sys.path.append('RANSAC_Flow/model/')
import model as model

import PIL.Image as Image
import os
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle
import pandas as pd
import kornia.geometry as tgm
# from scipy.misc import imresize
from itertools import product
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import matplotlib.pyplot as plt


def get_Avg_Image(Is, It) :

    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))


class imageAlign:
    def __init__(self):
        self._initNetwork()
        self._initCoarseModel()

    def _initNetwork(self):
        resumePth = 'RANSAC_Flow/model/pretrained/MegaDepth_Theta1_Eta001_Grad1_0.774.pth'
        kernelSize = 7
        self.Transform = outil.Homography
        self.nbPoint = 4
        network = {
            'netFeatCoarse' : model.FeatureExtractor(),
            'netCorr'       : model.CorrNeigh(kernelSize),
            'netFlowCoarse' : model.NetFlowCoarse(kernelSize),
            'netMatch'      : model.NetMatchability(kernelSize),
        }
        for key in list(network.keys()):
            network[key].cuda()
            self.typeData = torch.cuda.FloatTensor

        param = torch.load(resumePth)
        msg = 'Loading pretrained model from {}'.format(resumePth)
        print (msg)

        for key in list(param.keys()):
            network[key].load_state_dict(param[key])
            network[key].eval()
        self.network = network

    def _initCoarseModel(self):
        nbScale = 7
        coarseIter = 10000
        coarsetolerance = 0.05
        minSize = 224
        imageNet = True  # we can also use MOCO feature here
        scaleR = 1.2
        self.coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)

    def alignImages(self, sourceImage, targetImage, isPath=False, isTensor=False):
        if isPath:
            I1 = Image.open(sourceImage).convert('RGB')
            I2 = Image.open(targetImage).convert('RGB')
        elif isTensor:
            trans = transforms.ToPILImage()
            I1 = trans(sourceImage.cpu())
            I2 = trans(targetImage.cpu())
        else:
            I1 = sourceImage.convert('RGB')
            I2 = targetImage.convert('RGB')

        self.coarseModel.setSource(I1)
        self.coarseModel.setTarget(I2)

        I2w, I2h = self.coarseModel.It.size
        featt = F.normalize(self.network['netFeatCoarse'](self.coarseModel.ItTensor))

        #### -- grid
        gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
        gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
        grid = torch.cat((gridX, gridY), dim=3).cuda()
        warper = tgm.HomographyWarper(I2h,  I2w)

        bestPara, InlierMask = self.coarseModel.getCoarse(np.zeros((I2h, I2w)))
        bestPara = torch.from_numpy(bestPara).unsqueeze(0).cuda()

        flowCoarse = warper.warp_grid(bestPara)
        I1_coarse = F.grid_sample(self.coarseModel.IsTensor, flowCoarse)
        I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())

        featsSample = F.normalize(self.network['netFeatCoarse'](I1_coarse.cuda()))


        corr12 = self.network['netCorr'](featt, featsSample)
        flowDown8 = self.network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H

        flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
        flowUp = flowUp.permute(0, 2, 3, 1)

        flowUp = flowUp + grid

        flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

        I1_fine = F.grid_sample(self.coarseModel.IsTensor, flow12)
        I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())

        res = {
            'sourceImage' : I1,
            'targetImage' : I2,
            'alignedImage': I1_fine_pil,
            'alignedTensor':I1_fine,
        }
        return res


if __name__ == '__main__':
    I1_path = 'RANSAC_Flow/img/test_source_cat.png'
    I2_path = 'RANSAC_Flow/img/test_target_cat.png'

    iAlign = imageAlign()
    res = iAlign.alignImages(I1_path, I2_path, isPath=True)
    I1_fine_pil = res['alignedImage']
    I2 = res['targetImage']
    It = iAlign.coarseModel.It

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Source Image (Fine Alignment)')
    plt.imshow(I1_fine_pil)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Target Image')
    plt.imshow(I2)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Overlapped Image')
    plt.imshow(get_Avg_Image(I1_fine_pil, It))
    plt.show()
