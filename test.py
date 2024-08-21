import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset.floorplan_dataset_maps_functional_high_res import FloorplanGraphDataset, floorplan_collate_fn

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image, ImageDraw, ImageFont
import svgwrite
from models.models import Generator
# from models.models_improved import Generator

from misc.utils import _init_input, ID_COLOR, draw_masks, draw_graph, estimate_graph
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import glob
import cv2
import webcolors
import time
from tqdm import tqdm

CrPath = ''
current_index = '_j'

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--checkpoint", type=str, default='checkpoints/pretrained.pth', help="checkpoint path")
parser.add_argument("--data_path", type=str, default='data/', help="path to dataset list file")
parser.add_argument("--out", type=str, default='dump', help="output folder")
opt = parser.parse_args()
print(opt)

# Create output dir
os.makedirs(opt.out, exist_ok=True)

# Initialize generator and discriminator
model = Generator()


try:
    dict = torch.load(f"{CrPath}checkpoints/current{current_index}.pth", map_location=torch.device('cpu'))
    model.load_state_dict(dict['g'], strict=True)

except Exception as e:
    print(f"Не удалось прочитать генератор: {e}")

model = model.eval()

# Initialize variables
#if torch.cuda.is_available():
#    model.cuda()

# initialize dataset iterator
fp_dataset_test = FloorplanGraphDataset(opt.data_path, transforms.Normalize(mean=[0.5], std=[0.5]), split='test')
fp_loader = torch.utils.data.DataLoader(fp_dataset_test, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False)
# optimizers
#Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Tensor = torch.FloatTensor
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "balcony": 5, "entrance": 6, "dining room": 7,
              "study room": 8,
              "storage": 10 , "front door": 15, "unknown": 16, "interior_door": 17}
NeihboursForOuts = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
def TestInd(rec):
    return -1 if (np.array(rec) * NeihboursForOuts).sum()== 0 else 1
def AddOuts(given_masks_in, given_nds, given_eds, mks):
    ind = len(given_nds)
    outer_mks = torch.cat( (mks.unsqueeze(0), torch.ones(1, 64, 64)), 0 ).unsqueeze(0)
    given_masks_in = torch.cat( (given_masks_in, outer_mks), 0)
    outer_given_nds = torch.zeros((1, 18))
    outer_given_nds[0, 10] = 1
    outer_given_nds = torch.cat((given_nds, outer_given_nds), 0)

    z = torch.randn(len(outer_given_nds), 128).float()
    outer_given_eds = torch.tensor([[ind, TestInd(rec), room] for room, rec in enumerate(given_nds)])
    return z, given_masks_in, outer_given_nds, torch.cat( (given_eds, outer_given_eds), 0)

# run inference
def _infer(graph, model, prev_state, mks):
    
    # configure input to the network
    z, given_masks_in, given_nds, given_eds = _init_input(graph, prev_state)

    pmks = (0.5 - mks/2).prod(0)
    z, given_masks_in, given_nds, given_eds = AddOuts(given_masks_in, given_nds, given_eds, pmks)


    # run inference model
    with torch.no_grad():
        masks = model(z.to('cpu'), given_masks_in.to('cpu'), given_nds.to('cpu'), given_eds.to('cpu'))
        masks = masks.detach().cpu().numpy()
    return masks

def main():
    device = 'cpu'
    globalIndex = 0
    #for i, sample in enumerate(fp_loader):
    for i, (mks, nds) in enumerate(t := tqdm(fp_loader)):
        # draw real graph and groundtruth
        #mks, nds, eds, _, _ = sample
        z = torch.randn(1, 128).float().to(device)
        given_nds = nds.to(device)
        given_masks_in = mks[:, 0:1].to(device)
        '''
            Шум
            картинка (-1 - пустое пространство, 0 - наружа (только в моей реализации), 1 - объект )
            OHE тип комнаты или двери
            соседство комнат ( [Номер комнаты1, флаг, Номер комнаты 2]. -1 если не соседствуют, 1 - если да)

        '''

        # Генератор получает на вход картинки с учетом информации о блокировке, в формате [N, 2, :, :]
        masks = model(z, given_masks_in, given_nds)
        masks = masks.detach().cpu().numpy()

        real_mks = np.array(mks[0, 1:])
        im0 = draw_masks(masks.copy(), real_mks, 256, np.argmax(np.array(nds.cpu())[0], 1))
        im0 = torch.tensor(np.array(im0).transpose((2, 0, 1)))/255.0 
        save_image(im0, './{}/fp_init_{}.png'.format(opt.out, i), nrow=1, normalize=False) # visualize init image
        img = (1 - ((np.array(given_masks_in).reshape(64, 64, 1) / 2) + 0.5).astype(np.uint8)) * 255
        conturs = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
        cv2.imwrite('./{}/fp_init_{}_0rig.png'.format(opt.out, i), conturs)

        # generate per room type
        '''for _iter, _types in enumerate(selected_types):
            _fixed_nds = np.concatenate([np.where(real_nodes == _t)[0] for _t in _types]) \
                if len(_types) > 0 else np.array([]) 
            state = {'masks': masks, 'fixed_nodes': _fixed_nds}
            masks = _infer(graph, model, state)
            
        # save final floorplans
        imk = draw_masks(masks.copy(), real_nodes)
        imk = torch.tensor(np.array(imk).transpose((2, 0, 1)))/255.0 
        save_image(imk, './{}/fp_final_{}.png'.format(opt.out, i), nrow=1, normalize=False)
        '''
if __name__ == '__main__':
    main()