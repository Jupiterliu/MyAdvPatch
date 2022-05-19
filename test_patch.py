"""
Testing code for Physical Adversarial patch Attack
"""

import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from load_data import PatchTransformer, PatchApplier
import json
from DroNet_Pytorch.test import *
from DroNet_Pytorch.load_datasets import DronetDataset

if __name__ == '__main__':
    image_mode = "rgb"
    weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet_Pytorch/saved_models/test1_RGB_old_loss_200_nice/weights_199.pth"
    dronet = getModel((200, 200), image_mode, 1, weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing',
                                    image_mode ,augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16,
                                                     shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet_Pytorch/saved_models/test1_RGB_old_loss_200_nice"
    eval_path = "patch_test4_26"
    folder  = os.path.exists(os.path.join(test_path, eval_path))
    if not folder:
        os.makedirs(os.path.join(test_path, eval_path))

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/DroNet_patch/test4_old_loss_beta/20220519-122530_steer-0.0_coll-0.0_26.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True

    with torch.no_grad():
        testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch)
