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
from DroNet.dronet_evaluate import *
from DroNet.dronet_load_datasets import DronetDataset
from plot_result import *

if __name__ == '__main__':
    image_mode = "rgb"
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test3_weights_484.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode ,augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB"
    eval_path = "patch_test6_33"
    folder  = os.path.exists(os.path.join(test_path, eval_path))
    if not folder:
        os.makedirs(os.path.join(test_path, eval_path))

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test6_old_loss_beta10/20220520-122725_steer-0.0_coll-0.0_33.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True

    with torch.no_grad():
        testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch)

    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
    with open(fname_steer, 'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                                os.path.join(test_path, eval_path, "histograms.png"))

    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
    with open(fname_labels, 'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                            ['no collision', 'collision'],
                            img_name=os.path.join(test_path, eval_path, "confusion.png"))
