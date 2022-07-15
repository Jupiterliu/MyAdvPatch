import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from utils.median_pool import MedianPool2d
from utils.images import *
import random
import os
from PIL import Image
import numpy as np

patch_applier = PatchApplier()

patch_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test1_p200_lr01_k128_balance100-100_nobeta_nps001_tv25_nested3_scale01-37/patchs/20220714-180434_steer0.0_coll0.0_ep013.png"
patch_img = Image.open(patch_path)
adv_patch = transforms.ToTensor()(patch_img)

nested = 3
nested_size = 0.5

# Nested patch
if nested == 0:
    adv_p_ = adv_patch
elif nested == 1:
    pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size))/2
    adv_p1 = transforms.Resize((int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
        adv_patch)  # Defaults: the center (40,40) is the nested region
    mypad0 = nn.ConstantPad2d((int(pad_), int(pad_), int(pad_), int(pad_)), 0)
    adv_p1 = mypad0(adv_p1)
    adv_p_ = patch_applier(adv_patch, adv_p1)
elif nested == 2:
    pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
    pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
    adv_p1 = transforms.Resize((int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(adv_patch)   # Defaults: the center (40,40) is the nested region
    mypad0 = nn.ConstantPad2d((int(pad_), int(pad_), int(pad_), int(pad_)), 0)
    adv_p1 = mypad0(adv_p1)
    adv_p_ = patch_applier(adv_patch, adv_p1)
    adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size), int(adv_patch.size(-1) * nested_size * nested_size)))(adv_patch)
    mypad0 = nn.ConstantPad2d((int(pad__), int(pad__), int(pad__), int(pad__)), 0)
    adv_p2 = mypad0(adv_p2)
    adv_p_ = patch_applier(adv_p_, adv_p2)
elif nested == 3:
    pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
    pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
    pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size)) / 2
    adv_p1 = transforms.Resize(
        (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
        adv_patch)  # Defaults: the center (40,40) is the nested region
    mypad0 = nn.ConstantPad2d((int(pad_+0.5), int(pad_), int(pad_+0.5), int(pad_)), 0)
    adv_p1 = mypad0(adv_p1)
    adv_p_ = patch_applier(adv_patch, adv_p1)
    adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                int(adv_patch.size(-1) * nested_size * nested_size)))(
        adv_patch)
    mypad0 = nn.ConstantPad2d((int(pad__+0.5), int(pad__), int(pad__+0.5), int(pad__)), 0)
    adv_p2 = mypad0(adv_p2)
    adv_p_ = patch_applier(adv_p_, adv_p2)
    adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
        adv_patch)
    mypad0 = nn.ConstantPad2d((int(pad___+0.5), int(pad___), int(pad___+0.5), int(pad___)), 0)
    adv_p3 = mypad0(adv_p3)
    adv_p_ = patch_applier(adv_p_, adv_p3)
adv_p__ = transforms.Resize((100,100))(adv_p_)

img = transforms.ToPILImage('RGB')(adv_p_)
img_ = transforms.ToPILImage('RGB')(adv_p__)
plt.imshow(img)
img.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(7)))
img_.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(77)))
plt.show()