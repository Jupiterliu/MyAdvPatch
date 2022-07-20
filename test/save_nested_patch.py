from utils.images import *
import os
from PIL import Image
import numpy as np

patch_applier = PatchApplier()

patch_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test5_p400_lr01_balance100-100_beta1_nps001_tv25_nested4-05_scale01-20/patchs/20220715-175853_steer0.0_coll0.0_ep040.png"
patch_img = Image.open(patch_path)
adv_patch = transforms.ToTensor()(patch_img)

nested = 4
nested_size = 0.5
patch_name = "HA"
steer_target = 0.5
centre = True

adv_p_ = patch_nest(adv_patch, adv_patch, nested, nested_size, patch_name, steer_target, patch_applier, centre=centre)

img = transforms.ToPILImage('RGB')(adv_p_)
# img_ = transforms.ToPILImage('RGB')(adv_p__)
plt.imshow(img)
# plt.imshow(img_)
img.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(patch_name)))
# img_.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(22)))
plt.show()