from utils.images import *
import os
from PIL import Image
import numpy as np

patch_applier = PatchApplier()

patch_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test19_p400_lr005_balance10-10_beta10_nps001_tv25_nest5_scale02-35/patchs/20220727-221847_steer-0.5_coll0.0_ep066.png"
patch_img = Image.open(patch_path)
adv_patch = transforms.ToTensor()(patch_img)

nested = 5
nested_size = 0.5
patch_name = "YA"
steer_target = -0.5
centre = False

adv_p_ = patch_nest(adv_patch, adv_patch, nested, nested_size, patch_name, steer_target, patch_applier, centre=centre)

img = transforms.ToPILImage('RGB')(adv_p_)
# img_ = transforms.ToPILImage('RGB')(adv_p__)
plt.imshow(img)
# plt.imshow(img_)
img.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(patch_name)))
# img_.save(os.path.join("/root/Python_Program_Remote/MyAdvPatch/saved_patch/nested_patch", "{}.png".format(22)))
plt.show()