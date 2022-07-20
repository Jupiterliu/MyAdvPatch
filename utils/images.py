
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from utils.median_pool import MedianPool2d

import random
import os
from PIL import Image
import numpy as np
import math


# class PatchTransformer(nn.Module):
#     """PatchTransformer: transforms batch of patches
#
#     Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
#     contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
#     batch of labels, and pads them to the dimension of an image.
#
#     """
#
#     def __init__(self):
#         super(PatchTransformer, self).__init__()
#         self.min_contrast = 0.7  # 0.8
#         self.max_contrast = 1.3  # 1.2
#         self.min_brightness = -0.2  # -0.1
#         self.max_brightness = 0.2  # 0.1
#         self.min_scale = 0.1  # Scale the patch size from (patch_size * min_scale) to (patch_size * max_scale)
#         self.max_scale = 1.7
#         self.noise_factor = 0.1
#         self.minangle = 0#-10 / 180 * math.pi
#         self.maxangle = 0#10 / 180 * math.pi
#         self.medianpooler = MedianPool2d(7, same=True)
#
#     def forward(self, adv_patch, steer_true, img_size, do_rotate=True, rand_loc=True):
#         # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
#         adv_patch = transforms.Resize((100, 100))(adv_patch)
#         adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
#         # Determine size of padding
#         pad = (img_size - adv_patch.size(-1)) / 2
#         # Make a batch of patches
#         adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
#         adv_batch = adv_patch.expand(steer_true.size(0), 1, -1, -1, -1)
#         batch_size = torch.Size((steer_true.size(0), 1))
#         anglesize = steer_true.size(0)
#
#         # Contrast, brightness and noise transforms
#
#         # Create random contrast tensor
#         contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
#         contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
#         contrast = contrast.cuda()
#
#         # Create random brightness tensor
#         brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
#         brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
#         brightness = brightness.cuda()
#
#         # Create random scale tensor
#         scale = torch.cuda.FloatTensor(batch_size).uniform_(self.min_scale, self.max_scale)  # .fill_(0.85)
#         scale = scale.view(anglesize)
#         scale = scale.cuda()
#
#         # Create random noise tensor
#         noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
#
#         # Apply contrast/brightness/noise, clamp
#         adv_batch = adv_batch * contrast + brightness + noise
#
#         adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # compress to min-max, not standardize
#
#         # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
#         # cls_ids = torch.narrow(lab_batch, 2, 0, 1)
#         cls_ids = (torch.cuda.FloatTensor(batch_size).unsqueeze(-1)).fill_(0)
#         cls_mask = cls_ids.expand(-1, -1, 3)
#         cls_mask = cls_mask.unsqueeze(-1)
#         cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
#         cls_mask = cls_mask.unsqueeze(-1)
#         cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
#         msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask
#
#         # Pad patch and mask to image dimensions
#         mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
#         adv_batch = mypad(adv_batch)
#         msk_batch = mypad(msk_batch)
#
#         # Rotation and rescaling transforms
#         if do_rotate:
#             angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
#         else:
#             angle = torch.cuda.FloatTensor(anglesize).fill_(0)
#
#         # Resizes and rotates
#         target_x = torch.cuda.FloatTensor([0.5])
#         target_y = torch.cuda.FloatTensor([0.5])
#         targetoff_x = torch.cuda.FloatTensor([0.05])
#         targetoff_y = torch.cuda.FloatTensor([0.05])
#         if (rand_loc):
#             off_x = targetoff_x * (torch.cuda.FloatTensor(anglesize).uniform_(-4, 4))
#             target_x = target_x + off_x
#             off_y = targetoff_y * (torch.cuda.FloatTensor(anglesize).uniform_(-4, 4))
#             target_y = target_y + off_y
#
#         s = adv_batch.size()
#         adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
#         msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])
#
#         tx = (-target_x + 0.5) * 2
#         ty = (-target_y + 0.5) * 2
#         sin = torch.sin(angle)
#         cos = torch.cos(angle)
#
#         # Theta = rotation,rescale matrix
#         theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
#         theta[:, 0, 0] = cos / scale
#         theta[:, 0, 1] = sin / scale
#         theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
#         theta[:, 1, 0] = -sin / scale
#         theta[:, 1, 1] = cos / scale
#         theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale
#
#         b_sh = adv_batch.shape
#         grid = F.affine_grid(theta, adv_batch.shape)
#
#         adv_batch_t = F.grid_sample(adv_batch, grid)
#         msk_batch_t = F.grid_sample(msk_batch, grid)
#
#         adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
#         msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])
#
#         adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
#
#         return adv_batch_t * msk_batch_t


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.7  # 0.8
        self.max_contrast = 1.3  # 1.2
        self.min_brightness = -0.2  # -0.1
        self.max_brightness = 0.2  # 0.1
        # self.min_scale = 1 # Scale the patch size from (patch_size * min_scale) to (patch_size * max_scale)
        # self.max_scale = 3.6
        self.noise_factor = 0.1
        self.min_angle = -8  #-10 / 180 * math.pi
        self.max_angle = 8  #10 / 180 * math.pi
        # self.nested_size = 0.5
        self.min_distortion = 0.
        self.max_distortion = 0.2
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, steer_true, img_size, patch_size, patch_name, steer_target,
                do_rotate=True, do_pespective=True, nested=1, nested_size=0.5, centre=False,
                location="random", min_scale=1, max_scale=3.6):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = transforms.Resize((int(patch_size * 0.5), int(patch_size * 0.5)))(adv_patch)
        # adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(steer_true.size(0), -1, -1, -1)
        batch_size = torch.Size((steer_true.size(0), 1))

        patch_applier = PatchApplier().cuda()
        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)  # compress to min-max, not standardize

        if do_rotate:
            angle = torch.cuda.FloatTensor(steer_true.size(0)).uniform_(self.min_angle, self.max_angle)
        else:
            angle = torch.cuda.FloatTensor(steer_true.size(0)).fill_(0)
        if do_pespective:
            distortion = torch.cuda.FloatTensor(steer_true.size(0)).uniform_(self.min_distortion, self.max_distortion)
            p = torch.cuda.FloatTensor(steer_true.size(0)).uniform_(0.5, 1)
        else:
            distortion = torch.cuda.FloatTensor(steer_true.size(0)).fill_(0)
            p = torch.cuda.FloatTensor(steer_true.size(0)).fill_(0)
        scale = torch.cuda.FloatTensor(steer_true.size(0)).uniform_(min_scale, max_scale)

        for i in range(adv_batch.size(0)):

            # Nested patch
            adv_p_ = patch_nest(adv_patch, adv_batch[i, :, :, :], nested, nested_size, patch_name, steer_target,
                                patch_applier, centre=centre)

            # Scale
            adv_b_scaled = transforms.Resize((int(adv_batch.size(-1) * scale[i]), int(adv_batch.size(-1) * scale[i])))(adv_p_)

            # Perspective
            perspectives = transforms.RandomPerspective(distortion_scale=distortion[i], p=p[i].cpu(), fill=0)
            adv_b_perspective = perspectives(adv_b_scaled)
            # adv_b_perspective = ad_perspective(adv_b_scaled, img_size)

            # Rotation
            rotations = transforms.RandomRotation((angle[i], angle[i]), expand=True, fill=0, interpolation=Image.BILINEAR)  #Image.BILINEAR or 2
            adv_b_rotation = rotations(adv_b_perspective)

            # Location: random, centre, corner
            adv_b_pad = patch_pad(location, adv_b_rotation, img_size, patch_name, steer_target)
            # adv_b_pad = adv_b_rotation

            if i == 0:
                adv_batch_t = adv_b_pad.unsqueeze(0)
            else:
                adv_batch_t = torch.cat((adv_batch_t, adv_b_pad.unsqueeze(0)), 0)
        adv_batch_t = torch.clamp(adv_batch_t, 0., 0.9999999999)

        return adv_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()
    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch.unsqueeze(1), 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch


def patch_nest(adv_patch, adv_batch, nested, nested_size, patch_name, steer_target, patch_applier, centre=False):

    if nested == 0:
        adv_p_ = adv_batch
    elif nested == 1:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
        adv_p1 = transforms.Resize((int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad0 = nn.ConstantPad2d((int(pad_), int(pad_), int(pad_), int(pad_)), 0)
        adv_p1 = mypad0(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
    elif nested == 2:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
        adv_p1 = transforms.Resize((int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad0 = nn.ConstantPad2d((int(pad_), int(pad_), int(pad_), int(pad_)), 0)
        adv_p1 = mypad0(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size * nested_size), int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad0 = nn.ConstantPad2d((int(pad__), int(pad__), int(pad__), int(pad__)), 0)
        adv_p2 = mypad0(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
    elif nested == 3:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
        pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size)) / 2
        adv_p1 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad0 = nn.ConstantPad2d((int(pad_ + 0.5), int(pad_), int(pad_ + 0.5), int(pad_)), 0)
        adv_p1 = mypad0(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad0 = nn.ConstantPad2d((int(pad__ + 0.5), int(pad__), int(pad__ + 0.5), int(pad__)), 0)
        adv_p2 = mypad0(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
        adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad0 = nn.ConstantPad2d((int(pad___ + 0.5), int(pad___), int(pad___ + 0.5), int(pad___)), 0)
        adv_p3 = mypad0(adv_p3)
        adv_p_ = patch_applier(adv_p_, adv_p3)
    elif nested == 4 and centre:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
        pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size)) / 2
        pad____ = (adv_patch.size(-1) - int(
            adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)) / 2
        adv_p1 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad1 = nn.ConstantPad2d((int(pad_ + 0.5), int(pad_), int(pad_ + 0.5), int(pad_)), 0)
        adv_p1 = mypad1(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad2 = nn.ConstantPad2d((int(pad__ + 0.5), int(pad__), int(pad__ + 0.5), int(pad__)), 0)
        adv_p2 = mypad2(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
        adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad3 = nn.ConstantPad2d((int(pad___ + 0.5), int(pad___), int(pad___ + 0.5), int(pad___)), 0)
        adv_p3 = mypad3(adv_p3)
        adv_p_ = patch_applier(adv_p_, adv_p3)
        adv_p4 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size),
             int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad4 = nn.ConstantPad2d((int(pad____ + 0.5), int(pad____), int(pad____ + 0.5), int(pad____)), 0)
        adv_p4 = mypad4(adv_p4)
        adv_p_ = patch_applier(adv_p_, adv_p4)
    elif nested == 4 and (patch_name == "HA" or patch_name == "OA"):
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size)) / 2
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size)) / 2
        pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size)) / 2
        pad____ = (adv_patch.size(-1) - int(
            adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)) / 2
        adv_p1 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad1 = nn.ConstantPad2d((int(pad_ + 0.5), int(pad_), int(0), int(pad_ * 2)), 0)
        adv_p1 = mypad1(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad2 = nn.ConstantPad2d((int(pad__ + 0.5), int(pad__), int(0), int(pad__ * 2)), 0)
        adv_p2 = mypad2(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
        adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad3 = nn.ConstantPad2d((int(pad___ + 0.5), int(pad___), int(0), int(pad___ * 2)), 0)
        adv_p3 = mypad3(adv_p3)
        adv_p_ = patch_applier(adv_p_, adv_p3)
        adv_p4 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size),
             int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad4 = nn.ConstantPad2d((int(pad____ + 0.5), int(pad____), int(0), int(pad____ * 2)), 0)
        adv_p4 = mypad4(adv_p4)
        adv_p_ = patch_applier(adv_p_, adv_p4)
    elif nested == 4 and patch_name == "YA" and steer_target > 0:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size))
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size))
        pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size))
        pad____ = (adv_patch.size(-1) - int(
            adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size))
        adv_p1 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad1 = nn.ConstantPad2d((int(0), int(pad_ + 0.5), int(0), int(pad_ + 0.5)), 0)
        adv_p1 = mypad1(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad2 = nn.ConstantPad2d((int(0), int(pad__ + 0.5), int(0), int(pad__ + 0.5)), 0)
        adv_p2 = mypad2(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
        adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad3 = nn.ConstantPad2d((int(0), int(pad___ + 0.5), int(0), int(pad___ + 0.5)), 0)
        adv_p3 = mypad3(adv_p3)
        adv_p_ = patch_applier(adv_p_, adv_p3)
        adv_p4 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size),
             int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad4 = nn.ConstantPad2d((int(0), int(pad____ + 0.5), int(0), int(pad____ + 0.5)), 0)
        adv_p4 = mypad4(adv_p4)
        adv_p_ = patch_applier(adv_p_, adv_p4)
    elif nested == 4 and patch_name == "YA" and steer_target < 0:
        pad_ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size))
        pad__ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size))
        pad___ = (adv_patch.size(-1) - int(adv_patch.size(-1) * nested_size * nested_size * nested_size))
        pad____ = (adv_patch.size(-1) - int(
            adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size))
        adv_p1 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size), int(adv_patch.size(-1) * nested_size)))(
            adv_batch)  # Defaults: the center (40,40) is the nested region
        mypad1 = nn.ConstantPad2d((int(pad_ + 0.5), int(0), int(0), int(pad_ + 0.5)), 0)
        adv_p1 = mypad1(adv_p1)
        adv_p_ = patch_applier(adv_batch, adv_p1)
        adv_p2 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size)))(
            adv_batch)
        mypad2 = nn.ConstantPad2d((int(pad__ + 0.5), int(0), int(0), int(pad__ + 0.5)), 0)
        adv_p2 = mypad2(adv_p2)
        adv_p_ = patch_applier(adv_p_, adv_p2)
        adv_p3 = transforms.Resize((int(adv_patch.size(-1) * nested_size * nested_size * nested_size),
                                    int(adv_patch.size(-1) * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad3 = nn.ConstantPad2d((int(pad___ + 0.5), int(0), int(0), int(pad___ + 0.5)), 0)
        adv_p3 = mypad3(adv_p3)
        adv_p_ = patch_applier(adv_p_, adv_p3)
        adv_p4 = transforms.Resize(
            (int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size),
             int(adv_patch.size(-1) * nested_size * nested_size * nested_size * nested_size)))(
            adv_batch)
        mypad4 = nn.ConstantPad2d((int(pad____ + 0.5), int(0), int(0), int(pad____ + 0.5)), 0)
        adv_p4 = mypad4(adv_p4)
        adv_p_ = patch_applier(adv_p_, adv_p4)

    return adv_p_

def patch_pad(location, adv_b_rotation, img_size, patch_name, steer_target):
    length = adv_b_rotation.size(-1)
    if length > img_size:
        top = 0
        if patch_name == "HA" or patch_name == "OA":
            left = int((adv_b_rotation.size(-1) / 2) - 100)
        elif patch_name == "YA" and steer_target > 0:
            left = 0
        else:  # elif patch_name == "YA" and steer_target < 0:
            left = int(adv_b_rotation.size(-1) - 200)
        adv_b_pad = TF.crop(adv_b_rotation, top, left, 200, 200)
    else:
        if location == "random":
            pad_left = torch.cuda.FloatTensor(1).uniform_(0, img_size - length).int()
            pad_right = img_size - length - pad_left
            pad_top = torch.cuda.FloatTensor(1).uniform_(0, img_size - length).int()
            pad_bottom = img_size - length - pad_top
        elif location == "centre":
            pad_left = torch.cuda.FloatTensor(1).fill_((img_size - length) / 2).int()
            pad_right = img_size - length - pad_left
            pad_top = torch.cuda.FloatTensor(1).fill_((img_size - length) / 2).int()
            pad_bottom = img_size - length - pad_top
        else:  # elif location == "corner":
            pad_left = torch.cuda.FloatTensor(1).fill_(0).int()
            pad_right = img_size - length - pad_left
            pad_top = torch.cuda.FloatTensor(1).fill_(0).int()
            pad_bottom = img_size - length - pad_top
        mypad1 = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
        adv_b_pad = mypad1(adv_b_rotation)
    return  adv_b_pad


def ad_perspective(adv_b_scaled, img_size):
    bottom = adv_b_scaled.size(-1)
    height = int(bottom / 3)
    left_top = int(height * 2 / 3)
    right_top = bottom - left_top
    startpoint = [[0, bottom], [bottom, bottom], [0, 0], [bottom, 0]]
    endpoint = [[0, bottom], [bottom, bottom], [left_top, bottom - height], [right_top, bottom - height]]
    adv_p = TF.perspective(adv_b_scaled, startpoint, endpoint)
    adv_p_ = TF.crop(adv_p, bottom - height, 0, height, bottom)
    # Pad to img_size
    pad_left = torch.cuda.FloatTensor(1).uniform_(0, img_size - bottom).int()
    pad_right = img_size - bottom - pad_left
    pad_top = torch.cuda.FloatTensor(1).uniform_(0, img_size - height).int()
    pad_bottom = img_size - height - pad_top
    mypad0 = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)
    adv_b_perspective = mypad0(adv_p_)

    return adv_b_perspective

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if (random.randint(1, 10000) % 2):
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    flip = random.randint(1, 10000) % 2
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes, 5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace(
        '.png', '.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img, flip, dx, dy, sx, sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1. / sx, 1. / sy)
    return img, label

