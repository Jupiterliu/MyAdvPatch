import fnmatch
import math
import os
import sys
import time
from operator import itemgetter

import gc
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from median_pool import MedianPool2d

from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset
from plot_result import *


class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.min_contrast = 0.7  # 0.8
        self.max_contrast = 1.3  # 1.2
        self.min_brightness = -0.2  # -0.1
        self.max_brightness = 0.2  # 0.1
        self.min_scale = 0.15  # Scale the patch size from (patch_size * min_scale) to (patch_size * max_scale)
        self.max_scale = 0.8
        self.noise_factor = 0.10
        self.minangle = -10 / 180 * math.pi
        self.maxangle = 10 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, steer_true, img_size, do_rotate=True, rand_loc=True):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = transforms.Resize((100, 100))(adv_patch)
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(steer_true.size(0), 1, -1, -1, -1)
        batch_size = torch.Size((steer_true.size(0), 1))
        anglesize = steer_true.size(0)

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random scale tensor
        scale = torch.cuda.FloatTensor(batch_size).fill_(0.5)  #.uniform_(self.min_scale, self.max_scale)  # .fill_(0.85)
        scale = scale.view(anglesize)
        scale = scale.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # compress to min-max, not standardize

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids = (torch.cuda.FloatTensor(batch_size).unsqueeze(-1)).fill_(0)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        target_x = torch.cuda.FloatTensor([0.5])
        target_y = torch.cuda.FloatTensor([0.5])
        targetoff_x = torch.cuda.FloatTensor([0.])
        targetoff_y = torch.cuda.FloatTensor([0.])
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y
        # target_y = target_y - 0.05

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t

class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()
    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch

def testModel(model, testing_dataloader, adv_patch):
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    # go through all values
    all_true_steer = torch.cuda.FloatTensor()
    all_true_coll = torch.cuda.FloatTensor()
    all_pred_steer = torch.cuda.FloatTensor()
    all_pred_coll = torch.cuda.FloatTensor()
    all_exp_type = torch.cuda.FloatTensor()

    for idx, (img, steer_true, coll_true) in tqdm(enumerate(testing_dataloader), desc=f'Running epoch {0}', total=len(testing_dataloader)):
        img_cuda = img.cuda()
        true_steer = steer_true.squeeze(1).cuda()[:, 1]
        true_coll = coll_true.squeeze(1).cuda()[:, 1]
        exp_type = steer_true.squeeze(1).cuda()[:, 0]

        # patch projection
        adv_batch_t = patch_transformer(adv_patch, steer_true, 200, do_rotate=True, rand_loc=False)
        p_img_batch = patch_applier(img_cuda, adv_batch_t)
        img_cuda = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample

        steer_pred, coll_pred = model(img_cuda)
        pred_steer = steer_pred.squeeze(-1)
        pred_coll = coll_pred.squeeze(-1)
        # print(f'Dronet Ground Truth {steer_true.item()} angle and {coll_true.item()} collision.')
        # print(f'Dronet predicts {steer_pred.item()} angle and {coll_pred.item()} collision.\n')

        # All true and pred data:
        all_true_steer = torch.cat((all_true_steer, true_steer))
        all_true_coll = torch.cat((all_true_coll, true_coll))
        all_pred_steer = torch.cat((all_pred_steer, pred_steer))
        all_pred_coll = torch.cat((all_pred_coll, pred_coll))
        all_exp_type = torch.cat((all_exp_type, exp_type))

    # Param t. t=1 steering, t=0 collision
    t_mask = all_exp_type == 1

    # ************************* Steering evaluation ***************************
    # Predicted and real steerings
    pred_steerings = all_pred_steer[t_mask,].cpu().numpy()
    real_steerings = all_true_steer[t_mask,].cpu().numpy()

    # Evaluate predictions: EVA, residuals, and highest errors
    eva, rmse = evaluate_regression(pred_steerings, real_steerings)

    # *********************** Collision evaluation ****************************
    # Predicted probabilities and real labels
    pred_collisions = all_pred_coll[~t_mask,].cpu().numpy()
    pred_labels = np.zeros_like(pred_collisions)
    pred_labels[pred_collisions >= 0.5] = 1
    real_labels = all_true_coll[~t_mask, ].cpu().numpy()

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    ave_accuracy, precision, recall, f_score = evaluate_classification(pred_collisions, pred_labels, real_labels)

    return eva, rmse, ave_accuracy, precision, recall, f_score

def evaluate_regression(predictions, real_values):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    # highest_errors = compute_highest_regression_errors(predictions, real_values, n_errors=20)
    # dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(), "highest_errors": highest_errors.tolist()}
    return evas, rmse

def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse

def evaluate_classification(pred_prob, pred_labels, real_labels):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.precision_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    # highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
    #         n_errors=20)
    # dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
    #               "recall": recall.tolist(), "f_score": f_score.tolist(),
    #               "highest_errors": highest_errors.tolist()}
    # write_to_file(dictionary, fname)
    return ave_accuracy, precision, recall, f_score


if __name__ == '__main__':
    image_mode = "rgb"
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test3_weights_484.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode,
                                    augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle=True, num_workers=10)

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test6_old_loss_beta10/20220520-122725_steer-0.0_coll-0.0_33.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    with torch.no_grad():
        eva, rmse, ave_accuracy, precision, recall, f_score = testModel(dronet, testing_dataloader, adv_patch)
        print(eva, rmse, ave_accuracy, precision, recall, f_score)