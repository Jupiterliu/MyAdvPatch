
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attack_Loss(nn.Module):
    """AttackLoss: Calculate the detection loss, targted and non-targeted.

    """

    def __init__(self):
        super(Attack_Loss, self).__init__()
        self.beta = torch.Tensor([0]).float().cuda()

    def forward(self, k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target, is_targted,
                use_old_loss, beta):
        # Targeted:
        # Non-Targeted:
        if is_targted:
            attack_loss = self.targeted_attack_loss(k, steer_true, steer_pred, coll_true, coll_pred, steer_target,
                                                    coll_target, use_old_loss, beta)
        else:
            attack_loss = self.untargeted_attack_loss(k, steer_true, steer_pred, coll_true, coll_pred, use_old_loss, beta)
        return attack_loss

    def targeted_attack_loss(self, k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target,
                             use_old_loss, beta):
        # Steer angle: steer_target = torch.cuda.FloatTensor(torch.Size((steer_true.size(0), 1))).fill_(steer_target)
        balance_steer = 100
        balance_coll = 100
        target_steer = steer_true.clone()
        target_steer[:, 1] = steer_target
        target_coll = steer_true.clone()
        target_coll[:, 1] = coll_target
        if use_old_loss:
            loss1 = self.old_hard_mining_mse(k, steer_true, steer_pred)
            loss2 = self.old_hard_mining_mse(k, target_steer, steer_pred)
            # collision: coll_target = torch.cuda.FloatTensor(torch.Size((coll_true.size(0), 1))).fill_(coll_target)
            loss3 = self.old_hard_mining_entropy(k, coll_true, coll_pred)
            loss4 = self.old_hard_mining_entropy(k, target_coll, coll_pred)
            # print("loss1: ", loss1.item(), "loss2: ", loss2.item(), "loss3: ", loss3.item(), "loss4: ", loss4.item())
            return torch.mean( (balance_steer * loss2) + beta * (balance_coll * loss4))
        else:
            loss1 = self.hard_mining_mse(k, steer_true, steer_pred)
            loss2 = self.hard_mining_mse(k, target_steer, steer_pred)
            # collision: coll_target = torch.cuda.FloatTensor(torch.Size((coll_true.size(0), 1))).fill_(coll_target)
            loss3 = self.hard_mining_entropy(k, coll_true, coll_pred)
            loss4 = self.hard_mining_entropy(k, target_coll, coll_pred)
            return torch.mean((-loss1 + balance_steer * loss2) + beta * (-loss3 + balance_coll * loss4))

    def untargeted_attack_loss(self, k, steer_true, steer_pred, coll_true, coll_pred, use_old_loss, beta):
        # for steering angle
        mse_loss = self.hard_mining_mse(k, steer_true, steer_pred)
        # for collision probability
        bce_loss = self.beta * (self.hard_mining_entropy(k, coll_true, coll_pred))
        return -(mse_loss + bce_loss)  # Or 1 / (mse_loss + bce_loss)

    def hard_mining_mse(self, k, y_true, y_pred):
        t = y_true[:, 0]
        n_sample_steer = 0
        for i in t:
            if i == 1.0:
                n_sample_steer = n_sample_steer + 1
        if n_sample_steer == 0:
            return 0.0
        else:
            true_steer = y_true[:, 1]
            pred_steer = y_pred.squeeze(-1)
            loss_steer = torch.mul(((true_steer - pred_steer) ** 2), t)
            k_min = min(k, n_sample_steer)
            _, indices = torch.topk(loss_steer, k=k_min, dim=0)
            max_loss_steer = torch.gather(loss_steer, dim=0, index=indices)
            # mean square error
            hard_loss_steer = torch.div(torch.sum(max_loss_steer), k_min)
            return hard_loss_steer

    def hard_mining_entropy(self, k, y_true, y_pred):
        t = y_true[:, 0]
        n_sample_coll = 0
        for i in t:
            if i == 0.0:
                n_sample_coll = n_sample_coll + 1
        if n_sample_coll == 0:
            return 0.0
        else:
            true_coll = y_true[:, 1]
            pred_coll = y_pred.squeeze(-1)
            loss_coll = torch.mul(F.binary_cross_entropy(pred_coll, true_coll, reduction='none'), 1 - t)
            k_min = min(k, n_sample_coll)
            _, indices = torch.topk(loss_coll, k=k_min, dim=0)
            max_loss_coll = torch.gather(loss_coll, dim=0, index=indices)
            hard_loss_coll = torch.div(torch.sum(max_loss_coll), k_min)
            return hard_loss_coll

    def old_hard_mining_mse(self, k, y_true, y_pred):
        y_true = y_true[:, 1].unsqueeze(-1)
        loss_steer = (y_true - y_pred) ** 2
        # hard mining
        # get value of k that is minimum of batch size or the selected value of k
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_steer, k=k_min, dim=0)
        max_loss_steer = torch.gather(loss_steer, dim=0, index=indices)
        # mean square error
        hard_loss_steer = torch.div(torch.sum(max_loss_steer), k_min)
        return hard_loss_steer

    def old_hard_mining_entropy(self, k, y_true, y_pred):
        y_true = y_true[:, 1].unsqueeze(-1)
        loss_coll = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        k_min = min(k, y_true.shape[0])
        _, indices = torch.topk(loss_coll, k=k_min, dim=0)
        max_loss_coll = torch.gather(loss_coll, dim=0, index=indices)
        hard_loss_coll = torch.div(torch.sum(max_loss_coll), k_min)
        return hard_loss_coll


class PNorm_Loss(nn.Module):
    """

    """

    def __init__(self):
        super(PNorm_Loss, self).__init__()

    def forward(self, adv_patch, patched_img):
        pass


class NPS_Loss(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPS_Loss, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),
                                               requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)

        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, printability_file, side):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((side, side), red))
            printability_imgs.append(np.full((side, side), green))
            printability_imgs.append(np.full((side, side), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        return pa


class TV_Loss(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch

        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2

        return tv / torch.numel(adv_patch)