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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet import Darknet

from median_pool import MedianPool2d

# print('starting test read')
# im = Image.open('data/horse.jpg').convert('RGB')
# print('img read!')

class AttackLoss(nn.Module):
    """AttackLoss: Calculate the detection loss, targted and non-targeted.
    
    """
    def __init__(self):
        super(AttackLoss, self).__init__()
        self.beta = torch.Tensor([0]).float().cuda()
        
    def forward(self, k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target, is_targted):
        # Targeted:
        # Non-Targeted:
        if is_targted:
            attack_loss = self.targeted_attack_loss(k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target)
        else:
            attack_loss = self.untargeted_attack_loss(k, steer_true, steer_pred, coll_true, coll_pred)
        return attack_loss

    def targeted_attack_loss(self, k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target):
        # Steer angle
        # steer_target = torch.cuda.FloatTensor(torch.Size((steer_true.size(0), 1))).fill_(steer_target)
        target_steer = steer_true.clone()
        target_steer[:, 1] = steer_target
        loss1 = self.hard_mining_mse(k, steer_true, steer_pred)
        loss2 = self.hard_mining_mse(k, target_steer, steer_pred)
        # collision
        # coll_target = torch.cuda.FloatTensor(torch.Size((coll_true.size(0), 1))).fill_(coll_target)
        target_coll = steer_true.clone()
        target_coll[:, 1] = steer_target
        loss3 = self.hard_mining_entropy(k, coll_true, coll_pred)
        loss4 = self.hard_mining_entropy(k, target_coll, coll_pred)
        return torch.mean((-loss1 + loss2) + (-loss3 + loss4))

    def untargeted_attack_loss(self, k, steer_true, steer_pred, coll_true, coll_pred):
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

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_side):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_side),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score/torch.numel(adv_patch)

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


class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)


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
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, steer_true, coll_true, img_size, do_rotate=True, rand_loc=True):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(steer_true.size(0), steer_true.size(1), -1, -1, -1)
        batch_size = torch.Size((steer_true.size(0), steer_true.size(1)))

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

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # compress to min-max, not standardize

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids = (torch.cuda.FloatTensor(coll_true.size()).unsqueeze(-1)).fill_(0)
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
        anglesize = (steer_true.size(0) * steer_true.size(1))
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        # lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled = (torch.cuda.FloatTensor(coll_true.size()).fill_(0)).unsqueeze(-1)
        lab_batch_scaled = lab_batch_scaled.expand(-1, -1, 5)
        lab_batch_scaled[:, :, 1] = 0.861575186252594 * img_size
        lab_batch_scaled[:, :, 2] = 0.32314783334732056 * img_size
        lab_batch_scaled[:, :, 3] = 0.13357515633106232 * img_size
        lab_batch_scaled[:, :, 4] = 0.5612906217575073 * img_size
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
        target_x = torch.cuda.FloatTensor([0.561575186252594])
        target_y = torch.cuda.FloatTensor([0.52314783334732056])
        targetoff_x = torch.cuda.FloatTensor([0.13357515633106232])
        targetoff_y = torch.cuda.FloatTensor([0.5612906217575073])
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
            target_y = target_y + off_y
        target_y = target_y - 0.05
        scale = (torch.cuda.FloatTensor(coll_true.size()).fill_(0.7))  # target_size / current_patch_size
        scale = scale.view(anglesize)

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

        '''
        # Theta2 = translation matrix
        theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta2[:, 0, 0] = 1
        theta2[:, 0, 1] = 0
        theta2[:, 0, 2] = (-target_x + 0.5) * 2
        theta2[:, 1, 0] = 0
        theta2[:, 1, 1] = 1
        theta2[:, 1, 2] = (-target_y + 0.5) * 2

        grid2 = F.affine_grid(theta2, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch_t, grid2)
        msk_batch_t = F.grid_sample(msk_batch_t, grid2)
        '''
        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        # img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        # img = transforms.ToPILImage()(img)
        # img.show()
        # exit()

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

'''
class PatchGenerator(nn.Module):
    """PatchGenerator: network module that generates adversarial patches.

    Module representing the neural network that will generate adversarial patches.

    """

    def __init__(self, cfgfile, weightfile, img_dir, lab_dir):
        super(PatchGenerator, self).__init__()
        self.yolo = Darknet(cfgfile).load_weights(weightfile)
        self.dataloader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                                      batch_size=5,
                                                      shuffle=True)
        self.patchapplier = PatchApplier()
        self.nmscalculator = NMSCalculator()
        self.totalvariation = TotalVariation()

    def forward(self, *input):
        pass
'''

class InriaDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
        else:
            padded_lab = lab
        return padded_lab

if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()

    test_loader = torch.utils.data.DataLoader(InriaDataset(img_dir, lab_dir, shuffle=True),
                                              batch_size=3, shuffle=True)

    cfgfile = "cfg/yolov2.cfg"
    weightfile = "weights/yolo.weights"
    printfile = "non_printability/30values.txt"
    
    patch_size = 400

    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.cuda()
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()
    prob_extractor = MaxProbExtractor(0, 80).cuda()
    nms_calculator = NMSCalculator(printfile, patch_size)
    total_variation = TotalVariation()
    '''
    img = Image.open('data/horse.jpg').convert('RGB')
    img = img.resize((darknet_model.width, darknet_model.height))
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    img = torch.autograd.Variable(img)

    output = darknet_model(img)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    
    tl0 = time.time()
    tl1 = time.time()
    for i_batch, (img_batch, lab_batch) in enumerate(test_loader):
        tl1 = time.time()
        print('time to fetch items: ',tl1-tl0)
        img_batch = img_batch.cuda()
        lab_batch = lab_batch.cuda()
        adv_patch = Image.open('data/horse.jpg').convert('RGB')
        adv_patch = adv_patch.resize((patch_size, patch_size))
        transform = transforms.ToTensor()
        adv_patch = transform(adv_patch).cuda()
        img_size = img_batch.size(-1)
        print('transforming patches')
        t0 = time.time()
        adv_batch_t = patch_transformer.forward(adv_patch, lab_batch, img_size)
        print('applying patches')
        t1 = time.time()
        img_batch = patch_applier.forward(img_batch, adv_batch_t)
        img_batch = torch.autograd.Variable(img_batch)
        img_batch = F.interpolate(img_batch,(darknet_model.height, darknet_model.width))
        print('running patched images through model')
        t2 = time.time()

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    try:
                        print(type(obj), obj.size())
                    except:
                        pass
            except:
                pass

        print(torch.cuda.memory_allocated())

        output = darknet_model(img_batch)
        print('extracting max probs')
        t3 = time.time()
        max_prob = prob_extractor(output)
        t4 = time.time()
        nms = nms_calculator.forward(adv_patch)
        tv = total_variation(adv_patch)
        print('---------------------------------')
        print('        patch transformation : %f' % (t1-t0))
        print('           patch application : %f' % (t2-t1))
        print('             darknet forward : %f' % (t3-t2))
        print('      probability extraction : %f' % (t4-t3))
        print('---------------------------------')
        print('          total forward pass : %f' % (t4-t0))
        del img_batch, lab_batch, adv_patch, adv_batch_t, output, max_prob
        torch.cuda.empty_cache()
        tl0 = time.time()
