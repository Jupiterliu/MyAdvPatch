"""
Training code for Physical Adversarial patch Attack
"""

import PIL
import torch

import load_data
from tqdm import tqdm

import utils
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

# import DroNet_Pytorch.dronet_torch
from DroNet_Pytorch.dronet_torch import getModel
from DroNet_Pytorch.load_datasets import DronetDataset

class PatchTrainer(object):
    def __init__(self, mode):
        # Load config from args
        self.config = patch_config.patch_configs[mode]()

        # Load DroNet model from .pth & eval()
        weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet_Pytorch/saved_models/test1_RGB_old_loss_200_nice/weights_199.pth"
        self.dronet_model = getModel((200, 200), self.config.image_mode, 1, weights_path)
        self.dronet_model = self.dronet_model.eval().cuda()

        # Load patch Projection
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()

        # Load Methods to calcu Loss: AttackLoss, NPS Loss, TV Loss
        self.attack_loss = Attack_Loss().cuda()
        self.nps_loss = NPS_Loss(self.config.printfile, self.config.patch_size).cuda()
        self.tv_loss = TV_Loss().cuda()
        # self.pnorm_loss = NPS_Loss().cuda()

        # Load the recording Tensorboard
        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=tensorboard_runs', '--host=0.0.0.0'])  # Link tensorboard to Loss(det, nps, tv, total) and misc(epoch, lr)
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S") #, time.localtime())
            return SummaryWriter(f'tensorboard_runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate starting point: patch(gray or random)
        adv_patch_cpu = self.generate_patch("gray")  # gray or random
        adv_patch_cpu.requires_grad_(True)

        # Load my data from collision testing
        training_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'training',
                                            self.config.image_mode, augmentation=False)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=self.config.batch_size,
                                                            shuffle=True, num_workers=self.config.num_workers)
        self.epoch_length = len(training_dataloader)
        # print(f'One epoch is {len(training_dataloader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)  # Improving the performance by reducing the learning rate

        et0 = time.time()
        for epoch in range(self.config.n_epochs):
            ep_attack_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            other_val = (1 - torch.exp(torch.Tensor([-1 * (0.1) * (epoch - 10)]))).float().cuda()
            beta = torch.max(torch.Tensor([0]).float().cuda(), other_val)
            for i_batch, (img_batch, steer_true, coll_true) in tqdm(enumerate(training_dataloader),
                                                                    desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    steer_true = steer_true.cuda().squeeze(1)
                    coll_true = coll_true.cuda().squeeze(1)
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()

                    # patch projection
                    # if self.config.image_mode == "gray":
                    #     adv_patch = transforms.Grayscale()(adv_patch)
                    adv_batch_t = self.patch_transformer(adv_patch, steer_true, self.config.image_size, do_rotate=True, rand_loc=True)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample

                    for i in range(p_img_batch.size(0)):
                        Tensor = p_img_batch[i, :, :, :]
                        image = np.transpose(Tensor.detach().cpu().numpy(), (1, 2, 0))
                        plt.imshow(image)
                        plt.show()

                    # Prediction
                    steer_pred, coll_pred = self.dronet_model(p_img_batch)

                    # Cal 3 Losses from true and pred
                    attack_loss = self.attack_loss(self.config.k, steer_true, steer_pred, coll_true, coll_pred,
                                                    self.config.steer_target, self.config.coll_target,
                                                    self.config.is_targeted, self.config.use_old_loss,
                                                    beta)
                    nps = self.nps_loss(adv_patch)
                    tv = self.tv_loss(adv_patch)
                    # pnorm = self.pnorm_loss(adv_patch, p_img_batch)

                    # From the Loss weights to cal the total Loss
                    nps_loss = nps * 0.1     # 0.01
                    tv_loss = tv * 3.5    # 2.5
                    loss = attack_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())  # + nps_loss

                    ep_attack_loss += attack_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    # Optimizing adv_patch_cpu
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/attack_loss', attack_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(training_dataloader):
                        print('\n')
                    else:
                        del adv_batch_t, steer_pred, coll_pred, attack_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_attack_loss = ep_attack_loss/len(training_dataloader)
            ep_nps_loss = ep_nps_loss/len(training_dataloader)
            ep_tv_loss = ep_tv_loss/len(training_dataloader)
            ep_loss = ep_loss/len(training_dataloader)

            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            plt.imshow(im)
            im.save(f'DroNet_patch/test4_old_loss_beta/{time_str}_steer-{self.config.steer_target}_coll-{self.config.coll_target}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('   EPOCH NR: ', epoch),
                print(' EPOCH LOSS: ', ep_loss)
                print('ATTACK LOSS: ', ep_attack_loss)
                print('   NPS LOSS: ', ep_nps_loss)
                print('    TV LOSS: ', ep_tv_loss)
                print(' EPOCH TIME: ', et1-et0)
                del adv_batch_t, steer_pred, coll_pred, attack_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


