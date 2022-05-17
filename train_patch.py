"""
Training code for Adversarial patch training
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
from DroNet_Pytorch.dronet_torch_train import getModel
from DroNet_Pytorch.load_datasets import DronetDataset

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        # Load DroNet model from .pth & eval()
        weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet_Pytorch/saved_models/test1_RGB/weights_199.pth"
        self.dronet_model = getModel((200, 200), 3, 1, weights_path)
        self.dronet_model = self.dronet_model.eval().cuda()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.my_pred_loss = AttackLoss().cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

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

        # img_size = self.darknet_model.height
        img_size = 200
        # batch_size = self.config.batch_size
        batch_size = 16
        num_workers = 10
        n_epochs = 100

        is_targted = True  # or False
        steer_target = 0.
        coll_target = 0.
        k = 8  # hard-mining

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate starting point
        adv_patch_cpu = self.generate_patch("gray")
        adv_patch_cpu.requires_grad_(True)

        # Load my data from collision testing
        training_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets/collision/collision_dataset', 'training', augmentation=False)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.epoch_length = len(training_dataloader)
        print(f'One epoch is {len(training_dataloader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)  # Improving the performance by reducing the learning rate

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_pred_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, steer_true, coll_true) in tqdm(enumerate(training_dataloader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():  # Anomaly Detetion for Propagation and BackPropagation
                    img_batch = img_batch.cuda()
                    steer_true = steer_true.cuda().squeeze(1)
                    coll_true = coll_true.cuda().squeeze(1)
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, steer_true, coll_true, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    # p_img_batch = p_img_batch.swapaxes(1,-1).swapaxes(1,2).cpu().detach().numpy()
                    # p_img_batch_gray = transforms.Grayscale()(p_img_batch).squeeze(1).unsqueeze(-1).cpu().detach().numpy()
                    # p_img_batch_gray = transforms.Grayscale()(p_img_batch)
                    p_img_batch = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample

                    steer_pred, coll_pred = self.dronet_model(p_img_batch)
                    # pred_loss = self.dronet_model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
                    pred_loss = self.my_pred_loss(k, steer_true, steer_pred, coll_true, coll_pred, steer_target, coll_target, is_targted)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    nps_loss = nps*0.1  # 0.01
                    tv_loss = tv*5  # 2.5
                    loss = pred_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_pred_loss += pred_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/pred_loss', pred_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(training_dataloader):
                        print('\n')
                    else:
                        del adv_batch_t, steer_pred, coll_pred, pred_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_pred_loss = ep_pred_loss/len(training_dataloader)
            ep_nps_loss = ep_nps_loss/len(training_dataloader)
            ep_tv_loss = ep_tv_loss/len(training_dataloader)
            ep_loss = ep_loss/len(training_dataloader)

            im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            plt.imshow(im)
            # im.save(f'DroNet_result_patch/{time_str}_{self.config.patch_name}_{epoch}.png')
            im.save(f'DroNet_patch/test2_random_scale/{time_str}_steer-{steer_target}_coll-{coll_target}_{epoch}.png')
            # im.save(f'DroNet_result_patch/{time_str}_{is_targted}_{epoch}.png')
            # utils.save_images_from_Tesnor(adv_patch_cpu, epoch)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_pred_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                #im.save("saved_patches/patchnew1.jpg")
                del adv_batch_t, steer_pred, coll_pred, pred_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()
        # utils.save_images_from_Tesnor(adv_patch_cpu)

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

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
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


