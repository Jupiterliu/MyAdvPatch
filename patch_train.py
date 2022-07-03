"""
Training code for Physical Adversarial patch Attack
"""
import os

from tqdm import tqdm

from utils.images import *
from utils.losses import *
import matplotlib.pyplot as plt
from torch import autograd, optim
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

# import DroNet.dronet_torch
from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

class PatchTrainer(object):
    def __init__(self, mode):
        # Load config from args
        self.config = patch_config.patch_configs[mode]()

        # Load DroNet model from .pth & eval()
        best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test8_weights_346.pth"
        self.dronet_model = getModel((200, 200), self.config.image_mode, 1, best_weights_path)
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

        # adv_patch_circle = self.init_patch_circle(200, math.pi)
        # adv_patch_cpu =  torch.from_numpy(adv_patch_circle)

        adv_patch_cpu.requires_grad_(True)

        # Load my data from collision testing
        training_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'training',
                                            self.config.image_mode, augmentation=False)
        training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=self.config.batch_size,
                                                            shuffle=True, num_workers=self.config.num_workers)
        self.epoch_length = len(training_dataloader)
        # print(f'One epoch is {len(training_dataloader)}')

        root_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch"
        saved_patch_name = "test17_nopes_lr01_k128_balance100-100_beta10_gamma1_nps001_tv25_scale10-17"
        patch_path = os.path.join(root_path, saved_patch_name, "patchs")
        if not os.path.exists(patch_path):
            os.makedirs(patch_path)
        if not os.path.exists(os.path.join(root_path, saved_patch_name, "temp")):
            os.makedirs(os.path.join(root_path, saved_patch_name, "temp"))

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)  # Improving the performance by reducing the learning rate

        et0 = time.time()
        for epoch in range(self.config.n_epochs):
            ep_attack_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            other_val = (self.config.gamma - torch.exp(torch.Tensor([-1 * (0.1) * (epoch - self.config.beta)]))).float().cuda()
            beta = torch.max(torch.Tensor([0]).float().cuda(), other_val)
            # beta = torch.Tensor([1.0]).float().cuda()
            for i_batch, (img_batch, steer_true, coll_true) in tqdm(enumerate(training_dataloader),
                                                                    desc=f'Running epoch {epoch}', total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    steer_true = steer_true.cuda().squeeze(1)
                    coll_true = coll_true.cuda().squeeze(1)
                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()

                    # patch projection
                    if self.config.image_mode == "gray":
                        adv_patch = transforms.Grayscale()(adv_patch)
                    adv_batch_t = self.patch_transformer(adv_patch, steer_true, self.config.image_size,
                                                         do_rotate=True, do_pespective=False, do_nested=True, location="random")
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample

                    if self.config.is_save_temp:
                        for i in range(p_img_batch.size(0)):
                            Tensor = p_img_batch[i, :, :, :]
                            # image = np.transpose(Tensor.detach().cpu().numpy(), (1, 2, 0))
                            patcded_image = transforms.ToPILImage('RGB')(Tensor)
                            plt.imshow(patcded_image)
                            patcded_image.save(os.path.join(root_path, saved_patch_name, "temp", "temp_batch{:0>2d}_im{:0>2d}.png".format(i_batch, i)))

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
                    attack_loss = attack_loss * self.config.attack_loss_weight    # 1
                    nps_loss = nps * self.config.nps_loss_weight                  # 0.01
                    tv_loss = tv * self.config.tv_loss_weight                     # 2.5

                    loss = attack_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) + nps_loss

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
            im.save(os.path.join(patch_path, "{}_steer{}_coll{}_ep{:0>2d}.png".format(time_str, self.config.steer_target, self.config.coll_target, epoch)))

            scheduler.step(ep_loss)
            if True:
                print('   EPOCH NR: ', epoch),
                print(' EPOCH LOSS: ', ep_loss.item())
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

    def init_patch_circle(self, image_size, patch_size):
        image_size = image_size ** 2
        noise_size = image_size * patch_size
        radius = int(math.sqrt(noise_size / math.pi) / 2)
        patch = np.zeros((1, 3, radius * 2, radius * 2))
        for i in range(3):
            a = np.zeros((radius * 2, radius * 2))
            cx, cy = radius, radius  # The center of circle
            y, x = np.ogrid[-radius: radius, -radius: radius]
            index = x ** 2 + y ** 2 <= radius ** 2
            a[cy - radius:cy + radius, cx - radius:cx + radius][index] = np.random.rand()
            idx = np.flatnonzero((a == 0).all((1)))
            a = np.delete(a, idx, axis=0)
            patch[0][i] = np.delete(a, idx, axis=1)
        return patch.reshape(3, radius * 2, radius * 2)

def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)

    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


