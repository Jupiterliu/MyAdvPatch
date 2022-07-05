"""
Testing code for Physical Adversarial patch Attack
"""

from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

from utils.evaluation import *

if __name__ == '__main__':
    image_mode = "rgb"
    attack_mode = "HA"  # YA, CA
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test8_weights_346.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode ,augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test18_nopes_lr01_k128_balance100-100_beta10_gamma1_nps001_tv25_scale10-17/patchs/20220702-225515_steer0.0_coll0.0_ep91.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True

    for i in np.arange(0.5, 1, 0.1):
    # i = 1.0
        test_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test18_nopes_lr01_k128_balance100-100_beta10_gamma1_nps001_tv25_scale10-17"
        eval_path = "patch91_scale{}-{}-random".format(int(i*10), int(i*10))
        # eval_path = "patch91_scale10-36-random"
        results = os.path.join(test_path, eval_path)
        if not os.path.exists(results):
            os.makedirs(results)
        with torch.no_grad():
            testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch,
                      do_rotate=True, do_pespective=True, do_nested=True, location="random",
                      min_scale=i, max_scale=i)

            # Compute histograms from predicted and real steerings; confusion matrix from predicted and real labels
            fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
            with open(fname_steer, 'r') as f1:
                dict_steerings = json.load(f1)
            fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
            with open(fname_labels, 'r') as f2:
                dict_labels = json.load(f2)
            evaluation_metrics(dict_steerings['pred_steerings'], dict_steerings['real_steerings'], dict_labels['real_labels'], dict_labels['pred_probabilities'],
                                    ['no collision', 'collision'], attack_mode, title_name=eval_path, saved_path=results, ishow=True)
