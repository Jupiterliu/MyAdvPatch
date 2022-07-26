"""
Testing code for Physical Adversarial patch Attack
"""

from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

from utils.evaluation import *
from utils.plot import *

import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_mode = "rgb"
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test8_weights_346.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode ,augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test17_p400_lr005_balance10-10_beta5_nps001_tv25_nest0_scale01-35"
    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test17_p400_lr005_balance10-10_beta5_nps001_tv25_nest0_scale01-35/patchs/20220725-165720_steer0.0_coll1.0_ep095.png"
    test_num = 17
    patch_epoch = 95
    # adv_patch_cpu = torch.rand((3, 200, 200))
    # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
    # plt.imshow(im)
    # im.save(os.path.join(test_path, "random_patch.png"))

    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True

    is_distance_test = True
    fixed_distance = np.arange(0.5, 2.1, 0.1)
    metrics_results = np.zeros((len(fixed_distance), 6))
    min_scale = 2
    max_scale = 2
    do_rotate = True
    do_pespective = True
    nested = 0
    nested_size = 0.5
    location = "random"
    attack_mode = "OA"
    steer_target = 1
    centre = False

    plot_result = os.path.join(test_path, "fixed_dis_plot_result_nested")
    if not os.path.exists(plot_result):
        os.makedirs(plot_result)
    fixed_distance_result = os.path.join(test_path, "fixed_distance_nested")
    if not os.path.exists(fixed_distance_result):
        os.makedirs(fixed_distance_result)

    if is_distance_test:  # test the fixed distance
        index = 0
        for i in fixed_distance:
            eval_path = "test{}_patch{}_scale{}-{}".format(int(test_num),int(patch_epoch),int(i*10), int(i*10))
            # eval_path = "patch91_scale10-36-random"
            results = os.path.join(fixed_distance_result, eval_path)
            if not os.path.exists(results):
                os.makedirs(results)
            with torch.no_grad():
                testModel(dronet, testing_dataloader, fixed_distance_result, eval_path, is_patch_test, adv_patch,
                          attack_mode, steer_target, centre,
                          do_rotate=do_rotate, do_pespective=do_pespective, nested=nested, nested_size=nested_size, location=location,
                          min_scale=i, max_scale=i)

                # Compute histograms from predicted and real steerings; confusion matrix from predicted and real labels
                fname_steer = os.path.join(fixed_distance_result, eval_path, 'predicted_and_real_steerings.json')
                with open(fname_steer, 'r') as f1:
                    dict_steerings = json.load(f1)
                fname_labels = os.path.join(fixed_distance_result, eval_path, 'predicted_and_real_labels.json')
                with open(fname_labels, 'r') as f2:
                    dict_labels = json.load(f2)
                ASD, MAE, RMSE, mASR, mF1 = evaluation_metrics(dict_steerings['pred_steerings'], dict_steerings['real_steerings'], dict_labels['real_labels'], dict_labels['pred_probabilities'],
                                        ['no collision', 'collision'], attack_mode, steer_target, title_name=eval_path, saved_path=plot_result, ishow=True)
                metrics_results[index, 0] = i
                metrics_results[index, 1] = ASD
                metrics_results[index, 2] = MAE
                metrics_results[index, 3] = RMSE
                metrics_results[index, 4] = mASR
                metrics_results[index, 5] = mF1
                np.savetxt(os.path.join(plot_result, 'metrics_results.txt'), metrics_results, fmt="%f")

                index = index + 1
        plot_metrics_results(plot_result)

    else:
        eval_path = "test{}_patch{}_scale{}-{}".format(int(test_num),int(patch_epoch),int(min_scale*10), int(max_scale*10))
        results = os.path.join(test_path, eval_path)
        if not os.path.exists(results):
            os.makedirs(results)
        with torch.no_grad():
            testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch, attack_mode, steer_target,
                      do_rotate=do_rotate, do_pespective=do_pespective, nested=nested, nested_size=nested_size, location=location,
                      min_scale=min_scale, max_scale=max_scale)

            # Compute histograms from predicted and real steerings; confusion matrix from predicted and real labels
            fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
            with open(fname_steer, 'r') as f1:
                dict_steerings = json.load(f1)
            fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
            with open(fname_labels, 'r') as f2:
                dict_labels = json.load(f2)
            evaluation_metrics(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                               dict_labels['real_labels'], dict_labels['pred_probabilities'],
                               ['no collision', 'collision'], attack_mode, steer_target, title_name=eval_path, saved_path=results,
                               ishow=True)
