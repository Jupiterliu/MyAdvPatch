import numpy as np
np.set_printoptions(suppress=True)

from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

from utils.evaluation import *

if __name__ == '__main__':
    # Load testing data
    image_mode = "rgb"
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode,
                                    augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    env_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test6_RGB_old_loss_beta50_103"
    models_path = os.path.join(env_path, "models")
    print("Loaded weights path: ", models_path)
    eval_path = os.path.join(env_path, "eval_result")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    is_patch_test = False

    weights = sorted(os.listdir(models_path))
    all_criterion = np.zeros((len(weights), 7))
    index = 0
    for weight in weights:
        weight_path = os.path.join(models_path, weight)
        dronet = getModel((200, 200), image_mode, 1, weight_path)
        dronet = dronet.eval().cuda()
        weight_result_path = os.path.join(eval_path, weight)
        if not os.path.exists(weight_result_path):
            os.makedirs(weight_result_path)
        with torch.no_grad():
            eva, rmse, ave_accuracy, precision, recall, f_score = testModel(dronet, testing_dataloader,
                                                                            eval_path, weight, is_patch_test, None)
            all_criterion[index, 0] = index
            all_criterion[index, 1] = eva
            all_criterion[index, 2] = rmse
            all_criterion[index, 3] = ave_accuracy
            all_criterion[index, 4] = precision
            all_criterion[index, 5] = recall
            all_criterion[index, 6] = f_score
            index = index + 1
            np.savetxt(os.path.join(env_path, 'all_criterion.txt'), all_criterion, fmt="%f")

    # all_criterion