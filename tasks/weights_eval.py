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

    env_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test7_RGB_new_loss_beta20"
    models_path = os.path.join(env_path, "models")
    # print("Loaded weights path: ", models_path)
    folder = os.path.join(env_path, "multi_patchs_eval_result")
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_result_confusion = os.path.join(env_path, "confusion")
    if not os.path.exists(plot_result_confusion):
        os.makedirs(plot_result_confusion)
    plot_result_histograms = os.path.join(env_path, "histograms")
    if not os.path.exists(plot_result_histograms):
        os.makedirs(plot_result_histograms)

    is_patch_test = False

    weights = sorted(os.listdir(models_path))
    all_criterion = np.zeros((len(weights), 7))
    index = 0
    for weight in weights:
        weight_path = os.path.join(models_path, weight)
        dronet = getModel((200, 200), image_mode, 1, weight_path)
        dronet = dronet.eval().cuda()
        weight_result_path = os.path.join(folder, weight)
        if not os.path.exists(weight_result_path):
            os.makedirs(weight_result_path)
        with torch.no_grad():
            eva, rmse, ave_accuracy, precision, recall, f_score = testModel(dronet, testing_dataloader,
                                                                            folder, weight, is_patch_test, None)
            all_criterion[index, 0] = index
            all_criterion[index, 1] = eva
            all_criterion[index, 2] = rmse
            all_criterion[index, 3] = ave_accuracy
            all_criterion[index, 4] = precision
            all_criterion[index, 5] = recall
            all_criterion[index, 6] = f_score
            np.savetxt(os.path.join(env_path, 'all_criterion.txt'), all_criterion, fmt="%f")

            # Compute histograms from predicted and real steerings
            fname_steer = os.path.join(folder, weight, 'predicted_and_real_steerings.json')
            with open(fname_steer, 'r') as f1:
                dict_steerings = json.load(f1)
            make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                                        os.path.join(plot_result_histograms, "histograms_{}.png".format(index)),
                                        title_name = "patch_{}".format(index))

            # Compute confusion matrix from predicted and real labels
            fname_labels = os.path.join(folder, weight, 'predicted_and_real_labels.json')
            with open(fname_labels, 'r') as f2:
                dict_labels = json.load(f2)
            plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                                    ['no collision', 'collision'],
                                    img_name=os.path.join(plot_result_confusion, "confusion_{}.png".format(index)),
                                    title_name = "patch_{}".format(index))
            index = index + 1

    # all_criterion