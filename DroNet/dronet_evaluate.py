import numpy as np
np.set_printoptions(suppress=True)

from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

from utils.evaluation import *

if __name__ == '__main__':
    image_mode = "gray"
    # weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test4_GRAY_new_loss_500/weights_070.pth"
    # weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test3_RGB_old_loss_500/weights_435.pth"
    weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test9_GRAY_new_loss_beta50/models/weights_410.pth"
    dronet = getModel((200, 200), image_mode, 1, weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode, augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test9_GRAY_new_loss_beta50"
    eval_path = "evaluation_410"
    folder = os.path.exists(os.path.join(test_path, eval_path))
    if not folder:
        os.makedirs(os.path.join(test_path, eval_path))

    is_patch_test = False

    with torch.no_grad():
        testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, None)

    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
    with open(fname_steer, 'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                                os.path.join(test_path, eval_path, "histograms.png"))

    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
    with open(fname_labels, 'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                            ['no collision', 'collision'],
                            img_name=os.path.join(test_path, eval_path, "confusion.png"))
