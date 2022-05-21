"""
Testing code for Physical Adversarial patch Attack
"""

from torch.utils.data import Dataset
from torchvision import transforms

from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset

from utils.evaluation import *

if __name__ == '__main__':
    image_mode = "rgb"
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test3_weights_484.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'training', image_mode ,augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test8_old_loss_nobeta_balance5_npstvnotchange"
    eval_path = "patch_test59"
    folder  = os.path.exists(os.path.join(test_path, eval_path))
    if not folder:
        os.makedirs(os.path.join(test_path, eval_path))

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test8_old_loss_nobeta_balance5_npstvnotchange/patchs/20220521-014312_steer-0.0_coll-0.0_59.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True

    with torch.no_grad():
        testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch)

    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
    with open(fname_steer, 'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                                os.path.join(test_path, eval_path, "histograms.png"), title_name = "single_patch59")

    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
    with open(fname_labels, 'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                            ['no collision', 'collision'],
                            img_name=os.path.join(test_path, eval_path, "confusion.png"), title_name = "single_patch59")
