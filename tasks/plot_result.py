
from utils.evaluation import *


if __name__ == "__main__":
    # experiment_rootdir = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test1_RGB_old_loss_200_nice"
    # eval_path = "evaluation_199"
    experiment_rootdir = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB"
    eval_path = "patch_test6_9"


    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(experiment_rootdir, eval_path, 'predicted_and_real_steerings.json')
    with open(fname_steer, 'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                                os.path.join(experiment_rootdir, eval_path, "histograms.png"))


    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(experiment_rootdir, eval_path, 'predicted_and_real_labels.json')
    with open(fname_labels, 'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                            ['no collision', 'collision'],
                            img_name=os.path.join(experiment_rootdir, eval_path, "confusion.png"))

    # Plot Loss from losses.txt
    # fname_loss = os.path.join(experiment_rootdir, 'losses.txt')
    # plot_loss(experiment_rootdir, fname_loss)
