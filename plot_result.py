import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_loss(experiment_rootdir, fname):
    # Read log file
    try:
        log = np.loadtxt(fname, delimiter=',')
    except:
        raise IOError("Log file not found")

    train_loss = log[:, 0].reshape(-1, 1)
    val_loss = log[:, 1].reshape(-1, 1)
    timesteps = list(range(train_loss.shape[0]))

    # Plot losses
    plt.figure()
    plt.plot(timesteps, train_loss, 'r--', timesteps, val_loss, 'b--')
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(experiment_rootdir, "log.png"))
    plt.show()


def make_and_save_histograms(pred_steerings, real_steerings,
                             img_name="histograms.png"):
    """
    Plot and save histograms from predicted steerings and real steerings.

    # Arguments
        pred_steerings: List of predicted steerings.
        real_steerings: List of real steerings.
        img_name: Name of the png file to save the figure.
    """
    pred_steerings = np.array(pred_steerings)
    real_steerings = np.array(real_steerings)
    max_h = np.maximum(np.max(pred_steerings), np.max(real_steerings))
    min_h = np.minimum(np.min(pred_steerings), np.min(real_steerings))
    bins = np.linspace(min_h, max_h, num=50)
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Predicted', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Real', color='r')
    # plt.title('Steering angle')
    plt.legend(fontsize=10)
    plt.savefig(img_name, bbox_inches='tight')


def plot_confusion_matrix(real_labels, pred_prob, classes,
                          normalize=False,
                          img_name="confusion.png"):
    """
    Plot and save confusion matrix computed from predicted and real labels.

        # Arguments
        real_labels: List of real labels.
        pred_prob: List of predicted probabilities.
        normalize: Boolean, whether to apply normalization.
        img_name: Name of the png file to save the figure.
    """
    real_labels = np.array(real_labels)

    # Binarize predicted probabilities
    pred_prob = np.array(pred_prob)
    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1

    cm = confusion_matrix(real_labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(img_name)


if __name__ == "__main__":
    experiment_rootdir = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test1_RGB_old_loss_200_nice"
    eval_path = "evaluation_199"
    # experiment_rootdir = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test1_RGB_old_loss_200_nice"
    # eval_path = "patch_test4_38"


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
    fname_loss = os.path.join(experiment_rootdir, 'losses.txt')
    # plot_loss(experiment_rootdir, fname_loss)
