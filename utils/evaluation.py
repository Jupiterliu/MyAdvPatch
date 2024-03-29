from utils.images import *
from utils.losses import *
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
np.set_printoptions(suppress=True)

from sklearn import metrics

import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def testModel(model, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch, patch_name, steer_target, centre,
              do_rotate=True, do_pespective=True, nested=0, nested_size=0.5, location="random",
              min_scale=1, max_scale=3.6):
    '''
    tests the model with the following metrics:

    root mean square error (RMSE) for steering angle.

    expected variance for steering angle.

    accuracy % for probability of collision.

    f1 score for probability of collision.

    ## parameters

    `model`: `torch.nn.Module`: the dronet model.

    `weights_path`: `str`: the path to the file to get
    the weights from the trained model. No use in keeping it at the default of `None`,
    and having (very, very, very likely) horrible metrics.
    '''
    # go through all values
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    all_true_steer = torch.cuda.FloatTensor()
    all_true_coll = torch.cuda.FloatTensor()
    all_pred_steer = torch.cuda.FloatTensor()
    all_pred_coll = torch.cuda.FloatTensor()
    all_exp_type = torch.cuda.FloatTensor()

    for idx, (img, steer_true, coll_true) in tqdm(enumerate(testing_dataloader), desc=f'Running epoch {0}', total=len(testing_dataloader)):
        img_cuda = img.cuda()
        true_steer = steer_true.squeeze(1).cuda()[:, 1]
        true_coll = coll_true.squeeze(1).cuda()[:, 1]
        exp_type = steer_true.squeeze(1).cuda()[:, 0]
        # patch testing
        if is_patch_test:
            # patch projection
            adv_batch_t = patch_transformer(adv_patch, steer_true, 200, 400, patch_name, steer_target, do_rotate, do_pespective, nested, nested_size, centre, location, min_scale, max_scale)
            p_img_batch = patch_applier(img_cuda, adv_batch_t)
            # img_cuda = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample
            img_cuda = p_img_batch

        steer_pred, coll_pred = model(img_cuda)
        pred_steer = steer_pred.squeeze(-1)
        pred_coll = coll_pred.squeeze(-1)
        # print(f'Dronet Ground Truth {steer_true.item()} angle and {coll_true.item()} collision.')
        # print(f'Dronet predicts {steer_pred.item()} angle and {coll_pred.item()} collision.\n')

        # All true and pred data:
        all_true_steer = torch.cat((all_true_steer, true_steer))
        all_true_coll = torch.cat((all_true_coll, true_coll))
        all_pred_steer = torch.cat((all_pred_steer, pred_steer))
        all_pred_coll = torch.cat((all_pred_coll, pred_coll))
        all_exp_type = torch.cat((all_exp_type, exp_type))

    # Saving all steer and coll to txt
    steer = torch.cat((all_true_steer.unsqueeze(-1), all_pred_steer.unsqueeze(-1)), 1)
    coll = torch.cat((all_true_coll.unsqueeze(-1), all_pred_coll.unsqueeze(-1)), 1)

    # Param t. t=1 steering, t=0 collision
    t_mask = all_exp_type == 1

    coll = mul_columns_sort(coll[~t_mask, ].cpu().numpy())
    np.savetxt(os.path.join(test_path, eval_path, 'steerings.txt'), steer[t_mask,].cpu().numpy(), fmt="%f", delimiter=",")
    np.savetxt(os.path.join(test_path, eval_path, 'collision.txt'), coll, fmt="%f", delimiter=",")

    # ************************* Steering evaluation ***************************
    # Predicted and real steerings
    pred_steerings = all_pred_steer[t_mask, ].cpu().numpy()
    real_steerings = all_true_steer[t_mask, ].cpu().numpy()

    # Evaluate predictions: EVA, residuals, and highest errors
    abs_fname = os.path.join(test_path, eval_path, "test_regression.json")
    evas, rmse = evaluate_regression(pred_steerings, real_steerings, abs_fname)

    # Write predicted and real steerings
    dict_test = {'pred_steerings': pred_steerings.tolist(),
                    'real_steerings': real_steerings.tolist()}
    write_to_file(dict_test, os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json'))

    # *********************** Collision evaluation ****************************
    # Predicted probabilities and real labels
    pred_collisions = all_pred_coll[~t_mask, ].cpu().numpy()
    pred_labels = np.zeros_like(pred_collisions)
    pred_labels[pred_collisions >= 0.5] = 1
    real_labels = all_true_coll[~t_mask, ].cpu().numpy()

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    abs_fname = os.path.join(test_path, eval_path, "test_classification.json")
    ave_accuracy, precision, recall, f_score = evaluate_classification(pred_collisions, pred_labels, real_labels, abs_fname)

    # Write predicted probabilities and real labels
    dict_test = {'pred_probabilities': pred_collisions.tolist(), 'real_labels': real_labels.tolist()}
    write_to_file(dict_test, os.path.join(test_path, eval_path, 'predicted_and_real_labels.json'))
    return evas, rmse, ave_accuracy, precision, recall, f_score

def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values, n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(), "highest_errors": highest_errors.tolist()}
    write_to_file(dictionary, fname)
    return evas, rmse

def compute_explained_variance(predictions, real_values):
    """
    Computes the explained variance of prediction for each
    steering and the average of them
    """
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print("EVA = {}".format(ex_variance))
    return ex_variance

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y].
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def compute_rmse(predictions, real_values):
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = np.sqrt(mse)
    print("RMSE = {}".format(rmse))
    return rmse

def compute_highest_regression_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    highest_errors = sq_res.argsort()[-n_errors:][::-1]
    return highest_errors

def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary, f)
        print("Written file {}".format(fname))

def evaluate_classification(pred_prob, pred_labels, real_labels, fname):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.precision_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
            n_errors=20)
    dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
                  "recall": recall.tolist(), "f_score": f_score.tolist(),
                  "highest_errors": highest_errors.tolist()}
    write_to_file(dictionary, fname)
    return ave_accuracy, precision, recall, f_score

def compute_highest_classification_errors(predictions, real_values, n_errors=20):
    """
    Compute the indexes with highest error
    """
    assert np.all(predictions.shape == real_values.shape)
    dist = abs(predictions - real_values)
    highest_errors = dist.argsort()[-n_errors:][::-1]
    return highest_errors

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def mul_columns_sort(data):
    m2A = []
    for i in range(0, len(data)):
        m2A.append((data[i][0], data[i][1]))
    dtype = [('x', float), ('y', float)]
    tuple1 = np.array(m2A, dtype)
    tuple1 = np.sort(tuple1, order=['x', 'y'])
    print(tuple1.shape)
    inFile = np.array(totuple(tuple1))
    return inFile

def mean_ablosute_error(pred, true):
    return np.sum(np.abs(pred - true))/len(pred)

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


def evaluation_metrics(pred_steerings, real_steerings, real_labels, pred_prob, classes,
                       attack_mode, steer_target, normalize=True, title_name=None, saved_path=None, ishow=True):
    ### for the steering angle
    pred_steerings = np.array(pred_steerings) * 90
    real_steerings = np.array(real_steerings) * 90
    pred_steer_min = np.min(pred_steerings)
    pred_steer_max = np.max(pred_steerings)
    max_h = np.maximum(np.max(pred_steerings), np.max(real_steerings))
    min_h = np.minimum(np.min(pred_steerings), np.min(real_steerings))
    bins = np.linspace(min_h, max_h, num=100)
    font_blue = {'color': 'blue',
            'size': 16,
            'family': 'Times New Roman',
            # 'style':'italic',  # 斜体
            }
    font_red = {'color': 'red',
                 'size': 16,
                 'family': 'Times New Roman',
                 # 'style':'italic',  # 斜体
                 }
    plt.figure(1, figsize=(7, 7))
    plt.subplot(2, 1, 1)
    plt.title("Steering Angle")
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Patched Prediction', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Clean Label', color='r')
    plt.legend(fontsize=10)
    # plt.savefig(os.path.join(saved_path, "histograms.png"), bbox_inches='tight')


    ### for the collision probability
    real_labels = np.array(real_labels)

    # Binarize predicted probabilities
    pred_prob = np.array(pred_prob)
    pred_labels = np.zeros_like(pred_prob)
    pred_labels[pred_prob >= 0.5] = 1

    cm = confusion_matrix(real_labels, pred_labels)
    plt.figure(1)
    plt.subplot(2, 2, 3)
    plt.title("Collision Prob.")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix")
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)
    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 5)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Clean Label')
    plt.xlabel('Patched Prediction')
    # plt.savefig(os.path.join(saved_path, "confusion.png"))

    ### For the metrics: ASDR, EVA, RMSE, mASR, mF1-score
    plt.figure(1)
    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.title("Metrics")
    if attack_mode == "HA":
        ASD = np.mean(np.abs((pred_steerings - real_steerings)))
        MAE = mean_ablosute_error(pred_steerings, np.zeros(len(pred_steerings)))
        RMSE = compute_rmse(pred_steerings, np.zeros(len(pred_steerings)))
        mASR = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        mF1 = cm[1, 0] / (cm[1, 0] + (cm[0, 1] + cm[1, 1])/2)
    elif attack_mode == "YA":
        ASD = np.mean(np.abs((pred_steerings - real_steerings)))
        MAE = mean_ablosute_error(pred_steerings, np.ones(len(pred_steerings)) * steer_target * 90)
        RMSE = compute_rmse(pred_steerings, np.ones(len(pred_steerings)) * steer_target * 90)
        mASR = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        mF1 = cm[1, 0] / (cm[1, 0] + (cm[0, 1] + cm[1, 1]) / 2)
    elif attack_mode == "OA":
        # False
        ASD = np.mean(np.abs((pred_steerings - real_steerings)))
        MAE = mean_ablosute_error(pred_steerings, np.zeros(len(pred_steerings)))
        RMSE = compute_rmse(pred_steerings, np.zeros(len(pred_steerings)))
        mASR = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        mF1 = cm[0, 1] / (cm[0, 1] + (cm[0, 0] + cm[1, 0]) / 2)
    else:
        print("attack_mode is wrong!!!!")
    plt.text(0.05, 0.9,  "ASD:           " + str(np.around(ASD,  3)) + "°", fontdict=font_blue)
    plt.text(0.05, 0.75, "MAE:           " + str(np.around(MAE,  3)) + "°", fontdict=font_blue)
    plt.text(0.05, 0.6,  "RMSE:        "   + str(np.around(RMSE, 3)), fontdict=font_blue)
    plt.text(0.05, 0.45, "Min_pred:  "     + str(np.around(pred_steer_min, 3)) + "°", fontdict=font_blue)
    plt.text(0.05, 0.3,  "Max_pred:  "     + str(np.around(pred_steer_max, 3)) + "°", fontdict=font_blue)
    plt.text(0.05, 0.15, "mASR:        "   + str(np.around(mASR, 3) * 100) + "%", fontdict=font_red)
    plt.text(0.05, 0.,   "mF1:           " + str(np.around(mF1,  3) * 100) + "%", fontdict=font_red)
    if ishow:
        # plt.tight_layout()
        plt.suptitle(attack_mode + ":" + title_name)
        plt.savefig(os.path.join(saved_path, title_name))
        plt.show()
    return ASD, MAE, RMSE, mASR, mF1