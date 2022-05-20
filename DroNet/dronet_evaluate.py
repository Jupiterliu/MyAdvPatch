import torch
from DroNet.dronet_load_datasets import DronetDataset
from DroNet.dronet_model import DronetTorch
from DroNet.dronet_model import getModel
from plot_result import *
from load_data import *
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm
import sklearn
import re
import os
import numpy as np
np.set_printoptions(suppress=True)

from random import randint
from sklearn import metrics
import json


def testModel(model, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch):
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
            adv_batch_t = patch_transformer(adv_patch, steer_true, 200, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_cuda, adv_batch_t)
            img_cuda = F.interpolate(p_img_batch, (200, 200))  # Up or Down sample

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

    # Compute random and constant baselines for steerings
    random_steerings = random_regression_baseline(real_steerings)
    constant_steerings = constant_baseline(real_steerings)

    # Create dictionary with filenames
    dict_fname = {'test_regression.json': pred_steerings,
                    'random_regression.json': random_steerings,
                    'constant_regression.json': constant_steerings}

    # Evaluate predictions: EVA, residuals, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(test_path, eval_path, fname)
        evaluate_regression(pred, real_steerings, abs_fname)

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

    # Compute random, weighted and majorirty-class baselines for collision
    random_labels = random_classification_baseline(real_labels)

    # Create dictionary with filenames
    dict_fname = {'test_classification.json': pred_labels,
                    'random_classification.json': random_labels}

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    for fname, pred in dict_fname.items():
        abs_fname = os.path.join(test_path, eval_path, fname)
        evaluate_classification(pred_collisions, pred, real_labels, abs_fname)

    # Write predicted probabilities and real labels
    dict_test = {'pred_probabilities': pred_collisions.tolist(), 'real_labels': real_labels.tolist()}
    write_to_file(dict_test, os.path.join(test_path, eval_path, 'predicted_and_real_labels.json'))

def random_regression_baseline(real_values):
    mean = np.mean(real_values)
    std = np.std(real_values)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values):
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)

def evaluate_regression(predictions, real_values, fname):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values, n_errors=20)
    dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(), "highest_errors": highest_errors.tolist()}
    write_to_file(dictionary, fname)

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

def random_classification_baseline(real_values):
    """
    Randomly assigns half of the labels to class 0, and the other half to class 1
    """
    return [randint(0,1) for p in range(real_values.shape[0])]

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


if __name__ == '__main__':
    image_mode = "rgb"
    # weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test4_GRAY_new_loss_500/weights_070.pth"
    # weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test3_RGB_old_loss_500/weights_435.pth"
    weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test2_RGB_new_loss_500/models/weights_288.pth"
    dronet = getModel((200, 200), image_mode, 1, weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode, augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test2_RGB_new_loss_500"
    eval_path = "evaluation_288"
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
