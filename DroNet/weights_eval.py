import torch
from DroNet.dronet_load_datasets import DronetDataset
from DroNet.dronet_model import DronetTorch
from DroNet.dronet_model import getModel
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


def testModel(model, testing_dataloader):
    # go through all values
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

    # Param t. t=1 steering, t=0 collision
    t_mask = all_exp_type == 1

    # ************************* Steering evaluation ***************************
    # Predicted and real steerings
    pred_steerings = all_pred_steer[t_mask,].cpu().numpy()
    real_steerings = all_true_steer[t_mask,].cpu().numpy()

    # Evaluate predictions: EVA, residuals, and highest errors
    eva, rmse = evaluate_regression(pred_steerings, real_steerings)

    # *********************** Collision evaluation ****************************
    # Predicted probabilities and real labels
    pred_collisions = all_pred_coll[~t_mask,].cpu().numpy()
    pred_labels = np.zeros_like(pred_collisions)
    pred_labels[pred_collisions >= 0.5] = 1
    real_labels = all_true_coll[~t_mask, ].cpu().numpy()

    # Evaluate predictions: accuracy, precision, recall, F1-score, and highest errors
    ave_accuracy, precision, recall, f_score = evaluate_classification(pred_collisions, pred_labels, real_labels)

    return eva, rmse, ave_accuracy, precision, recall, f_score

def evaluate_regression(predictions, real_values):
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    # highest_errors = compute_highest_regression_errors(predictions, real_values, n_errors=20)
    # dictionary = {"evas": evas.tolist(), "rmse": rmse.tolist(), "highest_errors": highest_errors.tolist()}
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

def evaluate_classification(pred_prob, pred_labels, real_labels):
    ave_accuracy = metrics.accuracy_score(real_labels, pred_labels)
    print('Average accuracy = ', ave_accuracy)
    precision = metrics.precision_score(real_labels, pred_labels)
    print('Precision = ', precision)
    recall = metrics.precision_score(real_labels, pred_labels)
    print('Recall = ', recall)
    f_score = metrics.f1_score(real_labels, pred_labels)
    print('F1-score = ', f_score)
    # highest_errors = compute_highest_classification_errors(pred_prob, real_labels,
    #         n_errors=20)
    # dictionary = {"ave_accuracy": ave_accuracy.tolist(), "precision": precision.tolist(),
    #               "recall": recall.tolist(), "f_score": f_score.tolist(),
    #               "highest_errors": highest_errors.tolist()}
    # write_to_file(dictionary, fname)
    return ave_accuracy, precision, recall, f_score

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
    # Load testing data
    image_mode = "rgb"
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', image_mode,
                                    augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    models_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/test5_GRAY_old_loss_141/models"
    print("Loaded weights path: ", models_path)
    weights = sorted(os.listdir(models_path))
    all_criterion = np.zeros((len(weights), 7))
    index = 0
    for weight in weights:
        weight_path = os.path.join(models_path, weight)
        dronet = getModel((200, 200), image_mode, 1, weight_path)
        dronet = dronet.eval().cuda()
        with torch.no_grad():
            eva, rmse, ave_accuracy, precision, recall, f_score = testModel(dronet, testing_dataloader)
            all_criterion[index, 0] = index
            all_criterion[index, 1] = eva
            all_criterion[index, 2] = rmse
            all_criterion[index, 3] = ave_accuracy
            all_criterion[index, 4] = precision
            all_criterion[index, 5] = recall
            all_criterion[index, 6] = f_score
            index = index + 1
            np.savetxt(os.path.join(models_path, 'all_criterion.txt'), all_criterion, fmt="%f")

    # all_criterion