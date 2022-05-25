from torch.utils.data import Dataset
from torchvision import transforms

from DroNet.dronet_model import getModel
from DroNet.dronet_load_datasets import DronetDataset
from utils.median_pool import *
from tqdm import tqdm
import  numpy as np
import os
import json
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from PIL import Image
import random

torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

class testPatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(testPatchTransformer, self).__init__()
        self.min_contrast = 0.8  # 0.8
        self.max_contrast = 1.2  # 1.2
        self.min_brightness = -0.1  # -0.1
        self.max_brightness = 0.1  # 0.1
        self.min_scale = 1.0  # Scale the patch size from (patch_size * min_scale) to (patch_size * max_scale)
        self.max_scale = 1.7
        self.noise_factor = 0.1
        self.minangle = -10 / 180 * math.pi
        self.maxangle = 10 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, steer_true, img_size, x, y, x_off, y_off, min_scale, max_scale, do_rotate=True, rand_loc=True):
        # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
        adv_patch = transforms.Resize((100, 100))(adv_patch)
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
        adv_batch = adv_patch.expand(steer_true.size(0), 1, -1, -1, -1)
        batch_size = torch.Size((steer_true.size(0), 1))
        anglesize = steer_true.size(0)

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()

        # Create random scale tensor
        scale = torch.cuda.FloatTensor(batch_size).uniform_(min_scale, max_scale)  # .fill_(0.85)
        scale = scale.view(anglesize)
        scale = scale.cuda()

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)  # compress to min-max, not standardize

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        # cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_ids = (torch.cuda.FloatTensor(batch_size).unsqueeze(-1)).fill_(0)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

        # Pad patch and mask to image dimensions
        mypad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = mypad(adv_batch)
        msk_batch = mypad(msk_batch)

        # Rotation and rescaling transforms
        if do_rotate:
            angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = torch.cuda.FloatTensor(anglesize).fill_(0)

        # Resizes and rotates
        target_x = torch.cuda.FloatTensor([x])
        target_y = torch.cuda.FloatTensor([y])
        targetoff_x = torch.cuda.FloatTensor([x_off])
        targetoff_y = torch.cuda.FloatTensor([y_off])
        if (rand_loc):
            off_x = targetoff_x * (torch.cuda.FloatTensor(anglesize).uniform_(-1, 1.))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(anglesize).uniform_(-1, 1.))
            target_y = target_y + off_y
        else:
            off_x = targetoff_x * (torch.cuda.FloatTensor(anglesize).fill_(1))
            target_x = target_x + off_x
            off_y = targetoff_y * (torch.cuda.FloatTensor(anglesize).fill_(1))
            target_y = target_y + off_y

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        b_sh = adv_batch.shape
        grid = F.affine_grid(theta, adv_batch.shape)

        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t

class testPatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(testPatchApplier, self).__init__()
    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch

def testModel(model, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch,
                x, y, x_off, y_off, rotate, random_loc, min_scale, max_scale):
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
    patch_applier = testPatchApplier().cuda()
    patch_transformer = testPatchTransformer().cuda()

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
            adv_batch_t = patch_transformer(adv_patch, steer_true, 200, x, y, x_off, y_off, min_scale, max_scale, do_rotate=rotate, rand_loc=random_loc)
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


def make_and_save_histograms(pred_steerings, real_steerings, img_name="histograms.png", title_name = None):
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
    plt.figure()
    plt.title(title_name)
    plt.hist(pred_steerings, bins=bins, alpha=0.5, label='Predicted', color='b')
    plt.hist(real_steerings, bins=bins, alpha=0.5, label='Real', color='r')
    # plt.title('Steering angle')
    plt.legend(fontsize=10)
    plt.savefig(img_name, bbox_inches='tight')


def plot_confusion_matrix(real_labels, pred_prob, classes, normalize=False, img_name="confusion.png", title_name = None):
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
    plt.title(title_name)
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
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(img_name)
    plt.show()


if __name__ == '__main__':
    image_mode = "rgb"
    best_weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test3_weights_484.pth"
    dronet = getModel((200, 200), image_mode, 1, best_weights_path)
    # print(dronet)
    dronet = dronet.eval().cuda()

    # Load testing data
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'training', image_mode,
                                    augmentation=False)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True, num_workers=10)

    test_path = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test15_balance1_nps01_tv5_scale05-05"
    eval_path = "patch_test38_07_07-centre"
    folder = os.path.exists(os.path.join(test_path, eval_path))
    if not folder:
        os.makedirs(os.path.join(test_path, eval_path))

    patchfile = "/root/Python_Program_Remote/MyAdvPatch/saved_patch/test15_balance1_nps01_tv5_scale05-05/patchs/20220524-165846_steer0.0_coll0.0_ep38.png"
    adv_patch = Image.open(patchfile).convert('RGB')
    adv_patch = transforms.ToTensor()(adv_patch).cuda()

    is_patch_test = True
    rotate = True
    random_loc = True
    # x, y, x_off, y_off, min_scale, max_scale = 0.095, 0.095, 0.81, 0.81, 0.3, 0.3
    # x, y, x_off, y_off, min_scale, max_scale = 0.155, 0.155, 0.7, 0.7, 0.3, 0.5
    # x, y, x_off, y_off, min_scale, max_scale = 0.3, 0.3, 0.4, 0.4, 1.0, 1.0
    # x, y, x_off, y_off, min_scale, max_scale = 0.39, 0.39, 0.23, 0.23, 0.5, 1.3
    # x, y, x_off, y_off, min_scale, max_scale = 0.5, 0.5, 0., 0., 0.5, 1.7

    x, y, x_off, y_off, min_scale, max_scale = 0.5, 0.5, 0., 0., 0.7, 0.7

    with torch.no_grad():
        testModel(dronet, testing_dataloader, test_path, eval_path, is_patch_test, adv_patch,
                    x, y, x_off, y_off, rotate, random_loc, min_scale, max_scale)

    # Compute histograms from predicted and real steerings
    fname_steer = os.path.join(test_path, eval_path, 'predicted_and_real_steerings.json')
    with open(fname_steer, 'r') as f1:
        dict_steerings = json.load(f1)
    make_and_save_histograms(dict_steerings['pred_steerings'], dict_steerings['real_steerings'],
                             os.path.join(test_path, eval_path, "histograms.png"), title_name=eval_path)

    # Compute confusion matrix from predicted and real labels
    fname_labels = os.path.join(test_path, eval_path, 'predicted_and_real_labels.json')
    with open(fname_labels, 'r') as f2:
        dict_labels = json.load(f2)
    plot_confusion_matrix(dict_labels['real_labels'], dict_labels['pred_probabilities'],
                          ['no collision', 'collision'],
                          img_name=os.path.join(test_path, eval_path, "confusion.png"), title_name=eval_path)
