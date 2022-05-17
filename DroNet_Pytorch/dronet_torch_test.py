import torch
from dronet_datasets import DronetDataset
from dronet_torch import DronetTorch
from dronet_torch_train import getModel
import sklearn

def testModel(model):
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
    
    testing_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'testing', augmentation=False)

    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1,
                                            shuffle=True, num_workers=10)
    # go through all values
    cumulative = 0
    # cumulative_tensor_true = torch.cuda.FloatTensor(torch.Size((16,1))).fill_(0)
    # cumulative_tensor_pred = torch.cuda.FloatTensor(torch.Size((16,1))).fill_(0)
    cumulative_tensor_true = torch.cuda.FloatTensor()
    cumulative_tensor_pred = torch.cuda.FloatTensor()
    n = len(testing_dataloader)
    num_correct = 0
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    for idx, (img, steer_true, coll_true) in enumerate(testing_dataloader):
        img_cuda = img.cuda()
        steer_true = steer_true.cuda()
        coll_true = coll_true.cuda()
        steer_pred, coll_pred = model(img_cuda)
        print(f'Dronet Ground Truth {steer_true.item()} angle and {coll_true.item()} collision.')
        print(f'Dronet predicts {steer_pred.item()} angle and {coll_pred.item()} collision.\n')
        # rmse
        cumulative += ((steer_pred - steer_true)**2)

        # eva
        # shape is (n,1), will be flatten out
        cumulative_tensor_true = torch.cat((cumulative_tensor_true, steer_true))
        cumulative_tensor_pred = torch.cat((cumulative_tensor_pred, steer_pred))

        # accuracy
        concrete_probs_true = coll_true >= 0.5
        concrete_probs_pred = coll_pred >= 0.5
        num_correct += torch.sum(concrete_probs_pred == concrete_probs_true)


        # get precision and recall
        true_positives = torch.sum(concrete_probs_true)
        for i in range(len(concrete_probs_pred)):
            # thought was positive, but wasn't
            if concrete_probs_pred[i]==True and concrete_probs_true[i]==False:
                false_positives+=1
            # thoght was negative, but wasn't
            elif concrete_probs_pred[i]==False and concrete_probs_true[i]==True:
                false_negatives+=1
         

        # f1 score, that's interesting
    # p is the number of correct positive results divided by the 
    # number of all positive results returned by the classifier, and r is 
    # the number of correct positive results divided by the number of all relevant samples 
    # (all samples that should be positive)
    # calculated root mean square error
    rmse = torch.sqrt(torch.div(cumulative, n))

    # calculated expected variance
    cumulative_tensor_pred = cumulative_tensor_pred.flatten()
    cumulative_tensor_true = cumulative_tensor_true.flatten()
    subtracted_var = torch.var(cumulative_tensor_true - cumulative_tensor_pred)
    true_var = torch.var(cumulative_tensor_true)
    eva = torch.div(subtracted_var, true_var)
    
    # accuracy
    accuracy = torch.div(num_correct, n)
    # f1 scoreu
    precision = torch.div(true_positives, true_positives + false_positives)
    recall = torch.div(true_positives, true_positives + false_negatives)
    f1_score = 2 * (torch.div(precision * recall, precision + recall))
    print('Testing complete. Displaying results..')
    print('--------------------------')
    print('RMSE(steer): ', rmse.item())
    print('--------------------------')
    print('Expected Variance(steer): ', eva.item())
    print('--------------------------')
    print('Accuracy(coll): ', accuracy.item())
    print('--------------------------')
    print('Precision(coll): ', precision.item())
    print('--------------------------')
    print('Recall(coll): ', recall.item())
    print('--------------------------')
    print('F1 Score(coll): ', f1_score.cpu().numpy())
    print('**************************')

if __name__ == '__main__':
    weights_path = "/root/Python_Program_Remote/MyAdvPatch/DroNet_Pytorch/saved_models/test4_RGB_new_loss/weights_000.pth"
    dronet = getModel((200, 200), 3, 1, weights_path)
    dronet = dronet.eval().cuda()
    with torch.no_grad():
        testModel(dronet)
