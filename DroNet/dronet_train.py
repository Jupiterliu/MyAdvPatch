import sys
import numpy as np
import os

from DroNet.dronet_model import getModel
from DroNet.dronet_model import DronetTorch
from DroNet.dronet_load_datasets import DronetDataset
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm


def trainModel(model: DronetTorch, epochs, batch_size, steps_save, k, image_mode, exp_dir, use_old_loss):
    '''
    trains the model.

    ## parameters:


    '''
    model.to(model.device)

    model.train()
    # create dataloaders for validation and training
    training_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'training',
                                        image_mode, augmentation=False, grayscale=False)
    validation_dataset = DronetDataset('/root/Python_Program_Remote/MyAdvPatch/datasets_png', 'validation',
                                        image_mode, augmentation=False, grayscale=False)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                        shuffle=True, num_workers=10)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                        shuffle=False, num_workers=10)

    epoch_length = len(training_dataloader)
    # adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # epoch_loss = np.zeros((1, 2))
    epoch_loss_train = np.array([])
    epoch_loss_val = np.array([])
    for epoch in range(epochs):
        # scale the weights on the loss and the epoch number
        train_losses = []
        validation_losses = []
        # rip through the dataset
        other_val = (1 - torch.exp(torch.Tensor([-1 * model.decay * (epoch - 50)]))).float().to(model.device)
        model.beta = torch.max(torch.Tensor([0]).float().to(model.device), other_val)
        for batch_idx, (img, steer_true, coll_true) in tqdm(enumerate(training_dataloader),
                                                            desc=f'Running epoch {epoch}', total=epoch_length):
            img_cuda = img.float().to(model.device)  # .float()
            steer_pred, coll_pred = model(img_cuda)
            # get loss, perform hard mining
            steer_true = steer_true.squeeze(1).to(model.device)
            coll_true = coll_true.squeeze(1).to(model.device)

            # for i in range(img.size(0)):
            #     Tensor = img[i, :, :, :]
            #     image = np.transpose(Tensor.detach().cpu().numpy(), (1, 2, 0))
            #     plt.imshow(image)
            #     plt.show()

            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred, use_old_loss)
            # backpropagate loss
            loss.backward()
            # optimizer step
            optimizer.step()
            # zero gradients to prevestepnt accumulation, for now
            optimizer.zero_grad()
            # print(f'loss: {loss.item()}')
            train_losses.append(loss.item())
            # print(f'Training Images Epoch {epoch}: {batch_idx * batch_size}')
        train_loss = np.array(train_losses).mean()
        print(f'training loss: {train_loss.item()}')
        if epoch % steps_save == 0:
            print('Saving results...')

            weights_path = os.path.join(exp_dir, "models", f'weights_{epoch:03d}.pth')
            torch.save(model.state_dict(), weights_path)
        # evaluate on validation set
        for batch_idx, (img, steer_true, coll_true) in tqdm(enumerate(validation_dataloader),
                                                            desc=f'Running epoch {epoch}',
                                                            total=len(validation_dataloader)):
            img_cuda = img.float().to(model.device)
            steer_pred, coll_pred = model(img_cuda)
            steer_true = steer_true.squeeze(1).to(model.device)
            coll_true = coll_true.squeeze(1).to(model.device)
            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred)
            validation_losses.append(loss.item())
            # print(f'Validation Images: {batch_idx * 4}')

        validation_loss = np.array(validation_losses).mean()
        print(f'validation loss: {validation_loss.item()}')
        # epoch_loss[epoch, 0] = train_loss
        # epoch_loss[epoch, 1] = validation_loss
        epoch_loss_train = np.append(epoch_loss_train, train_loss)
        epoch_loss_val = np.append(epoch_loss_val, validation_loss)
        epoch_loss = np.concatenate((epoch_loss_train.reshape(-1, 1), epoch_loss_val.reshape(-1, 1)), axis=1)
        # Save training and validation losses.
        np.savetxt(os.path.join(exp_dir, 'losses.txt'), epoch_loss, fmt="%f", delimiter=", ")
    # save final results
    #np.savetxt(os.path.join('saved_model', 'test3_RGB_old_loss_500', 'losses.txt'), epoch_loss)



if __name__ == "__main__":
    # Train a model with gray or rgb input
    image_mode = "rgb"

    # Path to save models
    exp_root = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model"
    exp_name = "test6_RGB_old_loss_beta50"
    folder = os.path.join(exp_root, exp_name)
    if not os.path.exists(os.path.join(folder, "models")):
        os.makedirs(os.path.join(folder, "models"))

    # Create and train a Dronet model
    dronet = getModel((200, 200), image_mode, 1, None)
    # print(dronet)
    # Old loss behavior great during training model
    trainModel(dronet, 500, 16, 1, 8, image_mode, folder, use_old_loss = True)
