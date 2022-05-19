import sys
import numpy as np
import os

from DroNet_Pytorch import dronet_torch
from DroNet_Pytorch.load_datasets import DronetDataset
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm


def getModel(img_dims, image_mode, output_dim, weights_path):
    '''
      Initialize model.

      ## Arguments

        `img_dims`: Target image dimensions.

        `img_channels`: Target image channels.

        `output_dim`: Dimension of model output.

        `weights_path`: Path to pre-trained model.

      ## Returns
        `model`: the pytorch model
    '''
    if image_mode=="rgb":
        img_channels = 3
    else:
        img_channels = 1
    model = dronet_torch.DronetTorch(img_dims, img_channels, output_dim)
    # if weights path exists...
    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path))
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def trainModel(model: dronet_torch.DronetTorch, epochs, batch_size, steps_save, k, image_mode):
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
    epoch_loss = np.zeros((epochs, 2))
    for epoch in range(epochs):
        # scale the weights on the loss and the epoch number
        train_losses = []
        validation_losses = []
        # rip through the dataset
        other_val = (1 - torch.exp(torch.Tensor([-1 * model.decay * (epoch - 10)]))).float().to(model.device)
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

            loss = model.loss(k, steer_true, steer_pred, coll_true, coll_pred, use_old_loss = False)
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

            weights_path = os.path.join('saved_models', 'test4_GRAY_new_loss_500', f'weights_{epoch:03d}.pth')
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
        epoch_loss[epoch, 0] = train_loss
        epoch_loss[epoch, 1] = validation_loss
        # Save training and validation losses.
        np.savetxt(os.path.join('saved_models', 'test4_GRAY_new_loss_500', 'losses.txt'), epoch_loss, fmt="%f", delimiter=", ")
    # save final results
    #np.savetxt(os.path.join('saved_models', 'test3_RGB_old_loss_500', 'losses.txt'), epoch_loss)



if __name__ == "__main__":
    image_mode = "gray"
    dronet = getModel((200, 200), image_mode, 1, None)
    # print(dronet)
    trainModel(dronet, 500, 16, 1, 8, image_mode)
