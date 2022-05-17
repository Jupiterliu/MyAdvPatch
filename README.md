# Physical Adversarial Attack to Dronet
This repository is based on the Dronet, which is a uav navigation and collision avoidance model: https://github.com/uzh-rpg/rpg_public_dronet

Dronet is implemented by Keras/Tensorflow, but we here recurrence it by Pytorch in the file `DroNet_Pytorch`


# What you need
## Environments
Python == 3.6

Pytorch == 1.10.2

Torchvision == 0.11.3

OpenCV == 4.5.5.64

Tensorboard == 2.8.0

TensorboardX == 2.5

Also you need a device supported GPU.


## Datasets
Udacity dataset(Steering data): https://github.com/udacity/self-driving-car/tree/master/datasets/CH2

Collision data: http://rpg.ifi.uzh.ch/data/collision.zip

# Generating a patch
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment.

You can generate a physical patch by running:
```
python train_patch.py test2_random_scale
```
You also can evaluate the attacking effect of a generated patch by running:
```
...
```
