# Physical Adversarial Attack to Dronet
This repository is based on the Dronet, which is a uav navigation and collision avoidance model: https://github.com/uzh-rpg/rpg_public_dronet

Dronet is implemented by Keras/Tensorflow, but we here recurrence it by Pytorch.


# What you need
## Environments
Python 3.6

Pytorch 1.10.1

## Datasets
Udacity dataset(Steering data): https://github.com/udacity/self-driving-car/tree/master/datasets/CH2

Collision data: http://rpg.ifi.uzh.ch/data/collision.zip

# Generating a patch
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment.

You can generate a physical patch by running:
```
python train_patch.py test2_random_scale
```
