import torch
from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.training_dir = "/root/Python_Program_Remote/MyAdvPatch/datasets_png/training"
        self.validation_dir = "/root/Python_Program_Remote/MyAdvPatch/datasets_png/validation"
        self.testing_dir = "/root/Python_Program_Remote/MyAdvPatch/datasets_png/testing"
        self.weightfile = "/root/Python_Program_Remote/MyAdvPatch/DroNet/saved_model/best_model_RGB/test8_weights_346.pth"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 200
        self.image_size = 200

        self.start_learning_rate = 0.03

        self.patch_name = 'Base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)  # default: patience=40
        self.max_tv = 0.165

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls

class HA(BaseConfig):
    """
    Generate a physical patch for conducting Hiding Attack.
    """

    def __init__(self):
        super().__init__()

        self.n_epochs = 100  # 70 is already good
        self.batch_size = 128  # defaults: 64
        self.k = 128   # hard-mining
        self.num_workers = 10
        self.beta = 10
        self.gamma = 1

        self.patch_size = 200
        self.image_size = 200
        self.image_mode = "rgb"  # "rgb" or "gray"

        self.is_save_temp = False

        self.is_targeted = True  # False0
        self.use_old_loss = True  # False or True
        self.steer_target = 0.
        self.coll_target = 0.

        self.attack_loss_weight = 1  # origin: 1
        self.nps_loss_weight = 0.01  # origin: 0.01
        self.tv_loss_weight = 2.5  # origin: 2.5

        self.start_learning_rate = 0.1  # reduce by 10 times

        self.patch_name = 'HidingAttack'

class YA(BaseConfig):
    """
    Generate a physical patch for conducting Yaw Attack.
    """

    def __init__(self):
        super().__init__()

        self.n_epochs = 100  # 70 is already good
        self.batch_size = 64
        self.k = 64   # hard-mining
        self.num_workers = 10
        self.beta = 40
        self.gamma = 1

        self.patch_size = 200
        self.image_size = 200
        self.image_mode = "rgb"  # "rgb" or "gray"

        self.is_save_temp = False

        self.is_targeted = True  # False0
        self.steer_target = 0.5
        self.coll_target = 0.
        self.use_old_loss = True  # False or True

        self.attack_loss_weight = 1  # origin: 1
        self.nps_loss_weight = 0.01  # origin: 0.01
        self.tv_loss_weight = 2.5  # origin: 2.5

        self.patch_name = 'YawAttack'


class CA(BaseConfig):
    """
    Generate a physical patch for conducting Collision Attack.
    """

    def __init__(self):
        super().__init__()

        self.n_epochs = 100  # 70 is already good
        self.batch_size = 128  # defaults: 64
        self.k = 64   # hard-mining
        self.num_workers = 10
        self.beta = 40
        self.gamma = 1

        self.patch_size = 200
        self.image_size = 200
        self.image_mode = "rgb"  # "rgb" or "gray"

        self.is_save_temp = False

        self.is_targeted = True  # False0
        self.steer_target = 0.
        self.coll_target = 1.
        self.use_old_loss = True  # False or True

        self.attack_loss_weight = 1  # origin: 1
        self.nps_loss_weight = 0.01  # origin: 0.01
        self.tv_loss_weight = 2.5  # origin: 2.5

        self.patch_name = 'CollisionAttack'


patch_configs = {
    "base": BaseConfig,
    "hiding_attack": HA,
    "yaw_attack": YA,
    "collision_attack": CA
}
