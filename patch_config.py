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
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 200
        self.image_size = 200

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls

class HA(BaseConfig):
    """
    Generate a physical patch that attack the Dronet model.
    """

    def __init__(self):
        super().__init__()

        self.n_epochs = 120  # 70 is already good
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
        self.steer_target = 0.
        self.coll_target = 0.
        self.use_old_loss = True  # False or True

        self.attack_loss_weight = 1  # origin: 1
        self.nps_loss_weight = 0.01  # origin: 0.01
        self.tv_loss_weight = 2.5  # origin: 2.5

        self.patch_name = 'PhysicalAttack'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

class YA(BaseConfig):
    """
    Generate a physical patch that attack the Dronet model.
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

        self.patch_name = 'PhysicalAttack'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

class CA(BaseConfig):
    """
    Generate a physical patch that attack the Dronet model.
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
        self.steer_target = 0.
        self.coll_target = 1.
        self.use_old_loss = True  # False or True

        self.attack_loss_weight = 1  # origin: 1
        self.nps_loss_weight = 0.01  # origin: 0.01
        self.tv_loss_weight = 2.5  # origin: 2.5

        self.patch_name = 'PhysicalAttack'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

patch_configs = {
    "base": BaseConfig,
    "hiding_attack": HA,
    "yaw_attack": YA,
    "collision_attack": CA
}
