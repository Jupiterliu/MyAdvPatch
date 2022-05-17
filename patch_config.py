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
        self.patch_size = 300
        self.image_size = 416

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls

class PhysicalPatch(BaseConfig):
    """
    Generate a physical patch that attack the Dronet model.
    """

    def __init__(self):
        super().__init__()

        self.n_epochs = 500
        self.batch_size = 16
        self.k = 8   # hard-mining
        self.num_workers = 10

        self.patch_size = 100
        self.image_size = 200

        self.is_targeted = True  # False
        self.steer_target = 0.
        self.coll_target = 0.

        self.patch_name = 'PhysicalAttack'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "test2_random_scale": PhysicalPatch
}
