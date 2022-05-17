from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "/root/Python_Program_Remote/FASC/INRIAPerson/Train/pos"
        self.lab_dir = "/root/Python_Program_Remote/FASC/INRIAPerson/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)  # 通过降低网络的学习率来提高网络性能
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls

class PhysicalPatch(BaseConfig):
    """
    Generate a physical patch that attack the Dronet model.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 16
        self.patch_size = 100

        self.patch_name = 'PhysicalAttack'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "test2_random_scale": PhysicalPatch
}
