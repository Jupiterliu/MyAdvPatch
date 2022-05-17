import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np

labels_path = "/root/Python_Program_Remote/MyAdvPatch/datasets/collision/collision_dataset/training/Ch2_002/steering_training.txt"
labels_path = "/root/Python_Program_Remote/MyAdvPatch/datasets/collision/collision_dataset/testing/Ch2_001_HMB_3/steering_testing.txt"
exp_arr = np.loadtxt(labels_path, delimiter=',', usecols=0)

plt.plot(exp_arr)
plt.show()