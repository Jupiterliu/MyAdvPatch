import numpy as np
import matplotlib.pyplot as plt
import os

path = "/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL"
HA = np.loadtxt("/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/YA-HA-2.txt", delimiter=",")
name = "YA-HA-2"
# YA = np.loadtxt("/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/2022-07-16_22-55-30-YA-2/pred_labels/labels.txt", delimiter=",")

plt.figure(0, figsize=(20, 5))
min = 0
max_ha = len(HA)
x_ticks_ha = np.arange(min, max_ha + 1, 10)
plt.xticks(x_ticks_ha)

plt.subplot(2, 1, 1)
plt.title(name+ ": Steering Angle")
l1 = plt.plot(range(0, len(HA)), HA[:, 0] * 90, 'r--', label='Steering Angle')
plt.plot(range(0, len(HA)), HA[:, 0] * 90, "r.-") #o
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.xticks(x_ticks_ha)
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.title(name + ": Prob. Collision")
l2 = plt.plot(range(0, len(HA)), HA[:, 1], 'b--', label='Prob. Collision')
plt.plot(range(0, len(HA)), HA[:, 1], "b.-")  # v
plt.xlabel("Frame")
plt.ylabel("Prob.")
plt.xticks(x_ticks_ha)
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(path, name + '.png'))
plt.show()

# plt.figure(1, figsize=(20, 5))
# min = 0
# max_ya = len(YA)
# x_ticks_ya = np.arange(min, max_ya + 1, 10)
# plt.xticks(x_ticks_ya)
#
# plt.subplot(2, 1, 1)
# plt.title("YA-2: Steering Angle")
# l1 = plt.plot(range(0, len(YA)), YA[:, 0] * 90, 'r--', label='Steering Angle')
# plt.plot(range(0, len(YA)), YA[:, 0] * 90, "r.-")
# plt.xlabel("Frame")
# plt.ylabel("Angle")
# plt.xticks(x_ticks_ya)
# plt.grid()
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.title("YA-2: Prob. Collision")
# l2 = plt.plot(range(0, len(YA)), YA[:, 1], 'b--', label='Prob. Collision')
# plt.plot(range(0, len(YA)), YA[:, 1], "b.-")
# plt.xlabel("Frame")
# plt.ylabel("Prob.")
# plt.xticks(x_ticks_ya)
# plt.grid()
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(os.path.join("/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/2022-07-16_22-55-30-YA-2", 'YA-2.png'))
# plt.show()
