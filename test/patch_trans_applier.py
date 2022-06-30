from scipy import ndimage
from numpy import *
import matplotlib.pyplot as plt
import pylab


def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def random_pad(nums, sums):
    a = random.sample(range(0, sums), k=1)
    a.append(0)
    a.append(sums)


def image_pad(image, target_size):
    iw, ih = image.size
    w, h = target_size
    new_image = Image.new("RGB", target_size, (0, 0, 0))


a = transforms.ToPILImage('RGB')(adv_patch)
a = a.resize((100, 100))
plt.subplot(3, 3, 1)
plt.imshow(a)

b = transforms.ToPILImage('RGB')(img_batch[0, :, :, :])
plt.subplot(3, 3, 2)
plt.imshow(b)

a_10 = a.rotate(8.234234, expand=1)
width, height = a_10.size
a_10_tensor = transforms.ToTensor()(a_10)
plt.subplot(3, 3, 3)
plt.imshow(a_10)
# a_10.save("/root/Python_Program_Remote/MyAdvPatch/1.png")
# plt.show()

paras = [1, 1, 1, 1, 1, 1, 0, 0]
# coeffs = find_coeffs([(100,100),(100,100),(100,100),(100,100)],[(100,100),(100,100),(100,100),(100,100)])
coeffs_inv = find_coeffs(
    [(700, 732), (869, 754), (906, 916), (712, 906)],
    [(867, 652), (1020, 580), (1206, 666), (1057, 757)]
)
a_affine = a.transform((100, 100), Image.AFFINE, (0, 0, 0, 1, 1, 1), fillcolor="black")
b_affine = b.transform((1200, 800), Image.PERSPECTIVE, coeffs_inv, fillcolor="black")
plt.subplot(3, 3, 4)
plt.imshow(b_affine)

b.paste(a_10, (100, 100))
plt.subplot(3, 3, 5)
plt.imshow(b)
plt.show()
# pylab.show()