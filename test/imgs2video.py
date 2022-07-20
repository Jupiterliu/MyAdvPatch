import cv2
import os

image=cv2.imread('/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/2022-07-16_21-40-11-HA-1/raw_imgs/HA1.png')
# cv2.imshow("new window", image)
image_info=image.shape
height=image_info[0]
width=image_info[1]
size=(height,width)
print(size)
fps=30
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter('YA-2.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) #创建视频流对象-格式一

#video = cv2.VideoWriter('ss.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象-格式二

path = '/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/2022-07-16_22-55-30-YA-2/raw_imgs'
files = os.listdir(path)

"""
参数1 即将保存的文件路径
参数2 VideoWriter_fourcc为视频编解码器
    fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
    cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv 
    cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    文件名后缀为.mp4
参数3 为帧播放速率
参数4 (width,height)为视频帧大小

"""
for i in range(1, len(files)+1):
    file_name = '/root/Python_Program_Remote/MyAdvPatch/test/on_board_images/SITL/2022-07-16_22-55-30-YA-2/raw_imgs/YA' + str(i) + '.png'
    image=cv2.imread(file_name)
    video.write(image)
cv2.waitKey()