from PIL import Image  # pillow 基本上已经是Python平台上图像处理标准库了。他的功能很强大，但API却很简单实用。
from io import BytesIO
import requests
import matplotlib.pyplot as plt
'''
常有方法
Image.format 识别图像格式或来源，如果图像不是从文件读取的，值是None
Image.mode 图像的色彩模式 L灰度图像 RGB彩色图像 CMYK出版图像
Image.size 识别图像的宽高 单位为px，返回值是元组类型
im.info 输出与图像相关的数据的信息，返回字典类型
Image.open(img_path) 通过文件路径打开图像
Image.open(BytesIO(img_bytes)) 通过二进制文件流打开图像
Image.new(mode,size,color) 使用给定的模式、大小和颜色创建新图像
Image.show() 展示图像，会弹出一个展示窗口
Image.resise((size,size)) 调整图像宽高
Image.rotate(angle) 旋转图像
Image.save(filename,format) 保存图像，filename 图像名称或路径 format 保存模式，默认为RGB
img1.paste(img2, (10,10)) 将图像img2粘贴到图像img1上，并指定位置，默认为(0,0) 表示在左上点粘贴
'''


img1_url = 'https://nbjice.oss-cn-hangzhou.aliyuncs.com/10-sansheng/202001/9a594e5c-2dca-11ea-a72d-00163e09d976.jpg'
img2_url = 'https://nbjice.oss-cn-hangzhou.aliyuncs.com/10-sansheng/201912/e84203ee-255e-11ea-a72d-00163e09d976.png'

# 通过二进制流读取图片
img1=Image.open(BytesIO(requests.get(img1_url).content))
# 改变图像尺寸
img1 = img1.resize((640, 360))
img2=Image.open(BytesIO(requests.get(img2_url).content))
img2 = img2.resize((180, 180))
# 合并两个图像，img2 放到 img1上面 并指定坐标
img1.paste(img2, (430, 90))
# 展示图像
plt.figure()
plt.imshow(img1)
plt.show()
# 将文件保存为二进制文件流
img1_bytes = BytesIO()  # 创建一个二进制对象 并将图像内容写入到二进制对象中 获得一个二进制图像文件类型
# 以二进制文件类型保存图像，并指定图片类型 当然还可以直接保存为文件 img1.save('img1.png')
img1.save('.\img1.png')
# img1.save(img1_bytes, format="PNG")
