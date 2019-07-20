# IMPORT NECESSARY LIBRARIES

import scipy
import scipy.ndimage
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
import matplotlib

# IMPORTING IMAGE USING SCIPY AND TAKING R,G,B COMPONENTS

a = matplotlib.pyplot.imread("Einstein.jpg")
a_np = np.array(a)
a_r = a_np[:,:,0]
a_g = a_np[:,:,1]
a_b = a_np[:,:,2]

fig = plt.figure()


def comp_2d(image_2d,numpc): # FUNCTION FOR RECONSTRUCTING 2D MATRIX USING PCA
	cov_mat = image_2d - np.mean(image_2d , axis = 1)
	eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat)) # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
	p = np.size(eig_vec, axis =1)
	idx = np.argsort(eig_val)
	idx = idx[::-1]
	eig_vec = eig_vec[:,idx]
	eig_val = eig_val[idx]
	# numpc = 100 # THIS IS NUMBER OF PRINCIPAL COMPONENTS, YOU CAN CHANGE IT AND SEE RESULTS
	if numpc <p or numpc >0:
		eig_vec = eig_vec[:, range(numpc)]
	score = np.dot(eig_vec.T, cov_mat)
	recon = np.dot(eig_vec, score) + np.mean(image_2d, axis = 1).T # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
	recon_img_mat = np.uint8(np.absolute(recon)) # TO CONTROL COMPLEX EIGENVALUES
	return recon_img_mat

i=0
for numpc in range(1,100,10):
	i+=1
	a_r_recon, a_g_recon, a_b_recon = comp_2d(a_r,numpc), comp_2d(a_g,numpc), comp_2d(a_b,numpc) # RECONSTRUCTING R,G,B COMPONENTS SEPARATELY
	recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon)) # COMBINING R.G,B COMPONENTS TO PRODUCE COLOR IMAGE
	recon_color_img = Image.fromarray(recon_color_img)
	recon_color_img.save('./img/'+str(i)+'.jpg')
	# x=[i for i in range(5)]
	# y=[i**2 for i in range(5)]
	# ax = fig.add_subplot(241+recon_color_img)
	# ax.scatter(x,y,c='r',marker=shape[recon_color_img])
	# ax.set_title('第'+str(j))




import PIL.Image as Image
import os
 
IMAGES_PATH = './img/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG']  # 图片格式
IMAGE_SIZE = 640  # 每张小图片的大小
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'final.jpg'  # 图片转换后的地址
 
# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
 
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")
 
# 定义图像拼接函数

def image_compose():
	t=0
	to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
	for y in range(1, IMAGE_ROW + 1):
		for x in range(1, IMAGE_COLUMN + 1):
			
			from_image = Image.open(IMAGES_PATH + image_names[9-t]).resize((IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
			to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
			t=t+1
	return to_image.save(IMAGE_SAVE_PATH) 
image_compose() 





