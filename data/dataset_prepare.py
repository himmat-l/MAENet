import os
import os.path as osp
import shutil
from PIL import Image
from scipy.io import loadmat
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import h5py
import scipy

imgpath = './SUNRGBD'
SUNRGBDMeta_dir = './SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'
SUNRGBD2Dseg_dir = './SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
labeltxt = "./label.txt"
imagepath = './images'
labelpath = './labels'
depthpath = './depth'
visualpath = './visual'

# 将mat格式的文件转换为img
def mat_to_img(path):
	for p in path:
		if not osp.exists(p):
			os.makedirs(p)
	bin_colormap = np.random.randint(0, 255, (256, 3))  # 可视化的颜色
	bin_colormap = bin_colormap.astype(np.uint8)

	labels = []
	processed = []
	# load the data from the matlab file
	SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_dir, mode='r', libver='latest')
	SUNRGBDMeta = scipy.io.loadmat(SUNRGBDMeta_dir, squeeze_me=True,
	                               struct_as_record=False)['SUNRGBDMeta']
	seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']
	# classlabels
	seg37list = SUNRGBD2Dseg['seg37list']
	for i in range(seg37list.size):
		classstring = np.array(SUNRGBD2Dseg[seg37list[i][0]]).tostring().decode('utf-8')
		classstring = classstring.replace("\x00", "")
		print(classstring)
		labels.append(classstring)

# 将npy格式的文件转换为img
def npy_to_img():
	arr = np.load('/home/liuxiaohui/MAENet/data/NYUDv2/labels/1.npy')
	print(arr.shape)
	output_name = os.path.splitext(os.path.basename('/home/liuxiaohui/MAENet/data/NYUDv2/labels/1.npy'))[0]
	plt.imsave(os.path.join('/home/liuxiaohui/MAENet/data/NYUDv2/labels/', "{}.png".format(output_name)), arr, cmap='plasma')


if __name__ == '__main__':
	npy_to_img()







