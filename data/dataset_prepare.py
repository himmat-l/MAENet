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

imgpath = './SUNRGBD/SUNRGBD/SUNRGBD'
SUNRGBDMeta_dir = './SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'
SUNRGBD2Dseg_dir = './SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
labeltxt = '/home/liuxiaohui/MAENet/data/labels/label.txt'
imagepath = './images'
labelpath = './labels'
depthpath = './depth'
visualpath = './visual'

SUNRGBD_Label = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter',
                 'blinds','desk','shelves','curtain','dresser','pillow','mirror','floor_mat','clothes','ceiling','books',
                 'fridge','tv','paper','towel','shower_curtain','box','whiteboard','person','night_stand','toilet','sink',
                 'lamp','bathtub','bag']
# 将mat格式的文件转换为img
# path: [imagepath, labelpath, visualpath]
def sunrgbd_prepare(path):
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
    print('SUNRGBDMeta', SUNRGBDMeta.shape)
    seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']
    print('seglabel', seglabel.shape)
    # classlabels
    seg37list = SUNRGBD2Dseg['seg37list']
    for i in range(seg37list.size):
        classstring = np.array(SUNRGBD2Dseg[seg37list[i][0]]).tostring().decode('utf-8')
        classstring = classstring.replace("\x00", "")
        # print(classstring)
        labels.append(classstring)
        # print('labels',labels)
    with open(labeltxt, 'w') as f:
        content = ','.join(labels)
        f.write(content)

    for i, meta in tqdm(enumerate(SUNRGBDMeta)):
        # print(i)
        meta_dir = '/'.join(meta.rgbpath.split('/')[:-2])
        real_dir = meta_dir.split('/n/fs/sun3d/data/SUNRGBD/')[1]
        rgb_path = os.path.join(real_dir, 'image/' + meta.rgbname)

        # rgbimage
        srcname = osp.join(imgpath, rgb_path)
        t = "sun_{}".format(i)
        dstname = osp.join(imagepath, t)
        shutil.copy(srcname, dstname)
        rgbimg = Image.open(srcname)

        # labelimage
        label = np.array(
            SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0)). \
            astype(np.uint8)
        labelname = osp.join(labelpath, t.replace(".jpg", ".txt"))
        np.savetxt(labelname, label, fmt='%d')
        labelname = osp.join(labelpath, t.replace(".jpg", ".png"))
        labelimg = Image.fromarray(label, 'L')
        labelimg.save(labelname)

        # debug show
        # plt.subplot(1, 2, 1)
        # plt.imshow(rgbimg)
        # plt.subplot(1, 2, 2)
        # plt.imshow(labelimg)
        # plt.show()

        # visualimage
        visualname = osp.join(visualpath, t.replace(".jpg", ".png"))
        visualimg = Image.fromarray(label, "P")
        palette = bin_colormap  # long palette of 768 items
        visualimg.putpalette(palette)
        visualimg.save(visualname, format='PNG')


# 将npy格式的文件转换为img
def npy_to_img():
    arr = np.load('/home/liuxiaohui/MAENet/data/NYUDv2/labels/1.npy')
    print(arr.shape)
    output_name = os.path.splitext(os.path.basename('/home/liuxiaohui/MAENet/data/NYUDv2/labels/1.npy'))[0]
    plt.imsave(os.path.join('/home/liuxiaohui/MAENet/data/NYUDv2/labels/', "{}.png".format(output_name)), arr, cmap='plasma')

# 从文件夹中随机抽取一部分文件放在另一个文件夹
def moveFile(fileDir):
    refDir = '/home/liuxiaohui/MAENet/data/sunrgbd/label37/val/'
    pathDir = os.listdir(refDir)    #取图片的原始路径
    print(pathDir)
    filenumber = len(pathDir)
    print(filenumber)
    tarDir = '/home/liuxiaohui/MAENet/data/sunrgbd/depth/val/'
    # rate = 0.4    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    # picknumber = int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    # sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    # print(sample)
    for name in pathDir:
        shutil.move(fileDir+name, tarDir+name)
        # shutil.move(fileDir+'img-'+(name.split('0', 2)[2]).split('.')[0]+'.jpg', tarDir+'img-'+(name.split('0', 2)[2]).split('.')[0]+'.jpg')
    return

if __name__ == '__main__':
    # path_1 = [imagepath, labelpath, visualpath]
    # sunrgbd_prepare(path_1)
    moveFile('/home/liuxiaohui/MAENet/data/sunrgbd/depth/test/')
    # dir = '00000001.png'
    # print('/home/liuxiaohui/MAENet/data/sunrgbd/label37/test/' +'img-' + (dir.split('0',2)[2]).split('.')[0])







