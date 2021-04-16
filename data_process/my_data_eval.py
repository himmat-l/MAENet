import numpy as np
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import cv2
import random
import torchvision
import torch
from matplotlib import image as mpimg
import os
from scipy import misc

image_h = 480
image_w = 640
train_file = '/home/liuxiaohui/MAENet/data/sunrgbd/train37.txt'
test_file = '/home/liuxiaohui/MAENet/data/sunrgbd/test37.txt'


def make_dataset_fromlst(listfilename):
    """
    My Data list format:
    imagepath seglabelpath depthpath HHApath
    """
    images = []
    depths = []
    with open(listfilename, 'r') as f:
        content = f.readlines()
        # print('content:',content)
        # print('content_split:',content[0].strip().split(' '))
        for x in content:
            if x != '\n':
                imgname, depthname = x.strip().split(' ')
                images += [imgname]
                depths += [depthname]
    return {'images': images, 'depths': depths}


class ReadData(Dataset):
    def __init__(self, transform=None, data_dir=None):
        self.data_dir = data_dir
        self.transform = transform
        result = make_dataset_fromlst(data_dir)
        if 'train' in self.data_dir:
            self.img_dir_train = result['images']
            self.depth_dir_train = result['depths']
        else:
            self.img_dir_test = result['images']
            self.depth_dir_test = result['depths']

    def __len__(self):
        if 'train' in self.data_dir:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if 'train' in self.data_dir:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test

        # 当图像格式为npy
        # label = np.load(label_dir[idx])
        # depth = np.load(depth_dir[idx])
        # image = np.load(img_dir[idx])
        # 当图像格式为jpg或png   uint8   uint16
        depth = misc.imread(os.path.join('/home/liuxiaohui/MAENet/data/MyData', depth_dir[idx]))/10000
        image = mpimg.imread(os.path.join('/home/liuxiaohui/MAENet/data/MyData', img_dir[idx]))

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)
        # observe = torch.max(sample['label'])

        return sample

class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['image']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'image': img_new, 'depth': sample['depth']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth}

# 随机缩放
class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']

        # random.uniform(x,y):随机生成x，y之间的一个实数
        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth}

# 随机裁剪
class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w]}

# 随机翻转
class RandomFlip(object):
    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()

        return {'image': image, 'depth': depth}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        # depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
        #                                          std=[0.9932836506164299])(depth)
        sample['image'] = image
        # sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']

        # Generate different label scales
        # label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
        #                                   order=0, mode='reflect', preserve_range=True)
        # label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
        #                                   order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float()
                # 'label2': torch.from_numpy(label2).float(),
                # 'label3': torch.from_numpy(label3).float(),
                # 'label4': torch.from_numpy(label4).float(),
                # 'label5': torch.from_numpy(label5).float()
                }

if __name__ == '__main__':
    # label_1 = np.load('/home/liuxiaohui/MAENet/data/NYUDv2/depths/1.npy') #(480,640)
    # label_2 = mpimg.imread(os.path.join('/home/liuxiaohui/MAENet/data/sunrgbd', 'depth/train/00002790.png'))
    # print(label_1.shape)
    # print(label_2.shape)
    weight = np.load('/home/liuxiaohui/MAENet/data/sunrgbd/sunrgbd_classes_weights.npy')
    np.savetxt('/home/liuxiaohui/MAENet/data/sunrgbd_classes_weights.txt', weight, fmt='%.8f')
    print(weight)
    # np.savetxt('/home/liuxiaohui/MAENet/data/NYUDv2/labels/1.txt', label_1, fmt='%s', newline='\n')
    # print(label[:10][:10])


