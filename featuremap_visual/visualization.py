import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import os

# 该函数创建保存特征图的文件目录,以网络层号命名文件夹，如feature\\1\\..文件夹中保存的是模型第二层的输出特征图
def mkdir(path):

    isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
    if not isExists: # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False

# 图像预处理函数，将图像转换成[224,224]大小,并进行Normalize，返回[1,3,224,224]的四维张量
def preprocess_image(cv2im, resize_im=True):

    # 在ImageNet100万张图像上计算得到的图像的均值和标准差，它会使图像像素值大小在[-2.7,2.1]之间，但是整体图像像素值的分布会是标准正态分布（均值为0，方差为1）
    # 之所以使用这种方法，是因为这是基于ImageNet的预训练VGG16对输入图像的要求
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 改变图像大小并进行Normalize
    if resize_im:
        cv2im = cv2.resize(cv2im, dsize=(224,224),interpolation=cv2.INTER_CUBIC)
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # 将[W,H,C]的次序改变为[C,W,H]

    for channel, _ in enumerate(im_as_arr): # 进行在ImageNet上预训练的VGG16要求的ImageNet输入图像的Normalize
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]

    # 转变为三维Tensor,[C,W,H]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    im_as_ten = im_as_ten.unsqueeze_(0) # 扩充为四维Tensor,变为[1,C,W,H]

    return im_as_ten # 返回处理好的[1,3,224,224]四维Tensor


class FeatureVisualization():

    def __init__(self,img_path,selected_layer):
        '''
        :param img_path:  输入图像的路径
        :param selected_layer: 待可视化的网络层的序号
        '''
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrained_model = models.vgg16(pretrained=True).features # 调用预训练好的vgg16模型

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img, resize_im=False)
        return img

    def get_feature(self):

        input=self.process_image() # 读取输入图像
        # 以下是关键代码：根据给定的层序号，返回该层的输出
        x = input
        for index, layer in enumerate(self.pretrained_model):
            x = layer(x) # 将输入给到模型各层，注意第一层的输出要作为第二层的输入，所以才会复用x
            # print('x:', x.shape,'\n','index:', index)

            if (index == self.selected_layer): # 如果模型各层的索引序号等于期望可视化的选定层号
                return x # 返回模型当前层的输出四维特征图

    def get_single_feature(self):
        features = self.get_feature() # 得到期望模型层的输出四维特征图
        return features

    def save_feature_to_img(self):

        features=self.get_single_feature() # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
        for i in range(features.shape[1]):
            feature = features[:, i, :, :] # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
            feature = feature.view(feature.shape[1], feature.shape[2]) # batch为1，所以可以直接view成二维张量
            feature = feature.data.numpy() # 转为numpy

            # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
            feature = (feature - np.amin(feature))/(np.amax(feature) - np.amin(feature) + 1e-5) # 注意要防止分母为0！
            feature = np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
            mkdir('/home/liuxiaohui/MAENet/featuremap_visual/HHA/' + str(self.selected_layer))  # 创建保存文件夹，以选定可视化层的序号命名
            if feature.shape[0]<1080:
                tmp_img = feature.copy()
                tmp_img = cv2.resize(tmp_img, (1920, 1080), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite('/home/liuxiaohui/MAENet/featuremap_visual/HHA/' + str(self.selected_layer) + '/' + str(i) + '_r'+'.png',tmp_img)



            cv2.imwrite('/home/liuxiaohui/MAENet/featuremap_visual/HHA/' + str(self.selected_layer) + '/' + str(i) + '.png',feature)  # 保存当前层输出的每个channel上的特征图为一张图像


if __name__=='__main__':


    for k in [1,3,4,6,8,9,11,13,15,18,20,22,23,25,27,29,30]: # k代表选定的可视化的层的序号
        myClass = FeatureVisualization('/home/liuxiaohui/MAENet/featuremap_visual/fullres/hha_complete_SUN.png', k) # 实例化类
        # myClass.get_feature()
        # print (myClass.pretrained_model)
        myClass.save_feature_to_img() # 开始可视化，并将特征图保存成图像

