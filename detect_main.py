import sys
sys.path.append('./')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import argparse
from torchvision import models
import numpy as np
from tensorboardX import SummaryWriter
import os

from src.MultiTaskCNN1 import DetectCNN
from utils.yoloLoss import yoloLoss
from src.yolo.net import vgg16_bn
from src.yolo.resnet_yolo import resnet50
from utils.yoloDataloader import YoloDataset
from visualize import Visualizer

parser = argparse.ArgumentParser(description='object detection using tiny yolov3')
# train or test:
parser.add_argument('ACTION', type=str, help="'train' or 'test' the detector.")
parser.add_argument('--img-dir', dest='img_dir', type=str, default='../data/samples',
                        help="The path to the folder containing images to be detected or trained.")
parser.add_argument('--batch-size', dest='batch_size', type=int, default=4,
                        help="The number of sample in one batch during training or inference.")
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument("--img-size", dest='img_size', type=int, default=416,
                        help="The size of the image for training or inference.")
parser.add_argument('--annot-path', dest='annot_path', type=str, default=None,
                    help="TRAINING ONLY: The path to the file of the annotations for training.")
parser.add_argument('--no-augment', dest='data_augment', action='store_false',
                    help="TRAINING ONLY: use this option to turn off the data augmentation of the dataset."
                         "Currently only COCO dataset support data augmentation.")
parser.add_argument('--normalize', action='store_true', default=False,
                    help='whether to normolize loss')
# Yolov4的tricks应用
# mosaic 马赛克数据增强 True or False,实际测试时mosaic数据增强并不稳定，所以默认为False
parser.add_argument('--mosaic', action='store_true', default=False,
                    help='whether to use mosaic trick')
# Cosine_scheduler 余弦退火学习率 True or False
parser.add_argument('--cosine-lr', action='store_true', default=False,
                    help='whether to use mosaic trick')
# label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
parser.add_argument('--smooth-label', default=0, type=float,
                    metavar='SL', help='smooth label,always use 0.01,0.005')
args = parser.parse_args()


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

# 验证集与训练集的划分，通过val_split参数控制划分比例，默认为0.1，即训练集：验证集=9:1
def train_val(annotation_path):
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    return lines, num_train

# 输入图像的大小
input_shape = (448, 448)
# 是否对损失进行归一化，用于改变loss的大小。用于决定计算最终loss是除上batch_size还是除上正样本数量
normalize = False
# classes和anchor的路径，非常重要，训练前一定要修改classes_path，使其对应自己的数据集
anchors_path = './data/VOCdevkit/yolo_anchors.txt'
classes_path = './data/VOCdevkit/voc_classes.txt'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)
def train():
    # 记录数据在tensorboard中显示
    writer_loss = SummaryWriter(os.path.join(args.summary_dir, 'loss'))

    train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)


