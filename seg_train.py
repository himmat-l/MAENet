import sys

sys.path.append('./')
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1 ,2'
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torchstat import stat
import unittest
import inspect

from src.MAENet import MAENet
from data_process import data_eval
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from gpu_mem_track import  MemTracker

from torchsummary import summary, summary_string

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 参数定义
parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
args = parser.parse_args()

# 初始权重, len(nyuv2_frq)=40
nyuv2_frq = []
weight_path = './data/NYUDv2/nyuv2_40class_weight.txt'

with open(weight_path, 'r') as f:
    context = f.readlines()

for x in context[1:]:
    x = x.strip().strip('\ufeff')
    # 初始化权重
    nyuv2_frq.append(float(x))

device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

image_h = 480
image_w = 640
# 训练函数
def train():
    # 准备数据集
    train_data = data_eval.ReadNpy(transform=transforms.Compose([data_eval.scaleNorm(),
                                                                 data_eval.RandomScale((1.0, 1.4)),
                                                                 data_eval.RandomHSV((0.9, 1.1),
                                                                                     (0.9, 1.1),
                                                                                     (25, 25)),
                                                                 data_eval.RandomCrop(image_h, image_w),
                                                                 data_eval.RandomFlip(),
                                                                 data_eval.ToTensor(),
                                                                 data_eval.Normalize()]),
                                   data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)
    # data.keys: ['image', 'depth', 'label', 'label2', 'label3', 'label4', 'label5'] (train)
    data = iter(train_loader).next()
    print('data:', data['image'].shape)
    if args.last_ckpt:
        model = MAENet(num_classes=40, model_url=model_urls['resnet50'], pretrained=False)
    else:
        model = MAENet(num_classes=40, model_url=model_urls['resnet50'], pretrained=True)

    model = model.to(device)
    model.train()
    # cal_param(model, data)
    criteon = nn.CrossEntropyLoss()

    for epoch in range(int(args.start_epoch), args.epochs):
        for batch_idx, data in enumerate(train_loader):
            image = data['image'].to(device)
            depth = data['depth'].to(device)
            target_scales = [data[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            pred_scales = model(image, depth)
            print('pred_scales', pred_scales[0].shape)
            print('target_scales', target_scales[0].shape)


# 计算模型参数量
class torchsummaryTests(unittest.TestCase):
    def test_multiple_input(self):
        in_batch, in_h, in_w = 4, 480, 640
        in_rgb = (3,480,640)
        in_dep = (1,480,640)
        # initialblock = InitialBlock(64, is_attention=False)
        net = MAENet(num_classes=37, model_url=model_urls['resnet50'], use_psp=True)
        total_params, trainable_params = summary(
            net, [in_rgb, in_dep], device='cpu')
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)


if __name__ == '__main__':
    unittest.main(buffer=True)
