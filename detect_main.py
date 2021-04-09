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

from src.MultiTaskCNN1 import DetectCNN
from utils.yoloLoss import yoloLoss
from src.yolo.net import vgg16_bn
from src.yolo.resnet_yolo import resnet50
from data.voc_dataset import yoloDataset
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
# training only
parser.add_argument('--annot-path', dest='annot_path', type=str, default=None,
                    help="TRAINING ONLY: The path to the file of the annotations for training.")
parser.add_argument('--no-augment', dest='data_augment', action='store_false',
                    help="TRAINING ONLY: use this option to turn off the data augmentation of the dataset."
                         "Currently only COCO dataset support data augmentation.")
args = parser.parse_args()

file_root = '/home/liuxiaohui/MAENet/data/pascal voc/JPEGImages/'
learning_rate = 0.001
num_epochs = 50
batch_size = 24
device = torch.device("cuda:1"if args.cuda and torch.cuda.is_available() else "cpu")
net = DetectCNN(arch='resnet50')
net = net.to(device)
net.train()
# print(net)
criterion = yoloLoss(7, 2, 5, 0.5)

params = []
params_dict = dict(net.named_parameters())
print(params_dict.keys())
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    params += [{'params': [value], 'lr': learning_rate}]

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 数据集
train_dataset = yoloDataset(root=file_root,list_file=['./data/pascalvoc/voc2012.txt','./data/pascalvoc/voc2007.txt'], train=True, transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
test_dataset = yoloDataset(root=file_root, list_file='./data/pascalvoc/voc2007test.txt', train=False, transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# print('the dataset has %d images' % (len(train_dataset)))
print('the train dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='lxh_py37')
best_test_loss = np.inf

for epoch in range(num_epochs):
    # net.train()
    if epoch == 30:
        learning_rate = 0.0001
    if epoch == 40:
        learning_rate = 0.00001

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))

    total_loss = 0.
    print(1)
    print(train_loader)
    for i, (images, target) in enumerate(train_loader):
        print(2)
        images = Variable(images)
        target = Variable(target)
        print(images)
        if args.cuda:
            images, target = images.to(device), target.to(device)
        pred = net(images)
        loss = criterion(pred, target)
        total_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0], total_loss / (i + 1)))
            num_iter += 1
            # vis.plot_train_val(loss_train=total_loss / (i + 1))

    # validation
    validation_loss = 0.0
    net.eval()
    for i, (images, target) in enumerate(test_loader):
        images = Variable(images, volatile=True)
        target = Variable(target, volatile=True)
        if args.cuda:
            images, target = images.to(device), target.to(device)

        pred = net(images)
        loss = criterion(pred, target)
        validation_loss += loss.data[0]
    validation_loss /= len(test_loader)
    # vis.plot_train_val(loss_val=validation_loss)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(), './models/detect/best.pth')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
    logfile.flush()
    torch.save(net.state_dict(), 'yolo.pth')

def train()
