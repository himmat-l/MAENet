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

from src.MAENet import MAENet
from data_process import data_eval
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log

'''
NYUDv2_classes = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
           'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
           'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
           'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand',
           'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
'''
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

nyuv2_frq = []
weight_path = './data/NYUDv2/nyuv2_40class_weight.txt'

with open(weight_path, 'r') as f:
	context = f.readlines()

for x in context[1:]:
	x = x.strip().strip('\ufeff')
	# 初始化权重
	nyuv2_frq.append(float(x))
print('nyuv2_frq', len(nyuv2_frq))
# nyuv2_frq = torch.from_numpy(np.array(nyuv2_frq)).float()

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
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0")
# if torch.cuda.is_available() and args.cuda:
#     torch.cuda.set_device(0)
image_w = 640
image_h = 480


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
	global nyuv2_frq
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
	                          num_workers=args.workers, pin_memory=False)
	num_train = len(train_data)
	# 如果有已训练的checkpoint，则不使用预训练的backbone模型
	if args.last_ckpt:
		model = MAENet(num_classes=40, model_url=model_urls['resnet50'], pretrained=False)
	else:
		model = MAENet(num_classes=40, model_url=model_urls['resnet50'], pretrained=True)
	# 使用GPU的个数
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# nn.DataParallel(module, device_ids=None, output_device=None, dim=0):使用多块GPU进行计算
		model = nn.DataParallel(model)
	# 使用交叉熵损失函数
	# nyuv2_frq = nyuv2_frq.to(device)
	weight = (torch.from_numpy(np.array(nyuv2_frq))).to(device)
	CEL_weighted = utils.CrossEntropyLoss2d(weight=weight)
	model.train()
	model.to(device)
	CEL_weighted.cuda()
	# 使用SGD随机梯度下降，initial lr=2*10^(-3), weight_decay=10^(-4)(权重衰减，L2正则化), momentum=0.9
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
	                            momentum=args.momentum, weight_decay=args.weight_decay)
	global_step = 0
	# 如果有模型的训练权重，则获取global_step，start_epoch
	if args.last_ckpt:
		global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
	# 学习率下降规律，lr_decay_rate=0.8， lr_epoch_per_decay=100，scheduler用来自定义调整学习率
	lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
	scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
	# 记录数据在tensorboard中显示
	writer = SummaryWriter(args.summary_dir)
	epoch_size = num_train // args.batch_size

	# 开始训练
	for epoch in range(int(args.start_epoch), args.epochs):
		# with tqdm(total=epoch_size, desc=f'Train epoch {epoch +1} / {args.epochs}', postfix=dict,mininterval=0.3) as pbar:
		# optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,
		# 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
		scheduler.step(epoch)
		# local_count记录处理的图片数量
		local_count = 0
		last_count = 0
		end_time = time.time()
		# 每隔save_epoch_freq个epoch就保存一次权重
		if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
			save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
			          local_count, num_train)
		# 分批训练
		for batch_idx, sample in enumerate(train_loader):
			image = sample['image'].to(device)
			depth = sample['depth'].to(device)
			target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
			print(len(target_scales))
			# 梯度清零
			optimizer.zero_grad()
			pred_scales = model(image, depth)
			pred_scales = [pred_scales[s].to(device) for s in range(5)]
			print(len(pred_scales))
			# 计算loss值，CEL（CrossEntropyLoss）为交叉熵损失函数
			loss = CEL_weighted(pred_scales, target_scales)
			# loss反向传播
			loss.backward()
			# 对模型进行更新
			optimizer.step()
			local_count += image.data.shape[0]
			global_step += 1

			# 每迭代print_freq次就打印一次训练结果
			if global_step % args.print_freq == 0 or global_step == 1:
				time_inter = time.time() - end_time
				count_inter = local_count - last_count
				print_log(global_step, epoch, local_count, count_inter,
				          num_train, loss, time_inter)
				end_time = time.time()
				for name, param in model.named_parameters():
					writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
				grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
				writer.add_image('image', grid_image, global_step)
				grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
				writer.add_image('depth', grid_image, global_step)
				grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3,
				                       normalize=False,
				                       range=(0, 255))
				writer.add_image('Predicted label', grid_image, global_step)
				grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
				writer.add_image('Groundtruth label', grid_image, global_step)
				writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
				writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
				last_count = local_count
			# pbar.set_postfix(**{'loss': loss.data / (batch_idx + 1), 'lr': get_lr(optimizer)})
			# pbar.update(epoch_size)

	# save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
	#           0, num_train)

	print("Training completed ")


if __name__ =='__main__':
	train()
# 	2021/1/22 报错
#  File "/home/liuxiaohui/MAENet/utils/utils.py", line 87, in forward
# 	losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
# RuntimeError: CUDA error: an illegal memory access was encountered









