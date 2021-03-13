import sys
import os
import argparse
import torch
import imageio
import skimage.transform
import torchvision.transforms as transforms
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import datetime
import cv2
import torch.optim
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from data_process import data_eval
from src.MultiTaskCNN import MultiTaskCNN
import utils.utils as utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output', default='./result/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=38, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]


def inference():
	model = MultiTaskCNN(38, depth_channel=1, pretrained=False, arch='resnet18')
	load_ckpt(model, None, args.last_ckpt, device)
	model.eval()
	model = model.to(device)

	val_data = data_eval.ReadData(transform=torchvision.transforms.Compose([data_eval.scaleNorm(),
	                                                                       data_eval.ToTensor(),
	                                                                       Normalize()]),
	                             data_dir=args.data_dir
	                             )
	val_loader = DataLoader(val_data, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

	acc_meter = AverageMeter()
	intersection_meter = AverageMeter()
	union_meter = AverageMeter()
	a_meter = AverageMeter()
	b_meter = AverageMeter()

	with torch.no_grad():
		for batch_idx, sample in enumerate(val_loader):
			origin_image = sample['origin_image'].numpy()
			origin_depth = sample['origin_depth'].numpy()
			image = sample['image'].to(device)
			depth = sample['depth'].to(device)
			label = sample['label'].numpy()

			with torch.no_grad():
				pred = model(image, depth)
			output = torch.max(pred, 1)[1] + 1
			output = output.squeeze(0).cpu().numpy()

			acc, pix = accuracy(output, label)
			intersection, union = intersectionAndUnion(output, label, args.num_class)
			acc_meter.update(acc, pix)
			a_m, b_m = macc(output, label, args.num_class)
			intersection_meter.update(intersection)
			union_meter.update(union)
			a_meter.update(a_m)
			b_meter.update(b_m)
			if batch_idx % 50 == 0:
				print('[{}] iter {}, accuracy: {}'
				      .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				              batch_idx, acc))
				if args.visualize:
					visualize_result(origin_image, origin_depth, label - 1, output - 1, batch_idx, args)

	iou = intersection_meter.sum / (union_meter.sum + 1e-10)
	for i, _iou in enumerate(iou):
		print('class [{}], IoU: {}'.format(i, _iou))
	# mAcc:Prediction和Ground Truth对应位置的“分类”准确率（每个像素）
	mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
	print(mAcc.mean())
	print('[Eval Summary]:')
	print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
	      .format(iou.mean(), acc_meter.average() * 100))


class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        origin_image = image.clone()
        origin_depth = depth.clone()
        image = image / 255
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        sample['origin_image'] = origin_image
        sample['origin_depth'] = origin_depth
        sample['image'] = image
        sample['depth'] = depth

        return sample

def visualize_result(img, depth, label, preds, info, args):
    # segmentation
    img_list = []
    img = img.squeeze(0).transpose(0, 2, 1)
    img_list.append(img)
    dep = depth.squeeze(0).squeeze(0)
    dep = (dep*255/dep.max()).astype(np.uint8)
    dep = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep = dep.transpose(2,1,0)
    img_list.append(dep)
    seg_color = utils.color_label_eval(label)
    img_list.append(seg_color)
    # prediction
    pred_color = utils.color_label_eval(preds)
    img_list.append(pred_color)
    # aggregate images and save
    # im_vis = np.concatenate((img, dep, seg_color, pred_color),
    #                         axis=1).astype(np.uint8)
    # im_vis = im_vis.transpose(2, 1, 0)
    for i,im in enumerate(img_list):
        img_name = str(info)+str(i)
        cv2.imwrite(os.path.join(args.output,
                img_name+'.png'),im.astype(np.uint8).transpose(2,1,0))
    # img_name = str(info)
    # print('write check: ', im_vis.dtype)
    # cv2.imwrite(os.path.join(args.output,
    #             img_name+'.png'), im_vis)

if __name__ == '__main__':
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    inference()