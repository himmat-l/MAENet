import torch
from torch import nn

from .BiseNetModule1 import SpatialPathwithDW, FFMBlock, ARMBlock
from .backbones.models1 import ASPP, DetectPathWithYOLO, ASPP4812
from src.backbones import resnet, mobilenet


class MultiTaskCNN(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN, self).__init__()
		self.depth_path = SpatialPathwithDW(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18' or 'resnet18dilated':
			orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(256, 256)
			self.arm_module2 = ARMBlock(512, 512)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(896, num_classes)
			if use_aspp:
				self.gap = ASPP(512, 512)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
		elif arch == 'resnet50' or arch == 'resnet50dilated':
			orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(3328, num_classes)
			if use_aspp:
				self.gap = ASPP(2048, 2048)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
		# take pretrained resnet
		self.conv1 = orig_resnet.conv1
		self.bn1 = orig_resnet.bn1
		self.relu = orig_resnet.relu
		self.maxpool = orig_resnet.maxpool
		self.layer1 = orig_resnet.layer1
		self.layer2 = orig_resnet.layer2
		self.layer3 = orig_resnet.layer3
		self.layer4 = orig_resnet.layer4
		# for layer in [self.conv1, self.maxpool, self.layer1]
		# for param in self.conv1.parameters():
		# 	param.requires_grad = False


	def forward(self, x, depth):
		depth_out1, depth_out2 = self.depth_path(depth)
		# print('depth_out1', depth_out1.shape, '\ndepth_out2', depth_out2.shape)
		rgb = self.relu(self.bn1(self.conv1(x)))
		# print('rgb', rgb.shape)
		rgb = self.maxpool(rgb)
		rgb = self.layer1(rgb)
		# print('rgb', rgb.shape)
		fuse = depth_out1 + rgb
		rgb = self.layer2(fuse)
		fuse = depth_out2 + rgb
		rgb1 = self.layer3(fuse)
		rgb2 = self.layer4(rgb1)
		tail = self.gap(rgb2)
		rgb1 = self.arm_module1(rgb1)
		# print('rgb1', rgb1.shape)
		rgb2 = self.arm_module2(rgb2)
		rgb2 = torch.mul(rgb2, tail)
		# upsampling
		rgb1 = torch.nn.functional.interpolate(rgb1, size=depth_out2.size()[-2:], mode='bilinear')
		rgb2 = torch.nn.functional.interpolate(rgb2, size=depth_out2.size()[-2:], mode='bilinear')
		rgb_out = torch.cat((rgb1, rgb2), dim=1)
		if self.training:
			rgb1_sup = self.supervision1(rgb1)
			rgb2_sup = self.supervision2(rgb2)
			rgb1_sup = torch.nn.functional.interpolate(rgb1_sup, size=x.size()[-2:], mode='bilinear')
			rgb2_sup = torch.nn.functional.interpolate(rgb2_sup, size=x.size()[-2:], mode='bilinear')


		# output of feature fusion module
		result = self.ffm_module(depth_out2, rgb_out)

		# upsampling
		result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
		result = self.conv(result)

		if self.training:
			return result, rgb1_sup, rgb2_sup
		return result

	def channel_attention(self, num_channel, ablation=False):
		# add convolution here ACM通道注意力机制
		pool = nn.AdaptiveAvgPool2d(1)
		conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
		# bn = nn.BatchNorm2d(num_channel)
		activation = nn.Sigmoid()  # modify the activation function

		return nn.Sequential(*[pool, conv, activation])


class MultiTaskCNN_DA(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN_DA, self).__init__()
		self.depth_path = SpatialPathwithDW(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18' or 'resnet18dilated':
			orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(256, 256)
			self.arm_module2 = ARMBlock(512, 512)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(896, num_classes)
			if use_aspp:
				self.gap = ASPP4812(512, 512)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
		elif arch == 'resnet50' or arch == 'resnet50dilated':
			orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(3328, num_classes)
			if use_aspp:
				self.gap = ASPP4812(2048, 2048)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
		# take pretrained resnet
		self.conv1 = orig_resnet.conv1
		self.bn1 = orig_resnet.bn1
		self.relu = orig_resnet.relu
		self.maxpool = orig_resnet.maxpool
		self.layer1 = orig_resnet.layer1
		self.layer2 = orig_resnet.layer2
		self.layer3 = orig_resnet.layer3
		self.layer4 = orig_resnet.layer4


	def forward(self, x, depth):
		depth_out1, depth_out2 = self.depth_path(depth)
		# print('depth_out1', depth_out1.shape, '\ndepth_out2', depth_out2.shape)
		rgb = self.relu(self.bn1(self.conv1(x)))
		# print('rgb', rgb.shape)
		rgb = self.maxpool(rgb)
		rgb = self.layer1(rgb)
		# print('rgb', rgb.shape)
		fuse = depth_out1 + rgb
		rgb = self.layer2(fuse)
		fuse = depth_out2 + rgb
		rgb1 = self.layer3(fuse)
		rgb2 = self.layer4(rgb1)
		tail = self.gap(rgb2)
		rgb1 = self.arm_module1(rgb1)
		# print('rgb1', rgb1.shape)
		rgb2 = self.arm_module2(rgb2)
		rgb2 = torch.mul(rgb2, tail)
		# upsampling
		rgb1 = torch.nn.functional.interpolate(rgb1, size=depth_out2.size()[-2:], mode='bilinear')
		rgb2 = torch.nn.functional.interpolate(rgb2, size=depth_out2.size()[-2:], mode='bilinear')
		rgb_out = torch.cat((rgb1, rgb2), dim=1)
		if self.training:
			rgb1_sup = self.supervision1(rgb1)
			rgb2_sup = self.supervision2(rgb2)
			rgb1_sup = torch.nn.functional.interpolate(rgb1_sup, size=x.size()[-2:], mode='bilinear')
			rgb2_sup = torch.nn.functional.interpolate(rgb2_sup, size=x.size()[-2:], mode='bilinear')


		# output of feature fusion module
		result = self.ffm_module(depth_out2, rgb_out)

		# upsampling
		result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
		result = self.conv(result)

		if self.training:
			return result, rgb1_sup, rgb2_sup
		return result

	def channel_attention(self, num_channel, ablation=False):
		# add convolution here ACM通道注意力机制
		pool = nn.AdaptiveAvgPool2d(1)
		conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
		# bn = nn.BatchNorm2d(num_channel)
		activation = nn.Sigmoid()  # modify the activation function

		return nn.Sequential(*[pool, conv, activation])

class DetectCNN(nn.Module):
	def __init__(self, pretrained=False, arch='resnet18'):
		super(DetectCNN, self).__init__()
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18' or arch == 'resnet18dilated':
			orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
		elif arch == 'resnet50' or arch == 'resnet50dilated':
			orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
		else:
			print('Error: unspport context_path network \n')
		self.conv1 = orig_resnet.conv1
		self.bn1 = orig_resnet.bn1
		self.relu = orig_resnet.relu
		self.maxpool = orig_resnet.maxpool
		self.layer1 = orig_resnet.layer1
		self.layer2 = orig_resnet.layer2
		self.layer3 = orig_resnet.layer3
		self.layer4 = orig_resnet.layer4
		self.detect = DetectPathWithYOLO()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x1 = self.layer3(x)
		x2 = self.layer4(x1)
		x = self.detect(x2, x1)
		print('x', x.shape)
		return x

if __name__ =='__main__':
	from torchsummary import summary
	in_batch, in_h, in_w = 2, 416, 416
	in_rgb = torch.randn(4, 3, in_h, in_w)
	# out_dep = torch.randn(in_batch, 1, in_h, in_w)
	# net = MultiTaskCNN(40, depth_channel=1)(in_rgb, out_dep)
	# print(net[0].shape, net[1].shape, net[2].shape)
	net = DetectCNN(arch='resnet18')
	out = net(in_rgb)
	# print("o1", o1.shape, "\no2", o2.shape)
	# summary(net, (418, 418), 4)







