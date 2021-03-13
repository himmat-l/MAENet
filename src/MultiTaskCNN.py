import torch
from torch import nn

from .BiseNetModule import SpatialPath, FFMBlock, ARMBlock
from .backbones.models import ContextPath


class MultiTaskCNN(nn.Module):
	def __init__(self, num_classes, depth_channel=3, pretrained=False, arch='resnet18'):
		super(MultiTaskCNN, self).__init__()
		self.depth_path = SpatialPath(depth_channel)
		self.rgb_path = ContextPath.build_contextpath(arch=arch, pretrained=pretrained)
		if arch == 'resnet18' or 'resnet18dilated':
			self.arm_module1 = ARMBlock(256, 256)
			self.arm_module2 = ARMBlock(512, 512)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(1024, num_classes)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

	def forward(self, rgb, depth):
		depth_out = self.depth_path(depth)
		rgb1, rgb2, tail = self.rgb_path(rgb)
		# print('depth', depth_out.shape, '\nrgb1', rgb1.shape, '\nrgb2', rgb2.shape, '\ntail', tail.shape,)
		rgb1 = self.arm_module1(rgb1)
		# print('rgb1', rgb1.shape)
		rgb2 = self.arm_module2(rgb2)
		rgb2 = torch.mul(rgb2, tail)
		# upsampling
		rgb1 = torch.nn.functional.interpolate(rgb1, size=depth_out.size()[-2:], mode='bilinear')
		rgb2 = torch.nn.functional.interpolate(rgb2, size=depth_out.size()[-2:], mode='bilinear')
		rgb_out = torch.cat((rgb1, rgb2), dim=1)
		if self.training:
			rgb1_sup = self.supervision1(rgb1)
			rgb2_sup = self.supervision2(rgb2)
			rgb1_sup = torch.nn.functional.interpolate(rgb1_sup, size=rgb.size()[-2:], mode='bilinear')
			rgb2_sup = torch.nn.functional.interpolate(rgb2_sup, size=rgb.size()[-2:], mode='bilinear')

		# output of feature fusion module
		result = self.ffm_module(depth_out, rgb_out)

		# upsampling
		result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
		result = self.conv(result)

		if self.training:
			return result, rgb1_sup, rgb2_sup
		return result


if __name__ =='__main__':
	in_batch, in_h, in_w = 2, 480, 640
	in_rgb = torch.randn(in_batch, 3, in_h, in_w)
	out_dep = torch.randn(in_batch, 1, in_h, in_w)
	net = MultiTaskCNN(40, depth_channel=1)(in_rgb, out_dep)
	print(net[0].shape, net[1].shape, net[2].shape)






