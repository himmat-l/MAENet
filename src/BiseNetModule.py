import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride, padding):
	"3x3 convolution with padding"
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
	                 padding=padding, bias=False)


# 深度图提取分支(BiseNet中的Spatial Path)。输入为普通深度图或HHA编码后的深度图
class SpatialPath(nn.Module):
	def __init__(self, in_channels):
		super(SpatialPath, self).__init__()
		self.conv1 = conv3x3(in_channels, 64, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(64, 128, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = conv3x3(128, 256, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.relu3 = nn.ReLU(inplace=True)

	def forward(self, input):
		x = self.relu1(self.bn1(self.conv1(input)))
		x = self.relu2(self.bn2(self.conv2(x)))
		x = self.relu3(self.bn3(self.conv3(x)))
		return x


class ARMBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ARMBlock, self).__init__()
		self.in_channels = in_channels
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.bn = nn.BatchNorm2d(out_channels)
		self.sigmoid = nn.Sigmoid()
		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

	def forward(self, input):
		x = self.avgpool(input)
		assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
		x = self.conv(x)
		x = self.bn(x)
		x = self.sigmoid(x)
		x = torch.mul(input, x)
		return x


class FFMBlock(nn.Module):
	def __init__(self, in_channels, num_classes):
		super(FFMBlock, self).__init__()
		self.in_channels = in_channels
		self.conv1 = conv3x3(self.in_channels, num_classes, stride=1)
		self.bn = nn.BatchNorm2d(num_classes)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
		self.relu3 = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()
		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

	def forward(self, input1, input2):
		x = torch.cat((input1, input2), dim=1)
		assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
		feature = self.conv1(x)
		feature = self.bn(feature)
		feature = self.relu1(feature)

		x = self.avgpool(feature)
		x = self.relu2(self.conv2(x))
		x = self.relu3(self.conv3(x))
		x = torch.mul(x, feature)
		x = torch.add(x, feature)