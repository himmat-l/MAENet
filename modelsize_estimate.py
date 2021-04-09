import torch
import torch.nn as nn
import numpy as np
from src.MultiTaskCNN import MultiTaskCNN
from src.yolo.yolov3_tiny import TinyYoloNetTail


def modelsize(model, features, type_size=4):
	para = sum([np.prod(list(p.size())) for p in model.parameters()])
	# print('Model {} : Number of params: {}'.format(model._get_name(), para))
	print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
	print(model.parameters())
	inputs = []
	for feature in features:
		feature_ = feature.clone()
		feature_.requires_grad_(requires_grad=False)
		inputs.append(feature_)

	mods = list(model.modules())
	out_sizes = []

	for i in range(1, len(mods)):
		m = mods[i]
		if isinstance(m, nn.ReLU):
			if m.inplace:
				continue
				if len(features) == 1:
					out = m(inputs[0])
			out_sizes.append(np.array(out.size()))
			inputs[0] = out

	total_nums = 0
	for i in range(len(out_sizes)):
		s = out_sizes[i]
		nums = np.prod(np.array(s))
		total_nums += nums

	# print('Model {} : Number of intermedite variables without backward: {}'.format(model._get_name(), total_nums))
	# print('Model {} : Number of intermedite variables with backward: {}'.format(model._get_name(), total_nums*2))
	print('Model {} : intermedite variables: {:3f} M (without backward)'
	      .format(model._get_name(), total_nums * type_size / 1000 / 1000))
	print('Model {} : intermedite variables: {:3f} M (with backward)'
	      .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


def conv3x3(in_channels, out_channels, stride, padding=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
	                 padding=padding, bias=False)


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

	def forward(self, x):
		# print('x', x.size(1), '\nin_channels', self.in_channels)
		assert self.in_channels == x.size(1), 'In FFMBlock, in_channels of ConvBlock should be {}'.format(x.size(1))
		feature = self.conv1(x)
		feature = self.bn(feature)
		feature = self.relu1(feature)

		x = self.avgpool(feature)
		x = self.relu2(self.conv2(x))
		x = self.relu3(self.conv3(x))
		x = torch.mul(x, feature)
		x = torch.add(x, feature)
		return x


if __name__ == "__main__":
	in_batch, in_h, in_w = 4, 60, 80
	in_rgb = torch.randn(in_batch, 3328, in_h, in_w)
	out_dep = torch.randn(in_batch, 1, in_h, in_w)
	in1 = torch.randn(24, 512, 14, 14)
	in2 = torch.randn(24, 256, 28, 28)
	# model = MultiTaskCNN(38, depth_channel=1, pretrained=False, arch='resnet18')
	# net = FFMBlock(3328, 38)
	net = TinyYoloNetTail()
	modelsize(net, (in1, in2))
