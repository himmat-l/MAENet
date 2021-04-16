import torch
from torch import nn
from src.backbones import resnet, mobilenet

def conv3x3(in_channels, out_channels, stride, padding=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
	                 padding=padding, bias=False)


# 深度图提取分支(BiseNet中的Spatial Path)。输入为普通深度图或HHA编码后的深度图
# input[batch, channels, in_h, in_w]——》[batch, 64, in_h/2, in_w/2]
# ——》[batch, 128, in_h/4, in_w/4]——》[batch, 256, in_h/8, in_w/8]
class SpatialPath(nn.Module):
	def __init__(self, in_channels):
		super(SpatialPath, self).__init__()
		self.conv1 = conv3x3(in_channels, 64, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(64, 256, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = conv3x3(256, 512, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(512)
		self.relu3 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu1(self.bn1(self.conv1(x)))
		# print(x.shape)
		x1 = self.relu2(self.bn2(self.conv2(x)))
		# print(x1.shape)
		x2 = self.relu3(self.bn3(self.conv3(x1)))
		# print(x2.shape)
		return x1, x2


class SpatialPathwithDW(nn.Module):
	def __init__(self, in_channels):
		super(SpatialPathwithDW, self).__init__()
		self.dw1 = self._conv_dw(in_channels, 64, 2)
		self.dw2 = self._conv_dw(64, 256, 2)
		self.dw3 = self._conv_dw(256, 512, 2)

	def forward(self, x):
		x = self.dw1(x)
		# print(x.shape)
		x1 = self.dw2(x)
		# print(x1.shape)
		x2 = self.dw3(x1)
		# print(x2.shape)
		return x1, x2

	def _conv_dw(self, in_channels, out_channels, stride):
		return nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
			          bias=False),
			nn.BatchNorm2d(in_channels),
			nn.ReLU(),
			nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(),
		)

class ARMBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ARMBlock, self).__init__()
		self.in_channels = in_channels
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.bn = nn.BatchNorm2d(out_channels)
		self.sigmoid = nn.Sigmoid()
		self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

	def forward(self, x):
		out = self.avgpool(x)
		# print('out', out.shape)
		assert self.in_channels == out.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
		out = self.conv(out)
		out = self.bn(out)
		# print('out', out.shape)
		out = self.sigmoid(out)
		out = torch.mul(x, out)
		return out


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

class BiseNet(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18'):
		super(BiseNet, self).__init__()
		self.depth_path = SpatialPath(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18' or arch == 'resnet18dilated':
			orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(256, 256)
			self.arm_module2 = ARMBlock(512, 512)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(896, num_classes)
			self.gap = nn.AdaptiveAvgPool2d(1)
		elif arch == 'resnet50' or arch == 'resnet50dilated':
			orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(3584, num_classes)
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


	def forward(self, x):
		_, depth_out = self.depth_path(x)
		# print('depth_out1', depth_out1.shape, '\ndepth_out2', depth_out2.shape)
		rgb = self.relu(self.bn1(self.conv1(x)))
		# print('rgb', rgb.shape)
		rgb = self.maxpool(rgb)
		rgb = self.layer1(rgb)
		rgb = self.layer2(rgb)
		# print('rgb', rgb.shape)
		rgb1 = self.layer3(rgb)
		rgb2 = self.layer4(rgb1)
		tail = self.gap(rgb2)
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
			rgb1_sup = torch.nn.functional.interpolate(rgb1_sup, size=x.size()[-2:], mode='bilinear')
			rgb2_sup = torch.nn.functional.interpolate(rgb2_sup, size=x.size()[-2:], mode='bilinear')


		# output of feature fusion module
		result = self.ffm_module(depth_out, rgb_out)

		# upsampling
		result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
		result = self.conv(result)

		if self.training:
			return result, rgb1_sup, rgb2_sup
		return result


if __name__=="__main__":
	batch_size, in_h, in_w = 4, 30, 40
	in_rgb = torch.randn(batch_size, 256, in_h, in_w)
	in_depth = torch.randn(batch_size, 1, 480, 640)
	net = SpatialPath(1)
	# net = ARMBlock(256, 256)
	# print(net.parameters())
	# result = net(in_rgb, in_depth)
	result = net(in_depth)
	print(result.shape)

