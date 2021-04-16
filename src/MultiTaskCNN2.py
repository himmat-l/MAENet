import torch
from torch import nn

from .BiseNetModule1 import SpatialPathwithDW, SpatialPath,FFMBlock, ARMBlock
from .backbones.models1 import ASPP, DetectPathWithYOLO, ASPP4812
from src.backbones import resnet, mobilenet


class MultiTaskCNN(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN, self).__init__()
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
			self.ffm_module = FFMBlock(3584, num_classes)
			if use_aspp:
				self.gap = ASPP(2048, 2048)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
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
		# print('rgb', rgb.shape)
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


# ----------改进---------------------#
# 将ResNet中的7×7卷积换成3个3×3卷积
# -----------------------------------#
class MultiTaskCNN_3(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN_3, self).__init__()
		self.depth_path = SpatialPath(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18_3':
			orig_resnet = resnet.__dict__['resnet18_3'](pretrained=pretrained)
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
		elif arch == 'resnet50_3':
			orig_resnet = resnet.__dict__['resnet50_3'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(3584, num_classes)
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
		self.relu1 = orig_resnet.relu1
		self.conv2 = orig_resnet.conv2
		self.bn2 = orig_resnet.bn2
		self.relu2 = orig_resnet.relu2
		self.conv3 = orig_resnet.conv3
		self.bn3 = orig_resnet.bn3
		self.relu3 = orig_resnet.relu3
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
		rgb = self.relu1(self.bn1(self.conv1(x)))
		print('rgb1', rgb.shape)
		rgb = self.relu2(self.bn2(self.conv2(rgb)))
		print('rgb2', rgb.shape)
		rgb = self.relu3(self.bn3(self.conv3(rgb)))
		print('rgb3', rgb.shape)
		# print('rgb', rgb.shape)
		rgb = self.maxpool(rgb)
		rgb = self.layer1(rgb)
		# print('rgb', rgb.shape)
		fuse = depth_out1 + rgb
		rgb = self.layer2(fuse)
		# print('rgb', rgb.shape)
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

# ----------改进----------------#
# 加入ACNet中的ACM通道注意力模块
# ------------------------------#
class MultiTaskCNN_Atten(nn.Module):
	def  __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN_Atten, self).__init__()
		self.depth_path = SpatialPathwithDW(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18' or arch == 'resnet18dilated' or arch == 'resnet34':
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
			self.atten_rgb1 = self.channel_attention(64)
			self.atten_rgb2 = self.channel_attention(128)
			self.atten_depth1 = self.channel_attention(64)
			self.atten_depth2 = self.channel_attention(128)
		elif arch == 'resnet50' or arch == 'resnet50dilated':
			orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			# 512+1024+2048
			self.ffm_module = FFMBlock(3584, num_classes)
			if use_aspp:
				self.gap = ASPP(2048, 2048)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
			self.atten_rgb1 = self.channel_attention(256)
			self.atten_rgb2 = self.channel_attention(512)
			self.atten_depth1 = self.channel_attention(256)
			self.atten_depth2 = self.channel_attention(512)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)
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
		# 如果使用resnet18，要将depth_path中的通道数改为64,128
		depth_out1, depth_out2 = self.depth_path(depth)
		atten_depth1 = self.atten_depth1(depth_out1)
		atten_depth2 = self.atten_depth2(depth_out2)
		# print('depth1',depth_out1.shape, atten_depth1.shape)
		rgb = self.relu(self.bn1(self.conv1(x)))
		# print('rgb0', rgb.shape)
		rgb = self.maxpool(rgb)
		# print('rgb1', rgb.shape)
		rgb = self.layer1(rgb)
		# print('rgb2', rgb.shape)
		atten_rgb = self.atten_rgb1(rgb)
		# print('rgb3', rgb.shape, atten_rgb.shape)
		fuse = rgb.mul(atten_rgb) + depth_out1.mul(atten_depth1)
		rgb = self.layer2(fuse)
		atten_rgb = self.atten_rgb2(rgb)
		fuse = rgb.mul(atten_rgb) + depth_out2.mul(atten_depth2)
		# fuse = depth_out2 + rgb
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


# ------------改进-------------- #
# 加入MR、ACM
# ------------------------------ #
class MultiTaskCNN_Atten3(nn.Module):
	def __init__(self, num_classes, depth_channel=1, pretrained=False, arch='resnet18', use_aspp=False):
		super(MultiTaskCNN_Atten3, self).__init__()
		self.depth_path = SpatialPathwithDW(depth_channel)
		# self.rgb_path = ContextPath(arch=arch, pretrained=pretrained, use_aspp=use_aspp)
		arch = arch.lower()
		orig_resnet = None
		if arch == 'resnet18_3':
			orig_resnet = resnet.__dict__['resnet18_3'](pretrained=pretrained)
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
			self.atten_rgb1 = self.channel_attention(64)
			self.atten_rgb2 = self.channel_attention(128)
			self.atten_depth1 = self.channel_attention(64)
			self.atten_depth2 = self.channel_attention(128)
		elif arch == 'resnet50_3':
			orig_resnet = resnet.__dict__['resnet50_3'](pretrained=pretrained)
			self.arm_module1 = ARMBlock(1024, 1024)
			self.arm_module2 = ARMBlock(2048, 2048)
			# supervision block
			self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
			self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
			# build feature fusion module
			self.ffm_module = FFMBlock(3584, num_classes)
			if use_aspp:
				self.gap = ASPP(2048, 2048)
			else:
				self.gap = nn.AdaptiveAvgPool2d(1)
			self.atten_rgb1 = self.channel_attention(256)
			self.atten_rgb2 = self.channel_attention(512)
			self.atten_depth1 = self.channel_attention(256)
			self.atten_depth2 = self.channel_attention(512)
		else:
			print('Error: unspport context_path network \n')
		# build final convolution
		self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
		# take pretrained resnet
		self.conv1 = orig_resnet.conv1
		self.bn1 = orig_resnet.bn1
		self.relu1 = orig_resnet.relu1
		self.conv2 = orig_resnet.conv2
		self.bn2 = orig_resnet.bn2
		self.relu2 = orig_resnet.relu2
		self.conv3 = orig_resnet.conv3
		self.bn3 = orig_resnet.bn3
		self.relu3 = orig_resnet.relu3
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
		atten_depth1 = self.atten_depth1(depth_out1)
		atten_depth2 = self.atten_depth2(depth_out2)
		# print('depth_out1', depth_out1.shape, '\ndepth_out2', depth_out2.shape)
		rgb = self.relu1(self.bn1(self.conv1(x)))
		# print('rgb1', rgb.shape)
		rgb = self.relu2(self.bn2(self.conv2(rgb)))
		# print('rgb2', rgb.shape)
		rgb = self.relu3(self.bn3(self.conv3(rgb)))
		# print('rgb3', rgb.shape)
		# print('rgb', rgb.shape)
		rgb = self.maxpool(rgb)
		rgb = self.layer1(rgb)
		# print('rgb2', rgb.shape)
		atten_rgb = self.atten_rgb1(rgb)
		# print('rgb3', rgb.shape, atten_rgb.shape)
		fuse = rgb.mul(atten_rgb) + depth_out1.mul(atten_depth1)
		rgb = self.layer2(fuse)
		atten_rgb = self.atten_rgb2(rgb)
		fuse = rgb.mul(atten_rgb) + depth_out2.mul(atten_depth2)
		# fuse = depth_out2 + rgb
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
	out_dep = torch.randn(4, 1, in_h, in_w)
	net = MultiTaskCNN_DA(38, depth_channel=1, arch='resnet18')(in_rgb, out_dep)
	print(net[0].shape, net[1].shape, net[2].shape)
	# net = DetectCNN(arch='resnet18')
	# out = net(in_rgb)
	# print("o1", o1.shape, "\no2", o2.shape)
	# summary(net, (418, 418), 4)







