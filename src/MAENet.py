import sys
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
# 在服务器上运行程序时要把.去掉
from .backbones import resnet
# from .backbones.resnet import backbone_layers
# from ssd_module.ssd_layers import Detect, PriorBox, L2Norm
from utils.config import Config
sys.path.append('.')


model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class MAENet(nn.Module):
	def __init__(self, num_classes, model_url, is_attention=True, use_psp=True, pretrained=False):
		super(MAENet, self).__init__()

		self.model_urls = model_url
		self.initial_block = InitialBlock(64, is_attention)
		self.backbone = resnet.resnet50()
		self.seg_head = SemanticSegBranch(2048, num_classes, use_psp)
		# self.detect_head = ObjDetectBranch(num_classes)

		if pretrained:
			self._load_resnet_pretrained()

	def forward(self, rgb, depth):
		# out_initial.shape: [4, 64, 240, 320]
		out = self.initial_block(rgb, depth)
		# print('out', out.shape)
		out = self.backbone(out)
		# out_seg.shape: [4, 37, 480, 640]
		out_seg = self.seg_head(out)
		# out_detect.shape: [4, 37, 200, 5](phase_train=False), ([4, 28630, 4], [4, 28630, 37], [28630, 4])(phase_train=True)
		# out_detect = self.detect_head()
		# print('out_detect：', out_detect[0].shape, out_detect[1].shape, out_detect[2].shape)
		return out_seg

	def _load_resnet_pretrained(self):
		pretrain_dict = model_zoo.load_url(self.model_urls)
		model_dict = {}
		state_dict = self.state_dict()
		# print('state_dict:', state_dict.keys())
		for k, v in pretrain_dict.items():
			if k.startswith('conv1'):
				model_dict['initial_block.' + k[:]] = v
				model_dict[k.replace('conv1', 'initial_block.conv1_d')] = torch.mean(v, 1).view_as(
					state_dict[k.replace('conv1', 'initial_block.conv1_d')])
			elif k.startswith('bn1'):
				model_dict['initial_block.' + k[:]] = v
				model_dict[k.replace('bn1', 'initial_block.bn1_d')] = v
			elif k.startswith('layer'):
				model_dict['backbone.' + k[:]] = v

		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

	# print(state_dict.keys())

# RGB图与深度图融合的方式：
# 1）直接堆叠成四个通道，但是利用堆叠方式融合深度信息对室内语义分割的精确度的贡献是有限的
# 2）分支融合，利用两个相同的卷积神经网络分别提取两者的特征，并不断进行融合，最后输出融合的结果，
#    一般RGB分支作为主分支，depth分支作为次分支，特征融合的过程一般将两者的特征提取 结果进行拼接或叠加
class InitialBlock(nn.Module):
	def __init__(self, out_channel, is_attention=True):
		super().__init__()
		self.is_attention = is_attention
		self.conv1 = nn.Conv2d(3, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel)
		self.conv1_d = nn.Conv2d(1, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1_d = nn.BatchNorm2d(out_channel)
		self.relu = nn.ReLU(inplace=True)
		self.atten_rgb = self.channel_attention(out_channel)
		self.atten_depth = self.channel_attention(out_channel)

	def forward(self, r, d):
		# rgb输入
		out_rgb = self.conv1(r)
		out_rgb = self.bn1(out_rgb)
		out_rgb = self.relu(out_rgb)
		# depth输入
		out_depth = self.conv1_d(d)
		out_depth = self.bn1_d(out_depth)
		out_depth = self.relu(out_depth)
		#attention一般不会用在encoder里面，一般是用在decoder下。
		if self.is_attention:
			atten_rgb = self.atten_rgb(out_rgb)
			atten_depth = self.atten_depth(out_depth)
			fuse = out_rgb.mul(atten_rgb) + out_depth.mul(atten_depth)
		else:
			# 这部分的相加我觉得太浅层了，只有第一层相加得到的信息不会太丰富
			fuse = out_rgb + out_depth
		return fuse

	def channel_attention(self, num_channel):
		pool = nn.AdaptiveAvgPool2d(1)
		conv = nn.Conv2d(num_channel, num_channel, kernel_size=1)
		activation = nn.Sigmoid()
		return nn.Sequential(*[pool, conv, activation])


# 使用SSD检测分支
class ObjDetectBranch(nn.Module):
	def __init__(self, num_classes, phase_train=True, mbox=[4, 6, 6, 6, 4, 4]):
		super(ObjDetectBranch, self).__init__()
		self.mbox = mbox
		self.num_classes = num_classes
		self.phase_train = phase_train
		self.L2Norm = L2Norm(512, 20)
		self.cfg = Config
		self.basenet = backbone_layers
		self.extra_layer1 = self._make_layer(2048, 256)
		self.extra_layer2 = self._make_layer(512, 128)
		self.extra_layer3 = self._make_layer(256, 128)
		self.priorbox = PriorBox(self.cfg)
		with torch.no_grad():
			self.priors = self.priorbox()
		if self.phase_train == False:
			self.softmax = nn.Softmax(dim=-1)
			self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
	def forward(self):
		loc_layers = []
		conf_layers = []
		sources = []
		loc = []
		conf = []
		sources.append(self.L2Norm(backbone_layers[0]))
		sources += backbone_layers[1:]
		out = self.extra_layer1(backbone_layers[-1])
		print('out1:', out.shape)
		sources.append(out)
		out = self.extra_layer2(out)
		print('out2:', out.shape)
		sources.append(out)
		out = self.extra_layer3(out)
		print('out3:', out.shape)
		sources.append(out)
		for k, v in enumerate(sources):
			loc_layers += [nn.Conv2d(v.shape[1],
                                 self.mbox[k] * 4, kernel_size=3, padding=1)]
			conf_layers += [nn.Conv2d(v.shape[1],
                                 self.mbox[k] * self.num_classes, kernel_size=3, padding=1)]
		for (x, l, c) in zip(sources, loc_layers, conf_layers):
			loc.append(l(x).permute(0, 2, 3, 1).contiguous())
			conf.append(c(x).permute(0, 2, 3, 1).contiguous())
		# print('loc', loc[0].shape, loc[1].shape, loc[2].shape, loc[3].shape)
		# print('conf', conf[0].shape, conf[1].shape, conf[2].shape, conf[3].shape)
		# loc.shape:[4,114520], cof.shape:[4,1059310]
		loc = torch.cat([o.view(o.shape[0], -1) for o in loc], 1)
		conf = torch.cat([o.view(o.shape[0], -1) for o in conf], 1)
		if self.phase_train:
			output = (
				loc.view(loc.size(0), -1, 4),
				conf.view(conf.size(0), -1, self.num_classes),
				self.priors
			)
		else:
			output = self.detect(
				loc.view(loc.size(0), -1, 4),  # loc preds
				self.softmax(conf.view(conf.size(0), -1,
				                       self.num_classes)),  # conf preds
				self.priors
			)
		return output

	def _make_layer(self, in_channel, out_channel):
		return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
							nn.BatchNorm2d(out_channel),
							nn.ReLU(inplace=True),
							nn.Conv2d(out_channel, out_channel*2, kernel_size=3, stride=2),
							nn.BatchNorm2d(out_channel*2),
							nn.ReLU(inplace=True))


# 集成多种decoder模式
class SemanticSegBranch(nn.Module):
	def __init__(self, in_channel, num_classes, use_psp, sizes=(1, 2, 3, 6)):
		super(SemanticSegBranch, self).__init__()
		# transblock = TransBasicBlock
		self.use_psp = use_psp
		# PSP相关结构
		self.psp = PSPModule(in_channel, 512, sizes)
		self.drop_1 = nn.Dropout2d(p=0.3)
		# if self.use_psp:
		# 	self.up_1 = self._make_transpose(transblock, 1024, 512, 6, stride=2)
		# else:
		# 	self.up_1 = self._make_transpose(transblock, 2048, 512, 6, stride=2)
		# self.up_2 = self._make_transpose(transblock, 512, 256, 4, stride=2)
		# self.up_3 = self._make_transpose(transblock, 256, 64, 4, stride=2)
		# self.up_4 = self._make_transpose(transblock, 64, 64, 4, stride=2)
		# self.final = self._make_transpose(transblock, 64,  num_classes, 4, stride=2)
		if self.use_psp:
			self.up_1 = PSPUpsample(512, 512)
		else:
			self.up_1 = PSPUpsample(in_channel, 512)
		self.up_2 = PSPUpsample(512, 256)
		self.up_3 = PSPUpsample(256, 64)
		self.up_4 = PSPUpsample(64, 64)
		self.final = PSPUpsample(64, num_classes)
		self.out5_conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)
		self.out4_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
		self.out3_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
		self.out2_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
		self.drop_2 = nn.Dropout2d(p=0.15)

	def forward(self, p):
		if self.use_psp:
			# p = self.psp(x)
			p = self.psp(p)
			# print('psp:', p.shape)
			p = self.drop_1(p)
		# else:
		# 	p = self.drop_1(x) #一般不会在刚encoder完后就加上dropout层
		p = self.up_1(p)
		# print('up1:', p.shape)
		#可能要加上encoder结构里的那几层
		if self.training:
			# out5 = self.out5_conv(p)
			out5 = F.softmax(p)
		# p = self.drop_2(p)
		p = self.up_2(p)
		# print('up2:', p.shape)
		if self.training:
			# out4 = self.out4_conv(p)
			out4 = F.sigmoid(p)
		# p = self.drop_2(p)
		p = self.up_3(p)
		# print('up3:', p.shape)
		if self.training:
			# out3 = self.out3_conv(p)
			out3 = 10 * p
		# p = self.drop_2(p)
		p = self.up_4(p)
		if self.training:
			# out2 = self.out2_conv(p)
			out2 = 10 * p
		# p = self.drop_2(p)
		final = self.final(p)
		# print('final:', final.shape)

		if self.training:
			return final, out2, out3, out4, out5

		return final

	def _make_transpose(self, block, in_channel, out_channel, blocks, stride=1):
		upsample = None
		if stride != 1:
			upsample = nn.Sequential(
				nn.ConvTranspose2d(in_channel, out_channel,
				                   kernel_size=2, stride=stride,
				                   padding=0, bias=False),
				nn.BatchNorm2d(out_channel),
			)
		elif in_channel != out_channel:
			upsample = nn.Sequential(
				nn.Conv2d(in_channel, out_channel,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channel),
			)

		layers = []

		for i in range(1, blocks):
			layers.append(block(in_channel, in_channel))

		layers.append(block(in_channel, out_channel, stride, upsample))
		self.inplanes = out_channel

		return nn.Sequential(*layers)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = self.conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = self.conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

    def conv3x3(self, in_planes, out_planes, stride=1):
	    "3x3 convolution with padding"
	    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
	                     padding=1, bias=False)

class PSPModule(nn.Module):
	def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
		super().__init__()
		self.stages = []
		self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
		self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
		self.relu = nn.ReLU()

	def _make_stage(self, features, size):
		prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
		conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
		return nn.Sequential(prior, conv)

	def forward(self, feats):
		h, w = feats.size(2), feats.size(3)
		priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
		bottle = self.bottleneck(torch.cat(priors, 1))
		return self.relu(bottle)


class PSPUpsample(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, 3, padding=1),
			# nn.BatchNorm2d(out_channel),
			nn.PReLU())

	def forward(self, x):
		h, w = 2 * x.size(2), 2 * x.size(3)
		p = F.interpolate(input=x, size=(h, w), mode='bilinear')
		return self.conv(p)


if __name__ == "__main__":
	in_batch, in_h, in_w = 2, 480, 640
	in_rgb = torch.randn(in_batch, 3, in_h, in_w)
	out_dep = torch.randn(in_batch, 1, in_h, in_w)
	# initialblock = InitialBlock(64, is_attention=False)
	net = MAENet(num_classes=37, model_url=model_urls['resnet50'],use_psp=True)
	# net._load_resnet_pretrained()
	out_model = net(in_rgb, out_dep)
	print('out', len(out_model))

