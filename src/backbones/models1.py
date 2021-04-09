"""-MobileNetV2dilated; ResNet18/ResNet18dilated; ResNet50/ResNet50dilated; ResNet101/ResNet101dilated; HRNetV2 (W48)"""
import torch
import torch.nn as nn
from src.backbones import resnet, mobilenet
import math
import logging
import torch.nn.functional as F
from src.yolo.yolov3_tiny import TinyYoloNetTail


class DetectPath(nn.Module):
    def __init__(self):
        super(DetectPath, self).__init__()
        self.conv_end = nn.Conv2d(256, 30, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(30)
        self.layer = self._make_detnet_layer(in_channels=2048)
        self.avgpool = nn.AvgPool2d(2)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer(x)
        x = self.avgpool(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        print('x in models1', x.shape)
        x = self.sigmoid(x)  # 归一化到0-1
        print('x in models1', x.shape)
        # x = x.view(-1,7,7,30)
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,30)
        return x

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out

'''
tiny yolov3模型参考链接：
https://zhuanlan.zhihu.com/p/93809416
yolov3模型参考链接：
https://blog.csdn.net/litt1e/article/details/88907542
'''
class DetectPathWithYOLO(nn.Module):
    def __init__(self, nms=False, post=True):
        super(DetectPathWithYOLO, self).__init__()
        self.tiny_yolo_tail = TinyYoloNetTail()
        self.nms = nms
        self._post_process = post

    def forward(self, x1, x2):
        out1, out2 = self.tiny_yolo_tail(x1, x2)
        print('out1', out1.shape, '\nout2', out2.shape)
        out = torch.cat((out1, out2), 1)
        logging.debug("The dimension of the output before nms is {}".format(out.size()))
        return out

    def tiny_yolo_last_layers(self):
        _layers = [self.tiny_yolo_tail.conv6,
                   self.tiny_yolo_tail.conv3]
        return _layers

    def tiny_yolo_last_two_layers(self):
        _layers = self.yolo_last_layers() + \
                  [self.tiny_yolo_tail.conv5,
                   self.tiny_yolo_tail.conv2]
        return _layers

    def tiny_yolo_last_three_layers(self):
        _layers = self.yolo_last_two_layers() + \
                  [self.tiny_yolo_tail.conv4,
                   self.tiny_yolo_tail.conv1]
        return _layers

    def tiny_yolo_tail_layers(self):
        _layers = [self.yolo_tail]
        return _layers

    def tiny_yolo_last_n_layers(self, n):
        try:
            n = int(n)
        except ValueError:
            pass
        if n == 1:
            return self.tiny_yolo_last_layers()
        elif n == 2:
            return self.tiny_yolo_last_two_layers()
        elif n == 3:
            return self.tiny_yolo_last_three_layers()
        elif n == 'tail':
            return self.tiny_yolo_tail_layers()
        else:
            raise ValueError("n>3 not defined")




class ContextPath(nn.Module):
    def __init__(self, arch='resnet50', pretrained=False, use_aspp=False):
        super(ContextPath, self).__init__()
        self.net = None
        if arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            self.net = Resnet(orig_resnet)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            self.net = Resnet(orig_resnet)
        self.arch = arch.lower()
        self.gap = None
        if use_aspp and self.arch == 'resnet18':
            self.gap = ASPP(512, 512)
        elif use_aspp and self.arch == 'resnet50':
            self.gap = ASPP(2048, 2048)
        elif use_aspp and self.arch == 'mobilenetv2':
            self.gap = ASPP(1024, 1024)
        elif not use_aspp:
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            assert not self.gap,  'global average pooling should not be None'

    def forward(self, x, return_feature_maps=True):
        conv_out = self.net(x, return_feature_maps=return_feature_maps)
        tail = self.gap(conv_out[3])
        return conv_out, tail


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
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

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]



class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet featues
        self.features = orig_net.features[:-1] #[:-1]除了列表中最后一个元素，其余全取

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mult=1,
                 momentum=0.0003):
        super(ASPP, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          dilation=int(6 * mult), padding=int(6 * mult),
                          bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          dilation=int(12 * mult), padding=int(12 * mult),
                          bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(18 * mult), padding=int(18 * mult),
                               bias=False)
        self.aspp5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp2_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp3_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp4_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp5_bn = nn.BatchNorm2d(out_channels, momentum)
        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1,
                          bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.global_pooling(x)

        return x


class ASPP4812(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mult=1,
                 momentum=0.0003):
        super(ASPP4812, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          dilation=int(4 * mult), padding=int(4 * mult),
                          bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          dilation=int(8 * mult), padding=int(8 * mult),
                          bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               dilation=int(12* mult), padding=int(12 * mult),
                               bias=False)
        self.aspp5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp2_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp3_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp4_bn = nn.BatchNorm2d(out_channels, momentum)
        self.aspp5_bn = nn.BatchNorm2d(out_channels, momentum)
        self.conv2 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1,
                          bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.global_pooling(x)

        return x

if __name__ == '__main__':
    in_batch, in_h, in_w = 4, 480, 640
    rgb = torch.randn(in_batch, 3, in_h, in_w)
    # resnet = Resnet(resnet.__dict__['resnet18'](pretrained=False))
    # result = resnet(rgb)
    # aspp = ASPP(512, 512)
    # result = aspp(rgb)   #当输入为[4, 512, 15, 20]时，输出为[4, 256, 15, 20]
    # result = nn.AdaptiveAvgPool2d(1)(rgb)
    context_path = ContextPath(arch='resnet50', use_aspp=False)
    result, tail = context_path(rgb)  #return_feature_maps
    print(result[0].shape, result[1].shape, result[2].shape, result[3].shape, tail.shape)
    # print(result[0].shape, result[1].shape, result[2].shape)

