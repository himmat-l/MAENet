import torch
import torch.nn as nn
import math
from utils.utils import load_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# 'resnet18_3'、'resnet50_3'、'resnet101_3'为将7*7卷积核换成3个3*3卷积核后的预训练权重
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18_3': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50_3': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101_3': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# input: [batch, channels, in_h, in_w]
#                       resnet50--152                                           resnet18--34
# Conv:                 [batch, 64, in_h/2, in_w/2]                             [batch, 64, in_h/2, in_w/2]
# max_pool+layer1:      [batch, 256, in_h/4, in_w/4]                            [batch, 64, in_h/4, in_w/4]
# layer2:               [batch, 512, in_h/8, in_w/8]                            [batch, 128, in_h/8, in_w/8]
# layer3:               [batch, 1024, in_h/16, in_w/16]                         [batch, 256, in_h/16, in_w/16]
# layer4:               [batch, 2048, in_h/32, in_w/32]                         [batch, 512, in_h/32, in_w/32]
class ResNet(nn.Module):
    def __init__(self, block, layers, pretrained=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            print('----successfully initialize weights----')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('x:', x.shape)
        x = self.maxpool(x)
        x = self.layer1(x)
        # print('layer1:', x.shape)
        x = self.layer2(x)
        # print('layer2:', x.shape)
        x = self.layer3(x)
        # print('layer3:', x.shape)
        x = self.layer4(x)
        # print('layer4:', x.shape)

        return x

    def _make_layer(self, block, out_channel, blocks, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers.append(block(self.inplanes, out_channel, stride, downsample))
        self.inplanes = out_channel * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, out_channel))

        return nn.Sequential(*layers)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 加载预训练模型
def _load_pretrained_dict(model, urls):
    pretrained_dict = load_url(urls)
    net_state_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_1)
    model.load_state_dict(net_state_dict)
    print('----successfully load pretrained model----')

# pretrained: 如果没有自己训练的模型，就导入resnet的预训练模型
def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained is True:
        _load_pretrained_dict(model, model_urls['resnet18'])
    return model


def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained_dict(model, model_urls['resnet34'])
    return model


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained_dict(model, model_urls['resnet50'])
    return model


def resnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        _load_pretrained_dict(model, model_urls['resnet101'])
    return model


def resnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        _load_pretrained_dict(model, model_urls['resnet152'])
    return model


if __name__ == '__main__':
    in_batch, in_h, in_w = 4, 480, 640
    rgb = torch.randn(in_batch, 3, in_h, in_w)
    # resnet34 = ResNet(BasicBlock, [3,4,6,3])
    # resnet152 = ResNet(Bottleneck, [3, 8, 36, 3])
    resnet18 = resnet18(pretrained=False)
    ressult = resnet18(rgb)
    print('out：', ressult.shape)
