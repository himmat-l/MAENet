import torch
import torch.nn as nn
import math

# backbone_layers = []
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)
        print('x:', x.shape)
        x = self.layer1(x)
        print('layer1:', x.shape)
        x = self.layer2(x)
        # backbone_layers.append(x)
        print('layer2:', x.shape)
        x = self.layer3(x)
        # backbone_layers.append(x)
        print('layer3:', x.shape)
        x = self.layer4(x)
        # backbone_layers.append(x)
        print('layer4:', x.shape)

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
        self.conv1 = self.conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(out_channel, out_channel)
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

    def conv3x3(self, in_channel, out_channel, stride=1):
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                        padding=1, bias=False)


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

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152():
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model


if __name__ == '__main__':
    in_batch, in_h, in_w = 4, 480, 640
    rgb = torch.randn(in_batch, 3, in_h, in_w)
    # resnet34 = ResNet(BasicBlock, [3,4,6,3])
    # resnet152 = ResNet(Bottleneck, [3, 8, 36, 3])
    resnet50 = resnet50()
    ressult = resnet50(rgb)
    print('out：', ressult.shape)




