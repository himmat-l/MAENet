import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANCHORS, NUM_ANCHORS_PER_SCALE, NUM_CLASSES, NUM_ATTRIB, LAST_LAYER_DIM
Tensor = torch.Tensor


class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out


class TinyYoloLayer(nn.Module):

    def __init__(self, scale, stride):
        super(TinyYoloLayer, self).__init__()
        if scale == 'm':
            idx = (0, 1, 2)
        elif scale == 'l':
            idx = (3, 4, 5)
        else:
            idx = None
        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(num_batch,
                                NUM_ANCHORS_PER_SCALE,
                                NUM_ATTRIB,
                                num_grid,
                                num_grid).permute(0, 1, 3, 4, 2).contiguous().view(num_batch, -1, NUM_ATTRIB)
            return output_raw
        else:
            prediction_raw = x.view(num_batch,
                                    NUM_ANCHORS_PER_SCALE,
                                    NUM_ATTRIB,
                                    num_grid,
                                    num_grid).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            # Calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # Get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride # Center x
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride  # Center y
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view((num_batch, -1, 4)) #cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)  # Cls pred one-hot.

            output = torch.cat((bbox_pred, conf_pred, cls_pred), -1)
            return output


class TinyYoloNetTail(nn.Module):
    def __init__(self):
        super(TinyYoloNetTail, self).__init__()
        self.detect1 = TinyYoloLayer('l', 32)
        self.detect2 = TinyYoloLayer('m', 16)
        self.conv1 = ConvLayer(512, 256, 1)
        self.conv2 = ConvLayer(256, 512, 3)
        self.conv3 = nn.Conv2d(512, NUM_ANCHORS_PER_SCALE * (4 + 1 + NUM_CLASSES), 1, bias=True, padding=0)
        self.conv4 = ConvLayer(256, 128, 1)
        self.conv5 = ConvLayer(384, 256, 3)
        self.conv6 = nn.Conv2d(256, NUM_ANCHORS_PER_SCALE * (4 + 1 + NUM_CLASSES), 1, bias=True, padding=0)

    def forward(self, x1, x2):
        # print('x1', x1.shape, '\nx2', x2.shape)
        branch = self.conv1(x1)
        # print('branch', branch.shape)
        tmp = self.conv2(branch)
        # print('tmp', tmp.shape)
        tmp = self.conv3(tmp)
        # print('tmp', tmp.shape)
        out1 = self.detect1(tmp)
        # print('out1', out1.shape)
        tmp = self.conv4(branch)
        # print('tmp', tmp.shape)
        tmp = F.interpolate(tmp, scale_factor=2)
        # print('tmp', tmp.shape)
        tmp = torch.cat((tmp, x2), 1)
        print('cat tmp', tmp.shape)
        tmp = self.conv5(tmp)
        print('conv5', tmp.shape)
        tmp = self.conv6(tmp)
        print('conv6', tmp.shape)
        out2 = self.detect2(tmp)

        return out1, out2
