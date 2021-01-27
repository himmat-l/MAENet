from __future__ import division
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd import Variable
from math import sqrt as sqrt
from itertools import product as product
import numpy as np
from utils.box_utils import decode, nms
from utils.config import Config


class Detect(nn.Module):
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = Config['variance']

    def forward(self, loc_data, conf_data, prior_data):
        loc_data = loc_data.cpu()
        conf_data = conf_data.cpu()
        print('conf_data:', conf_data.shape)
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        print('conf_preds:', conf_preds.shape)
        # 对每一张图片进行处理
        for i in range(num):
            # 对先验框解码获得预测框
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # 对每一类进行非极大抑制
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                # print('scores:', scores.shape)
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # print('boxes:', boxes.shape)
                # 进行非极大抑制
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

'''
def PriorBox(cfg):
    mean = []
    for v in cfg['variance']:
        if v <= 0:
            raise ValueError('Variances must be greater than 0')
    for k, f in enumerate(cfg['feature_maps']):
        x, y = np.meshgrid(np.arange(f[0]), np.arange(f[1]))
        x = x.reshape(-1)
        y = y.reshape(-1)
        for i, j in zip(y, x):
            f_k = (cfg['min_dim'] / cfg['steps'][k][0], cfg['min_dim'] / cfg['steps'][k][1])
            # 计算网格中心
            cx = (j + 0.5) / f_k[0]
            cy = (i +0.5) / f_k[1]
            # 求短边（小正方形检测框）
            s_k = cfg['min_sizes'][k] / cfg['min_dim']
            mean += [cx, cy, s_k, s_k]
            # 求长边（大正方形检测框）
            s_k_prime = sqrt(s_k * (cfg['max_sizes'][k] / cfg['min_dim']))
            mean += [cx, cy, s_k_prime, s_k_prime]
            # 获得长方形检测框
            for ar in cfg['aspect_ratios'][k]:
                mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
    # 获得所有的先验框
    output = torch.Tensor(mean).view(-1, 4)
    if cfg['clip']:
        output.clamp_(max=1, min=0)
    return output
'''

class PriorBox(nn.Module):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # np.meshgrid(x, y)：生成坐标矩阵，输入的x，y，就是网格点的横纵坐标列向量（非矩阵）
            x,y = np.meshgrid(np.arange(f[0]),np.arange(f[1]))
            x = x.reshape(-1)
            y = y.reshape(-1)
            for i, j in zip(y,x):
                f_k = (self.image_size/self.steps[k][0], self.image_size/self.steps[k][1])
                # 计算网格的中心
                cx = (j + 0.5) / f_k[0]
                cy = (i + 0.5) / f_k[1]

                # 求短边
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 求长边
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 获得长方形
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # 获得所有的先验框
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

if __name__ == '__main__':
    b = PriorBox(Config)
    print(b.shape)
