import sys
import numpy as np
from torch import nn
import torch
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

med_frq = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
           0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
           2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
           0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
           1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
           4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
           3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
           0.750738, 4.040773]

'''
SUN RGBD class:background,wall,floor,cabinet柜子,bed,chair,sofa,table,door,window,
               bookshelf书架,picture,counter柜台,blinds百叶窗,desk,shelves架子,curtain窗帘,dresser化妆台,pillow枕头,
               mirror,floor_mat地毯,clothes,ceiling天花板,books,fridge冰箱,tv,paper,towel毛巾,
               shower_curtain浴帘,box,whiteboard,person,night_stand床头柜,toilet,sink水槽,lamp,bathtub浴缸,bag

NYUDv2_classes = ['background', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
           'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain',
           'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
           'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand',
           'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
'''
label_colours = [(0, 0, 0),
                 # 0=background
                 (148, 65, 137), (255, 116, 69), (223, 213, 19),
                 (202, 179, 158), (189, 61, 64), (161, 107, 108),
                 (133, 160, 103), (186, 25, 203), (84, 62, 35),
                 (44, 80, 130), (31, 184, 157), (101, 144, 77),
                 (23, 197, 62), (141, 168, 145), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (198, 244, 2),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),(139, 110, 246)]
# 剩下的颜色映射值：(161, 176, 169), (80, 29, 135), (177, 105, 197),(139, 110, 246)

# 测试时的颜色映射
# label_colours = [(0, 0, 0),
#                  # 0=background
#                  (108, 191, 119), (1, 140, 187), (33, 43, 237),
#                  (54, 77, 98), (67, 195, 192), (161, 107, 108),
#                  (133, 160, 103), (186, 25, 203), (84, 62, 35),
#                  (44, 80, 130), (31, 184, 157), (101, 144, 77),
#                  (23, 197, 62), (141, 168, 145), (142, 151, 136),
#                  (115, 201, 77), (100, 216, 255), (57, 156, 36),
#                  (88, 108, 129), (105, 129, 112), (42, 137, 126),
#                  (155, 108, 249), (166, 148, 143), (81, 91, 87),
#                  (100, 124, 51), (73, 131, 121), (157, 210, 220),
#                  (134, 181, 60), (221, 223, 147), (123, 108, 131),
#                  (161, 66, 179), (163, 221, 160), (198, 244, 2),
#                  (99, 121, 30), (49, 89, 240), (116, 108, 9),
#                  (161, 176, 169), (80, 29, 135), (177, 105, 197),(139, 110, 246)]


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        # self.weight = weight
        # self.weight = torch.from_numpy(np.array(weight)).float()
        # 关于参数size_average=False，根据pytorch的官方文档，size_average默认情况下是True，
        # 对每个小批次的损失取平均值。 但是，如果字段size_average设置为False，则每个小批次的损失将被相加。如果参数reduce=False，则忽略。

        self.ce_loss = nn.CrossEntropyLoss(weight.float(), size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            print('inputs:', inputs.shape)
            print('targets:', targets.shape)
            # targets:cuda,[4,240,320], requires_grad=False     targets_m:cuda, [4,480,640], requires_grad=False
            mask = (targets > 0)    #mask: cuda, [4, 480, 640], requires_grad=False
            targets_m = targets.clone()
            targets_m[mask] -= 1
            print('targets_m:', targets[mask].shape)
            loss_all = self.ce_loss(inputs, targets_m.long())     #loss_all: requires_grad=True, [4,480,640]
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))  #losses: requires_grad=True, scalar
            # losses.append(torch.sum(inputs)/1000000)
        total_loss = sum(losses)
        # print(total_loss)
        return total_loss

# hxx add, focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                                 size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1))**2 * F.log_softmax(input, 1), target)


class FocalLoss2d(nn.Module):
    def __init__(self, weight=med_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.fl_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label_eval(label):
    # label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    return torch.from_numpy(colored[np.newaxis, ...].transpose([0,1, 3,2]))
    # return colored.transpose([0, 2, 1])

def color_label(label):
    label = label.clone().cpu().data.numpy()
    colored_label = np.vectorize(lambda x: label_colours[int(x)])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])
    return colored.transpose([0, 2, 1])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.data, time_inter, count_inter))


def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        # model = model.to(device)
        # 加进来会报错 一部分tensor在GPU，一部分在cpu
        # if optimizer:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)


# added by hxx for iou calculation
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1 # hxx
    # imLab += 1 # label 应该是不用加的
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def accuracy(preds, label):
    valid = (label > 0) # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def macc(preds, label, num_class):
    a = np.zeros(num_class)
    b = np.zeros(num_class)
    for i in range(num_class):
        mask = (label == i+1)
        a_sum = (mask * preds == i+1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a,b

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


# 学习率更新方式，采用poly方式
def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, lr_decay_iter=1,
                       power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr


