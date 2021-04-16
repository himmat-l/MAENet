import sys

sys.path.append('./')
import argparse
import os
import datetime
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
torch.cuda.set_device(2)
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import time
from torchstat import stat
import unittest
import inspect

from src.MultiTaskCNN2 import MultiTaskCNN, MultiTaskCNN_Atten, MultiTaskCNN_Atten3
from data_process import data_eval
from utils import utils
from utils.utils import save_ckpt, load_ckpt, print_log, poly_lr_scheduler
import utils.utils as utils
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc
from gpu_mem_track import  MemTracker

from torchsummary import summary, summary_string


# 参数定义
parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--train-data-dir', default=None, metavar='DIR',
                    help='path to dataset-train')
parser.add_argument('--val-data-dir', default=None, metavar='DIR',
                    help='path to dataset-val')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save_epoch_freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')
parser.add_argument('--optimizer', default='sgd', help='optimizer, support rmsprop, sgd, adam')
parser.add_argument('--context_path', type=str, default='resnet18',
                    help='The context path model you are using')
parser.add_argument('--num_class', type=str, default=38,
                    help='num classes')
args = parser.parse_args()

# 设置每个类别的权重，在计算损失时使用,当训练集不平衡时该参数十分有用。 len(nyuv2_frq)=40

device = torch.device("cuda:2"if args.cuda and torch.cuda.is_available() else "cpu")  #if args.cuda and torch.cuda.is_available() else "cpu"
image_h = 480
image_w = 640
log_file = '/home/liuxiaohui/MAENet/summary/rdbinet-acm/log.txt'
# 训练函数
def train():
    # 记录数据在tensorboard中显示
    writer_loss = SummaryWriter(os.path.join(args.summary_dir, 'loss'))
    # writer_loss1 = SummaryWriter(os.path.join(args.summary_dir, 'loss', 'loss1'))
    # writer_loss2 = SummaryWriter(os.path.join(args.summary_dir, 'loss', 'loss2'))
    # writer_loss3 = SummaryWriter(os.path.join(args.summary_dir, 'loss', 'loss3'))
    writer_acc = SummaryWriter(os.path.join(args.summary_dir, 'macc'))

    # 准备数据集
    train_data = data_eval.ReadData(transform=transforms.Compose([data_eval.scaleNorm(),
                                                                 data_eval.RandomScale((1.0, 1.4)),
                                                                 data_eval.RandomHSV((0.9, 1.1),
                                                                                     (0.9, 1.1),
                                                                                     (25, 25)),
                                                                 data_eval.RandomCrop(image_h, image_w),
                                                                 data_eval.RandomFlip(),
                                                                 data_eval.ToTensor(),
                                                                 data_eval.Normalize()]),
                                   data_dir=args.train_data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)
    val_data = data_eval.ReadData(transform=transforms.Compose([data_eval.scaleNorm(),
                                                                 data_eval.RandomScale((1.0, 1.4)),
                                                                 data_eval.RandomCrop(image_h, image_w),
                                                                 data_eval.ToTensor(),
                                                                 data_eval.Normalize()]),
                                   data_dir=args.val_data_dir)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)
    num_train = len(train_data)
    # num_val = len(val_data)

    # build model
    if args.last_ckpt:
        model = MultiTaskCNN_Atten(38, depth_channel=1, pretrained=False, arch='resnet50', use_aspp=True)
    else:
        model = MultiTaskCNN_Atten(38, depth_channel=1, pretrained=True, arch='resnet50', use_aspp=True)


    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    global_step = 0
    max_miou_val = 0
    loss_count = 0
    # 如果有模型的训练权重，则获取global_step，start_epoch
    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)
    # if torch.cuda.device_count() > 1 and args.cuda and torch.cuda.is_available():
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)
    model.train()
    # cal_param(model, data)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(int(args.start_epoch), args.epochs):
        torch.cuda.empty_cache()
        # if epoch <= freeze_epoch:
        #     for layer in [model.conv1, model.maxpool,model.layer1, model.layer2, model.layer3, model.layer4]:
        #         for param in layer.parameters():
        #             param.requires_grad = False
        tq = tqdm(total=len(train_loader) * args.batch_size)
        if loss_count >= 10:
            args.lr = 0.5 * args.lr
            loss_count = 0
        lr = poly_lr_scheduler(optimizer, args.lr, iter=epoch, max_iter=args.epochs)
        optimizer.param_groups[0]['lr'] = lr
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5)
        tq.set_description('epoch %d, lr %f' % (epoch, args.lr))
        loss_record = []
        # loss1_record = []
        # loss2_record = []
        # loss3_record = []
        local_count = 0
        # print('1')
        for batch_idx, data in enumerate(train_loader):
            image = data['image'].to(device)
            depth = data['depth'].to(device)
            label = data['label'].long().to(device)
            # print('label', label.shape)
            output, output_sup1, output_sup2 = model(image, depth)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            local_count += image.data.shape[0]
            # writer_loss.add_scalar('loss_step', loss, global_step)
            # writer_loss1.add_scalar('loss1_step', loss1, global_step)
            # writer_loss2.add_scalar('loss2_step', loss2, global_step)
            # writer_loss3.add_scalar('loss3_step', loss3, global_step)
            loss_record.append(loss.item())
            # loss1_record.append(loss1.item())
            # loss2_record.append(loss2.item())
            # loss3_record.append(loss3.item())
            if global_step % args.print_freq == 0 or global_step == 1:
                for name, param in model.named_parameters():
                    writer_loss.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                writer_loss.add_graph(model,[image, depth])
                grid_image1 = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                writer_loss.add_image('image', grid_image1, global_step)
                grid_image2 = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
                writer_loss.add_image('depth', grid_image2, global_step)
                grid_image3 = make_grid(utils.color_label(torch.max(output[:3], 1)[1]), 3,
                                        normalize=False,
                                        range=(0, 255))
                writer_loss.add_image('Predicted label', grid_image3, global_step)
                grid_image4 = make_grid(utils.color_label(label[:3]), 3, normalize=False, range=(0, 255))
                writer_loss.add_image('Groundtruth label', grid_image4, global_step)


        tq.close()
        loss_train_mean = np.mean(loss_record)
        with open(log_file,'a') as f:
            f.write(str(epoch) + '\t' + str(loss_train_mean))
        # loss1_train_mean = np.mean(loss1_record)
        # loss2_train_mean = np.mean(loss2_record)
        # loss3_train_mean = np.mean(loss3_record)
        writer_loss.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        # writer_loss1.add_scalar('epoch/sub_loss_epoch_train', float(loss1_train_mean), epoch)
        # writer_loss2.add_scalar('epoch/sub_loss_epoch_train', float(loss2_train_mean), epoch)
        # writer_loss3.add_scalar('epoch/sub_loss_epoch_train', float(loss3_train_mean), epoch)
        print('loss for train : %f' % loss_train_mean)
        print('----validation starting----')
        # tq_val = tqdm(total=len(val_loader) * args.batch_size)
        # tq_val.set_description('epoch %d' % epoch)
        model.eval()

        val_total_time = 0
        with torch.no_grad():
            sys.stdout.flush()
            tbar = tqdm(val_loader)
            acc_meter = AverageMeter()
            intersection_meter = AverageMeter()
            union_meter = AverageMeter()
            a_meter = AverageMeter()
            b_meter = AverageMeter()
            for batch_idx, sample in enumerate(tbar):

                # origin_image = sample['origin_image'].numpy()
                # origin_depth = sample['origin_depth'].numpy()
                image_val = sample['image'].to(device)
                depth_val = sample['depth'].to(device)
                label_val = sample['label'].numpy()

                with torch.no_grad():
                    start = time.time()
                    pred = model(image_val, depth_val)
                    end = time.time()
                    duration = end - start
                    val_total_time += duration
                # tq_val.set_postfix(fps ='%.4f' % (args.batch_size / (end - start)))
                print_str = 'Test step [{}/{}].'.format(batch_idx + 1, len(val_loader))
                tbar.set_description(print_str)

                output_val = torch.max(pred, 1)[1]
                output_val = output_val.squeeze(0).cpu().numpy()

                acc, pix = accuracy(output_val, label_val)
                intersection, union = intersectionAndUnion(output_val, label_val, args.num_class)
                acc_meter.update(acc, pix)
                a_m, b_m = macc(output_val, label_val, args.num_class)
                intersection_meter.update(intersection)
                union_meter.update(union)
                a_meter.update(a_m)
                b_meter.update(b_m)
        fps = len(val_loader) / val_total_time
        print('fps = %.4f' % fps)
        tbar.close()
        mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
        with open(log_file, 'a') as f:
            f.write('                    ' + str(mAcc.mean()) + '\n')
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        writer_acc.add_scalar('epoch/Acc_epoch_train', mAcc.mean(), epoch)
        print('----validation finished----')
        model.train()
        # # 每隔save_epoch_freq个epoch就保存一次权重
        if epoch != args.start_epoch:
            if iou.mean() >= max_miou_val:
                print('mIoU:', iou.mean())
                if not os.path.isdir(args.ckpt_dir):
                    os.mkdir(args.ckpt_dir)
                save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train)
                max_miou_val = iou.mean()
                # max_macc_val = mAcc.mean()
            else:
                loss_count += 1
        torch.cuda.empty_cache()






if __name__ == '__main__':
    train()










