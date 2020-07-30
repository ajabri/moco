#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet

import moco.loader
import moco.builder
import moco.modulate

import sys
import numpy as np


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# Mask options
parser.add_argument('--mloss_coef', type=float, default=0.0)
parser.add_argument('--maloss_coef', type=float, default=0.0)
parser.add_argument('--maloss_mode', type=str, default='relu')
parser.add_argument('--mask_mode', type=str, default='bilinear')
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--xray', action='store_true', default=False)

parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to an encoder checkpoint (will do filtered state_dict load) (default: none)')

parser.add_argument('--image_size', type=int, default=224)

parser.add_argument('--name', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--use-same-head', action='store_true', default=False)

import moco.utils as utils
import wandb
import torchvision

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    wandb.init(project="unrel", group="debug", notes=args.name)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    masker = moco.modulate.Instance(
        inp_dim=2048, #args.moco_dim,
        H=10 if args.mask_mode == 'bilinear' else 100,
        out_dim=args.moco_dim,
        mode=args.mask_mode,
        nonlin=args.maloss_mode)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        getattr(resnet, args.arch),
        dim=args.moco_dim, 
        K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp,
        masker=masker,
        dist=False)

    if args.use_same_head:
        masker.head = model.encoder_q.fc

    print(model)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module'):# and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0

            msg = model.load_state_dict(state_dict, strict=False)
            assert not any(['encoder_q' in k for k in set(msg.missing_keys)]), 'missing encoder_q keys!'

            model._reinit_encoder_k()
            # import pdb; pdb.set_trace()

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    kl_criterion = nn.KLDivLoss(reduction='batchmean').cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(masker.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            mdict, cdict = model.state_dict().keys(), checkpoint['state_dict'].keys()
            print('notin 1', [k for k in mdict if k not in cdict])
            print('notin 2', [k for k in cdict if k not in mdict])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    cropsize = args.image_size

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(cropsize, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(cropsize, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # vis = utils.Visualize(args)
    # vis.vis.close()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)


        # train for one epoch
        train(train_loader, model, criterion, kl_criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, kl_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mlosses = AverageMeter('Mask Loss', ':.4e')
    malosses = AverageMeter('Mask Aux Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, mlosses, malosses],
        prefix="Epoch: [{}]".format(epoch))


    # switch to train mode
    model.train()

    end = time.time()

    strip = lambda x: x.detach().cpu()
    fq, fk, x1, x2, mm = [], [], [], [], []

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, (output_m, target_m, masks, maloss), q, k, mq, mk = \
            model(im_q=images[0], im_k=images[1])
        # output, target = model(im_q=images[0], im_k=images[1])

        # output and target are lists of pairs
        loss = criterion(output, target)
    
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0].item(), images[0].size(0))
        top5.update(acc5[0].item(), images[0].size(0))

        if args.mloss_coef > 0:
            if target_m.ndim == 1:
                mloss = criterion(output_m, target_m)
            else:
                output_m = torch.nn.functional.softmax(output_m, dim=-1).log()
                # assert all(target_m.sum(-1) == 1)
                mloss = kl_criterion(output_m, target_m)
            mlosses.update(mloss.item())
            malosses.update(maloss.item())

            # import pdb; pdb.set_trace()
            loss += args.mloss_coef * mloss + args.maloss_coef * maloss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if i > 50:
                wandb.log(
                    dict(
                        loss=loss.item(),
                        top1=acc1[0].item(), top5=acc5[0].item(),
                        mloss=mloss.item(), maloss=maloss.item(),
                    )
                )

        if len(fq) < 20:
            # visualize
            fq += [strip(q)]
            fk += [strip(k)]
            mm += [strip(masks)]
            x1 += [strip(images[0])]
            x2 += [strip(images[1])]
            

        # [(print(m.avg)) for m in progress.meters[:]]
        # [(vis.log(m.name, m.avg)) for m in progress.meters[:]]
        # vis.vis.text('', opts=dict(width=10000, height=1), win='metric_header')

        elif ((epoch+1) % 1) == 0 and ((i+1) % 1000) == 0:
            masker = model.masker if not hasattr(model, 'module') else model.module.masker
            encoder = model.encoder_q if not hasattr(model, 'module') else model.module.encoder_q

            head = encoder.fc.cpu()
            masker = masker.cpu()

            fq_hid, fk_hid, mm, x1, x2 = (torch.cat(_, dim=0) for _ in (fq, fk, mm, x1, x2))
            
            # import pdb; pdb.set_trace()

            # VISUALIZE NN
            nvis_q, nvis_k = 2, 4
            nnn = 18

            fq, fk = head(fq_hid), head(fk_hid)

            for i in range(nvis_q):
                # fq, fk = head(fq_hid), head(fk_hid)

                D = [strip(torch.einsum('ik,jk->ij', (fq[i][None],  fk)))]

                ids = torch.argsort(D[0], descending=True).squeeze()

                uncond_img = torch.cat([x1[i][None], x1[i][None]*0, x2[ids[:nnn]]])
                uncond_img -= uncond_img.min(); uncond_img /= uncond_img.max()
                uncond_img = torchvision.utils.make_grid(uncond_img, nrow=10, padding=2, pad_value=0)

                wandb.log({'nn %s' % (i): [wandb.Image(uncond_img)]})

                ids = ids[:nvis_k]
                fk_hids = fk_hid[ids]
                m, m_aux_loss, _, _ = masker(fq_hid[i][None].expand_as(fk_hids), fk_hids)

                # conditoined lookups: select a random query, select
                for n, j in enumerate(ids):
                    _fq = F.normalize(masker.condition(fq_hid[i][None], m[n]))
                    _fk = fk
                    # _fk = F.normalize(masker.condition(fk.cuda(), m[n].cuda()))

                    D += [strip(torch.einsum('ij,kj->ik', _fq, _fk))]
                    _D, I = torch.sort(D[-1], descending=True, axis=-1)

                    # import pdb; pdb.set_trace()

                    nn_img = torch.cat([x1[i][None], x2[j][None], x2[I[0,:nnn]]])
                    nn_img -= nn_img.min(); nn_img /= nn_img.max()
                    nn_img = torchvision.utils.make_grid(nn_img, nrow=10, padding=2, pad_value=0)
                    wandb.log({
                        'nn %s:%s' % (i,n): [wandb.Image(nn_img)]
                    })
                    # vis.vis.bar(_D[0][::-1][:20], opts=dict(height=150, width=500), win='patch_affinity_%s_%s' % (n, i))

            encoder.fc = encoder.fc.cuda()
            masker = masker.cuda()

            fq = []
            fk = []
            mm = []
            x1 = []
            x2 = []


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
