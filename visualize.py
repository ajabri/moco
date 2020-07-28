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

import moco.loader
import moco.builder

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

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--name', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--nvis', default=2e4, type=int,
                    help='number of images to use in visualization')

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

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,
        dist=False)
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

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data) #, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
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
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    plain = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation), orig_transform=plain))
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, 0, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    f1 = []
    f2 = []
    idxs = []

    X_ = []
    X1 = []
    X2 = []

    end = time.time()
    n = 0
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X1 += [images[0]]
        X2 += [images[1]]
        X_ += [images[2]]

        if args.gpu is not None:
            q_img = images[2].cuda(args.gpu, non_blocking=True)
            k_img = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, q, k = model(im_q=q_img, im_k=k_img, return_feats=True)
        loss = criterion(output, target)

        f1.append(q.cpu().detach())
        f2.append(k.cpu().detach())
        idxs.append(torch.arange(0, q.shape[0]))

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        n += images[0].shape[0]

        if n > args.nvis:
            break

        torch.cuda.empty_cache()

    idxs = torch.cat(idxs+idxs, dim=0)
    f = torch.cat(f1+f2, dim=0)
    X12 = torch.cat(X1+X2, dim=0)

    xray(model, X1, X2, f1, f2)
    # do_pca(f, X12, idxs)

    # Z = torch.cat(X_ + X_, dim=0)
    X1, X2 = torch.cat(X1), torch.cat(X2)

    import pdb; pdb.set_trace()

    # show_augs(torch.cat(X_), X1, X2)

    ## noaug
    # do_pca(torch.cat(f1), torch.cat(X_), idxs[:idxs.shape[0]//2], args.name + '_noaug')
    # do_pca(torch.cat(f2), X2, idxs[idxs.shape[0]//2:], args.name + '_aug')
    # do_pca(f,  X12, idxs, args.name + '_noplusaug')
    
def show_augs(X, X1, X2):
    import visdom
    import torchvision

    vis = visdom.Visdom(port=8095, env='moco_augs')
    vis.close()

    unnorm = lambda x: (x - x.min()) / (x - x.min()).max()
    X, X1, X2 = unnorm(X), unnorm(X1), unnorm(X2)

    for i in range(min(X.shape[0], 20)):
        vis.images(torch.stack([X[i], X1[i], X2[i]]))

class L2Normalize(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=1) / 0.07

def xray(model, X1, X2, Q, K):
    import visdom
    import torchvision
    import cv2
    from torchray.attribution.grad_cam import grad_cam
    from torchray.attribution.excitation_backprop import contrastive_excitation_backprop, excitation_backprop
    from torchray.attribution.rise import rise
    from torchray.attribution.deconvnet import deconvnet
    from torchray.benchmark import get_example_data, plot_example
    from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward

    from torchray.utils import imsc
    from matplotlib import cm
    color = cm.get_cmap('jet')

    vis = visdom.Visdom(port=8095, env='moco_xray')
    vis.close()


    def do_saliency(x, k, one_vs_all=False):
        head = torch.nn.Linear(k.shape[-1], k.shape[-2], bias=False)
        head.weight.data[:] = k.data[:]
        head = head.cuda()

        contrast_model = torch.nn.Sequential(
            model.encoder_q,
            L2Normalize(),
            head,
            torch.nn.Softmax(dim=-1))

        x = x.cuda()

        def colorize(saliency):
            resized = cv2.resize(
                    saliency.permute(0,2,3,1)[0].detach().cpu().numpy(),
                    (x.shape[-2], x.shape[-1]), interpolation=cv2.INTER_LANCZOS4) * 255
                    
            # import pdb; pdb.set_trace() #* 255.

            # import pdb; pdb.set_trace()
            return color(resized)[..., :3]
            # return np.stack([resized]*3, axis=-1)
            # return np.stack([resized]*3, dim=-1)

        def _saliency(x, func):
            s = []
            for i in range(x.shape[0]):
                saliency = func(contrast_model, x[i][None].cuda(), i)
                # import pdb; pdb.set_trace()
                saliency = colorize(saliency)
                s.append(saliency)
            return np.stack(s).transpose(0, -1, 1, 2)     

        def gcam(m, x, i):
            g = grad_cam(m, x, i, saliency_layer=model.encoder_q.layer4)
            return g
        
        def gcam_l3(m, x, i):
            return grad_cam(m, x, i, saliency_layer=model.encoder_q.layer3)

        def excite_c1(m, x, i):
            saliency = excitation_backprop(m, x, i, saliency_layer=model.encoder_q.conv1)  
            saliency = imsc(saliency[0], quiet=True)[0][None]

            return saliency

        def excite_l1(m, x, i):
            saliency = excitation_backprop(m, x, i, saliency_layer=model.encoder_q.layer1)  
            saliency = imsc(saliency[0], quiet=True)[0][None]

            return saliency

        def ceb(m, x, i):
            # Contrastive excitation backprop.
            return contrastive_excitation_backprop(m, x, i,
                saliency_layer=model.encoder_q.layer2[-1],
                contrast_layer=model.encoder_q.layer4[-1],
                classifier_layer=model.encoder_q.fc[-1]
            )

            # import excitationbp as eb


        def ep(area):
            def _ep(m, x, i):
                # Extremal perturbation backprop.
                masks_1, _ = extremal_perturbation(
                    m, x, i,
                    reward_func=contrastive_reward,
                    debug=True,
                    areas=[area],
                )
                masks_1 = imsc(masks_1[0], quiet=True)[0][None]

                return masks_1
            return _ep
        
        def decnn(m, x, i):
            saliency = deconvnet(m, x, i)
            saliency = imsc(saliency[0], quiet=True)[0][None]

            return saliency



        if one_vs_all:
            '''
            compare first x to all k
            '''
            import time

            assert x.shape[0] == 1, 'expected just one instance'

            t0 = time.time()
            rise_saliency = rise(contrast_model, x).transpose(0, 1)
            print('rise took ************', time.time() - t0)

            rise_saliency = torch.stack([imsc(rs, quiet=True)[0] for rs in rise_saliency])
            # import pdb; pdb.set_trace()

            # rise_saliency = np.stack([colorize(rs[None]) for rs in rise_saliency])
            rise_saliency = np.stack([rise_saliency.detach().cpu().numpy()] * 3, axis=-1)[:, 0]
            # import pdb; pdb.set_trace()

            rise_saliency = rise_saliency.transpose(0,-1,1,2)

            # gcam_saliency = 

            return dict(
                # rise=rise_saliency,
                # grad_cam=_saliency(torch.cat([x]*k.shape[0]), gcam),
            )            
            
        else:
            out = dict(
                # grad_cam=_saliency(x, gcam),
                # grad_cam_l3=_saliency(x, gcam_l3),
                contrastive_excitation_backprop=_saliency(x, ceb),

                # excite_c1=_saliency(x, excite_c1),
                # excite_l1=_saliency(x, excite_l1),
                # deconvnet=_saliency(x, decnn),
                # rise=rise_saliency
            )

            for k_ep in [0.05]:#, 0.05, 0.12]:
                out['extremal_perturbation_%s' % k_ep] = _saliency(x, ep(k_ep))
            
            return out

    bsz = 5
    for b in range(0, len(X1), bsz):
        x1, x2, q, k = X1[b:b+bsz], X2[b:b+bsz], Q[b:b+bsz], K[b:b+bsz]
        x1, x2, q, k = (torch.cat(xx) for xx in (x1, x2, q, k))

        unnorm = lambda x: (x - x.min()) / (x - x.min()).max() # * 255.0
        _x1, _x2 = unnorm(x1), unnorm(x2)

        x1, x2 = _x1[:2], _x2[:2]
        S1 = do_saliency(x1, k)
        S2 = do_saliency(x2, q)

        for name in S1:
            s1, s2 = torch.from_numpy(S1[name]).float(), torch.from_numpy(S2[name]).float()
            row = [x1, s1, s2, x2]

            out = torch.stack(row, dim=1)
            out = out.reshape(-1, *out.shape[-3:])
            vis.images(out, nrow=len(row), opts=dict(title=name))


        # x1, x2 = _x1, _x2

        # S_one_v_all = do_saliency(x1[0][None], k, one_vs_all=True)

        # for name in S_one_v_all:
        #     s1 = torch.from_numpy(S_one_v_all[name]).float()
        #     # import pdb; pdb.set_trace()

        #     row = [torch.stack([x1[0]]*s1.shape[0]), s1, x2]

        #     out = torch.stack(row, dim=1)
        #     out = out.reshape(-1, *out.shape[-3:])
        #     vis.images(out, nrow=len(row), opts=dict(title='one-vs-all-%s' % name))

        # import pdb; pdb.set_trace()


        vis.text('', opts=dict(height=1, width=10000))

    return


def do_pca(f, X, idxs, name=''):    
    from sklearn.decomposition import PCA, FastICA
    import visdom
    import torchvision

    vis = visdom.Visdom(port=8095, env='moco_nn_%s' % name)
    vis.close()

    # idxs = torch.cat(idxs+idxs, dim=0)
    # f = torch.cat(f1+f2, dim=0)
    # X = torch.cat(X1+X2, dim=0)

    D = torch.matmul(f,  f.t())
    X -= X.min(); X /= X.max()

    # f1 = torch.cat(f1, dim=0)
    # f2 = torch.cat(f2, dim=0)

    ########################### PCA ###########################
    K = 50
    # # pca = PCA(n_components=K, svd_solver='auto', whiten=False)
    # pca = FastICA(n_components=K, whiten=False)

    # import pdb; pdb.set_trace()

    # p_f = pca.fit_transform(f.numpy())

    # l = []
    # import math
    # step = math.ceil(p_f.shape[0]/300)
    # i_f = np.argsort(p_f, axis=0)[::step]

    # for k in range(0, K):
    #     vis.image(torchvision.utils.make_grid(X[i_f[:, k]], nrow=10, padding=2, pad_value=0).cpu().numpy(),
    #         opts=dict(title='Component %s' % k))

    # vis.text('NN', opts=dict(width=1000, h=1))


    # import pdb; pdb.set_trace()

    ########################### NN  ###########################
    V, I = torch.topk(D, 50, dim=-1)

    for _k in range(K):
        k = np.random.randint(X.shape[0])
        vis.image(torchvision.utils.make_grid(X[I[k]], nrow=10, padding=2, pad_value=0).cpu().numpy(),
            opts=dict(title='Example %s' % k))



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
