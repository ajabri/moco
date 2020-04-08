import math
import torch
from torch import nn
import torch.nn.functional as F

import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info

class Bilinear(nn.Module):
    def __init__(self, inp_dim, H):
        super(Bilinear, self).__init__()
        self.w = nn.Bilinear(inp_dim, inp_dim, H, bias=True)

    def forward(self, x12):
        # import pdb; pdb.set_trace()
        L = x12.shape[-1]//2
        # return torch.einsum('ij,kjl,il->ik', x12[:, :L], self.W, x12[:, L:])
        return self.w(x12[:, :L], x12[:, L:])


# class Bilinear(nn.Module):
#     def __init__(self, inp_dim, H):
#         super(Bilinear, self).__init__()
#         self.W = nn.Parameter(torch.Tensor(H, inp_dim, inp_dim))

#     def forward(self, x12):
#         L = x12.shape[-1]//2
#         return torch.einsum('ij,kjl,il->ik', x12[:, :L], self.W, x12[:, L:])


class Inferer(nn.Module):
    def __init__(self, inp_dim, H, mode):
        super(Inferer, self).__init__()

        self.inp_dim = inp_dim
        self.H = H
        self.mode = mode

        self.g = []

        # TODO:
        # more fusion other than cat
        if mode == 'regress':
            nhid = 2*self.inp_dim
            self.g += [
                nn.Linear(self.inp_dim, nhid),
                nn.ReLU(inplace=True),
                nn.Linear(nhid, nhid),
                nn.ReLU(inplace=True),
                nn.Linear(nhid, self.inp_dim)
            ]
        
        elif mode == 'linear' or mode == 'class':
            self.g += [
                nn.Linear(self.inp_dim, self.inp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.inp_dim, self.H)
            ]

            self.M = nn.Linear(self.H, self.inp_dim, bias=False)

        elif mode == 'bilinear':
            self.g += [
                Bilinear(self.inp_dim//2, self.H),
            ]

            self.M = nn.Linear(self.H, self.inp_dim, bias=False)

        self.g = nn.Sequential(*self.g)

        # import pdb; pdb.set_trace()

    def forward(self, q, k):
        m = self.g(torch.cat([q, k], dim=-1))
        
        if self.mode == 'regress':
            return m
        else:
            if self.mode == 'class':
                m = F.softmax(m, dim=-1)
            return self.M(m)




class Masker(nn.Module):
    def __init__(self, inp_dim, H, mode='bilinear', nonlin='relu', prior='l1'):
        super(Masker, self).__init__()
        self.inp_dim = inp_dim
        self.H = H
        self.mode = mode
        self.nonlin = nonlin
        self.prior = prior

        self.g = Inferer(2*inp_dim, H, mode)

    def forward(self, q, k):
        m = self.g(q, k)
        n_m = self.apply_nonlin(m)
        aux_loss = self.aux_loss(n_m, m)

        # import pdb; pdb.set_trace()
        # mq, mk = n_m*q, n_m*k
        mq, mk = F.normalize(n_m*q), F.normalize(n_m*k)

        m_out = torch.einsum('ij,ij->i', mq, mk)
        return n_m, m_out, aux_loss

    def apply_nonlin(self, m):

        if self.nonlin == 'softmax':
            m = F.softmax(m, dim=-1)
        else:
            m = getattr(F, self.nonlin)(m)
            # print(m)

        # elif self.nonlin == 'relu':
        #     m = F.leaky_relu(m,)
        # elif self.nonlin == 'sigmoid':
        #     m = F.sigmoid(m)
        # elif self.nonlin == 'tanh':
        #     m = F.tanh(m)

        return m

    def aux_loss(self, n_m, m):

        if self.prior == 'l2':
            return torch.norm(n_m, dim=-1, p=2).mean()

        elif self.prior == 'l1':
            return torch.norm(n_m, dim=-1, p=1).mean()

        elif self.prior == 'kl':
            return torch.nn.functional.kl_div(m, torch.ones(m.shape).cuda()*0.05)

        return torch.Tensor(0).cuda()

        # MI objective
        # elif 'top' in self.prior:
        #     k = int(self.prior[-1])
        #     # mask = torch.topk()



'''
ToDo:
    * N aug dataloader q, k where k is now a list? or has an extra dim


    Late:
        Pass each augmentation through each head, or categorical conditional batchnorm

        * Either run assignment optimization, pick that head for key as positive, other augs under that head as negatives
        * MI objective, unrel
        
    Condition on key itself:
        * 
    

'''

def make_mlp(nhid)
    '''
    construct an mlp, where nhid is the size of every layer, starting with input
    '''
    out = []
    for d1,d2 in zip(nhid[:-1], nhid[1:]):
        out += [nn.Linear(d1, d2), nn.ReLU()]
    out = out[:-1]

    return nn.Sequential(*out)


class kMLP(nn.Module):
    def __init__(self, nhids, H, ):
        self.H = H
        self.nhid = nhids
        self.heads = [make_mlp(nhids) for _ in H]
    
    def forward(self, x)
        out = [h(x) for h in self.heads]

        return out


class Modulate(nn.Module):
    def __init__(self, inp_dim, nc, mode='mlp', H=10, residual=False):
        self.inp_dim = inp_dim
        self.infer = Inferer(inp_dim, H, mode)

        self.map = torch.nn.Linear(inp_dim, nc*2)
        self.residual = residual

    def forward(self, x):
        z = self.infer(x) + (x if self.residual else  0)
        gam, bet = torch.split(self.map(z), 2, dim=-1)

        out = (1 + gam) * x + bet

        return out


import scipy
from scipy.optimize import linear_sum_assignment

class Hydra(nn.Module):
    '''
        Some transformation of base embedding that can be modulated in H different ways

        * H mlps
        * Conditional batchnorm with H classes

        # positive:
            # conditioning key
        
        # negative:
            # other augmentations?
            # unconditioned key?
            
        # Need several positives for q now
            # Either n augmentations
            # Or n nearest neighbors
    '''
    def __init__(self, inp_dim, H):
        super(Hydra, self).__init__()
        self.inp_dim = inp_dim
        self.H = H

        self.heads = [make_mlp(nhids) for _ in H]

    def forward(self, q, k):
        # q is batch of keys, k is list of batch of m augmentations

        # 1. Map to M subspaces (unconditional subspaces)
        q_k = self.heads(q)

        q_k, k_m = torch.stack(q_k, dim=1), torch.stack(k, dim=1)
        q_k, k_m = F.normalize(q_k, dim=1), F.normalize(n_m*k, dim=1)

        costs = torch.einsum('nhd,nmd->nmh', q_k, k_m)


    def route(self, costs):
        # costs here is N x M x H

        # 1. Hungarian
        routes = [linear_sum_assignment(c) for c in costs]

        import pdb; pdb.set_trace()

