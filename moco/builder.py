# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import moco.modulate

import numpy as np

########################################################
# ToDo

# Use same MLP for conditioned rediction...?

# Build positives


########################################################


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, dist=True, masker=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dist = dist

        print(K, m, T)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.masker = masker

        self._reinit_encoder_k()

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_2", torch.randn(2048, K))

        # self.register_buffer("queue", torch.randn(dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_2", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _reinit_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.dist:
            # gather keys before updating queue
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T if hasattr(keys, 'T') else keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_2(self, keys):
        if self.dist:
            # gather keys before updating queue
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_2[:, ptr:ptr + batch_size] = keys.T if hasattr(keys, 'T') else keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_2[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, return_feats=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_hid = self.encoder_q(im_q, skip_fc=True)  # queries: NxC
        q_fc = self.encoder_q.fc(q_hid)
        q = nn.functional.normalize(q_fc, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.dist:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_hid = self.encoder_k(im_k, skip_fc=True)  # keys: NxC
            k_fc = self.encoder_k.fc(k_hid)
            k = nn.functional.normalize(k_fc, dim=1)

            if self.dist:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        queue = self.queue.clone().detach()
        queue_norm = nn.functional.normalize(queue, dim=0)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue_norm])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        if self.masker is not None:
            if isinstance(self.masker, moco.modulate.Symmetric):
                # self-magnify
                masks, m_aux_loss, q_p, k_p = self.masker(q, k)
                logits_m = torch.einsum('ij,ij->i', q_p, k_p)

                logits_m = torch.cat([logits_m[:, None], l_pos], dim=1)
                logits_m /= self.T

                labels_m = torch.zeros(logits_m.shape[0], dtype=torch.long).cuda()

            elif isinstance(self.masker, moco.modulate.Instance) and False:
                K = 10

                ########################################################################
                # obtain suitable keys for conditioning
                logits_nn, ids_nn = torch.topk(l_neg, k=K, dim=-1)
                
                k_nn = queue[:, ids_nn].permute(1,2,0)      # neighbour embeddings
                q_nn = q[:, None].expand_as(k_nn)           # expanded query embeddings

                ########################################################################

                # condition the query embeddings
                masks, m_aux_loss, q_p, k_p = self.masker(q_nn.flatten(0,1), k_nn.flatten(0,1))

                q_p = q_p.view(*q_nn.shape[:2], q_p.shape[-1]) # B x NN x D  (same as k_nn)
                # for each pi, produce positive set:
                # q|pi with q, pi
                

                # negative set:
                # q|pj, pj, k
                # compute <q|pi, pj>

                # q|pi v.s. pj
                logits_m = torch.einsum('njc,nlc->njl', q_p, k_nn).flatten(0, 1)
                labels_m = torch.Tensor(list(range(logits_m.shape[-1])) * q_p.shape[0]).long().cuda()

                if torch.rand(1).item() > 0.5:
                    _logits_m = torch.einsum('nc,njc->nj', q, q_p).flatten(0, 1)
                    logits_m[:, 0] = _logits_m

                    # logits_m_pos = torch.einsum('nc,njc->nj', q, q_p)
                    # logits_m_neg = torch.einsum('njc,ck->njk', q_p, k)
                    # K = 100
                    # _, ids_nn = torch.topk(l_neg, k=K, dim=-1)


                logits_m /= self.T

            elif isinstance(self.masker, moco.modulate.Instance):
                K = 10

                ########################################################################
                # obtain suitable keys for conditioning
                logits_nn, ids_nn = torch.topk(l_neg, k=K, dim=-1, largest=True)
                # ids_not_nn = torch.topk(l_neg, k=queue.shape[-1]-K, dim=-1, largest=False)
                ids_not_nn = np.delete(torch.arange(0, self.queue.shape[-1]), ids_nn.cpu(), axis=0)

                # k_nn = queue[:, ids_nn].permute(1,2,0)       # neighbour embeddings
                # q_nn = q_fc[:, None].expand_as(k_nn)            # expanded query embeddings

                k_nn = self.queue_2[:, ids_nn].permute(1,2,0)       # neighbour embeddings
                k_nn_norm = queue_norm[:, ids_nn].permute(1,2,0)
                q_nn = q_hid[:, None].expand_as(k_nn)            # expanded query embeddings

                ########################################################################

                # condition the query embeddings
                masks, m_aux_loss, q_p, k_p = self.masker(q_nn.flatten(0,1), k_nn.flatten(0,1))

                q_p = q_p.view(*q_nn.shape[:2], q_p.shape[-1]) # B x NN x D  (same as k_nn)

                ########################################################################
                pos_logits = []
                neg_logits = []

                # q|pi v.s. q (positives)
                logits_q_pi__q = torch.einsum('nc,njc->nj', q, q_p).flatten(0, 1)
                pos_logits.append(logits_q_pi__q[:, None])

                # q|pi v.s. q|pj
                logits_q_pi__q_pj = torch.einsum('njc,nlc->njl', q_p, q_p).flatten(0, 1)
                # pos_logits.append(logits_q_pi__q_pj[:, 0])
                neg_logits.append(logits_q_pi__q_pj[:, 1:])


                # q|pi v.s. pj
                # import pdb; pdb.set_trace()
                logits_q_pi__pj = torch.einsum('njc,nlc->njl', q_p, k_nn_norm).flatten(0, 1)
                # logits_q_pi__pj = torch.einsum('njc,nlc->njl', q_p, nn.functional.normalize(k_nn, dim=-1)).flatten(0, 1)
                pos_logits.append(logits_q_pi__pj[:, 0:1])
                neg_logits.append(logits_q_pi__pj[:, 1:])

                # q|pi v.s. k (for k not in pj)
                logits_q_pi__k = torch.einsum('njc,cl->njl', q_p, queue_norm[:, ids_not_nn[::4]]).flatten(0, 1)
                # import pdb; pdb.set_trace()
                # pos_logits.append(logits_)
                neg_logits.append(logits_q_pi__k)

                logits_m = torch.cat(pos_logits + neg_logits, axis=-1)
                labels_m = torch.zeros(logits_m.shape).cuda()
                labels_m[:, :len(pos_logits)] = 1/len(pos_logits)

                # if torch.rand(1).item() > 0.5:
                #     _logits_m = torch.einsum('nc,njc->nj', q, q_p).flatten(0, 1)
                #     logits_m[:, 0] = _logits_m

                    # logits_m_pos = torch.einsum('nc,njc->nj', q, q_p)
                    # logits_m_neg = torch.einsum('njc,ck->njk', q_p, k)
                    # K = 100
                    # _, ids_nn = torch.topk(l_neg, k=K, dim=-1)

                logits_m /= self.T

                # logits[:, 0] = q_p

            # elif isinstance(self.masker, moco.modulate.):

            # dequeue and enqueue
            self._dequeue_and_enqueue(k_fc)
            self._dequeue_and_enqueue_2(k_hid)

            return logits, labels, (logits_m, labels_m, masks, m_aux_loss), q_hid, k_hid, q_p, k_p

        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_fc)
        self._dequeue_and_enqueue_2(k_hid)

        if return_feats:
            return logits, labels, q, k

        return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
