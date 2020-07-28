
# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io

import visdom
import time
import PIL
import torchvision
import skimage

def partial_load(pretrained_dict, model):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    skipped_keys = [k for k in pretrained_dict if k not in filtered_dict]
    print('Skipped keys: ',  skipped_keys)

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def pca_feats(ff, solver='auto', img_normalize=True):
    ## expect ff to be   N x C x H x W
        
    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3,
        svd_solver=solver,
        whiten=True
    )
#     print(ff.shape)
    ff = ff.transpose(1, 2).transpose(2, 3)
#     print(ff.shape)
    ff = ff.reshape(N*H*W, C).numpy()
#     print(ff.shape)
    
    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3)
#     print(pca_ff.shape)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    if img_normalize:
        pca_ff = (pca_ff - pca_ff.min()) / (pca_ff.max() - pca_ff.min())


    return pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        # print(video.min(), video.max())
        
        video = (video*255).astype(np.uint8)
#         video = video.chunk(video.shape[0])
        
    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration = 0.2)


from matplotlib import cm
import time
import cv2

def draw_matches(x1, x2, i1, i2):
    # x1, x2 = f1, f2/
    detach = lambda x: x.detach().cpu().numpy().transpose(1,2,0) * 255
    i1, i2 = detach(i1), detach(i2)
    i1, i2 = cv2.resize(i1, (400, 400)), cv2.resize(i2, (400, 400))

    for check in [True]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)
        # matches = bf.match(x1.permute(0,2,1).view(-1, 128).cpu().detach().numpy(), x2.permute(0,2,1).view(-1, 128).cpu().detach().numpy())

        h = int(x1.shape[-1]**0.5)
        matches = bf.match(x1.t().cpu().detach().numpy(), x2.t().cpu().detach().numpy())

        scale = i1.shape[-2] / h
        grid = torch.stack([torch.arange(0, h)[None].repeat(h, 1), torch.arange(0, h)[:, None].repeat(1, h)])
        
        grid = grid.view(2, -1)
        grid = grid * scale + scale//2

        kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

        matches = sorted(matches, key = lambda x:x.distance)

        # img1 = img2 = np.zeros((40, 40, 3))
        out = cv2.drawMatches(i1, kps, i2,kps,matches[:], None, flags=2).transpose(2,0,1)

    return out

class PatchGraph(object):
    
    color = cm.get_cmap('jet')
    pad = 0
    
    def blend(self, i):

        y, x = i // self.W, i % self.W
        cx, cy = [int((self.w + self.pad) * (x  + 0.5)), int((self.h + self.pad) * (y  + 0.5))]

        def _blend(x):
            x = x[...,:-self.pad, :-self.pad] if self.pad > 0 else x
            x = (0.5 * self.maps[i] + 0.5 * x).copy() * 255
            return x

        img1 = self.grid[0]*255.0
        img2 = _blend(self.grid[1])
        img1[:, cy-5:cy+5, cx-5:cx+5] = 255
        # import pdb; pdb.set_trace()

        return np.concatenate([img1, img2], axis=-1), None

    def update(self):
        self.viz.image(self.curr[0], win=self.win_id, env=self.viz.env+'_pg')
        # self.viz.image(self.curr[1], win=self.win_id2, env=self.viz.env+'_pg')

    def make_canvas(self, I, orig, N):
        # import pdb; pdb.set_trace()
        # if N == 1:
        #     grid = [cv2.resize(o.numpy().transpose(1,2,0), (800, 800), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1) for o in orig]
        # else:
        grid = []
        for i in range(I.shape[1]):
            grid += [torchvision.utils.make_grid(I[:, i], nrow=int(N**0.5), padding=self.pad, pad_value=0).cpu().numpy()]
        
        for i in range(len(grid)):
            grid[i] -= grid[i].min()
            grid[i] /= grid[i].max()
        
        # if orig is not None:
        #     self.orig = cv2.resize(orig[0].numpy().transpose(1,2,0), self.grid.shape[-2:], interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        #     self.orig -= self.orig.min()
        #     self.orig /= self.orig.max()
        # else:
        #     self.orig = None
        
        return grid

    def __init__(self, viz, I, A, win='patchgraph', orig=None):
        self._win = win
        self.viz = viz
        self._birth = time.time()

        P, C, T, h, w = I.shape
        N = A.shape[-1]
        H = W = int(N ** 0.5)
        self.N, self.H, self.W, self.h, self.w = N, H, W, h, w

        if P == 1:
            self.w, self.h = self.w // W, self.h // H

        I = I.cpu()
        orig = orig.cpu()
        A = A.view(H * W, H, W).cpu() #.transpose(-1, -2)

        # psize = min(2000 // H, I.shape[-1])
        # if psize < I.shape[-1]:
        #     I = [cv2.resize(ii, (psize, psize)) for ii in I]

        ####################################################################
        # Construct image data

        self.grid = self.make_canvas(I, orig, N)

        ####################################################################
        # Construct map data

        # pstride = utils.get_stride(orig.shape[-1], h, H)   # stride length used to gen patches, h is patch size, H is n patch per side
        # map_sz_ratio = (pstride * H ) / orig.shape[-1]     # compute percentage of image spanned by affinity map overlay
        # map_sz = int(map_sz_ratio * self.orig.shape[-1])
        # lpad = int(((h-pstride)//2 / orig.shape[-1]) * self.orig.shape[-1])
        # rpad = self.orig.shape[-1] - map_sz - lpad

        map_sz = self.grid[0].shape[-1]
        lpad, rpad = 0, 0

        zeros = np.zeros(self.grid[0].shape).transpose(1,2,0)
        maps = []
        for a in A[:, :, :, None].cpu().detach().numpy():
            _a = cv2.resize(a, (map_sz, map_sz), interpolation=cv2.INTER_NEAREST)
            _a = self.color(_a)[...,:3]
            a = zeros.copy()
            if lpad > 0 and rpad > 0:
                a[lpad:-rpad, lpad:-rpad, :] = _a
            else:
                a = _a
            maps.append(a)

        self.maps = np.array(maps).transpose(0, -1, 1, 2)

        ####################################################################
        # Set first image

        self.curr_id = N//2
        self.curr = self.blend(self.curr_id)
        # viz.text('', opts=dict(width=10000, height=2), env=viz.env+'_pg')
        
        self.win_id = self._win 
        self.win_id2 = self._win+'2'

        self.update()
        ####################################################################

        def str2inttuple(s):
            try:
                ss = s.split(',')
                assert(len(ss) == 2)
                return int(ss[0]), int(ss[1])
            except:
                return False

        def callback(event):
            # nonlocal win_id #, win_id_text
            # print(event['event_type'])

            #TODO make the enter key recompute the A under a
            if event['event_type'] == 'KeyPress':
                # print(event['key'], 'KEYYYYY')

                if 'Arrow' in event['key']:
                    self.curr_id += {'ArrowLeft':-1, 'ArrowRight': 1, 'ArrowUp': -self.W, 'ArrowDown': self.W}[event['key']]
                    # print('hello!!', self.curr_id)
                    self.curr_id = min(max(self.curr_id, 0), N)
                    self.curr = self.blend(self.curr_id)
                    self.update()

                # curr_txt = event['pane_data']['content']

                # print(event['key'], 'KEYYYYY')
                # if event['key'] == 'Enter':
                #     itup = str2inttuple(curr_txt)
                #     if itup:
                #         self.curr = self.blend(itup[0]*H + itup[1])
                #         viz.image(self.curr, win=self.win_id, env=viz.env+'_pg')
                #         curr_txt='Set %s' % curr_txt
                #     else:
                #         curr_txt='Invalid position tuple'

                # elif event['key'] == 'Backspace':
                #     curr_txt = curr_txt[:-1]
                # elif event['key'] == 'Delete':
                #     curr_txt = ''
                # elif len(event['key']) == 1:
                #     curr_txt += event['key']
                

                # viz.text(curr_txt, win=self.win_id_text, env=viz.env+'_pg')

            if event['event_type'] == 'Click':
                # print(event.keys())
                # import pdb; pdb.set_trace()
                # viz.text(event)
                coords = "x: {}, y: {};".format(
                    event['image_coord']['x'], event['image_coord']['y']
                )
                viz.text(coords, win=self.win_id_text, env=viz.env+'_pg')

                self.curr = self.blend(np.random.randint(N))

                viz.image(self.curr, win=self.win_id, env=viz.env+'_pg')


        viz.register_event_handler(callback, self.win_id)
        # viz.register_event_handler(callback, self.win_id_text)
        # import pdb; pdb.set_trace()


class Visualize(object):
    def __init__(self, args, suffix='metrics', log_interval=10):
        self._env_name = "%s-%s" % (args.name, suffix)
        self.vis = visdom.Visdom(
            port=getattr(args, 'port', 8095),
            server='http://%s' % getattr(args, 'server', 'localhost'),
            env=self._env_name,
        )
        self.data = dict()

        self.log_interval = log_interval
        self._last_flush = time.time()

        self.buffer = dict()

    def log(self, key, value):
        if not key in self.data:
            self.data[key] = [[],[]]

        if isinstance(value, tuple):
            self.data[key][0].append(value[0])
            self.data[key][1].append(value[1])
        else:
            self.data[key][1].append(value)
            self.data[key][0].append(len(self.data[key][1]) * 1.0)
            # import pdb; pdb.set_trace()

        if (time.time() - self._last_flush) > (self.log_interval):
            for k in self.data:
                self.vis.line(
                    X=np.array(self.data[k][0]),
                    Y=np.array(self.data[k][1]),
                    win=k,
                    opts=dict( title=k )
                )
            self._last_flush = time.time()

    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)

    def save(self):
        self.vis.save(self._env_name)

def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz)//(res-1)
    return stride

def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy() #.transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)
    
    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' %(prefix))


    for n,i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))

import torchvision
import torchvision.transforms as transforms
def unnormalize(x):
    t = transforms.Normalize(
        -np.array([0.4914, 0.4822, 0.4465])/np.array([0.2023, 0.1994, 0.2010]),
        1/np.array([0.2023, 0.1994, 0.2010]).tolist()
    )
    return t(x)


def nn_pca(f, X, name='', vis=None):    
    from sklearn.decomposition import PCA, FastICA
    import visdom
    import torchvision

    if vis is None:
        vis = visdom.Visdom(port=8095, env='%s_nn' % name)
        vis.close()

    # ########################### PCA ###########################
    K = 50
    # # pca = PCA(n_components=K, svd_solver='auto', whiten=False)
    # pca = FastICA(n_components=K, whiten=False)

    # # import pdb; pdb.set_trace()

    # p_f = pca.fit_transform(f.numpy())

    # l = []
    # import math
    # step = math.ceil(p_f.shape[0]/300)
    # i_f = np.argsort(p_f, axis=0)[::step]

    # for k in range(0, K):
    #     vis.image(torchvision.utils.make_grid(X[i_f[:, k]], nrow=10, padding=2, pad_value=0).cpu().numpy(),
    #         opts=dict(title='Component %s' % k))

    # f = torch.cat(f1+f2, dim=0)
    # X = torch.cat(X1+X2, dim=0)

    D = torch.matmul(f,  f.t())
    X -= X.min(); X /= X.max()

    # f1 = torch.cat(f1, dim=0)
    # f2 = torch.cat(f2, dim=0)

    # vis.text('NN', opts=dict(width=1000, h=1))


    # import pdb; pdb.set_trace()

    ########################### NN  ###########################
    V, I = torch.topk(D, 50, dim=-1)

    for _k in range(K):
        k = np.random.randint(X.shape[0])
        vis.image(torchvision.utils.make_grid(X[I[k]], nrow=10, padding=2, pad_value=0).cpu().numpy(),
            opts=dict(title='Example %s' % k))

