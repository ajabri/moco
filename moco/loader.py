# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, orig_transform=None):
        self.base_transform = base_transform
        self.orig_transform = orig_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        if self.orig_transform is not None:
            return [q, k, self.orig_transform(x)]

        return [q, k]

class NCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, orig_transform=None, N=2):
        self.base_transform = base_transform
        self.orig_transform = orig_transform
        self.N = N

    def __call__(self, x):
        out = [self.base_transform(x) for _ in range(self.N)]

        if self.orig_transform is not None:
            return out + [self.orig_transform(x)]

        return out

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
