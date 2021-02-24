# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math

import torch
import torch.distributed as dist

from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_transform(
    is_train: bool, 
    image_size: int = 224,
    color_jitter: float = 0.4, 
    aa: str = "rand-m9-mstd0.5-inc1",
    train_interpolation: str = "bicubic",
    reprob: float = 0.25, 
    remode: str = "pixel", 
    recount: int = 1
):
    resize_im = image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation=train_interpolation,
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(image_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * image_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


# Similar to data.distributed.DistributedSampler
# https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, seed: int = 0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        # TODO: change to logging later
        print(f'Initializing RASampler, num_replicas = {num_replicas}, and rank = {rank}')
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))

    def __iter__(self):
        # Deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # generate the repeated augmentation samples.
        indices = [ele for ele in indices for i in range(3)]
        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
