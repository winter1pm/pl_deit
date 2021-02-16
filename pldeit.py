# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import io
import json
import math
import numpy as np
import os
import sys
import time

from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, ModelEma, accuracy, get_state_dict


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


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


def build_dataset(
    is_train, 
    data_dir: str,
    image_size: int = 224,
    color_jitter: float = 0.4, 
    aa: str = "rand-m9-mstd0.5-inc1",
    train_interpolation: str = "bicubic",
    reprob: float = 0.25, 
    remode: str = "pixel", 
    recount: int = 1
):
    """ Build the ImageNet dataset. """
    transforms = build_transform(
        is_train, 
        image_size, 
        color_jitter, 
        aa, 
        train_interpolation, 
        reprob, 
        remode, 
        recount
    )

    root = os.path.join(data_dir, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transforms)
    nb_classes = 1000

    return dataset, nb_classes


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(3)]
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


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(
        self, 
        base_criterion: torch.nn.Module, 
        teacher_model: torch.nn.Module,
        distillation_type: str, 
        alpha: float, 
        tau: float
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



class DeiT(pl.LightningModule):

    def __init__(
        self,
        args
    ):
        super().__init__()
        self.args = args
        self.mixup_fn = None
        self.mixup_active = (
            args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        )
        if self.mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup, 
                cutmix_alpha=args.cutmix, 
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, 
                switch_prob=args.mixup_switch_prob, 
                mode=args.mixup_mode,
                label_smoothing=args.smoothing, 
                num_classes=args.nb_classes
            )

        # Model creation
        # TODO: load pretrained model / continue training model
        self.model_arch = args.model_arch 
        self.model = create_model(
            self.model_arch,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )
        # TODO: how to log message using pytorch-lightning?
        #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print('number of params:', n_parameters)        
        # TODO: how to manage the EMA of model on multiple devices?
        if self.args.model_ema is True:
            self.model_ema = ModelEma(
                self.model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else next(self.model.parameters()).device,
                resume=''
            )
        else:
            self.model_ema = None

        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        self.teacher_model = None
        if args.distillation_type != 'none':
            assert args.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {args.teacher_model}")
            self.teacher_model = create_model(
                args.teacher_model,
                pretrained=False,
                num_classes=args.nb_classes,
                global_pool='avg',
            )
            if args.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.teacher_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['model'])
            self.teacher_model.eval()

        self.train_criterion = DistillationLoss(
            criterion, 
            self.teacher_model, 
            args.distillation_type, 
            args.distillation_alpha, 
            args.distillation_tau
        )
        self.eval_criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = create_optimizer(args, self.model)

        self.epoch = 0

    def load_finetune_checkpoint(self):
        pass 

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        pass 
    
    def on_train_epoch_end(self):
        self.epoch += 1
        self.lr_scheduler.step(epoch)
        if self.args.output_dir:
            checkpoint_paths = [self.args.output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': self.args,
                }, checkpoint_path)

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)

        samples, targets = batch[0], batch[1]
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.model(samples)
        loss = self.train_criterion(samples, outputs, targets)
        # TODO: how to configure the customized optimization step, e.g. 
        # https://github.com/rwightman/pytorch-image-models/blob/80078c47bb5b60d337f294594433bbd6809f3975/timm/utils/cuda.py#L34
        # in timm?
        return loss 

    # TODO: do we need to no_grad(), or with torch.cuda.amp.autocast()?
    # TODO: how to return accuracy? https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/6c307c1292d26562b849c35695819cc93c104a8f/pl_bolts/models/self_supervised/moco/moco2_module.py#L291
    def validation_step(self, batch, batch_idx):
        samples, targets = batch[0], batch[1]
        outputs = self.model(samples)
        loss = self.eval_criterion(outputs, targets)
        #acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        return loss
    
    def configure_optimizers(self):
        return self.optimizer


class ImagenetDataModule(pl.LightningDataModule):
    """
    Modified on top of pl_bolts/datamodules/imagenet_datamodule.py

    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)

    Imagenet train, val and test dataloaders.

    The train set is the imagenet train.

    The val set is taken from the train set with `num_imgs_per_val_class` images per class.
    For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.

    The test set is the official imagenet validation set.

     Example::

        from pl_bolts.datamodules import ImagenetDataModule

        dm = ImagenetDataModule(IMAGENET_PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = 'imagenet'

    def __init__(
        self,
        args,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            meta_dir: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_mem: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__()
        self.args = args

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f'a {split} Imagenet split was not found in {data_dir},'
                f' make sure the folder contains a subfolder named {split}'
            )

    # reference: https://pytorch-lightning.readthedocs.io/en/stable/new-project.html#lightningdatamodules
    # `prepare_data` is called only on 1 GPU/machine
    # `setup` is called for every GPU/machine (assigning state is OK)
    def prepare_data(self) -> None:
        self._verify_splits(self.args.data_dir, 'train')
        self._verify_splits(self.args.data_dir, 'val')

        self.dataset_train, self.nb_classes = build_dataset(
            is_train=True, 
            data_dir=self.args.data_dir,
            image_size=self.args.image_size,
            color_jitter=self.args.color_jitter,
            aa=self.args.aa,
            train_interpolation=self.args.train_interpolation,
            reprob=self.args.reprob,
            remode=self.args.remode,
            recount=self.args.recount
        )

        self.dataset_valid, _ = build_dataset(
            is_train=False, 
            data_dir=self.args.data_dir,
            image_size=self.args.image_size,
            color_jitter=self.args.color_jitter,
            aa=self.args.aa,
            train_interpolation=self.args.train_interpolation,
            reprob=self.args.reprob,
            remode=self.args.remode,
            recount=self.args.recount
        )

    def setup(self, stage):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        
        # TODO; how to configure / use e.g. RASampler which directly configure hardware relevant properties?
        if self.args.repeated_aug:
            self.sampler_train = RASampler(
                self.dataset_train, num_replicas=None, rank=None, shuffle=True
            )
        else:
            self.sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=None, rank=None, shuffle=True
            )

    # TODO: when is this method called? During .fit() or before .fit()?
    def train_dataloader(self) -> DataLoader:
        # pytorch_lightning.utilities.exceptions.MisconfigurationException: 
        # You seem to have configured a sampler in your DataLoader. 
        # This will be replaced  by `DistributedSampler` since `replace_sampler_ddp` is True 
        # and you are using distributed training. 
        # Either remove the sampler from your DataLoader or 
        # set `replace_sampler_ddp`=False if you want to use your custom sampler.
        loader_train = DataLoader(
            self.dataset_train, 
            #sampler=self.sampler_train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=True,
        )
        return loader_train

    def val_dataloader(self) -> DataLoader:
        """
        Uses the part of the train split of imagenet2012  that was not used for training via `num_imgs_per_val_class`

        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        loader_val = DataLoader(
            self.dataset_valid, 
            batch_size=int(1.5 * self.args.batch_size),
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_mem,
            drop_last=False
        )
        return loader_val



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model-arch', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--image-size', default=224, type=int, help='images input size')
    parser.add_argument('--nb-classes', default=1000, type=int, help='images classes number')
    
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-dir', default='/home/dongguo/FastData/ImageNet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='/home/dongguo/FastData/ImageNet/ckpts',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    deit_model = DeiT(args)
    data_module = ImagenetDataModule(args)
    trainer = pl.Trainer.from_argparse_args(args) 
    trainer.fit(deit_model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DeiT training and evaluation script', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
