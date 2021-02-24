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

from pytorch_lightning.callbacks import ModelCheckpoint
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


from .optim_factory import create_optimizer


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

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
        model_arch: str = "deit_tiny_patch16_224",
        mixup: float = 0.8,
        cutmix: float = 1.0,
        cutmix_minmax: Optional[float] = None,
        mixup_prob: float = 1.0,
        mixup_switch_prob: float = 0.5,
        mixup_mode: str = "batch",
        smoothing: float = 0.1,
        nb_classes: int = 2000,
        drop: float = 0.0,
        drop_path: float = 0.1,
        model_ema: bool = False,
        model_ema_decay: float = 0.99996,
        model_ema_force_cpu: bool = False,
        finetune: bool = False,
        distillation_type: str = "none",
        distillation_alpha: float = 0.5,
        distillation_tau: float = 1.0,
        teacher_model: str = "regnety_160",
        teacher_path: str = "",
        opt: str = "torch.optim.AdamW",
        opt_eps: float = 1e-08,
        momentum: float = 0.9,
        weight_decay: float = 0.05,
        sched: str = "cosine",
        lr: float = 0.0005,
        lr_noise: Optional[float] = None,
        lr_noise_pct: float = 0.67,
        lr_noise_std: float = 1.0,
        warmup_lr: float = 1e-06,
        min_lr: float = 1e-05,
        decay_epochs: int = 30,
        warmup_epochs: int = 5,
        cooldown_epochs: int = 10,
        patience_epochs: int = 10,
        decay_rate: float = 0.1,
    ):
        super().__init__()
        self.model_arch = model_arch
        self.mixup = mixup
        self.cutmix = cutmix
        self.cutmix_minmax = cutmix_minmax
        self.mixup_prob = mixup_prob
        self.mixup_switch_prob = mixup_switch_prob
        self.mixup_mode = mixup_mode
        self.smoothing = smoothing
        self.nb_classes = nb_classes
        self.drop = drop
        self.drop_path = drop_path
        self.finetune = finetune
        self.model_ema = model_ema
        self.model_ema_decay = model_ema_decay
        self.model_ema_force_cpu= model_ema_force_cpu
        self.distillation_type = distillation_type
        self.distillation_alpha = distillation_alpha
        self.distillation_tau = distillation_tau
        self.teacher_model = teacher_model
        self.teacher_path = teacher_path
        self.opt = opt
        self.opt_eps = opt_eps
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.sched = sched
        self.lr = lr
        self.lr_noise_pct = lr_noise_pct
        self.lr_noise_std = lr_noise_std
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.decay_epochs = decay_epochs
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.patience_epochs = patience_epochs
        self.decay_rate = decay_rate

        self.mixup_fn = None
        self.mixup_active = (
            self.mixup > 0 or self.cutmix > 0. or self.cutmix_minmax is not None
        )
        if self.mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.mixup, 
                cutmix_alpha=self.cutmix, 
                cutmix_minmax=self.cutmix_minmax,
                prob=self.mixup_prob, 
                switch_prob=self.mixup_switch_prob, 
                mode=self.mixup_mode,
                label_smoothing=self.smoothing, 
                num_classes=self.nb_classes
            )

        self.model_arch = self.model_arch 
        self.model = create_model(
            self.model_arch,
            pretrained=False,
            num_classes=self.nb_classes,
            drop_rate=self.drop,
            drop_path_rate=self.drop_path,
            drop_block_rate=None,
        )
        if self.model_ema is True:
            self.model_ema = ModelEma(
                self.model,
                decay=self.model_ema_decay,
                device='cpu' if self.model_ema_force_cpu else next(self.model.parameters()).device,
                resume=''
            )
        else:
            self.model_ema = None

        if self.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif self.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        self.teacher_model = None
        if self.distillation_type != 'none':
            assert self.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {self.teacher_model}")
            self.teacher_model = create_model(
                self.teacher_model,
                pretrained=False,
                num_classes=self.nb_classes,
                global_pool='avg',
            )
            if self.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.teacher_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['model'])
            self.teacher_model.eval()

        self.train_criterion = DistillationLoss(
            criterion, 
            self.teacher_model, 
            self.distillation_type, 
            self.distillation_alpha, 
            self.distillation_tau
        )
        self.eval_criterion = torch.nn.CrossEntropyLoss()

        self.epoch = 0

    def load_finetune_checkpoint(self):
        pass 

    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_start(self):
        pass 
    
    def on_train_epoch_end(self, outputs):
        self.epoch += 1
        #self.lr_scheduler.step(epoch)

    def training_step(self, batch, batch_idx):
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        samples, targets = batch[0], batch[1]
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.model(samples)
        loss = self.train_criterion(samples, outputs, targets)
        # TODO: how to configure the customized optimization step, e.g. 
        # https://github.com/rwightman/pytorch-image-models/blob/80078c47bb5b60d337f294594433bbd6809f3975/timm/utils/cuda.py#L34
        # in timm?
        # Possivle solution, 'def optimizer_step'
        # reference: https://pytorch-lightning.readthedocs.io/en/0.7.0/optimizers.html
        # 
        self.log('train_loss', loss)
        return loss 

    # TODO: do we need to no_grad(), or with torch.cuda.amp.autocast()?
    # TODO: how to return accuracy? https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/6c307c1292d26562b849c35695819cc93c104a8f/pl_bolts/models/self_supervised/moco/moco2_module.py#L291
    def validation_step(self, batch, batch_idx):
        samples, targets = batch[0], batch[1]
        outputs = self.model(samples)
        loss = self.eval_criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        self.log('valid_loss', loss)
        self.log('valid_acc1', acc1)
        self.log('valid_acc5', acc5)
        return loss
    
    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.opt, 
            self.lr, 
            self.weight_decay, 
            self.opt_eps, 
            opt_betas=None, 
            momentum=self.momentum, 
            model=self.model
        )
        #optimizer = create_optimizer(args, self.model)
        return optimizer
