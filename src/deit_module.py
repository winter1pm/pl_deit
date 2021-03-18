import os
import logging

import hydra 
import pytorch_lightning as pl
import torch

from .deit_lib import create_vit_model, create_teacher_model
from .losses import DistillationLoss
from omegaconf import DictConfig, OmegaConf 
from pytorch_lightning.utilities import AMPType
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, accuracy
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional

class DeiT(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.mixup_fn = None
        self.mixup_active = (
            self.cfg.model.mixup > 0 
            or self.cfg.cutmix > 0. 
            or self.cfg.model.cutmix_minmax is not None
        )
        if self.mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=self.cfg.model.mixup, 
                cutmix_alpha=self.cfg.model.cutmix, 
                cutmix_minmax=self.cfg.model.cutmix_minmax,
                prob=self.cfg.model.mixup_prob, 
                switch_prob=self.cfg.model.mixup_switch_prob, 
                mode=self.cfg.model.mixup_mode,
                label_smoothing=self.cfg.model.smoothing, 
                num_classes=self.cfg.model.num_classes
            )

        self.model = create_vit_model(
            model_name=self.cfg.model.model_name,
            pretrained=False,
            num_classes=self.cfg.model.num_classes,
            drop_rate=self.cfg.model.drop,
            drop_path_rate=self.cfg.model.drop_path,
        )
        # TODO: there is an option to initialize with pretrained model. It 
        # has lower priority at this time. Add it later. 
        if self.cfg.model.model_ema is True:
            self.model_ema = ModelEma(
                self.model,
                decay=self.cfg.model.model_ema_decay,
                device='cpu' if self.cfg.model.model_ema_force_cpu else next(self.model.parameters()).device,
                resume=''
            )
        else:
            self.model_ema = None

        if self.cfg.model.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif self.cfg.model.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=self.cfg.model.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        self.teacher_model = None
        if self.cfg.model.distillation_type != 'none':
            self.teacher_model = create_teacher_model(
                model_name=self.cfg.model.teacher_model,
                pretrained=False,
                num_classes=self.cfg.model.num_classes,
                global_pool='avg',
            )
            
            checkpoint = torch.load(os.path.join(
                self.cfg.ckpts.pretrained_ckpt_path, 
                self.cfg.ckpts.pretrained_models[self.cfg.model.teacher_model]
            ))
            try:
                self.teacher_model.load_state_dict(checkpoint['model'])
            except:
                self.teacher_model.load_state_dict(checkpoint)
            self.teacher_model.eval()

        self.train_criterion = DistillationLoss(
            criterion, 
            self.teacher_model, 
            self.cfg.model.distillation_type, 
            self.cfg.model.distillation_alpha, 
            self.cfg.model.distillation_tau
        )
        self.eval_criterion = torch.nn.CrossEntropyLoss()
        self.epoch = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # TODO: run model ema experiment
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        samples, targets = batch[0], batch[1]
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.model(samples)
        loss = self.train_criterion(samples, outputs, targets)
        if isinstance(outputs, torch.Tensor):
            acc1, acc5 = accuracy(outputs, batch[1], topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs[0], batch[1], topk=(1, 5))
        self.log('train_loss', loss, on_epoch=True, logger=True)
        # TODO: rethink about on_step and on_epoch options here.
        # TODO: e.g.1. is on_step = True by default?
        # TODO: e.g.2. is on_epoch = True useful?
        # self.log('lr', optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # TODO: remove this assert after experiment run
        assert self.model.training == False
        samples, targets = batch[0], batch[1]
        outputs = self.model(samples)
        loss = self.eval_criterion(outputs, targets)

        if isinstance(outputs, torch.Tensor):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs[0], targets, topk=(1, 5))
        self.log('Loss/val', loss, on_epoch=True, logger=True)
        self.log('Accuracy/top1_val', acc1, on_epoch=True, logger=True)
        self.log('Accuracy/top5_val', acc5, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        # TODO: remove this assert after experiment run
        assert self.model.training == False
        samples, targets = batch[0], batch[1]
        outputs = self.model(samples)
        loss = self.eval_criterion(outputs, targets)

        if isinstance(outputs, torch.Tensor):
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(outputs[0], targets, topk=(1, 5))
        self.log('Loss/test', loss, on_epoch=True, logger=True)
        self.log('Accuracy/top1_test', acc1, on_epoch=True, logger=True)
        self.log('Accuracy/top5_test', acc5, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optim, self.model)
        lr_scheduler, _ = create_scheduler(self.cfg.scheduler, optimizer)
        return [optimizer], [lr_scheduler]
        
    """
    # In first (successful) version, do not use mix precision training
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        # reference_url = 
        # "https://github.com/PyTorchLightning/lightning-bolts/blob/\
        # 61d3a26f45b8f01232e76e18d22d55eecb4e6c77/pl_bolts/models/\
        # self_supervised/simsiam/simsiam_module.py#L268"
        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
    """
