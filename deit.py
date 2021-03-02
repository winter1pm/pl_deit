import hydra 
import pytorch_lightning as pl
import torch

from conf import DeiTConfig
from src.models.vits import (
    create_model, 
    DistillationLoss, 
    LabelSmoothingCrossEntropy, 
    SoftTargetCrossEntropy
)
from src.optim import create_optimizer
from src.scheduler import create_scheduler
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

class DeiT(pl.LightningModule):

    def __init__(self, cfg: DeiTConfig):
        super().__init__()
        self.cfg = cfg

        self.mixup_fn = None
        self.mixup_active = (
            self.cfg.model.mixup > 0 or self.cfg.cutmix > 0. or self.cfg.model.cutmix_minmax is not None
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
                num_classes=self.cfg.model.nb_classes
            )

        self.model = create_model(
            model_arch=self.cfg.model.model_arch,
            pretrained=False,
            num_classes=self.cfg.model.nb_classes,
            drop_rate=self.cfg.model.drop,
            drop_path_rate=self.cfg.model.drop_path,
        )
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
            assert self.cfg.model.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {self.cfg.model.teacher_model}")
            self.teacher_model = create_model(
                self.cfg.model.teacher_model,
                pretrained=False,
                num_classes=self.cfg.model.nb_classes,
                global_pool='avg',
            )
            if self.cfg.model.teacher_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.cfg.model.teacher_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(self.cfg.model.teacher_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['model'])
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
        if self.model_ema is not None:
            self.model_ema.update(self.model)

        samples, targets = batch[0], batch[1]
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
        outputs = self.model(samples)
        loss = self.train_criterion(samples, outputs, targets)
        # TODO: specify train_cls_loss and train_distill_loss
        tensorboard_logs = {'train_tb_loss': loss}
        progress_bar_metrics = tensorboard_logs

        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def validation_step(self, batch, batch_idx):
        samples, targets = batch[0], batch[1]
        outputs = self.model(samples)
        loss = self.eval_criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        tensorboard_logs = {
            'valid_loss': loss, 
            'valid_acc1': acc1, 
            'valid_acc5': acc5
        }
        progress_bar_metrics = tensorboard_logs
        return {'val_loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}
    
    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer, self.model)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.scheduler.epochs)
        return [optimizer], [lr_scheduler]
        #return optimizer
        #lr_scheduler, _ = create_scheduler(self.cfg.scheduler, optimizer)

        

