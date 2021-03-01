import collections

from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf 
from typing import List, Any, Optional

import hydra 
from hydra.core.config_store import ConfigStore 

HYDRA_FULL_ERROR=1


project_default = [
    {"optimizer": "adamw"},
    {"scheduler": "cosine"},
    {"model": "deit_tiny_patch16_224"},
    {"data": "imnet_data"},
    {"trainer": "gpus_trainer"},
    {"logger": "tblogger"}
]


@dataclass 
class OptimizerConfig:
    opt: str = 'adamw'
    lr: float = 0.0005
    opt_eps: float = 1e-08
    weight_decay: float = 0.05
    opt_betas: Optional[float] = MISSING
    momentum: Optional[float] = MISSING


@dataclass 
class SchedulerConfig:
    sched: str = "cosine"
    lr_noise: Optional[float] = MISSING
    lr_noise_pct: float = 0.67
    lr_noise_std: float = 1.0
    warmup_lr: float = 1e-06
    min_lr: float = 1e-05
    decay_rate: float = 0.1
    epochs: int = 300
    decay_epochs: int = 30
    warmup_epochs: int = 5
    cooldown_epochs: int = 10


@dataclass 
class DataConfig:
    _target_: str = "src.datasets.ImagenetDataModule"
    data_dir: str = "/home/dongguo/FastData/ImageNet/images"
    batch_size: int = 64
    image_size: int = 224
    nb_classes: int = 2000
    color_jitter: float = 0.4
    aa: str = "rand-m9-mstd0.5-inc1"
    train_interpolation: str = "bicubic"
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    repeated_aug: bool = True
    num_workers: int = 10
    pin_mem: bool = True


@dataclass 
class TrainerConfig:
    gpus: int = 2
    accelerator: str = "ddp"
    replace_sampler_ddp: bool = False
    max_epochs: int = 300


@dataclass 
class LoggerConfig:
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"
    tb_dir: str = "tb_ckpts"


@dataclass 
class ModelConfig:
    model_arch: str = "deit_tiny_patch16_224"
    mixup: float = 0.8
    cutmix: float = 1.0
    cutmix_minmax: Optional[float] = MISSING
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    smoothing: float = 0.1
    nb_classes: int = 2000
    drop: float = 0.0
    drop_path: float = 0.1
    model_ema: bool = False
    model_ema_decay: float = 0.99996
    model_ema_force_cpu: bool = False
    distillation_type: str = "none"
    distillation_alpha: float = 0.5
    distillation_tau: float = 1.0
    teacher_model: str = "regnety_160"
    teacher_path: str = ""


@dataclass 
class DeiTConfig:
    optimizer: OptimizerConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    model: ModelConfig = MISSING 
    data: DataConfig = MISSING 
    trainer: TrainerConfig = MISSING
    logger: LoggerConfig = MISSING 
    defaults: List[Any] = field(default_factory=lambda: project_default)


cs = ConfigStore.instance()
cs.store(group="optimizer", name="adamw", node=OptimizerConfig)
cs.store(group="scheduler", name="cosine", node=SchedulerConfig)
cs.store(group="model", name="deit_tiny_patch16_224", node=ModelConfig)
cs.store(group="data", name="imnet_data", node=DataConfig)
cs.store(group="trainer", name="gpus_trainer", node=TrainerConfig)
cs.store(group="logger", name="tblogger", node=LoggerConfig)
cs.store(name="deit", node=DeiTConfig)