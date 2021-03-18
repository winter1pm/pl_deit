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

import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, MISSING, OmegaConf 
from pytorch_lightning.callbacks import (
    LearningRateMonitor, 
    ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from src import DeiT, ImagenetDataModule


HYDRA_FULL_ERROR=1


@hydra.main(config_path="conf", config_name="deit/config")
def main(cfg: DictConfig) -> None:
    print(cfg)

    #if cfg.model.distillation_type != 'none' and cfg.model.finetune and not cfgmodel.eval:
    #    raise NotImplementedError("Finetuning with distillation not yet supported")

    print('sanity check config:')
    print(cfg)
    
    deit_model = DeiT(cfg)
    data_module = ImagenetDataModule(**OmegaConf.to_container(cfg.dataset))
    
    trainer_params = OmegaConf.to_container(cfg.trainer, resolve=True)

    callbacks = trainer_params.get("callbacks", [])
    callbacks.append(LearningRateMonitor())
    trainer_params["callbacks"] = callbacks
    
    trainer_params["checkpoint_callback"] = ModelCheckpoint(
        monitor='Loss/val',
        dirpath='/home/dongguo/fastProjects/pl_deit/train_ckpts',
        filename='sample-deit-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min',
    )

    # TODO: config logger directory and name in conf/logger
    trainer_params["logger"] = TensorBoardLogger(
        save_dir='/home/dongguo/fastProjects/pl_deit/train_ckpts/tb_logs', 
        name=cfg.model.model_name
    )
    trainer = pl.Trainer(**trainer_params)
    
    trainer.fit(deit_model, datamodule=data_module)
    trainer.test(deit_model, datamodule=data_module)
    

if __name__ == '__main__':
    main()
