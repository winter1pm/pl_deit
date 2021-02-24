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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

import src
#import src.models.deit_models.DeiT

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg)

    if cfg.model.distillation_type != 'none' and cfg.model.finetune and not cfgmodel.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    print('model config')
    print(cfg.model)

    print('\ntrainer config')
    print(cfg.trainer)

    print('\ndata config')
    print(cfg.datamodule)

    print('Instantiating DeiT model')
    deit_model = hydra.utils.instantiate(cfg.model)

    print('Instantiating IMNET dataset')
    data_module = hydra.utils.instantiate(cfg.datamodule)

    
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='/home/dongguo/fastProjects/pl_deit/ckpts',
        filename='sample-deit-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    #args.callbacks=[checkpoint_callback]
    #args.max_epochs = args.epochs
    trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback])

    #print(f'calling trainer.fit on device {dist.get_rank()}')
    trainer.fit(deit_model, datamodule=data_module)
    

if __name__ == '__main__':
    #if args.output_dir:
    #    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main()
