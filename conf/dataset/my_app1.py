# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from omegaconf import DictConfig, OmegaConf

import hydra


@hydra.main(config_name="imnet_online")
def my_app(cfg: DictConfig) -> None:
    print('YAML format')
    print(OmegaConf.to_yaml(cfg))

    print('flatten format')
    print(OmegaConf.to_container(cfg))

if __name__ == "__main__":
    my_app()
