import os
import time

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import ImageFolder

from .data_utils import build_transform, RASampler


def build_imnet_dataset(
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

    #root = os.path.join(data_dir, 'train' if is_train else 'val')
    root = os.path.join(data_dir, 'val' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transforms)
    nb_classes = 1000

    return dataset, nb_classes


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
        data_dir,
        nb_classes,
        image_size,
        color_jitter,
        aa,
        train_interpolation,
        reprob,
        remode,
        recount,
        repeated_aug,
        batch_size,
        num_workers,
        pin_mem,
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
        self.data_dir = data_dir
        self.nb_classes = nb_classes
        self.image_size = image_size
        self.color_jitter = color_jitter
        self.aa = aa
        self.train_interpolation = train_interpolation
        self.reprob= reprob
        self.remode = remode
        self.recount= recount
        self.repeated_aug = repeated_aug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_mem = pin_mem

    @property
    def num_classes(self) -> int:
        """
        Return:
            number of classes (1000 in ImageNet dataset)
        """
        return self.nb_classes

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
        #print(f'running prepare_data() on device {dist.get_rank()}')
        print(f'running prepare_data() at {time.time()}')
        self._verify_splits(self.data_dir, 'train')
        self._verify_splits(self.data_dir, 'val')
        print(f'Data prepared (downloaded and split')

    def setup(self, stage):
        print(f'running setup() on device {dist.get_rank()}, at time {time.time()}')
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        self.dataset_train, self.nb_classes = build_imnet_dataset(
            is_train=True, 
            data_dir=self.data_dir,
            image_size=self.image_size,
            color_jitter=self.color_jitter,
            aa=self.aa,
            train_interpolation=self.train_interpolation,
            reprob=self.reprob,
            remode=self.remode,
            recount=self.recount
        )

        self.dataset_valid, _ = build_imnet_dataset(
            is_train=False, 
            data_dir=self.data_dir,
            image_size=self.image_size,
            color_jitter=self.color_jitter,
            aa=self.aa,
            train_interpolation=self.train_interpolation,
            reprob=self.reprob,
            remode=self.remode,
            recount=self.recount
        )

        print(
            'Train & valid datasets prepared, of size '
            + f'{len(self.dataset_train)} and {len(self.dataset_valid)} respectively.'
        )

        if self.repeated_aug:
            self.sampler_train = RASampler(
                self.dataset_train, num_replicas=None, rank=None
            )
        else:
            self.sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=None, rank=None, shuffle=True
            )
        print(f'Deit Model setup on device {dist.get_rank()}, sampler_train instatiated, at time {time.time()}')

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
            sampler=self.sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
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
            batch_size=int(1.5 * self.batch_size),
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False
        )
        return loader_val
