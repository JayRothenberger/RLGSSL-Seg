import os
import torch
import numpy as np
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
import gc
from sklearn.model_selection import train_test_split
import pickle
import random
from collections import Counter
from functorch.experimental.control_flow import map
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import Any, Dict, List
from copy import deepcopy as copy

SPEC_DIR = {
    "siglipv2_tree": {
        "unlabeled": {"split": "unlabeled"},
        "train": {"split": "train"},
        "test": {"split": "val"},
        "classes": 38,
        "download_path": "../",
    },
    "dinov3_tree": {
        "unlabeled": {"split": "unlabeled"},
        "train": {"split": "train"},
        "test": {"split": "val"},
        "classes": 38,
        "download_path": "../",
    },
    "pascal_voc_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 21,
        "download_path": "../",
    },
    "in1k_dinov2_1": {
        "unlabeled": {"split": "unlabeled_1"},
        "train": {"split": "train_1"},
        "test": {"split": "val_1"},
        "classes": 1000,
        "download_path": "../",
    },
    "in1k_clip_1": {
        "unlabeled": {"split": "unlabeled_1"},
        "train": {"split": "train_1"},
        "test": {"split": "val_1"},
        "classes": 1000,
        "download_path": "../",
    },
    "in1k_siglipv2_1": {
        "unlabeled": {"split": "unlabeled_1"},
        "train": {"split": "train_1"},
        "test": {"split": "val_1"},
        "classes": 1000,
        "download_path": "../",
    },
    "in1k_dinov3_1": {
        "unlabeled": {"split": "unlabeled_1"},
        "train": {"split": "train_1"},
        "test": {"split": "val_1"},
        "classes": 1000,
        "download_path": "../",
    },
    "inat_siglipv2_1010": {
        "unlabeled": {"split": "unlabeled_1010"},
        "train": {"split": "train_1010"},
        "test": {"split": "val_1010"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_dinov3_1010": {
        "unlabeled": {"split": "unlabeled_1010"},
        "train": {"split": "train_1010"},
        "test": {"split": "val_1010"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_siglipv2_3030": {
        "unlabeled": {"split": "unlabeled_3030"},
        "train": {"split": "train_3030"},
        "test": {"split": "val_3030"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_dinov3_3030": {
        "unlabeled": {"split": "unlabeled_3030"},
        "train": {"split": "train_3030"},
        "test": {"split": "val_3030"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_siglipv2_5050": {
        "unlabeled": {"split": "unlabeled_5050"},
        "train": {"split": "train_5050"},
        "test": {"split": "val_5050"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_dinov3_5050": {
        "unlabeled": {"split": "unlabeled_5050"},
        "train": {"split": "train_5050"},
        "test": {"split": "val_5050"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_siglipv2_10100": {
        "unlabeled": {"split": "unlabeled_10100"},
        "train": {"split": "train_10100"},
        "test": {"split": "val_10100"},
        "classes": 1010,
        "download_path": "../",
    },
    "inat_dinov3_10100": {
        "unlabeled": {"split": "unlabeled_10100"},
        "train": {"split": "train_10100"},
        "test": {"split": "val_10100"},
        "classes": 1010,
        "download_path": "../",
    },
    "food101_siglipv2_101": {
        "unlabeled": {"split": "unlabeled_101"},
        "train": {"split": "train_101"},
        "test": {"split": "val_101"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_dinov3_101": {
        "unlabeled": {"split": "unlabeled_101"},
        "train": {"split": "train_101"},
        "test": {"split": "val_101"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_siglipv2_303": {
        "unlabeled": {"split": "unlabeled_303"},
        "train": {"split": "train_303"},
        "test": {"split": "val_303"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_dinov3_303": {
        "unlabeled": {"split": "unlabeled_303"},
        "train": {"split": "train_303"},
        "test": {"split": "val_303"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_siglipv2_505": {
        "unlabeled": {"split": "unlabeled_505"},
        "train": {"split": "train_505"},
        "test": {"split": "val_505"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_dinov3_505": {
        "unlabeled": {"split": "unlabeled_505"},
        "train": {"split": "train_505"},
        "test": {"split": "val_505"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_siglipv2_1010": {
        "unlabeled": {"split": "unlabeled_1010"},
        "train": {"split": "train_1010"},
        "test": {"split": "val_1010"},
        "classes": 101,
        "download_path": "../",
    },
    "food101_dinov3_1010": {
        "unlabeled": {"split": "unlabeled_1010"},
        "train": {"split": "train_1010"},
        "test": {"split": "val_1010"},
        "classes": 101,
        "download_path": "../",
    },
    "in1k_dinov2_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 1000,
        "download_path": "../",
    },
    "in1k_clip_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 1000,
        "download_path": "../",
    },
    "svhn_siglipv2_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_dinov3_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_siglipv2_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_dinov3_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_siglipv2_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_dinov3_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_siglipv2_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "svhn_dinov3_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_siglipv2_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_dinov3_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_siglipv2_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_dinov3_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_siglipv2_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_dinov3_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_siglipv2_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "stl10_dinov3_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_siglipv2_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_dinov3_10": {
        "unlabeled": {"split": "unlabeled_10"},
        "train": {"split": "train_10"},
        "test": {"split": "val_10"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_siglipv2_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_dinov3_30": {
        "unlabeled": {"split": "unlabeled_30"},
        "train": {"split": "train_30"},
        "test": {"split": "val_30"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_siglipv2_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_dinov3_50": {
        "unlabeled": {"split": "unlabeled_50"},
        "train": {"split": "train_50"},
        "test": {"split": "val_50"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_siglipv2_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar10_dinov3_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_siglipv2_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_dinov3_100": {
        "unlabeled": {"split": "unlabeled_100"},
        "train": {"split": "train_100"},
        "test": {"split": "val_100"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_siglipv2_300": {
        "unlabeled": {"split": "unlabeled_300"},
        "train": {"split": "train_300"},
        "test": {"split": "val_300"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_dinov3_300": {
        "unlabeled": {"split": "unlabeled_300"},
        "train": {"split": "train_300"},
        "test": {"split": "val_300"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_siglipv2_500": {
        "unlabeled": {"split": "unlabeled_500"},
        "train": {"split": "train_500"},
        "test": {"split": "val_500"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_dinov3_500": {
        "unlabeled": {"split": "unlabeled_500"},
        "train": {"split": "train_500"},
        "test": {"split": "val_500"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_siglipv2_1000": {
        "unlabeled": {"split": "unlabeled_1000"},
        "train": {"split": "train_1000"},
        "test": {"split": "val_1000"},
        "classes": 10,
        "download_path": "../",
    },
    "cifar100_dinov3_1000": {
        "unlabeled": {"split": "unlabeled_1000"},
        "train": {"split": "train_1000"},
        "test": {"split": "val_1000"},
        "classes": 10,
        "download_path": "../",
    },
    "caltech101": {
        "unlabeled": None,
        "train": None,
        "test": None,
        "classes": 101,
        "download_path": "/ourdisk/hpc/ai2es/datasets/caltech101",
    },
    "caltech256": {
        "unlabeled": None,
        "train": None,
        "test": None,
        "classes": 257,
        "download_path": "/ourdisk/hpc/ai2es/datasets/caltech256",
    },
    "food101": {
        "unlabeled": None,
        "train": {"split": "train"},
        "test": {"split": "test"},
        "classes": 101,
        "download_path": "/ourdisk/hpc/ai2es/datasets/Food101",
    },
    "inat2021": {
        "unlabeled": None,
        "train": {"version": "2021_train"},
        "test": {"version": "2021_train"},
        "classes": 10_000,
        "download_path": "/ourdisk/hpc/ai2es/datasets/iNat2021",
    },
    "cifar10": {
        "unlabeled": None,
        "train": {"train": True},
        "test": {"train": False},
        "classes": 10,
        "download_path": "../",  # "/ourdisk/hpc/ai2es/datasets/CIFAR10",
    },
    "CIFAR10_DINOv2": {
        "unlabeled": None,
        "train": {"split": "train"},
        "test": {"split": "val"},
        "classes": 10,
        "download_path": "../",  # "/ourdisk/hpc/ai2es/datasets/CIFAR10",
    },
    "mnist": {
        "unlabeled": None,
        "train": {"train": True},
        "test": {"train": False},
        "classes": 10,
        "download_path": "../mnist",  # "/ourdisk/hpc/ai2es/datasets/MNIST",
    },
    "cifar100": {
        "unlabeled": None,
        "train": {"train": True},
        "test": {"train": False},
        "classes": 100,
        "download_path": "../",  # "/ourdisk/hpc/ai2es/datasets/CIFAR100",
    },
}


def equal_class_counts_inds(labels, count):
    # identify unique elements
    unique, inverse = torch.unique(torch.tensor(labels), return_inverse=True)

    # interleave unique elements
    unique_inds = []
    for v in unique:
        unique_inds.append(torch.arange(labels.shape[0])[inverse == v])

    smallest_frequency = min([len(u) for u in unique_inds])
    total = smallest_frequency * len(unique_inds)
    
    interleaved = torch.stack([u[:smallest_frequency] for u in unique_inds], dim=0).permute(1, 0).reshape(total, -1).squeeze()

    assert interleaved.shape == (total, ), (interleaved.shape, labels.shape)

    return [int(y) for y in interleaved[:count]]


class IndexSubsetDataset:
    def __init__(self, ds, inds):
        self.ds = ds
        self.inds = inds

    def __getitem__(self, i, index=False):
        if index:
            return self.inds[i], *self.ds[self.inds[i]]

        return self.ds[self.inds[i]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.inds)


class IndexedDataset:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        return i, *self.ds[i]

    def __iter__(self):
        for x in self:
            yield x

    def __len__(self):
        return len(self.ds)


class ConcatDataset:
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

    def __getitem__(self, i):
        if i >= len(self.ds1):
            return self.ds2[i - len(self.ds1)]
        return self.ds1[i]

    def __iter__(self):
        for x in self:
            yield x

    def __len__(self):
        return len(self.ds1) + len(self.ds2)


class IndexedIndexSubsetDataset:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        return self.ds.__getitem__(i, True)

    def __iter__(self):
        for x in self:
            yield x

    def __len__(self):
        return len(self.ds)


class InfiniteDataset:
    def __init__(self, ds, seed=None, index=False):
        self.ds = ds
        self.index = index

        self.rng = np.random.default_rng(seed=seed)

    def __getitem__(self, i):
        ind = self.rng.integers(0, len(self.ds) - 1, (1, ))[0]

        if not self.index:
            return self.ds[ind]
        
        return ind, *self.ds[ind]

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

    def __len__(self):
        return int(2**31)


class WeakStrongDataset:
    def __init__(self, ds, weak, strong):
        self.ds = ds
        self.weak = weak
        self.strong = strong

    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.weak(x), self.strong(x), y

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

    def __len__(self):
        return len(self.ds)


class IterLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for b in self.loader:
            yield b

    def __next__(self):
        for b in self.loader:
            yield b


class BaseDataset:
    def __init__(
        self,
        args,
        preprocessing_transform,
        weak_augmentation_transform=torch.nn.Identity(),
        strong_augmentation_transform=torch.nn.Identity(),
        infinite_index=False
    ):
        """
        this object is used for both the labeled and unlabeled datasets

        It should be understood that the preprocessing transform is always applied and then the weak and strong augmentations are applied separately.
        """
        self.args = args
        self.infinite_index = infinite_index

        if int(int(os.environ["WORLD_SIZE"]) / 2) > 0:
            self.seed = int(os.environ['RANK']) % int(int(os.environ["WORLD_SIZE"]) / 2)
        else:
            self.seed = int(os.environ['RANK'])

        # we will use the weak augmentations only here and apply the strong augmentations later upon request
        self.ds_unlabeled, self.ds_train, self.ds_val = select_dataset(
            args,
            DATA_DIR,
            SPEC_DIR,
            preprocessing_transform,
        )

        self.ds_unlabeled, self.ds_train = (
            WeakStrongDataset(
                self.ds_unlabeled,
                weak_augmentation_transform,
                strong_augmentation_transform,
            ),
            WeakStrongDataset(
                self.ds_train,
                weak_augmentation_transform,
                strong_augmentation_transform,
            ),
        )

        self.labeled = IndexSubsetDataset(self, list(range(len(self.ds_train))))
        self.unlabeled = IndexSubsetDataset(
            self,
            list(
                range(len(self.ds_train), len(self.ds_train) + len(self.ds_unlabeled))
            ),
        )
        self.validation = self.ds_val

        self.pseudo_labels = torch.full((len(self.unlabeled), *self.labeled[0][-1].shape), -1)

        self.samplers = None
        self.loaders = None
        self.iterable_loaders = None

        self.infinite_labeled = InfiniteDataset(self.labeled, seed=self.seed, index=self.infinite_index)
        self.infinite_unlabeled = InfiniteDataset(ConcatDataset(self.unlabeled, self.labeled), seed=self.seed, index=self.infinite_index)

    def get_unlabeled(self, i):
        return self.unlabeled[i], -1

    def get_labeled(self, i):
        return self.labeled[i]

    def get_validation(self, i):
        return self.ds_val[i]

    def add_pl_to_labeled(self, labels, inds):
        assert labels.shape[0] == inds.shape[0], (
            f"labels and indicies must have the same size along axis 0, found {labels.shape[0]} and {inds.shape[0]}"
        )
        self.pseudo_labels[inds] = labels

        inds = [int(i + len(self.ds_train)) for i in inds]

        labeled_inds = set(self.labeled.inds)
        labeled_inds.update(set(inds))

        self.labeled.inds = list(labeled_inds)

        if set(inds):
            assert set(inds).issubset(set(self.unlabeled.inds)), (set(inds) & set(self.unlabeled.inds))

        torch.distributed.barrier()

        self.unlabeled.inds = list(set(self.unlabeled.inds) - set(inds))

        self.validation = self.ds_val

        self.samplers = None
        self.loaders = None
        self.iterable_loaders = None

        self.infinite_labeled = InfiniteDataset(self.labeled, seed=self.seed, index=self.infinite_index)
        self.infinite_unlabeled = InfiniteDataset(ConcatDataset(self.unlabeled, self.labeled), seed=self.seed, index=self.infinite_index)

    def clear_loaders(self):
        del self.samplers
        del self.loaders
        del self.iterable_loaders

        self.samplers = None
        self.loaders = None
        self.iterable_loaders = None

        gc.collect()

        torch.distributed.barrier()


    def clear_pl(self):
        self.labeled = IndexSubsetDataset(self, list(range(len(self.ds_train))))
        self.unlabeled = IndexSubsetDataset(
            self,
            list(
                range(len(self.ds_train), len(self.ds_train) + len(self.ds_unlabeled))
            ),
        )
        self.validation = self.ds_val

        self.pseudo_labels = torch.full((len(self.unlabeled), *self.labeled[0][1].shape), -1)

        self.clear_loaders()

        self.infinite_labeled = InfiniteDataset(self.labeled, seed=self.seed, index=self.infinite_index)
        self.infinite_unlabeled = InfiniteDataset(ConcatDataset(self.unlabeled, self.labeled), seed=self.seed, index=self.infinite_index)

        # self.get_loaders()

    def get_loaders(self):
        if self.loaders is not None:
            return self.loaders
        # return loaders of the infinite versions of the datasets
        samplers = self.get_samplers()
        # set the loaders
        self.loaders = {
            "labeled": DataLoader(
                self.infinite_labeled,
                batch_size=self.args.train_batch_size,
                shuffle=False,
                sampler=None, # samplers["labeled"],
                num_workers=self.args.train_workers,
                drop_last=True
            ),
            "unlabeled": DataLoader(
                self.infinite_unlabeled,
                batch_size=self.args.unlabeled_batch_size,
                shuffle=False,
                sampler=None, # samplers["unlabeled"],
                num_workers=self.args.unlabeled_workers,
                drop_last=True
            ),
            "validation": DataLoader(
                self.validation,
                batch_size=self.args.validation_batch_size,
                shuffle=False,
                sampler=samplers["validation"],
                num_workers=self.args.validation_workers,
            ),
        }

        self.iterable_loaders = {k: iter(v) for k, v in self.loaders.items()}

        return self.loaders

    def get_samplers(self):
        # this needs to not return new samplers but allow the modification of sampler state
        # for the single view datasets we do not need to worry too much about which process we are with the sampler
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        self.samplers = {
            "labeled": DistributedSampler(
                self.infinite_labeled,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=13,
            ),
            "unlabeled": DistributedSampler(
                self.infinite_unlabeled,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=13,
            ),
            "validation": DistributedSampler(
                self.validation,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=13,
            ),
        }

        return self.samplers

    def __getitem__(self, i):
        if i < len(self.ds_train):
            return tuple([torch.tensor(x) for x in self.ds_train[i]])
        elif i < (len(self.ds_train) + len(self.ds_unlabeled)):
            x, x_aug, _ = self.ds_unlabeled[i - len(self.ds_train)]
            assert _.shape == torch.tensor(self.pseudo_labels[i - len(self.ds_train)].type(torch.long)).shape, (_.shape, torch.tensor(self.pseudo_labels[i - len(self.ds_train)].type(torch.long)).shape)
            return x, x_aug, torch.tensor(self.pseudo_labels[i - len(self.ds_train)].type(torch.long))
        else:
            raise IndexError("Index out of bounds.")
        
    def __len__(self):
         return len(self.ds_train) + len(self.ds_unlabeled)


class RLPLDataset(BaseDataset):
    def __init__(
        self,
        args,
        preprocessing_transform,
        weak_augmentation_transform=torch.nn.Identity(),
        strong_augmentation_transform=torch.nn.Identity(),
        rl_fraction=-1,
        infinite_index=False
    ):
        
        super().__init__(
            args,
            preprocessing_transform,
            weak_augmentation_transform=weak_augmentation_transform,
            strong_augmentation_transform=strong_augmentation_transform,
            infinite_index=infinite_index
        )

        self.infinite_index = infinite_index
        self.rl_fraction = rl_fraction

        # Extract labels from the dataset
        labels = [y for _, _, y in self.ds_train]

        # Generate indices for splitting
        indices = list(range(len(self.ds_train)))

        # Perform stratified split
        if 0 < self.rl_fraction < 1:
            rl_inds, train_inds = train_test_split(
                indices, test_size=(1 - self.rl_fraction), random_state=13, stratify=labels
            )
        else:
            rl_inds, train_inds = indices[::5], indices

        ds_rl = IndexSubsetDataset(self.ds_train, rl_inds)
        self.ds_train = IndexSubsetDataset(self.ds_train, train_inds)

        self.rl_data = (
            torch.stack([x for x, x_aug, y in ds_rl]).to(torch.cuda.current_device()),
            torch.tensor([y for x, x_aug, y in ds_rl], dtype=torch.int64).to(
                torch.cuda.current_device()
            ),
        )

        self.labeled = IndexSubsetDataset(self, list(range(len(self.ds_train))))
        self.unlabeled = IndexSubsetDataset(
            self,
            list(
                range(len(self.ds_train), len(self.ds_train) + len(self.ds_unlabeled))
            ),
        )

        self.pseudo_labels = torch.full((len(self.unlabeled),), -1)

        self.infinite_labeled = InfiniteDataset(self.labeled, seed=self.seed, index=self.infinite_index)
        self.infinite_unlabeled = InfiniteDataset(ConcatDataset(self.unlabeled, self.labeled), seed=self.seed, index=self.infinite_index)


class DINOv2ImageNetDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        lock_path = os.path.join(download_path, 'download.lock')
        if os.path.isfile(lock_path):
            os.remove(lock_path)

        # check if the download path already exists

        assert os.path.isdir(download_path)

        # load the files

        with open(os.path.join(download_path, 'IN1k', 'DINOv2_IN1K_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'IN1k', 'DINOv2_IN1K_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        splits = ['train_1', 'train_10', 'val_1', 'val_10', 'unlabeled_1', 'unlabeled_10']

        assert split in splits

        if split.endswith('_1'):
            with open(os.path.join(download_path, 'IN1k', '1percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        elif split.endswith('_10'):
            with open(os.path.join(download_path, 'IN1k', '10percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        else:
            idx = torch.arange(X_train.shape[0])

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class VOCLabelTransform():
    mapping: Dict
    
    def __init__(self, rez=520):
        self.rez = rez
        self.mapping = self.build_mapping()
        
    def build_mapping(self):
        return {
            255: 255
        }

    def apply_mapping(self, target):
        arr = np.array(target)
        
        out_arr = arr.copy()
        for old_val, new_val in self.mapping.items():
            # create list of indices we care about for this rule
            idxs = arr == old_val
            out_arr[idxs] = new_val
        
        return torch.tensor(out_arr)
    
    def __call__(self, target):
        return torchvision.transforms.Resize(
            (self.rez, self.rez), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )(torch.tensor(np.array(target)).to(torch.int64).unsqueeze(0)).squeeze(0)


class PascalVOC:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        lock_path = os.path.join(download_path, 'download.lock')
        if os.path.isfile(lock_path):
            os.remove(lock_path)

        # check if the download path already exists

        assert os.path.isdir(download_path)

        # load the dataset for pascal VOC

        rez = 520

        train_trans = torch.nn.Identity()

        val_trans = transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (rez, rez), antialias=True, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        target_trans = transforms.Compose(
            [
                VOCLabelTransform(rez),
            ]
        )

        val_target_trans = transforms.Compose(
            [
                VOCLabelTransform(rez),
            ]
        )

        ds_train = torchvision.datasets.VOCSegmentation(
            download_path,
            image_set="train",
            transform=train_trans,
            target_transform=target_trans,
            download=True,
        )
        ds_val = torchvision.datasets.VOCSegmentation(
            download_path,
            image_set="val",
            transform=val_trans,
            target_transform=val_target_trans,
            download=True,
        )
        
        # check for the 1% or 10% split

        splits = ['train_10', 'val_10', 'unlabeled_10']

        assert split in splits

        if split.endswith('_10'):
            # generate indices for the 10% train
            idx = list(range(len(ds_train)))[::8]
        else:
            idx = list(range(len(ds_train)))
        
        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(len(ds_train)))) - labeled_idx

        assert len(unlabeled_idx) == (len(ds_train) - len(labeled_idx))

        if split.startswith('val'):
            self.ds = ds_val
        elif split.startswith('train'):
            labeled_idx = sorted(list(labeled_idx))
            self.ds = IndexSubsetDataset(ds_train, labeled_idx)
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = IndexSubsetDataset(ds_train, unlbl_idx)
            # for i, (x, y) in enumerate(tqdm(self.ds)):
            #     assert x.shape == self.ds[0][0].shape and y.shape == self.ds[0][1].shape, (i, self.ds[i].shape, )
    
    def __getitem__(self, i):
        x, y = self.ds[i]
        # print(i, x.shape, y.shape)
        return self.transform(x), self.target_transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3TREEDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # check if the download path already exists
        lock_path = os.path.join(download_path, 'download.lock')
        if os.path.isfile(lock_path):
            os.remove(lock_path)


        assert os.path.isdir(download_path)

        # load the files
        splits = ['train', 'val', 'unlabeled']

        assert split in splits

        with open(os.path.join(download_path, 'TREES', 'dino_vitl_train_embeds.pkl'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'TREES', 'dino_vitl_val_embeds.pkl'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        else:
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLIP2TREEDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # check if the download path already exists
        lock_path = os.path.join(download_path, 'download.lock')
        if os.path.isfile(lock_path):
            os.remove(lock_path)


        assert os.path.isdir(download_path)

        # load the files
        splits = ['train', 'val', 'unlabeled']

        assert split in splits

        with open(os.path.join(download_path, 'TREES', 'siglip2_vitl_train_embeds.pkl'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'TREES', 'siglip2_vitl_val_embeds.pkl'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        else:
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3ImageNetDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files

        with open(os.path.join(download_path, 'IN1k', 'DINOv3_IN1K_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'IN1k', 'DINOv3_IN1K_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        splits = ['train_1', 'train_10', 'val_1', 'val_10', 'unlabeled_1', 'unlabeled_10']

        assert split in splits

        if split.endswith('_1'):
            with open(os.path.join(download_path, 'IN1k', '1percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        elif split.endswith('_10'):
            with open(os.path.join(download_path, 'IN1k', '10percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        else:
            idx = torch.arange(X_train.shape[0])

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class CLIPImageNetDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()


        # load the files

        with open(os.path.join(download_path, 'IN1k', 'CLIP_IN1K_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'IN1k', 'CLIP_IN1K_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        splits = ['train_1', 'train_10', 'val_1', 'val_10', 'unlabeled_1', 'unlabeled_10']

        assert split in splits

        if split.endswith('_1'):
            with open(os.path.join(download_path, 'IN1k', '1percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        elif split.endswith('_10'):
            with open(os.path.join(download_path, 'IN1k', '10percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        else:
            idx = torch.arange(X_train.shape[0])

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class SigLiPv2ImageNetDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()


        # load the files

        with open(os.path.join(download_path, 'IN1k', 'SigLiPv2_IN1K_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'IN1k', 'SigLiPv2_IN1K_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        splits = ['train_1', 'train_10', 'val_1', 'val_10', 'unlabeled_1', 'unlabeled_10']

        assert split in splits

        if split.endswith('_1'):
            with open(os.path.join(download_path, 'IN1k', '1percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        elif split.endswith('_10'):
            with open(os.path.join(download_path, 'IN1k', '10percent_idx.pkl'), 'rb') as fp:
                idx = pickle.load(fp)
        else:
            idx = torch.arange(X_train.shape[0])

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLiPv2iNatDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()


        # load the files

        with open(os.path.join(download_path, 'iNat', 'SigLiPv2_iNat_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'iNat', 'SigLiPv2_iNat_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        counts = dict(Counter(y_train))
        most_common = [int(x) for x, y in sorted([(x, y) for (x, y) in counts.items()], reverse=True, key=lambda k: k[-1])[:1010]]

        label_map = {x: i for i, x in enumerate(sorted(list(most_common)))}
        
        train_mask = torch.tensor([True if int(y_train[i]) in most_common else False for i in range(y_train.shape[0])])
        val_mask = torch.tensor([True if int(y_val[i]) in most_common else False for i in range(y_val.shape[0])])

        X_train, y_train = torch.tensor(X_train[train_mask]), torch.tensor([label_map[int(x)] for x in y_train[train_mask]])
        X_val, y_val = torch.tensor(X_val[val_mask]), torch.tensor([label_map[int(x)] for x in y_val[val_mask]])
        
        # check for the 1% or 10% split

        counts = ['_1010', '_3030', '_5050', '_10100', '_30300']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        assert split in splits

        if split.endswith('_1010'):
            idx = equal_class_counts_inds(y_train, 1010)
        elif split.endswith('_3030'):
            idx = equal_class_counts_inds(y_train, 3030)
        elif split.endswith('_5050'):
            idx = equal_class_counts_inds(y_train, 5050)
        elif split.endswith('_10100'):
            idx = equal_class_counts_inds(y_train, 10100)
        elif split.endswith('_30300'):
            idx = equal_class_counts_inds(y_train, 30300)
        else:
            raise NotImplementedError()

        assert torch.equal(torch.unique(y_train[idx]), torch.unique(y_train)), torch.unique(y_train[idx], return_counts=True)

        print(torch.unique(y_train[idx], return_counts=True))

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(X_val, y_val)
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(X_train[idx], y_train[idx])
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(X_train[unlbl_idx], y_train[unlbl_idx])
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class SigLiPv2Food101Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files

        with open(os.path.join(download_path, 'Food101', 'SigLiPv2_Food101_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'Food101', 'SigLiPv2_Food101_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_101', '_303', '_505', '_1010']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_101'):
            idx = equal_class_counts_inds(y_train, 101)
        elif split.endswith('_303'):
            idx = equal_class_counts_inds(y_train, 303)
        elif split.endswith('_505'):
            idx = equal_class_counts_inds(y_train, 505)
        elif split.endswith('_1010'):
            idx = equal_class_counts_inds(y_train, 1010)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3iNatDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files

        with open(os.path.join(download_path, 'iNat', 'DINOv3_iNat_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'iNat', 'DINOv3_iNat_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        counts = dict(Counter(y_train))

        most_common = [int(x) for x, y in sorted([(x, y) for (x, y) in counts.items()], reverse=True, key=lambda k: k[-1])[:1010]]

        label_map = {x: i for i, x in enumerate(sorted(list(most_common)))}
        
        train_mask = torch.tensor([True if int(y_train[i]) in most_common else False for i in range(y_train.shape[0])])
        val_mask = torch.tensor([True if int(y_val[i]) in most_common else False for i in range(y_val.shape[0])])

        X_train, y_train = torch.tensor(X_train[train_mask]), torch.tensor([label_map[int(x)] for x in y_train[train_mask]])
        X_val, y_val = torch.tensor(X_val[val_mask]), torch.tensor([label_map[int(x)] for x in y_val[val_mask]])

        assert (y_train < 10000).all(), y_train.max()
        
        # check for the 1% or 10% split

        counts = ['_1010', '_3030', '_5050', '_10100', '_30300']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        assert split in splits

        if split.endswith('_1010'):
            idx = equal_class_counts_inds(y_train, 1010)
        elif split.endswith('_3030'):
            idx = equal_class_counts_inds(y_train, 3030)
        elif split.endswith('_5050'):
            idx = equal_class_counts_inds(y_train, 5050)
        elif split.endswith('_10100'):
            idx = equal_class_counts_inds(y_train, 10100)
        elif split.endswith('_30300'):
            idx = equal_class_counts_inds(y_train, 30300)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx)), (len(unlabeled_idx), X_train.shape[0], len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(X_val, y_val)
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(X_train[idx], y_train[idx])
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(X_train[unlbl_idx], y_train[unlbl_idx])
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3Food101Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files

        with open(os.path.join(download_path, 'Food101', 'DINOv3_Food101_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, 'Food101', 'DINOv3_Food101_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_101', '_303', '_505', '_1010']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_101'):
            idx = equal_class_counts_inds(y_train, 101)
        elif split.endswith('_303'):
            idx = equal_class_counts_inds(y_train, 303)
        elif split.endswith('_505'):
            idx = equal_class_counts_inds(y_train, 505)
        elif split.endswith('_1010'):
            idx = equal_class_counts_inds(y_train, 1010)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3CIFAR10Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'CIFAR10'

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()


        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLiPv2CIFAR10Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'CIFAR10'

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()
        
        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3CIFAR100Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'CIFAR100'

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_100', '_300', '_500', '_1000']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits, (split, splits)

        if split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        elif split.endswith('_300'):
            idx = equal_class_counts_inds(y_train, 300)
        elif split.endswith('_500'):
            idx = equal_class_counts_inds(y_train, 500)
        elif split.endswith('_1000'):
            idx = equal_class_counts_inds(y_train, 1000)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLiPv2CIFAR100Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'CIFAR100'

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)
        
        # check for the 1% or 10% split
        counts = ['_100', '_300', '_500', '_1000']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits, (split, splits)

        if split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        elif split.endswith('_300'):
            idx = equal_class_counts_inds(y_train, 300)
        elif split.endswith('_500'):
            idx = equal_class_counts_inds(y_train, 500)
        elif split.endswith('_1000'):
            idx = equal_class_counts_inds(y_train, 1000)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3SVHNDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'SVHN'

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLiPv2SVHNDataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'SVHN'

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        
        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()

        labeled_idx = set(idx)
        unlabeled_idx = set(list(range(X_train.shape[0]))) - labeled_idx

        assert len(unlabeled_idx) == (X_train.shape[0] - len(labeled_idx))

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            unlbl_idx = sorted(list(unlabeled_idx))
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[unlbl_idx]), torch.tensor(y_train[unlbl_idx]))
        
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


class DINOv3STL10Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'STL10'

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'DINOv3_{dataset_string}_unlabeled.ds'), 'rb') as fp:
            X_unlabeled, y_unlabeled = pickle.load(fp)
        
        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_unlabeled), torch.tensor(y_unlabeled))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)
    

class SigLiPv2STL10Dataset:
    def __init__(self, download_path, transform=None, target_transform=None, download=False, split='train'):
        self.split = split
        self.transform = transform if transform is not None else torch.nn.Identity()
        self.target_transform = target_transform if target_transform is not None else torch.nn.Identity()

        # load the files
        dataset_string = 'STL10'

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_train.ds'), 'rb') as fp:
            X_train, y_train = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_val.ds'), 'rb') as fp:
            X_val, y_val = pickle.load(fp)

        with open(os.path.join(download_path, dataset_string, f'SigLiPv2_{dataset_string}_unlabeled.ds'), 'rb') as fp:
            X_unlabeled, y_unlabeled = pickle.load(fp)
        
        # check for the 1% or 10% split

        counts = ['_10', '_30', '_50', '_100']

        splits = ['train' + c for c in counts] + ['val' + c for c in counts] + ['unlabeled' + c for c in counts]

        assert split in splits

        if split.endswith('_10'):
            idx = equal_class_counts_inds(y_train, 10)
        elif split.endswith('_30'):
            idx = equal_class_counts_inds(y_train, 30)
        elif split.endswith('_50'):
            idx = equal_class_counts_inds(y_train, 50)
        elif split.endswith('_100'):
            idx = equal_class_counts_inds(y_train, 100)
        else:
            raise NotImplementedError()

        if split.startswith('val'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        elif split.startswith('train'):
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_train[idx]), torch.tensor(y_train[idx]))
        else:
            self.ds = torch.utils.data.TensorDataset(torch.tensor(X_unlabeled), torch.tensor(y_unlabeled))
            
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.transform(x), self.transform(y)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ds)


def select_dataset(args, data_dir, spec_dir, preprocessing_transform):
    """
    Use the dataset directory and the data augmentations to build the dataset lists to pass into the
    dataset objects.
    
    lock_path = os.path.join(spec_dir[args.dataset]["download_path"], 'download.lock')
    if not os.path.isfile(lock_path):
        fp = open(lock_path, 'a')
        fp.close()
    else:
        while os.path.isfile(lock_path):
            time.sleep(1)
    """

    ds_train = data_dir[args.dataset](
        spec_dir[args.dataset]["download_path"],
        transform=preprocessing_transform,
        download=True,
        **spec_dir[args.dataset]["train"],
    )

    if spec_dir[args.dataset]["unlabeled"] is not None:
        # if the unlabeled dataset is defined in the dataset directory then we just load that
        ds_unlabeled = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=preprocessing_transform,
            download=True,
            **spec_dir[args.dataset]["unlabeled"],
        )

    else:
        # otherwise then we have to split the train dataset into train and unlabeled
        ds = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=preprocessing_transform,
            download=True,
            **spec_dir[args.dataset]["train"],
        )
        # Perform stratified split
        labeled_indices = []
        unlabeled_indices = []

        indices = np.random.permutation(len(ds))

        if args.balanced:
            # TODO: continuous targets are not suitable for this type of operations (this may be out of scope for some time)
            num_classes = (
                vars(args).get("num_classes")
                if vars(args).get("num_classes") is not None
                else spec_dir[args.dataset]["classes"]
            )
            setattr(args, "num_classes", num_classes)
            class_counters = [[] for i in range(num_classes)]

            for i in indices:
                dp = ds[i]
                y = dp[1]
                class_counters[y].append(i)

            while len(labeled_indices) < int(len(ds) * (1 - args.unlabeled_fraction)):
                for j in range(num_classes):
                    labeled_indices.append(class_counters[j].pop(0))

            unlabeled_indices = sum(class_counters, [])
        else:
            assert vars(args).get("num_classes") is None, (
                "num classes not supported if args.balanced is not True"
            )

            setattr(args, "num_classes", spec_dir[args.dataset]["classes"])
            unlabeled_indices = indices[: int(len(ds) * args.unlabeled_fraction)]
            labeled_indices = indices[int(len(ds) * args.unlabeled_fraction) :]

        ds_train = IndexSubsetDataset(ds, list(labeled_indices))
        ds_unlabeled = IndexSubsetDataset(ds, list(unlabeled_indices))

    if spec_dir[args.dataset]["train"] is not None and spec_dir[args.dataset]["test"]:
        # if there is a test dataset which is defined in the dataset directory then we load that
        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=preprocessing_transform,
            download=False,
            **spec_dir[args.dataset]["test"],
        )
    else:
        # otherwise the train dataset is going to be split into train and test, having already held out the labeled data
        ds_train = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=preprocessing_transform,
            download=False,
        )

        ds_train = IndexSubsetDataset(
            ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
        )

        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=preprocessing_transform,
            download=False,
        )
        ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])

    """try:
        os.remove(lock_path)
    except OSError as e:
        pass"""

    return ds_unlabeled, ds_train, ds_val


DATA_DIR = {  # (TODO) add the MNIST, SVHN, and ImageNet
    "pascal_voc_10": PascalVOC,
    "caltech101": torchvision.datasets.Caltech101,
    "caltech256": torchvision.datasets.Caltech256,
    "food101": torchvision.datasets.Food101,
    "inat2021": torchvision.datasets.INaturalist,
    "cifar10": torchvision.datasets.CIFAR10,
    "mnist": torchvision.datasets.MNIST,
    "cifar100": torchvision.datasets.CIFAR100,
    "in1k_dinov2_1": DINOv2ImageNetDataset,
    "in1k_clip_1": CLIPImageNetDataset,
    "in1k_dinov3_1": DINOv3ImageNetDataset,
    "in1k_siglipv2_1": SigLiPv2ImageNetDataset,
    "in1k_dinov2_10": DINOv2ImageNetDataset,
    "in1k_clip_10": CLIPImageNetDataset,
    "in1k_dinov3_10": DINOv3ImageNetDataset,
    "in1k_siglipv2_10": SigLiPv2ImageNetDataset,
    "inat_dinov3_1010": DINOv3iNatDataset,
    "inat_siglipv2_1010": SigLiPv2iNatDataset,
    "inat_dinov3_3030": DINOv3iNatDataset,
    "inat_siglipv2_3030": SigLiPv2iNatDataset,
    "inat_dinov3_5050": DINOv3iNatDataset,
    "inat_siglipv2_5050": SigLiPv2iNatDataset,
    "inat_dinov3_10100": DINOv3iNatDataset,
    "inat_siglipv2_10100": SigLiPv2iNatDataset,
    "food101_dinov3_101": DINOv3Food101Dataset,
    "food101_siglipv2_101": SigLiPv2Food101Dataset,
    "food101_dinov3_303": DINOv3Food101Dataset,
    "food101_siglipv2_303": SigLiPv2Food101Dataset,
    "food101_dinov3_505": DINOv3Food101Dataset,
    "food101_siglipv2_505": SigLiPv2Food101Dataset,
    "food101_dinov3_1010": DINOv3Food101Dataset,
    "food101_siglipv2_1010": SigLiPv2Food101Dataset,
    "cifar10_dinov3_10": DINOv3CIFAR10Dataset,
    "cifar10_siglipv2_10": SigLiPv2CIFAR10Dataset,
    "cifar10_dinov3_30": DINOv3CIFAR10Dataset,
    "cifar10_siglipv2_30": SigLiPv2CIFAR10Dataset,
    "cifar10_dinov3_50": DINOv3CIFAR10Dataset,
    "cifar10_siglipv2_50": SigLiPv2CIFAR10Dataset,
    "cifar10_dinov3_100": DINOv3CIFAR10Dataset,
    "cifar10_siglipv2_100": SigLiPv2CIFAR10Dataset,
    "cifar100_dinov3_100": DINOv3CIFAR100Dataset,
    "cifar100_siglipv2_100": SigLiPv2CIFAR100Dataset,
    "cifar100_dinov3_300": DINOv3CIFAR100Dataset,
    "cifar100_siglipv2_300": SigLiPv2CIFAR100Dataset,
    "cifar100_dinov3_500": DINOv3CIFAR100Dataset,
    "cifar100_siglipv2_500": SigLiPv2CIFAR100Dataset,
    "cifar100_dinov3_1000": DINOv3CIFAR100Dataset,
    "cifar100_siglipv2_1000": SigLiPv2CIFAR100Dataset,
    "stl10_dinov3_50": DINOv3STL10Dataset,
    "stl10_siglipv2_50": SigLiPv2STL10Dataset,
    "stl10_dinov3_10": DINOv3STL10Dataset,
    "stl10_siglipv2_10": SigLiPv2STL10Dataset,
    "stl10_dinov3_30": DINOv3STL10Dataset,
    "stl10_siglipv2_30": SigLiPv2STL10Dataset,
    "stl10_dinov3_100": DINOv3STL10Dataset,
    "stl10_siglipv2_100": SigLiPv2STL10Dataset,
    "svhn_dinov3_10": DINOv3SVHNDataset,
    "svhn_siglipv2_10": SigLiPv2SVHNDataset,
    "svhn_dinov3_30": DINOv3SVHNDataset,
    "svhn_siglipv2_30": SigLiPv2SVHNDataset,
    "svhn_dinov3_50": DINOv3SVHNDataset,
    "svhn_siglipv2_50": SigLiPv2SVHNDataset,
    "svhn_dinov3_100": DINOv3SVHNDataset,
    "svhn_siglipv2_100": SigLiPv2SVHNDataset,
    "dinov3_tree": DINOv3TREEDataset,
    "siglipv2_tree": SigLIP2TREEDataset
}