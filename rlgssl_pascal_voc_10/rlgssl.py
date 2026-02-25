from dahps import DistributedAsynchronousRandomSearch as DARS
from dahps.torch_utils import sync_parameters
from rlgssl_config import config

import torch
from torch import nn
import os
import argparse
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F
import math
import random
from typing import Union
from functools import partial

from stablessl.train_utils import (
    model_run,
    setup,
    cleanup,
)

from stablessl.data.dataset import BaseDataset
from stablessl.algorithms.rlgssl import RLGSSLAlgorithm
from torchvision.models.segmentation import deeplabv3_resnet50
from pathlib import Path

from torch.optim.lr_scheduler import LambdaLR
import wandb


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_wait_steps=0,
    num_cycles=0.5,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(
                max(1, num_warmup_steps + num_wait_steps)
            )

        progress = float(current_step - num_warmup_steps - num_wait_steps) / float(
            max(1, num_training_steps - num_warmup_steps - num_wait_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


torch.manual_seed(42)


def wandb_init(args):
    wandb.init(
        "ai2es",
        args.project,
        config=vars(args),
        name=f"{args.experiment_type} {args.dataset}",
    )

def construct_deeplabv3_resnet50(
        weights: Path | None,
        num_classes
    ):

    # if weights:
    #     assert weights.exists()

    model = deeplabv3_resnet50(
        weights=weights,
        num_classes=num_classes
        # i think everything else should be okay as defaults, we're directly
        # comparing against a pascal-VOC baseline.
    )

    return model

class DeepLabV3_Resnet50(nn.Module):
    def __init__(self, n_classes=21, weights: Path | None = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, **kwargs):
        super().__init__()
        module = construct_deeplabv3_resnet50(weights, n_classes)

        self.deeplab = module
        
    def forward(self, x, **kwargs):

        return self.deeplab(x)['out']


class Cutout(nn.Module):
    """
    Cutout data augmentation: randomly masks out one or more square regions
    of an image during training.

    Based on the paper:
        "Improved Regularization of Convolutional Neural Networks with Cutout"
        DeVries & Taylor, 2017 — https://arxiv.org/abs/1708.04552

    Operates on a torch.Tensor image of shape (C, H, W) with values in [0, 1]
    or [0, 255], or a PIL Image (converted internally).

    Args:
        n_holes     : Number of square patches to cut out.
        length      : Side length of each square patch (pixels).
                      Can also be a (min, max) tuple for random sizing.
        fill_value  : Value to fill the cutout region with.
                        - float/int: fills all channels with that value.
                        - 'random': fills with uniform random noise.
                        - 'mean': fills with the per-channel mean of the image.
        p           : Probability of applying the transform. Default 1.0.
    """

    def __init__(
        self,
        n_holes: int = 1,
        length: Union[int, tuple[int, int]] = 16,
        fill_value: Union[float, str] = 0.0,
        p: float = 1.0,
    ):
        super().__init__()

        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        if isinstance(length, int):
            if length <= 0:
                raise ValueError(f"length must be positive, got {length}")
            self.length_min = self.length_max = length
        else:
            self.length_min, self.length_max = length
            if self.length_min <= 0 or self.length_min > self.length_max:
                raise ValueError(f"length tuple must satisfy 0 < min <= max, got {length}")
        if isinstance(fill_value, str) and fill_value not in ("random", "mean"):
            raise ValueError(f"fill_value string must be 'random' or 'mean', got '{fill_value}'")

        self.n_holes   = n_holes
        self.fill_value = fill_value
        self.p          = p

    # ------------------------------------------------------------------
    def _sample_length(self) -> int:
        if self.length_min == self.length_max:
            return self.length_min
        return random.randint(self.length_min, self.length_max)

    def _make_fill(self, img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Return a (C, h, w) fill tensor matching img's dtype and device."""
        c = img.shape[0]
        if self.fill_value == "random":
            return torch.empty(c, h, w, dtype=img.dtype, device=img.device).uniform_(
                img.min().item(), img.max().item()
            )
        elif self.fill_value == "mean":
            mean = img.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
            return mean.expand(c, h, w)
        else:
            return torch.full(
                (c, h, w), self.fill_value, dtype=img.dtype, device=img.device
            )

    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img : Tensor of shape (C, H, W).

        Returns:
            Augmented tensor of the same shape.
        """
        if random.random() > self.p:
            return img

        # Accept PIL Images for compatibility with torchvision pipelines
        was_pil = not isinstance(img, torch.Tensor)
        if was_pil:
            img = F.to_tensor(img)

        img = img.clone()                       # don't mutate the original
        _, H, W = img.shape

        for _ in range(self.n_holes):
            length = self._sample_length()

            # Centre of the patch (uniformly random over the full image)
            cx = random.randint(0, W - 1)
            cy = random.randint(0, H - 1)

            # Clamp patch boundaries to image edges
            x1 = max(cx - length // 2, 0)
            y1 = max(cy - length // 2, 0)
            x2 = min(cx + length // 2, W)
            y2 = min(cy + length // 2, H)

            ph, pw = y2 - y1, x2 - x1          # actual (possibly clipped) patch size
            img[:, y1:y2, x1:x2] = self._make_fill(img, ph, pw)

        if was_pil:
            img = F.to_pil_image(img)

        return img

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        length_str = (
            str(self.length_min)
            if self.length_min == self.length_max
            else f"({self.length_min}, {self.length_max})"
        )
        return (
            f"{self.__class__.__name__}("
            f"n_holes={self.n_holes}, "
            f"length={length_str}, "
            f"fill_value={self.fill_value!r}, "
            f"p={self.p})"
        )


def training_process(agent, args, rank, world_size):
    wandb_init(args)

    setattr(args, 'autocast_type', torch.float16)

    # constructors for the optimization objects

    def model_fn():
        return DeepLabV3_Resnet50()
    

    def optimizer_fn(named_parameters):
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [
                        param
                        for name, param in named_parameters
                        if "backbone" in name
                    ],
                    "lr": args.learning_rate,
                },
                {
                    "params": [
                        param
                        for name, param in named_parameters
                        if "backbone" not in name
                    ],
                    "lr": args.learning_rate * 10,
                },
            ],
            lr=args.learning_rate,
            # momentum=0.9,
            # weight_decay=1e-4,
        )

        return optimizer


    scheduler_fn = partial(get_cosine_schedule_with_warmup, num_warmup_steps=0, num_training_steps=args.epochs * args.steps_per_epoch, num_wait_steps=0)

    rez = 520

    weak_transform = transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (rez, rez), antialias=True, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.ColorJitter(0.25, 0.25, 0.25, 0.125),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(5, (0.01, 1)),
                Cutout(1, int(rez/8)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    strong_transform = transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (rez, rez), antialias=True, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.25),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(5, (0.01, 2)),
                Cutout(1, int(rez/8)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    
    def identity(x):
        return x

    dataset = BaseDataset(
        args,
        torch.nn.Identity(),
        weak_augmentation_transform=weak_transform,
        strong_augmentation_transform=strong_transform
    )

    algorithm = RLGSSLAlgorithm(agent, args, model_fn, optimizer_fn, scheduler_fn, dataset, loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255))

    print("training model...")

    model, acc = model_run(args, algorithm, dataset)

    return model, acc


def create_parser():
    parser = argparse.ArgumentParser(description="StableSSL RLGSSL")

    parser.add_argument(
        "--epochs", type=int, default=300, help="training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./stableSSL_rlgssl",
        help="path for the hyperparameter search data",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="meta_debug",
        help="wandb project identifier",
    )

    return parser


def main(args, rank, world_size):
    setup(rank, world_size)

    device = rank % torch.cuda.device_count()
    print(f"rank {rank} running on device {device} (of {torch.cuda.device_count()})")
    torch.cuda.set_device(device)

    agent = DARS.from_config(args.path, config)

    agent = sync_parameters(rank, agent)

    args = agent.update_namespace(args)

    print(args)

    states, metric = training_process(agent, args, rank, world_size)

    if rank == 0:
        agent.finish_combination(metric)

    print("cleanup")
    cleanup()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method("spawn")

    main(args, rank, world_size)
