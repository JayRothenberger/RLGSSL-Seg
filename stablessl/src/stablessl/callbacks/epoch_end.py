from stablessl.callbacks.base import BaseCallback
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from stablessl.data.dataset import IndexedIndexSubsetDataset, IndexedDataset
from stablessl.data.utils import balance_samples
from stablessl.train_utils.metrics import Accuracy
from torch import distributed as dist
import os
import torch
from enum import Enum
from tqdm import tqdm 
import wandb
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm


class MetaCoTrainingEvalCallback(BaseCallback):
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def __call__(self):
        # create a loader for this so we do not observe the shuffling behaviour
        rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        sampler = DistributedSampler(
                self.algorithm.dataset.validation,
                num_replicas=1,
                rank=0,
                shuffle=False,
                seed=777,
            )
        val_loader = DataLoader(
                IndexedDataset(self.algorithm.dataset.validation),
                batch_size=self.algorithm.dataset.args.validation_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.algorithm.dataset.args.validation_workers,
            )
        
        pl_batches = []
        label_batches = []

        self.model = self.algorithm.model
        freeze_batchnorm(self.model)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.algorithm.args.autocast_type):
                for i, X, y in tqdm(val_loader):
                    pl_batches.append(self.model(X.to(torch.cuda.current_device())))
                    label_batches.append(y.to(torch.cuda.current_device()))

        pl_tensor = torch.cat(pl_batches, 0)

        # now we need to make sure that all of the indices match

        label_tensor = torch.cat(label_batches, 0) 
        label_tensors = [label_tensor.clone() for i in range(world_size)]

        torch.distributed.all_gather(label_tensors, label_tensor)

        for b in label_tensors:
            assert torch.equal(b, label_tensors[0]), 'label tensors were not all equal on all ranks'

        torch.distributed.all_reduce(pl_tensor)
        
        acc = Accuracy(pl_tensor, label_tensor)

        wandb.log({
            'epoch co_accuracy': acc.metric
        })

        unfreeze_batchnorm(self.model)
