from stablessl.callbacks.base import BaseCallback
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from stablessl.data.dataset import IndexedIndexSubsetDataset, IndexedDataset
from stablessl.data.utils import balance_samples
from stablessl.train_utils.metrics import Accuracy
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm
from torch import distributed as dist
import os
import torch
from enum import Enum
from tqdm import tqdm 
import wandb
import gc

class SelfTrainingStrategy(Enum):
    PERCENT = 0 # a percentage of the pseudo labels brought in
    TOP_K = 1 # a constant number of examples will be brought in
    THRESHOLD = 2 # a confidence threshold will be used to select pseudo-labels


class SelfTrainingPLCallback(BaseCallback):
    def __init__(self, algorithm, autocast_type, with_replacement, strategy: SelfTrainingStrategy, k=None, threshold=None, percent=None):
        self.algorithm = algorithm
        self.model = None
        self.strategy = strategy
        self.autocast_type = autocast_type
        self.with_replacement = with_replacement

        if self.strategy == SelfTrainingStrategy.PERCENT.value:
            assert percent is not None, 'must set percent argument if strategy is PERCENT'
            assert 0 < percent <= 1, f'must set percent to a value in (0, 1], found {percent}'
            self.percent = percent
        elif self.strategy == SelfTrainingStrategy.TOP_K.value:
            assert k is not None, 'must set k if strategy is TOP_K'
            assert k > 0, f'k must be greater than 0, found {k}'
            self.k = k
        elif self.strategy == SelfTrainingStrategy.THRESHOLD.value:
            assert threshold is not None, 'must set threshold if strategy is THRESHOLD'
            assert 0 <= threshold < 1, f'must set threshold to a value in [0, 1), found {threshold}' 
            self.threshold = threshold
        else:
            raise NotImplementedError(f"unrecognized strategy type: {self.strategy}")


    def __call__(self):
        # modify the dataset object in the following way:
        # 1. provide pseudo-labels for all of the elements of the current shard
        # 2. sort the pseudo-labels by confidence
        # 3. pick the top confident labels (by either fraction, confidence level, or constant number)
        # 4. add these indices to the labeled dataset
        # 5. if we are doing this with replacement then remove the pseudo-labels previously added.

        if self.with_replacement:
            self.algorithm.dataset.clear_pl()

        self.algorithm.dataset.clear_loaders()

        # create a loader for this so we do not observe the shuffling behaviour
        rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        sampler = DistributedSampler(
                self.algorithm.dataset.unlabeled,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=777,
            )
        U_PL_loader = DataLoader(
                IndexedIndexSubsetDataset(self.algorithm.dataset.unlabeled),
                batch_size=self.algorithm.dataset.args.unlabeled_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.algorithm.dataset.args.unlabeled_workers,
            )
        
        pl_batches = []
        index_batches = []

        self.algorithm.load_checkpoint_state()
        self.model = self.algorithm.model
        freeze_batchnorm(self.model)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_type):
                for i, U, _, _ in tqdm(U_PL_loader):
                    pl_batches.append(torch.nn.functional.softmax(self.model(U.to(torch.cuda.current_device())), -1))
                    index_batches.append(i.to(torch.cuda.current_device()))

        # concatenate the pseudo-labels and the indicies into a single tensor
        index_tensor = torch.cat(index_batches, 0) - len(self.algorithm.dataset.ds_train)
        pl_tensor = torch.cat(pl_batches, 0)

        gather_pl = [pl_tensor.clone() for i in range(world_size)]
        gather_index = [index_tensor.clone() for i in range(world_size)]

        dist.all_gather(gather_index, index_tensor)
        dist.all_gather(gather_pl, pl_tensor)

        index_tensor = torch.cat(gather_index, 0).cpu()
        pl_tensor = torch.cat(gather_pl, 0).cpu()

        confidences, labels = torch.max(pl_tensor, -1)

        # balance the incoming labels batch
        if vars(self.algorithm.args).get('balance_factor') is not None:
            balanced_inds = balance_samples(labels, sorted=True, factor=self.algorithm.args.balance_factor)

            confidences = confidences[balanced_inds]
            index_tensor = index_tensor[balanced_inds]
            labels = labels[balanced_inds]

        # sort the pseudo-labels by confidence

        confidences_descending = torch.argsort(confidences, 0, descending=True)

        if self.strategy == SelfTrainingStrategy.TOP_K.value:
            pl_inds = confidences_descending[:self.k]
            inds = index_tensor[pl_inds]
            labels = labels[pl_inds]
        elif self.strategy == SelfTrainingStrategy.PERCENT.value:
            pl_inds = confidences_descending[:int(len(index_tensor) * self.percent)]
            inds = index_tensor[pl_inds]
            labels = labels[pl_inds]
        elif self.strategy == SelfTrainingStrategy.THRESHOLD.value:
            inds = index_tensor[confidences > self.threshold]
            labels = labels[confidences > self.threshold]
        else:
            raise NotImplementedError(f"unrecognized strategy: {self.strategy}")
        
        # add those indices to the labeled set with the appropriate pseudo_label
        self.algorithm.dataset.add_pl_to_labeled(labels, inds)

        unfreeze_batchnorm(self.model)

        
class MaybeResetModelCallback(BaseCallback):
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.args = algorithm.args

    def __call__(self):
        if vars(self.args).get('from_scratch'):
            self.algorithm.reset_model()


class CoTrainingPLCallback(BaseCallback):
    def __init__(self, algorithm, autocast_type, with_replacement, strategy: SelfTrainingStrategy, k=None, threshold=None, percent=None):
        self.algorithm = algorithm
        self.model = None
        self.strategy = strategy
        self.autocast_type = autocast_type
        self.with_replacement = with_replacement

        if self.strategy == SelfTrainingStrategy.PERCENT.value:
            assert percent is not None, 'must set percent argument if strategy is PERCENT'
            assert 0 < percent <= 1, f'must set percent to a value in (0, 1], found {percent}'
            self.percent = percent
        elif self.strategy == SelfTrainingStrategy.TOP_K.value:
            assert k is not None, 'must set k if strategy is TOP_K'
            assert k > 0, f'k must be greater than 0, found {k}'
            self.k = k
        elif self.strategy == SelfTrainingStrategy.THRESHOLD.value:
            assert threshold is not None, 'must set threshold if strategy is THRESHOLD'
            assert 0 <= threshold < 1, f'must set threshold to a value in [0, 1), found {threshold}' 
            self.threshold = threshold
        else:
            raise NotImplementedError(f"unrecognized strategy type: {self.strategy}")
        
    def __call__(self):

        if self.with_replacement:
            self.algorithm.dataset.clear_pl()
            self.algorithm.other_dataset.clear_pl()

        self.algorithm.dataset.clear_loaders()
        self.algorithm.other_dataset.clear_loaders()

        # create a loader for this so we do not observe the shuffling behaviour
        rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        sampler = DistributedSampler(
                self.algorithm.dataset.unlabeled,
                num_replicas=int(world_size // 2),
                rank=int(rank // 2),
                shuffle=False,
                seed=777,
            )
        U_PL_loader = DataLoader(
                IndexedIndexSubsetDataset(self.algorithm.dataset.unlabeled),
                batch_size=self.algorithm.dataset.args.unlabeled_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.algorithm.dataset.args.unlabeled_workers,
            )
        
        pl_batches = []
        index_batches = []

        self.algorithm.load_checkpoint_state()
        self.model = self.algorithm.model
        freeze_batchnorm(self.model)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_type):
                for i, U, _, _ in tqdm(U_PL_loader):
                    pl_batches.append(torch.nn.functional.softmax(self.model(U.to(torch.cuda.current_device())), -1).type(self.autocast_type))
                    index_batches.append(i.to(torch.cuda.current_device()))

        # concatenate the pseudo-labels and the indicies into a single tensor
        index_tensor = torch.cat(index_batches, 0) - len(self.algorithm.dataset.ds_train)
        pl_tensor = torch.cat(pl_batches, 0)

        del pl_batches
        gc.collect()

        index_tensor = index_tensor.cpu()
        pl_tensor = pl_tensor.cpu()

        print(world_size, pl_tensor.dtype)

        gather_pl = [pl_tensor.clone().to(torch.cuda.current_device()) for i in range(world_size)]

        dist.barrier()

        dist.all_gather(gather_pl, pl_tensor.to(torch.cuda.current_device()))

        gather_pl = [pl_tensor.cpu() for pl_tensor in gather_pl]

        gather_index = [index_tensor.clone().to(torch.cuda.current_device()) for i in range(world_size)]

        dist.barrier()

        dist.all_gather(gather_index, index_tensor.to(torch.cuda.current_device()))
         
        gather_index = [index_tensor.cpu() for index_tensor in gather_index]

        # need to make sure that these guys are coming from the correct ranks and storing all of the indices
        A = set([a.item() for a in list(torch.cat(gather_index[::2], 0).cpu())])
        B = set([b.item() for b in list(torch.cat(gather_index[1::2], 0).cpu())])
        assert A == B, (sorted(A), sorted(B))

        current_view = int(os.environ["RANK"]) % 2
        other_view = 1 - (int(os.environ["RANK"]) % 2)

        current_index_tensor = torch.cat(gather_index[current_view::2], 0).cpu()
        current_pl_tensor = torch.cat(gather_pl[current_view::2], 0).cpu()

        other_index_tensor = torch.cat(gather_index[other_view::2], 0).cpu()
        other_pl_tensor = torch.cat(gather_pl[other_view::2], 0).cpu()

        current_confidences, current_labels = torch.max(current_pl_tensor, -1)
        other_confidences, other_labels = torch.max(other_pl_tensor, -1)

        # ensure that the conflicting labels are droppped (after this there are still the same number of pseudo-labels)

        current_index_tensor, current_index_order = torch.sort(current_index_tensor, 0, descending=False)
        other_index_tensor, other_index_order = torch.sort(other_index_tensor, 0, descending=False)

        current_confidences, current_labels = current_confidences[current_index_order], current_labels[current_index_order]
        other_confidences, other_labels = other_confidences[other_index_order], other_labels[other_index_order]

        agreement_mask = current_labels == other_labels

        current_confidences, other_confidences = current_confidences[agreement_mask], other_confidences[agreement_mask]
        current_index_tensor, other_index_tensor = current_index_tensor[agreement_mask], other_index_tensor[agreement_mask]
        current_labels, other_labels = current_labels[agreement_mask], other_labels[agreement_mask]

        # balance the incoming labels batch
        if vars(self.algorithm.args).get('balance_factor') is not None:
            balanced_inds = balance_samples(current_labels, sorted=True, factor=self.algorithm.args.balance_factor)

            current_confidences, other_confidences = current_confidences[balanced_inds], other_confidences[balanced_inds]
            index_tensor = index_tensor[balanced_inds]
            current_labels, other_labels = current_labels[balanced_inds], other_labels[balanced_inds]

        # sort the pseudo-labels by confidence

        current_confidences_descending = torch.argsort(current_confidences, 0, descending=True)
        other_confidences_descending = torch.argsort(other_confidences, 0, descending=True)

        if self.strategy == SelfTrainingStrategy.TOP_K.value:
            current_pl_inds = current_confidences_descending[:self.k]
            other_pl_inds = other_confidences_descending[:self.k]

            current_inds = index_tensor[current_pl_inds]
            current_labels = current_labels[current_pl_inds]

            other_inds = index_tensor[other_pl_inds]
            other_labels = other_labels[other_pl_inds]
        elif self.strategy == SelfTrainingStrategy.PERCENT.value:
            current_pl_inds = current_confidences_descending[:int(len(current_index_tensor) * self.percent)]
            other_pl_inds = other_confidences_descending[:int(len(other_index_tensor) * self.percent)]

            current_inds = index_tensor[current_pl_inds]
            current_labels = current_labels[current_pl_inds]

            other_inds = index_tensor[other_pl_inds]
            other_labels = other_labels[other_pl_inds]
        elif self.strategy == SelfTrainingStrategy.THRESHOLD.value:
            current_inds = index_tensor[current_confidences > self.threshold]
            current_labels = current_labels[current_confidences > self.threshold]

            other_inds = index_tensor[other_confidences > self.threshold]
            other_labels = other_labels[other_confidences > self.threshold]
        else:
            raise NotImplementedError(f"unrecognized strategy: {self.strategy}")
        
        # add those indices to the labeled set with the appropriate pseudo_label
        self.algorithm.other_dataset.add_pl_to_labeled(current_labels, current_inds)
        self.algorithm.dataset.add_pl_to_labeled(other_labels, other_inds)

        unfreeze_batchnorm(self.model)

class CoTrainingEvalCallback(BaseCallback):
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

        self.algorithm.load_checkpoint_state()
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
            'iteration co_accuracy': acc.metric
        })

        unfreeze_batchnorm(self.model)