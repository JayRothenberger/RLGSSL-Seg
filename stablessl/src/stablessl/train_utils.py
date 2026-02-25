import torch
import os
import gc
import wandb
from tqdm import tqdm
import torch.distributed as dist
import time

import glob
from pathlib import Path
import numpy as np


class IndexSubsetDataset:
    def __init__(self, ds, inds):
        self.ds = ds
        self.inds = inds

    def __getitem__(self, i):
        return self.ds[self.inds[i]]
    
    def __iter__(self):
        for i in self.inds:
            yield self.ds[i]

    def __len__(self):
        return len(self.inds)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_learning_rate(optimizer, scheduler, lr, it, warmup_steps, wait_steps=0.0):
    warmup_steps = warmup_steps + wait_steps
    if it > wait_steps > 0 and wait_steps > it:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0
    elif it > warmup_steps > 0 and warmup_steps > it:
        # do warm up lr
        lr = (float(it) / float(warmup_steps)) * lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test_print(model, val_loader):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda().long()
        input = input.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def global_contrast_normalize(X, scale=55.0, min_divisor=1e-8):
    # https://github.com/Jongchan/unsupervised_data_augmentation_pytorch/blob/master/cifar/main.py
    X = X.view(X.size(0), -1)
    X = X - X.mean(dim=1, keepdim=True)

    normalizers = torch.sqrt(torch.pow(X, 2).sum(dim=1, keepdim=True)) / scale
    normalizers[normalizers < min_divisor] = 1.0
    X /= normalizers

    return X.view(X.size(0), 3, 32, 32)


class UDADataset:
    def __init__(self, ds, transform, target_transform):
        self.ds = ds
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        x, _ = self.ds[i]
        return self.transform(x), self.target_transform(x)

    def __iter__(self):
        for i in self:
            yield i

    def __len__(self):
        return len(self.ds)


def repeat3(x):
    return x.unsqueeze(0).repeat(3, 1, 1)[:3]


def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def model_run(
    model,
    scheduler_actor,
    scheduler_critic,
    epochs,
    train_loader,
    unlabeled_loader,
    val_loader,
    patience,
    sampler_train=None,
    sampler_unlabeled=None,
    stop_early=False
):
    best_val_acc = 0
    best_state = None
    epochs_since_improvement = 0

    if int(os.environ.get("WORLD_SIZE")) > 1:
        print(
            "because the world size is greater than 1 I assume you want to aggregate performance metrics across ranks"
        )
        print("please make sure each rank is running the training process")

    test_fn = test_ddp if int(os.environ.get("WORLD_SIZE")) > 1 else test
    train_fn = train_ddp if int(os.environ.get("WORLD_SIZE")) > 1 else train

    if stop_early:
        print("early stopping is enabled, be aware that the stopping criterion is a local property and if training multiple models some may stop before others.")

    print("training model...")
    for epoch in range(epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        if sampler_unlabeled is not None:
            sampler_unlabeled.set_epoch(epoch)
        print("epoch", epoch)
        train_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "train_loader": train_loader,
            "unlabeled_loader": unlabeled_loader,
            "scheduler_actor": scheduler_actor,
            "scheduler_critic": scheduler_critic,
        }
        avg_reward, avg_loss = train_fn(**train_args)
        gc.collect()
        test_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "loader": val_loader,
            "loss_fn": model.loss,
        }
        val_loss, val_acc = test_fn(**test_args)

        test_args = {
            "rank": int(os.environ.get("RANK"))
            if os.environ.get("RANK") is not None
            else None,
            "model": model.to(torch.cuda.current_device()),
            "loader": train_loader,
            "loss_fn": model.loss,
        }

        wandb.log(
            {
                "val loss": val_loss,
                "val acc": val_acc,
                "avg reward": avg_reward,
                "avg loss": avg_loss,
            }
        )

        # early stopping

        if val_acc > best_val_acc:
            model.agent.save_checkpoint(model.state_dict())
            epochs_since_improvement = 0
            best_val_acc = val_acc

        elif stop_early:
            if epochs_since_improvement >= patience:
                return best_state, best_val_acc
            epochs_since_improvement += 1

    return best_state, best_val_acc


## single-gpu local versions of model training functions
def train(
    model, unlabeled_loader, train_loader, scheduler_actor, scheduler_critic, **kwargs
):
    model.train()

    total = 0
    avg_h = 0
    avg_l = 0

    for (I, U, U_aug), (X, y) in tqdm(zip(unlabeled_loader, train_loader)):
        X, y = (
            X.to(torch.cuda.current_device()),
            y.to(torch.cuda.current_device()),
        )
        I, U, U_aug = (
            I.to(torch.cuda.current_device()),
            U.to(torch.cuda.current_device()),
            U_aug.to(torch.cuda.current_device()),
        )

        l, h = model.step(I, U, U_aug, X, y)

        avg_h += h.item()
        avg_l += l.item()
        total += 1

        scheduler_actor.step()
        scheduler_critic.step()

    return avg_h / total, avg_l / total


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device("cpu"))
    target = target.to(torch.device("cpu"))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def epoch_accuracy(loader_s, student):
    student.eval()

    out_epoch_s = [
        accuracy(student(L.to(torch.cuda.current_device())), y)[0].detach().cpu().item()
        for L, y in loader_s
    ]

    student.train()

    return sum(out_epoch_s) / len(out_epoch_s)


def test(model, loader, loss_fn, **kwargs):
    test_print(model, loader)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_losses = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data.to(torch.cuda.current_device()))
            test_loss += loss_fn(output, target.to(torch.cuda.current_device())).item() * output.shape[0]
            pred = output.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target.data.view_as(pred)).sum()
            total += target.shape[0]
        test_loss /= total
        test_losses.append(test_loss)

    return test_losses[-1], correct / total


def train_ddp(
    model, train_loader, unlabeled_loader, scheduler_actor, scheduler_critic, **kwargs
):
    step = 0
    ddp_loss = torch.zeros(5).to(torch.cuda.current_device())
    model.train()

    for (I, U, U_aug), (X, y) in tqdm(zip(unlabeled_loader, train_loader)):
        X, y = (
            X.to(torch.cuda.current_device()),
            y.to(torch.cuda.current_device()),
        )
        I, U, U_aug = (
            I.to(torch.cuda.current_device()),
            U.to(torch.cuda.current_device()),
            U_aug.to(torch.cuda.current_device()),
        )

        avg_h = model.step(I, U, U_aug, X, y)

        ddp_loss[0] += avg_h.item()

        step += 1

        scheduler_actor.step()
        scheduler_critic.step()

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.AVG)

    return avg_h


def test_ddp(model, loader, loss_fn, **kwargs):
    ddp_loss = torch.zeros(5).to(torch.cuda.current_device())
    model.eval()

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())
            output = model(X, with_variance=False, update_precision=False)
            loss = loss_fn(output.type(torch.float32), y)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (
                torch.where(y != loss_fn.ignore_index, (output.argmax(1) == y), 0)
                .type(torch.float)
                .sum()
                .item()
            )

            ddp_loss[2] += y.numel()
            ddp_loss[4] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2]
    test_loss = ddp_loss[0] / ddp_loss[2]

    return test_acc, test_loss


### supercomputer utility functions for data movement


def move_data_to_lscratch(rank, args, spec_dir):
    start = time.time()

    if rank == 0:
        LSCRATCH = os.environ["LSCRATCH"] + "/"
        os.mkdir(f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}")
        os.system(
            f"scp -r {spec_dir[args.dataset]['download_path']}/* {LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}"
        )

        print(
            glob.glob(
                f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/*.tar.gz"
            )
        )
        print(glob.glob(LSCRATCH))

        for path in glob.glob(
            f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/*.tar.gz"
        ):
            os.system(
                f"tar -xzf {path} -C {LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/"
            )

        for path in glob.glob(
            f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/*.tar"
        ):
            os.system(
                f"tar -xf {path} -C {LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/"
            )

        print(
            glob.glob(
                f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/*"
            )
        )
        print(glob.glob(f"{LSCRATCH}{spec_dir[args.dataset]['download_path']}/*"))
        print(glob.glob(LSCRATCH))

        spec_dir[args.dataset]["download_path"] = (
            f"{LSCRATCH}{spec_dir[args.dataset]['download_path'].split('/')[-1]}/"
        )

        if args.dataset == "food101":
            args.dataset_path = (
                args.dataset_path
                + "/"
                + spec_dir[args.dataset]["download_path"].split("/")[-1]
            )

        Path(
            os.path.join("/scratch/jroth/", f"done_{os.environ['SLURM_JOB_ID']}")
        ).touch()

    else:
        while not os.path.isfile(
            os.path.join("/scratch/jroth/", f"done_{os.environ['SLURM_JOB_ID']}")
        ):
            time.sleep(10)
        while os.path.getmtime(
            os.path.join("/scratch/jroth/", f"done_{os.environ['SLURM_JOB_ID']}")
        ) < (start + 10):
            time.sleep(10)

    print("data unzipped")


### utility functions for selecting objects for training (data, models) using config elements


def select_dataset(args, data_dir, spec_dir, unlbl_transform, train_transform, val_transform):

    ds_train = data_dir[args.dataset](
        spec_dir[args.dataset]["download_path"],
        transform=train_transform,
        download=False,
        **spec_dir[args.dataset]["train"],
    )

    if spec_dir[args.dataset]["unlabeled"] is not None:
        ds_unlabeled = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=unlbl_transform,
            download=False,
            **spec_dir[args.dataset]["unlabeled"],
        )

    else:
        ds_unlabeled = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=None,
            download=False,
            **spec_dir[args.dataset]["train"],
        )

        # Generate indices for splitting
        indices = list(range(len(ds_train)))

        # Perform stratified split
        labelled_indices = []
        unlabelled_indices = []

        indices = np.random.permutation(len(ds_unlabeled))
        class_counters = list([0] * 10)
        max_counter = 4000 // 10

        for i in indices:
            dp = ds_unlabeled[i]
            y = dp[1]
            c = class_counters[y]
            if c < max_counter:
                class_counters[y] += 1
                labelled_indices.append(i)

        unlabelled_indices = indices

        ds_unlabeled = UDADataset(
            IndexSubsetDataset(ds_unlabeled, list(unlabelled_indices)),
            transform=train_transform,
            target_transform=unlbl_transform,
        )

        ds_train = IndexSubsetDataset(ds_train, list(labelled_indices))

    # selector for the type of dataset we will train on.  We will need to download it to a special directory from the args to make use of lscratch
    if spec_dir[args.dataset]["train"] is not None and spec_dir[args.dataset]["test"]:
        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=val_transform,
            download=False,
            **spec_dir[args.dataset]["test"],
        )
    else:
        ds_train = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=train_transform,
            download=False,
        )

        ds_train = IndexSubsetDataset(
            ds_train, sum([list(range(len(ds_train)))[i::5] for i in range(1, 5)], [])
        )

        ds_val = data_dir[args.dataset](
            spec_dir[args.dataset]["download_path"],
            transform=val_transform,
            download=False,
        )
        ds_val = IndexSubsetDataset(ds_val, list(range(len(ds_val)))[0::5])

    print(len(ds_unlabeled), len(ds_train), len(ds_val))

    return ds_unlabeled, ds_train, ds_val


def select_model(args, model_dir, spec_dir):
    if args.model in model_dir:
        model_fn = model_dir[args.model]

        model = torch.nn.Sequential(
            model_fn(), torch.nn.Linear(1000, spec_dir[args.dataset]["classes"])
        )
    else:
        raise NotImplementedError(f"unrecognized model type: {args.model}")

    return model
