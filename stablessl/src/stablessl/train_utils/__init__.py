import torch
import os
import wandb
from stablessl.train_utils.metrics import TrainingMetrics, ValidationMetrics
from tqdm import tqdm
import warnings


def freeze_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.eval()

def unfreeze_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.train()


def model_run(
    args,
    model,
    dataset,
):
    best_val_acc = 0

    if int(os.environ.get("WORLD_SIZE")) > 1:
        print("please make sure each rank is running the training process")

    test_fn = test
    train_fn = train

    iterations = vars(args).get('iterations')
    iterations = iterations if iterations is not None else 1

    print(f"training model for {iterations} iterations at {args.epochs} epochs each...")
    for iteration in range(iterations):
        # create the loaders and samplers for the dataset if they do not exist
        dataset.get_loaders()
        # start iteration callback
        model.iteration_start_callback()
        print(f'starting iteration {iteration}')
        epochs_since_improvement = 0
        for epoch in range(args.epochs):
            model.epoch_start_callback()

            for sampler in dataset.samplers.values():
                sampler.set_epoch(epoch)

            train_stats = train_fn(args, model, dataset)
            validation_stats = test_fn(args, model, dataset)

            wandb.log(
                {
                    'iteration': iteration,
                    'epoch': epoch,
                    **dict(train_stats),
                    **dict(validation_stats)
                }
            )

            if validation_stats['val_acc'] > best_val_acc:
                epochs_since_improvement = 0
                best_val_acc = validation_stats['val_acc']
                model.checkpoint_state()
            else:
                if epochs_since_improvement >= args.patience:
                    break
                epochs_since_improvement += 1
            
            model.epoch_end_callback()
        model.iteration_end_callback()

    model.load_checkpoint_state()

    dataset.get_loaders()

    validation_stats = test_fn(args, model, dataset)

    wandb.log(
        {
            **{k + ' final': v for k, v in dict(train_stats).items()},
            **{k + ' final': v for k, v in dict(validation_stats).items()}
        }
    )

    return model, best_val_acc


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


def train(args, model, dataset):
    model.train()

    statistics = TrainingMetrics()

    for _ in tqdm(range(args.steps_per_epoch)):
        statistics.update(*model.step())

    # statistics.reduce()

    return statistics


def test(args, model, dataset):
    loader = dataset.loaders['validation']

    model.eval()

    try:
        loss_fn = model.loss_fn
    except Exception as e:
        loss_fn = torch.nn.CrossEntropyLoss()

    statistics = ValidationMetrics()

    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(torch.cuda.current_device()), y.to(torch.cuda.current_device())
            output = model(X)
            loss = loss_fn(output.type(torch.float32), y)
            
            statistics.update(output.cpu(), y.cpu())

    return statistics


def setup(rank, world_size):
    if int(os.environ['LOCAL_WORLD_SIZE']) > torch.cuda.device_count():
        warnings.warn('Detected that there are more processes on this node than GPUs.  Falling back to GLOO backend which may cause some distributed computations to fail with CUDA tensors.')
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()
