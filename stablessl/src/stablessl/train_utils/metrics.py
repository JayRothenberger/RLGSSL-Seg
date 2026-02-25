import torch
import os
from torchmetrics.classification import MulticlassJaccardIndex

def args_to_tensor(func):

    def wrapper_func(self, *args, **kwargs):
        args = [torch.tensor(arg) if not isinstance(arg, torch.Tensor) else arg for arg in args]

        for k in kwargs:
            if not isinstance(kwargs[k], torch.Tensor):
                kwargs[k] = torch.tensor(kwargs[k])

        return func(self, *args, **kwargs)

    return wrapper_func

class Metric:
    def __init__(self, y_pred, y_true, name='metric'):
        self.name = name
        self.y_pred = None
        self.y_true = None
        self.metric = None
        self.figure = None

        if (y_true is not None) and (y_pred is not None):
            self.update(y_pred, y_true)

    def update(self, y_pred, y_true):
        raise NotImplementedError()
    
    def __iter__(self):
        yield self.name, self.metric
    
    def reset(self):
        self.y_pred = None
        self.y_true = None
        self.metric = None
        self.figure = None

    def reduce(self):
        pred_tensors = [torch.tensor(self.y_pred).to(torch.cuda.current_device()).clone() for _ in range(int(os.environ['WORLD_SIZE']))]
        true_tensors = [torch.tensor(self.y_true).to(torch.cuda.current_device()).clone() for _ in range(int(os.environ['WORLD_SIZE']))]

        torch.distributed.all_gather(pred_tensors, self.y_pred.clone().to(torch.cuda.current_device()))
        torch.distributed.all_gather(true_tensors, self.y_true.clone().to(torch.cuda.current_device()))

        y_pred = torch.cat(pred_tensors, 0)
        y_true = torch.cat(true_tensors, 0)

        self.reset()

        self.update(y_pred, y_true)


class Accuracy(Metric):
    def __init__(self, y_pred=None, y_true=None, name='accuracy'):
        super().__init__(y_pred, y_true, name=name)

    @args_to_tensor
    def update(self, y_pred, y_true):
        if len(y_pred.shape) >= 2:
            y_pred = torch.argmax(y_pred, 1)

        assert (len(y_true) == len(y_pred)) and len(y_pred), f'Length of inputs must match and be greater than zero, found: {len(y_true)}, {len(y_pred)}'

        if self.y_true is not None:
            self.y_pred = torch.cat((y_pred.cpu(), self.y_pred.cpu()), 0)
            self.y_true = torch.cat((y_true.cpu(), self.y_true.cpu()), 0)
        else:
            self.y_pred = y_pred.cpu()
            self.y_true = y_true.cpu()

        self.metric = (self.y_pred == self.y_true).type(torch.float64).mean().item()


class MJaccard(Metric):
    def __init__(self, y_pred=None, y_true=None, name='jaccard'):
        super().__init__(y_pred, y_true, name=name)
        self.count = 0
        self.sum_jac = 0
    
    def reset(self):
        self.count = 0
        self.sum_jac = 0

    @args_to_tensor
    def update(self, y_pred, y_true):
        jaccard = MulticlassJaccardIndex(
                num_classes=y_pred.shape[1], ignore_index=255
            )
        
        self.sum_jac += jaccard(y_pred.argmax(1), y_true)
        self.count += 1
        
        self.metric = self.sum_jac / self.count

class TrainingMetrics:
    def __init__(self):
        self.metrics = [Accuracy(name='train_acc'), MJaccard(name='train_jaccard')]

    def update(self, y_pred, y_true):
        for metric in self.metrics:
            metric.update(y_pred, y_true)

    def __getitem__(self, k):
        return {k: v for metric in self.metrics for k, v in dict(metric).items()}[k]

    def __iter__(self):
        for metric in self.metrics:
            for k, v in dict(metric).items():
                yield k, v
    
    def reduce(self):
        for metric in self.metrics:
            metric.reduce()

class ValidationMetrics:
    def __init__(self):
        self.metrics = [Accuracy(name='val_acc'), MJaccard(name='val_jaccard')]

    def update(self, y_pred, y_true):
        for metric in self.metrics:
            metric.update(y_pred, y_true)

    def __getitem__(self, k):
        return {k: v for metric in self.metrics for k, v in dict(metric).items()}[k]
    
    def __iter__(self):
        for metric in self.metrics:
            for k, v in dict(metric).items():
                yield k, v
    
    def reduce(self):
        for metric in self.metrics:
            metric.reduce()