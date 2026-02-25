from stablessl.algorithms.base import AlgorithmBase
from stablessl.callbacks import SelfTrainingPLCallback, MaybeResetModelCallback
import torch

class SelfTrainingAlgorithm(AlgorithmBase):
    def __init__(self, agent, args, model_fn, optimizer_fn, scheduler_fn, dataset):
        self.args = args
        self.dataset = dataset

        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

        model = self.model_fn().to(torch.cuda.current_device())
        optimizer = self.optimizer_fn(model.parameters())
        scheduler = self.scheduler_fn(optimizer)
        scaler = torch.amp.GradScaler("cuda")

        (
            iteration_start_callbacks,
            epoch_start_callbacks,
            update_start_callbacks,
            iteration_end_callbacks,
            epoch_end_callbacks,
            update_end_callbacks,
        ) = [], [], [], [SelfTrainingPLCallback(self, args.autocast_type, args.with_replacement, args.strategy, args.k, args.threshold, args.percent), MaybeResetModelCallback(self)], [], []

        super().__init__(
            agent,
            args,
            dataset,
            iteration_start_callbacks,
            epoch_start_callbacks,
            update_start_callbacks,
            iteration_end_callbacks,
            epoch_end_callbacks,
            update_end_callbacks,
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

    def step(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        predictions = []
        ground_truth = []
        
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            for i in range(self.args.accumulate):
                X, X_aug, y = next(self.dataset.iterable_loaders['labeled'])

                pred = self.model(X.to(torch.cuda.current_device()))

                predictions.append(pred), ground_truth.append(y)

                loss = loss_fn(pred, y.to(torch.cuda.current_device()))

                self.scaled_clipped_gradient_update(self.model, loss, self.optimizer, self.scaler, i < (self.args.accumulate - 1))

        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()
    
    def reset_model(self):
        self.model = self.model_fn().to(torch.cuda.current_device())
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scheduler = self.scheduler_fn(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda")

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)