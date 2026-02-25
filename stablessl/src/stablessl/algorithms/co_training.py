from stablessl.algorithms.base import AlgorithmBase
from stablessl.callbacks import CoTrainingPLCallback, MaybeResetModelCallback, CoTrainingEvalCallback
import torch
import os

class CoTrainingAlgorithm(AlgorithmBase):
    def __init__(self, agent, args, v1_model_fn, v1_optimizer_fn, v1_scheduler_fn, v2_model_fn, v2_optimizer_fn, v2_scheduler_fn, v1_dataset, v2_dataset):
        self.args = args

        view = (int(os.environ["RANK"]) % 2)
        is_view_1 = view == 0

        self.dataset = v1_dataset if is_view_1 else v2_dataset
        self.other_dataset = v2_dataset if is_view_1 else v1_dataset

        self.v1_model_fn = v1_model_fn
        self.v1_optimizer_fn = v1_optimizer_fn
        self.v1_scheduler_fn = v1_scheduler_fn

        self.v2_model_fn = v2_model_fn
        self.v2_optimizer_fn = v2_optimizer_fn
        self.v2_scheduler_fn = v2_scheduler_fn

        assert (int(os.environ["WORLD_SIZE"]) % 2) == 0 and int(os.environ["WORLD_SIZE"]), 'co-training is a multi-model method and thus must be launched with an even non-zero world size'

        self.model_fn = self.v1_model_fn if is_view_1 else self.v2_model_fn
        self.optimizer_fn = self.v1_optimizer_fn if is_view_1 else self.v2_optimizer_fn
        self.scheduler_fn = self.v1_scheduler_fn if is_view_1 else self.v2_scheduler_fn

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
        ) = [], [], [], [CoTrainingPLCallback(self, args.autocast_type, args.with_replacement, args.strategy, args.k, args.threshold, args.percent), CoTrainingEvalCallback(self), MaybeResetModelCallback(self)], [], []

        super().__init__(
            agent,
            args,
            self.dataset,
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
                _, X, X_aug, y = next(self.dataset.iterable_loaders['labeled'])

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