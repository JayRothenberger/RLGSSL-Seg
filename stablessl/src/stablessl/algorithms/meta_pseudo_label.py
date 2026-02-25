from stablessl.algorithms.base import AlgorithmBase
from stablessl.callbacks import (
    CoTrainingPLCallback,
    MaybeResetModelCallback,
    CoTrainingEvalCallback,
)
import torch
import os
import torch.distributed as dist
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm


class MetaPseudoLabelAlgorithm(AlgorithmBase):
    def __init__(self, agent, args, model_fn, optimizer_fn, scheduler_fn, dataset):
        self.args = args

        self.is_teacher = int(os.environ["RANK"]) >= int(
            int(os.environ["WORLD_SIZE"]) // 2
        )

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
        ) = [], [], [], [], [], []

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

    def step_student(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        predictions = []
        ground_truth = []

        loss_initial, loss_final = 0, 0

        # measure the initial loss on the labeled batch
        batches = []
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for i in range(self.args.accumulate):
                    X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
                    batches.append((X, y))

                    pred = self.model(X.to(torch.cuda.current_device()))

                    loss_initial += (
                        loss_fn(pred, y.to(torch.cuda.current_device()))
                        / self.args.accumulate
                    )

        # gather the unlabeled instances to ensure consistency
        zero_tensor = torch.zeros(
            (self.args.unlabeled_batch_size, X.shape[1], X.shape[2], X.shape[3])
        ).to(torch.cuda.current_device())
        gather_tensors_unlabeled = [
            zero_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]

        dist.all_gather(gather_tensors_unlabeled, zero_tensor)

        # gather the teacher's prediction
        zero_tensor = torch.zeros((self.args.unlabeled_batch_size, pred.shape[-1])).to(
            torch.cuda.current_device()
        )
        gather_tensors_pseudo_label = [
            zero_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]

        dist.all_gather(gather_tensors_pseudo_label, zero_tensor)

        # harden the pseudo label
        U, pl = (
            gather_tensors_unlabeled[
                int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
            ],
            gather_tensors_pseudo_label[
                int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
            ].argmax(-1),
        )

        # update this model
        unfreeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            pred = self.model(U.to(torch.cuda.current_device()))
            loss = loss_fn(pred, pl.to(torch.cuda.current_device()))

            self.scaled_clipped_gradient_update(
                self.model, loss, self.optimizer, self.scaler, False
            )

        # measure the change in loss
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for X, y in batches:
                    # measure the initial loss on the labeled batch
                    pred = self.model(X.to(torch.cuda.current_device()))

                    predictions.append(pred)
                    ground_truth.append(y)

                    loss_final += (
                        loss_fn(pred, y.to(torch.cuda.current_device()))
                        / self.args.accumulate
                    )

        h = loss_initial - loss_final
        # reduce the change in loss
        h_tensor = h.to(
            torch.cuda.current_device()
        )

        gather_tensors_h = [
            h_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]

        dist.all_gather(gather_tensors_h, h_tensor)

        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def step_teacher(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        predictions = []
        ground_truth = []

        unlabeled_batches = []
        PLs = []

        # predict on the unlabeled data
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for i in range(self.args.accumulate):
                    U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
                    unlabeled_batches.append(U)

                    pred = self.model(U.to(torch.cuda.current_device()))

                    PLs.append(pred)


        U = torch.cat(unlabeled_batches, 0).to(torch.cuda.current_device())
        PL = torch.cat(PLs, 0).to(torch.cuda.current_device())

        # gather the unlabeled instances to ensure consistency for the student
        gather_tensors_unlabeled = [
            U.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]
        dist.all_gather(gather_tensors_unlabeled, U)

        # gather the teacher's prediction
        gather_tensors_pseudo_label = [
            PL.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]
        dist.all_gather(gather_tensors_pseudo_label, PL)

        # update this model based on the student's performance
        zero_tensor = torch.tensor(0.0).to(
            torch.cuda.current_device()
        )
        gather_tensors_h = [
            zero_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]
        
        dist.all_gather(gather_tensors_h, zero_tensor)

        h = gather_tensors_h[int(os.environ['RANK']) - int(int(os.environ['WORLD_SIZE']) / 2)]

        # update based on the labeled data (optional)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            for i in range(self.args.accumulate):
                X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
                # measure the initial loss on the labeled batch

                pred = self.model(X_aug.to(torch.cuda.current_device()))

                loss = loss_fn(pred, y.to(torch.cuda.current_device()))

                self.scaled_clipped_gradient_update(
                    self.model,
                    loss,
                    self.optimizer,
                    self.scaler,
                    True, # this is only accumulated so that later it can be applied
                )

                predictions.append(pred.detach().clone()), ground_truth.append(y)

            for i, (U_batch, pl) in enumerate(zip(unlabeled_batches, PLs)):
                loss = h * loss_fn(self.model(U_batch.to(torch.cuda.current_device())), pl.argmax(-1).to(torch.cuda.current_device()))

                self.scaled_clipped_gradient_update(
                    self.model,
                    loss,
                    self.optimizer,
                    self.scaler,
                    i < (self.args.accumulate - 1),
                )


        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def step(self):
        step_fn = self.step_teacher if self.is_teacher else self.step_student

        predictions, ground_truth = step_fn()

        return predictions, ground_truth

    def reset_model(self):
        self.model = self.model_fn().to(torch.cuda.current_device())
        self.optimizer = self.optimizer_fn(self.model.parameters())
        self.scheduler = self.scheduler_fn(self.optimizer)
        self.scaler = torch.amp.GradScaler("cuda")

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)
