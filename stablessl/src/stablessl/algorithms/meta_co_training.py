from stablessl.algorithms.base import AlgorithmBase
from stablessl.callbacks.epoch_end import MetaCoTrainingEvalCallback
import torch
import os
import torch.distributed as dist
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm


class MetaCoTrainingAlgorithm(AlgorithmBase):
    def __init__(
        self,
        agent,
        args,
        v1_model_fn,
        v1_optimizer_fn,
        v1_scheduler_fn,
        v2_model_fn,
        v2_optimizer_fn,
        v2_scheduler_fn,
        v1_dataset,
        v2_dataset,
    ):
        self.args = args

        self.is_view_1 = int(os.environ["RANK"]) < int(
            int(os.environ["WORLD_SIZE"]) // 2
        )

        self.dataset = v1_dataset if self.is_view_1 else v2_dataset
        self.other_dataset = v2_dataset if self.is_view_1 else v1_dataset

        self.v1_model_fn = v1_model_fn
        self.v1_optimizer_fn = v1_optimizer_fn
        self.v1_scheduler_fn = v1_scheduler_fn

        self.v2_model_fn = v2_model_fn
        self.v2_optimizer_fn = v2_optimizer_fn
        self.v2_scheduler_fn = v2_scheduler_fn

        assert (int(os.environ["WORLD_SIZE"]) % 2) == 0 and int(
            os.environ["WORLD_SIZE"]
        ), (
            "meta co-training is a multi-model method and thus must be launched with an even non-zero world size"
        )

        self.model_fn = self.v1_model_fn if self.is_view_1 else self.v2_model_fn
        self.optimizer_fn = (
            self.v1_optimizer_fn if self.is_view_1 else self.v2_optimizer_fn
        )
        self.scheduler_fn = (
            self.v1_scheduler_fn if self.is_view_1 else self.v2_scheduler_fn
        )

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
        ) = [], [], [], [], [MetaCoTrainingEvalCallback(self)], []

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
        loss_initial, loss_final = 0, 0

        predictions = []
        ground_truth = []

        unlabeled_batches = []
        PLs = []
        inds = []

        # predict on the unlabeled data
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for i in range(self.args.accumulate):
                    I_u, U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
                    I_u, order_u = torch.sort(I_u)
                    U, U_aug = U[order_u], U_aug[order_u]

                    unlabeled_batches.append(U)

                    pred = self.model(U.to(torch.cuda.current_device()))

                    PLs.append(pred)
                    inds.append(I_u)

        # measure the initial loss on the labeled batch
        batches = []
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for i in range(self.args.accumulate):
                    I_l, X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
                    I_l, order_l = torch.sort(I_l)
                    X, X_aug, y = X[order_l], X_aug[order_l], y[order_l]

                    batches.append((X, X_aug, y))

                    pred = self.model(X.to(torch.cuda.current_device()))

                    loss_initial += (
                        loss_fn(pred, y.to(torch.cuda.current_device()))
                        / self.args.accumulate
                    )

        # gather the unlabeled instances to ensure consistency for the student
        U = torch.cat(unlabeled_batches, 0).to(torch.cuda.current_device())
        PL = torch.cat(PLs, 0).to(torch.cuda.current_device())
        I_u = I_u.to(torch.cuda.current_device())

        gather_tensors_I = [I_u.clone() for i in range(int(os.environ["WORLD_SIZE"]))]

        torch.distributed.all_gather(gather_tensors_I, I_u)

        if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
            assert (
                I_u
                == gather_tensors_I[
                    int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
                ]
            ).all()
        else:
            assert (
                I_u
                == gather_tensors_I[
                    int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                ]
            ).all()

        # gather the teacher's prediction
        gather_tensors_pseudo_label = [
            PL.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]
        dist.all_gather(gather_tensors_pseudo_label, PL)
        # student update
        # harden the pseudo label
        if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
            pl = gather_tensors_pseudo_label[
                int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
            ].argmax(-1)
        else:
            pl = gather_tensors_pseudo_label[
                int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
            ].argmax(-1)

        unfreeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            pred = self.model(U.to(torch.cuda.current_device()))
            loss = loss_fn(pred, pl.to(torch.cuda.current_device()))

            self.scaled_clipped_gradient_update(
                self.model, loss, self.optimizer, self.scaler, True
            )
        # measure the change in loss
        freeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                for X, _, y in batches:
                    # measure the initial loss on the labeled batch
                    pred = self.model(X.to(torch.cuda.current_device()))

                    loss_final += (
                        loss_fn(pred, y.to(torch.cuda.current_device()))
                        / self.args.accumulate
                    )

        h = loss_initial - loss_final
        # reduce the change in loss
        h_tensor = h.to(torch.cuda.current_device())

        gather_tensors_h = [
            h_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
        ]

        dist.all_gather(gather_tensors_h, h_tensor)

        if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
            h = gather_tensors_h[
                int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
            ]
        else:
            h = gather_tensors_h[
                int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
            ]

        # update based on the labeled data (optional)
        unfreeze_batchnorm(self.model)
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            for i, (X, X_aug, y) in enumerate(batches):
                # measure the initial loss on the labeled batch

                pred = self.model(X_aug.to(torch.cuda.current_device()))

                loss = loss_fn(pred, y.to(torch.cuda.current_device()))

                self.scaled_clipped_gradient_update(
                    self.model,
                    loss,
                    self.optimizer,
                    self.scaler,
                    True,  # this is only accumulated so that later it can be applied
                )

                predictions.append(pred.detach().clone()), ground_truth.append(y)

            for i, (U_batch, pl) in enumerate(zip(unlabeled_batches, PLs)):
                loss = h * loss_fn(
                    self.model(U_batch.to(torch.cuda.current_device())),
                    torch.cat(PLs, 0).argmax(-1).to(torch.cuda.current_device()),
                )

                self.scaled_clipped_gradient_update(
                    self.model,
                    loss,
                    self.optimizer,
                    self.scaler,
                    i < (self.args.accumulate - 1),
                )

        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)
