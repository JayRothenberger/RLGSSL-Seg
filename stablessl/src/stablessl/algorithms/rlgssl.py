import torch
import torch.nn as nn
from copy import deepcopy as copy
from stablessl.algorithms.base import AlgorithmBase
import numpy as np
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm


class ModelEMA:
    """
    EMA model
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                name
            ]
            self.shadow[name] = new_average.clone()

        for name, buffer in self.model.named_buffers():
            self.shadow[name] = buffer.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}

    def eval(self):
        freeze_batchnorm(self.model)

    def train(self):
        unfreeze_batchnorm(self.model)


@torch.no_grad()
def mixup(x, y, u, pl, alpha=0.5, is_bias=True):
    """Returns mixed inputs, mixed targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(u.size(0)).to(x.device)

    assert x.shape == u.shape, (u.shape, x.shape)

    mixed_x = lam * u + (1 - lam) * x[index]
    mixed_y = lam * pl + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


class RLGSSLAlgorithm(AlgorithmBase):
    def __init__(
        self,
        agent,
        args,
        model_fn,
        optimizer_fn,
        scheduler_fn,
        dataset,
        ema_decay=0.999,
        mean_reward=False,
        loss_fn = torch.nn.CrossEntropyLoss()
    ):
        self.args = args
        self.dataset = dataset

        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

        model = self.model_fn().to(torch.cuda.current_device())
        optimizer = self.optimizer_fn(model.named_parameters())
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

        self.loss_fn = loss_fn

        self.model = model
        self.model_ema = ModelEMA(model, ema_decay)
        self.model_ema.register()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.mean_reward = mean_reward

        self.step_number = 0

    def step(self):
        loss_fn = self.loss_fn
        
        predictions = []
        ground_truth = []

        # compute the soft pseudo label data
        for i in range(self.args.accumulate):
            with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
                U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
                X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])

                X, X_aug, U, U_aug, y = (
                    X.to(torch.cuda.current_device()),
                    X_aug.to(torch.cuda.current_device()),
                    U.to(torch.cuda.current_device()),
                    U_aug.to(torch.cuda.current_device()),
                    y.to(torch.cuda.current_device()),
                )

                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda", dtype=self.args.autocast_type
                    ):
                        freeze_batchnorm(self.model)
                        self.model_ema.apply_shadow()
                        unlabeled_logits = self.model(U)
                        self.model_ema.restore()

                        unlabeled_probs = torch.softmax(
                            unlabeled_logits, dim=1
                        ).detach()

                        # generate the mixup data
                        temp_num_classes = max(y.max().item() + 1, unlabeled_logits.shape[1])

                        one_hot_labels = torch.nn.functional.one_hot(
                            y, temp_num_classes
                        )[..., :unlabeled_logits.shape[1]]
                        
                        if len(unlabeled_probs.shape) == 4:
                            one_hot_labels = one_hot_labels.permute(0, 3, 1, 2)

                        assert one_hot_labels.shape == unlabeled_probs.shape, (one_hot_labels.shape, unlabeled_probs.shape)

                        mixed_x, mixed_y, _ = mixup(
                            X, one_hot_labels, U, unlabeled_probs
                        )

                        # calculate the reward
                        freeze_batchnorm(self.model)
                        mixed_predictions = self.model(mixed_x)
                unfreeze_batchnorm(self.model)
                # compute the RL loss
                unlabeled_preds = self.model(U)

                assert mixed_predictions.shape == mixed_y.shape, (mixed_predictions.shape, mixed_y.shape)

                reward = -torch.nn.functional.mse_loss(torch.softmax(mixed_predictions, dim=1), mixed_y, reduction='none')

                kl_div = torch.nn.KLDivLoss(reduction="none")(
                    torch.full_like(unlabeled_preds, 1 / unlabeled_logits.shape[-1]),
                    unlabeled_preds,
                )

                if self.mean_reward:
                    rl_loss = reward.mean().detach() * kl_div.sum(1).mean()
                else:
                    rl_loss = (reward.mean(1).detach() * kl_div.sum(1)).mean()

                # compute the Consistency loss

                cons_loss = torch.nn.KLDivLoss(reduction="none")(
                    unlabeled_preds, unlabeled_probs
                ).sum(1).mean()

                # compute the Supervised loss
                pred = self.model(X)

                predictions.append(pred), ground_truth.append(y)

                sup_loss = loss_fn(pred, y)

                # backprop the combined loss
                self.step_number += 1

                if self.step_number >= 0 * self.args.accumulate:
                    total_loss = (1 * rl_loss) + (0.1 * sup_loss) + (0.1 * cons_loss)
                else:
                    total_loss = sup_loss

                self.scaled_clipped_gradient_update(
                    self.model,
                    total_loss,
                    self.optimizer,
                    self.scaler,
                    i < (self.args.accumulate - 1),
                )
        else:
            # update the teacher
            self.model_ema.update()

        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)
