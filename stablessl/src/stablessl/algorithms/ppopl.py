from stablessl.algorithms.base import AlgorithmBase
from stablessl.callbacks.epoch_end import MetaCoTrainingEvalCallback
import torch
import os
import torch.distributed as dist
import copy
import gc
import warnings
import math
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm
import wandb
from torch.distributions.categorical import Categorical


def shap_approx(A, B):
    # but the weights are all the same if we are using the same batch size for each batch
    # W = (A.shape[-1] - 1) / (torch_choice(A.shape[-1], A.sum(-1)) * A.sum(-1) * (A.shape[-1] - A.sum(-1)))
    # so then we can use the torch lstsq method instead of writing our own.
    A, B = A.to(torch.cuda.current_device()), B.to(torch.cuda.current_device())
    X = torch.linalg.lstsq(A, B).solution
    # X = torch.linalg.pinv(A.T @ A) @ A.T @ B

    return X


def detach_tensors(od):
    if isinstance(od, torch.Tensor):
        return od.detach().clone()
    if isinstance(od, float):
        return float(od)
    if isinstance(od, list):
        return [detach_tensors(i) for i in od]
    if isinstance(od, dict):
        for name, tensor in od.items():
            od[name] = detach_tensors(tensor)
        return od

    return copy.deepcopy(od)


class SHAPEstimator:
    def __init__(
        self,
        algorithm,
        est_iters=512,
        batch_size=0.125,
    ):
        self.step = 0

        self.algorithm = algorithm
        self.est_iters = est_iters
        self.batch_size = batch_size

    def on_policy_shap(self, masks, hs):
        weights = shap_approx(masks.type(torch.float32), hs).squeeze()

        # weights = (weights - weights.mean()) / torch.std(weights)

        # weights = (weights - weights.mean()) / torch.std(weights)

        # new = (new > 1).type(torch.float32)

        return weights

    def shap_bayes(self, masks, hs):
        hs, masks = (
            hs.squeeze().to(torch.cuda.current_device()),
            masks.to(torch.cuda.current_device()).type(torch.float32),
        )

        counts = masks.sum(0)

        masks = masks * hs.unsqueeze(-1)

        # intersection = masks[hs > 0]

        new = masks.sum(0) / counts  # (intersection.sum(0) / (masks.abs().sum(0) + 1)).squeeze()

        new = (new - new.mean()) / torch.std(new)

        # new = (new > 1).type(torch.float32)

        return torch.clamp(new, -100, 100)

    def checkpoint(self):
        self.initial_state = detach_tensors(
            copy.deepcopy(self.algorithm.model.state_dict().copy())
        )
        self.initial_opt_state = detach_tensors(
            copy.deepcopy(self.algorithm.optimizer.state_dict().copy())
        )
        self.initial_scaler_state = detach_tensors(
            copy.deepcopy(self.algorithm.scaler.state_dict().copy())
        )

    def load_checkpoint(self):
        self.algorithm.model.load_state_dict(self.initial_state)
        self.algorithm.optimizer.load_state_dict(self.initial_opt_state)
        self.algorithm.scaler.load_state_dict(self.initial_scaler_state)

    def free_checkpoint(self):
        self.initial_state = None
        self.initial_opt_state = None
        gc.collect()

    def __call__(self, X, y, U, a):
        if 0 < self.batch_size <= 1:
            warnings.warn(
                f"interpreting batch size for training as fraction of unlabeled batch size {self.batch_size * 100}% of {U.shape[0]} is {math.ceil(self.batch_size * U.shape[0])}"
            )
            self.batch_size = math.ceil(self.batch_size * U.shape[0])

        self.step += 1

        masks = torch.rand((self.est_iters, a.shape[0]))
        masks = masks <= torch.sort(masks, -1)[0][..., self.batch_size - 1].unsqueeze(
            -1
        )

        hs = []

        opt = self.algorithm.optimizer

        self.algorithm.optimizer = torch.optim.Adam(
            self.algorithm.model.parameters(), lr=1e-4, betas=(1e-12, 1e-12)
        )

        self.checkpoint() # checkpoint model and optimizer states

        freeze_batchnorm(self.algorithm.model)

        for m, mask in enumerate(masks):
            # TODO: function that steps the model here
            h = self.algorithm.environment_step(X, y, U[mask], a[mask])

            hs.append(h)

            self.load_checkpoint()

        self.algorithm.optimizer = opt

        hs = torch.tensor(hs).unsqueeze(-1).to(torch.cuda.current_device())

        weights = self.on_policy_shap(masks, hs)

        self.algorithm.model.train()

        return a, weights


class PPOPLAlgorithm(AlgorithmBase):
    def __init__(
        self,
        agent,
        args,
        model_fn,
        optimizer_fn,
        scheduler_fn,
        dataset,
        target_kl=0.1,
        maxit=1,
        warmup=0,
    ):
        self.args = args
        self.dataset = dataset

        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn

        model = self.model_fn().to(torch.cuda.current_device())
        optimizer = self.optimizer_fn(model.parameters())
        scheduler = self.scheduler_fn(optimizer)
        scaler = torch.amp.GradScaler("cuda")

        self.estimator = SHAPEstimator(self)

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

        self.maxit = maxit
        self.target_kl = target_kl

        self.rl_data = dataset.rl_data
        self.opt_step = 0
        self.warmup = warmup

    def environment_step(self, X, y, U, a):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                loss_initial = (
                    loss_fn(
                        self.model(
                            X.to(torch.cuda.current_device()), deterministic=True
                        ),
                        y.to(torch.cuda.current_device()),
                    )
                    .detach()
                    .clone()
                )

            critic_unlabeled_logits = self.model(U, deterministic=True)
            tmp_loss = loss_fn(
                critic_unlabeled_logits, a.to(torch.cuda.current_device())
            )

            self.scaled_clipped_gradient_update(
                self.model, tmp_loss, self.optimizer, self.scaler, False
            )

            # now we can compute the final loss of the value network on the labeled data
            with torch.no_grad():
                loss_final = (
                    loss_fn(
                        self.model(
                            X.to(torch.cuda.current_device()), deterministic=True
                        ),
                        y.to(torch.cuda.current_device()),
                    )
                    .detach()
                    .clone()
                )

        h = (loss_initial.mean() - loss_final.mean()).to(torch.cuda.current_device())

        return h

    def step(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        predictions = []
        ground_truth = []

        if self.accumulate != 1:
            raise NotImplementedError("Gradient Accumulation not implemented for PPOPL")

        U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
        X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])

        X, U, U_aug, y = (
            X.to(torch.cuda.current_device()),
            U.to(torch.cuda.current_device()),
            U_aug.to(torch.cuda.current_device()),
            y.to(torch.cuda.current_device()),
        )

        if self.opt_step > self.warmup:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    freeze_batchnorm(self.model)
                    actor_unlabeled_logits, log_probs_initial = self.model(
                        U, log_prob=True, deterministic=False
                    )

                    a = torch.softmax(actor_unlabeled_logits * 4, dim=-1).detach()

            # use the shap estimator
            freeze_batchnorm(self.model)
            a, h = self.estimator(*self.rl_data, U, a)

            kl, it = torch.tensor(0), 0

            unfreeze_batchnorm(self.model)

            while it < self.maxit:
                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    if it > 0:
                        aul, dist = self.model._distribution(U)

                        log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                        kl = (log_probs_initial - log_probs).mean().item()
                        if kl > (1.5 * self.target_kl):
                            break
                    else:
                        aul, dist = self.model._distribution(U)

                        log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                        kl = (log_probs_initial - log_probs).mean().item()

                    assert log_probs.shape == log_probs_initial.shape, (
                        log_probs.shape,
                        log_probs_initial.shape,
                    )

                    ratio = torch.exp(log_probs - log_probs_initial)

                    log_probs_final = log_probs.detach().clone()

                    assert h.shape == ratio.shape, (h.shape, ratio.shape)

                    student = torch.nn.CrossEntropyLoss(reduction="none")(
                        aul, a.detach().clone()
                    )

                    clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * (
                        h * student
                    )

                loss_pi = -(torch.min(ratio * (h * student), clip_adv)).mean()

                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    self.model.eval()
                    output = self.model(X, deterministic=True)

                    loss_sup = loss_fn(output, y)
                    predictions.append(output.detach().clone())
                    ground_truth.append(y.clone())

                total_loss = (0.1 * loss_pi) + loss_sup

                self.scaled_clipped_gradient_update(
                    self.model,
                    total_loss,
                    self.optimizer,
                    self.scaler,
                )

                it += 1
        else:
            with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                unfreeze_batchnorm(self.model)
                output = self.model(X, deterministic=True)

                loss_sup = loss_fn(output, y)

                predictions.append(output.detach().clone())
                ground_truth.append(y.clone())

            total_loss = loss_sup

            self.scaled_clipped_gradient_update(
                self.model,
                total_loss,
                self.optimizer,
                self.scaler,
            )
        self.opt_step += 1
        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x, deterministic=True)


class CoPPOPLAlgorithm(AlgorithmBase):
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
        target_kl=0.01,
        maxit=1,
        warmup=0,
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

        self.estimator = SHAPEstimator(self)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.maxit = maxit
        self.target_kl = target_kl

        self.rl_data = self.dataset.rl_data
        self.opt_step = 0
        self.warmup = warmup

    def environment_step(self, X, y, U, a):
        loss_fn = torch.nn.CrossEntropyLoss()

        freeze_batchnorm(self.model)

        with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
            with torch.no_grad():
                loss_initial = (
                    loss_fn(
                        self.model(
                            X.to(torch.cuda.current_device()), deterministic=True
                        ),
                        y.to(torch.cuda.current_device()),
                    )
                    .detach()
                    .clone()
                )

            critic_unlabeled_logits = self.model(U, deterministic=True)
            tmp_loss = loss_fn(
                critic_unlabeled_logits, a.to(torch.cuda.current_device())
            )

            self.scaled_clipped_gradient_update(
                self.model, tmp_loss, self.optimizer, self.scaler, False
            )

            # now we can compute the final loss of the value network on the labeled data
            with torch.no_grad():
                loss_final = (
                    loss_fn(
                        self.model(
                            X.to(torch.cuda.current_device()), deterministic=True
                        ),
                        y.to(torch.cuda.current_device()),
                    )
                    .detach()
                    .clone()
                )

        h = (loss_initial.mean() - loss_final.mean()).to(torch.cuda.current_device())

        return h

    def step(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        predictions = []
        ground_truth = []

        unlabeled_batches = []
        PLs = []
        clean_PLs = []
        inds = []

        if self.accumulate != 1:
            raise NotImplementedError("Gradient Accumulation not implemented for CoPPOPL")

        I_u, U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])

        # I_u, U, U_aug = torch.cat((I_u, I_u), 0), torch.cat((U, U), 0), torch.cat((U_aug, U_aug), 0)

        I_u, order_u = torch.sort(I_u)

        U, U_aug = U[order_u], U_aug[order_u]

        U, U_aug = (
            U.to(torch.cuda.current_device()),
            U_aug.to(torch.cuda.current_device()),
        )

        if self.opt_step >= self.warmup:
            if self.opt_step == self.warmup:
                self.optimizer = self.optimizer_fn(self.model.parameters())
                self.scheduler = self.scheduler_fn(self.optimizer)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    freeze_batchnorm(self.model)
                    actor_unlabeled_logits, log_probs_initial = self.model(
                        U, log_prob=True, deterministic=False
                    )

                    clean_unlabeled_logits = self.model(
                        U, log_prob=False, deterministic=True
                    )

                    unlabeled_batches.append(U)

                    pl = torch.softmax(actor_unlabeled_logits, dim=-1).detach() 
                    clean_pl = torch.softmax(clean_unlabeled_logits, dim=-1).detach() 

                    pl = torch.nn.functional.one_hot(torch.distributions.categorical.Categorical(probs=clean_pl).sample(), pl.shape[-1])

                    PLs.append(pl)
                    clean_PLs.append(clean_pl)


            # gather the unlabeled instances to ensure consistency for the student
            U = torch.cat(unlabeled_batches, 0).to(torch.cuda.current_device())
            PL = torch.cat(PLs, 0).to(torch.cuda.current_device())
            clean_PL = torch.cat(clean_PLs, 0).to(torch.cuda.current_device())

            I_u = I_u.to(torch.cuda.current_device())

            gather_tensors_I = [
                I_u.clone() for i in range(int(os.environ["WORLD_SIZE"]))
            ]

            torch.distributed.all_gather(gather_tensors_I, I_u)

            # assert torch.unique(I_u).shape[0] == (I_u.shape[0] // 2), (torch.unique(I_u).shape[0] / I_u.shape[0])

            if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
                assert (
                    I_u
                    == gather_tensors_I[
                        int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
                    ]
                ).all(), I_u
            else:
                assert (
                    I_u
                    == gather_tensors_I[
                        int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                    ]
                ).all(), I_u

            # gather the teacher's prediction
            gather_tensors_pseudo_label = [
                PL.clone() for i in range(int(os.environ["WORLD_SIZE"]))
            ]
            torch.distributed.all_gather(gather_tensors_pseudo_label, PL)

            gather_tensors_clean_pseudo_label = [
                clean_PL.clone() for i in range(int(os.environ["WORLD_SIZE"]))
            ]
            torch.distributed.all_gather(gather_tensors_clean_pseudo_label, clean_PL)
             
            # student update
            # harden the pseudo label

            if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
                clean_pl, pl = (
                    gather_tensors_clean_pseudo_label[
                        int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                    ],
                    gather_tensors_pseudo_label[
                        int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
                    ]
                )
            else:
                clean_pl, pl = (
                    gather_tensors_clean_pseudo_label[
                        int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                    ],
                    gather_tensors_pseudo_label[
                        int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                    ]
                )

            # assert not (gather_tensors_pseudo_label[0] == gather_tensors_pseudo_label[1]).all(), pl

            # use the shap estimator
            freeze_batchnorm(self.model)

            I_l, X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
            I_l, order_l = torch.sort(I_l)
            X, X_aug, y = X[order_l], X_aug[order_l], y[order_l]

            X, y = (
                X.to(torch.cuda.current_device()),
                y.to(torch.cuda.current_device()),
            )

            a, h = self.estimator(X, y, U, pl.argmax(-1))

            h_tensor = h.to(torch.cuda.current_device())

            gather_tensors_h = [
                h_tensor.clone() for i in range(int(os.environ["WORLD_SIZE"]))
            ]

            torch.distributed.all_gather(gather_tensors_h, h_tensor)

            assert int(int(os.environ["WORLD_SIZE"]) / 2) > 0

            if int(os.environ["RANK"]) < int(int(os.environ["WORLD_SIZE"]) / 2):
                h = gather_tensors_h[
                    int(os.environ["RANK"]) + int(int(os.environ["WORLD_SIZE"]) / 2)
                ]
            else:
                h = gather_tensors_h[
                    int(os.environ["RANK"]) - int(int(os.environ["WORLD_SIZE"]) / 2)
                ]

            # assert not (gather_tensors_h[0] == gather_tensors_h[1]).all(), h

            kl, it = torch.tensor(0), 0

            while it < self.maxit:
                I_l, X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
                I_l, order_l = torch.sort(I_l)
                X, X_aug, y = X[order_l], X_aug[order_l], y[order_l]

                X, y = (
                    X.to(torch.cuda.current_device()),
                    y.to(torch.cuda.current_device()),
                )

                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    if it > 0:
                        freeze_batchnorm(self.model)
                        aul, dist = self.model._distribution(U)

                        log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                        kl = (log_probs_initial - log_probs).mean().item()

                        kl_penalty = (log_probs_initial - log_probs).mean()

                        if kl > (1.5 * self.target_kl):
                            break
                    else:
                        unfreeze_batchnorm(self.model)
                        aul, dist = self.model._distribution(U)

                        log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                        kl = (log_probs_initial - log_probs).mean().item()

                        kl_penalty = (log_probs_initial - log_probs).mean()

                    assert log_probs.shape == log_probs_initial.shape, (
                        log_probs.shape,
                        log_probs_initial.shape,
                    )

                    # ratio = torch.exp(torch.clamp(log_probs - log_probs_initial, -32, 32))

                    # assert h.shape == ratio.shape, (h.shape, ratio.shape)

                    # clip_adv = (
                    #     torch.clamp(ratio, 0.8, 1.2) * h
                    # )

                
                # loss_pi = -(log_probs[h > 0] * h[h > 0]).sum() 
                # -(torch.min(ratio * h, clip_adv)).mean()

                loss_pi = torch.nn.CrossEntropyLoss(reduction='none')(aul, PL.argmax(-1))

                assert not loss_pi.isinf().any()
                assert not loss_pi.isnan().any()
                assert loss_pi.shape == h.shape

                loss_pi = (loss_pi * h).sum()

                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                    freeze_batchnorm(self.model)

                    clean_logits = self.model(
                        U, log_prob=False, deterministic=True
                    )

                    loss_student = loss_fn(clean_logits, (clean_PL + clean_pl).argmax(-1))

                    unfreeze_batchnorm(self.model)

                    output = self.model(X, deterministic=True)

                    loss_sup = loss_fn(output, y)

                    predictions.append(output.detach().clone())
                    ground_truth.append(y.clone())

                # if loss_sup < 0.1:
                #     loss_sup = torch.tensor(0)

                total_loss = (self.args.pg_weight * loss_pi) + loss_sup + (self.args.student_weight * loss_student)

                self.scaled_clipped_gradient_update(
                    self.model,
                    total_loss,
                    self.optimizer,
                    self.scaler,
                )

                wandb.log({'loss_sup': loss_sup.item(), 'loss_pi': loss_pi.item(), 'kl': kl, 'student': loss_student.item()})

                it += 1
        else:
            I_l, X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])
            I_l, order_l = torch.sort(I_l)
            X, X_aug, y = X[order_l], X_aug[order_l], y[order_l]

            X, y = (
                X.to(torch.cuda.current_device()),
                y.to(torch.cuda.current_device()),
            )

            with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
                unfreeze_batchnorm(self.model)
                output = self.model(X, deterministic=True)

                loss_sup = loss_fn(output, y)

                predictions.append(output.detach().clone())
                ground_truth.append(y.clone())

            self.scaled_clipped_gradient_update(
                self.model,
                loss_sup,
                self.optimizer,
                self.scaler,
            )


        self.opt_step += 1

        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x, deterministic=True)
