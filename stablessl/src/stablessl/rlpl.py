import torch
from abc import ABC
from .models import ModelEMA
from .estimators import NaiveEstimator
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import wandb

class CriticModel(torch.nn.Module):
    def __init__(self, backbone, predictor):
        super().__init__()
        self.backbone = backbone
        self.predictor = predictor

    def forward(self, x, **kwargs):
        return self.predictor(self.backbone(x), **kwargs)
    
    def _distribution(self, obs):
        return self.predictor._distribution(self.backbone(obs))

def freeze_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.eval()

def unfreeze_batchnorm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.train()

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_wait_steps=0,
    num_cycles=0.5,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(
                max(1, num_warmup_steps + num_wait_steps)
            )

        progress = float(current_step - num_warmup_steps - num_wait_steps) / float(
            max(1, num_training_steps - num_warmup_steps - num_wait_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ReplayBuffer:
    def __init__(self, batch_size=1024, max_size=4096):
        self.batch_size = batch_size
        self.max_size = max_size
        self.tensors = None

    def __iter__(self):
        return self

    def __len__(self):
        return self.tensors[0].shape[0]

    def add(self, batch):
        if self.tensors is None:
            self.tensors = list(batch)
            return

        for i, (b, t) in enumerate(zip(batch, self.tensors)):
            self.tensors[i] = torch.cat((b, t), 0)
            self.tensors[i] = self.tensors[i][: self.max_size]

    def __next__(self):
        perm = torch.randperm(len(self))
        idx = perm[: self.batch_size]
        return [t[idx] for t in self.tensors]

    def __bool__(self):
        if self.tensors is None:
            return False

        return True


def torch_choice(n, k):
    return torch.lgamma(n + 1).exp() / (
        torch.lgamma((n - k) + 1).exp() * torch.lgamma(k + 1).exp()
    )


def random_probability_vectors(b, n, m=int(1e6)):
    rand_nums = np.sort(np.random.choice(m - 1, (b, n - 1), replace=False) + 1)
    left = np.concatenate((np.zeros((b, 1)), rand_nums), -1)
    right = np.concatenate((rand_nums, np.full((b, 1), m)), -1)
    result = (right - left) / m
    return result


class RLPLModel(ABC):
    def __init__(
        self,
        uda_weight=8.0,
        uda_warmup=5000,
        uda_wait=0,
        uda_threshold=0.6,
        uda_temp=1.0,
        acast_type=torch.float32,
        rl_data=None,
        pl_temperature=1.0,
        clip=1e9,
        supervised_loss=True,
        pl_max_batch_size=64,
    ):
        # TODO: organize and document these attributes.
        self.supervised_loss = supervised_loss
        self.pl_max_batch_size = pl_max_batch_size

        self.uda_step = 0
        self.current_step = 0
        self.uda_wait = uda_wait
        self.uda_warmup = uda_warmup + uda_wait
        self.uda_weight = uda_weight
        self.uda_threshold = uda_threshold
        self.uda_temp = uda_temp
        self.uda_crit = torch.nn.KLDivLoss(reduction="batchmean").to(
            torch.cuda.current_device()
        )

        self.gradscaler_actor = torch.amp.GradScaler("cuda")
        self.gradscaler_critic = torch.amp.GradScaler("cuda")
        self.gradscaler_estimator = torch.amp.GradScaler("cuda")

        self.acast_type = acast_type

        self.productive_updates = 0

        self.estimator = None
        self.rl_data = rl_data
        self.pl_temperature = pl_temperature
        self.clip = clip

    def UDA(self, actor_unlabeled_logits, actor_augmented_logits):
        self.uda_step += 1

        if self.uda_step < self.uda_wait:
            return 0

        soft_pseudo_label = torch.softmax(
            actor_unlabeled_logits.detach() / self.pl_temperature, dim=-1
        )
        max_probs, _ = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(self.uda_threshold).float()

        uda_loss = torch.mean(
            -(
                soft_pseudo_label * torch.log_softmax(actor_augmented_logits, dim=-1)
            ).sum(dim=-1)
            * mask
        )

        return (
            uda_loss
            * min(1, self.uda_step / (self.uda_warmup - self.uda_wait))
            * self.uda_weight
        )

    def environment_step(self, U, X, y, a):
        with torch.no_grad():
            loss_initial = (
                self.loss(
                    self.critic(X.to(torch.cuda.current_device()), deterministic=True),
                    y.to(torch.cuda.current_device()),
                )
                .detach()
                .clone()
            )
        with torch.autocast(device_type="cuda", dtype=self.acast_type):
            critic_unlabeled_logits = self.critic(U, deterministic=True)
            tmp_loss = self.loss(
                critic_unlabeled_logits, a.to(torch.cuda.current_device())
            )

        self.scaled_clipped_gradient_update(
            self.critic, tmp_loss, self.critic_opt, self.gradscaler_estimator
        )

        # now we can compute the final loss of the value network on the labeled data
        with torch.no_grad():
            loss_final = (
                self.loss(
                    self.critic(X.to(torch.cuda.current_device()), deterministic=True),
                    y.to(torch.cuda.current_device()),
                )
                .detach()
                .clone()
            )

        h = (loss_initial.mean() - loss_final.mean()).to(torch.cuda.current_device())


        return h

    def set_estimator(self, estimator):
        self.estimator = estimator

    def step(self, I, u, u_aug, x, y):
        self.current_step += 1
        if self.estimator is None:
            self.set_estimator(NaiveEstimator(self))
        return self.rlpl(I, u, u_aug, x, y)

    def scaled_clipped_gradient_update(self, model, loss, opt, scaler):
        loss.backward()
        opt.step()
        opt.zero_grad()
        return
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    def forward(self, x):
        with torch.no_grad():
            return self.critic(x.to(torch.cuda.current_device()))


class PPOPLModel(torch.nn.Module, RLPLModel):
    def __init__(
        self,
        backbone,
        predictor,
        predictor_optimizer,
        value,
        value_optimizer,
        loss=torch.nn.CrossEntropyLoss(),
        est_steps=1,
        use_shap=True,
        batch_size=1.0,
        use_uda=True,
        reset_weights=False,
        ema_policy=0.0,
        ema_value=0.0,
        value_network_wait=5000,
        value_network_warmup=5000,
        epsilon=0.0,
        gamma=0.0,
        target_kl=0.01,
        clip=0.2,
        maxit=16,
        **kwargs,
    ):
        super().__init__()
        RLPLModel.__init__(self, **kwargs)

        self.backbone = backbone
        self.predictor = predictor
        self.critic = CriticModel(backbone, predictor)
        self.critic_opt = predictor_optimizer

        # arguments associated with
        self.loss = loss
        self.use_uda = use_uda

        # arguments associated with reward approximation
        self.use_shap = use_shap
        self.est_steps = est_steps
        self.reset_weights = reset_weights
        self.batch_size = batch_size

        self.moving_dot_product = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

        self.moving_mean = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )
        self.moving_square = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

        # arguments associated with warmup and learning scheduling
        self.value_network_wait = value_network_wait
        self.value_network_warmup = self.value_network_wait + value_network_warmup

        self.buffer = ReplayBuffer(self.pl_max_batch_size, 2**12)
        self.target_kl = target_kl
        self.clip = clip
        self.maxit = maxit

    def rlpl(self, I, U, U_aug, L, y):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.acast_type): 
                # U = U[:self.pl_max_batch_size]
                # U = torch.repeat_interleave(U, 16, 0)
                _, _ = self.critic(torch.cat((L, U,)), log_prob=True)

                actor_logits, log_probs_initial = self.critic(torch.cat((L, U)), log_prob=True)
                log_probs_initial = log_probs_initial[L.shape[0] :]

                actor_labeled_logits = actor_logits[: L.shape[0]]
                actor_unlabeled_logits = actor_logits[L.shape[0] :]
                del actor_logits
                # log_probs_initial = torch.log_softmax(
                #     actor_unlabeled_logits.detach() / self.pl_temperature, dim=-1
                # )
                # _, a = torch.max(log_probs_initial, dim=-1)  
                a = torch.softmax(actor_unlabeled_logits, dim=-1).detach()

        self.critic.apply(freeze_batchnorm)
        a, h = self.estimator(I, a, U, L, y)
        self.critic.apply(unfreeze_batchnorm)
        print(torch.histogram(h.cpu(), 10, range=(-1, 1))[0])
        # self.ema_critic.update_parameters(self.critic)

        kl, it = torch.tensor(0), 0

        while it < self.maxit: 
            with torch.autocast(device_type="cuda", dtype=self.acast_type):
                if it > 0:
                    self.critic.apply(freeze_batchnorm)
                    aul, dist = self.critic._distribution(U)
                    self.critic.apply(unfreeze_batchnorm)
                    log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                    kl = (log_probs_initial - log_probs).mean().item()
                    print(it, kl)
                    if kl > (1.5 * self.target_kl):
                        break
                else:
                    self.critic.apply(freeze_batchnorm)
                    aul, dist = self.critic._distribution(U)
                    self.critic.apply(unfreeze_batchnorm)
                    log_probs = dist.log_prob(actor_unlabeled_logits).sum(-1)

                    kl = (log_probs_initial - log_probs).mean().item()
                    print(it, kl)

                # we can compute the log probability of all of these logits to estimate the KL divergence if we would like.
                # the estimate would be better, but it would be more expensive to compare.
                self.critic.apply(freeze_batchnorm)
                aal, _ = self.critic._distribution(U_aug)
                self.critic.apply(unfreeze_batchnorm)

                assert log_probs.shape == log_probs_initial.shape, (log_probs.shape, log_probs_initial.shape)

                ratio = torch.exp(
                    log_probs - log_probs_initial
                )

                log_probs_final = log_probs.detach().clone()

                assert h.shape == ratio.shape, (h.shape, ratio.shape)

                student = torch.nn.CrossEntropyLoss(reduction='none')(aul, a.detach().clone())
                entropy = -(torch.log_softmax(aul, -1) * torch.nn.functional.softmax(aul, -1)).sum(-1)

                clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * (h * student)

            # it is good when h is positive and the ratio is higher than 1.
            # The ratio is in the range [inf, 0] it can only increase by increasing the probability of predicting the action.
            # so increasing the ratio for a positive h means decreasing the negative product of those things.
            loss_pi = -(torch.min(ratio * (h * student), clip_adv)).mean() # - (entropy.mean() * 1e-2)

            # uda_loss = self.UDA(
            #     aul, aal
            # )

            with torch.autocast(device_type="cuda", dtype=self.acast_type):
                self.critic.apply(freeze_batchnorm)
                output = self.critic(L, deterministic=True)
                self.critic.apply(unfreeze_batchnorm)
                loss_sup = self.loss(output, y)

            total_loss = loss_pi + loss_sup # + uda_loss

            self.scaled_clipped_gradient_update(
                self.critic,
                total_loss,
                self.critic_opt,
                self.gradscaler_critic,
            )

            it += 1

        wandb.log(
            {"KL": kl, "ratio": ratio.mean().item(), 'value_iters': it, 'rl_loss': loss_pi}
        )

        return torch.tensor(0), torch.tensor(0)

    def forward(self, x):
        with torch.no_grad():
            return self.critic(x.to(torch.cuda.current_device()), deterministic=True)


class PGPLModel(torch.nn.Module, RLPLModel):
    def __init__(
        self,
        critic,
        critic_opt,
        actor,
        actor_opt,
        loss=torch.nn.CrossEntropyLoss(),
        est_steps=1,
        use_shap=True,
        batch_size=1.0,
        use_uda=True,
        reset_weights=False,
        **kwargs,
    ):
        super().__init__()
        RLPLModel.__init__(self, **kwargs)
        """

        U is the source of data that gives us our trajectories.  It makes sense to provide it separately as a data loader
        we will want to change elements of this loader, add a sampler, add augmentations, adjust the number of workers and
        change other elements of the data so that it can provide different kinds of observations for us to optimize over

        """
        self.critic = critic
        self.critic_opt = critic_opt
        self.ema_critic = ModelEMA(critic, 0.995)  # TODO: make this an argument

        # we cannoot just infer what the actor is supposed to be in the case models are wrapped for distributed training
        self.actor = actor
        self.actor_opt = actor_opt

        # arguments associated with
        self.loss = loss
        self.use_uda = use_uda

        # arguments associated with reward approximation
        self.use_shap = use_shap
        self.est_steps = est_steps
        self.reset_weights = reset_weights
        self.batch_size = batch_size

        self.moving_dot_product = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

    def rlpl(self, I, U, U_aug, L, y):
        # first, we estimate the individual rewards for the policy by sampling some actions and observing rewards.
        with torch.autocast(device_type="cuda", dtype=self.acast_type):
            if self.use_uda:
                # there may be some cases in which we do not want to do this consistency regularization (such as in the case when we are using self-supervised representations)
                actor_logits = self.actor(torch.cat((L, U, U_aug)))
                actor_labeled_logits = actor_logits[: L.shape[0]]
                actor_unlabeled_logits, actor_augmented_logits = actor_logits[
                    L.shape[0] :
                ].chunk(2)
                del actor_logits
            else:
                actor_logits = self.actor(torch.cat((L, U)))
                actor_labeled_logits = actor_logits[: L.shape[0]]
                actor_unlabeled_logits = actor_logits[L.shape[0] :]
                actor_augmented_logits = actor_unlabeled_logits
                del actor_logits

            soft_pseudo_label = torch.softmax(
                actor_unlabeled_logits.detach() / self.pl_temperature, dim=-1
            )
            _, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)

        if self.rl_data is not None:
            rl_x, rl_y = next(self.rl_data)
            a, h = self.estimator(I, hard_pseudo_label, U_aug, rl_x, rl_y)
        else:
            a, h = self.estimator(I, hard_pseudo_label, U_aug, L, y)

        with torch.autocast(device_type="cuda", dtype=self.acast_type):
            # we can choose to recompute the forward pass of the actor so as to avoid storing multiple gradients.
            _, hard_pseudo_label = torch.max(actor_augmented_logits.detach(), dim=-1)

            log_probs = -torch.nn.functional.log_softmax(
                actor_augmented_logits, dim=-1
            )[torch.arange(a.shape[0]), a]

            # l1 = torch.nn.functional.cross_entropy(actor_augmented_logits.detach(), hard_pseudo_label.detach())
            # l2 = log_probs.detach().mean()

            # assert torch.allclose(l1, l2, atol=1e-4), (l1 - l2)

            uda_loss = 0

            actor_loss = (log_probs * h).mean()

            if self.use_uda:
                uda_loss += self.UDA(actor_unlabeled_logits, actor_augmented_logits)

            if self.supervised_loss:
                uda_loss += self.loss(actor_labeled_logits, y)

            total_loss = actor_loss + uda_loss

        self.scaled_clipped_gradient_update(
            self.actor, total_loss, self.actor_opt, self.gradscaler_actor
        )

        self.ema_critic.update_parameters(self.actor)

        return h.mean()

    def forward(self, x):
        with torch.no_grad():
            return self.ema_critic(x.to(torch.cuda.current_device()))


class DPGPLModel(torch.nn.Module, RLPLModel):
    def __init__(
        self,
        backbone,
        predictor,
        predictor_optimizer,
        value,
        value_optimizer,
        loss=torch.nn.CrossEntropyLoss(),
        est_steps=1,
        use_shap=True,
        batch_size=1.0,
        use_uda=True,
        reset_weights=False,
        epsilon=0.0,
        gamma=0.0,
        ema_policy=0.999,
        ema_value=0.99,
        value_network_wait=5000,
        value_network_warmup=5000,
        **kwargs,
    ):
        super().__init__()
        RLPLModel.__init__(self, **kwargs)
        """

        U is the source of data that gives us our trajectories.  It makes sense to provide it separately as a data loader
        we will want to change elements of this loader, add a sampler, add augmentations, adjust the number of workers and
        change other elements of the data so that it can provide different kinds of observations for us to optimize over

        """
        self.backbone = backbone
        self.predictor = predictor
        self.critic = CriticModel(backbone, predictor)
        self.critic_opt = predictor_optimizer
        self.ema_backbone = ModelEMA(
            self.backbone, ema_policy
        )  # TODO: make this an argument

        # we cannoot just infer what the actor is supposed to be in the case models are wrapped for distributed training
        self.actor = value
        self.actor_opt = value_optimizer
        self.ema_actor = ModelEMA(self.actor, ema_value)  # TODO: make this an argument

        # arguments associated with
        self.loss = loss
        self.use_uda = use_uda

        # arguments associated with reward approximation
        self.use_shap = use_shap
        self.est_steps = est_steps
        self.reset_weights = reset_weights
        self.batch_size = batch_size
        self.epsilon = epsilon  # mixing coefficient for random noise
        self.gamma = (
            gamma  # probability of mixing a soft pseudo label with random noise
        )

        self.moving_dot_product = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

        self.moving_mean = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )
        self.moving_square = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

        # arguments associated with warmup and learning scheduling
        self.value_network_wait = value_network_wait
        self.value_network_warmup = self.value_network_wait + value_network_warmup

        self.buffer = ReplayBuffer(self.pl_max_batch_size, 2**12)

    def rlpl(self, I, U, U_aug, L, y):
        # this version of RLPL requires a special model that has multiple heads, each of which need to be optimized separately.
        # first, we estimate the individual rewards for the policy by sampling some actions and observing rewards.
        with torch.autocast(device_type="cuda", dtype=self.acast_type):
            ### BACKBONE NETWORK ###
            logits = self.backbone(torch.cat((L, U)))
            labeled_logits = logits[: L.shape[0]]
            unlabeled_logits = logits[L.shape[0] :]
            augmented_logits = unlabeled_logits

            output = self.predictor(labeled_logits)
            loss_sup = self.loss(output, y)

            uda_loss = 0

            if self.current_step > self.value_network_warmup:
                a, log_prob = self.critic(U[-self.pl_max_batch_size :], log_prob=True)
                action_value = self.actor(
                            unlabeled_logits[-self.pl_max_batch_size :].detach(),
                            torch.log_softmax(
                                self.critic(U[-self.pl_max_batch_size :]), dim=-1
                            ),
                )
                value_loss = - action_value.mean()

            self.buffer.add((U, self.predictor(unlabeled_logits).detach()))
            print(len(self.buffer))

        if self.current_step > self.value_network_warmup:

            rl_x, rl_y = self.rl_data.tensors
            with torch.no_grad():
                initial_loss_rl_data = (
                    self.loss(
                        self.critic(rl_x.to(torch.cuda.current_device())),
                        rl_y.to(torch.cuda.current_device()),
                    )
                    .cpu()
                    .item()
                )
            
            self.scaled_clipped_gradient_update(
                self.critic,
                value_loss + uda_loss,
                self.critic_opt,
                self.gradscaler_critic,
            )

            with torch.no_grad():
                final_loss_rl_data = (
                    self.loss(
                        self.critic(rl_x.to(torch.cuda.current_device())),
                        rl_y.to(torch.cuda.current_device()),
                    )
                    .cpu()
                    .item()
                )

        ### PREDICTOR NETWORK ###

        i = 0
        ema_trivial = None
        ema_measured = None
        running_h_mean = 0
        running_h_var = 0

        shap_buffer = ReplayBuffer(64, 2**16)

        while True:
            i += 1
            # we have to call this the critic i think so that the estimator works correctly
            if self.current_step > self.value_network_wait:
                ### POLICY NETWORK ###
                if i <= 1:
                    with torch.no_grad():
                        # this could be faster but I don't want to mess with the backbone right now
                        if self.buffer:
                            U, a = next(self.buffer)
                        else:
                            assert False
                            U, a = (
                                U[-self.pl_max_batch_size :].detach(),
                                self.predictor(
                                    unlabeled_logits[-self.pl_max_batch_size :]
                                ).detach(),
                            )

                        rl_x, rl_y = next(self.rl_data)
                        U = torch.repeat_interleave(U, 16, 0)
                        a = torch.repeat_interleave(a, 16, 0)
                        # U[-128:] = rl_x[:128]

                        ema_logits = self.backbone(U)
                        a = self.predictor(ema_logits).detach()

                        # a[-128:] = torch.nn.functional.one_hot(rl_y[:128], a.shape[-1])

                        soft_pseudo_label = torch.softmax(a, dim=-1).detach()

                        """if self.epsilon > 0 and self.gamma > 0:
                            soft_pseudo_label_random = (
                                (1 - self.epsilon) * soft_pseudo_label
                            ) + torch.tensor(
                                random_probability_vectors(
                                    soft_pseudo_label.shape[0],
                                    soft_pseudo_label.shape[-1],
                                )
                                * self.epsilon
                            ).to(torch.cuda.current_device()).type(torch.float16)
                            mask = torch.rand((soft_pseudo_label_random.shape[0],))
                            mask = (
                                mask
                                <= torch.sort(mask)[0][
                                    min(
                                        int(mask.shape[0] * self.gamma),
                                        mask.shape[0] - 1,
                                    )
                                ]
                            )

                            soft_pseudo_label[mask] = soft_pseudo_label_random[mask]

                            # soft_pseudo_label = (torch.nn.functional.one_hot(torch.argmax(soft_pseudo_label, -1), soft_pseudo_label.shape[-1]) + 0.01) / 1.1"""

                    if self.rl_data is not None:
                        pl, h = self.estimator(
                            I.detach().clone(),
                            soft_pseudo_label.detach().clone(),
                            U.detach().clone(),
                            self.rl_data.tensors[0].detach().clone(),
                            self.rl_data.tensors[1].detach().clone(),
                        )
                    else:
                        assert False
                        a, h = self.estimator(
                            I,
                            soft_pseudo_label[: self.pl_max_batch_size],
                            U[: self.pl_max_batch_size],
                            L,
                            y,
                        )

                    good_mask = h > 0.0
                    bad_mask = (h == 0)

                    bad = h[h > 0.0].shape[0] // 10

                    shap_buffer.add(
                        (
                            torch.cat(
                                (
                                    ema_logits[good_mask],
                                    ema_logits[bad_mask],
                                ),
                                0,
                            ).detach(),
                            torch.cat(
                                (
                                    soft_pseudo_label[good_mask],
                                    soft_pseudo_label[bad_mask],
                                ),
                                0,
                            ).detach(),
                            torch.cat(
                                (
                                    h[good_mask],
                                    h[bad_mask],
                                ),
                                0,
                            ).detach(),
                        )
                    )

                    for tensor in shap_buffer.tensors:
                        assert tensor.shape[0] == shap_buffer.tensors[0].shape[0]
                    
                    continue

                ema_logits, soft_pseudo_label, h = shap_buffer.tensors

                running_h_mean = (h.mean() * (1 / i)) + (((i - 1) / i) * running_h_mean)
                running_h_var = (torch.var(h) * (1 / i)) + (
                    ((i - 1) / i) * running_h_var
                )

                # h = (h - running_h_mean) / torch.sqrt(running_h_var)
                # h *= 10

                # h = ((h > h.mean()).type(torch.float32) * 2) - 1

                assert not torch.isnan(h).any()

                ### VALUE NETWORK ###

                with torch.autocast(device_type="cuda", dtype=self.acast_type):
                    # the goal of the value network is different here, we are just predicting h.  SHAP is incompatible with the value estimation network.
                    actor_input = (
                            ema_logits.detach().clone(),
                            torch.log(soft_pseudo_label.detach().clone()),
                    )

                    predicted_value = self.actor(*actor_input)

                    assert h.squeeze().shape == predicted_value.squeeze().shape, (
                        h.squeeze().shape,
                        predicted_value.squeeze().shape,
                    )
                    # actor_loss = ((h.squeeze() - predicted_value.squeeze()) ** 2).mean()

                    actor_loss = -(
                        (h.squeeze() * torch.log(predicted_value.squeeze()))
                        + ((1 - h.squeeze()) * torch.log(1 - predicted_value.squeeze()))
                    ).mean()

                self.scaled_clipped_gradient_update(
                    self.actor, actor_loss, self.actor_opt, self.gradscaler_actor
                )
                # actor_scheduler.step()

                self.ema_actor.update_parameters(self.actor)

                with torch.no_grad():
                    actor_loss = -(
                            (h.squeeze() * torch.log(self.actor(*actor_input).squeeze()))
                            + ((1 - h.squeeze()) * torch.log(1 - predicted_value.squeeze()))
                        ).mean()

                trivial_loss = (
                    -(
                        (h.squeeze() * torch.log(h.mean()))
                        + ((1 - h.squeeze()) * torch.log(1 - torch.exp(torch.log(h).mean())))
                    )
                    .detach()
                    .clone()
                    .mean()
                    .cpu()
                    .item()
                )

                # trivial_loss = ((h - h.mean()) ** 2).detach().clone().mean().cpu().item()
                measured_loss = actor_loss.detach().clone().mean().cpu().item()

                if ema_measured is None:
                    ema_measured = measured_loss
                    ema_trivial = trivial_loss

                ema_measured = (0.99 * ema_measured) + (0.01 * measured_loss)
                ema_trivial = (0.99 * ema_trivial) + (0.01 * trivial_loss)

                # print(i, round(ema_measured - (ema_trivial * 0.01), 5), round(ema_measured, 4), round(ema_trivial, 4),  end='\r')
                # wandb.log({'ema_measured': ema_measured, 'ema_trivial' :ema_trivial})

                if (ema_measured < ema_trivial) or (i >= 10 and self.current_step > 2):
                    wandb.log(
                        {"ema_measured": measured_loss, "ema_trivial": trivial_loss, 'value_iters': i, 'rl_loss': final_loss_rl_data}
                    )
                    print(i)
                    break

        if self.current_step > self.value_network_wait:
            print(torch.histogram(shap_buffer.tensors[-1].cpu(), 10, range=(0, 1))[0])
            return  torch.tensor(measured_loss), torch.tensor(trivial_loss)
        else:
            return torch.tensor(0), torch.tensor(0)

    def forward(self, x):
        with torch.no_grad():
            return self.critic(x.to(torch.cuda.current_device()), deterministic=True)


class SupervisedModel(torch.nn.Module, RLPLModel):
    def __init__(
        self,
        critic,
        critic_opt,
        actor,
        actor_opt,
        loss=torch.nn.CrossEntropyLoss(),
        est_steps=1,
        use_shap=True,
        batch_size=1.0,
        use_uda=True,
        reset_weights=False,
        **kwargs,
    ):
        super().__init__()
        RLPLModel.__init__(self, **kwargs)
        """

        U is the source of data that gives us our trajectories.  It makes sense to provide it separately as a data loader
        we will want to change elements of this loader, add a sampler, add augmentations, adjust the number of workers and
        change other elements of the data so that it can provide different kinds of observations for us to optimize over

        """
        self.critic = critic
        self.critic_opt = critic_opt
        self.ema_critic = ModelEMA(critic, 0.995)

        # we cannoot just infer what the actor is supposed to be in the case models are wrapped for distributed training
        self.actor = actor
        self.actor_opt = actor_opt

        # arguments associated with
        self.loss = loss
        self.use_uda = use_uda

        # arguments associated with reward approximation
        self.use_shap = use_shap
        self.est_steps = est_steps
        self.reset_weights = reset_weights
        self.batch_size = batch_size

        self.moving_dot_product = torch.nn.Parameter(
            data=torch.zeros((1,)), requires_grad=False
        )

    def rlpl(self, I, U, U_aug, L, y):
        # first, we estimate the individual rewards for the policy by sampling some actions and observing rewards.
        with torch.autocast(device_type="cuda", dtype=self.acast_type):
            if self.use_uda:
                # there may be some cases in which we do not want to do this consistency regularization (such as in the case when we are using self-supervised representations)
                actor_logits = self.actor(torch.cat((L, U, U_aug)))
                actor_labeled_logits = actor_logits[: L.shape[0]]
                actor_unlabeled_logits, actor_augmented_logits = actor_logits[
                    L.shape[0] :
                ].chunk(2)
                del actor_logits
            else:
                actor_logits = self.actor(torch.cat((L, U)))
                actor_labeled_logits = actor_logits[: L.shape[0]]
                actor_unlabeled_logits = actor_logits[L.shape[0] :]
                actor_augmented_logits = actor_unlabeled_logits
                del actor_logits

            soft_pseudo_label = torch.softmax(
                actor_unlabeled_logits.detach() / self.pl_temperature, dim=-1
            )
            _, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)

            if self.use_uda:
                uda_loss = self.UDA(
                    actor_unlabeled_logits, actor_augmented_logits
                ) + self.loss(actor_labeled_logits, y)
            elif self.supervised_loss:
                uda_loss = self.loss(actor_labeled_logits, y)
            else:
                uda_loss = 0
        self.ema_critic.update_parameters(self.critic)

        a, h = self.estimator(I, hard_pseudo_label, U_aug, L, y)

        total_loss = uda_loss

        self.scaled_clipped_gradient_update(
            self.actor, total_loss, self.actor_opt, self.gradscaler_actor
        )

        return h.mean()

    def forward(self, x):
        with torch.no_grad():
            return self.ema_critic(x.to(torch.cuda.current_device()))


class UDAModel(torch.nn.Module):
    def __init__(self, model, opt, **kwargs):
        super().__init__()
        self.model = model
        self.actor_opt = opt
        self.uda_step = 0
        self.uda_steps = 20000
        self.loss = None

    def step(self, I, U, U_aug, L, y):
        self.actor_opt.zero_grad()

        assert not torch.allclose(U, U_aug)

        self.uda_step += 1
        with torch.no_grad():
            output_unsup = self.model(U)
        output_unsup_aug = self.model(U_aug)

        # import ipdb;ipdb.set_trace()
        loss_unsup = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(output_unsup_aug, dim=1),
            torch.nn.functional.softmax(output_unsup, dim=1).detach(),
            reduction="batchmean",
        )  # * self.uda_weight

        loss_sup = torch.nn.functional.cross_entropy(self.model(L), y, reduction="mean")

        # loss_unsup = self.UDA(U, U_aug)
        total_loss = loss_sup + loss_unsup  # + loss_l1
        total_loss.backward()
        self.actor_opt.step()

        return torch.tensor(0)

    def forward(self, x):
        self.model.eval()
        y = self.model(x)
        self.model.train()
        return y
