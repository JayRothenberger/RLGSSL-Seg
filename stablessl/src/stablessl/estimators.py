import torch
from abc import ABC
import math
import warnings
import os
import gc
import copy
import numpy as np


def torch_choice(n, k):
    return torch.lgamma(n + 1).exp() / (
        torch.lgamma((n - k) + 1).exp() * torch.lgamma(k + 1).exp()
    )


def shap_approx(A, B):
    # but the weights are all the same if we are using the same batch size for each batch
    # W = (A.shape[-1] - 1) / (torch_choice(A.shape[-1], A.sum(-1)) * A.sum(-1) * (A.shape[-1] - A.sum(-1)))
    # so then we can use the torch lstsq method instead of writing our own.
    A, B = A.to(torch.cuda.current_device()), B.to(torch.cuda.current_device())
    X = torch.linalg.lstsq(A, B).solution
    # X = torch.linalg.pinv(A.T @ A) @ A.T @ B

    if os.environ.get("WORLD_SIZE") is not None:
        X = X.to(torch.cuda.current_device())
        torch.distributed.broadcast(X, 0)

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

class RewardEstimator(ABC):
    def __init__(self, rl_process, update=False):
        self.rl_process = rl_process
        self.step = 0
        self.batch_size = 1
        self.binarize_rewards = False
        self.update = update

    def __call__(self, U, X, y):
        self.step += 1

        if 0 < self.batch_size <= 1:
            warnings.warn(
                f"interpreting batch size for training as fraction of unlabeled batch size {self.batch_size * 100}% of {U.shape[0]} is {math.ceil(self.batch_size * U.shape[0])}"
            )
            self.batch_size = math.ceil(self.batch_size * U.shape[0])

    def epsilon(self):
        return 1 - (0.985 ** (self.step**0.5))

    def sample_actions(self, U):
        with torch.autocast(device_type="cuda", dtype=self.rl_process.acast_type):
            pi = self.rl_process.actor(U.to(torch.cuda.current_device())) / 0.7

        # then we sample from that distribution (this could theoretically occur multiple times, probably shouldn't)

        a = torch.tensor(
            [
                np.random.choice(
                    np.arange(pi.shape[-1]),
                    None,
                    False,
                    torch.nn.Softmax(-1)(
                        torch.clamp(
                            torch.nan_to_num(pi.type(torch.float32)), -(2**14), 2**14
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()[xi],
                )
                for xi in range(pi.shape[0])
            ]
        ).to(torch.cuda.current_device())

        amax = torch.argmax(pi.detach().clone(), -1).to(torch.cuda.current_device())

        # with some prob select a random action
        a_mask = torch.rand((amax.shape[0],))
        a_mask = (
            a_mask
            <= torch.sort(a_mask, -1)[0][..., int(amax.shape[0] * self.epsilon())]
        )
        # a_mask = a_mask <= torch.sort(a_mask, -1)[0][..., int(amax.shape[0] * 0.95)]

        a = torch.where(
            a_mask.to(torch.cuda.current_device()),
            amax,
            torch.randint(low=0, high=pi.shape[-1], size=amax.shape).to(
                torch.cuda.current_device()
            ),
        )

        return pi, a, amax

    def checkpoint(self):
        self.initial_state = detach_tensors(copy.deepcopy(self.rl_process.critic.state_dict()))
        self.initial_opt_state = detach_tensors(copy.deepcopy(self.rl_process.critic_opt.state_dict().copy()))
        self.initial_scaler_state = detach_tensors(copy.deepcopy(self.rl_process.gradscaler_critic.state_dict().copy()))

        #for name, tensor in self.initial_state.items():
        #    self.initial_state[name] = tensor.detach().clone()

        #torch.save({'critic_state': self.initial_state, 'opt_state': self.initial_opt_state, 'scaler_state': self.initial_scaler_state}, 'states4.pt')

    def load_checkpoint(self):
        #states = torch.load('states4.pt')
        self.rl_process.critic.load_state_dict(self.initial_state)
        self.rl_process.critic_opt.load_state_dict(self.initial_opt_state)
        self.rl_process.gradscaler_critic.load_state_dict(self.initial_scaler_state)

    def free_checkpoint(self):
        self.initial_state = None
        self.initial_opt_state = None
        gc.collect()


class NaiveEstimator(RewardEstimator):
    def __call__(self, I, a, U, X, y):
        super().__call__(U, X, y)
        # first we get the action distribution
        # TODO: sampling the actions should be a command line option
        # pi, a, amax = self.sample_actions(U)
        if not self.update:
            self.checkpoint()

        h = self.rl_process.environment_step(U, X, y, a)

        if not self.update:
            self.load_checkpoint()

        if os.environ.get("WORLD_SIZE") is not None:
            torch.distributed.all_reduce(h, torch.distributed.ReduceOp.AVG)

        # TODO: make this an option argument
        # self.moving_dot_product.data = (self.moving_dot_product * 0.99) + (h.mean() * 0.01)
        # h = h - self.moving_dot_product

        if self.binarize_rewards:
            return a, torch.sign(torch.ones((U.shape[0],)).to(torch.cuda.current_device()) * h.mean()).type(torch.float32)
        else:
            return a, torch.ones((U.shape[0],)).to(torch.cuda.current_device()) * h.mean()



class SHAPEstimator(RewardEstimator):
    def __init__(
        self,
        rl_process,
        est_iters=1,
        batch_size=0.5,
        online=True,
        online_lr=0.0001,
        online_decay=2e-2,
        binarize_rewards=False,
        **kwargs
    ):
        super().__init__(rl_process, **kwargs)
        self.rl_process = rl_process

        self.est_iters = est_iters
        self.batch_size = batch_size
        self.online = online
        self.binarize_rewards = binarize_rewards
        self.online_lr = online_lr
        self.online_decay = online_decay

        # if the estimation is to be made online then we need a structure to hold the values
        # this is not the best structure and if we knew the size of this array we would
        # this structure will not work when we do parallel runs we would need to reduce over the ranks for each index (TODO)
        self.online_shap_weights = None
        self.online_shap_flags = None

    def on_policy_shap(self, masks, hs):
        weights = shap_approx(masks.type(torch.float32), hs).squeeze()

        # weights = torch.nan_to_num(
        #     (
        #         masks.to(torch.cuda.current_device()).type(torch.float32)
        #         * weights.squeeze().unsqueeze(0)
        #     ).sum(0)
        #     / masks.to(torch.cuda.current_device()).type(torch.float32).sum(0)
        # )

        mu = weights.mean()

        # weights = (weights - mu) / weights.std()


        return weights # torch.clamp(weights / 2, -1, 1)

    def online_shap(self, a, masks, hs, inds, pi):
        assert False
        # let's select only the ones where
        input_actions = a.shape[0]
        non_zero_rewards = masks.type(torch.float32).sum(0) > 0
        # we will select only the positions for which we evaluated that action
        input_inds = inds
        inds = inds.cpu()[non_zero_rewards]
        a = a.cpu()[non_zero_rewards]
        masks = masks[..., non_zero_rewards]
        # if we have not yet initialized the structure with the weights we can do that here
        if self.online_shap_weights is None:
            self.online_shap_weights = torch.zeros(
                inds.max() + 1, pi.shape[-1], dtype=torch.float16
            )
            # we will use these to indicate whether or not a position has already been set with an initial value
            self.online_shap_flags = torch.zeros(
                inds.max() + 1, pi.shape[-1], dtype=torch.bool
            )
        elif inds.max() >= self.online_shap_weights.shape[0]:
            print(self.online_shap_weights.shape[0], inds.max())
            print(f"step {self.step} adding new rows to flags and weights...")
            self.online_shap_weights = torch.cat(
                (
                    self.online_shap_weights,
                    torch.zeros(
                        (inds.max() - self.online_shap_weights.shape[0]) + 1,
                        self.online_shap_weights.shape[-1],
                        dtype=torch.float16,
                    ),
                ),
                0,
            )
            self.online_shap_flags = torch.cat(
                (
                    self.online_shap_flags,
                    torch.zeros(
                        (inds.max() - self.online_shap_flags.shape[0]) + 1,
                        self.online_shap_flags.shape[-1],
                        dtype=torch.bool,
                    ),
                ),
                0,
            )

        # if there are weights which are not initialized it may negatively impact the weight updates so we will initialize them with the solution to a shap estimation
        # this should also speed up the initial estimation

        W = self.online_shap_weights[inds, a]

        # compute the predicted value of the batch from the shap values
        y = masks.type(torch.float16) @ W.unsqueeze(-1)

        # update the shap values with an iteration of gradient descent
        update = (
            self.online_lr * ((hs.cpu() - y.cpu()) * masks).mean(0).unsqueeze(-1)
        ).squeeze()
        # assert update.shape == W.shape, (update.shape, W.shape)

        # TODO: make this an argument option
        ema_weight = 0.01
        # ((W + update) * ema_weight) + (W * (1 - ema_weight))

        self.online_shap_weights -= self.online_decay * self.online_shap_weights
        self.online_shap_weights[inds, a] += update.cpu().type(torch.float16)

        # then we either do the EMA or a step of gradient decent to update the online estimate

        # finally we return the new estimate as the weights
        rax = torch.zeros((input_actions,)).to(torch.cuda.current_device())
        rax[non_zero_rewards] = (
            self.online_shap_weights[inds, a]
            .squeeze()
            .to(torch.cuda.current_device())
            .type(torch.float32)
        )

        # we can compute the action with the best reward reward.
        values, a = torch.max(self.online_shap_weights[inds], -1)

        return values.to(torch.cuda.current_device()).type(torch.float32), a.to(
            torch.cuda.current_device()
        )

    def __call__(self, I, a, U, X, y):
        super().__call__(U, X, y)
        # first we get the action distribution
        # TODO: make sampling actions a part of the RL algorithm
        # pi, a, amax = self.sample_actions(U)
        # then we need to generate some binary vectors that will serve as instance masks
        masks = torch.rand((self.est_iters, a.shape[0]))
        masks = masks <= torch.sort(masks, -1)[0][..., self.batch_size - 1].unsqueeze(
            -1
        )

        hs = []

        self.checkpoint()

        opt = self.rl_process.critic_opt
        self.rl_process.critic_opt = torch.optim.Adam(self.rl_process.critic.parameters(), lr=0.0001, betas=(1e-5, 1e-5))
        self.rl_process.critic.eval()

        for m, mask in enumerate(masks):
            h = self.rl_process.environment_step(U[mask], X, y, a[mask])

            if os.environ.get("WORLD_SIZE") is not None:
                torch.distributed.all_reduce(h, torch.distributed.ReduceOp.AVG)

            hs.append(h)

            self.load_checkpoint()
        else:
            # self.free_checkpoint()
            if self.update:
                raise NotImplementedError()
                self.rl_process.environment_step(U, X, y, amax)

        self.rl_process.critic_opt = opt

        self.load_checkpoint()

        hs = torch.tensor(hs).unsqueeze(-1).to(torch.cuda.current_device())

        if self.binarize_rewards:
            hs = torch.sign(hs)

        if self.online:
            raise NotImplementedError()
            weights, action = self.online_shap(a, masks, hs, I, pi)
        else:
            weights = self.on_policy_shap(masks, hs)

        # can take the sign of h here so that the task is to estimate helpful v.s. unhelpful rather than the reward magnitude which will drift.
        if os.environ.get("WORLD_SIZE") is not None:
            tensors = [
                masks.clone().to(torch.cuda.current_device())
                for _ in range(int(os.environ["WORLD_SIZE"]))
            ]
            torch.distributed.all_gather(tensors, masks.to(torch.cuda.current_device()))
            masks = torch.cat(tensors, -1)

        self.rl_process.critic.train()

        return a, weights

        if self.binarize_rewards:
            return a, torch.sign(weights).type(torch.float32)
        else:
            return a, weights


class SHAPBayes(RewardEstimator):
    def __init__(
        self,
        rl_process,
        est_iters=1,
        batch_size=0.5,
        online=True,
        online_lr=0.0001,
        online_decay=2e-2,
        binarize_rewards=False,
        **kwargs
    ):
        super().__init__(rl_process, **kwargs)
        self.rl_process = rl_process

        self.est_iters = est_iters
        self.batch_size = batch_size
        self.online = online
        self.binarize_rewards = binarize_rewards
        self.online_lr = online_lr
        self.online_decay = online_decay

        # if the estimation is to be made online then we need a structure to hold the values
        # this is not the best structure and if we knew the size of this array we would
        # this structure will not work when we do parallel runs we would need to reduce over the ranks for each index (TODO)
        self.online_shap_weights = None
        self.online_shap_flags = None

    def shap_bayes(self, U, a, masks, hs):
        hs, masks = hs.squeeze().to(torch.cuda.current_device()), masks.to(torch.cuda.current_device()).type(torch.float32)

        masks = masks * hs.unsqueeze(-1)

        intersection = masks[hs > 0]

        new = masks.sum(0) # (intersection.sum(0) / (masks.abs().sum(0) + 1)).squeeze()

        new = (new - new.mean()) / torch.std(new)

        # new[new < -1] = 0
        # new[(new > -1) * (new <= 3)] = (1 + new[(new > -1) * (new <= 3)]) / 4
        # new[new > 3] = 1

        return torch.clamp(new / 3, -1, 1)


    def __call__(self, I, a, U, X, y):
        super().__call__(U, X, y)
        # first we get the action distribution
        # TODO: make sampling actions a part of the RL algorithm
        # pi, a, amax = self.sample_actions(U)
        # then we need to generate some binary vectors that will serve as instance masks
        masks = torch.rand((self.est_iters, a.shape[0]))
        masks = masks <= torch.sort(masks, -1)[0][..., self.batch_size - 1].unsqueeze(
            -1
        )

        hs = []

        self.checkpoint()

        opt = copy.copy(self.rl_process.critic_opt)
        self.rl_process.critic_opt = torch.optim.Adam(self.rl_process.critic.parameters(), lr=1e-4, betas=(1e-7, 1e-7))

        for m, mask in enumerate(masks):
            h = self.rl_process.environment_step(U[mask], X, y, a[mask])

            if os.environ.get("WORLD_SIZE") is not None:
                torch.distributed.all_reduce(h, torch.distributed.ReduceOp.AVG)

            hs.append(h)

            self.load_checkpoint()
        else:
            # self.free_checkpoint()
            if self.update:
                raise NotImplementedError()
                self.rl_process.environment_step(U, X, y, amax)

        self.rl_process.critic_opt = opt

        hs = torch.tensor(hs).unsqueeze(-1).to(torch.cuda.current_device())

        if self.online:
            raise NotImplementedError()
            weights, action = self.online_shap(a, masks, hs, I, pi)
        else:
            weights = self.shap_bayes(U, a, masks, hs)

        # can take the sign of h here so that the task is to estimate helpful v.s. unhelpful rather than the reward magnitude which will drift.
        if os.environ.get("WORLD_SIZE") is not None:
            tensors = [
                masks.clone().to(torch.cuda.current_device())
                for _ in range(int(os.environ["WORLD_SIZE"]))
            ]
            torch.distributed.all_gather(tensors, masks.to(torch.cuda.current_device()))
            masks = torch.cat(tensors, -1)

        return a, weights

        if self.binarize_rewards:
            return a, torch.sign(weights).type(torch.float32)
        else:
            return a, weights
