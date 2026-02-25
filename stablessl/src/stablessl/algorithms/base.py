from abc import ABC
import torch
import os

def freeze_norm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.eval()

def unfreeze_norm(m):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
        m.train()

class AlgorithmBase(ABC, torch.nn.Module):
    def __init__(self, agent, args, dataset, iteration_start_callbacks, epoch_start_callbacks, update_start_callbacks, iteration_end_callbacks,
                 epoch_end_callbacks, update_end_callbacks):
        super().__init__()
        # common arguments
        self.agent = agent
        self.args = args
        self.num_classes = args.num_classes

        self.epochs = args.epochs
        self.clip = args.clip
        self.checkpoint_path = agent.path
        self.load_path = vars(args).get('load_path')
        self.resume = vars(args).get('resume')
        self.accumulate = args.accumulate

        # common optimization parameters (optimizers and models are defined on a per-algorithm basis)
        self.it = 0
        self.start_epoch = 0
        self.best_eval_acc, self.best_it = 0.0, 0

        # build dataset
        self.dataset = dataset

        # set callback on initialization
        self.iteration_start_callbacks = iteration_start_callbacks
        self.epoch_start_callbacks = epoch_start_callbacks
        self.update_start_callbacks = update_start_callbacks

        self.iteration_end_callbacks = iteration_end_callbacks
        self.epoch_end_callbacks = epoch_end_callbacks
        self.update_end_callbacks = update_end_callbacks

    def scaled_clipped_gradient_update(self, model, loss, opt, scaler, accumulate=False):
        scaler.scale(loss).backward()

        if not accumulate:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

    def iteration_start_callback(self):
        for callback in self.iteration_start_callbacks:
            callback()

    def epoch_start_callback(self):
        for callback in self.epoch_start_callbacks:
            callback()

    def update_start_callback(self):
        for callback in self.update_start_callbacks:
            callback()

    def iteration_end_callback(self):
        for callback in self.iteration_end_callbacks:
            callback()

    def epoch_end_callback(self):
        for callback in self.epoch_end_callbacks:
            callback()

    def update_end_callback(self):
        for callback in self.update_end_callbacks:
            callback()

    def forward(self, x):
        raise NotImplementedError("")
    
    def step(self, args, dataset):
        raise NotImplementedError("")
    
    def set_dataset(self, dataset):
        self.dataset = dataset

    def checkpoint_state(self):
        torch.save(self.state_dict(), self.agent.path + f'_{os.environ["RANK"]}')

    def load_checkpoint_state(self):
        with open(self.agent.path + f'_{os.environ["RANK"]}', "rb") as fp:
            state_dict = torch.load(fp)
        
        self.load_state_dict(state_dict)