from stablessl.algorithms.base import AlgorithmBase
import torch
import numpy as np
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm

# https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/utils/ops.py#L49
@torch.no_grad()
def mixup_one_target(x, y, alpha=0.5, is_bias=True):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


class MixMatchAlgorithm(AlgorithmBase):
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


    def step(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        predictions = []
        ground_truth = []

        for i in range(self.accumulate):

            U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
            X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])

            unlabeled_instances = torch.cat((U, U_aug), 0)

            X, unlabeled_instances = X.to(torch.cuda.current_device()), unlabeled_instances.to(torch.cuda.current_device())
            y = y.to(torch.cuda.current_device())
            freeze_batchnorm(self.model)
            with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
            # compute the logits for unlabeled augmented and unaugmented instances
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=self.args.autocast_type): 
                        unlabeled_logits = self.model(unlabeled_instances)

                        probs = torch.softmax(unlabeled_logits, dim=-1).detach()
                        weak, strong = probs.chunk(2)

                        avg_probs = (weak + strong) / 2 # average predicted probabilities

                        # sharpen the predicted average
                        sharpen_probs = avg_probs ** 2
                        sharpen_probs = (sharpen_probs / sharpen_probs.sum(dim=-1, keepdim=True))
                
                # compute labeled logits
                labeled_logits = self.model(X)
                predictions.append(labeled_logits)
                ground_truth.append(y)
                # mixup 
                one_hot_labels = torch.nn.functional.one_hot(y, sharpen_probs.shape[-1])
                all_instances = torch.cat((X, unlabeled_instances), 0)
                all_logits = torch.cat((one_hot_labels, sharpen_probs, sharpen_probs), 0)

                mixed_instances, mixed_targets, lam = mixup_one_target(all_instances, all_logits)
                # predict on the mixed up chunks
                mixed_labeled = mixed_instances[:labeled_logits.shape[0]]
                mixed_unlabeled = mixed_instances[labeled_logits.shape[0]:]

                # labeled loss with CE between mixed labeled instance logits and mixed labels
                unfreeze_batchnorm(self.model)
                supervised_loss = loss_fn(self.model(mixed_labeled), mixed_targets[:labeled_logits.shape[0]])
                # MSE unlabeled consistency loss
                freeze_batchnorm(self.model)
                consistency_loss = torch.nn.MSELoss()(self.model(mixed_unlabeled), mixed_targets[labeled_logits.shape[0]:])

                self.scaled_clipped_gradient_update(
                        self.model,
                        supervised_loss + 0.1 * consistency_loss,
                        self.optimizer,
                        self.scaler,
                        i < (self.accumulate - 1)
                    )
        
        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()
    
    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)