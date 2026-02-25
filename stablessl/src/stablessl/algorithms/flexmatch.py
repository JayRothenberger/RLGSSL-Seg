from stablessl.algorithms.base import AlgorithmBase
import torch
from collections import Counter
from copy import deepcopy
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm


class FlexMatchAlgorithm(AlgorithmBase):
    def __init__(self, agent, args, model_fn, optimizer_fn, scheduler_fn, dataset, pl_cutoff=0.95):
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

        self.selected_labels = None
        self.class_frequencies = None
        self.pl_cutoff = pl_cutoff


    @torch.no_grad()
    def update_frequencies(self):
        pseudo_counter = Counter(self.selected_labels.tolist())
        max_freq = max(pseudo_counter.values())
        if max_freq < self.args.unlabeled_batch_size:
            assert -1 not in pseudo_counter.keys()
            for i in range(self.num_classes):
                self.class_frequencies[i] = ((pseudo_counter[i] / max_freq) * 0.01) + (self.class_frequencies * 0.99)


    def step(self):
        loss_fn = torch.nn.CrossEntropyLoss()

        predictions = []
        ground_truth = []

        for i in range(self.accumulate):

            U, U_aug, _ = next(self.dataset.iterable_loaders["unlabeled"])
            X, X_aug, y = next(self.dataset.iterable_loaders["labeled"])

            if self.selected_labels is None or self.class_frequencies is None:
                self.class_frequencies = torch.ones((U.shape[0], ), device=torch.cuda.current_device(), dtype=torch.float32)
                self.class_frequencies /= self.class_frequencies.sum()

            unlabeled_instances = torch.cat((U, U_aug), 0)

            X, unlabeled_instances = X.to(torch.cuda.current_device()), unlabeled_instances.to(torch.cuda.current_device())
            y = y.to(torch.cuda.current_device())
            unfreeze_batchnorm(self.model)
            with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
            # compute the logits for unlabeled augmented and unaugmented instances
                with torch.autocast(device_type="cuda", dtype=self.args.autocast_type): 
                    unlabeled_logits = self.model(unlabeled_instances)

                    probs = torch.softmax(unlabeled_logits, dim=-1).detach()
                    weak, strong = probs.chunk(2)

                    max_probs, idx = weak.max(-1)

                    mask = max_probs.ge(self.pl_cutoff * (self.class_frequencies[idx] / (2. - self.class_frequencies[idx])))

                    self.selected_labels = idx[max_probs > self.pl_cutoff]
                
                # compute labeled logits
                freeze_batchnorm(self.model)
                labeled_logits = self.model(X)
                predictions.append(labeled_logits)
                ground_truth.append(y)

                # labeled loss with CE between mixed labeled instance logits and mixed labels
                supervised_loss = loss_fn(labeled_logits, y)
                # CE unlabeled consistency loss
                if mask.any():
                    consistency_loss = loss_fn(unlabeled_logits[U.shape[0]:][mask], weak.argmax(-1)[mask])
                else:
                    consistency_loss = 0

                self.scaled_clipped_gradient_update(
                        self.model,
                        supervised_loss + consistency_loss,
                        self.optimizer,
                        self.scaler,
                        i < (self.accumulate - 1)
                    )
        
        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    
    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)