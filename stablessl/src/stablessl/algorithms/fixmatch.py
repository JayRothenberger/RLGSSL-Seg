from stablessl.algorithms.base import AlgorithmBase
import torch
from stablessl.train_utils import freeze_batchnorm, unfreeze_batchnorm

class FixMatchAlgorithm(AlgorithmBase):
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
        self.pl_cutoff = pl_cutoff


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
            unfreeze_batchnorm(self.model)
            with torch.autocast(device_type="cuda", dtype=self.args.autocast_type):
            # compute the logits for unlabeled augmented and unaugmented instances

                unlabeled_logits = self.model(unlabeled_instances)

                probs = torch.softmax(unlabeled_logits, dim=-1).detach()
                weak, strong = probs.chunk(2)

                mask = weak.max(-1)[0] > self.pl_cutoff
                
                # compute labeled logits
                freeze_batchnorm(self.model)
                labeled_logits = self.model(X_aug.to(torch.cuda.current_device()))
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
                        supervised_loss + (0.1 * consistency_loss),
                        self.optimizer,
                        self.scaler,
                        i < (self.accumulate - 1)
                    )
        
        return torch.cat(predictions, 0).cpu(), torch.cat(ground_truth, 0).cpu()

    
    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=self.args.autocast_type):
            return self.model(x)