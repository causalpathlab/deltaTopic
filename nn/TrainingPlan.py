import torch
from inspect import getfullargspec
from typing import Union, Literal
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DeltaTopic.nn.base_model import BaseModuleClass

class TrainingPlan(pl.LightningModule):
    """
    Lightning module task to train deltaTopic modules.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    lr
        Learning rate used for optimization.
    weight_decay
        Weight decay used in optimizatoin.
    eps
        eps used for optimization.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`).
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        eps: float = 0.01,
        optimizer: Literal["Adam", "AdamW"] = "Adam",
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation", "elbo_train"
        ] = "elbo_train",
        lr_min: float = 0,
        **loss_kwargs,
    ):
        super(TrainingPlan, self).__init__()
        self.module = module
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.optimizer_name = optimizer
        self.n_steps_kl_warmup = n_steps_kl_warmup
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_threshold = lr_threshold
        self.lr_min = lr_min
        self.loss_kwargs = loss_kwargs

        self._n_obs_training = None

        # automatic handling of kl weight
        self._loss_args = getfullargspec(self.module.loss)[0]
        if "kl_weight" in self._loss_args:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

    @property
    def n_obs_training(self):
        """
        Number of observations in the training set.

        This will update the loss kwargs for loss rescaling.
        """
        return self._n_obs_training

    @n_obs_training.setter
    def n_obs_training(self, n_obs: int):
        if "n_obs" in self._loss_args:
            self.loss_kwargs.update({"n_obs": n_obs})
        self._n_obs_training = n_obs

    def forward(self, *args, **kwargs):
        """Passthrough to `model.forward()`."""
        return self.module(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        reconstruction_loss_spliced = scvi_loss.reconstruction_loss_spliced
        reconstruction_loss_unspliced = scvi_loss.reconstruction_loss_unspliced
        kl_beta = scvi_loss.kl_beta
        kl_rho = scvi_loss.kl_rho
        kl_delta = scvi_loss.kl_delta
        # pytorch lightning automatically backprops on "loss"
        self.log("train_loss", scvi_loss.loss, on_epoch=True)
        return {
            "loss": scvi_loss.loss,
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            #"kl_global": scvi_loss.kl_global,
            "kl_beta_sum": kl_beta.sum(),
            "kl_rho_sum": kl_rho.sum(),
            "kl_delta_sum": kl_delta.sum(),
            "reconstruction_loss_spliced_sum": reconstruction_loss_spliced.sum(),
            "reconstruction_loss_unspliced_sum": reconstruction_loss_unspliced.sum(),
            "n_obs": reconstruction_loss.shape[0],
        }

    def training_epoch_end(self, outputs):
        n_obs, elbo, rec_loss, kl_local, rec_loss_spliced, rec_loss_unspliced, kl_beta, kl_rho, kl_delta  = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            rec_loss_spliced += tensors['reconstruction_loss_spliced_sum']
            rec_loss_unspliced += tensors['reconstruction_loss_unspliced_sum']
            kl_local += tensors["kl_local_sum"]
            kl_beta += tensors["kl_beta_sum"]
            kl_rho += tensors["kl_rho_sum"]
            kl_delta += tensors["kl_delta_sum"]
            n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        #kl_global = outputs[0]["kl_global"]
        #elbo += kl_global
        self.log("elbo_train", elbo / n_obs)
        self.log("reconstruction_loss_train", rec_loss / n_obs)
        self.log("kl_local_train", kl_local / n_obs)
        self.log("kl_beta_train", kl_beta / n_obs)
        self.log("kl_rho_train", kl_rho / n_obs)
        self.log("kl_delta_train", kl_delta / n_obs)
        self.log("reconstruction_loss_spliced_train", rec_loss_spliced / n_obs)
        self.log("reconstruction_loss_unspliced_train", rec_loss_unspliced / n_obs)
        #self.log("kl_global_train", kl_global)

    def validation_step(self, batch, batch_idx):
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        reconstruction_loss = scvi_loss.reconstruction_loss
        reconstruction_loss_spliced = scvi_loss.reconstruction_loss_spliced
        reconstruction_loss_unspliced = scvi_loss.reconstruction_loss_unspliced
        kl_beta = scvi_loss.kl_beta
        kl_rho = scvi_loss.kl_rho
        kl_delta = scvi_loss.kl_delta
        self.log("validation_loss", scvi_loss.loss, on_epoch=True)
        return {
            "reconstruction_loss_sum": reconstruction_loss.sum(),
            "kl_local_sum": scvi_loss.kl_local.sum(),
            #"kl_global": scvi_loss.kl_global,
            "kl_beta_sum": kl_beta.sum(),
            "kl_rho_sum": kl_rho.sum(),
            "kl_delta_sum": kl_delta.sum(),
            "reconstruction_loss_spliced_sum": reconstruction_loss_spliced.sum(),
            "reconstruction_loss_unspliced_sum": reconstruction_loss_unspliced.sum(),
            "n_obs": reconstruction_loss.shape[0],
        }

    def validation_epoch_end(self, outputs):
        """Aggregate validation step information."""
        #n_obs, elbo, rec_loss, kl_local = 0, 0, 0, 0
        n_obs, elbo, rec_loss, kl_local, rec_loss_spliced, rec_loss_unspliced, kl_beta, kl_rho, kl_delta  = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for tensors in outputs:
            elbo += tensors["reconstruction_loss_sum"] + tensors["kl_local_sum"]
            rec_loss += tensors["reconstruction_loss_sum"]
            rec_loss_spliced = tensors['reconstruction_loss_spliced_sum']
            rec_loss_unspliced = tensors['reconstruction_loss_unspliced_sum']
            kl_local += tensors["kl_local_sum"]
            kl_beta += tensors["kl_beta_sum"]
            kl_rho += tensors["kl_rho_sum"]
            kl_delta += tensors["kl_delta_sum"]
            n_obs += tensors["n_obs"]
        # kl global same for each minibatch
        #kl_global = outputs[0]["kl_global"]
        #elbo += kl_global
        self.log("elbo_validation", elbo / n_obs)
        self.log("reconstruction_loss_validation", rec_loss / n_obs)
        self.log("kl_local_validation", kl_local / n_obs)
        self.log("kl_beta_validation", kl_beta / n_obs)
        self.log("kl_rho_validation", kl_rho / n_obs)
        self.log("kl_delta_validation", kl_delta / n_obs)
        self.log("reconstruction_loss_spliced_validation", rec_loss_spliced / n_obs)
        self.log("reconstruction_loss_unspliced_validation", rec_loss_unspliced / n_obs)
        #self.log("kl_global_validation", kl_global)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        if self.optimizer_name == "Adam":
            optim_cls = torch.optim.Adam
        elif self.optimizer_name == "AdamW":
            optim_cls = torch.optim.AdamW
        else:
            raise ValueError("Optimizer not understood.")
        optimizer = optim_cls(
            params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": scheduler,
                    "monitor": self.lr_scheduler_metric,
                },
            )
        return config

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        epoch_criterion = self.n_epochs_kl_warmup is not None
        step_criterion = self.n_steps_kl_warmup is not None
        if epoch_criterion:
            kl_weight = min(1.0, self.current_epoch / self.n_epochs_kl_warmup)
        elif step_criterion:
            kl_weight = min(1.0, self.global_step / self.n_steps_kl_warmup)
        else:
            kl_weight = 1.0
        return kl_weight

