import logging
import os
import pickle
import warnings
from itertools import cycle
from typing import List, Optional, Union, Dict, Sequence
from typing_extensions import Literal

import numpy as np
import torch
from anndata import AnnData, read
from torch import nn as nn

from scvi import _CONSTANTS
from scvi.data import transfer_anndata_setup
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _get_var_names_from_setup_anndata, parse_use_gpu_arg
from scvi.model.base import RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass
from scvi.train import TrainRunner
from task.TotalDeltaETM_TrainingPlan import TrainingPlan

from module.DeltaETM_module import TotalDeltaETM_module, TotalDeltaETM_module_decoupled, BETM_module, BayesianETM_module
from scvi.train import TrainRunner

logger = logging.getLogger(__name__)

def _unpack_tensors(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    unspliced = tensors[_CONSTANTS.PROTEIN_EXP_KEY].squeeze_(0)
    local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY].squeeze_(0)
    local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY].squeeze_(0)
    batch_index = tensors[_CONSTANTS.BATCH_KEY].squeeze_(0)
    y = tensors[_CONSTANTS.LABELS_KEY].squeeze_(0)
    return x, unspliced, local_l_mean, local_l_var, batch_index, y

def _unpack_tensors_BETM(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY].squeeze_(0)
    local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY].squeeze_(0)
    batch_index = tensors[_CONSTANTS.BATCH_KEY].squeeze_(0)
    y = tensors[_CONSTANTS.LABELS_KEY].squeeze_(0)
    return x, local_l_mean, local_l_var, batch_index, y

def set_up_adata_pathway(adata_pathway, minGenes=10):
    """
    remove pathways with less than "minGenes" genes
    """
    # remove bad pathways
    mask = adata_pathway.X.copy()
    pathway_size = np.sum(mask,1)
    bad_pathway = pathway_size < minGenes
    mask[bad_pathway,:] = 0
    adata_pathway.X = mask
    print(f'Removing {np.sum(bad_pathway)} pathways less than {minGenes} genes\n')
    return(adata_pathway)
 
class TotalDeltaETM(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    DeltaETM

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_seq
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> model = TotalDeltaEMT(adata_seq adata_pathway)
    >>> model.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 4,
        combine_latent: str = 'concat',
        **model_kwargs,
    ):
        super(TotalDeltaETM, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
        self.scvi_setup_dicts_ = self.adata.uns["_scvi"]
        if adata_pathway is not None:
            # condition check
            var_names = _get_var_names_from_setup_anndata(adata_seq)
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(var_names) == set(pathway_var_names):
                raise ValueError("expression genes needs to be equal to pathway genes")
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
        
        self.summary_stats = adata_seq.uns['_scvi']["summary_stats"]    
        self.module = TotalDeltaETM_module(
            dim_input_list = [self.summary_stats["n_vars"],self.summary_stats["n_vars"]],
            total_genes = self.summary_stats["n_vars"],
            mask=self.mask,
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            combine_method = combine_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "TotalDelETM Model with the following params: \nn_latent: {},  n_genes: {}, "
            + "n_batch: {}, combine_latent: {}"
        ).format(n_latent, self.summary_stats["n_vars"], self.summary_stats["n_batch"], combine_latent)
        self.init_params_ = self._get_init_params(locals())    
        #wandb.watch(self.module.decoder, log_freq=10, log="all")
    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = False,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        #adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        '''if adversarial_classifier is None:
            imputation = (
                True if "totalvi_batch_mask" in self.scvi_setup_dict_.keys() else False
            )
            adversarial_classifier = True if imputation else False
        '''
        #adversarial_classifier = False    
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            #"adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            #"check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            **kwargs,
        )
        return runner()
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        batch_size: int = 128,
        output_softmax_z: bool = True,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        output_softmax_z
            If true, return the softmax of the latent space embedding.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch,
                sample_batch_unspliced,
                *_,
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, sample_batch_unspliced, deterministic=deterministic, output_softmax_z = output_softmax_z)
            latent_z.append(z_dict["z"])                

        latent_z = torch.cat(latent_z).cpu().detach().numpy()
        
        print(f'Deterministic: {deterministic}\nOutput_softmax_z: {output_softmax_z}')
        return latent_z
    
    @torch.no_grad()
    def get_weights(
        self,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
       
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()
        log_sftm = nn.LogSoftmax(dim=-1)
        log_delta = self.module.decoder.delta.detach()
        log_rho = self.module.decoder.rho.detach()
        
        delta = torch.exp(log_sftm(log_delta))
        rho = torch.exp(log_sftm(log_rho))
        return delta.cpu().numpy(), rho.cpu().numpy(), log_delta.cpu().numpy(), log_rho.cpu().numpy()
    
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
            varnames_save_path = os.path.join(
                dir_path, "var_names.csv"
            )

            var_names = self.adata.var_names.astype(str)
            var_names = var_names.to_numpy()
            np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        #adatas = [adata_seq, adata_spatial]
        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        #spatial_var_names = np.genfromtxt(
        #    spatial_var_names_path, delimiter=",", dtype=str
        #)
        var_names = seq_var_names
        #for i, adata in enumerate(adatas):
        saved_var_names = var_names
        user_var_names = adata.var_names.astype(str)
        if not np.array_equal(saved_var_names, user_var_names):
            warnings.warn(
                "var_names for adata passed in does not match var_names of "
                "adata used to train the model. For valid results, the vars "
                "need to be the same and in the same order as the adata used to train the model."
            )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts, adata_seq)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

class TotalDeltaETM_decoupled(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    DeltaETM

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_seq
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> model = TotalDeltaEMT(adata_seq adata_pathway)
    >>> model.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 4,
        combine_latent: str = 'concat',
        **model_kwargs,
    ):
        super(TotalDeltaETM_decoupled, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
        self.scvi_setup_dicts_ = self.adata.uns["_scvi"]
        if adata_pathway is not None:
            # condition check
            var_names = _get_var_names_from_setup_anndata(adata_seq)
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(var_names) == set(pathway_var_names):
                raise ValueError("expression genes needs to be equal to pathway genes")
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
        
        self.summary_stats = adata_seq.uns['_scvi']["summary_stats"]    
        self.module = TotalDeltaETM_module_decoupled(
            dim_input_list = [self.summary_stats["n_vars"],self.summary_stats["n_vars"]],
            total_genes = self.summary_stats["n_vars"],
            mask=self.mask,
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            combine_method = combine_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "TotalDelETM decpoupled Model with the following params: \nn_latent: {},  n_genes: {}, "
            + "n_batch: {}, combine_latent: {}"
        ).format(n_latent, self.summary_stats["n_vars"], self.summary_stats["n_batch"], combine_latent)
        self.init_params_ = self._get_init_params(locals())    
        #wandb.watch(self.module.decoder, log_freq=10, log="all")
    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = False,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        #adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        '''if adversarial_classifier is None:
            imputation = (
                True if "totalvi_batch_mask" in self.scvi_setup_dict_.keys() else False
            )
            adversarial_classifier = True if imputation else False
        '''
        #adversarial_classifier = False    
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            #"adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            #"check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            **kwargs,
        )
        return runner()
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        batch_size: int = 128,
        output_softmax_z: bool = True,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        batch_size
            Minibatch size for data loading into model.
        output_softmax_z
            If true, return the softmax of the latent space embedding.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch,
                sample_batch_unspliced,
                *_,
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, sample_batch_unspliced, deterministic=deterministic, output_softmax_z = output_softmax_z)
            latent_z.append(z_dict["z"])                

        latent_z = torch.cat(latent_z).cpu().detach().numpy()
        
        print(f'Deterministic: {deterministic}\nOutput_softmax_z: {output_softmax_z}')
        return latent_z
    
    @torch.no_grad()
    def get_weights(
        self,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
       
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()
        log_sftm = nn.LogSoftmax(dim=-1)
        log_delta = self.module.decoder.delta.detach()
        log_rho = self.module.decoder.rho.detach()
        
        delta = torch.exp(log_sftm(log_delta))
        rho = torch.exp(log_sftm(log_rho))
        return delta.cpu().numpy(), rho.cpu().numpy(), log_delta.cpu().numpy(), log_rho.cpu().numpy()
    
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
            varnames_save_path = os.path.join(
                dir_path, "var_names.csv"
            )

            var_names = self.adata.var_names.astype(str)
            var_names = var_names.to_numpy()
            np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        #adatas = [adata_seq, adata_spatial]
        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        #spatial_var_names = np.genfromtxt(
        #    spatial_var_names_path, delimiter=",", dtype=str
        #)
        var_names = seq_var_names
        #for i, adata in enumerate(adatas):
        saved_var_names = var_names
        user_var_names = adata.var_names.astype(str)
        if not np.array_equal(saved_var_names, user_var_names):
            warnings.warn(
                "var_names for adata passed in does not match var_names of "
                "adata used to train the model. For valid results, the vars "
                "need to be the same and in the same order as the adata used to train the model."
            )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts, adata_seq)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

class BETM(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    BaeyainETM

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_seq
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> model = BETM(adata_seq adata_pathway)
    >>> model.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 32,
        **model_kwargs,
    ):
        super(BETM, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
        self.scvi_setup_dicts_ = self.adata.uns["_scvi"]
        if adata_pathway is not None:
            # condition check
            var_names = _get_var_names_from_setup_anndata(adata_seq)
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(var_names) == set(pathway_var_names):
                raise ValueError("expression genes needs to be equal to pathway genes")
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
        
        self.summary_stats = adata_seq.uns['_scvi']["summary_stats"]    
        self.module = BETM_module(
            dim_input = self.summary_stats["n_vars"],
            total_genes = self.summary_stats["n_vars"],
            mask=self.mask,
            #n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "BETM with the following params: \nn_latent: {},  n_genes: {}, "
            + "n_batch: {}"
        ).format(n_latent, self.summary_stats["n_vars"], self.summary_stats["n_batch"])
        self.init_params_ = self._get_init_params(locals()) 
           
        #wandb.watch(self.module.decoder, log_freq=10, log="all")
    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = False,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        #adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        '''if adversarial_classifier is None:
            imputation = (
                True if "totalvi_batch_mask" in self.scvi_setup_dict_.keys() else False
            )
            adversarial_classifier = True if imputation else False
        '''
        #adversarial_classifier = False    
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            #"adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            #"check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, 

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.    
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch,
                *_,
            ) = _unpack_tensors_BETM(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])                

        latent_z = torch.cat(latent_z).cpu().detach().numpy()
        
        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z
    
    
    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None, 
        overwrite = False,
    ) -> List[np.ndarray]:
        """return the spike logit, slab mean, slab lnvar for rho"""
        
        self.module.eval()
        decoder = self.module.decoder
        
        
        if not os.path.exists(os.path.join(save_dir,"model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir,"model_parameters"), exist_ok=overwrite)
            
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "spike_logit_rho.txt"
            ), decoder.spike_logit.cpu().numpy())
        

        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_rho.txt"
            ), decoder.slab_mean.cpu().numpy())
        
    
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_rho.txt"
            ), decoder.slab_lnvar.cpu().numpy())
        
 
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_gene_rho.txt"
            ), decoder.bias_d.cpu().numpy())
        
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_topic_rho.txt"
            ), decoder.bias_k.cpu().numpy())
        
        
    @torch.no_grad()
    def get_weights(
        self,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
       
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()
        rho = self.module.decoder.get_rho()
        
        return rho.cpu().numpy()
    
    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = 128,
    ) -> Union[float, Dict[str, float]]:
        r"""
        Return the reconstruction error for the data.

        This is typically written as :math:`p(x \mid z)`, the likelihood term given one posterior sample.
        Note, this is not the negative likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        batch_size
            Minibatch size for data loading into model.
        """
        
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()
        
        reconstruction_loss_sum = 0
        n = 0
        
        for tensors in scdl:
            (
                sample_batch,
                *_,
            ) = _unpack_tensors(tensors)
            reconstruction_loss  = self.module.get_reconstruction_loss(sample_batch)
            reconstruction_loss_sum += torch.sum(reconstruction_loss)
            
            n += sample_batch.shape[0]
            
        recon = reconstruction_loss_sum/n
        
        return recon.cpu().numpy()
    
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
            varnames_save_path = os.path.join(
                dir_path, "var_names.csv"
            )

            var_names = self.adata.var_names.astype(str)
            var_names = var_names.to_numpy()
            np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        #adatas = [adata_seq, adata_spatial]
        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        #spatial_var_names = np.genfromtxt(
        #    spatial_var_names_path, delimiter=",", dtype=str
        #)
        var_names = seq_var_names
        #for i, adata in enumerate(adatas):
        saved_var_names = var_names
        user_var_names = adata.var_names.astype(str)
        if not np.array_equal(saved_var_names, user_var_names):
            warnings.warn(
                "var_names for adata passed in does not match var_names of "
                "adata used to train the model. For valid results, the vars "
                "need to be the same and in the same order as the adata used to train the model."
            )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts, adata_seq)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

class BDeltaTopic(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    DeltaETM

    Parameters
    ----------
    adata_seq
        Spliced and unspliced count AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains data.
    adata_pathway
        Pathway AnnData object that has been registered via :func:`~scvi.data.setup_anndata`
        and contains pathway information. Note that genes needs to be equal to the input genes in 
        adata_seq
    mask 
        Binary torch.tensor with the shape of of [n_pathways, n_genes]. Note that this option 
        is only avaiable when adata_pathway is None 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    **model_kwargs
        Keyword args for :class:`~module.DeltaETM_module`

    Examples
    --------
    >>> adata_seq = anndata.read_h5ad(path_to_anndata_seq)
    >>> scvi.data.setup_anndata(adata_seq)
    >>> model = BDeltaTopic(adata_seq adata_pathway)
    >>> model.train(n_epochs=400)


    """

    def __init__(
        self,
        adata_seq: AnnData,
        adata_pathway: AnnData = None,
        mask: torch.Tensor = None,
        n_latent: int = 32,
        combine_latent: str = 'add',
        **model_kwargs,
    ):
        super(BDeltaTopic, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
        self.scvi_setup_dicts_ = self.adata.uns["_scvi"]
        if adata_pathway is not None:
            # condition check
            var_names = _get_var_names_from_setup_anndata(adata_seq)
            pathway_var_names = _get_var_names_from_setup_anndata(adata_pathway)
            if not set(var_names) == set(pathway_var_names):
                raise ValueError("expression genes needs to be equal to pathway genes")
        
        self.adata_pathway = None
        if adata_pathway is not None:
            self.adata_pathway = set_up_adata_pathway(adata_pathway)
            self.mask = torch.from_numpy(self.adata_pathway.X)
            print("mask is from Anndata object")
        elif mask is not None:
            self.mask = mask
            print("mask is taken from user-specified input\n")
        else:
            self.mask = None
            print("No pathways, use fully-connected layers\n")
        
        self.summary_stats = adata_seq.uns['_scvi']["summary_stats"]    
        self.module = BayesianETM_module(
            dim_input_list = [self.summary_stats["n_vars"],self.summary_stats["n_vars"]],
            total_genes = self.summary_stats["n_vars"],
            mask=self.mask,
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            combine_method = combine_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "BDeltaTopic with the following params: \nn_latent: {},  n_genes: {}, "
            + "n_batch: {}, combine_latent: {}"
        ).format(n_latent, self.summary_stats["n_vars"], self.summary_stats["n_batch"], combine_latent)
        self.init_params_ = self._get_init_params(locals()) 
           
        #wandb.watch(self.module.decoder, log_freq=10, log="all")
    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 256,
        early_stopping: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = False,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        #adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        '''if adversarial_classifier is None:
            imputation = (
                True if "totalvi_batch_mask" in self.scvi_setup_dict_.keys() else False
            )
            adversarial_classifier = True if imputation else False
        '''
        #adversarial_classifier = False    
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            #"adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            #"check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = TrainingPlan(self.module, **plan_kwargs)
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()
    
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: AnnData = None,
        deterministic: bool = True,
        output_softmax_z: bool = True,
        batch_size: int = 128,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a Gaussian sample.
        output_softmax_z
            if true, output probability, otherwise output z.    
        batch_size
            Minibatch size for data loading into model.
        """
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()

        latent_z = []
        for tensors in scdl:
            (
                sample_batch,
                sample_batch_unspliced,
                *_,
            ) = _unpack_tensors(tensors)
            z_dict  = self.module.sample_from_posterior_z(sample_batch, sample_batch_unspliced, deterministic=deterministic, output_softmax_z=output_softmax_z)
            latent_z.append(z_dict["z"])                

        latent_z = torch.cat(latent_z).cpu().detach().numpy()
        
        print(f'Deterministic: {deterministic}, output_softmax_z: {output_softmax_z}' )
        return latent_z
    
    
    @torch.no_grad()
    def get_parameters(
        self,
        save_dir = None, 
        overwrite = False,
    ) -> List[np.ndarray]:
        """return the spike logit, slab mean, slab lnvar for rho and delta"""
        
        self.module.eval()
        decoder = self.module.decoder
        
        
        if not os.path.exists(os.path.join(save_dir,"model_parameters")) or overwrite:
            os.makedirs(os.path.join(save_dir,"model_parameters"), exist_ok=overwrite)
            
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "spike_logit_delta.txt"
            ), decoder.spike_logit_delta.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "spike_logit_rho.txt"
            ), decoder.spike_logit_rho.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_delta.txt"
            ), decoder.slab_mean_delta.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_mean_rho.txt"
            ), decoder.slab_mean_rho.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_delta.txt"
            ), decoder.slab_lnvar_delta.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "slab_lnvar_rho.txt"
            ), decoder.slab_lnvar_rho.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_gene_delta.txt"
            ), decoder.bias_d_delta.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_gene_rho.txt"
            ), decoder.bias_d_rho.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_topic_delta.txt"
            ), decoder.bias_k_delta.cpu().numpy())
        
        np.savetxt(os.path.join(
                save_dir,"model_parameters", "bias_topic_rho.txt"
            ), decoder.bias_k_rho.cpu().numpy())
        
        '''return(decoder.spike_logit_delta.cpu().numpy(),
            decoder.spike_logit_rho.cpu().numpy(),

            decoder.slab_mean_delta.cpu().numpy(),
            decoder.slab_mean_rho.cpu().numpy(),
            
            decoder.slab_lnvar_delta.cpu().numpy(),
            decoder.slab_lnvar_rho.cpu().numpy(),
            
            decoder.bias_d_delta.cpu().numpy(),
            decoder.bias_d_rho.cpu().numpy(), 
            
            decoder.bias_k_delta.cpu().numpy(), 
            decoder.bias_k_rho.cpu().numpy(),
        )'''
        
    @torch.no_grad()
    def get_weights(
        self,
    ) -> List[np.ndarray]:
        """
        Return the latent space embedding for each dataset, i.e., spliced and unspliced

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
       
        batch_size
            Minibatch size for data loading into model.
        """
        self.module.eval()
        rho, delta = self.module.decoder.get_rho_delta()
        
        return delta.cpu().numpy(), rho.cpu().numpy()
    
    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = 128,
    ) -> Union[float, Dict[str, float]]:
        r"""
        Return the reconstruction error for the data.

        This is typically written as :math:`p(x \mid z)`, the likelihood term given one posterior sample.
        Note, this is not the negative likelihood, higher is better.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        batch_size
            Minibatch size for data loading into model.
        """
        
        if adata is None:
            adata = self.adata
        scdl = self._make_data_loader(adata, batch_size=batch_size)
        self.module.eval()
        
        reconstruction_loss_spliced_sum = 0
        reconstruction_loss_unspliced_sum = 0
        n_spliced = 0
        n_unspliced = 0
        
        for tensors in scdl:
            (
                sample_batch_spliced,
                sample_batch_unspliced,
                *_,
            ) = _unpack_tensors(tensors)
            reconstruction_loss_spliced, reconstruction_loss_unspliced  = self.module.get_reconstruction_loss(sample_batch_spliced, sample_batch_unspliced)
            reconstruction_loss_spliced_sum += torch.sum(reconstruction_loss_spliced)
            reconstruction_loss_unspliced_sum += torch.sum(reconstruction_loss_unspliced)
            n_spliced += sample_batch_spliced.shape[0]
            n_unspliced += sample_batch_unspliced.shape[0]

        recon_spliced = reconstruction_loss_spliced_sum/n_spliced
        recon_unspliced = reconstruction_loss_unspliced_sum/n_unspliced
        return recon_spliced.cpu().numpy(), recon_unspliced.cpu().numpy()
    
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save the state of the model.

        Neither the trainer optimizer state nor the trainer history are saved.
        Model files are not expected to be reproducibly saved and loaded across versions
        until we reach version 1.0.

        Parameters
        ----------
        dir_path
            Path to a directory.
        overwrite
            Overwrite existing data or not. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_anndata
            If True, also saves the anndata
        anndata_write_kwargs
            Kwargs for anndata write function
        """
        # get all the user attributes
        user_attributes = self._get_user_attributes()
        # only save the public attributes with _ at the very end
        user_attributes = {a[0]: a[1] for a in user_attributes if a[0][-1] == "_"}
        # save the model state dict and the trainer state dict only
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists. Please provide an unexisting directory for saving.".format(
                    dir_path
                )
            )
        if save_anndata:
            save_path = os.path.join(
                dir_path, "adata.h5ad"
            )
            self.adata.write(save_path)
            varnames_save_path = os.path.join(
                dir_path, "var_names.csv"
            )

            var_names = self.adata.var_names.astype(str)
            var_names = var_names.to_numpy()
            np.savetxt(varnames_save_path, var_names, fmt="%s")
            # saving pathways
            if self.adata_pathway is not None:
                save_path = os.path.join(
                    dir_path, "adata_pathways.h5ad"
                )
                self.adata_pathway.write(save_path)
                varnames_save_path = os.path.join(
                    dir_path, "var_names_pathways.csv"
                )
                var_names = self.adata_pathway.var_names.astype(str)
                var_names = var_names.to_numpy()
                np.savetxt(varnames_save_path, var_names, fmt="%s")
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata_seq: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        adata_seq
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            AnnData must be registered via :func:`~scvi.data.setup_anndata`.
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        Examples
        --------
        >>> vae = GIMVI.load(adata_seq, adata_spatial, save_path)
        >>> vae.get_latent_representation()
        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")
        path_var_names_path = os.path.join(dir_path, "var_names_pathways.csv")

        if adata_seq is None and os.path.exists(seq_data_path):
            adata_seq = read(seq_data_path)
        elif adata_seq is None and not os.path.exists(seq_data_path):
            raise ValueError(
                "Save path contains no saved anndata and no adata was passed."
            )
        
        if os.path.exists(path_data_path):
            adata_path = read(path_data_path)
        elif not os.path.exists(path_data_path):
            adata_path = None
            print("no pathways saved")

        #adatas = [adata_seq, adata_spatial]
        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
        #spatial_var_names = np.genfromtxt(
        #    spatial_var_names_path, delimiter=",", dtype=str
        #)
        var_names = seq_var_names
        #for i, adata in enumerate(adatas):
        saved_var_names = var_names
        user_var_names = adata.var_names.astype(str)
        if not np.array_equal(saved_var_names, user_var_names):
            warnings.warn(
                "var_names for adata passed in does not match var_names of "
                "adata used to train the model. For valid results, the vars "
                "need to be the same and in the same order as the adata used to train the model."
            )

        with open(setup_dict_path, "rb") as handle:
            attr_dict = pickle.load(handle)

        scvi_setup_dicts = attr_dict.pop("scvi_setup_dicts_")
        transfer_anndata_setup(scvi_setup_dicts, adata_seq)
      
        # get the parameters for the class init signiture
        init_params = attr_dict.pop("init_params_")

        # new saving and loading, enable backwards compatibility
        if "non_kwargs" in init_params.keys():
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = init_params["non_kwargs"]
            kwargs = init_params["kwargs"]

            # expand out kwargs
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        else:
            # grab all the parameters execept for kwargs (is a dict)
            non_kwargs = {
                k: v for k, v in init_params.items() if not isinstance(v, dict)
            }
            kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
            kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        
        # the default init require this way of loading models
        if adata_path is not None:    
            model = cls(adata_seq, **non_kwargs, adata_pathway=adata_path, **kwargs)
        elif adata_path is None:
            model = cls(adata_seq, **non_kwargs, **kwargs)

        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        _, device = parse_use_gpu_arg(use_gpu)
        model.module.load_state_dict(torch.load(model_path, map_location=device))
        model.module.eval()
        model.to_device(device)
        return model

