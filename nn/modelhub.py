import logging
import os
import pickle
import warnings
import torch
import numpy as np
from torch import nn
from anndata import AnnData, read
from typing import List, Optional, Union, Dict
from DeltaTopic.nn.util import _get_var_names_from_setup_anndata, parse_use_gpu_arg, _CONSTANTS, DataSplitter, TrainRunner, BaseModelClass
from DeltaTopic.nn.TrainingPlan import TrainingPlan
from DeltaTopic.nn.module import BALSAM_module, DeltaTopic_module

logger = logging.getLogger(__name__)

def _unpack_tensors(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    unspliced = tensors[_CONSTANTS.PROTEIN_EXP_KEY].squeeze_(0)
    return x, unspliced

def _unpack_tensors_BETM(tensors):
    x = tensors[_CONSTANTS.X_KEY].squeeze_(0)
    return x
    
class BALSAM(BaseModelClass):
    """
    Bayesian Latent topic analysis with Sparse Association Matrix (BALSAM).
    
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~DeltaTopic.nn.util.setup_anndata`.
    n_latent
        Dimensionality of the latent space    
    **model_kwargs
        Keyword args for :class:`~DeltaTopic.nn.module.BALSAM_module`    
    
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> DeltaTopic.nn.util.setup_anndata(adata)
    >>> model = DeltaTopic.nn.modelhub.BALSAM(adata)
    >>> model.train(100)
    """

    def __init__(
        self,
        adata_seq: AnnData,
        n_latent: int = 32,
        **model_kwargs,
    ):
        super(BALSAM, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
         
        self.module = BALSAM_module(
            n_genes = self.adata.n_vars,
            n_latent=n_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "BALSAM with the following params: \nn_latent: {},  n_genes: {}"
        ).format(n_latent, self.adata.n_vars)
           
    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
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
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        """
       
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
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
    ):
        """
        Return the latent space (topic proportions).
        
        Parameters
        ----------
        adatas
            adata registered with setup_anndata.
        deterministic
            If true, use the mean of the encoder instead of a stochastic sample
        output_softmax_z
            If true, output probability, otherwise output z (unnormalized probability).    
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
                sample_batch
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
    ):
        """
        Save the spike and slab parameters to the specificed directory.
        
        Parameters
        ----------
        save_dir
            Save directory.
        overwrite
            If true, overwrite the existing files.
        """
        
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
                save_dir,"model_parameters", "bias_gene.txt"
            ), decoder.bias_d.cpu().numpy())
                
    def save(
        self,
        dir_path: str,
        overwrite: bool = False,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ):
        """
        Save model parameters to the specified directory.
        
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
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.module.state_dict(), model_save_path)

class DeltaTopic(BaseModelClass):
    """
    Dynamically-Encoded Latent Transcriptomic pattern Analysis by Topic modelling (DeltaTopic).
    
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~DeltaTopic.nn.util.setup_anndata`.
    n_latent
        Dimensionality of the latent space    
    **model_kwargs
        Keyword args for :class:`~DeltaTopic.nn.module.DeltaTopic_module`    
    
    Examples
    --------
    >>> adata= anndata.read_h5ad(path_to_anndata_spliced)
    >>> X_unspliced = sc.read(path_to_anndata_spliced)
    >>> adata.obsm["unspliced_expression"] = (X_unspliced.X.copy()
    >>> DeltaTopic.nn.util.setup_anndata(adata, layer="counts", unspliced_obsm_key = "unspliced_expression")
    >>> model = DeltaTopic.nn.modelhub.DeltaTopic(adata)
    >>> model.train(100)
    """

    def __init__(
        self,
        adata_seq: AnnData,
        n_latent: int = 32,
        **model_kwargs,
    ):
        super(DeltaTopic, self).__init__()
        self.n_latent = n_latent
        self.adata = adata_seq
         
        self.module = DeltaTopic_module(
            n_genes = self.adata.n_vars,
            n_latent=n_latent,
            **model_kwargs,
        )
        
        self._model_summary_string = (
            "DeltaTopic with the following params: \nn_latent: {},  n_genes: {} "
        ).format(n_latent, self.adata.n_vars)
        
    def train(
        self,
        max_epochs: Optional[int] = 1000,
        lr: float = 1e-3,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
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
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.

        """
       
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )

        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
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
    ):
        """
        Return the latent space (topic proportions) for spliced and unspliced.

        Parameters
        ----------
        adatas
            List of adata_spliced and adata_unspliced.
        deterministic
            If true, use the mean of the encoder instead of a stochastic sample.
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
    ):
        """
        Save the spike and slab parameters to the specified directory.
        
        Parameters
        ----------
        save_dir
            Directory to save the parameters.
        overwrite
            If true, overwrite the existing parameters.
        """
        
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
                save_dir,"model_parameters", "bias_gene.txt"
            ), decoder.bias_gene.cpu().numpy())
        

    @torch.no_grad()
    def get_reconstruction_error(
        self,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = 128,
    ):
        """
        Return the reconstruction error for the data.


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
        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")

        torch.save(self.module.state_dict(), model_save_path)

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
        dir_path
            Path to saved outputs.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

        """
        model_path = os.path.join(dir_path, "model_params.pt")
        setup_dict_path = os.path.join(dir_path, "attr.pkl")
        seq_data_path = os.path.join(dir_path, "adata.h5ad")
        path_data_path = os.path.join(dir_path, "adata_pathways.h5ad")
        seq_var_names_path = os.path.join(dir_path, "var_names.csv")

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

        adata = adata_seq
        seq_var_names = np.genfromtxt(seq_var_names_path, delimiter=",", dtype=str)
       
        var_names = seq_var_names
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

