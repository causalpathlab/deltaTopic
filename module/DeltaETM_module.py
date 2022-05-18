# -*- coding: utf-8 -*-
"""Main module."""
from typing import List, Optional, Tuple, Optional
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import _CONSTANTS
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data

from nn.base_components import TotalMultiMaskedEncoder, TotalDeltaETMDecoder

torch.backends.cudnn.benchmark = True

def etm_llik(xx, pr, eps=1e-8):
    return torch.sum(xx * torch.log(pr+eps),dim=1)

class TotalDeltaETM_module(BaseModuleClass):
    """

    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset. If
            the datasets have different sizes, the dataloader will loop on the
            smallest until it reaches the size of the longest one
    total_genes
        Total number of different genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    n_layers_encoder_shared
        number of shared layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    n_layers_decoder_individual
        number of layers that are conditionally batchnormed in the encoder
    n_layers_decoder_shared
        number of shared layers in the decoder
    dim_hidden_decoder_individual
        dimension of the individual hidden layers in the decoder
    dim_hidden_decoder_shared
        dimension of the shared hidden layers in the decoder
    dropout_rate_encoder
        dropout encoder
    dropout_rate_decoder
        dropout decoder
    n_batch
        total number of batches
    n_labels
        total number of labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        mask: torch.Tensor = None, 
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 2,
        dim_hidden_encoder: int = 32,
        n_layers_decoder: int = 1, # by default, the decoder has no hidden layers
        dim_hidden_decoder: int = 32, # not in effect when n_layers_decoder = 1
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        n_batch: int = 0,
        n_labels: int = 0,
        log_variational: bool = True,
        combine_method: str = "concat",
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.mask = mask
        self.n_latent = n_latent
        self.combine_method = combine_method
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.log_variational = log_variational

        self.z_encoder = TotalMultiMaskedEncoder(
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            log_variational = self.log_variational,
            combine_method = self.combine_method,          
        )

        # TODO: use self.total_genes is dangerous, if we have dfferent sets of genes in spliced and un unspliced
        self.decoder = TotalDeltaETMDecoder(self.n_latent , self.total_genes)


    def sample_from_posterior_z(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        deterministic: bool = False, 
        output_softmax_z: bool = True,
    ) -> torch.Tensor:
        """
        Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not
        output_softmax_z
            bool - whether to output the softmax of the z or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """

    
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return dict(z=z)
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = None
        reconstruction_loss = -etm_llik(x,recon)
        
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY],y = tensors[_CONSTANTS.PROTEIN_EXP_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        #batch_index = tensors[_CONSTANTS.BATCH_KEY]
        #y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        x_ = x
        y_ = y

        qz_m, qz_v, z = self.z_encoder(x_, y_)
        
        return dict(qz_m=qz_m, qz_v=qz_v, z=z)


    @auto_move_data
    def generative(self,z: torch.Tensor) -> dict:

        log_beta_spliced, log_beta_unspliced, hh, log_softmax_rho, log_softmax_delta  = self.decoder(z)

        
        return dict(log_beta_spliced=log_beta_spliced, log_beta_unspliced=log_beta_unspliced, hh=hh, log_softmax_rho = log_softmax_rho, log_softmax_delta = log_softmax_delta)
    
    # this is for the purpose of computing the integrated gradient, output z but not dict
    def get_latent_representation(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
        output_softmax_z: bool = True, 
    ):
        inference_out = self.inference(x,y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return z

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Returns the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        deterministic
            bool - whether to sample or not
        Returns
        -------
        type
            tensor of means of the scaled frequencies

        """
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
    
        gen_out = self.generative(z)
        
        log_beta_spliced = gen_out["log_beta_spliced"]
        log_beta_unspliced = gen_out["log_beta_unspliced"]
        hh = gen_out["hh"]
        
        recon_spliced = torch.mm(hh,torch.exp(log_beta_spliced))
        recon_unspliced = torch.mm(hh,torch.exp(log_beta_unspliced))
        
        
        reconstruction_loss_spliced = self.reconstruction_loss(x, recon_spliced)
        reconstruction_loss_unspliced = self.reconstruction_loss(y, recon_unspliced)
        return reconstruction_loss_spliced, reconstruction_loss_unspliced

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variances of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences

        """
        x = tensors[_CONSTANTS.X_KEY]
        y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        reconstruction_loss_spliced, reconstruction_loss_unspliced = self.get_reconstruction_loss(x, y)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_local = kl_divergence_z
        reconstruction_loss = reconstruction_loss_spliced + reconstruction_loss_unspliced
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss_spliced,
                            reconstruction_loss_unspliced=reconstruction_loss_unspliced)

class TotalDeltaETM_module_decoupled(BaseModuleClass):
    """
    This is the test idea to fix the mode collapsing problem.
    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset. If
            the datasets have different sizes, the dataloader will loop on the
            smallest until it reaches the size of the longest one
    total_genes
        Total number of different genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    n_layers_encoder_shared
        number of shared layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    n_layers_decoder_individual
        number of layers that are conditionally batchnormed in the encoder
    n_layers_decoder_shared
        number of shared layers in the decoder
    dim_hidden_decoder_individual
        dimension of the individual hidden layers in the decoder
    dim_hidden_decoder_shared
        dimension of the shared hidden layers in the decoder
    dropout_rate_encoder
        dropout encoder
    dropout_rate_decoder
        dropout decoder
    n_batch
        total number of batches
    n_labels
        total number of labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        mask: torch.Tensor = None, 
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 2,
        dim_hidden_encoder: int = 32,
        n_layers_decoder: int = 1, # by default, the decoder has no hidden layers
        dim_hidden_decoder: int = 32, # not in effect when n_layers_decoder = 1
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        n_batch: int = 0,
        n_labels: int = 0,
        log_variational: bool = True,
        combine_method: str = "concat",
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.mask = mask
        self.n_latent = n_latent
        self.combine_method = combine_method
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.log_variational = log_variational

        self.z_encoder = TotalMultiMaskedEncoder(
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            log_variational = self.log_variational,
            combine_method = self.combine_method,          
        )

        # TODO: use self.total_genes is dangerous, if we have dfferent sets of genes in spliced and un unspliced
        self.decoder = TotalDeltaETMDecoder(self.n_latent , self.total_genes)


    def sample_from_posterior_z(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        deterministic: bool = False, 
        output_softmax_z: bool = True,
    ) -> torch.Tensor:
        """
        Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not
        output_softmax_z
            bool - whether to output the softmax of the z or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """

    
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return dict(z=z)
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = None
        reconstruction_loss = -etm_llik(x,recon)
        
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY],y = tensors[_CONSTANTS.PROTEIN_EXP_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        #batch_index = tensors[_CONSTANTS.BATCH_KEY]
        #y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        x_ = x
        y_ = y

        qz_m, qz_v, z = self.z_encoder(x_, y_)
        
        return dict(qz_m=qz_m, qz_v=qz_v, z=z)


    @auto_move_data
    def generative(self,z: torch.Tensor) -> dict:

        log_beta_spliced, log_beta_unspliced, hh, log_softmax_rho, log_softmax_delta  = self.decoder(z)

        
        return dict(log_beta_spliced=log_beta_spliced, log_beta_unspliced=log_beta_unspliced, hh=hh, log_softmax_rho = log_softmax_rho, log_softmax_delta = log_softmax_delta)
    
    # this is for the purpose of computing the integrated gradient, output z but not dict
    def get_latent_representation(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
        output_softmax_z: bool = True, 
    ):
        inference_out = self.inference(x,y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return z

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Returns the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        deterministic
            bool - whether to sample or not
        Returns
        -------
        type
            tensor of means of the scaled frequencies

        """
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
    
        gen_out = self.generative(z)
        log_softmax_rho = gen_out["log_softmax_rho"]
        log_softmax_delta = gen_out["log_softmax_delta"]
        
        #log_beta_spliced = gen_out["log_beta_spliced"]
        #log_beta_unspliced = gen_out["log_beta_unspliced"]
        
        hh = gen_out["hh"]
        
        recon_unspliced = torch.mm(hh,torch.exp(log_softmax_rho))
        recon_spliced = torch.mm(hh,torch.exp(log_softmax_rho)) * torch.mm(hh,torch.exp(log_softmax_delta))
        
        reconstruction_loss_spliced = self.reconstruction_loss(x, recon_spliced)
        reconstruction_loss_unspliced = self.reconstruction_loss(y, recon_unspliced)
        return reconstruction_loss_spliced, reconstruction_loss_unspliced

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variances of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences

        """
        x = tensors[_CONSTANTS.X_KEY]
        y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        reconstruction_loss_spliced, reconstruction_loss_unspliced = self.get_reconstruction_loss(x, y)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_local = kl_divergence_z
        reconstruction_loss = reconstruction_loss_spliced + reconstruction_loss_unspliced
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss_spliced,
                            reconstruction_loss_unspliced=reconstruction_loss_unspliced)

class ETM_module(BaseModuleClass):
    """
    This is the test idea to fix the mode collapsing problem.
    Parameters
    ----------
    dim_input_list
        List of number of input genes for each dataset. If
            the datasets have different sizes, the dataloader will loop on the
            smallest until it reaches the size of the longest one
    total_genes
        Total number of different genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    n_layers_encoder_shared
        number of shared layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    n_layers_decoder_individual
        number of layers that are conditionally batchnormed in the encoder
    n_layers_decoder_shared
        number of shared layers in the decoder
    dim_hidden_decoder_individual
        dimension of the individual hidden layers in the decoder
    dim_hidden_decoder_shared
        dimension of the shared hidden layers in the decoder
    dropout_rate_encoder
        dropout encoder
    dropout_rate_decoder
        dropout decoder
    n_batch
        total number of batches
    n_labels
        total number of labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.

    """

    def __init__(
        self,
        dim_input_list: List[int],
        total_genes: int,
        mask: torch.Tensor = None, 
        n_latent: int = 10,
        n_layers_encoder_individual: int = 1,
        n_layers_encoder_shared: int = 2,
        dim_hidden_encoder: int = 32,
        n_layers_decoder: int = 1, # by default, the decoder has no hidden layers
        dim_hidden_decoder: int = 32, # not in effect when n_layers_decoder = 1
        dropout_rate_encoder: float = 0.1,
        dropout_rate_decoder: float = 0.1,
        n_batch: int = 0,
        n_labels: int = 0,
        log_variational: bool = True,
        combine_method: str = "concat",
    ):
        super().__init__()

        self.n_input_list = dim_input_list
        self.total_genes = total_genes
        self.mask = mask
        self.n_latent = n_latent
        self.combine_method = combine_method
        self.n_batch = n_batch
        self.n_labels = n_labels

        self.log_variational = log_variational

        self.z_encoder = TotalMultiMaskedEncoder(
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            mask = self.mask,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            n_layers_shared=n_layers_encoder_shared,
            dropout_rate=dropout_rate_encoder,
            log_variational = self.log_variational,
            combine_method = self.combine_method,          
        )

        # TODO: use self.total_genes is dangerous, if we have dfferent sets of genes in spliced and un unspliced
        self.decoder = TotalDeltaETMDecoder(self.n_latent , self.total_genes)


    def sample_from_posterior_z(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        deterministic: bool = False, 
        output_softmax_z: bool = True,
    ) -> torch.Tensor:
        """
        Sample tensor of latent values from the posterior.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        mode
            head id to use in the encoder
        deterministic
            bool - whether to sample or not
        output_softmax_z
            bool - whether to output the softmax of the z or not

        Returns
        -------
        type
            tensor of shape ``(batch_size, n_latent)``

        """

    
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return dict(z=z)
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:

        reconstruction_loss = None
        reconstruction_loss = -etm_llik(x,recon)
        
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY],y = tensors[_CONSTANTS.PROTEIN_EXP_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        #batch_index = tensors[_CONSTANTS.BATCH_KEY]
        #y = tensors[_CONSTANTS.LABELS_KEY]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        x_ = x
        y_ = y

        qz_m, qz_v, z = self.z_encoder(x_, y_)
        
        return dict(qz_m=qz_m, qz_v=qz_v, z=z)


    @auto_move_data
    def generative(self,z: torch.Tensor) -> dict:

        log_beta_spliced, log_beta_unspliced, hh, log_softmax_rho, log_softmax_delta  = self.decoder(z)

        
        return dict(log_beta_spliced=log_beta_spliced, log_beta_unspliced=log_beta_unspliced, hh=hh, log_softmax_rho = log_softmax_rho, log_softmax_delta = log_softmax_delta)
    
    # this is for the purpose of computing the integrated gradient, output z but not dict
    def get_latent_representation(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
        output_softmax_z: bool = True, 
    ):
        inference_out = self.inference(x,y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["hh"]      
        return z

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Returns the tensor of scaled frequencies of expression.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        deterministic
            bool - whether to sample or not
        Returns
        -------
        type
            tensor of means of the scaled frequencies

        """
        inference_out = self.inference(x, y)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
    
        gen_out = self.generative(z)
        log_softmax_rho = gen_out["log_softmax_rho"]
        log_softmax_delta = gen_out["log_softmax_delta"]
        
        #log_beta_spliced = gen_out["log_beta_spliced"]
        #log_beta_unspliced = gen_out["log_beta_unspliced"]
        
        hh = gen_out["hh"]
        
        recon_unspliced = torch.mm(hh,torch.exp(log_softmax_rho))
        recon_spliced = recon_unspliced
        
        reconstruction_loss_spliced = self.reconstruction_loss(x, recon_spliced)
        reconstruction_loss_unspliced = self.reconstruction_loss(y, recon_unspliced)
        return reconstruction_loss_spliced, reconstruction_loss_unspliced

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Return the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
            or ``(batch_size, n_input_fish)`` depending on the mode
        local_l_mean
            tensor of means of the prior distribution of latent variable l
            with shape (batch_size, 1)
        local_l_var
            tensor of variances of the prior distribution of latent variable l
            with shape (batch_size, 1)
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        y
            tensor of cell-types labels with shape (batch_size, n_labels)
        mode
            indicates which head/tail to use in the joint network


        Returns
        -------
        the reconstruction loss and the Kullback divergences

        """
        x = tensors[_CONSTANTS.X_KEY]
        y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        
        reconstruction_loss_spliced, reconstruction_loss_unspliced = self.get_reconstruction_loss(x, y)

        # KL Divergence
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        kl_local = kl_divergence_z
        reconstruction_loss = reconstruction_loss_spliced + reconstruction_loss_unspliced
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) * x.size(0)

        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss_spliced,
                            reconstruction_loss_unspliced=reconstruction_loss_unspliced)
