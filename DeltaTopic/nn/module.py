# -*- coding: utf-8 -*-
"""Main module."""
from typing import List, Tuple
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from DeltaTopic.nn.util import _CONSTANTS
from DeltaTopic.nn.base_model import BaseModuleClass, LossRecorder, auto_move_data
from DeltaTopic.nn.base_components import DeltaTopicEncoder, BALSAMDecoder, BALSAMEncoder, DeltaTopicDecoder

torch.backends.cudnn.benchmark = True

class BALSAM_module(BaseModuleClass):
    """
    BALASM module
    
    Parameters
    ----------
    n_genes
        number of genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    pip0_rho
        scaling factor for rho loss, default 0.1
    kl_weight_beta: 
        scaling factor for KL, default 1.0
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    """

    def __init__(
        self,
        n_genes: int,
        n_latent: int = 32,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        log_variational: bool = True,
        pip0_rho: float = 0.1,
        kl_weight_beta: float = 1.0,        
    ):
        super().__init__()

        self.n_input = n_genes
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.pip0_rho = pip0_rho
        self.kl_weight_beta = kl_weight_beta
            
        self.z_encoder = BALSAMEncoder(
            n_input=self.n_input,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational = self.log_variational,    
        )

        self.decoder = BALSAMDecoder(self.n_latent , 
                                    self.n_input, 
                                    pip0 = self.pip0_rho,
                                    )

    def dir_llik(self, 
                 xx: torch.Tensor, 
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Compute the Dirichlet log-likelihood.
        '''
        reconstruction_loss = None 
        
        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor) -> dict:
        x_ = x

        qz_m, qz_v, z = self.z_encoder(x_)
        
        return dict(qz_m=qz_m, qz_v=qz_v, z=z)


    @auto_move_data
    def generative(self, z) -> dict:

        rho, rho_kl, theta  = self.decoder(z)

        return dict(rho = rho, rho_kl = rho_kl, theta = theta)
    
    def sample_from_posterior_z(
        self, 
        x: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True, 
    ):
        """Sample from the posterior z
        """
        inference_out = self.inference(x)
        if deterministic:
            z = inference_out["qz_m"]
        else:
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["theta"]      
        return dict(z=z)
    
    @auto_move_data
    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the reconstruction loss for a batch of data. 
        
        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        
        Returns
        -------
        type
            tensor of means of the scaled frequencies
        """
        inference_out = self.inference(x)
        z = inference_out["z"]
        gen_out = self.generative(z)
        theta = gen_out["theta"]
               
        rho = gen_out["rho"]
        log_aa = torch.clamp(torch.mm(theta, rho), -10, 10)
        aa = torch.exp(log_aa)
        
        reconstruction_loss = -self.dir_llik(x, aa)

        return reconstruction_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
        #kl_weight_beta = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Agrregates the likelihood and KL divergences to form the loss function.
        """
        kl_weight_beta = self.kl_weight_beta
        x = tensors[_CONSTANTS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        rho_kl = generative_outputs["rho_kl"]
        
        # [batch_size]
        reconstruction_loss = self.get_reconstruction_loss(x)

        # KL Divergence for z [batch_size]
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        ) # suming over all the latent dimensinos
        # kl_divergence for beta, rho_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = rho_kl
        kl_local = kl_divergence_z
        
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) + kl_weight_beta * kl_divergence_beta/x.shape[1]
        
        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss,
                            reconstruction_loss_unspliced=torch.Tensor(0), 
                            kl_beta = kl_divergence_beta, 
                            kl_rho = rho_kl, 
                            kl_delta = torch.Tensor(0)) 

class DeltaTopic_module(BaseModuleClass):
    """
    DeltaTopic module.
    
    Parameters
    ----------
    n_genes
        number of genes
    n_latent
        dimension of latent space
    n_layers_encoder_individual
        number of individual layers in the encoder
    dim_hidden_encoder
        dimension of the hidden layers in the encoder
    pip0_rho
        scaling factor for rho loss, default 0.1
    pip0_delta
        scaling factor for delta loss, default 0.1
    kl_weight_beta: 
        scaling factor for KL, default 1.0
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    """

    def __init__(
        self,
        n_genes: int,
        n_latent: int = 10,
        n_layers_encoder_individual: int = 2,
        dim_hidden_encoder: int = 128,
        pip0_rho: float = 0.1,
        pip0_delta: float = 0.1,
        kl_weight_beta: float = 1.0,
        log_variational: bool = True,
        
    ):
        super().__init__()
        dim_input_list = [n_genes, n_genes]
        
        self.n_input_list = dim_input_list
        self.total_genes = n_genes
        self.n_latent = n_latent
        self.pip0_rho = pip0_rho
        self.pip0_delta = pip0_delta
        self.log_variational = log_variational
        self.kl_weight_beta = kl_weight_beta
        
        
        self.z_encoder = DeltaTopicEncoder(
            n_input_list=dim_input_list,
            n_output=self.n_latent,
            n_hidden=dim_hidden_encoder,
            n_layers_individual=n_layers_encoder_individual,
            log_variational = self.log_variational,
             
        )
        
        # TODO: use self.total_genes is dangerous, if we have dfferent sets of genes in spliced and unspliced
        self.decoder = DeltaTopicDecoder(self.n_latent , 
                                          self.total_genes, 
                                          pip0_rho = self.pip0_rho,
                                          pip0_delta = self.pip0_delta,
                                        )

    def dir_llik(self, 
                 xx: torch.Tensor, 
                 aa: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Return the Dirichlet log-likelihood for a batch.
        '''
        reconstruction_loss = None 
        
        term1 = (torch.lgamma(torch.sum(aa, dim=-1)) -
                torch.lgamma(torch.sum(aa + xx, dim=-1))) #[n_batch]
        term2 = torch.sum(torch.where(xx > 0,
                            torch.lgamma(aa + xx) -
                            torch.lgamma(aa),
                            torch.zeros_like(xx)),
                            dim=-1) #[n_batch
        reconstruction_loss = term1 + term2 #[n_batch
        return reconstruction_loss


    def _get_inference_input(self, tensors):
        return dict(x=tensors[_CONSTANTS.X_KEY],y = tensors[_CONSTANTS.PROTEIN_EXP_KEY])

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        return dict(z=z)

    @auto_move_data
    def inference(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        x_ = x
        y_ = y

        q_m, q_v, z = self.z_encoder(x_, y_)
        return dict(z = z, q_m = q_m, q_v = q_v)

    @auto_move_data
    def generative(self, z) -> dict:

        rho, delta, rho_kl, delta_kl, theta  = self.decoder(z)

        return dict(rho = rho, delta = delta, rho_kl = rho_kl, delta_kl = delta_kl, theta = theta)
    
    def sample_from_posterior_z(
        self, 
        x: torch.Tensor,
        y: torch.Tensor,
        deterministic: bool = True,
        output_softmax_z: bool = True, 
    ):
        """
        sample from the posterior of latent space z
        """
        
        inference_out = self.inference(x,y)
        if deterministic: # average of the two means WITHOUT sampling
            z = inference_out["q_m"]
        else: # sampling 
            z = inference_out["z"]
        if output_softmax_z:
            generative_outputs = self.generative(z)
            z = generative_outputs["theta"]      
        return dict(z=z)
    
    @auto_move_data
    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the reconstruction loss for the given batch.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input)``
        y  
            tensor of values with shape ``(batch_size, n_input)``

        """
        inference_out = self.inference(x, y)
        z = inference_out["z"]
        gen_out = self.generative(z)
        theta = gen_out["theta"]
               
        rho = gen_out["rho"]
        log_aa_spliced = torch.clamp(torch.mm(theta, rho), -10, 10)
        aa_spliced = torch.exp(log_aa_spliced)
        
        delta = gen_out["delta"]
        log_aa_unspliced = torch.clamp(torch.mm(theta, rho + delta), -10, 10)
        aa_unspliced = torch.exp(log_aa_unspliced)
        
        reconstruction_loss_spliced = -self.dir_llik(x, aa_spliced)
        reconstruction_loss_unspliced = -self.dir_llik(y, aa_unspliced)
        return reconstruction_loss_spliced, reconstruction_loss_unspliced

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs, # this is important to include
        kl_weight=1.0,
        kl_weight_beta = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Aggregate the kl and likelihood to form the loss.

        """
        kl_weight_beta = self.kl_weight_beta
        x = tensors[_CONSTANTS.X_KEY]
        y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        q_m = inference_outputs["q_m"]
        q_v = inference_outputs["q_v"]
        
        rho_kl = generative_outputs["rho_kl"]
        delta_kl = generative_outputs["delta_kl"]
        
        # [batch_size]
        reconstruction_loss_spliced, reconstruction_loss_unspliced = self.get_reconstruction_loss(x, y)

        # KL Divergence for z [batch_size]
        mean = torch.zeros_like(q_m)
        scale = torch.ones_like(q_v)
        kl_divergence = kl(Normal(q_m, torch.sqrt(q_v)), Normal(mean, scale)).sum(
            dim=1
        )

        # suming over all the topics
        # kl_divergence for beta, rho_kl, tensor of torch.size([]) <- torch.sum([N_topics, N_genes])
        kl_divergence_beta = rho_kl + delta_kl
        kl_local = kl_divergence
        reconstruction_loss = reconstruction_loss_spliced + reconstruction_loss_unspliced
        loss = torch.mean(reconstruction_loss + kl_weight * kl_local) + kl_weight_beta * kl_divergence_beta/x.shape[1]
        
        return LossRecorder(loss, reconstruction_loss, kl_local,
                            reconstruction_loss_spliced=reconstruction_loss_spliced,
                            reconstruction_loss_unspliced=reconstruction_loss_unspliced, 
                            kl_beta = kl_divergence_beta, 
                            kl_rho = rho_kl, 
                            kl_delta = delta_kl)
