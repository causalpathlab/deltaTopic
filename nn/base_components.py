# -*- coding: utf-8 -*-
"""base components"""
import collections
from typing import Iterable, List
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
from scvi.nn import FCLayers
import torch.nn.functional as F
from scvi.nn import one_hot

torch.backends.cudnn.benchmark = True

def identity(x):
    return x

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class MaskedLinear(nn.Linear):
    """ 
    same as Linear except has a configurable mask on the weights 
    """
    
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', mask)
        
    def forward(self, input):
        #mask = Variable(self.mask, requires_grad=False)
        if self.bias is None:
            return F.linear(input, self.weight*self.mask)
        else:
            return F.linear(input, self.weight*self.mask, self.bias)

class MaskedLinearLayers(FCLayers):
    """
    This incorporates the one-hot encoding for for category input.
    A helper class to build Masked Linear layers compatible with FClayer
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    mask
        The mask, should be dimension n_out * n_in
    mask_first
        wheather mask linear layer should be before or after fully-connected layers, default is true;
        False is useful to construct an decoder with the oposite strucutre (mask linear after fully connected)
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self, 
        n_in: int,
        n_out: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU
        ):
            
        super().__init__(
            n_in=n_in,
            n_out=n_out,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_activation=use_activation,
            bias=bias,
            inject_covariates=inject_covariates,
            activation_fn=activation_fn
            )

        self.mask = mask ## out_features, in_features

        if mask is None:
            print("No mask input, use all fully connected layers")

        
        if mask is not None:
            if mask_first:
                layers_dim = [n_in] + [mask.shape[0]] + (n_layers - 1) * [n_hidden] + [n_out]
            else:
                layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [mask.shape[0]] + [n_out]
        else:    
            layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)

        # concatnat one hot encoding to mask if available
        if cat_dim>0:
            mask_input = torch.cat((self.mask, torch.ones(cat_dim, self.mask.shape[1])), dim=0)
        else:
            mask_input = self.mask        

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )
        if mask is not None:
            if mask_first:
                # change the first layer to be MaskedLinear
                self.fc_layers[0] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[0] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[1],
                                                mask_input,
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )
            else:
                # change the last layer to be MaskedLinear
                self.fc_layers[-1] = nn.Sequential(
                                            MaskedLinear(
                                                layers_dim[-2] + cat_dim * self.inject_into_layer(0),
                                                layers_dim[-1],
                                                torch.transpose(mask_input,0,1),
                                                bias=bias,
                                            ),
                                            # non-default params come from defaults in original Tensorflow implementation
                                            nn.BatchNorm1d(layers_dim[-1], momentum=0.01, eps=0.001)
                                            if use_batch_norm
                                            else None,
                                            nn.LayerNorm(layers_dim[-1], elementwise_affine=False)
                                            if use_layer_norm
                                            else None,
                                            activation_fn() if use_activation else None,
                                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                                            )


    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.
        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor
        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if (isinstance(layer, nn.Linear) or isinstance(layer, MaskedLinear)) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x

class MultiMaskedEncoder(nn.Module):
    """
    Maksed latent encoder with two input heads --> shared latent space
    Options to incorporate categorical variables as one-hot vectors
    """
    def __init__(
        self,
        n_heads: int,
        n_input_list: List[int],
        n_output: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                MaskedLinearLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                )
                for i in range(n_heads)
            ]
        )

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, head_id: int, *cat_list: int):
        q = self.encoders[head_id](x, *cat_list)
        
        q = self.encoder_shared(q, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -4.0, 4.0)/2.)
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent

class DeltaETMDecoder(nn.Module):
    """
    The decoder for DeltaETM model
    - Model the topic specific post-transcription factor for each gene between spliced and unplisced transcripts 

    Testing script:
import torch
from nn.base_components import DeltaETMDecoder
test_decoder = DeltaETMDecoder(n_input=10, n_output=100)
input = torch.randn(2, 10) # batch_size 2, 10 LVs
output = test_decoder(input, 0)

log_beta = test_decoder.beta(test_decoder.rho.expand([test_decoder.n_input,-1]))
hh = test_decoder.hid(input)
pr = torch.mm(torch.exp(hh),torch.exp(log_beta))

    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        # global parameters 
        self.rho = nn.Parameter(torch.randn(self.n_input, self.n_output)) # topic-by-gene matrix, shared effect
        self.delta= nn.Parameter(torch.randn(self.n_input,self.n_output)) # topic-by-gene matrix, differnces
       
        # Log softmax operations
        self.logsftm_rho = nn.LogSoftmax(dim=-1) 
        self.logsftm_delta = nn.LogSoftmax(dim=-1) 
        self.hid = nn.LogSoftmax(dim=-1) # to topics loadings on each gene

    def forward(
        self,
        z: torch.Tensor,
        dataset_id: int, # 0 for spliced count and 1 for unspliced count, The order is IMPORTANT
    ):
        
        self.log_softmax_rho = self.logsftm_rho(self.rho)
        self.log_softmax_delta = self.logsftm_delta(self.delta)
        
        # The order matters here, the spliced needs to be passed as adata1
        if dataset_id == 0: # spliced count 
            log_beta = self.log_softmax_rho + self.log_softmax_delta
        elif dataset_id == 1: # unplisced count
            log_beta =  self.log_softmax_rho
        else:
            raise ValueError("DeltaETMDecoder dataset_id should be 0 (spliced) or 1 (unspliced)")
        
        hh = torch.exp(self.hid(z))     

        return torch.mm(hh,torch.exp(log_beta)), hh, self.log_softmax_rho, self.log_softmax_delta

class TotalMultiMaskedEncoder(nn.Module):
    """
    Maksed latent encoder with two input heads --> one shared latent space
    Options to incorporate categorical variables as one-hot vectors
    """
    def __init__(
        self,
        n_input_list: List[int],
        n_output: int,
        mask: torch.Tensor = None,
        mask_first: bool = True,
        n_hidden: int = 128,
        n_layers_individual: int = 1,
        n_layers_shared: int = 2,
        n_cat_list: Iterable[int] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        log_variational: bool = True,
        combine_method: str = "concat",
    ):
        super().__init__()
        self.log_variational = log_variational
        self.combine_method = combine_method
        self.encoders = ModuleList(
            [
                MaskedLinearLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    mask=mask,
                    mask_first=mask_first,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                )
                for i in range(len(n_input_list))
            ]
        )
        if self.combine_method == 'concat':
            dim_encoder_shared = n_hidden + n_hidden
        elif self.combine_method == 'add':
            dim_encoder_shared = n_hidden
        else:
            raise ValueError("combine method must choose from concat or add") 
        
        self.encoder_shared = FCLayers(
            n_in=dim_encoder_shared,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, y: torch.Tensor, *cat_list: int):
        
        if self.log_variational:
            x_ = torch.log(1 + x)
            y_ = torch.log(1 + y)
        
        q_x = self.encoders[0](x_, *cat_list)
        q_y = self.encoders[1](y_, *cat_list)
        
        if self.combine_method == 'concat':
            q = torch.cat([q_x, q_y], dim=-1)
        elif self.combine_method == 'add':
            q = (q_x + q_y)/2.
        else:
            raise ValueError("combine method must choose from concat or add")  
         
        q = self.encoder_shared(q, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -4.0, 4.0)/2.)
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent

class TotalDeltaETMDecoder(nn.Module):
    """
    The decoder for DeltaETM model
    - Model the topic specific post-transcription factor for each gene between spliced and unplisced transcripts 

    Testing script:
import torch
from nn.base_components import DeltaETMDecoder
test_decoder = DeltaETMDecoder(n_input=10, n_output=100)
input = torch.randn(2, 10) # batch_size 2, 10 LVs
output = test_decoder(input, 0)

log_beta = test_decoder.beta(test_decoder.rho.expand([test_decoder.n_input,-1]))
hh = test_decoder.hid(input)
pr = torch.mm(torch.exp(hh),torch.exp(log_beta))

    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        # global parameters 
        self.rho = nn.Parameter(torch.randn(self.n_input, self.n_output)) # topic-by-gene matrix, shared effect
        self.delta= nn.Parameter(torch.randn(self.n_input,self.n_output)) # topic-by-gene matrix, differnces
       
        # Log softmax operations
        self.logsftm_rho = nn.LogSoftmax(dim=-1) 
        self.logsftm_delta = nn.LogSoftmax(dim=-1) 
        self.hid = nn.LogSoftmax(dim=-1) # to topics loadings on each gene

    def forward(
        self,
        z: torch.Tensor,
    ):
        
        self.log_softmax_rho = self.logsftm_rho(self.rho)
        self.log_softmax_delta = self.logsftm_delta(self.delta)
        

        log_beta_spliced = self.log_softmax_rho + self.log_softmax_delta
        log_beta_unspliced =  self.log_softmax_rho
        hh = torch.exp(self.hid(z))     
        # torch.mm(hh,torch.exp(log_beta))
        return log_beta_spliced, log_beta_unspliced, hh, self.log_softmax_rho, self.log_softmax_delta
        



