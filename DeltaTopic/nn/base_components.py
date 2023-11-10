import collections
from typing import Iterable, List
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

def identity(x):
    return x

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
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
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
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

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

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
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
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

        #if mask is None:
            #print("No mask input, use all fully connected layers")

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

class DeltaTopicEncoder(nn.Module):
    """
    A two-headed encoder that maps the two inputs into a shared latent space through a stack of individual and shared fully-connected layers.

    Parameters
    ----------
    n_input_list
        List of the dimension of two input tensors
    n_output
        The dimensionality of the output
    mask
        The mask to apply to the first layer (experimental)
    mask_first
        Transpose the mask if set to false (experimental)
    n_hidden
        The number of nodes per hidden layer
    n_layers_individual
        The number of fully-connected hidden layers for the individual encoder
    n_layers_shared
        The number of fully-connected hidden layers for the shared encoder
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    log_variational
        Whether to apply log(1+x) transformation to the input
    combine_method
        the method to combine the two latent space, either "add" or "concatenate"
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
        combine_method: str = "add",
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
        '''
        Forward pass for DeltaTopicEncoder
        
        Parameters
        ----------
        x
            First input tensor, e.g., spliced RNA count
        y   
            Second input tensorm, e.g., unsplice RNA count
    
        '''
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
    
class DeltaTopicDecoder(nn.Module):
    """
    Decoder network for DeltaTopic, a generative network with spike and slab prior for rho and delta.

    Parameters
    ----------
    n_input
        The dimensionality of the input
    n_output
        The dimensionality of the output
    pip0_rho
        posterior inclusion probability prior for rho
    pip0_delta
        posterior inclusion probability prior for delta
    v0_rho
        variance for rho slab
    vo_delta
        variance for delta slab            
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        pip0_rho = 0.1,
        pip0_delta = 0.1,
        v0_rho = 1,
        v0_delta = 1,
    ):
        super().__init__()
        self.n_input = n_input # topics
        self.n_output = n_output # genes
        
        # gene-level bias, shared across all topics 
        self.bias_gene = nn.Parameter(torch.zeros(1, n_output))
        # for shared effect（rho）
        self.logit_0_rho = nn.Parameter(torch.logit(torch.ones(1)* pip0_rho, eps=1e-6), requires_grad = False)
        self.lnvar_0_rho = nn.Parameter(torch.log(torch.ones(1) * v0_rho), requires_grad = False)
        self.slab_mean_rho = nn.Parameter(torch.randn(n_input, n_output) * torch.sqrt(torch.ones(1) * v0_rho))
        self.slab_lnvar_rho = nn.Parameter(torch.ones(n_input, n_output) * torch.log(torch.ones(1) * v0_rho))
        self.spike_logit_rho = nn.Parameter(torch.zeros(n_input, n_output) * self.logit_0_rho)

        # delta effect
        self.logit_0_delta = nn.Parameter(torch.logit(torch.ones(1)*pip0_delta, eps=1e-6), requires_grad = False)
        self.lnvar_0_delta = nn.Parameter(torch.log(torch.ones(1)*v0_delta), requires_grad = False)
        self.slab_mean_delta = nn.Parameter(torch.randn(n_input, n_output) * torch.sqrt(torch.ones(1) * v0_delta))
        self.slab_lnvar_delta = nn.Parameter(torch.ones(n_input, n_output) * torch.log(torch.ones(1) * v0_delta))
        self.spike_logit_delta = nn.Parameter(torch.zeros(n_input, n_output) * self.logit_0_delta)
         
        # Log softmax operations
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    
    def forward(
        self,
        z: torch.Tensor,
    ):
        '''
        forward pass for DeltaTopicDecoder
        
        Parameters
        ----------
        z
            the input the of the decoder, e.g., latent variable from DeltaTopicEncoder
        '''
        theta = self.soft_max(z)
        rho = self.get_beta(self.spike_logit_rho, self.slab_mean_rho, self.slab_lnvar_rho, self.bias_gene)
        rho_kl = self.sparse_kl_loss(self.logit_0_rho, self.lnvar_0_rho, self.spike_logit_rho, self.slab_mean_rho, self.slab_lnvar_rho)
        
        delta = self.get_beta(self.spike_logit_delta, self.slab_mean_delta, self.slab_lnvar_delta, self.bias_gene)
        delta_kl = self.sparse_kl_loss(self.logit_0_delta, self.lnvar_0_delta, self.spike_logit_delta, self.slab_mean_delta, self.slab_lnvar_delta)
        
        return rho, delta, rho_kl, delta_kl, theta 
    
    def get_rho_delta(
        self,
    ):
        '''
        Helper function to get rho and delta
        '''
        rho = self.get_beta(self.spike_logit_rho, self.slab_mean_rho, self.slab_lnvar_rho, self.bias_gene)
        delta = self.get_beta(self.spike_logit_delta, self.slab_mean_delta, self.slab_lnvar_delta, self.bias_gene)
        
        return rho, delta 
    
    def get_beta(self, 
        spike_logit: torch.Tensor,
        slab_mean: torch.Tensor, 
        slab_lnvar: torch.Tensor,
        bias_gene: torch.Tensor, 
    ): 
        '''
        Get a spike and slab sample using reparameterization trick
        
        Parameters
        ----------
        spike_logit
            logit of spike probability
        slab_mean
            mean of slab
        slab_lnvar
            log variance of slab
        bias_gene
            gene-level bias
        '''
        pip = torch.sigmoid(spike_logit)
        mean = slab_mean * pip
        var = pip * (1 - pip) * torch.square(slab_mean)
        var = var + pip * torch.exp(slab_lnvar)
        eps = torch.randn_like(var)

        return mean + eps * torch.sqrt(var) - bias_gene
    
    def soft_max(self, 
                 z: torch.Tensor,
    ):  
        '''
        softmax function
        
        Parameters
        ----------
        z
            input tensor
        '''  
        return torch.exp(self.log_softmax(z))
    
    def sparse_kl_loss(
        self,
        logit_0, 
        lnvar_0,
        spike_logit,
        slab_mean,
        slab_lnvar,
    ):  
        '''
        Compute KL divergence between spike and slab piors and posteriors
        
        Parameters
        ----------
        logit_0
            logit of spike probability (prior)
        lnvar_0
            log variance of slab (prior)
        spike_logit
            logit of spike probability (posterior)
        slab_mean
            mean of slab (posterior)            
        slab_lnvar
            log variance of slab (posterior)        
        '''                         
        ## PIP KL between p and p0
        ## p * ln(p / p0) + (1-p) * ln(1-p/1-p0)
        ## = p * ln(p / 1-p) + ln(1-p) +
        ##   p * ln(1-p0 / p0) - ln(1-p0)
        ## = sigmoid(logit) * logit - softplus(logit)
        ##   - sigmoid(logit) * logit0 + softplus(logit0)
        pip_hat = torch.sigmoid(spike_logit)
        kl_pip_1 = pip_hat * (spike_logit - logit_0)
        kl_pip = kl_pip_1 - nn.functional.softplus(spike_logit) + nn.functional.softplus(logit_0)
        ## Gaussian KL between N(μ,ν) and N(0, v0) 
        sq_term = torch.exp(-lnvar_0) * (torch.square(slab_mean) + torch.exp(slab_lnvar))
        kl_g = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)
        ## Combine both logit and Gaussian KL
        return torch.sum(kl_pip + pip_hat * kl_g) # return a number sum over [N_topics, N_genes]

class BALSAMDecoder(nn.Module):
    """
    Decoder network for BALSAM model, a generative network with spike and slab prior for beta parameter.
    
    Parameters
    ----------
    n_input: int
        The input dimension of the decoder, e.g., number of topics.
    n_output: int
        The output dimension of decoder, e.g., tumber of genes.
    pip0: float
        The prior probability of spike in the spike and slab prior.
    v0: float
        The prior variance of slab in the spike and slab prior.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        pip0 = 0.1,
        v0 = 1,
    ):
        super().__init__()
        self.n_input = n_input # topics
        self.n_output = n_output # genes
        
        # for shared effect（rho）
        self.logit_0 = nn.Parameter(torch.logit(torch.ones(1)* pip0, eps=1e-6), requires_grad = False)
        self.lnvar_0 = nn.Parameter(torch.log(torch.ones(1) * v0), requires_grad = False)
        self.bias_d = nn.Parameter(torch.zeros(1, n_output))
        self.slab_mean = nn.Parameter(torch.randn(n_input, n_output) * torch.sqrt(torch.ones(1) * v0))
        self.slab_lnvar = nn.Parameter(torch.ones(n_input, n_output) * torch.log(torch.ones(1) * v0))
        self.spike_logit = nn.Parameter(torch.zeros(n_input, n_output) * self.logit_0)

        # Log softmax operations
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    
    def forward(
        self,
        z: torch.Tensor,
    ):
        '''
        forward pass of the decoder network.
        
        Parameters
        ----------
        z: torch.Tensor
            The input tensor of the decoder, e.g., the latent representation from the encoder.
        '''
        theta = self.soft_max(z)
        rho = self.get_beta(self.spike_logit, self.slab_mean, self.slab_lnvar, self.bias_d)
        rho_kl = self.sparse_kl_loss(self.logit_0, self.lnvar_0, self.spike_logit, self.slab_mean, self.slab_lnvar)
        
        return rho, rho_kl, theta 
    
    def get_rho(
        self,
    ):
        '''
        A helper function to get rho.
        '''
        rho = self.get_beta(self.spike_logit, self.slab_mean, self.slab_lnvar, self.bias_d)
        
        return rho
    
    def get_beta(self, 
        spike_logit: torch.Tensor,
        slab_mean: torch.Tensor, 
        slab_lnvar: torch.Tensor,
        bias_d: torch.Tensor,
    ): 
        '''
        Sample beta using the repameterization trick
        
        Parameters
        ----------
        spike_logit: torch.Tensor
            The logit of spike probability.
        slab_mean: torch.Tensor
            The mean of slab.
        slab_lnvar: torch.Tensor
            The log variance of slab.
        bias_d: torch.Tensor
            The bias term in the GLM model.    
        '''
        pip = torch.sigmoid(spike_logit)
        mean = slab_mean * pip
        var = pip * (1 - pip) * torch.square(slab_mean)
        var = var + pip * torch.exp(slab_lnvar)
        eps = torch.randn_like(var)

        return mean + eps * torch.sqrt(var) - bias_d
    
    def soft_max(self, 
                 z: torch.Tensor,
    ):  
        ''' 
        softmax function
        
        Parameters
        ----------
        z: torch.Tensor
            The input tensor.
        ''' 
        return torch.exp(self.log_softmax(z))
    
    def sparse_kl_loss(
        self,
        logit_0, 
        lnvar_0,
        spike_logit,
        slab_mean,
        slab_lnvar,
    ):  
        '''
        Compute the KL divergence between spike and slab prior and the posterior.
        
        Parameters
        ----------
        logit_0: torch.Tensor
            The logit prior of spike probability.
        lnvar_0: torch.Tensor
            The log variance prior of slab. 
        spike_logit: torch.Tensor
            The logit of spike probability.
        slab_mean: torch.Tensor
            The mean of slab.
        slab_lnvar: torch.Tensor
            The log variance of slab.        
        '''                         
        ## PIP KL between p and p0
        ## p * ln(p / p0) + (1-p) * ln(1-p/1-p0)
        ## = p * ln(p / 1-p) + ln(1-p) +
        ##   p * ln(1-p0 / p0) - ln(1-p0)
        ## = sigmoid(logit) * logit - softplus(logit)
        ##   - sigmoid(logit) * logit0 + softplus(logit0)
        pip_hat = torch.sigmoid(spike_logit)
        kl_pip_1 = pip_hat * (spike_logit - logit_0)
        kl_pip = kl_pip_1 - nn.functional.softplus(spike_logit) + nn.functional.softplus(logit_0)
        ## Gaussian KL between N(μ,ν) and N(0, v0) 
        sq_term = torch.exp(-lnvar_0) * (torch.square(slab_mean) + torch.exp(slab_lnvar))
        kl_g = -0.5 * (1. + slab_lnvar - lnvar_0 - sq_term)
        ## Combine both logit and Gaussian KL
        return torch.sum(kl_pip + pip_hat * kl_g) # return a number sum over [N_topics, N_genes]
                              
class BALSAMEncoder(nn.Module):
    """
    Encoder for BALSAM model, encodes the input data into a latent topic representation.
    
    Parameters
    ----------
    n_input: int
        The number of input features.
    n_output: int
        The number of output features.
    n_hidden: int
        The number of hidden units.
    n_layers_individual: int
        The number of layers in the network.
    use_batch_norm: bool
        Whether to use batch normalization.
    log_variational: bool
        Whether to apply log(1+x) to the input.         
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers_individual: int = 3,
        use_batch_norm: bool = True,
        log_variational: bool = True,
    ):
        super().__init__()
        self.log_variational = log_variational
        
        self.encoder = FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_cat_list=None,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=0,
                    use_batch_norm = use_batch_norm
                )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        '''
        forward pass of the encoder
        '''
        if self.log_variational:
            x_ = torch.log(1 + x)
    
        q = self.encoder(x_, *cat_list)    
        q_m = self.mean_encoder(q)
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -4.0, 4.0)/2.)
        latent = reparameterize_gaussian(q_m, q_v)

        return q_m, q_v, latent