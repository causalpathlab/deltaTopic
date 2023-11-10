from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union, Callable, Any
from collections.abc import Mapping, Sequence
from functools import wraps
import torch
import torch.nn as nn
from torch.nn import Module

def auto_move_data(fn: Callable) -> Callable:
    """
    Decorator for :class:`~torch.nn.Module` methods to move data to correct device.

    Input arguments are moved automatically to the correct device.
    It has no effect if applied to a method of an object that is not an instance of
    :class:`~torch.nn.Module` and is typically applied to ``__call__``
    or ``forward``.

    Parameters
    ----------
    fn
        A nn.Module method for which the arguments should be moved to the device
        the parameters are on.
    """

    @wraps(fn)
    def auto_transfer_args(self, *args, **kwargs):
        if not isinstance(self, Module):
            return fn(self, *args, **kwargs)

        # decorator only necessary after training
        if self.training:
            return fn(self, *args, **kwargs)

        device = list(set(p.device for p in self.parameters()))
        if len(device) > 1:
            raise RuntimeError("Model tensors on multiple devices.")
        else:
            device = device[0]
        args = _move_data_to_device(args, device)
        kwargs = _move_data_to_device(kwargs, device)
        return fn(self, *args, **kwargs)

    return auto_transfer_args

def _move_data_to_device(batch: Any, device: torch.device):
    """
    Transfers a collection of data to the given device.

    Any object that defines a method ``to(device)`` will be moved and all other objects
    in the collection will be left untouched.

    Parameters
    ----------
    batch
        A tensor or collection of tensors or anything that has a method `.to(...)`.
        See :func:`apply_to_collection` for a list of supported collection types.
    device
        The device to which the data should be moved

    Returns
    -------
        The same collection but with all contained tensors residing on the new device.
    """

    def batch_to(data):
        kwargs = dict(non_blocking=True) if isinstance(data, torch.Tensor) else {}
        return data.to(device, **kwargs)

    return _apply_to_collection(batch, dtype=torch.Tensor, function=batch_to)

def _apply_to_collection(
    data: Any, dtype: Union[type, tuple], function: Callable, *args, **kwargs
) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.

    Parameters
    ----------
    data
        The collection to apply the function to
    dtype
        The given function will be applied to all elements of this dtype
    function
        The function to apply
    *args
        positional arguments (will be forwarded to calls of ``function``)
    **kwargs
        keyword arguments (will be forwarded to calls of ``function``)

    Returns
    -------
    The resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    elif isinstance(data, Mapping):
        return elem_type(
            {
                k: _apply_to_collection(v, dtype, function, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(
            *(_apply_to_collection(d, dtype, function, *args, **kwargs) for d in data)
        )
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type(
            [_apply_to_collection(d, dtype, function, *args, **kwargs) for d in data]
        )

    # data is neither of dtype, nor a collection
    return data

class LossRecorder:
    """
    Loss signature for models.

    This class provides an organized way to record the model loss, as well as
    the components of the ELBO. This may also be used in MLE, MAP, EM methods.
    The loss is used for backpropagation during inference. The other parameters
    are used for logging/early stopping during inference.

    Parameters
    ----------
    loss
        Tensor with loss for minibatch. Should be one dimensional with one value.
        Note that loss should be a :class:`~torch.Tensor` and not the result of `.item()`.
    reconstruction_loss
        Reconstruction loss for each observation in the minibatch.
    kl_local
        KL divergence associated with each observation in the minibatch.
    kl_global
        Global kl divergence term. Should be one dimensional with one value.
    **kwargs
        Additional metrics can be passed as keyword arguments and will
        be available as attributes of the object.
    """

    def __init__(
        self,
        loss: Union[Dict[str, torch.Tensor], torch.Tensor],
        reconstruction_loss: Union[
            Dict[str, torch.Tensor], torch.Tensor
        ] = torch.Tensor([0]),
        kl_local: Union[Dict[str, torch.Tensor], torch.Tensor] = torch.Tensor([0]),
        kl_global: Union[Dict[str, torch.Tensor], torch.Tensor] = torch.Tensor([0]),
        **kwargs,
    ):
        self._loss = loss if isinstance(loss, dict) else dict(loss=loss)
        self._reconstruction_loss = (
            reconstruction_loss
            if isinstance(reconstruction_loss, dict)
            else dict(reconstruction_loss=reconstruction_loss)
        )
        self._kl_local = (
            kl_local if isinstance(kl_local, dict) else dict(kl_local=kl_local)
        )
        self._kl_global = (
            kl_global if isinstance(kl_global, dict) else dict(kl_global=kl_global)
        )
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _get_dict_sum(dictionary):
        total = 0.0
        for value in dictionary.values():
            total += value
        return total

    @property
    def loss(self) -> torch.Tensor:
        return self._get_dict_sum(self._loss)

    @property
    def reconstruction_loss(self) -> torch.Tensor:
        return self._get_dict_sum(self._reconstruction_loss)

    @property
    def kl_local(self) -> torch.Tensor:
        return self._get_dict_sum(self._kl_local)

    @property
    def kl_global(self) -> torch.Tensor:
        return self._get_dict_sum(self._kl_global)

class BaseModuleClass(nn.Module):
    """Abstract class for deltaTopic modules."""

    def __init__(
        self,
    ):
        super().__init__()

    @auto_move_data
    def forward(
        self,
        tensors,
        get_inference_input_kwargs: Optional[dict] = None,
        get_generative_input_kwargs: Optional[dict] = None,
        inference_kwargs: Optional[dict] = None,
        generative_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        compute_loss=True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, LossRecorder],
    ]:
        """
        Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for `_get_inference_input()`
        get_generative_input_kwargs
            Keyword args for `_get_generative_input()`
        inference_kwargs
            Keyword args for `inference()`
        generative_kwargs
            Keyword args for `generative()`
        loss_kwargs
            Keyword args for `loss()`
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        inference_kwargs = _get_dict_if_none(inference_kwargs)
        generative_kwargs = _get_dict_if_none(generative_kwargs)
        loss_kwargs = _get_dict_if_none(loss_kwargs)
        get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
        get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

        inference_inputs = self._get_inference_input(
            tensors, **get_inference_input_kwargs
        )
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)
        if compute_loss:
            losses = self.loss(
                tensors, inference_outputs, generative_outputs, **loss_kwargs
            )
            return inference_outputs, generative_outputs, losses
        else:
            return inference_outputs, generative_outputs

    @abstractmethod
    def _get_inference_input(self, tensors: Dict[str, torch.Tensor], **kwargs):
        """Parse tensors dictionary for inference related values."""

    @abstractmethod
    def _get_generative_input(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """Parse tensors dictionary for generative related values."""

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> dict:
        """
        Run the inference (recognition) model.

        In the case of variational inference, this function will perform steps related to
        computing variational distribution parameters. In a VAE, this will involve running
        data through encoder networks.

        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """

    @abstractmethod
    def generative(self, *args, **kwargs) -> dict:
        """
        Run the generative model.

        This function should return the parameters associated with the likelihood of the data.
        This is typically written as :math:`p(x|z)`.

        This function should return a dictionary with str keys and :class:`~torch.Tensor` values.
        """

    @abstractmethod
    def loss(self, *args, **kwargs) -> LossRecorder:
        """
        Compute the loss for a minibatch of data.

        This function uses the outputs of the inference and generative functions to compute
        a loss. This many optionally include other penalty terms, which should be computed here.

        This function should return an object of type :class:`~module.base.LossRecorder`.
        """

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Generate samples from the learned model."""

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param
