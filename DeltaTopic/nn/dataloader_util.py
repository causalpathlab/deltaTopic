import copy
import logging
from typing import Optional, Union, Dict, List
import anndata
import numpy as np
import torch
import h5py
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from anndata._core.sparse_dataset import SparseDataset

logger = logging.getLogger(__name__)

def get_from_registry(adata: anndata.AnnData, key: str) -> np.ndarray:
    """
    Returns the object in AnnData associated with the key in ``.uns['deltaTopic']['data_registry']``.

    Parameters
    ----------
    adata
        anndata object already setup with `setup_anndata()`
    key
        key of object to get from ``adata.uns['deltaTopic]['data_registry']``

    Returns
    -------
    The requested data

    Examples
    --------
    """
    data_loc = adata.uns["deltaTopic"]["data_registry"][key]
    attr_name, attr_key = data_loc["attr_name"], data_loc["attr_key"]

    data = getattr(adata, attr_name)
    if attr_key != "None":
        if isinstance(data, pd.DataFrame):
            data = data.loc[:, attr_key]
        else:
            data = data[attr_key]
    if isinstance(data, pd.Series):
        data = data.to_numpy().reshape(-1, 1)
    return data

class AnnTorchDataset(Dataset):
    """Extension of torch dataset to get tensors from anndata."""

    def __init__(
        self,
        adata: anndata.AnnData,
        getitem_tensors: Union[List[str], Dict[str, type]] = None,
    ):
        self.adata = adata
        self.attributes_and_types = None
        self.getitem_tensors = getitem_tensors
        self.setup_getitem()
        self.setup_data_attr()

    @property
    def registered_keys(self):
        """Returns the keys of the mappings in deltaTopic data registry."""
        return self.adata.uns["deltaTopic"]["data_registry"].keys()

    def setup_data_attr(self):
        """
        Sets data attribute.

        Reduces number of times anndata needs to be accessed
        """
        self.data = {
            key: get_from_registry(self.adata, key)
            for key, _ in self.attributes_and_types.items()
        }

    def setup_getitem(self):
        """
        Sets up the __getitem__ function used by Pytorch.
        """
        registered_keys = self.registered_keys
        getitem_tensors = self.getitem_tensors
        if isinstance(getitem_tensors, List):
            keys = getitem_tensors
            keys_to_type = {key: np.float32 for key in keys}
        elif isinstance(getitem_tensors, Dict):
            keys = getitem_tensors.keys()
            keys_to_type = getitem_tensors
        elif getitem_tensors is None:
            keys = registered_keys
            keys_to_type = {key: np.float32 for key in keys}
        else:
            raise ValueError(
                "getitem_tensors invalid type. Expected: List[str] or Dict[str, type] or None"
            )
        for key in keys:
            assert (
                key in registered_keys
            ), "{} not in anndata.uns['deltaTopic']['data_registry']".format(key)

        self.attributes_and_types = keys_to_type

    def __getitem__(self, idx: List[int]) -> Dict[str, torch.Tensor]:
        """Get tensors in dictionary from anndata at idx."""
        data_numpy = {}
        for key, dtype in self.attributes_and_types.items():
            data = self.data[key]
            # for backed anndata
            if isinstance(data, h5py.Dataset) or isinstance(data, SparseDataset):
                # need to sort idxs for h5py datasets
                if hasattr(idx, "shape"):
                    argsort = np.argsort(idx)
                else:
                    argsort = idx
                data = data[idx[argsort]]
                # now unsort
                i = np.empty_like(argsort)
                i[argsort] = np.arange(argsort.size)
                # this unsorts it
                idx = i

            if isinstance(data, np.ndarray):
                data_numpy[key] = data[idx].astype(dtype)
            elif isinstance(data, pd.DataFrame):
                data_numpy[key] = data.iloc[idx, :].to_numpy().astype(dtype)
            else:
                data_numpy[key] = data[idx].toarray().astype(dtype)

        return data_numpy

    def get_data(self, scvi_data_key):
        tensors = self.__getitem__(idx=[i for i in range(self.__len__())])
        return tensors[scvi_data_key]

    def __len__(self):
        return self.adata.shape[0]

class BatchSampler(torch.utils.data.sampler.Sampler):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.

    Parameters
    ----------
    indices
        list of indices to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """

    def __init__(
        self,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: Union[bool, int] = False,
    ):
        self.indices = indices
        self.n_obs = len(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if drop_last > batch_size:
            raise ValueError(
                "drop_last can't be greater than batch_size. "
                + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
            )

        last_batch_len = self.n_obs % self.batch_size
        if (drop_last is True) or (last_batch_len < drop_last):
            drop_last_n = last_batch_len
        elif (drop_last is False) or (last_batch_len >= drop_last):
            drop_last_n = 0
        else:
            raise ValueError("Invalid input for drop_last param. Must be bool or int.")

        self.drop_last_n = drop_last_n

    def __iter__(self):
        if self.shuffle is True:
            idx = torch.randperm(self.n_obs).tolist()
        else:
            idx = torch.arange(self.n_obs).tolist()

        if self.drop_last_n != 0:
            idx = idx[: -self.drop_last_n]

        data_iter = iter(
            [
                self.indices[idx[i : i + self.batch_size]]
                for i in range(0, len(idx), self.batch_size)
            ]
        )
        return data_iter

    def __len__(self):
        from math import ceil

        if self.drop_last_n != 0:
            length = self.n_obs // self.batch_size
        else:
            length = ceil(self.n_obs / self.batch_size)
        return length

class AnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.

    Parameters
    ----------
    adata
        An anndata objects
    shuffle
        Whether the data should be shuffled
    indices
        The indices of the observations in the adata to load
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (`adata.uns["deltaTopic"]`)
        and value equal to desired numpy loading type (later made into torch tensor).
        If `None`, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        shuffle=False,
        indices=None,
        batch_size=128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ):

        if "deltaTopic" not in adata.uns.keys():
            raise ValueError("Please run setup_anndata() on your anndata object first.")

        if data_and_attributes is not None:
            data_registry = adata.uns["deltaTopic"]["data_registry"]
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(
                        "{} required for model but not included when setup_anndata was run".format(
                            key
                        )
                    )

        self.dataset = AnnTorchDataset(adata, getitem_tensors=data_and_attributes)

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        super().__init__(self.dataset, **self.data_loader_kwargs)
