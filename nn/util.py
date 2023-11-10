# utilies for data handling and builing deltaTopic model with spliced and unspliced
import torch
import os
import pickle
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import anndata
from anndata import AnnData, read
from math import ceil, floor
from typing import Optional, Union, Literal, Sequence, Tuple, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

from DeltaTopic.nn.dataloader_util import AnnDataLoader

logger = logging.getLogger(__name__)

def _get_var_names_from_setup_anndata(adata):
    """Gets var names by checking if using raw."""
    var_names = adata.var_names
    return var_names

def parse_use_gpu_arg(
    use_gpu: Optional[Union[str, int, bool]] = None,
    return_device=True,
):
    """
    Parses the use_gpu arg in codebase.

    Returned gpus are is compatible with PytorchLightning's gpus arg.
    If return_device is True, will also return the device.

    Parameters
    ----------
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    return_device
        If True, will return the torch.device of use_gpu.
    """
    gpu_available = torch.cuda.is_available()
    if (use_gpu is None and not gpu_available) or (use_gpu is False):
        gpus = 0
        device = torch.device("cpu")
    elif (use_gpu is None and gpu_available) or (use_gpu is True):
        current = torch.cuda.current_device()
        device = torch.device(current)
        gpus = [current]
    elif isinstance(use_gpu, int):
        device = torch.device(use_gpu)
        gpus = [use_gpu]
    elif isinstance(use_gpu, str):
        device = torch.device(use_gpu)
        # changes "cuda:0" to "0,"
        gpus = use_gpu.split(":")[-1] + ","
    else:
        raise ValueError("use_gpu argument not understood.")

    if return_device:
        return gpus, device
    else:
        return gpus

def validate_data_split(
    n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val

def _setup_unspliced_expression(
    adata, unspliced_obsm_key, unspliced_names_uns_key, batch_key
):
    assert (
        unspliced_obsm_key in adata.obsm.keys()
    ), "{} is not a valid key in adata.obsm".format(unspliced_obsm_key)

    logger.info(
        "Using protein expression from adata.obsm['{}']".format(
            unspliced_obsm_key
        )
    )
    pro_exp = adata.obsm[unspliced_obsm_key]
    
    # setup protein names
    if unspliced_names_uns_key is None and isinstance(
        adata.obsm[unspliced_obsm_key], pd.DataFrame
    ):
        logger.info(
            "Using protein names from columns of adata.obsm['{}']".format(
                unspliced_obsm_key
            )
        )
        unscplied_names = list(adata.obsm[unspliced_obsm_key].columns)
    elif unspliced_names_uns_key is not None:
        logger.info(
            "Using unspliced names from adata.uns['{}']".format(unspliced_names_uns_key)
        )
        unscplied_names = adata.uns[unspliced_names_uns_key]
    else:
        logger.info("Generating sequential protein names")
        unscplied_names = np.arange(adata.obsm[unspliced_obsm_key].shape[1])

    adata.uns["deltaTopic"]["unscplied_names"] = unscplied_names

    return unspliced_obsm_key

def _setup_x(adata, layer):
    if layer is not None:
        assert (
            layer in adata.layers.keys()
        ), "{} is not a valid key in adata.layers".format(layer)
        logger.info('Using data from adata.layers["{}"]'.format(layer))
        x_loc = "layers"
        x_key = layer
        x = adata.layers[x_key]
    else:
        logger.info("Using data from adata.X")
        x_loc = "X"
        x_key = "None"
        x = adata.X

    return x_loc, x_key

def _register_anndata(adata, data_registry_dict: Dict[str, Tuple[str, str]]):
    adata.uns["deltaTopic"]["data_registry"] = data_registry_dict.copy()

def setup_anndata(
    adata: anndata.AnnData,
    batch_key: Optional[str] = None,
    layer: Optional[str] = None,
    unspliced_obsm_key: Optional[str] = None,
    unspliced_names_uns_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:

    if copy:
        adata = adata.copy()

    if adata.is_view:
        raise ValueError(
            "Please run `adata = adata.copy()` or use the copy option in this function."
        )

    adata.uns["deltaTopic"] = {}
    x_loc, x_key = _setup_x(adata, layer)

    data_registry = {
        _CONSTANTS.X_KEY: {"attr_name": x_loc, "attr_key": x_key},
    }

    if unspliced_obsm_key is not None:
        unspliced_obsm_key = _setup_unspliced_expression(
            adata, unspliced_obsm_key, unspliced_names_uns_key, batch_key
        )
        data_registry[_CONSTANTS.PROTEIN_EXP_KEY] = {
            "attr_name": "obsm",
            "attr_key": unspliced_obsm_key,
        }

    # add the data_registry to anndata
    _register_anndata(adata, data_registry_dict=data_registry)
    logger.debug("Registered keys:{}".format(list(data_registry.keys())))

    logger.info("Please do not further modify adata until model is trained.")

    if copy:
        return adata
    
class _CONSTANTS:
    X_KEY = "X"
    BATCH_KEY = "batch_indices"
    LOCAL_L_MEAN_KEY = "local_l_mean"
    LOCAL_L_VAR_KEY = "local_l_var"
    LABELS_KEY = "labels"
    PROTEIN_EXP_KEY = "unspliced_expression"
    CAT_COVS_KEY = "cat_covs"
    CONT_COVS_KEY = "cont_covs"

class BaseModelClass(ABC):
    """Abstract class for deltaTopic models."""

    def __init__(self, adata: Optional[AnnData] = None):
        
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.test_indices_ = None
        self.validation_indices_ = None
        self.history_ = None
        self._data_loader_cls = AnnDataLoader

    def to_device(self, device: Union[str, int]):
        """
        Move model to device.

        Parameters
        ----------
        device
            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.

        Examples
        --------
        >>> model.to_device('cpu')      # moves model to CPU
        >>> model.to_device('cuda:0')   # moves model to GPU 0
        >>> model.to_device(0)          # also moves model to GPU 0
        """
        my_device = torch.device(device)
        self.module.to(my_device)

    @property
    def device(self):
        """Device model is on."""
        device = list(set(p.device for p in self.module.parameters()))
        if len(device) > 1:
            raise RuntimeError("Model tensors on multiple devices.")
        return device[0]

    def _make_data_loader(
        self,
        adata: AnnData,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        data_loader_class=None,
        **data_loader_kwargs,
    ):
        """
        Create a AnnDataLoader object for data iteration.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `settings.batch_size`.
        shuffle
            Whether observations are shuffled each iteration though
        data_loader_kwargs
            Kwargs to the class-specific data loader class
        """
        if batch_size is None:
            batch_size = 128
        if indices is None:
            indices = np.arange(adata.n_obs)
        if data_loader_class is None:
            data_loader_class = self._data_loader_cls

        if "num_workers" not in data_loader_kwargs:
            data_loader_kwargs.update({"num_workers": 2})

        dl = data_loader_class(
            adata,
            shuffle=shuffle,
            indices=indices,
            batch_size=batch_size,
            **data_loader_kwargs,
        )
        return dl

    @property
    def is_trained(self):
        return self.is_trained_

    @property
    def test_indices(self):
        return self.test_indices_

    @property
    def train_indices(self):
        return self.train_indices_

    @property
    def validation_indices(self):
        return self.validation_indices_

    @train_indices.setter
    def train_indices(self, value):
        """indices for training set."""
        self.train_indices_ = value

    @test_indices.setter
    def test_indices(self, value):
        """indices for test set."""
        self.test_indices_ = value

    @validation_indices.setter
    def validation_indices(self, value):
        """indices for validation set."""
        self.validation_indices_ = value

    @is_trained.setter
    def is_trained(self, value):
        """Flag to indicate whether model is trained."""
        self.is_trained_ = value

    @property
    def history(self):
        """Returns computed metrics during training."""
        return self.history_
        
    @abstractmethod
    def train(self):
        """Trains the model."""

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
            Kwargs for :meth:`~anndata.AnnData.write`
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
            self.adata.write(
                os.path.join(dir_path, "adata.h5ad"), **anndata_write_kwargs
            )

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        varnames_save_path = os.path.join(dir_path, "var_names.csv")

        var_names = self.adata.var_names.astype(str)
        var_names = var_names.to_numpy()
        np.savetxt(varnames_save_path, var_names, fmt="%s")

        torch.save(self.module.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(user_attributes, f)

    @classmethod
    def load(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Instantiate a model from the saved output.

        Parameters
        ----------
        dir_path
            Path to saved outputs.
        adata
            AnnData organized in the same way as data used to train model.

        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).

        Returns
        -------
        Model with loaded state dictionaries.

    
        """
        load_adata = adata is None
        use_gpu, device = parse_use_gpu_arg(use_gpu)

        (
            scvi_setup_dict,
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = _load_saved_files(dir_path, load_adata, map_location=device)
        adata = new_adata if new_adata is not None else adata


        model = _initialize_model(cls, adata, attr_dict)

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)


        model.to_device(device)
        model.module.eval()
        
        return model

def _load_saved_files(
    dir_path: str,
    load_adata: bool,
    map_location: Optional[Literal["cpu", "cuda"]] = None,
):
    """Helper to load saved files."""
    setup_dict_path = os.path.join(dir_path, "attr.pkl")
    adata_path = os.path.join(dir_path, "adata.h5ad")
    varnames_path = os.path.join(dir_path, "var_names.csv")
    model_path = os.path.join(dir_path, "model_params.pt")

    if os.path.exists(adata_path) and load_adata:
        adata = read(adata_path)
    elif not os.path.exists(adata_path) and load_adata:
        raise ValueError("Save path contains no saved anndata and no adata was passed.")
    else:
        adata = None

    var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)

    with open(setup_dict_path, "rb") as handle:
        attr_dict = pickle.load(handle)

    scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")

    model_state_dict = torch.load(model_path, map_location=map_location)

    return scvi_setup_dict, attr_dict, var_names, model_state_dict, adata

def _initialize_model(cls, adata, attr_dict):
    """Helper to initialize a model."""
    if "init_params_" not in attr_dict.keys():
        raise ValueError(
            "No init_params_ were saved by the model. Check out the "
            "developers guide if creating custom models."
        )
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
        non_kwargs = {k: v for k, v in init_params.items() if not isinstance(v, dict)}
        kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        non_kwargs.pop("use_cuda")

    model = cls(adata, **non_kwargs, **kwargs)
    return model

class DataSplitter(pl.LightningDataModule):
    """
    Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata
        AnnData to split into train/test/val sets
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    **kwargs
        Keyword args for data loader. 

    Examples
    --------
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    def __init__(
        self,
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata = adata
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""
        n = self.adata.n_obs
        n_train, n_val = validate_data_split(n, self.train_size, self.validation_size)
        random_state = np.random.RandomState(seed=666)
        permutation = random_state.permutation(n)
        self.val_idx = permutation[:n_val]
        self.train_idx = permutation[n_val : (n_val + n_train)]
        self.test_idx = permutation[(n_val + n_train) :]

        gpus, self.device = parse_use_gpu_arg(self.use_gpu, return_device=True)
        self.pin_memory = (
            True if (gpus != 0) else False
        )

    def train_dataloader(self):
        return AnnDataLoader(
            self.adata,
            indices=self.train_idx,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.val_idx,
                shuffle=True,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata,
                indices=self.test_idx,
                shuffle=True,
                drop_last=3,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

class TrainRunner:
    """
    TrainRunner calls Trainer.fit() and handles pre and post training procedures.

    Parameters
    ----------
    model
        model to train
    training_plan
        initialized TrainingPlan
    data_splitter
        initialized class:`DataSplitter`
    max_epochs
        max_epochs to train for
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    trainer_kwargs
        Extra kwargs for :class:`Trainer`

    Examples
    --------
    >>> # Following code should be within a subclass of BaseModelClass
    >>> data_splitter = DataSplitter(self.adata)
    >>> training_plan = TrainingPlan(self.module, len(data_splitter.train_idx))
    >>> runner = TrainRunner(
    >>>     self,
    >>>     training_plan=trianing_plan,
    >>>     data_splitter=data_splitter,
    >>>     max_epochs=max_epochs)
    >>> runner()
    """

    def __init__(
        self,
        model: BaseModelClass,
        training_plan: pl.LightningModule,
        data_splitter: DataSplitter,
        max_epochs: int,
        use_gpu: Optional[Union[str, int, bool]] = None,
        **trainer_kwargs,
    ):
        self.training_plan = training_plan
        self.data_splitter = data_splitter
        self.model = model
        gpus, device = parse_use_gpu_arg(use_gpu)
        self.gpus = gpus
        self.device = device
        self.trainer = Trainer(max_epochs=max_epochs, gpus=gpus, **trainer_kwargs)

    def __call__(self):
        self.data_splitter.setup()
        self.model.train_indices = self.data_splitter.train_idx
        self.model.test_indices = self.data_splitter.test_idx
        self.model.validation_indices = self.data_splitter.val_idx

        self.training_plan.n_obs_training = len(self.model.train_indices)

        self.trainer.fit(self.training_plan, self.data_splitter)
        try:
            self.model.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None

        self.model.module.eval()
        self.model.is_trained_ = True
        self.model.to_device(self.device)
        self.model.trainer = self.trainer

class Trainer(pl.Trainer):
    """
    Lightweight wrapper of Pytorch Lightning Trainer.


    Parameters
    ----------
    gpus
        Number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
    benchmark
        If true enables cudnn.benchmark, which improves speed when inputs are fixed size
    check_val_every_n_epoch
        Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`.
    max_epochs
        Stop training once this number of epochs is reached.
    
    num_sanity_val_steps
        Sanity check runs n validation batches before starting the training routine.
        Set it to -1 to run all batches in all validation dataloaders.
    weights_summary
        Prints a summary of the weights when training begins.
    early_stopping
        Whether to perform early stopping with respect to the validation set. This
        automatically adds a :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
        instance. A custom instance can be passed by using the callbacks argument and
        setting this to `False`.
    early_stopping_monitor
        Metric logged during validation set epoch. The available metrics will depend on
        the training plan class used. We list the most common options here in the typing.
    early_stopping_min_delta
        Minimum change in the monitored quantity to qualify as an improvement,
        i.e. an absolute change of less than min_delta, will count as no improvement.
    early_stopping_patience
        Number of validation epochs with no improvement after which training will be stopped.
    early_stopping_mode
            In 'min' mode, training will stop when the quantity monitored has stopped decreasing
            and in 'max' mode it will stop when the quantity monitored has stopped increasing.
    progress_bar_refresh_rate
        How often to refresh progress bar (in steps). Value 0 disables progress bar.
    simple_progress_bar
        Use  simple progress bar (per epoch rather than per batch)
    logger
        A valid pytorch lightning logger. Defaults to a simple dictionary logger.
        If `True`, defaults to the default pytorch lightning logger.
    **kwargs
        Other keyword args for :class:`~pytorch_lightning.trainer.Trainer`
    """

    def __init__(
        self,
        gpus: Union[int, str] = 1,
        benchmark: bool = True,
        check_val_every_n_epoch: Optional[int] = None,
        max_epochs: int = 400,
        #checkpoint_callback: bool = False,
        num_sanity_val_steps: int = 0,
        #weights_summary: Optional[Literal["top", "full"]] = None,
        early_stopping: bool = False,
        early_stopping_monitor: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        early_stopping_min_delta: float = 0.00,
        early_stopping_patience: int = 45,
        early_stopping_mode: Literal["min", "max"] = "min",
        #progress_bar_refresh_rate: int = 1,
        simple_progress_bar: bool = True,
        logger: Union[Optional[Logger], bool] = None,
        **kwargs
    ):

        kwargs["callbacks"] = (
            [] if "callbacks" not in kwargs.keys() else kwargs["callbacks"]
        )
        if early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor=early_stopping_monitor,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
            )
            kwargs["callbacks"] += [early_stopping_callback]
            check_val_every_n_epoch = 1
        else:
            check_val_every_n_epoch = (
                check_val_every_n_epoch
                if check_val_every_n_epoch is not None
                # needs to be an integer, np.inf does not work
                else sys.maxsize
            )

        if logger is None:
            logger = SimpleLogger()

        super().__init__(
            gpus=gpus,
            benchmark=benchmark,
            check_val_every_n_epoch=check_val_every_n_epoch,
            max_epochs=max_epochs,
            #checkpoint_callback=checkpoint_callback,
            num_sanity_val_steps=num_sanity_val_steps,
            #weights_summary=weights_summary,
            logger=logger,
            #progress_bar_refresh_rate=progress_bar_refresh_rate,
            **kwargs,
        )

    def fit(self, *args, **kwargs):

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", category=UserWarning, message="The dataloader,"
            )
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="you defined a validation_step but have no val_dataloader",
            )
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="One of given dataloaders is None and it will be skipped",
            )
            super().fit(*args, **kwargs)

class SimpleLogger(Logger):
    def __init__(self):
        super().__init__()
        self._data = {}

    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Record metrics."""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if "epoch" in metrics.keys():
            time_point = metrics.pop("epoch")
            time_point_name = "epoch"
        elif "step" in metrics.keys():
            time_point = metrics.pop("step")
            time_point_name = "step"
        else:
            time_point = step
            time_point_name = "step"
        for metric, value in metrics.items():
            if metric not in self._data:
                self._data[metric] = pd.DataFrame(columns=[metric])
                self._data[metric].index.name = time_point_name
            self._data[metric].loc[time_point, metric] = _handle_value(value)

    @property
    def history(self):
        return self._data

    @property
    def version(self):
        return "1"

    @property
    def name(self):
        return "SimpleLogger"
