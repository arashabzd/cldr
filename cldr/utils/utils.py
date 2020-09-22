import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.jit import trace
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from disentanglement_lib.data.ground_truth.named_data import get_named_ground_truth_data


def export_model(model, path, input_shape=(1, 3, 64, 64), use_script_module=True):
    """
    Exports the model. If the model is a `ScriptModule`, it is saved as is. If not,
    it is traced (with the given input_shape) and the resulting ScriptModule is saved
    (this requires the `input_shape`, which defaults to the competition default).

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        Pytorch Module or a ScriptModule.
    path : str
        Path to the file where the model is saved. Defaults to the value set by the
        `get_model_path` function above.
    input_shape : tuple or list
        Shape of the input to trace the module with. This is only required if model is not a
        torch.jit.ScriptModule.
    use_script_module : True or False (default = True)
        If True saves model as torch.jit.ScriptModule -- this is highly recommended.
        Setting it to False may cause later evaluation to fail.

    Returns
    -------
    str
        Path to where the model is saved.
    """
    model = deepcopy(model).cpu().eval()
    if isinstance(model, torch.jit.ScriptModule):
        assert use_script_module, "Provided model is a ScriptModule, please set use_script_module to True."
    if use_script_module:
        if not isinstance(model, torch.jit.ScriptModule):
            assert input_shape is not None, "`input_shape` must be provided since model is not a " \
                                            "`ScriptModule`."
            traced_model = trace(model, torch.zeros(*input_shape))
        else:
            traced_model = model
        torch.jit.save(traced_model, path)
    else:
        torch.save(model, path) # saves model as a nn.Module
    return path


def import_model(path):
    """
    Imports a model (as torch.jit.ScriptModule or torch.nn.Module) from file.
    By default the file is imported as torch.jit.ScriptModule. If it fails due to saved model being torch.nn.Module, the file is imported as torch.nn.Module.

    Parameters
    ----------
    path : str
        Path to where the model is saved. Defaults to the return value of the `get_model_path`

    Returns
    -------
    torch.jit.ScriptModule / torch.nn.Module
        The model file.
    """
    try:
        return torch.jit.load(path)
    except RuntimeError:
        try:
            return torch.load(path) # loads model as a nn.Module
        except Exception as e:
            raise IOError("Could not load file. Please save as torch.jit.ScriptModule instead.") from e


def make_representor(model, cuda):
    """
    Encloses the pytorch ScriptModule in a callable that can be used by `disentanglement_lib`.

    Parameters
    ----------
    model : torch.nn.Module or torch.jit.ScriptModule
        The Pytorch model.
    cuda : bool
        Whether to use CUDA for inference. Defaults to the return value of the `use_cuda`
        function defined above.

    Returns
    -------
    callable
        A callable function (`representation_function` in dlib code)
    """
    model = model.cuda() if cuda else model.cpu()
    
    def _represent(x):
        assert isinstance(x, np.ndarray), \
            "Input to the representation function must be a ndarray."
        assert x.ndim == 4, \
            "Input to the representation function must be a four dimensional NHWC tensor."
        # Convert from NHWC to NCHW
        x = np.moveaxis(x, 3, 1)
        # Convert to torch tensor and evaluate
        x = torch.from_numpy(x).float().to('cuda' if cuda else 'cpu')
        with torch.no_grad():
            y = model(x)
        y = y.cpu().numpy()
        assert y.ndim == 2, \
            "The returned output from the representor must be two dimensional (NC)."
        return y

    return _represent


class DLIBDataset(Dataset):
    """
    No-bullshit data-loading from Disentanglement Library, but with a few sharp edges.

    Sharp edge:
        Unlike a traditional Pytorch dataset, indexing with _any_ index fetches a random batch.
        What this means is dataset[0] != dataset[0]. Also, you'll need to specify the size
        of the dataset, which defines the length of one training epoch.

        This is done to ensure compatibility with disentanglement_lib.
    """

    def __init__(self, name, seed=0, iterator_len=50000):
        """
        Parameters
        ----------
        name : str
            Name of the dataset use. You may use `get_dataset_name`.
        seed : int
            Random seed.
        iterator_len : int
            Length of the dataset. This defines the length of one training epoch.
        """
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.iterator_len = iterator_len
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return get_named_ground_truth_data(self.name)

    def __len__(self):
        return self.iterator_len

    def __getitem__(self, item):
        assert item < self.iterator_len
        y, x = self.dataset.sample(1, random_state=self.random_state)
        # Convert output to CHW from HWC
        x = torch.from_numpy(np.moveaxis(x[0], 2, 0))
        y = torch.from_numpy(y)[0]
        return x, y


def get_loader(name, batch_size=64, seed=0, iterator_len=50000, num_workers=0,
               **dataloader_kwargs):
    """
    Makes a dataset and a data-loader.

    Parameters
    ----------
    name : str
        Name of the dataset use.
    batch_size : int
        Batch size.
    seed : int
        Random seed.
    iterator_len : int
        Length of the dataset. This defines the length of one training epoch.
    num_workers : int
        Number of processes to use for multiprocessed data-loading.
    dataloader_kwargs : dict
        Keyword arguments for the data-loader.

    Returns
    -------
    DataLoader
    """
    dlib_dataset = DLIBDataset(name, seed=seed, iterator_len=iterator_len)
    loader = DataLoader(dlib_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, **dataloader_kwargs)
    return loader


class BatchTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([self.transform(xi) for xi in x])
