import os
import typing as T
from abc import abstractmethod

import idx2numpy
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.datasets import make_blobs, make_checkerboard, make_circles, make_moons
from torch import Tensor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10


# Base dataset classes
class DequantizedDataset(torch.utils.data.Dataset):
    """
    A dataset that dequantizes the data by adding uniform noise to each pixel.
    """

    def __init__(
        self,
        dataset: T.Union[os.PathLike, torch.utils.data.Dataset, np.ndarray],
        num_bits: int = 8,
        device: torch.device = None, 
    ):
        if isinstance(dataset, torch.utils.data.Dataset) or isinstance(
            dataset, np.ndarray
        ) or isinstance(dataset, torch.Tensor):
            self.dataset = dataset
        else:
            self.dataset = pd.read_csv(dataset).values

        #
        self.dataset = self.dataset.to(device)
        self.num_bits = num_bits
        self.num_levels = 2**num_bits
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / self.num_levels),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / self.num_levels),
            ]
        )

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = Tensor(self.transform(x))
        return x, y

    def __len__(self):
        return len(self.dataset)


class DataSplit:
    def __init__(*agrs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_test(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def get_val(self) -> torch.utils.data.Dataset:
        raise NotImplementedError


class SimpleSplit(DataSplit):
    """
    Split of dataset
    """

    def __init__(
        self,
        train: torch.utils.data.Dataset,
        test: torch.utils.data.Dataset,
        val: torch.utils.data.Dataset,
    ):
        """Create split of dataset

        Args:
            train (torch.utils.data.Dataset): training set
            test (torch.utils.data.Dataset): test set
            val (torch.utils.data.Dataset): validation set
        """
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val

def make_transformed_laplace(dim: int, n_samples: int, transform: Tensor = None) -> Tensor:
    """Create uniform dataset

    Args:
        dim (int): dimensionality of dataset
        num_samples (int): number of samples
        transform (Tensor, optional): transformation matrix. Defaults to None which applies no transformation.

    Returns:
        np.ndarray: dataset
    """
    sample = torch.distributions.Laplace(torch.zeros(dim), torch.ones(dim)).sample([n_samples])
    
    if transform is not None:
        inconsistent = len(transform.shape) != 2
        inconsistent = inconsistent or (transform.shape[0] != transform.shape[1])
        inconsistent = inconsistent or (transform.shape[0] != dim)
        if inconsistent:
            raise ValueError(f"transform must be {dim}X{dim} got {transform.shape}.")

        sample = (transform @ sample.T).T
    
    return sample  
    
    

class FlattenedDataset(torch.utils.data.Dataset):
    """
    A dataset that flattens the data.
    """

    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        x = x.flatten()
        return x, y

    def __len__(self):
        return len(self.dataset)
    
class DataSplitFromCSV(DataSplit):
    def __init__(self, train: os.PathLike, test: os.PathLike, val: os.PathLike):
        self.train = train
        self.test = test
        self.val = val

    def get_train(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.train).values

    def get_test(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.test).values

    def get_val(self) -> torch.utils.data.Dataset:
        return pd.read_csv(self.val).values


# Synthetic datasets


GENERATORS = {
    "moons": make_moons,
    "blobs": make_blobs,
    "checkerboard": make_checkerboard,
    "circles": make_circles,
    "transformed": make_transformed_laplace,
}

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset from generator function
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: [dict]: parameters for generator function
        """
        super().__init__(*args, **kwargs)
        if isinstance(generator, str):
            generator = GENERATORS[generator]

        self.dataset = generator(**params)[0]

    def __getitem__(self, index: int):
        if isinstance(self.dataset, np.ndarray):
            x = self.dataset[index].copy()
        else:
            x = self.dataset[index]
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return x, torch.zeros_like(x)

    def __len__(self):
        return len(self.dataset)


class SyntheticSplit(SimpleSplit):
    """
    Split of synthetic dataset
    """

    def __init__(
        self,
        generator: T.Union[T.Callable[..., np.ndarray], str],
        params_train: T.Dict[str, T.Any],
        params_test: T.Dict[str, T.Any],
        params_val: T.Dict[str, T.Any],
        *args,
        **kwargs
    ):
        """Create dataset from generator function

        Args:
            generator (function): generator function
            params: ]dict]: parameters for generator function
        """
        if isinstance(generator, str):
            generator = GENERATORS[generator]

        train = SyntheticDataset(generator, params_train)
        test = SyntheticDataset(generator, params_test)
        val = SyntheticDataset(generator, params_val)
        super().__init__(train=train, test=test, val=val, *args, **kwargs)


# FashonMNIST
class FashionMnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        label: T.Optional[int] = None,
        scale: bool = False
    ):
        rel_path = (
            "FashionMNIST/raw/train-images-idx3-ubyte"
            if train
            else "FashionMNIST/raw/t10k-images-idx3-ubyte"
        )
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            FashionMNIST(dataloc, train=train, download=True)
        # TODO: remove hardcoding of 3x3 downsampling, vectorizing
        dataset = idx2numpy.convert_from_file(path)
        if scale:
            dataset = dataset[:, ::3, ::3]
        dataset = dataset.reshape(dataset.shape[0], -1)
        if label is not None:
            rel_path = (
                "FashionMNIST/raw/train-labels-idx1-ubyte"
                if train
                else "FashionMNIST/raw/t10k-labels-idx1-ubyte"
            )
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == label]
        super().__init__(dataset, num_bits=8)

    def __getitem__(self, index: int):
        x = Tensor(self.dataset[index].copy())
        x = self.transform(x)
        return x, 0


class FashionMnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        label: T.Optional[int] = None,
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = FashionMnistDequantized(self.dataloc, train=True, label=label)
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = FashionMnistDequantized(self.dataloc, train=False, label=label)

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val


# MNIST
class MnistDequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        digit: T.Optional[int] = None,
        flatten=True,
        scale: bool = False,
        device: torch.device = None
    ):
        if train:
            rel_path = "MNIST/raw/train-images-idx3-ubyte"
        else:
            rel_path = "MNIST/raw/t10k-images-idx3-ubyte"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            MNIST(dataloc, train=train, download=True)

        dataset = idx2numpy.convert_from_file(path)
        if scale:
            dataset = dataset[:, ::3, ::3]
        if flatten:
            dataset = dataset.reshape(dataset.shape[0], -1)
        if digit is not None:
            if train:
                rel_path = "MNIST/raw/train-labels-idx1-ubyte"
            else:
                rel_path = "MNIST/raw/t10k-labels-idx1-ubyte"
            path = os.path.join(dataloc, rel_path)
            labels = idx2numpy.convert_from_file(path)
            dataset = dataset[labels == digit]
        super().__init__(torch.Tensor(dataset), num_bits=8, device=device)

    def __getitem__(self, index: int):
        if not isinstance(self.dataset, torch.Tensor):
            x = Tensor(self.dataset[index].copy())
        else:
            x = self.dataset[index]
        x = self.transform(x)
        return x, 0

class MnistSplit(DataSplit):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        val_split: float = 0.1,
        digit: T.Optional[int] = None,
        scale: bool = False,
        device: torch.device = None
    ):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = MnistDequantized(self.dataloc, train=True, digit=digit, scale=scale, device=device)
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(
            self.train, shuffle[: int(len(self.train) * val_split)]
        )
        self.train = torch.utils.data.Subset(
            self.train, shuffle[int(len(self.train) * val_split) :]
        )
        self.test = MnistDequantized(self.dataloc, train=False, digit=digit, scale=scale, device=device)

    def get_train(self) -> torch.utils.data.Dataset:
        return self.train

    def get_test(self) -> torch.utils.data.Dataset:
        return self.test

    def get_val(self) -> torch.utils.data.Dataset:
        return self.val

# CFAIR-10

class Cifar10Dequantized(DequantizedDataset):
    def __init__(
        self,
        dataloc: os.PathLike = None,
        train: bool = True,
        label: T.Optional[int] = None,
    ):
        if train:
            rel_path = "CIFAR10/raw/data_batch_1"
        else:
            rel_path = "CIFAR10/raw/test_batch"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            CIFAR10(dataloc, train=train, download=True)
            

class CreditData(SimpleSplit):
    def __init__(self,
                 dataloc: os.PathLike,
                 train: bool = True
                 ):
        print("fetching credit data")
        path = dataloc
        if not os.path.exists(path):
            print(f'Dataset not found {path}')
        else:
            cred_data = pd.read_csv(path)
            if cred_data is None:
                print(f'Credit data is none')
            self.dataloc = dataloc
            #cred_data = cred_data.drop(["Id"], axis=1)
            # Split into 60% train, 20% validate and 20% test set.
            train_nd, val_nd, test_nd = \
                np.split(cred_data.sample(frac=1, random_state=42),
                         [int(.6 * len(cred_data)), int(.8 * len(cred_data))])

            self.train = torch.from_numpy(train_nd.to_numpy(copy=True).copy()).float()
            self.val = torch.from_numpy(val_nd.to_numpy(copy=True).copy()).float()
            self.test = torch.from_numpy(test_nd.to_numpy(copy=True).copy()).float()

            print("-------train-------------")
            print(self.train)
            print("-------val-------------")
            print(self.val)
            print("-------test-------------")
            print(self.test)
            print("--------------------")


class HelocData(SimpleSplit):
    def __init__(self,
                 dataloc: os.PathLike,
                 train: bool = True
                 ):
        print("fetching heloc data")
        path = dataloc
        if not os.path.exists(path):
            print(f'Dataset not found {path}')
        else:
            cred_data = pd.read_csv(path)
            if cred_data is None:
                print(f'Heloc data is none')
            self.dataloc = dataloc
            train_nd, val_nd, test_nd = \
                np.split(cred_data.sample(frac=1, random_state=42),
                         [int(.6 * len(cred_data)), int(.8 * len(cred_data))])

            self.train = torch.from_numpy(train_nd.to_numpy(copy=True).copy()).float()
            self.val = torch.from_numpy(val_nd.to_numpy(copy=True).copy()).float()
            self.test = torch.from_numpy(test_nd.to_numpy(copy=True).copy()).float()

            print("-------train-------------")
            print(self.train)
            print("-------val-------------")
            print(self.val)
            print("-------test-------------")
            print(self.test)
            print("--------------------")

