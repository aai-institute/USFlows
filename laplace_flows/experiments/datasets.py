from abc import abstractmethod
import torch
from torch import Tensor
from torchvision.datasets import FashionMNIST, MNIST
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import typing as T

import idx2numpy

class DequantizedDataset(torch.utils.data.Dataset):
    """
    A dataset that dequantizes the data by adding uniform noise to each pixel.
    """
    def __init__(self, dataset: T.Union[os.PathLike, torch.utils.data.Dataset, np.ndarray], num_bits: int = 8):
        if isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, np.ndarray):
            self.dataset = dataset
        else:
            self.dataset = pd.read_csv(dataset).values
            
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x / self.num_levels),
            transforms.Lambda(lambda x: x + torch.rand_like(x) / self.num_levels)
            ])

    def __getitem__(self, index: int):

        x, y = self.dataset[index]
        x = Tensor(self.transform(x))
        return x, y

    def __len__(self):
        return len(self.dataset)
    
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
    
class FashionMnistDequantized(DequantizedDataset):
    def __init__(self, dataloc: os.PathLike = None, train: bool = True):
        rel_path = "FashionMNIST/raw/train-images-idx3-ubyte" if train else "FashionMNIST/raw/t10k-images-idx3-ubyte"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            FashionMNIST(path, train=train, download=True)
        dataset = idx2numpy.convert_from_file(path)
        dataset = dataset.reshape(dataset.shape[0], 1, 28, 28)     
        super().__init__(dataset, num_bits=8)
    
    def __getitem__(self, index: int):

        x = Tensor(self.dataset[index].copy())
        x = self.transform(x)
        return x, 0

class MnistDequantized(DequantizedDataset):
    def __init__(self, dataloc: os.PathLike = None, train: bool = True):
        rel_path = "MNIST/raw/train-images-idx3-ubyte" if train else "MNIST/raw/t10k-images-idx3-ubyte"
        path = os.path.join(dataloc, rel_path)
        if not os.path.exists(path):
            MNIST(path, train=train, download=True)
        # TODO: remove hardcoding of 3x3 downsampling
        dataset = idx2numpy.convert_from_file(path)[:, ::3, ::3]
        dataset = dataset.reshape(dataset.shape[0], -1)     
        super().__init__(dataset, num_bits=8)
    
    def __getitem__(self, index: int):

        x = Tensor(self.dataset[index].copy())
        x = self.transform(x)
        return x, 0

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
    
class FashionMnistSplit(DataSplit):
    def __init__(self, dataloc: os.PathLike = None, val_split: float = .1):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = FashionMnistDequantized(self.dataloc, train=True)
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(self.train, shuffle[:int(len(self.train) * val_split)])
        self.train = torch.utils.data.Subset(self.train, shuffle[int(len(self.train) * val_split):])
        self.test = FashionMnistDequantized(self.dataloc, train=False)
    
    def get_train(self) -> torch.utils.data.Dataset:
        return self.train
    
    def get_test(self) -> torch.utils.data.Dataset:
        return self.test
    
    def get_val(self) -> torch.utils.data.Dataset:
        return self.val
    
class MnistSplit(DataSplit):
    def __init__(self, dataloc: os.PathLike = None, val_split: float = .1):
        if dataloc is None:
            dataloc = os.path.join(os.getcwd(), "data")
        self.dataloc = dataloc
        self.train = MnistDequantized(self.dataloc, train=True)
        shuffle = torch.randperm(len(self.train))
        self.val = torch.utils.data.Subset(self.train, shuffle[:int(len(self.train) * val_split)])
        self.train = torch.utils.data.Subset(self.train, shuffle[int(len(self.train) * val_split):])
        self.test = MnistDequantized(self.dataloc, train=False)
    
    def get_train(self) -> torch.utils.data.Dataset:
        return self.train
    
    def get_test(self) -> torch.utils.data.Dataset:
        return self.test
    
    def get_val(self) -> torch.utils.data.Dataset:
        return self.val